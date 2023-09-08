from functools import partial
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.datasets import ZINC

from pygho import SparseTensor, MaskedTensor
from pygho.hodata import SpDataloader, Sppretransform, Mapretransform, MaDataloader, ParallelPreprocessDataset
from pygho.hodata.SpTupleSampler import KhopSampler
from pygho.hodata.MaTupleSampler import spdsampler

from pygho.honn.SpXOperator import parse_precomputekey

from pygho.backend.utils import torch_scatter_reduce
from pygho.honn.Conv import NGNNConv, GNNAKConv, DSSGNNConv, SSWLConv, SUNConv
from pygho.honn.TensorOp import OpPoolingSubg2D
from pygho.honn.MaXOperator import OpPooling
from pygho.honn.utils import MLP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--aggr", choices=["sum", "mean", "max"])
parser.add_argument("--conv",
                    choices=["NGNN", "GNNAK", "DSSGNN", "SSWL", "SUN"])
parser.add_argument("--npool", choices=["mean", "sum", "max"])
parser.add_argument("--lpool", choices=["mean", "sum", "max"])
parser.add_argument("--cpool", choices=["mean", "sum", "max"])

args = parser.parse_args()


class InputEncoderMa(nn.Module):

    def __init__(self, hiddim: int) -> None:
        super().__init__()
        self.x_encoder = nn.Embedding(32, hiddim)
        self.ea_encoder = nn.Embedding(16, hiddim, padding_idx=0)
        self.tuplefeat_encoder = nn.Embedding(16, hiddim)

    def forward(self, datadict: dict) -> dict:
        datadict["x"] = datadict["x"].tuplewiseapply(
            lambda x: self.x_encoder(x.squeeze(-1)))
        datadict["A"] = datadict["A"].tuplewiseapply(self.ea_encoder)
        datadict["X"] = MaskedTensor(datadict["X"].data, torch.logical_and(datadict["X"].mask, datadict["X"].data<4).squeeze(), 0, False)
        datadict["X"] = datadict["X"].tuplewiseapply(self.tuplefeat_encoder)
        return datadict


class InputEncoderSp(nn.Module):

    def __init__(self, hiddim: int) -> None:
        super().__init__()
        self.x_encoder = nn.Embedding(32, hiddim)
        self.ea_encoder = nn.Embedding(16, hiddim)
        self.tuplefeat_encoder = nn.Embedding(16, hiddim)

    def forward(self, datadict: dict) -> dict:
        datadict["x"] = self.x_encoder(datadict["x"].flatten())
        datadict["A"] = datadict["A"].tuplewiseapply(self.ea_encoder)
        datadict["X"] = datadict["X"].tuplewiseapply(self.tuplefeat_encoder)
        return datadict



def transfermlpparam(mlp: dict):
    mlp = mlp.copy()
    mlp.update({"tailact": True, "numlayer": 1})
    return mlp


spconvdict = {
    "SSWL":
    lambda dim, mlp: SSWLConv(dim, dim, args.aggr, "SS", transfermlpparam(mlp)
                              ),
    "DSSGNN":
    lambda dim, mlp: DSSGNNConv(dim, dim, args.aggr, args.aggr, args.cpool,
                                "SS", transfermlpparam(mlp)),
    "GNNAK":
    lambda dim, mlp: GNNAKConv(dim, dim, args.aggr, args.cpool, "SS",
                               transfermlpparam(mlp), transfermlpparam(mlp)),
    "SUN":
    lambda dim, mlp: SUNConv(dim, dim, args.aggr, args.cpool, "SS",
                             transfermlpparam(mlp), transfermlpparam(mlp)),
    "NGNN":
    lambda dim, mlp: NGNNConv(dim, dim, args.aggr, "SS", transfermlpparam(mlp))
}

maconvdict = {
    "SSWL":
    lambda dim, mlp: SSWLConv(dim, dim, args.aggr, "DD", transfermlpparam(mlp)
                              ),
    "DSSGNN":
    lambda dim, mlp: DSSGNNConv(dim, dim, args.aggr, args.aggr, args.cpool,
                                "DD", transfermlpparam(mlp)),
    "GNNAK":
    lambda dim, mlp: GNNAKConv(dim, dim, args.aggr, args.cpool, "DD",
                               transfermlpparam(mlp), transfermlpparam(mlp)),
    "SUN":
    lambda dim, mlp: SUNConv(dim, dim, args.aggr, args.cpool, "DD",
                             transfermlpparam(mlp), transfermlpparam(mlp)),
    "NGNN":
    lambda dim, mlp: NGNNConv(dim, dim, args.aggr, "DD", transfermlpparam(mlp))
}


class MaModel(nn.Module):

    def __init__(self,
                 convfn: Callable,
                 num_tasks=1,
                 num_layer=5,
                 hiddim=128,
                 npool="mean",
                 lpool="max",
                 residual=True,
                 outlayer: int = 2,
                 ln_out: bool = False,
                 mlp: dict = {}):
        '''
            num_tasks (int): number of labels to be predicted
        '''
        super().__init__()
        self.num_layer = num_layer
        self.hiddim = hiddim
        self.num_tasks = num_tasks

        self.lin_tupleinit0 = nn.Linear(hiddim, hiddim)
        self.lin_tupleinit1 = nn.Linear(hiddim, hiddim)

        self.residual = residual
        self.subggnns = nn.ModuleList(
            [convfn(hiddim, mlp) for _ in range(num_layer)])

        self.npool = OpPooling(1, pool=npool)
        self.lpool = OpPoolingSubg2D("D", pool=lpool)
        self.data_encoder = InputEncoderMa(hiddim)

        outdim = self.hiddim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(
            MLP(outdim, num_tasks, outlayer, tailact=False, **mlp),
            nn.LayerNorm(num_tasks, elementwise_affine=False)
            if ln_out else nn.Identity())

    def tupleinit(self, X: MaskedTensor, x: MaskedTensor) -> MaskedTensor:
        return X.tuplewiseapply(
            lambda val: self.lin_tupleinit0(x.fill_masked(0)).unsqueeze(
                1) * self.lin_tupleinit1(x.fill_masked(0)).unsqueeze(2) * val)

    def forward(self, datadict: dict):
        '''
        TODO: !warning input must be coalesced
        '''
        datadict = self.data_encoder(datadict)
        A = datadict["A"]
        X = datadict["X"]
        x = datadict["x"]
        X = self.tupleinit(datadict["X"], datadict["x"])
        for conv in self.subggnns:
            tX = conv.forward(A, X, datadict)
            if self.residual:
                X = X.add(tX, samesparse=True)
            else:
                X = tX
        x = self.lpool(X)
        h_graph = self.npool.forward(x).fill_masked(0)
        return self.pred_lin(h_graph)


class SpModel(nn.Module):

    def __init__(self,
                 convfn: Callable,
                 num_tasks=1,
                 num_layer=5,
                 hiddim=128,
                 npool="mean",
                 lpool="max",
                 residual=True,
                 outlayer: int = 1,
                 ln_out: bool = False,
                 mlp: dict = {}):
        '''
            num_tasks (int): number of output dimensions
            npool: node level pooling
            lpool: subgraph pooling
            aggr: aggregation scheme in MPNN on each subgraph
            ln_out: use layernorm in output, 
                a normalization method for classification problem
        '''

        super().__init__()
        self.num_layer = num_layer
        self.hiddim = hiddim
        self.num_tasks = num_tasks

        self.lin_tupleinit0 = nn.Linear(hiddim, hiddim)
        self.lin_tupleinit1 = nn.Linear(hiddim, hiddim)

        self.residual = residual
        self.subggnns = nn.ModuleList(
            [convfn(hiddim, mlp) for _ in range(num_layer)])

        self.npool = npool
        self.lpool = OpPoolingSubg2D("S", lpool)
        self.data_encoder = InputEncoderSp(hiddim)

        outdim = self.hiddim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(
            MLP(outdim, num_tasks, outlayer, tailact=False, **mlp),
            nn.LayerNorm(num_tasks, elementwise_affine=False)
            if ln_out else nn.Identity())

    def tupleinit(self, X: SparseTensor, x):
        return X.tuplewiseapply(lambda val: self.lin_tupleinit0(x)[X.indices[
            0]] * self.lin_tupleinit1(x)[X.indices[1]] * val)

    def forward(self, datadict: dict):
        datadict = self.data_encoder(datadict)
        A = datadict["A"]
        X = datadict["X"]
        x = datadict["x"]
        X = self.tupleinit(X, x)
        for conv in self.subggnns:
            tX = conv.forward(A, X, datadict)
            if self.residual:
                X = X.add(tX, True)
            else:
                X = tX
        x = self.lpool(X)
        h_graph = torch_scatter_reduce(0, x, datadict["batch"], datadict["num_graphs"], self.npool)
        return self.pred_lin(h_graph)


if args.sparse:
    model = SpModel(spconvdict[args.conv],
                    mlp={
                        "dp": 0,
                        "norm": "ln",
                        "act": "relu",
                        "normparam": 0.1
                    })
else:
    model = MaModel(maconvdict[args.conv],
                    mlp={
                        "dp": 0,
                        "norm": "ln",
                        "act": "silu",
                        "normparam": 0.1
                    })

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

device = torch.device("cuda")

trn_dataset = ZINC("dataset/ZINC", subset=True, split="train")
val_dataset = ZINC("dataset/ZINC", subset=True, split="val")
tst_dataset = ZINC("dataset/ZINC", subset=True, split="test")
if args.sparse:
    keys = parse_precomputekey(model)
    trn_dataset = ParallelPreprocessDataset(
        "dataset/ZINC_trn", trn_dataset,
        Sppretransform(None, partial(KhopSampler, hop=3), [""], keys), 0)
    val_dataset = ParallelPreprocessDataset(
        "dataset/ZINC_val", val_dataset,
        Sppretransform(None, partial(KhopSampler, hop=3), [""], keys), 0)
    tst_dataset = ParallelPreprocessDataset(
        "dataset/ZINC_tst", tst_dataset,
        Sppretransform(None, partial(KhopSampler, hop=3), [""], keys), 0)
    trn_dataloader = SpDataloader(trn_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True)
    val_dataloader = SpDataloader(val_dataset, batch_size=128)
    tst_dataloader = SpDataloader(tst_dataset, batch_size=128)
else:
    trn_dataset = ParallelPreprocessDataset("dataset/ZINC_trn",
                                            trn_dataset,
                                            pre_transform=Mapretransform(
                                                None, partial(spdsampler,
                                                              hop=4)),
                                            num_worker=0)
    val_dataset = ParallelPreprocessDataset("dataset/ZINC_val",
                                            val_dataset,
                                            pre_transform=Mapretransform(
                                                None, partial(spdsampler,
                                                              hop=4)),
                                            num_worker=0)

    tst_dataset = ParallelPreprocessDataset("dataset/ZINC_tst",
                                            tst_dataset,
                                            pre_transform=Mapretransform(
                                                None, partial(spdsampler,
                                                              hop=4)),
                                            num_worker=0)

    trn_dataloader = MaDataloader(trn_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True)
    val_dataloader = MaDataloader(val_dataset, batch_size=128)
    tst_dataloader = MaDataloader(tst_dataset, batch_size=128)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=40 *
                                                       len(trn_dataloader))

model = model.to(device)


def train(dataloader):
    model.train()
    losss = []
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        datadict = batch.to_dict()
        datadict["num_graphs"] = batch.num_graphs
        pred = model(datadict)
        loss = F.l1_loss(datadict["y"].unsqueeze(-1), pred, reduction="mean")
        loss.backward()
        optimizer.step()
        scheduler.step()
        losss.append(loss)
    losss = np.average(list(map(lambda x: x.item(), losss)))
    return losss


@torch.no_grad()
def eval(dataloader):
    model.eval()
    loss = 0
    size = 0
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        datadict = batch.to_dict()
        datadict["num_graphs"] = batch.num_graphs
        pred = model(datadict)
        loss += F.l1_loss(datadict["y"].unsqueeze(-1), pred, reduction="sum")
        size += pred.shape[0]
    return loss / size


best_val = float("inf")
tst_score = float("inf")

for epoch in range(1, 1001):
    t1 = time.time()
    losss = train(trn_dataloader)
    t2 = time.time()
    val_score = eval(val_dataloader)
    if val_score < best_val:
        best_val = val_score
        tst_score = eval(tst_dataloader)
    t3 = time.time()
    print(
        f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f}  l1loss {losss:.4f} val MAE {val_score:.4f} tst MAE {tst_score:.4f}"
    )
