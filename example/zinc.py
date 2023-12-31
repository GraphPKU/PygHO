import numpy as np

from functools import partial
import time
from typing import Callable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import ZINC

from lr_scheduler import CosineAnnealingWarmRestarts

from pygho import SparseTensor, MaskedTensor
from pygho.hodata import SpDataloader, Sppretransform, Mapretransform, MaDataloader, ParallelPreprocessDataset
from pygho.hodata.SpTupleSampler import KhopSampler, I2Sampler
from pygho.hodata.MaTupleSampler import spdsampler

from pygho.honn.SpOperator import parse_precomputekey

from pygho.backend.utils import torch_scatter_reduce
from pygho.honn.Conv import NGNNConv, GNNAKConv, DSSGNNConv, SSWLConv, SUNConv, PPGNConv, I2Conv
from pygho.honn.TensorOp import OpPoolingSubg2D, OpPoolingSubg3D
from pygho.honn.MaOperator import OpPooling
from pygho.honn.utils import MLP

import argparse

torch.set_float32_matmul_precision('high')
parser = argparse.ArgumentParser()
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--aggr", choices=["sum", "mean", "max"], default="sum")
parser.add_argument("--conv",
                    choices=["NGNN", "NGAT", "GNNAK", "DSSGNN", "SSWL", "SUN", "PPGN", "I2GNN"], default="NGNN")
parser.add_argument("--npool", choices=["mean", "sum", "max"], default="sum")
parser.add_argument("--lpool", choices=["mean", "sum", "max"], default="mean")
parser.add_argument("--cpool", choices=["mean", "sum", "max"], default="mean")
parser.add_argument("--mlplayer", type=int, default=2)
parser.add_argument("--outlayer", type=int, default=2)
parser.add_argument("--norm", choices=["ln", "bn", "none"], default="bn")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--minlr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--dp", type=float, default=0.0)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--normparam", type=float, default=0.1)
parser.add_argument("--cosT", type=int, default=100)
parser.add_argument("--K", type=float, default=0)
parser.add_argument("--K2", type=float, default=0)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

# 1 Models Definition


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


class InputEncoderI2(nn.Module):

    def __init__(self, hiddim: int) -> None:
        super().__init__()
        self.x_encoder = nn.Embedding(32, hiddim)
        self.ea_encoder = nn.Embedding(16, hiddim)
        self.tuplefeat_encoder1 = nn.Embedding(16, hiddim)
        self.tuplefeat_encoder2 = nn.Embedding(16, hiddim)

    def forward(self, datadict: dict) -> dict:
        datadict["x"] = self.x_encoder(datadict["x"].flatten())
        datadict["A"] = datadict["A"].tuplewiseapply(self.ea_encoder)
        datadict["X"] = datadict["X"].tuplewiseapply(lambda x: self.tuplefeat_encoder1(x[:, 0]) + self.tuplefeat_encoder2(x[:, 1]))
        return datadict

def transfermlpparam(mlp: dict):
    mlp = mlp.copy()
    mlp.update({"tailact": True, "numlayer": args.mlplayer})
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
    lambda dim, mlp: NGNNConv(dim, dim, args.aggr, "SS", transfermlpparam(mlp)
                              ),
    "PPGN":
    lambda dim, mlp: PPGNConv(dim, dim, args.aggr, "SS", transfermlpparam(mlp)
                              ),
    "I2GNN":
    lambda dim, mlp: I2Conv(dim, dim, args.aggr, "SS", transfermlpparam(mlp))
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
    lambda dim, mlp: NGNNConv(dim, dim, args.aggr, "DD", transfermlpparam(mlp)
                              ),
    "PPGN":
    lambda dim, mlp: PPGNConv(dim, dim, args.aggr, "DD", transfermlpparam(mlp)),
    "I2GNN":
    lambda dim, mlp: I2Conv(dim, dim, args.aggr, "DD", transfermlpparam(mlp))
}


class MaModel(nn.Module):
    residual: Final[bool]
    def __init__(self,
                 convfn: Callable,
                 num_tasks=1,
                 num_layer=6,
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
        self.poolmlp = MLP(hiddim, hiddim, args.mlplayer, tailact=True, **mlp)
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
            lambda val: self.lin_tupleinit0(x.fill_masked(0.)).unsqueeze(
                1) * self.lin_tupleinit1(x.fill_masked(0.)).unsqueeze(2) * val)

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
        x = x.tuplewiseapply(self.poolmlp)
        h_graph = self.npool.forward(x).fill_masked(0.)
        return self.pred_lin(h_graph)


class SpModel(nn.Module):

    def __init__(self,
                 convfn: Callable,
                 num_tasks=1,
                 num_layer=6,
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
        self.lin_tupleinit2 = nn.Linear(hiddim, hiddim)

        self.residual = residual
        self.subggnns = nn.ModuleList(
            [convfn(hiddim, mlp) for _ in range(num_layer)])

        self.npool = npool       
        self.lpool =  nn.Sequential(OpPoolingSubg3D("S", lpool), OpPoolingSubg2D("S", lpool)) if args.conv in ["I2GNN"] else OpPoolingSubg2D("S", lpool)
        self.poolmlp = MLP(hiddim, hiddim, args.mlplayer, tailact=True, **mlp)
        self.data_encoder = InputEncoderI2(hiddim) if args.conv in ["I2GNN"] else  InputEncoderSp(hiddim) 

        outdim = self.hiddim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(
            MLP(outdim, num_tasks, outlayer, tailact=False, **mlp),
            nn.LayerNorm(num_tasks, elementwise_affine=False)
            if ln_out else nn.Identity())

    def tupleinit(self, X: SparseTensor, x):
        if args.conv in ["I2GNN"]:
            return X.tuplewiseapply(lambda val: self.lin_tupleinit0(x)[X.indices[
                0]] * self.lin_tupleinit1(x)[X.indices[1]] * self.lin_tupleinit2(x)[X.indices[1]] * val)
        else:
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
        x = self.poolmlp(x)
        h_graph = torch_scatter_reduce(0, x, datadict["batch"],
                                       datadict["num_graphs"], self.npool)
        return self.pred_lin(h_graph)

# 2 build models

mlpdict = {"dp": args.dp, "norm": args.norm, "act": "silu", "normparam": args.normparam}
if args.sparse:
    model = SpModel(spconvdict[args.conv], npool=args.npool, lpool=args.lpool, outlayer=args.outlayer, mlp=mlpdict)
else:
    model = MaModel(maconvdict[args.conv], npool=args.npool, lpool=args.lpool, outlayer=args.outlayer, mlp=mlpdict)

device = torch.device("cuda")
# 3 data set preprocessing
trn_dataset = ZINC("dataset/ZINC", subset=True, split="train")
val_dataset = ZINC("dataset/ZINC", subset=True, split="val")
tst_dataset = ZINC("dataset/ZINC", subset=True, split="test")
if args.sparse:
    keys = parse_precomputekey(model)
    if args.conv in ["I2GNN"]:
        trn_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_trn", trn_dataset,
            Sppretransform(partial(I2Sampler, hop=3), [""], keys), 0)
        val_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_val", val_dataset,
            Sppretransform(partial(I2Sampler, hop=3), [""], keys), 0)
        tst_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_tst", tst_dataset,
            Sppretransform(partial(I2Sampler, hop=3), [""], keys), 0)
    else:
        trn_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_trn", trn_dataset,
            Sppretransform(partial(KhopSampler, hop=3), [""], keys), 0)
        val_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_val", val_dataset,
            Sppretransform(partial(KhopSampler, hop=3), [""], keys), 0)
        tst_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_tst", tst_dataset,
            Sppretransform(partial(KhopSampler, hop=3), [""], keys), 0)
    trn_dataloader = SpDataloader(trn_dataset,
                                  batch_size=args.bs,
                                  shuffle=True,
                                  drop_last=True,
                                  device=device)
    val_dataloader = SpDataloader(val_dataset, batch_size=args.bs, device=device)
    tst_dataloader = SpDataloader(tst_dataset, batch_size=args.bs, device=device)
else:
    trn_dataset = ParallelPreprocessDataset("dataset/ZINC_trn",
                                            trn_dataset,
                                            pre_transform=Mapretransform(
                                                partial(spdsampler,
                                                              hop=4)),
                                            num_worker=0)
    val_dataset = ParallelPreprocessDataset("dataset/ZINC_val",
                                            val_dataset,
                                            pre_transform=Mapretransform(
                                                partial(spdsampler,
                                                              hop=4)),
                                            num_worker=0)

    tst_dataset = ParallelPreprocessDataset("dataset/ZINC_tst",
                                            tst_dataset,
                                            pre_transform=Mapretransform(
                                                partial(spdsampler,
                                                              hop=4)),
                                            num_worker=0)

    trn_dataloader = MaDataloader(trn_dataset,
                                  batch_size=args.bs,
                                  shuffle=True,
                                  drop_last=True,
                                  device=device)
    val_dataloader = MaDataloader(val_dataset, batch_size=args.bs, device=device)
    tst_dataloader = MaDataloader(tst_dataset, batch_size=args.bs, device=device)

# 4 training process
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
    return (loss / size).item()

out = []
for i in range(args.repeat):
    print(f"runs {i}")
    if args.sparse:
        model = SpModel(spconvdict[args.conv], npool=args.npool, lpool=args.lpool, outlayer=args.outlayer, mlp=mlpdict)
    else:
        model = MaModel(maconvdict[args.conv], npool=args.npool, lpool=args.lpool, outlayer=args.outlayer, mlp=mlpdict)
    model = torch.compile(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                        T_0=args.cosT * len(trn_dataloader), eta_min=args.minlr, K=args.K, K2=args.K2)

    best_val = float("inf")
    tst_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        losss = train(trn_dataloader)
        t2 = time.time()
        val_score = eval(val_dataloader)
        if val_score < best_val:
            best_val = val_score
            tst_score = eval(tst_dataloader)
        t3 = time.time()
        print(
            f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} tst MAE {tst_score:.4f}", flush=True
        )
        if np.isnan(losss) or np.isnan(val_score):
            break
    out.append(tst_score)

print(f"All {np.average(tst_score)} {np.std(tst_score)}")