import numpy as np
from torch import Tensor
from functools import partial
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import ZINC
from pygho import SparseTensor
from pygho.hodata import SpDataloader, Sppretransform, ParallelPreprocessDataset
from pygho.hodata.SpTupleSampler import KhopSampler
from pygho.honn.SpOperator import parse_precomputekey
from pygho.backend.utils import torch_scatter_reduce

from pygho.honn.Conv import NGNNConv
from pygho.honn.TensorOp import OpPoolingSubg2D
from pygho.honn.utils import MLP


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


class SpModel(nn.Module):

    def __init__(self, num_tasks=1, num_layer=6, hiddim=128, mlp: dict = {}):
        '''
            num_tasks (int): number of output dimensions
            npool: node level pooling
            lpool: subgraph pooling
            aggr: aggregation scheme in MPNN on each subgraph
            ln_out: use layernorm in output, 
                a normalization method for classification problem
        '''

        super().__init__()

        self.lin_tupleinit0 = nn.Linear(hiddim, hiddim)
        self.lin_tupleinit1 = nn.Linear(hiddim, hiddim)

        self.npool = "sum"
        self.lpool = OpPoolingSubg2D("S", "mean")
        self.poolmlp = MLP(hiddim, hiddim, 1, tailact=True, **mlp)
        self.data_encoder = InputEncoderSp(hiddim)

        self.pred_lin = MLP(hiddim, num_tasks, 2, tailact=False, **mlp)

        mlp.update({"numlayer": 1, "tailact": True})
        self.subggnns = nn.ModuleList([
            NGNNConv(hiddim, hiddim, "sum", "SS", mlp)
            for _ in range(num_layer)
        ])

    def tupleinit(self, X: SparseTensor, x: Tensor):
        subgx0 = X.unpooling_fromdense1dim(0, self.lin_tupleinit0(x))
        subgx1 = X.unpooling_fromdense1dim(1, self.lin_tupleinit1(x))
        return X.tuplewiseapply(lambda val: subgx0.values * subgx1.values * val)

    def forward(self, datadict: dict):
        datadict = self.data_encoder(datadict)
        A = datadict["A"]
        X = datadict["X"]
        x = datadict["x"]
        X = self.tupleinit(X, x)
        for conv in self.subggnns:
            tX = conv.forward(A, X, datadict)
            X = X.add(tX, True)
        x = self.lpool(X)
        x = self.poolmlp(x)
        h_graph = torch_scatter_reduce(0, x, datadict["batch"],
                                       datadict["num_graphs"], self.npool)
        return self.pred_lin(h_graph)


# 2 build models

mlpdict = {
    "norm": "bn",
    "act": "silu",
    "dp": 0.0
}  # hyperparameter for multi-layer perceptrons in model. dropout ratio=0, use batchnorm, use SiLU activition function

model = SpModel(mlp=mlpdict)

device = torch.device("cuda")
# 3 data set preprocessing

# load pyg data
trn_dataset = ZINC("dataset/ZINC", subset=True, split="train")
val_dataset = ZINC("dataset/ZINC", subset=True, split="val")
tst_dataset = ZINC("dataset/ZINC", subset=True, split="test")

# initialize tuple feature
keys = parse_precomputekey(model)
trn_dataset = ParallelPreprocessDataset(
    "dataset/ZINC_trn", trn_dataset,
    Sppretransform(partial(KhopSampler, hop=3), [""], keys), 0)
val_dataset = ParallelPreprocessDataset(
    "dataset/ZINC_val", val_dataset,
    Sppretransform(partial(KhopSampler, hop=3), [""], keys), 0)
tst_dataset = ParallelPreprocessDataset(
    "dataset/ZINC_tst", tst_dataset,
    Sppretransform(partial(KhopSampler, hop=3), [""], keys), 0)

# create sparse dataloader
batch_size=128
trn_dataloader = SpDataloader(trn_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              device=device)
val_dataloader = SpDataloader(val_dataset,
                              batch_size=batch_size,
                              device=device)
tst_dataloader = SpDataloader(tst_dataset,
                              batch_size=batch_size,
                              device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model = model.to(device)


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

best_val = float("inf")
tst_score = float("inf")
for epoch in range(1, 100 + 1):
    t1 = time.time()
    losss = train(trn_dataloader)
    t2 = time.time()
    val_score = eval(val_dataloader)
    if val_score < best_val:
        best_val = val_score
        tst_score = eval(tst_dataloader)
    t3 = time.time()
    print(
        f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} tst MAE {tst_score:.4f}"
    )
    if np.isnan(losss) or np.isnan(val_score):
        break
out.append(tst_score)

print(f"All {np.average(tst_score)} {np.std(tst_score)}")