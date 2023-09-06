import torch
from torch_geometric.datasets import ZINC
from pygho.hodata import SpDataloader, Sppretransform, SubgDatasetClass
from pygho.hodata.SpTupleSampler import KhopSampler
from functools import partial
import torch
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from pygho.honn.Spconv import Convs, NestedConv, DSSGINConv, SUNConv
from pygho.honn.SpXOperator import pooling2nodes, diag2nodes
import torch.nn as nn
from pygho import SparseTensor
from pygho.honn.utils import MLP
from torch_geometric.data import DataLoader as PygDataloader
import torch.nn.functional as F
import numpy as np

'''
model definition
'''

class InputEncoder(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.x_encoder = nn.Embedding(32, emb_dim)
        self.ea_encoder = nn.Embedding(16, emb_dim)
        self.tuplefeat_encoder = nn.Embedding(16, emb_dim)

    def forward(self, datadict: dict) -> dict:
        datadict["x"] = self.x_encoder(datadict["x"].flatten())
        datadict["edge_attr"] = self.ea_encoder(datadict["edge_attr"])
        datadict["tuplefeat"] = self.tuplefeat_encoder(
            datadict["tuplefeat"])
        return datadict


pool_dict = {
    "sum": SumAggregation,
    "mean": MeanAggregation,
    "max": MaxAggregation
}


class NestedGNN(nn.Module):

    def __init__(self,
                 num_tasks=1,
                 num_layer=5,
                 emb_dim=256,
                 aggr="sum",
                 npool="sum",
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
            ln_out: use layernorm in output, a normalization method for classification problem
        '''

        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        self.lin_tupleinit = nn.Linear(emb_dim, emb_dim)

        ### GNN to generate node embeddings
        self.subggnns = Convs(
            [NestedConv(emb_dim, 1, aggr, mlp) for i in range(num_layer)],
            residual=residual)
        ### Pooling function to generate whole-graph embeddings
        self.npool = {
            "sum": SumAggregation,
            "mean": MeanAggregation,
            "max": MaxAggregation
        }[npool]()
        self.lpool = lpool
        self.data_encoder = InputEncoder(emb_dim)

        outdim = self.emb_dim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(
            MLP(outdim, num_tasks, outlayer, tailact=False, **mlp),
            nn.LayerNorm(num_tasks, elementwise_affine=False)
            if ln_out else nn.Identity())

    def tupleinit(self, tupleid, tuplefeat, x):
        return x[tupleid[0]] * self.lin_tupleinit(x)[tupleid[1]] * tuplefeat

    def forward(self, datadict: dict):
        datadict = self.data_encoder(datadict)
        A = SparseTensor(datadict["edge_index"],
                         datadict["edge_attr"],
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
        X = SparseTensor(datadict["tupleid"],
                         self.tupleinit(datadict["tupleid"],
                                        datadict["tuplefeat"], datadict["x"]),
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
        X = self.subggnns.forward(X, A, datadict)
        x = pooling2nodes(X, dims=[1], pool=self.lpool)
        h_graph = self.npool(x, datadict["batch"], dim=0)
        return self.pred_lin(h_graph)


class GNNAK(NestedGNN):

    def __init__(self, num_tasks=1, num_layer=5, emb_dim=256, aggr="sum", npool="sum", lpool="max", residual=True, outlayer: int = 1, ln_out: bool = False, mlp: dict = {}):
        super().__init__(num_tasks, num_layer, emb_dim, aggr, npool, lpool, residual, outlayer, ln_out, mlp)
        self.mergelin= MLP(3*emb_dim, emb_dim, 1, tailact=True, **mlp) 
        self.subggnns = Convs(
            [SUNConv(emb_dim, 1, aggr, mlp) for i in range(num_layer)],
            residual=residual)

    def forward(self, datadict: dict):
        '''
        TODO: !warning input must be coalesced
        '''
        datadict = self.data_encoder(datadict)
        A = SparseTensor(datadict["edge_index"],
                         datadict["edge_attr"],
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] + list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
        X = SparseTensor(datadict["tupleid"],
                         self.tupleinit(datadict["tupleid"],
                                        datadict["tuplefeat"], datadict["x"]),
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
        X = self.subggnns.forward(X, A, datadict)
        x1 = pooling2nodes(X, dims=1, pool=self.lpool)
        x2 = pooling2nodes(X, dims=0, pool=self.lpool)
        x3 = diag2nodes(X, dims=[0, 1])
        x = self.mergelin(torch.concat((x1,x2,x3), dim=-1))
        h_graph = self.npool(x, datadict["batch"], dim=0)
        return self.pred_lin(h_graph)

class DSSGIN(NestedGNN):

    def __init__(self, num_tasks=1, num_layer=5, emb_dim=256, aggr="sum", npool="sum", lpool="max", residual=True, outlayer: int = 1, ln_out: bool = False, mlp: dict = {}):
        super().__init__(num_tasks, num_layer, emb_dim, aggr, npool, lpool, residual, outlayer, ln_out, mlp)
        self.subggnns = Convs(
            [DSSGINConv(emb_dim, 1, "sum", mlp) for i in range(num_layer)],
            residual=residual)

    def forward(self, datadict: dict):
        '''
        TODO: !warning input must be coalesced
        '''
        datadict = self.data_encoder(datadict)
        A = SparseTensor(datadict["edge_index"],
                         datadict["edge_attr"],
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
        X = SparseTensor(datadict["tupleid"],
                         self.tupleinit(datadict["tupleid"],
                                        datadict["tuplefeat"], datadict["x"]),
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
        X = self.subggnns.forward(X, A, datadict)
        x = pooling2nodes(X, dims=1, pool=self.lpool)
        h_graph = self.npool(x, datadict["batch"], dim=0)
        return self.pred_lin(h_graph)


'''
main func
'''

model = NestedGNN(mlp={
    "dp": 0,
    "norm": "ln",
    "act": "relu",
    "normparam": 0.1
})

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
trn_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="train",
                   pre_transform=Sppretransform(partial(KhopSampler, hop=3), ["X_1_A_0_acd"]))
val_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="val",
                   pre_transform=Sppretransform(partial(KhopSampler, hop=3), ["X_1_A_0_acd"]))
tst_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="test",
                   pre_transform=Sppretransform(partial(KhopSampler, hop=3), ["X_1_A_0_acd"]))
trn_dataloader = SpDataloader(trn_dataset, batch_size=256, shuffle=True, drop_last=True)
val_dataloader = SpDataloader(val_dataset, batch_size=256)
tst_dataloader = SpDataloader(tst_dataset, batch_size=256)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=40*len(trn_dataloader))

device = torch.device("cuda")
model = model.to(device)
def train(dataloader):
    model.train()
    losss = []
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        datadict = batch.to_dict()
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
        pred = model(datadict)
        loss += F.l1_loss(datadict["y"].unsqueeze(-1), pred, reduction="sum")
        size += pred.shape[0]
    return loss / size

best_val = float("inf")
tst_score = float("inf")
import time

for epoch in range(1, 1001):
    t1 = time.time()
    losss = train(trn_dataloader)
    t2 = time.time()
    val_score = eval(val_dataloader)
    if val_score < best_val:
        best_val = val_score
        tst_score = eval(tst_dataloader)
    t3 = time.time()
    print(f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f}  l1loss {losss:.4f} val MAE {val_score:.4f} tst MAE {tst_score:.4f}")

