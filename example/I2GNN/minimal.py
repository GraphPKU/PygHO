import torch
from torch_geometric.datasets import ZINC
from subgdata.SpData import sp_datapreprocess
from subgdata.SpSubgSampler import I2Sampler
from functools import partial
import torch
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from subgnn.Spconv import Convs, I2Conv
from subgnn.SpXOperator import pooling2nodes, pooling2tuple
import torch.nn as nn
from backend.SpTensor import SparseTensor
from subgnn.utils import MLP
from torch_geometric.data import DataLoader as PygDataloader
import torch.nn.functional as F
import numpy as np


class InputEncoder(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.x_encoder = nn.Embedding(32, emb_dim)
        self.ea_encoder = nn.Embedding(16, emb_dim)
        self.tuplefeat_encoder = nn.Embedding(16, emb_dim // 2)

    def forward(self, datadict: dict) -> dict:
        # print(datadict["x"].shape, datadict["edge_attr"].shape, datadict["tuplefeat"].shape)
        datadict["x"] = self.x_encoder(datadict["x"].flatten())
        datadict["edge_attr"] = self.ea_encoder(datadict["edge_attr"])
        datadict["tuplefeat"] = self.tuplefeat_encoder(datadict["tuplefeat"]).flatten(-2, -1)
        return datadict


class I2GNN(nn.Module):

    def __init__(self,
                 num_tasks=1,
                 num_layer=4,
                 emb_dim=256,
                 npool="sum",
                 epool="sum",
                 lpool="max",
                 residual=True,
                 outlayer: int = 1,
                 ln_out: bool = False,
                 mlp: dict = {}):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        self.lin_tupleinit = nn.Linear(emb_dim, emb_dim)

        ### GNN to generate node embeddings
        self.subggnns = Convs(
            [I2Conv(emb_dim, 1, "sum", mlp) for i in range(num_layer)],
            residual=residual)
        ### Pooling function to generate whole-graph embeddings
        self.npool = {
            "sum": SumAggregation,
            "mean": MeanAggregation,
            "max": MaxAggregation
        }[npool]()
        self.lpool = lpool
        self.epool = epool
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
        X = pooling2tuple(X, dims=[2], pool=self.lpool)
        x = pooling2nodes(X, dims=[1], pool=self.epool)
        h_graph = self.npool(x, datadict["batch"], dim=0)
        return self.pred_lin(h_graph)


model = I2GNN(mlp={
            "dp": 0,
            "norm": "ln",
            "act": "relu",
            "normparam": 0.1
        })
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
trn_dataset = ZINC("dataset/ZINC",
                   subset=True,
                   split="train",
                   pre_transform=partial(sp_datapreprocess,
                                         subgsampler=partial(I2Sampler, hop=3),
                                         keys=["X_2_A_0_acd"]))
dataloader = PygDataloader(trn_dataset, batch_size=128)
device = torch.device("cuda")
model = model.to(device)
for epoch in range(1, 101):
    losss = []
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        datadict = batch.to_dict()
        pred = model(datadict)
        loss = F.l1_loss(datadict["y"].unsqueeze(-1), pred, reduction="mean")
        loss.backward()
        optimizer.step()
        losss.append(loss)
    losss = np.average(list(map(lambda x: x.item(), losss)))
    print(f"epoch {epoch} l1loss {losss:.4f} ")

