import torch
from torch import Tensor
from torch_geometric.datasets import ZINC
from pygho.hodata import SubgDatasetClass, Mapretransform, MaDataloader
from pygho.hodata.MaTupleSampler import spdsampler
from functools import partial
import torch
import torch.nn as nn
from pygho.honn.Maconv import CrossSubgConv, NestedConv, Convs
from pygho.honn.MaXOperator import pooling_tuple
from pygho.honn.utils import MLP
from pygho import MaskedTensor
import torch.nn.functional as F
import numpy as np


class InputEncoder(nn.Module):

    def __init__(self,
                 emb_dim: int) -> None:
        super().__init__()
        self.x_encoder = nn.Embedding(32, emb_dim)
        self.ea_encoder = nn.Embedding(16, emb_dim)
        self.tuplefeat_encoder = nn.Embedding(16, emb_dim)

    def forward(self, datadict: dict) -> dict:
        datadict["x"] = self.x_encoder(datadict["x"].squeeze(-1))
        datadict["A"] = datadict["A"].tuplewiseapply(self.ea_encoder)
        datadict["tuplefeat"] = self.tuplefeat_encoder(datadict["tuplefeat"])
        return datadict

class SSWL(nn.Module):

    def __init__(self,
                 num_tasks=1,
                 num_layer=1,
                 emb_dim=128,
                 gpool="mean",
                 lpool="mean",
                 residual=True,
                 outlayer: int = 2,
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
        self.subggnns = Convs(sum([[NestedConv(emb_dim, 2, "sum", mlp)] +
                                   [CrossSubgConv(emb_dim, 2, "sum", mlp)]
                                   for i in range(num_layer)],
                                  start=[]),
                              residual=residual)
        ### Pooling function to generate whole-graph embeddings
        self.gpool = gpool
        self.lpool = lpool

        self.data_encoder = InputEncoder(emb_dim)

        outdim = self.emb_dim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(
            MLP(outdim, num_tasks, outlayer, tailact=False, **mlp),
            nn.LayerNorm(num_tasks, elementwise_affine=False)
            if ln_out else nn.Identity())

    def tupleinit(self, tuplefeat: Tensor, x: Tensor, tuplemask: Tensor)->MaskedTensor:
        return MaskedTensor(x.unsqueeze(1) * self.lin_tupleinit(x).unsqueeze(2) * tuplefeat, mask=tuplemask)

    def forward(self, datadict: dict):
        '''
        TODO: !warning input must be coalesced
        '''
        datadict = self.data_encoder(datadict)
        A = datadict["A"]
        X = self.tupleinit(datadict["tuplefeat"], datadict["x"], datadict["tuplemask"])
        X = self.subggnns.forward(X, A)
        x = pooling_tuple(X, dim=1, pool=self.lpool)
        x = MaskedTensor(x, datadict["nodemask"])
        h_graph = getattr(x, self.gpool)(dim=1)
        return self.pred_lin(h_graph)


model = SSWL(mlp={
            "dp": 0,
            "norm": "ln",
            "act": "silu",
            "normparam": 0.1
        })
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
trn_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="train",
                   pre_transform=Mapretransform(partial(spdsampler, hop=4)))
val_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="val",
                   pre_transform=Mapretransform(partial(spdsampler, hop=4)))
tst_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="test",
                   pre_transform=Mapretransform(partial(spdsampler, hop=4)))
device = torch.device("cuda")
trn_dataloader = MaDataloader(trn_dataset, batch_size=256, device=device, shuffle=True, drop_last=True)
val_dataloader = MaDataloader(val_dataset, batch_size=256, device=device)
tst_dataloader = MaDataloader(tst_dataset, batch_size=256, device=device)
model = model.to(device)


def train(dataloader):
    model.train()
    losss = []
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        # num_nodes =  batch.ptr[2]-batch.ptr[1]
        # print(batch.x[1][:num_nodes], num_nodes, batch.nodemask[1][:num_nodes+1], batch.A.indices, batch.tuplemask[1][:num_nodes+1, :num_nodes+1], batch.tuplefeat[1][:num_nodes, :num_nodes].flatten())
        optimizer.zero_grad()
        datadict = batch.to_dict()
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