import torch
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from torch_geometric.data import Data
from subgnn.Spconv import Convs, SubgConv
from subgnn.SpXOperator import pooling_tuple
import torch.nn as nn
from backend.SpTensor import SparseTensor
from utils import MLP
from Emb import x2dims, MultiEmbedding, SingleEmbedding
from typing import List, Optional
from torch import Tensor
from torch_geometric.utils import degree


class InputEncoder(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 x_exdims: List[int],
                 tuple_exdims: List[int],
                 x_zeropad: bool = False,
                 tuple_zeropad: bool = False,
                 dataset=None,
                 **kwargs) -> None:
        super().__init__()
        if dataset is None:
            self.x_encoder = MultiEmbedding(
                emb_dim,
                dims=x_exdims,
                lastzeropad=len(x_exdims) if not x_zeropad else 0,
                **kwargs["emb"])
            self.ea_encoder = lambda *args: None
        else:
            x = dataset.data.x
            ea = dataset.data.edge_attr
            tuplefeat = dataset.data.tuplefeat
            if x is None:
                raise NotImplementedError
            elif x.dtype in [torch.float, torch.float16, torch.float64]:
                self.x_encoder = MLP(x.shape[-1],
                                     emb_dim,
                                     1,
                                     tailact=True,
                                     **kwargs["mlp"])
            elif x.dtype == torch.long:
                dims = x2dims(x)
                self.x_encoder = MultiEmbedding(
                    emb_dim,
                    dims=dims + x_exdims,
                    lastzeropad=len(x_exdims) if not x_zeropad else 0,
                    **kwargs["emb"])
            else:
                raise NotImplementedError

            if tuplefeat is None:
                raise NotImplementedError
            elif tuplefeat.dtype in [
                    torch.float, torch.float16, torch.float64
            ]:
                self.tuple_encoder = MLP(tuplefeat.shape[-1],
                                         emb_dim,
                                         1,
                                         tailact=True,
                                         **kwargs["mlp"])
            elif tuplefeat.dtype == torch.long:
                self.tuple_encoder = SingleEmbedding(
                    emb_dim,
                    torch.max(tuplefeat).item() + 1,
                    **kwargs["emb"])
            else:
                raise NotImplementedError

            if ea is None:
                self.ea_encoder = lambda *args: None
            elif ea.dtype != torch.int64:
                self.ea_encoder = MLP(ea.shape[-1],
                                      emb_dim,
                                      1,
                                      tailact=True,
                                      **kwargs["mlp"])
            elif ea.dtype == torch.int64:
                dims = x2dims(ea)
                self.ea_encoder = MultiEmbedding(emb_dim,
                                                 dims=dims,
                                                 **kwargs["emb"])
            else:
                raise NotImplementedError

    def forward(self, datadict: dict) -> dict:
        datadict["x"] = self.x_encoder(datadict["x"])
        datadict["edge_attr"] = self.ea_encoder(datadict["edge_attr"])
        datadict["tuplefeat"] = self.tuple_encoder(datadict["tuplefeat"])
        return datadict


pool_dict = {
    "sum": SumAggregation,
    "mean": MeanAggregation,
    "max": MaxAggregation
}


class NestedGNN(nn.Module):

    def __init__(self,
                 dataset,
                 num_tasks,
                 num_layer=1,
                 emb_dim=300,
                 gpool="mean",
                 lpool="mean",
                 residual=True,
                 outlayer: int = 1,
                 ln_out: bool = False,
                 **kwargs):
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
            [SubgConv(emb_dim, **kwargs["conv"]) for i in range(num_layer)],
            residual=residual)
        ### Pooling function to generate whole-graph embeddings
        self.gpool = pool_dict[gpool]()
        self.lpool = lpool

        self.data_encoder = InputEncoder(emb_dim, [], [], False, False, dataset, **kwargs)
        self.label_encoder = SingleEmbedding(emb_dim, 100, 1, **kwargs["emb"])

        outdim = self.emb_dim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(
            MLP(outdim, num_tasks, outlayer, tailact=False, **kwargs["mlp"]),
            nn.LayerNorm(num_tasks, elementwise_affine=False)
            if ln_out else nn.Identity())

    def tupleinit(self, tupleid, tuplefeat, x):
        return x[tupleid[0]] * self.lin_tupleinit(x)[tupleid[1]] * tuplefeat

    def forward(self, datadict: dict):
        '''
        TODO: !warning input must be coalesced
        '''
        datadict = self.data_encoder(datadict)
        datadict["XA_tar"] = datadict["tupleid"]
        #datadict["XA_acd"] = None
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
        x = pooling_tuple(X, dim=1, pool=self.lpool)
        h_graph = self.gpool(x, datadict["batch"], dim=0)
        return self.pred_lin(h_graph)
