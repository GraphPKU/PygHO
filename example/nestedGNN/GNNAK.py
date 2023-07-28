from NestedGnn import InputEncoder
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from subgnn.Spconv import Convs, NestedConv
from subgnn.SpXOperator import pooling2nodes, diag2nodes
import torch.nn as nn
from backend.SpTensor import SparseTensor
from subgnn.utils import MLP
from subgnn.Emb import SingleEmbedding
import torch


pool_dict = {
    "sum": SumAggregation,
    "mean": MeanAggregation,
    "max": MaxAggregation
}


class GNNAK(nn.Module):

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
            [NestedConv(emb_dim, **kwargs["conv"]) for i in range(num_layer)],
            residual=residual)
        ### Pooling function to generate whole-graph embeddings
        self.gpool = pool_dict[gpool]()
        self.lpool = lpool

        self.data_encoder = InputEncoder(emb_dim, [], [], False, False, dataset, **kwargs)
        self.label_encoder = SingleEmbedding(emb_dim, 100, 1, **kwargs["emb"])

        outdim = self.emb_dim
        if ln_out:
            print("warning: output is normalized")
        self.mergelin= MLP(3*outdim, outdim, 1, tailact=True, **kwargs["mlp"]),
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
        x1 = pooling2nodes(X, dim=1, pool=self.lpool)
        x2 = pooling2nodes(X, dim=0, pool=self.lpool)
        x3 = diag2nodes(X, dims=None)
        x = self.mergelin(torch.concat((x1,x2,x3), dim=-1))
        h_graph = self.gpool(x, datadict["batch"], dim=0)
        return self.pred_lin(h_graph)
