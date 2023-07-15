import torch
from torch_geometric.nn.aggr import Set2Set, SumAggregation, MeanAggregation, MaxAggregation
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from conv import GNN_node
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add, scatter_min
from utils import MLP
from Emb import x2dims, MultiEmbedding, SingleEmbedding
from typing import List, Optional
from torch import Tensor
from torch_geometric.utils import degree


class InputEncoder(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 exdims: List[int],
                 zeropad: bool = False,
                 dataset=None,
                 **kwargs) -> None:
        super().__init__()
        if dataset is None:
            self.x_encoder = MultiEmbedding(
                emb_dim,
                dims=exdims,
                lastzeropad=len(exdims) if not zeropad else 0, **kwargs["emb"])
            self.ea_encoder = lambda *args: None
        else:
            x = dataset.data.x
            ea = dataset.data.edge_attr

            if x is None:
                raise NotImplementedError
            elif x.dtype != torch.int64:
                self.x_encoder = MLP(x.shape[-1],
                                     emb_dim,
                                     1,
                                     tailact=True,
                                     **kwargs["mlp"])
            elif x.dtype == torch.int64:
                dims = x2dims(x)
                self.x_encoder = MultiEmbedding(
                    emb_dim,
                    dims=dims + exdims,
                    lastzeropad=len(exdims) if not zeropad else 0, **kwargs["emb"])
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
                self.ea_encoder = MultiEmbedding(emb_dim, dims=dims, **kwargs["emb"])
            else:
                raise NotImplementedError

    def forward(self, batched_data: Data):
        batched_data.x = self.x_encoder(batched_data.x)
        batched_data.edge_attr = self.ea_encoder(batched_data.edge_attr)
        batched_data.subg_edge_attr = self.ea_encoder(batched_data.subg_edge_attr)
        return batched_data


pool_dict = {"sum": SumAggregation, "mean": MeanAggregation, "max": MaxAggregation}

class NestedGNN(nn.Module):

    def __init__(self,
                 dataset,
                 num_tasks,
                 num_layer=1,
                 emb_dim=300,
                 gpool="mean",
                 lpool="mean",
                 outlayer: int = 1,
                 ln_out: bool=False,
                 **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        ### GNN to generate node embeddings
        self.ggnn_nodes = nn.ModuleList([GNN_node(emb_dim=emb_dim,
                                     **kwargs["ggnn"]) for _ in range(num_layer)])
        self.lgnn_nodes = nn.ModuleList([GNN_node(emb_dim=emb_dim,
                                     **kwargs["lgnn"]) for _ in range(num_layer)])
        ### Pooling function to generate whole-graph embeddings
        self.gpool = pool_dict[gpool]()
        self.lpool = pool_dict[lpool]()

        self.data_encoder = InputEncoder(emb_dim, [], False, dataset, **kwargs)
        self.label_encoder = SingleEmbedding(emb_dim, 100, 1, **kwargs["emb"])
        
        outdim = self.emb_dim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(MLP(outdim,
                            num_tasks,
                            outlayer,
                            tailact=False,
                            **kwargs["mlp"]), nn.LayerNorm(num_tasks, elementwise_affine=False) if ln_out else nn.Identity())
        
    def subgforward(self, batched_data, layer):
        h_node = self.lgnn_nodes[layer](batched_data, issubgraph=True)
        subg_nodeidx = batched_data.subg_nodeidx
        batched_data.x = batched_data.x + self.lpool(h_node, subg_nodeidx, dim_size=batched_data.num_nodes, dim=-2)
        return batched_data
    
    def graphforward(self, batched_data, layer):
        h_node = self.ggnn_nodes[layer](batched_data)
        batched_data.x = batched_data.x + h_node
        return batched_data

    def forward(self, batched_data):
        batched_data = self.data_encoder(batched_data)
        batched_data.subg_nodelabel = self.label_encoder(batched_data.subg_nodelabel.to(torch.long))
        for layer in range(self.num_layer):
            self.subgforward(batched_data, layer)
            self.graphforward(batched_data, layer)
        h_graph = self.gpool(batched_data.x, batched_data.batch, dim=-2)
        return self.pred_lin(h_graph)

