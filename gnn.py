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
                                     multiparams=1,
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
                                      multiparams=1,
                                      **kwargs["mlp"])
            elif ea.dtype == torch.int64:
                dims = x2dims(ea)
                self.ea_encoder = MultiEmbedding(emb_dim, dims=dims, **kwargs["emb"])
            else:
                raise NotImplementedError

    def forward(self, batched_data: Data):
        batched_data.x = self.x_encoder(batched_data.x)
        batched_data.edge_attr = self.ea_encoder(batched_data.edge_attr)
        return batched_data


class GNN(nn.Module):

    def __init__(self,
                 num_tasks,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 graph_pooling="mean",
                 outlayer: int = 1,
                 ln_out: bool=False,
                 multiconv: int = 1,
                 **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        #kwargs["mlp"]["bn"] = False
        #kwargs["mlp"]["ln"] = True

        ### GNN to generate node embeddings
        if virtual_node:
            raise NotImplementedError
        else:
            self.gnn_node = GNN_node(num_layer,
                                     emb_dim,
                                     JK=JK,
                                     residual=residual,
                                     norm=norm,
                                     multiconv=multiconv,
                                     **kwargs)
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumAggregation()
        elif self.graph_pooling == "mean":
            self.pool = MeanAggregation()
        elif self.graph_pooling == "max":
            self.pool = MaxAggregation()
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")
        outdim = 2 * self.emb_dim if graph_pooling == "set2set" else self.emb_dim
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(MLP(outdim,
                            num_tasks,
                            outlayer,
                            tailact=False,
                            **kwargs["mlp"]), nn.LayerNorm(num_tasks, elementwise_affine=False) if ln_out else nn.Identity())

    def forward(self, batched_data, idx: int):
        h_node = self.gnn_node(batched_data, idx)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.pred_lin(h_graph)


class UniAnchorGNN(GNN):

    def __init__(self,
                 num_tasks,
                 num_anchor=1,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 graph_pooling="mean",
                 rand_anchor: bool = False,
                 multi_anchor: int = 1,
                 outlayer: int = 1,
                 policy_detach: bool = False,
                 dataset=None,
                 randinit=False,
                 fullsample=False,
                 **kwargs):
        super().__init__(num_tasks,
                         num_layer,
                         emb_dim,
                         norm,
                         virtual_node,
                         residual,
                         JK,
                         graph_pooling,
                         outlayer=outlayer,
                         multiconv=num_anchor+1,
                         **kwargs)
        self.fullsample = fullsample
        if fullsample:
            assert num_anchor < 2, "TODO"
        self.randinit = randinit
        if randinit:
            print("warning: model using random init")
        self.policy_detach = policy_detach
        self.num_anchor = num_anchor
        self.multi_anchor = multi_anchor
        self.rand_anchor = rand_anchor
        self.data_encoder = InputEncoder(emb_dim, [], False, dataset, **kwargs)
        self.anchor_encoder = SingleEmbedding(emb_dim, num_anchor + 1, 1, **kwargs["emb"])
        self.h_node = None

    def get_h_node(self, batched_data, idx: int):
        self.h_node = self.gnn_node(batched_data, idx)

    def fresh_h_node(self):
        self.h_node = None

    def graph_forward(self, batched_data):
        assert self.h_node is not None
        h_graph = self.pool(self.h_node, batched_data.batch)
        return self.pred_lin(h_graph)

    def preprocessdata(self, batched_data: Data):
        batched_data = self.data_encoder(batched_data)
        batched_data.x = batched_data.x.unsqueeze(0).expand(self.multi_anchor, -1, -1)
        return batched_data
    
    def forward(self, batched_data, T: float = 1):
        batched_data = self.preprocessdata(batched_data)
        if self.randinit:
            batched_data.x = batched_data.x + torch.rand_like(batched_data.x)
        preds = []
        self.get_h_node(batched_data, self.num_anchor)
        preds.append(self.graph_forward(batched_data))
        finalpred = preds[-1].mean(dim=0)
        self.fresh_h_node()
        return torch.stack(preds, dim=0), None, None, finalpred
