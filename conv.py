import torch
from torch_geometric.nn import MessagePassing
from Emb import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
import torch.nn as nn
from utils import MLP
from torch import Tensor
from utils import BatchNorm

### GCN convolution along the graph structure
class GCNConv(MessagePassing):

    def __init__(self, emb_dim, use_elin: bool=False, elin_layer: int=1, norm: str="gcn", **kwargs):
        if norm in ["gcn", "sum"]:
            super(GCNConv, self).__init__(aggr='add', node_dim=-2)
        elif norm in "mean":
            super(GCNConv, self).__init__(aggr="mean", node_dim=-2)
        elif norm in "max":
            super(GCNConv, self).__init__(aggr="max", node_dim=-2)
        else:
            raise NotImplementedError
        self.norm = norm
        self.bond_encoder = nn.Identity() if not use_elin else MLP(emb_dim, emb_dim, elin_layer, False, **kwargs["mlp"])

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        norm = None
        if self.norm == "gcn":
            row, col = edge_index

            deg = degree(row, x.shape[-2], dtype=x.dtype) 
            deg += deg == 0
            deg.rsqrt_()

            norm = deg[row] * deg[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) 

    def message(self, x_j, edge_attr, norm):
        if norm is None:
            return x_j * edge_attr
        else:
            return norm.view(-1, 1) * x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 JK="last",
                 residual=False,
                 norm="gcn",
                 use_elin=False,
                 mlplayer=1,
                 dims=None,
                 **kwargs):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super().__init__()
        self.num_layer = num_layer
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim, dims=dims)
        self.bond_encoder = BondEncoder(emb_dim)
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GCNConv(emb_dim, use_elin, norm, **kwargs))
            self.lins.append(MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"]))

    def pregnn(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        h_list = [self.atom_encoder(x)]
        edge_attr = self.bond_encoder(edge_attr)
        return h_list, edge_index, edge_attr, batch

    def postgnn(self, h_list):
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        else:
            raise NotImplementedError
        return node_representation

    def forward(self, batched_data):
        h_list, edge_index, edge_attr, batch = self.pregnn(batched_data)

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.lins[layer](h)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        return self.postgnn(h_list)


if __name__ == "__main__":
    pass