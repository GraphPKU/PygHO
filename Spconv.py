import torch
from torch import Tensor
from backend.SpTensor import SparseTensor
import torch.nn as nn
from backend.SpXOperator import messagepassing_tuple, pooling_tuple
from torch_geometric.utils import degree
import torch.nn as nn
from utils import MLP
from typing import List

### GCN convolution along the graph structure
class SubgConv(nn.Module):

    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor, datadict: dict)->SparseTensor:
        tX = SparseTensor(X.indices, self.lin(X.values), shape=X.shape, is_coalesced=True)
        return messagepassing_tuple(A, tX, "XA", datadict, self.aggr)
    

### GCN convolution along the graph structure
class CrossSubgConv(nn.Module):

    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor, datadict: dict)->SparseTensor:
        tX = SparseTensor(X.indices, self.lin(X.values), shape=X.shape, is_coalesced=True)
        return messagepassing_tuple(A, tX, "AX", datadict, self.aggr)


### GNN to generate node embedding
class Convs(nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self,
                 convlist: List[nn.Module],
                 residual=False,
                 **kwargs):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super().__init__()
        self.num_layer = len(convlist)
        ### add residual connection or not
        self.residual = residual

        ###List of GNNs
        self.convs = nn.ModuleList(convlist)

    def forward(self, A: SparseTensor, X: SparseTensor, datadict)->SparseTensor:
        for conv in self.convs:
            tX = conv(A, X, datadict)
            if self.residual:
                X = SparseTensor(X.indices, tX.values+X.values, X.shape, is_coalesced=True)
            else:
                X = tX
        return X


if __name__ == "__main__":
    pass