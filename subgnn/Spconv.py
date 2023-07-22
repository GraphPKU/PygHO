from backend.SpTensor import SparseTensor
import torch.nn as nn
from .SpXOperator import messagepassing_tuple, pooling_tuple
from torch_geometric.utils import degree
import torch.nn as nn
from .utils import MLP
from typing import List

class SubgConv(nn.Module):
    '''
    message passing within each subgraph
    '''
    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor, datadict: dict)->SparseTensor:
        tX = SparseTensor(X.indices, self.lin(X.values), shape=X.shape, is_coalesced=True)
        # print(tX.nnz, A.nnz, datadict["XA_tar"].shape, datadict["XA_acd"].max(dim=-1)[0])
        ret = messagepassing_tuple(A, tX, "XA", datadict, self.aggr)
        return ret 

class CrossSubgConv(nn.Module):
    '''
    message passing across subgraph
    '''
    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor, datadict: dict)->SparseTensor:
        tX = SparseTensor(X.indices, self.lin(X.values), shape=X.shape, is_coalesced=True)
        return messagepassing_tuple(A, tX, "AX", datadict, self.aggr)


class TwoFWLConv(nn.Module):
    '''
    output = X1X2
    '''
    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin1 = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])
        self.lin2 = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, datadict: dict)->SparseTensor:
        X1 = SparseTensor(X.indices, self.lin1(X.values), shape=X.shape, is_coalesced=True)
        X2 = SparseTensor(X.indices, self.lin2(X.values), shape=X.shape, is_coalesced=True)
        return messagepassing_tuple(X1, X2, "XX", datadict, self.aggr)

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

    def forward(self, X: SparseTensor, A: SparseTensor, datadict: dict)->SparseTensor:
        for conv in self.convs:
            tX = conv(X, A, datadict)
            if self.residual:
                X = SparseTensor(X.indices, tX.values+X.values, X.shape, is_coalesced=True)
            else:
                X = tX
        return X


if __name__ == "__main__":
    pass