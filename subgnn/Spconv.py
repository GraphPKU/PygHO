'''
convolution layers for sparse tuple representation
'''
from backend.SpTensor import SparseTensor
import torch.nn as nn
from .SpXOperator import messagepassing_tuple
from torch_geometric.utils import degree
import torch.nn as nn
from .utils import MLP
from typing import List


class NestedConv(nn.Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        tX = X.tuplewiseapply(self.lin)
        ret = messagepassing_tuple(tX, 1, A, 0, "X_1_A_0", datadict, self.aggr)
        return ret


class I2Conv(nn.Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        tX = X.tuplewiseapply(self.lin)
        ret = messagepassing_tuple(tX, 2, A, 0, "X_2_A_0", datadict, self.aggr)
        return ret


class CrossSubgConv(nn.Module):
    '''
    message passing across subgraph
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        tX = SparseTensor(X.indices,
                          self.lin(X.values),
                          shape=X.shape,
                          is_coalesced=True)
        return messagepassing_tuple(A, 1, tX, 0, "A_1_X_0", datadict,
                                    self.aggr)


class TwoFWLConv(nn.Module):
    '''
    output = X1X2
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin1 = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])
        self.lin2 = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: SparseTensor, datadict: dict) -> SparseTensor:
        X1 = X.tuplewiseapply(self.lin1)
        X2 = X.tuplewiseapply(self.lin2)
        return messagepassing_tuple(X1, 1, X2, 0, "X_1_X_0", datadict,
                                    self.aggr)


class Convs(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, convlist: List[nn.Module], residual=False, **kwargs):
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

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        for conv in self.convs:
            tX: SparseTensor = conv(X, A, datadict)
            if self.residual:
                X = tX.tuplewiseapply(lambda val: val + X.values)
            else:
                X = tX
        return X


if __name__ == "__main__":
    pass