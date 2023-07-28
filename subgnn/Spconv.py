'''
convolution layers for sparse tuple representation
'''
from backend.SpTensor import SparseTensor
import torch.nn as nn
import torch
from .SpXOperator import messagepassing_tuple, diag2nodes, unpooling4node, pooling2nodes, messagepassing_node
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
                 mlp: dict = {}):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **mlp)

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        tX = X.tuplewiseapply(self.lin)
        ret = messagepassing_tuple(tX, 1, A, 0, "X_1_A_0", datadict, self.aggr)
        return ret


class DSSGINConv(nn.Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 subgpool: str = "max",
                 mlp: dict = {}):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **mlp)
        self.nestedconv = NestedConv(emb_dim, mlplayer, aggr, mlp)
        self.subgpool = subgpool

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        ret1 = self.nestedconv(X, A, datadict)
        nodex = self.lin(pooling2nodes(X, dims=1, pool=self.subgpool))
        nodex = messagepassing_node(A, nodex, self.aggr)
        ret2 = unpooling4node(nodex, X, dim=1)
        return ret2.tuplewiseapply(lambda x: x + ret1.values)


class SUNConv(nn.Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 mlp: dict = {}):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(6 * emb_dim, emb_dim, mlplayer, True, **mlp)

    def forward(self, X: SparseTensor, A: SparseTensor,
                datadict: dict) -> SparseTensor:
        tX = X.tuplewiseapply(self.lin)
        x1 = messagepassing_tuple(tX, 1, A, 0, "X_1_A_0", datadict, self.aggr)
        x2 = diag2nodes(X, [0, 1])
        x2_1 = unpooling4node(x2, X, 0)
        x2_2 = unpooling4node(x2, X, 1)
        x3 = unpooling4node(pooling2nodes(X, 1, "sum"), X, 1)
        x4 = pooling2nodes(X, 0, "sum")
        x4_1 = unpooling4node(x4, X, 0)
        x5 = messagepassing_node(A, x4, "sum")
        x5_1 = unpooling4node(x5, X, 0)
        ret = X.tuplewiseapply(lambda x: self.lin(
            torch.concat((x1.values, x2_1.values, x2_2.values, x3.values, x4_1.
                          values, x5_1.values))))
        return ret


class I2Conv(nn.Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 emb_dim: int,
                 mlplayer: int,
                 aggr: str = "sum",
                 mlp: dict = {}):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **mlp)

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
                 mlp: dict = {}):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, mlp)

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
                 mlp: dict = {}):
        super().__init__()
        self.aggr = aggr
        self.lin1 = MLP(emb_dim, emb_dim, mlplayer, True, **mlp)
        self.lin2 = MLP(emb_dim, emb_dim, mlplayer, True, **mlp)

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

    def __init__(self, convlist: List[nn.Module], residual=False):
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