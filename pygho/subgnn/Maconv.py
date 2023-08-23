'''
convolution layers for dense tuple representation
'''
from backend.MaTensor import MaskedTensor
from backend.SpTensor import SparseTensor
import torch.nn as nn
from .MaXOperator import messagepassing_tuple
import torch.nn as nn
from .utils import MLP
from typing import List, Union


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

    def forward(self, X: MaskedTensor, A: Union[MaskedTensor,
                                                SparseTensor]) -> MaskedTensor:
        tX = X.tuplewiseapply(self.lin)
        # print(tX.nnz, A.nnz, datadict["XA_tar"].shape, datadict["XA_acd"].max(dim=-1)[0])
        ret = messagepassing_tuple(tX, A, self.aggr)
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
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **mlp)

    def forward(self, X: MaskedTensor, A: Union[MaskedTensor,
                                                SparseTensor]) -> SparseTensor:
        tX = X.tuplewiseapply(self.lin)
        return messagepassing_tuple(A, tX, self.aggr)


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

    def forward(self, X: MaskedTensor) -> SparseTensor:
        X1 = X.tuplewiseapply(self.lin1)
        X2 = X.tuplewiseapply(self.lin2)
        return messagepassing_tuple(X1, X2, self.aggr)


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

    def forward(self, X: MaskedTensor, A: Union[SparseTensor,
                                                MaskedTensor]) -> MaskedTensor:
        for conv in self.convs:
            tX: MaskedTensor = conv(X, A)
            if self.residual:
                X = tX.tuplewiseapply(lambda x: x + X.data)
            else:
                X = tX
        return X


if __name__ == "__main__":
    pass