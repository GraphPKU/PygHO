from torch import Tensor
from backend.MaTensor import MaskedTensor
from backend.SpTensor import SparseTensor
import torch.nn as nn
from .MaXOperator import messagepassing_tuple, pooling_tuple
from torch_geometric.utils import degree
import torch.nn as nn
from utils import MLP
from typing import List, Union

class SubgConv(nn.Module):
    '''
    message passing within each subgraph
    '''
    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: MaskedTensor, A: Union[MaskedTensor, SparseTensor])->MaskedTensor:
        tX = MaskedTensor(self.lin(X.data), X.mask)
        # print(tX.nnz, A.nnz, datadict["XA_tar"].shape, datadict["XA_acd"].max(dim=-1)[0])
        ret = messagepassing_tuple(A, tX, "XA", self.aggr)
        return ret 

class CrossSubgConv(nn.Module):
    '''
    message passing across subgraph
    '''
    def __init__(self, emb_dim: int, mlplayer: int, aggr: str="sum", **kwargs):
        super().__init__()
        self.aggr = aggr
        self.lin = MLP(emb_dim, emb_dim, mlplayer, True, **kwargs["mlp"])

    def forward(self, X: MaskedTensor, A: Union[MaskedTensor, SparseTensor])->SparseTensor:
        tX = MaskedTensor(self.lin(X.data), X.mask)
        return messagepassing_tuple(A, tX, "AX", self.aggr)


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

    def forward(self, X: MaskedTensor, A: Union[SparseTensor, MaskedTensor])->MaskedTensor:
        for conv in self.convs:
            tX = conv(X, A)
            if self.residual:
                X = MaskedTensor(X.data + tX.data, mask=tX.mask, is_filled=(X.padvalue==tX.padvalue) and (X.padvalue==0))
            else:
                X = tX
        return X


if __name__ == "__main__":
    pass