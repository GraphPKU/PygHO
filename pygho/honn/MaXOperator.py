import torch
from torch import Tensor
from ..backend.SpTensor import SparseTensor
from ..backend.Spmamm import spmamm, maspmm
from ..backend.Mamamm import mamamm
from typing import Union
from ..backend.MaTensor import MaskedTensor


def messagepassing_tuple(A: Union[SparseTensor, MaskedTensor],
                         B: Union[SparseTensor, MaskedTensor],
                         aggr: str = "sum") -> MaskedTensor:
    '''
    A means adjacency matrix, X means tuple representations for key="AX", "XA". 
    A, X means X1, X2 for key = "XX"
    '''
    if isinstance(A, SparseTensor):
        assert isinstance(B, MaskedTensor)
        return spmamm(A, B, B.mask, aggr)
    elif isinstance(B, SparseTensor):
        return maspmm(A, B)
    if isinstance(A, Tensor):
        assert isinstance(B, MaskedTensor)
        assert aggr == "sum", "dense adjacency only support sum aggr"
        return mmamm(A, B, B.mask)
    elif isinstance(B, Tensor):
        assert isinstance(A, MaskedTensor)
        assert aggr == "sum", "dense adjacency only support sum aggr"
        return mamm(A, B, A.mask)
    raise NotImplementedError


def pooling_tuple(X: MaskedTensor, dim=1, pool: str = "sum") -> MaskedTensor:
    assert dim in [1, 2]
    return getattr(X, pool)(dim=dim)


def unpooling_node(nodeX: Tensor, tarX: MaskedTensor, dim=1) -> MaskedTensor:
    assert dim in [1, 2]
    return MaskedTensor(nodeX.unsqueeze(dim).expand_as(tarX.data),
                        mask=tarX.mask)