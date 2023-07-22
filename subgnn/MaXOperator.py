import torch
from torch import Tensor
from backend.SpTensor import SparseTensor
from backend.Spmamm import spmamm, maspmm
from backend.Mamamm import mamamm, mamm, mmamm
from typing import Union
from backend.MaTensor import MaskedTensor


def messagepassing_tuple(A: Union[SparseTensor, Tensor, MaskedTensor],
                         X: MaskedTensor,
                         key: str = "AX",
                         aggr: str = "sum") -> MaskedTensor:
    '''
    A means adjacency matrix, X means tuple representations for key="AX", "XA". 
    A, X means X1, X2 for key = "XX"
    '''
    if isinstance(A, SparseTensor):
        if key == "AX":
            return spmamm(A, X, X.mask, aggr)
        elif key == "XA":
            return maspmm(X, A, X.mask, aggr)
        else:
            raise NotImplementedError
    elif isinstance(A, MaskedTensor):
        assert aggr == "sum", "dense adjacency only supports ordinary matrix multiplication"
        if key == "AX":
            return mamamm(A, X, mask=X.mask)
        elif key == "XA":
            return mamamm(X, A, mask=X.mask)
        elif key == "XX":
            return mamamm(A, X, mask=X.mask)
        else:
            raise NotImplementedError
    elif A.layout == torch.strided:
        assert aggr == "sum", "dense adjacency only supports ordinary matrix multiplication"
        if key == "AX":
            return mmamm(A, X, mask=X.mask)
        elif key == "XA":
            return mamm(X, A, mask=X.mask)
        else:
            raise NotImplementedError
    raise NotImplementedError


def pooling_tuple(X: MaskedTensor, dim=1, pool: str = "sum") -> MaskedTensor:
    assert dim in [1, 2]
    if pool == "sum":
        return X.sum(dim=dim)
    elif pool == "max":
        return X.max(dim=dim)
    elif pool == "mean":
        return X.mean(dim=dim)


def unpooling_node(nodeX: Tensor, tarX: MaskedTensor, dim=1) -> MaskedTensor:
    assert dim in [1, 2]
    return MaskedTensor(nodeX.unsqueeze(dim).expand_as(tarX.data),
                        mask=tarX.mask)