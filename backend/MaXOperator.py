import torch
from torch import Tensor
from .Spmamm import spmamm, maspmm
from .Mamamm import mamamm, mamm, mmamm
from .Spmm import spmm
from torch_scatter import scatter
from .MaTensor import MaskedTensor

def messagepassing_tuple(A: Tensor, X: MaskedTensor, key: str="AX"):
    if A.is_sparse():
        if key=="AX":
            return spmamm(A, X)
        elif key=="XA":
            return maspmm(X, A)
        else:
            raise NotImplementedError
    elif isinstance(A, MaskedTensor):
        if key=="AX":
            return mamamm(A, X)
        elif key=="XA":
            return mamamm(X, A)
        else:
            raise NotImplementedError
    elif A.layout == torch.strided:
        if key=="AX":
            return mmamm(A, X)
        elif key=="XA":
            return mamm(X, A)
        else:
            raise NotImplementedError
    raise NotImplementedError

def pooling_tuple(X: MaskedTensor, dim=1, pool: str="sum"):
    assert dim in [1, 2]
    if pool == "sum":
        return X.sum(dim=dim)
    elif pool == "max":
        return X.max(dim=dim)
    elif pool == "mean":
        return X.mean(dim=dim)

def unpooling_node(nodeX: Tensor, tarX: MaskedTensor, dim=1):
    assert dim in [1, 2]
    return nodeX.unsqueeze(dim) + tarX

def messagepassing_node(A: Tensor, nodeX: Tensor):
    return spmm(A, nodeX)