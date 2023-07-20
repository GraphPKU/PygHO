from .MaTensor import MaskedTensor
import torch
from torch import BoolTensor
from typing import Optional
from torch_scatter import scatter
from .SpTensor import SparseTensor
from .Spmm import spmm


def spmamm(A: SparseTensor,
           B: MaskedTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    A: (B, n, m, *) SparseTensor
    B: (B, m, *) MaskedTensor
    '''
    assert A.sparse_dim == 3
    b, n = A.shape[0], A.shape[1]
    bij = A.indices
    Aval = A.values
    mult = Aval * B.fill_masked({"sum":0, "mean":0, "max": -torch.inf, "min": torch.inf}[aggr])[bij[0], bij[2]]
    val = scatter(mult,
                  bij[0] * n + bij[1],
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n))
    return MaskedTensor(ret, mask if mask is not None else B.mask)


def maspmm(B: MaskedTensor,
           A: SparseTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    B: (B, k, m, *) MaskedTensor
    A: (B, m, n, *) SparseTensor
    '''
    assert A.sparse_dim == 3
    b, n = A.shape[0], A.shape[2]
    bij = A.indices
    Aval = A.values
    mult = Aval * B.fill_masked({"sum":0, "mean":0, "max": -torch.inf, "min": torch.inf}[aggr]).transpose(1, 2)[bij[0], bij[1]]
    val = scatter(mult,
                  bij[0] * n + bij[2],
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n)).transpose(1, 2)
    return MaskedTensor(ret, mask if mask is not None else B.mask)