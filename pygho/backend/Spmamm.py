from .MaTensor import MaskedTensor, filterinf
import torch
from torch import BoolTensor
from typing import Optional
from torch_scatter import scatter
from .SpTensor import SparseTensor

filled_value_dict = {"sum": 0, "mean": 0, "max": -torch.inf, "min": torch.inf}
filter_inf_ops = ["max", "min"]


def spmamm(A: SparseTensor,
           B: MaskedTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    A: (B, n, m, *) SparseTensor
    B: (B, m, *) MaskedTensor
    ?? max pooling with Aval?
    '''
    assert A.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    b, n = A.shape[0], A.shape[1]
    bij = A.indices
    Aval = A.values
    if Aval is None:
        mult = B.fill_masked(filled_value_dict[aggr])[bij[0], bij[2]]
    else:
        mult = torch.einsum("z...,zm...->zm...", Aval, B.fill_masked(filled_value_dict[aggr])[bij[0], bij[2]])
    val = scatter(mult,
                  bij[0] * n + bij[1],
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n))
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else B.mask)


def maspmm(A: MaskedTensor,
           B: SparseTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    A: (B, k, m, *) MaskedTensor
    B: (B, m, n, *) SparseTensor
    '''
    assert B.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    b, n = B.shape[0], B.shape[2]
    bij = B.indices
    Bval = B.values
    if Bval is None:
        mult = A.fill_masked(filled_value_dict[aggr])[bij[0], :, bij[1]]
    else:
        mult = torch.einsum(
            "z...,zm...->zm...", Bval,
            A.fill_masked(filled_value_dict[aggr])[bij[0], :, bij[1]])
    val = scatter(mult,
                  bij[0] * n + bij[2],
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n)).transpose(1, 2)
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else A.mask)

