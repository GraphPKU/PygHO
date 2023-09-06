from .MaTensor import MaskedTensor, filterinf
import torch
from torch import BoolTensor
from typing import Optional
from torch_scatter import scatter
from .SpTensor import SparseTensor

filled_value_dict = {"sum": 0, "max": -torch.inf, "min": torch.inf}
filter_inf_ops = ["max", "min"]


def spmamm(A: SparseTensor,
           B: MaskedTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    A: (B, n, m, *) SparseTensor
    B: (B, m, *) MaskedTensor
    '''
    assert A.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    assert aggr != "mean", "not implemented"
    b, n = A.shape[0], A.shape[1]
    bij = A.indices
    Aval = A.values
    mult = Aval.unsqueeze(1) * B.data[bij[0], bij[2]]
    validmask = B.__rawmask[bij[0], bij[2]]
    mult.masked_fill(torch.logical_not(validmask), filled_value_dict[aggr])
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
    assert aggr != "mean", "not implemented"
    b, n = B.shape[0], B.shape[2]
    bij = B.indices
    Bval = B.values
    mult = Bval.unsqueeze(1)* A.data[bij[0], :, bij[1]]
    validmask = A.__rawmask[bij[0], :, bij[1]]
    mult.masked_fill(torch.logical_not(validmask), filled_value_dict[aggr])
    val = scatter(mult,
                  (bij[0] * n + bij[2]),
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n)).transpose(1, 2)
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else A.mask)



def spmamm_ge(A: SparseTensor,
           dim1: int,
           B: MaskedTensor,
           dim2: int,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    A: (B, n, m, *) SparseTensor
    B: (B, m, *) MaskedTensor
    '''
    assert A.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    assert aggr != "mean", "not implemented"
    if dim1 == 1:
        b, n = A.shape[0], A.shape[2]
        bij = A.indices[0], A.indices[1]
        tar_ind = n*A.indices[0] + A.indices[2]
    elif dim1 == 2:
        b, n = A.shape[0], A.shape[1]
        bij = A.indices[0], A.indices[2]
        tar_ind = n*A.indices[0] + A.indices[1]
    else:
        raise NotImplementedError
    Aval = A.values
    tB = torch.movedim(B.data, dim2, 1)
    tBmask = torch.movedim(B.__rawmask, dim2, 1)
    if Aval is not None:
        mult = Aval.unsqueeze(1) * tB[bij[0], bij[1]]
    else:
        mult = tB[bij[0], bij[1]]
    validmask = tBmask[bij[0], bij[1]]
    mult.masked_fill(torch.logical_not(validmask), filled_value_dict[aggr])
    val = scatter(mult,
                  tar_ind,
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n))
    ret = torch.movedim(ret, 1, dim2)
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else B.mask)

