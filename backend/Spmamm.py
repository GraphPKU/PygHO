from .MaTensor import MaskedTensor
import torch
from torch import BoolTensor, LongTensor
from typing import Optional
from torch_scatter import scatter_add


def spmamm(A: torch.Tensor, B: MaskedTensor, mask: Optional[BoolTensor]=None):
    '''
    A: (B, n, m, *) SparseTensor
    B: (B, m, *) MaskedTensor
    '''
    assert A.sparse_dim() == 3
    b, n = A.shape[0:2]
    bij = A.indices()
    Aval = A.values()
    mult = Aval * B.get_data()[bij[0], bij[2]]
    val = scatter_add(mult, bij[0]*n + bij[1], dim=0, dim_size=b*n)
    ret = val.unflatten(0, (b, n))
    return MaskedTensor(ret, mask if mask is not None else B.get_mask())

def maspmm(B: MaskedTensor, A: torch.Tensor, mask: Optional[BoolTensor]=None):
    '''
    B: (B, k, m, *) MaskedTensor
    A: (B, m, n, *) SparseTensor
    '''
    assert A.sparse_dim() == 3
    b, n = A.shape[0], A.shape[2]
    bij = A.indices()
    Aval = A.values()
    mult = Aval * B.get_data()[bij[0], :, bij[1]]
    val = scatter_add(mult, bij[0]*n + bij[2], dim=0, dim_size=b*n)
    ret = val.unflatten(0, (b, n)).transpose(1, 2)
    return MaskedTensor(ret, mask if mask is not None else B.get_mask())