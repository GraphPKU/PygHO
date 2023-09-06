from .MaTensor import MaskedTensor
import torch
from torch import BoolTensor, Tensor
from typing import Optional, Tuple


def batched_tensordot(A: Tensor, catdim1: int, dim1: int, B: Tensor, catdim2: int, dim2: int) -> Tensor:
    '''
    A (catdim1, densedim)
    B (catdim2, densedim)
    return (catdim1\dim1, catdim2\dim2, densedim)
    '''
    assert dim1 < catdim1, "contract the masked dim only"
    assert dim2 < catdim2, "contract the masked dim only"
    ndim1 = A.ndim
    densedim1 = ndim1 - catdim1
    ndim2 = B.ndim
    densedim2 = ndim2 - catdim2
    assert densedim1 == densedim2, "must of the same dense shape"
    
    A = torch.movedim(A, dim1, -1)
    B = torch.movedim(B, dim2, -1)

    for _ in range(catdim2-1): A.unsqueeze(catdim1-1)
    for _ in range(catdim1-1): B.unsqueeze(0)

    C = torch.inner(A, B)
    return C


def broadcast_denseshape(A: Tensor, densedim1: int, B: Tensor, densedim2: int) -> Tuple[Tensor, Tensor]:
    while densedim1 < densedim2:
        A.unsqueeze(-densedim1-1)
        densedim1 += 1
    while densedim2 < densedim1:
        B.unsqueeze(-densedim2-1)
        densedim2 += 1
    return A, B

def mamamm(A: MaskedTensor,
           dim1: int,
           B: MaskedTensor,
           dim2: int,
           mask: BoolTensor,
           broadcast_firstdim: bool=True)->MaskedTensor:
    '''
    A: (B, maskeddims1, *) MaskedTensor
    B: (B, maskeddims2, *) MaskedTensor
    * should be broadcastable
    out: (B, maskeddims1\dim1, maskeddims2\dim2, *)
    '''
    tA = A.fill_masked(0)
    tB = B.fill_masked(0)
    densedim1 = A.dense_dim
    densedim2 = B.dense_dim
    tA, tB = broadcast_denseshape(tA, tB)
    densedim = max(densedim1, densedim2)

    if broadcast_denseshape:
        assert dim1 > 0, "0 dim of A is batch, need to be broadcasted"
        assert dim2 > 0, "0 dim of B is batch, need to be broadcasted"

    if broadcast_firstdim:
       tA = torch.movedim(tA, 0, -densedim-1)
       tB = torch.movedim(tB, 0, -densedim-1)
       densedim += 1
       prod = batched_tensordot(tA, A.masked_dim-1, dim1-1, tB, B.masked_dim-1, dim2-1)
       prod = torch.movedim(prod, -densedim, 0)
    else:
       prod = batched_tensordot(tA, A.masked_dim, dim1, tB, B.masked_dim, dim2)
    return MaskedTensor(prod, mask)
