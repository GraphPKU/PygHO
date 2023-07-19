from .Tensor import MaskedTensor
import torch
from torch import BoolTensor, LongTensor
from typing import Optional


def mamamm(A: MaskedTensor, B: MaskedTensor, mask: Optional[BoolTensor]=None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    '''
    return MaskedTensor(torch.einsum("bnmd,bmkd->bnkd", A.get_data(), B.get_data()), mask if mask is not None else A.get_mask())
