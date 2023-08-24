from .MaTensor import MaskedTensor
import torch
from torch import BoolTensor, Tensor
from typing import Optional


def mamamm(A: MaskedTensor,
           B: MaskedTensor,
           mask: Optional[BoolTensor] = None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    * should be broadcastable
    '''
    return MaskedTensor(
        torch.einsum("bnm...,bmk...->bnk...", A.fill_masked(0),
                     B.fill_masked(0)), mask if mask is not None else A.mask)


def mamm(A: MaskedTensor, B: Tensor, mask: Optional[BoolTensor] = None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    * should be broadcastable
    '''
    return MaskedTensor(
        torch.einsum("bnm...,bmk...->bnk...", A.fill_masked(0), B),
        mask if mask is not None else A.mask)


def mmamm(A: Tensor, B: MaskedTensor, mask: Optional[BoolTensor] = None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    '''
    return MaskedTensor(
        torch.einsum("bnm...,bmk...->bnk...", A, B.fill_masked(0)),
        mask if mask is not None else B.mask)