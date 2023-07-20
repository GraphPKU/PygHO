from MaTensor import MaskedTensor
import torch
from torch import BoolTensor, Tensor
from typing import Optional


def mamamm(A: MaskedTensor, B: MaskedTensor, mask: Optional[BoolTensor]=None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    * should be broadcastable
    '''
    return MaskedTensor(torch.einsum("bnm...,bmk...->bnk...", A.fill_masked(0), B.fill_masked(0)), mask if mask is not None else A.mask)


def mamm(A: MaskedTensor, B: Tensor, mask: Optional[BoolTensor]=None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    * should be broadcastable
    '''
    return MaskedTensor(torch.einsum("bnm...,bmk...->bnk...", A.fill_masked(0), B), mask if mask is not None else A.mask)


def mmamm(A: Tensor, B: MaskedTensor, mask: Optional[BoolTensor]=None):
    '''
    A: (B, n, m, *) MaskedTensor
    B: (B, m, k, *) MaskedTensor
    '''
    return MaskedTensor(torch.einsum("bnm...,bmk...->bnk...", A, B.fill_masked(0)), mask if mask is not None else B.mask)


if __name__ == "__main__":
    b, n, m, l, d = 3, 10, 20, 17, 13
    A = torch.randn((b, n, m, d))
    mask = torch.randn((b, n, l))>=1
    MA = MaskedTensor(A, A.mean(dim=-1)>=0)
    B = torch.randn((b, m, l, 1))
    MB = MaskedTensor(B, B.mean(dim=-1)>=0)
    MC = mamamm(MA, MB, mask)
    MD = mamm(MA, B, mask)
    ME = mmamm(A, MB, mask)
    print(MC.shape, MD.shape, ME.shape)