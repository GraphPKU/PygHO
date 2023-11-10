from .MaTensor import MaskedTensor
import torch
from torch import BoolTensor, Tensor
from typing import Optional, Tuple


def mamamm(A: MaskedTensor,
           dim1: int,
           B: MaskedTensor,
           dim2: int,
           mask: BoolTensor,
           broadcast_dims: int = 1) -> MaskedTensor:
    """
    Batched masked matrix multiplication of two MaskedTensors.

    This function performs batched matrix multiplication between two MaskedTensors `A` and `B`, where the masked dimensions `dim1` and `dim2` are contracted. The result is a new MaskedTensor with the specified mask.

    Args:

    - A (MaskedTensor): The first MaskedTensor with shape (B,\* maskedshape1,\*denseshapeshape).
    - dim1 (int): The masked dimension to contract in the first tensor `A`.
    - B (MaskedTensor): The second MaskedTensor with shape (B,\* maskedshape2,\*denseshapeshape).
    - dim2 (int): The masked dimension to contract in the second tensor `B`.
    - mask (BoolTensor): The mask to apply to the resulting MaskedTensor.
    - broadcast_firstdim (bool, optional): If True, broadcast the first dimension (batch dimension) of `A` and `B` to ensure compatibility. Default is True.

    Returns:

    - MaskedTensor: A new MaskedTensor with shape (B,\* maskedshape1\dim1,\* maskedshape2\dim2,\*denseshapeshape) and the specified mask.

    Notes:

    - This function performs batched matrix multiplication between two MaskedTensors, contracting the specified masked dimensions.
    """
    tA, tB = A.fill_masked(0.), B.fill_masked(0.)
    catdim1, catdim2 = A.masked_dim, B.masked_dim
    if broadcast_dims > 0:
        assert dim1 > 0, "0 dim of A is batch, need to be broadcasted"
        assert dim2 > 0, "0 dim of B is batch, need to be broadcasted"
        assert tA.shape[:broadcast_dims] == tB.shape[:broadcast_dims], "broadcast dims should match"
        if broadcast_dims > 1:
            broadcastshape = tA.shape[:broadcast_dims]
            tA, tB = tA.flatten(0, broadcast_dims-1), tB.flatten(0, broadcast_dims-1)
        tA = torch.movedim(tA, 0, -1)
        tB = torch.movedim(tB, 0, -1)
        dim1 -= broadcast_dims
        dim2 -= broadcast_dims
        catdim1 -= broadcast_dims
        catdim2 -= broadcast_dims
    if catdim1 == 1:
        tA.unsqueeze(catdim1)
        catdim1 += 1
    if catdim2 == 1:
        tB.unsqueeze(catdim2)
        catdim2 += 1
    assert catdim1 >= 2, "bug"
    assert catdim2 >= 2, "bug"
    tA, tB = tA.movedim(dim1, -1), tB.movedim(dim2, -1)
    catshape1, catshape2 = tA.shape[:catdim1-1], tB.shape[:catdim2-1]
    tA, tB = tA.flatten(0, catdim1 - 2), tB.flatten(0, catdim2 - 2)
    tA, tB = tA.movedim(0, -2), tB.movedim(0, -1)
    prod = torch.matmul(tA, tB)

    prod = prod.flatten(-2, -1).movedim(-1, 0)
    prod = prod.unflatten(0, catshape1 + catshape2)
    if broadcast_dims > 0:
        prod = prod.movedim(-1, 0)
        if broadcast_dims > 1:
            prod = prod.unflatten(broadcastshape)
    return MaskedTensor(prod, mask)
