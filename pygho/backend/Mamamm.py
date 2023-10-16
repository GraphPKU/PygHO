from .MaTensor import MaskedTensor
import torch
from torch import BoolTensor, Tensor
from typing import Optional, Tuple


def batched_tensordot(A: Tensor, catdim1: int, dim1: int, B: Tensor,
                      catdim2: int, dim2: int) -> Tensor:
    """
    Perform a batched tensordot matrix operation.

    This function computes the tensordot product of two tensors `A` and `B`, where `A` and `B` are batched tensors with specified concatenation dimensions `catdim1` and `catdim2`, and contraction dimensions `dim1` and `dim2`.

    Args:

    - A (Tensor): The first batched tensor of shape (catshape1, broadcastshape).
    - catdim1 (int): The length of catshape1.
    - dim1 (int): The contraction dimension along `catdim1` of the first tensor.
    - B (Tensor): The second batched tensor of shape (catshape2, broadcastshape)..
    - catdim2 (int): The length of catshape2.
    - dim2 (int): The contraction dimension along `catdim2` of the second tensor.

    Returns:

    - Tensor: The result of the batched tensordot operation of shape (\*catshape1\\dim1, \*catshape2\\dim2, \*broadcastshape), where densedim is the common dense dimension of `A` and `B`.

    Notes:

    - `catdim1` and `catdim2` specify the number of concatenation dimensions of `A` and `B`, respectively.
    - `dim1` and `dim2` specify the contraction dimensions along `catdim1` and `catdim2`, respectively.
    - The function uses optimized paths for specific cases (e.g., when `catdim1=2` and `catdim2=2`).

    """
    assert dim1 < catdim1, "contract the masked dim only"
    assert dim2 < catdim2, "contract the masked dim only"
    # print(A.shape, catdim1, dim1, B.shape, catdim2, dim2)
    ndim1 = A.ndim
    densedim1 = ndim1 - catdim1
    ndim2 = B.ndim
    densedim2 = ndim2 - catdim2
    assert densedim1 == densedim2, "must of the same dense shape"

    if catdim1 == 2 and catdim2 == 2:
        if dim1 == 0:
            A = A.transpose(0, 1)
        if dim2 == 1:
            B = B.transpose(0, 1)
        A = torch.movedim(torch.movedim(A, 0, -1), 0, -1)
        B = torch.movedim(torch.movedim(B, 0, -1), 0, -1)
        C = A @ B
        C = torch.movedim(torch.movedim(C, -1, 0), -1, 0)
        return C
    # TODO more special case to apply bmm for acceleration?
    else:
        A = torch.movedim(A, dim1, -1)
        B = torch.movedim(B, dim2, -1)
        for _ in range(catdim2 - 1):
            A = A.unsqueeze(catdim1 - 1)
        for _ in range(catdim1 - 1):
            B = B.unsqueeze(0)

        C = torch.sum(torch.multiply(A, B), dim=-1)
        return C


def broadcast_denseshape(A: Tensor, densedim1: int, B: Tensor,
                         densedim2: int) -> Tuple[Tensor, Tensor]:
    """
    This function broadcasts the dense shapes of tensors `A` and `B` to the same by adding dimensions of size 1.

    Args:

    - A (Tensor): The first tensor.
    - densedim1 (int): The number of dense dimension of the first tensor.
    - B (Tensor): The second tensor.
    - densedim2 (int): The number of dense dimension of the second tensor.

    Returns:

    - Tuple[Tensor, Tensor]: A tuple containing the broadcasted tensors `A` and `B` with compatible dense shapes.

    Notes:

    - This function adds dimensions with size 1 to the smaller dense shape until both dense shapes match.

    """
    while densedim1 < densedim2:
        A.unsqueeze(-densedim1 - 1)
        densedim1 += 1
    while densedim2 < densedim1:
        B.unsqueeze(-densedim2 - 1)
        densedim2 += 1
    return A, B


def mamamm(A: MaskedTensor,
           dim1: int,
           B: MaskedTensor,
           dim2: int,
           mask: BoolTensor,
           broadcast_firstdim: bool = True) -> MaskedTensor:
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
    tA = A.fill_masked(0)
    tB = B.fill_masked(0)
    densedim1 = A.dense_dim
    densedim2 = B.dense_dim

    tA, tB = broadcast_denseshape(tA, densedim1, tB, densedim2)

    densedim = max(densedim1, densedim2)

    if broadcast_firstdim:
        assert dim1 > 0, "0 dim of A is batch, need to be broadcasted"
        assert dim2 > 0, "0 dim of B is batch, need to be broadcasted"
        tA = torch.movedim(tA, 0, -densedim - 1)
        tB = torch.movedim(tB, 0, -densedim - 1)
        densedim += 1
        prod = batched_tensordot(tA, A.masked_dim - 1, dim1 - 1, tB,
                                 B.masked_dim - 1, dim2 - 1)
        prod = torch.movedim(prod, -densedim, 0)
    else:
        prod = batched_tensordot(tA, A.masked_dim, dim1, tB, B.masked_dim,
                                 dim2)
    return MaskedTensor(prod, mask)
