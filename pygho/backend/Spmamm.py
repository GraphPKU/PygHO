from .MaTensor import MaskedTensor, filterinf
import torch
from torch import BoolTensor, Tensor
from typing import Optional
from .SpTensor import SparseTensor
from .utils import torch_scatter_reduce

filled_value_dict = {"sum": 0, "max": -torch.inf, "min": torch.inf}
filter_inf_ops = ["max", "min"]


def spmamm(A: SparseTensor,
           dim1: int,
           B: MaskedTensor,
           dim2: int,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    """
    SparseTensor-MaskedTensor multiplication.

    This function performs multiplication between a SparseTensor `A` and a MaskedTensor `B`. The specified dimensions `dim1` and `dim2` are contracted during the multiplication, and the result is returned as a MaskedTensor.

    Args:

    - A (SparseTensor): The SparseTensor with shape (B, n, m, \*shape).
    - dim1 (int): The dimension to contract in the SparseTensor `A`.
    - B (MaskedTensor): The MaskedTensor with shape (B, m, \*shape).
    - dim2 (int): The dimension to contract in the MaskedTensor `B`.
    - mask (BoolTensor, optional): The mask to apply to the resulting MaskedTensor. Default is None.
    - aggr (str, optional): The aggregation method for reduction during multiplication (e.g., "sum", "max"). Default is "sum".

    Returns:

    - MaskedTensor: A new MaskedTensor with shape (B, n,\*denseshapeshape) and the specified mask.

    Notes:
    - This function performs multiplication between a SparseTensor and a MaskedTensor, contracting the specified dimensions.
    - The `aggr` parameter controls the reduction operation during multiplication.
    - The result is returned as a MaskedTensor.

    """
    assert A.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    assert aggr != "mean", "not implemented"
    if dim1 == 1:
        b, n = A.shape[0], A.shape[2]
        bij = A.indices[0], A.indices[1]
        tar_ind = n * A.indices[0] + A.indices[2]
    elif dim1 == 2:
        b, n = A.shape[0], A.shape[1]
        bij = A.indices[0], A.indices[2]
        tar_ind = n * A.indices[0] + A.indices[1]
    else:
        raise NotImplementedError
    Aval = A.values
    tB = torch.movedim(B.data, dim2, 1)
    tBmask = torch.movedim(B.mask, dim2, 1)
    if Aval is not None:
        mult = Aval.unsqueeze(1) * tB[bij[0], bij[1]]
    else:
        mult = tB[bij[0], bij[1]]
    validmask = tBmask[bij[0], bij[1]]
    mult.masked_fill(torch.logical_not(validmask), filled_value_dict[aggr])
    val = torch_scatter_reduce(0, mult, tar_ind, b*n, aggr)
    ret = val.unflatten(0, (b, n))
    ret = torch.movedim(ret, 1, dim2)
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else B.mask)