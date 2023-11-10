from .MaTensor import MaskedTensor, filterinf
import torch
from torch import BoolTensor, Tensor
from typing import Optional
from .SpTensor import SparseTensor, indicehash_tight
from .utils import torch_scatter_reduce

filled_value_dict = {"sum": 0, "max": -torch.inf, "min": torch.inf}
filter_inf_ops = ["max", "min"]


def spmamm(A: SparseTensor,
           dim1: int,
           B: MaskedTensor,
           dim2: int,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum",
           broadcast_dim: int=1) -> MaskedTensor:
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
    assert broadcast_dim >= 1, " at least 1 broadcast_dim "
    assert A.sparse_dim == 2 + broadcast_dim, f"A should have 2 sparse dims other than {broadcast_dim} broadcast dim, but input has {A.sparse_dim}"
    assert B.masked_dim == 1 + broadcast_dim, f"B should have 1 masked dims other than {broadcast_dim} broadcast dim, but input has {B.masked_dim}"
    assert aggr != "mean", "not implemented"
    relativedim = dim1 - broadcast_dim
    otherdim = 1 - relativedim + broadcast_dim
    b = torch.LongTensor(A.shape[:broadcast_dim]+(A.shape[otherdim],), device=A.indices.device)
    bij = indicehash_tight(A.indices[:broadcast_dim], b[:-1]), A.indices[dim1]
    tar_ind = b[-1] * bij[0] + A.indices[otherdim]

    Aval = A.values
    tB = torch.movedim(B.data.flatten(0, broadcast_dim-1), dim2-broadcast_dim+1, 1)
    tBmask = torch.movedim(B.mask.flatten(0, broadcast_dim-1), dim2-broadcast_dim+1, 1)
    if Aval is not None:
        mult = Aval.unsqueeze(1) * tB[bij[0], bij[1]]
    else:
        mult = tB[bij[0], bij[1]]
    
    validmask = tBmask[bij[0], bij[1]]
    mult.masked_fill(torch.logical_not(validmask), filled_value_dict[aggr])
    val = torch_scatter_reduce(0, mult, tar_ind, torch.prod(b), aggr)
    ret = val.unflatten(0, b)
    if aggr in filter_inf_ops:
        ret = filterinf(ret)

    return MaskedTensor(ret, mask if mask is not None else B.mask)