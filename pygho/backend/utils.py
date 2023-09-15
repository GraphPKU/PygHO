import torch
from torch import Tensor, LongTensor
from typing import Tuple


def torch_scatter_reduce(dim: int, src: Tensor, ind: LongTensor, dim_size: int,
                         aggr: str) -> Tensor:
    """
    Applies a reduction operation to scatter elements from `src` to `dim_size`
    locations based on the indices in `ind`.

    This function is a wrapper for `torch.Tensor.scatter_reduce_` and is designed
    to scatter elements from `src` to `dim_size` locations based on the specified
    dimension `dim` and the indices in `ind`. The reduction operation is specified
    by the `aggr` parameter, which can be 'sum', 'mean', 'min', 'max'.

    Args:
    - dim (int): The dimension along which to scatter elements (only dim=0 is currently supported).
    - src (Tensor): The source tensor of shape (nnz, denseshape).
    - ind (LongTensor): The indices tensor of shape (nnz).
    - dim_size (int): The size of the target dimension for scattering.
    - aggr (str): The reduction operation to apply ('sum', 'mean', 'min', 'max', 'mul', 'any').

    Returns:
    - Tensor: A tensor of shape (dim_size, denseshape) resulting from the scatter operation.

    Raises:
    - AssertionError: If `dim` is not 0, or if `ind` is not 1-dimensional.

    Example:
    ```python
    src = torch.tensor([[1, 2], [4, 5], [7, 8], [9, 10]], dtype=torch.float)
    ind = torch.tensor([2, 2, 0, 1], dtype=torch.long)
    dim_size = 3
    aggr = 'sum'
    result = torch_scatter_reduce(0, src, ind, dim_size, aggr)
    ```
    """
    assert dim == 0, "other dim not implemented"
    assert ind.ndim == 1, "indice must be 1-d"
    if aggr in ["min", "max"]:
        aggr = "a" + aggr
    onedim = src.ndim - 1
    dim_size = dim_size
    ret = torch.zeros_like(src[[0]].expand((dim_size, ) + (-1, ) * onedim))
    ret.scatter_reduce_(dim,
                        ind.reshape((-1, ) + (1, ) * onedim).expand_as(src),
                        src,
                        aggr,
                        include_self=False)
    return ret