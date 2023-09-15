from .SpTensor import SparseTensor
from torch import Tensor
import torch
from .utils import torch_scatter_reduce

def spmm(A: SparseTensor, dim1: int, X: Tensor, aggr: str = "sum") -> Tensor:
    """
    SparseTensor, Tensor matrix multiplication.

    This function performs a matrix multiplication between a SparseTensor `A` and a dense tensor `X` along the specified dimension `dim1`. The result is a dense tensor. The `aggr` parameter specifies the reduction operation used for merging the resulting values.

    Args:
    - A (SparseTensor): The SparseTensor used for multiplication.
    - dim1 (int): The dimension along which `A` is reduced.
    - X (Tensor): The dense tensor to be multiplied with `A`. It dim 0 will be reduced.
    - aggr (str, optional): The reduction operation to use for merging edge features ("sum", "min", "max", "mean"). Defaults to "sum".

    Returns:
    - Tensor: A dense tensor containing the result of the matrix multiplication.

    Example:
    ```python
    A = SparseTensor(...)  # Initialize A
    X = torch.Tensor(...)  # Initialize X
    result = spmm(A, dim1=0, X)
    ```

    Notes:
    - `A` should be a 2-dimensional SparseTensor.
    - The dense shapes of `A` and `X` other than `dim1` must be broadcastable.

    """
    assert A.sparse_dim == 2, "can only use 2-dim sparse tensor"
    val = A.values
    if dim1 == 0:
        srcind = A.indices[0]
        tarind = A.indices[1]
        tarshape = A.shape[1]
    else:
        srcind = A.indices[1]
        tarind = A.indices[0]
        tarshape = A.shape[0]
    if val is None:
        mult = X[srcind]
    else:
        mult = val * X[srcind]
    ret = torch_scatter_reduce(0, mult, tarind, tarshape, aggr)
    return ret
