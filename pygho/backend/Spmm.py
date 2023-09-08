from .SpTensor import SparseTensor
from torch import Tensor
import torch
from .utils import torch_scatter_reduce

def spmm(A: SparseTensor, dim1: int, X: Tensor, aggr: str = "sum") -> Tensor:
    '''
    SparseTensor, Tensor Matrix multiplication.
    Dense shapes of A and X shape other than dim2 must be broadcastable. 
    '''
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
