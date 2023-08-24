from torch_scatter import scatter
from .SpTensor import SparseTensor
from torch import Tensor


def spmm(A: SparseTensor, X: Tensor, aggr: str = "sum") -> Tensor:
    '''
    SparseTensor, Tensor Matrix multiplication.
    Dense shapes of A and X.shape[1:] must be broadcastable. 
    '''
    assert A.sparse_dim == 2, "A should be adjacency matrix with two sparse dim "
    ind, val = A.indices, A.values
    if val is None:
        mult = X[ind[1]]
    else:
        mult = val * X[ind[1]]
    return scatter(mult, ind[0], dim=0, dim_size=A.shape[0], reduce=aggr)


def mspmm(X: Tensor, A: SparseTensor, aggr: str = "sum") -> Tensor:
    '''
    Tensor, SparseTensor Matrix multiplication.
    Dense shapes of A and X.shape[:1]+X.shape[2:] must be broadcastable. 
    '''
    assert A.sparse_dim == 2, "A should be adjacency matrix with two sparse dim "
    X = X.transpose(0, 1)
    ind, val = A.indices, A.values
    if val is None:
        mult = X[ind[0]]
    else:
        mult = val * X[ind[0]]
    return scatter(mult, ind[1], dim=0, dim_size=A.shape[1],
                   reduce=aggr).transpose(0, 1)
