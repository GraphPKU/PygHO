import torch
from torch_scatter import scatter
from .SpTensor import SparseTensor


def spmm(A: SparseTensor, X: TabError, aggr: str = "sum") -> SparseTensor:
    assert A.sparse_dim == 2, "A should be adjacency matrix with two sparse dim "
    ind, val = A.indices, A.values
    if val is None:
        mult = val * X[ind[1]]
    else:
        mult = X[ind[1]]
    return scatter(mult, ind, dim=0, dim_size=A.shape[0], reduce=aggr)