import torch
from torch_scatter import scatter

def spmm(A: torch.Tensor, X: torch.Tensor):
    assert A.sparse_dim() == 2, "A should be adjacency matrix with two sparse dim "
    ind, val = A.indices(), A.values()
    mult = val * X[ind[1]]
    return scatter(mult, ind, dim=0, dim_size=A.shape[0])