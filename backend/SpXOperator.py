import torch
from torch import Tensor
from .Spspmm import spspmm
from .Spmm import spmm
from torch_scatter import scatter

def messagepassing_tuple(A: Tensor, X: Tensor, key: str="AX", datadict: dict={}):
    if key=="AX":
        return spspmm(A, X, akl=datadict.get(f"{key}_akl", None), bkl=datadict.get(f"{key}_bkl", None), tar_ij=datadict.get(f"{key}_tar", None))
    elif key=="XA":
        return spspmm(X, A, akl=datadict.get(f"{key}_akl", None), bkl=datadict.get(f"{key}_bkl", None), tar_ij=datadict.get(f"{key}_tar", None))
    else:
        return spspmm(A, X, akl=datadict.get(f"{key}_akl", None), bkl=datadict.get(f"{key}_bkl", None), tar_ij=datadict.get(f"{key}_tar", None))

def pooling_tuple(X: Tensor, dim=1, pool: str="sum"):
    assert dim in [0, 1]
    ind, val = X.indices(), X.values()
    return scatter(val, ind[1-dim], dim=0, dim_size=X.shape[1-dim], reduce=pool)

def unpooling_node(nodeX: Tensor, tarX: Tensor, dim=1):
    assert dim in [0, 1]
    ind = tarX.indices()
    val = nodeX[ind[dim]]
    return torch.sparse_coo_tensor(ind, val, size=list(tarX.shape[:2]) + list(val.shape[1:]))

def messagepassing_node(A: Tensor, nodeX: Tensor):
    return spmm(A, nodeX)