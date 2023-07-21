import torch
from torch import Tensor
from backend.Spspmm import spspmm
from backend.Spmm import spmm
from backend.SpTensor import SparseTensor
from torch_scatter import scatter


def messagepassing_tuple(A: SparseTensor,
                         X: SparseTensor,
                         key: str = "AX",
                         datadict: dict = {},
                         aggr: str = "sum") -> SparseTensor:
    '''
    A means adjacency matrix, X means tuple representations for key="AX", "XA". 
    A, X means X1, X2 for key = "XX"
    '''
    if key == "AX":
        return spspmm(A,
                      X,
                      aggr,
                      acd=datadict.get(f"{key}_acd", None),
                      bcd=datadict.get(f"{key}_bcd", None),
                      tar_ij=datadict.get(f"{key}_tar", None))
    elif key == "XA":
        return spspmm(X,
                      A,
                      aggr,
                      acd=datadict.get(f"{key}_acd", None),
                      bcd=datadict.get(f"{key}_bcd", None),
                      tar_ij=datadict.get(f"{key}_tar", None))
    elif key == "XX":
        return spspmm(A,
                      X,
                      aggr,
                      acd=datadict.get(f"{key}_acd", None),
                      bcd=datadict.get(f"{key}_bcd", None),
                      tar_ij=datadict.get(f"{key}_tar", None))
    else:
        raise NotImplementedError

def pooling_tuple(X: SparseTensor, dim=1, pool: str = "sum") -> Tensor:
    '''
    ret_{i} = pool(X_{ij}) for dim = 1
    ret_{i} = pool(X_{ji}) for dim = 0
    '''
    assert X.sparse_dim == 2, "high-order sparse tensor not implemented"
    assert dim in [0, 1], "can only pool sparse dim "
    ind, val = X.indices, X.values
    return scatter(val,
                   ind[1 - dim],
                   dim=0,
                   dim_size=X.shape[1 - dim],
                   reduce=pool)


def unpooling_node(nodeX: Tensor, tarX: SparseTensor, dim=1) -> SparseTensor:
    '''
    X_{ij} = nodeX_{i} for dim = 1
    X_{ij} = nodeX_{j} for dim = 0 
    '''
    assert tarX.sparse_dim == 2, "high-order sparse tensor not implemented"
    assert dim in [0, 1], "can only pool sparse dim "
    ind = tarX.indices
    val = nodeX[ind[1-dim]]
    return SparseTensor(tarX.indices,
                        val,
                        shape=list(tarX.shape[:2]) + list(val.shape[1:]),
                        is_coalesced=True)


def messagepassing_node(A: SparseTensor,
                        nodeX: Tensor,
                        aggr: str = "sum") -> Tensor:
    return spmm(A, nodeX, aggr)