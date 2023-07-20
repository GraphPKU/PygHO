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
    else:
        return spspmm(A,
                      X,
                      aggr,
                      acd=datadict.get(f"{key}_acd", None),
                      bcd=datadict.get(f"{key}_bcd", None),
                      tar_ij=datadict.get(f"{key}_tar", None))


def pooling_tuple(X: SparseTensor, dim=1, pool: str = "sum") -> Tensor:
    assert dim in [0, 1]
    ind, val = X.indices, X.values
    return scatter(val,
                   ind[1 - dim],
                   dim=0,
                   dim_size=X.shape[1 - dim],
                   reduce=pool)


def unpooling_node(nodeX: Tensor, tarX: SparseTensor, dim=1) -> SparseTensor:
    assert dim in [0, 1]
    ind = tarX.indices
    val = nodeX[ind[dim]]
    return SparseTensor(tarX.indices,
                        val,
                        shape=list(tarX.shape[:2]) + list(val.shape[1:]),
                        is_coalesced=True)


def messagepassing_node(A: SparseTensor,
                        nodeX: Tensor,
                        aggr: str = "sum") -> Tensor:
    return spmm(A, nodeX, aggr)