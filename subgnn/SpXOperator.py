from torch import Tensor
from backend.Spspmm import spspmm
from backend.Spmm import spmm
from backend.SpTensor import SparseTensor
from torch_scatter import scatter


def messagepassing_tuple(A: SparseTensor,
                         dim1: int,
                         B: SparseTensor,
                         dim2: int,
                         key: str = "A_1_X_0",
                         datadict: dict = {},
                         aggr: str = "sum") -> SparseTensor:
    '''
    A means adjacency matrix, X means tuple representations for key="AX", "XA". 
    A, X means X1, X2 for key = "XX"
    '''
    return spspmm(A,
                  dim1,
                  B,
                  dim2,
                  aggr,
                  acd=datadict.get(f"{key}_acd", None),
                  bcd=datadict.get(f"{key}_bcd", None),
                  tar_ind=datadict.get(f"tupleid", None))


def pooling2nodes(X: SparseTensor, dims=1, pool: str = "sum") -> Tensor:
    '''
    ret_{i} = pool(X_{ij}) for dim = 1
    ret_{i} = pool(X_{ji}) for dim = 0
    '''
    assert len(set(dims)) == X.sparse_dim - 1
    return getattr(X, pool)(dims, return_sparse=False)


def pooling2tuple(X: SparseTensor, dims=1, pool: str = "sum") -> Tensor:
    '''
    ret_{i} = pool(X_{ij}) for dim = 1
    ret_{i} = pool(X_{ji}) for dim = 0
    '''
    return getattr(X, pool)(dims, return_sparse=True)


def unpooling4node(nodeX: Tensor, tarX: SparseTensor, dim=1) -> SparseTensor:
    '''
    X_{ij} = nodeX_{i} for dim = 1
    X_{ij} = nodeX_{j} for dim = 0 
    tarX is used for provide indice for the output
    '''
    return tarX.unpooling_fromdense1dim(dim, nodeX)


def unpooling4tuple(srcX: SparseTensor, tarX: SparseTensor, dims=1) -> SparseTensor:
    '''
    X_{ij} = nodeX_{i} for dim = 1
    X_{ij} = nodeX_{j} for dim = 0 
    tarX is used for provide indice for the output
    '''
    return srcX.unpooling(dims, tarX)


def messagepassing_node(A: SparseTensor,
                        nodeX: Tensor,
                        aggr: str = "sum") -> Tensor:
    return spmm(A, nodeX, aggr)