from torch import Tensor, LongTensor

from pygho.backend.SpTensor import SparseTensor
from ..backend.Spspmm import spspmm
from ..backend.Spmm import spmm
from ..backend.SpTensor import SparseTensor
from typing import Optional, Iterable, Dict, Union, List, Tuple
from torch.nn import Module

KEYSEP = "___"


def parse_precomputekey(model: Module) -> List[str]:
    ret = []
    for mod in model.modules():
        if isinstance(mod, OpMessagePassing):
            ret.append(mod.precomputekey)
    return sorted(list(set(ret)))


class OpNodeMessagePassing(Module):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__()
        self.aggr = aggr

    def forward(self, A: SparseTensor, X: Tensor, tarX: Tensor) -> Tensor:
        assert A.sparse_dim == 2, "A is adjacency matrix of the whole graph of shape nxn"
        return spmm(A, 1, X, self.aggr)


class OpMessagePassing(Module):

    def __init__(self,
                 op0: str = "X",
                 op1: str = "X",
                 dim1: int = 1,
                 op2: str = "A",
                 dim2: int = 0,
                 aggr: str = "sum") -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.precomputekey = f"{op0}{KEYSEP}{op1}{KEYSEP}{dim1}{KEYSEP}{op2}{KEYSEP}{dim2}"
        self.aggr = aggr

    def forward(self,
                A: SparseTensor,
                B: SparseTensor,
                datadict: Dict,
                tarX: Optional[SparseTensor] = None) -> SparseTensor:
        return spspmm(
            A,
            self.dim1,
            B,
            self.dim2,
            self.aggr,
            acd=datadict.get(f"{self.precomputekey}{KEYSEP}acd", None),
            bcd=datadict.get(f"{self.precomputekey}{KEYSEP}bcd", None),
            tar_ind=datadict.get(f"{self.precomputekey}{KEYSEP}tarind", None)
            if tarX is None else tarX.indices)


class Op2FWL(OpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__("X", "X", 1, "X", 0, aggr)

    def forward(self,
                X1: SparseTensor,
                X2: SparseTensor,
                datadict: Dict,
                tarX: SparseTensor | None = None) -> SparseTensor:
        assert X1.sparse_dim == 2, "X1 should be nxn adjacency matrix "
        assert X2.sparse_dim == 2, "X2 should be 2d representations"
        return super().forward(X1, X2, datadict, tarX)


class OpMessagePassingOnSubg2D(OpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__("X", "X", 1, "A", 0, aggr)

    def forward(self,
                A: SparseTensor,
                X: SparseTensor,
                datadict: Dict,
                tarX: SparseTensor | None = None) -> SparseTensor:
        assert A.sparse_dim == 2, "A should be nxn adjacency matrix "
        assert X.sparse_dim == 2, "X should be 2d representations"
        return super().forward(X, A, datadict, tarX)


class OpMessagePassingOnSubg3D(OpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__("X", "X", 2, "A", 0, aggr)

    def forward(self,
                A: SparseTensor,
                X: SparseTensor,
                datadict: Dict,
                tarX: SparseTensor | None = None) -> SparseTensor:
        assert A.sparse_dim == 2, "A should be nxn adjacency matrix "
        assert X.sparse_dim == 3, "X should be 3d representations"
        return super().forward(X, A, datadict, tarX)


class OpMessagePassingCrossSubg2D(OpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__("X", "A", 1, "X", 0, aggr)

    def forward(self,
                A: SparseTensor,
                X: SparseTensor,
                datadict: Dict,
                tarX: SparseTensor | None = None) -> SparseTensor:
        assert A.sparse_dim == 2, "A should be nxn adjacency matrix "
        assert X.sparse_dim == 2, "X should be 2d representations"
        return super().forward(A, X, datadict, tarX)


class OpDiag(Module):

    def __init__(self,
                 dims: Iterable[int],
                 return_sparse: bool = False) -> None:
        super().__init__()
        self.dims = sorted(list(set(dims)))
        self.return_sparse = return_sparse

    def forward(self, A: SparseTensor) -> Union[Tensor, SparseTensor]:
        return A.diag(self.dims, return_sparse=self.return_sparse)


class OpDiag2D(OpDiag):

    def __init__(self) -> None:
        super().__init__([0, 1], False)

    def forward(self, X: SparseTensor) -> Tensor:
        assert X.sparse_dim == 2, "X should be 2d representations"
        return X.diag(self.dims, return_sparse=self.return_sparse)


class OpPooling(Module):

    def __init__(self,
                 dims: Union[int, Iterable[int]],
                 pool: str = "sum",
                 return_sparse: bool = False) -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = [dims]
        self.dims = sorted(list(set(dims)))
        self.pool = pool
        self.return_sparse = return_sparse

    def forward(self, X: SparseTensor) -> Union[SparseTensor, Tensor]:
        return getattr(X, self.pool)(self.dims,
                                     return_sparse=self.return_sparse)


class OpPoolingSubg2D(OpPooling):

    def __init__(self, pool) -> None:
        super().__init__(1, pool, False)

    def forward(self, X: SparseTensor) -> Tensor:
        assert X.sparse_dim == 2, "X should be 2d representations"
        return super().forward(X)


class OpPoolingSubg3D(OpPooling):

    def __init__(self, pool) -> None:
        super().__init__(2, pool, True)

    def forward(self, X: SparseTensor) -> SparseTensor:
        assert X.sparse_dim == 3, "X should be 3d representations"
        return super().forward(X)


class OpPoolingCrossSubg2D(OpPooling):

    def __init__(self, pool) -> None:
        super().__init__(0, pool, False)

    def forward(self, X: SparseTensor) -> Tensor:
        assert X.sparse_dim == 2, "X should be 2d representations"
        return super().forward(X)


class OpUnpooling(Module):

    def __init__(self,
                 dims: Union[int, Iterable[int]],
                 fromdense1dim: bool = True) -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = [dims]
        self.dims = sorted(list(set(dims)))
        self.fromdense1dim = fromdense1dim

    def forward(self, X: Union[Tensor, SparseTensor],
                tarX: SparseTensor) -> SparseTensor:
        if isinstance(X, Tensor):
            leftdim = list(set(range(tarX.sparse_dim)) - set(self.dims))
            assert len(leftdim) == 1, "canonly pooling from 1 dim"
            return tarX.unpooling_fromdense1dim(leftdim[0], X)
        else:
            return X.unpooling(self.dims, tarX)


class OpUnpoolingSubgNodes2D(OpUnpooling):

    def __init__(self) -> None:
        super().__init__(1, True)


class OpUnpoolingRootNodes2D(OpUnpooling):

    def __init__(self) -> None:
        super().__init__(0, True)