import torch
from torch import Tensor, BoolTensor

from pygho.backend.MaTensor import MaskedTensor
from pygho.backend.SpTensor import SparseTensor
from ..backend.SpTensor import SparseTensor
from ..backend.Spmamm import spmamm
from ..backend.Mamamm import mamamm
from typing import Any, Union, Iterable, Literal, List, Tuple, Dict
from torch.nn import Module
from ..backend.MaTensor import MaskedTensor


class OpNodeMessagePassing(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: MaskedTensor, X: MaskedTensor,
                tarX: MaskedTensor) -> Tensor:
        return mamamm(A, 2, X, 1, tarX.mask)


class OpSpNodeMessagePassing(Module):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__()
        self.aggr = aggr

    def forward(self, A: SparseTensor, X: MaskedTensor,
                tarX: MaskedTensor) -> Tensor:
        return spmamm(A, 2, X, 1, tarX.mask, self.aggr)


class OpMessagePassing(Module):

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, A: MaskedTensor, B: MaskedTensor,
                tarX: MaskedTensor) -> MaskedTensor:
        tarmask = tarX.mask
        return mamamm(A, self.dim1, B, self.dim2, tarmask, True)


class Op2FWL(OpMessagePassing):

    def __init__(self) -> None:
        super().__init__(2, 1)

    def forward(self, X1: MaskedTensor, X2: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X1.masked_dim == 3, "X1 should be bxnxn adjacency matrix "
        assert X2.masked_dim == 3, "X2 should be bxnxn 2d representations"
        return super().forward(X1, X2, tarX)


class OpMessagePassingOnSubg2D(OpMessagePassing):

    def __init__(self) -> None:
        super().__init__(2, 1)

    def forward(self, A: MaskedTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert A.masked_dim == 3, "A should be bxnxn adjacency matrix "
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X, A, tarX)


class OpMessagePassingOnSubg3D(OpMessagePassing):

    def __init__(self, ) -> None:
        super().__init__(3, 1)

    def forward(self, A: MaskedTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert A.masked_dim == 3, "A should be bxnxn adjacency matrix "
        assert X.masked_dim == 4, "X should be bxnxnxn 3d representations"
        return super().forward(X, A, tarX)


class OpMessagePassingCrossSubg2D(OpMessagePassing):

    def __init__(self) -> None:
        super().__init__(1, 1)

    def forward(self, A: MaskedTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert A.masked_dim == 3, "A should be bxnxn adjacency matrix "
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(A, X, tarX)


class OpSpMessagePassing(Module):

    def __init__(self, dim1: int, dim2: int, aggr: str = "sum") -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.aggr = aggr

    def forward(self, A: SparseTensor, X: MaskedTensor,
                tarX: MaskedTensor) -> MaskedTensor:
        assert A.sparse_dim == 3, "A should be bxnxn adjacency matrix "
        return spmamm(A, self.dim1, X, self.dim2, tarX.mask, self.aggr)


class OpSpMessagePassingOnSubg2D(OpSpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__(1, 2, aggr)

    def forward(self, A: SparseTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2D representation "
        return super().forward(A, X, tarX)


class OpSpMessagePassingOnSubg3D(OpSpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__(1, 3, aggr)

    def forward(self, A: SparseTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxnxn 3D representation "
        return super().forward(A, X, tarX)


class OpSpMessagePassingCrossSubg2D(OpSpMessagePassing):

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__(1, 1, aggr)

    def forward(self, A: SparseTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2D representation "
        return super().forward(A, X, tarX)


class OpDiag(Module):

    def __init__(self, dims: Iterable[int]) -> None:
        super().__init__()
        self.dims = sorted(list(set(dims)))

    def forward(self, A: MaskedTensor) -> MaskedTensor:
        return A.diag(self.dims)


class OpDiag2D(OpDiag):

    def __init__(self) -> None:
        super().__init__([1, 2])

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X)


class OpPooling(Module):

    def __init__(self,
                 dims: Union[int, Iterable[int]],
                 pool: str = "sum") -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = [dims]
        self.dims = sorted(list(set(dims)))
        self.pool = pool

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        return getattr(X, self.pool)(dims=self.dims, keepdim=False)


class OpPoolingSubg2D(OpPooling):

    def __init__(self, pool: str = "sum") -> None:
        super().__init__([2], pool)

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X)


class OpPoolingSubg3D(OpPooling):

    def __init__(self, pool: str = "sum") -> None:
        super().__init__([3], pool)

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 4, "X should be bxnxnxn 3d representations"
        return super().forward(X)


class OpPoolingCrossSubg2D(OpPooling):

    def __init__(self, pool: str = "sum") -> None:
        super().__init__([1], pool)

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X)


class OpUnpooling(Module):

    def __init__(self, dims: Union[int, Iterable[int]]) -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = [dims]
        self.dims = sorted(list(set(dims)))

    def forward(self, X: MaskedTensor, tarX: MaskedTensor) -> MaskedTensor:
        return X.unpooling(self.dims, tarX)


class OpUnpoolingSubgNodes2D(OpUnpooling):

    def __init__(self) -> None:
        super().__init__([2])


class OpUnpoolingRootNodes2D(OpUnpooling):

    def __init__(self) -> None:
        super().__init__([1])
