from torch import Tensor
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor
from typing import Union, Tuple, List, Iterable, Literal, Dict, Optional
from . import SpXOperator
from . import MaXOperator
from torch.nn import Module


class OpNodeMessagePassing(Module):

    def __init__(self,
                 mode: Literal["SS", "SD", "DD"] = "SS",
                 aggr: str = "sum") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpXOperator.OpNodeMessagePassing(aggr)
        elif mode == "SD":
            self.mod = MaXOperator.OpSpNodeMessagePassing(aggr)
        elif mode == "DD":
            assert aggr == "sum", f"aggr {aggr} is not implemented for DD"
            self.mod = MaXOperator.OpNodeMessagePassing()

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[Tensor, MaskedTensor]) -> Union[Tensor, MaskedTensor]:
        return self.mod.forward(A, X, X)


class Op2FWL(Module):

    def __init__(self,
                 mode: Literal["SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpXOperator.Op2FWL(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaXOperator.Op2FWL()
        else:
            raise NotImplementedError

    def forward(
        self,
        X1: Union[SparseTensor, MaskedTensor],
        X2: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(X1, X2, datadict, tarX)


class OpMessagePassingOnSubg2D(Module):

    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpXOperator.OpMessagePassingOnSubg2D(aggr)
        elif mode == "SD":
            self.mod = MaXOperator.OpSpMessagePassingOnSubg2D(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaXOperator.OpMessagePassingOnSubg2D()
        else:
            raise NotImplementedError

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(A, X, datadict, tarX)


class OpMessagePassingOnSubg3D(Module):

    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpXOperator.OpMessagePassingOnSubg3D(aggr)
        elif mode == "SD":
            self.mod = MaXOperator.OpSpMessagePassingOnSubg3D(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaXOperator.OpMessagePassingOnSubg3D()
        else:
            raise NotImplementedError

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(A, X, datadict, tarX)


class OpMessagePassingCrossSubg2D(Module):

    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpXOperator.OpMessagePassingCrossSubg2D(aggr)
        elif mode == "SD":
            self.mod = MaXOperator.OpMessagePassingCrossSubg2D(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaXOperator.OpMessagePassingCrossSubg2D()
        else:
            raise NotImplementedError

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(A, X, datadict, tarX)


class OpDiag2D(Module):

    def __init__(self, mode: Literal["D", "S"] = "S") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpXOperator.OpDiag2D()
        elif mode == "D":
            self.mod = MaXOperator.OpDiag2D()
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return self.mod.forward(X)


class OpPoolingSubg2D(Module):

    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpXOperator.OpPoolingSubg2D(pool)
        elif mode == "D":
            self.mod = MaXOperator.OpPoolingSubg2D(pool)
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return self.mod(X)


class OpPoolingSubg3D(Module):

    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpXOperator.OpPoolingSubg3D(pool)
        elif mode == "D":
            self.mod = MaXOperator.OpPoolingSubg3D(pool)
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return self.mod(X)


class OpPoolingCrossSubg2D(Module):

    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpXOperator.OpPoolingCrossSubg2D(pool)
        elif mode == "D":
            self.mod = MaXOperator.OpPoolingCrossSubg2D(pool)
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return self.mod(X)


class OpUnpoolingSubgNodes2D(Module):

    def __init__(self, mode: Literal["S", "D"] = "S") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpXOperator.OpUnpoolingSubgNodes2D()
        elif mode == "D":
            self.mod = MaXOperator.OpUnpoolingSubgNodes2D()

    def forward(
        self, X: Union[Tensor, MaskedTensor], tarX: Union[SparseTensor,
                                                          MaskedTensor]
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(X, tarX)


class OpUnpoolingRootNodes2D(Module):

    def __init__(self, mode: Literal["S", "D"] = "S") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpXOperator.OpUnpoolingRootNodes2D()
        elif mode == "D":
            self.mod = MaXOperator.OpUnpoolingRootNodes2D()

    def forward(
        self, X: Union[Tensor, MaskedTensor], tarX: Union[SparseTensor,
                                                          MaskedTensor]
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(X, tarX)
