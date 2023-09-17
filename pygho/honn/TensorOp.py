'''
Wrappers unifying operators for sparse and masked tensors
'''

from torch import Tensor
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor
from typing import Union, Tuple, List, Iterable, Literal, Dict, Optional
from . import SpOperator
from . import MaOperator
from torch.nn import Module


class OpNodeMessagePassing(Module):
    """
    Perform node-wise message passing with support for both sparse and masked tensors.

    This class wraps the message passing operator, allowing it to be applied to both sparse and masked tensors.
    It can perform node-wise message passing based on the provided mode and aggregation method.

    Args:
    
    - mode (Literal["SS", "SD", "DD"], optional): The mode indicating tensor types (default: "SS"). 
      SS means sparse adjacency and sparse X, SD means sparse adjacency and dense X, DD means dense adjacency and dense X.
    - aggr (str, optional): The aggregation method for message passing (default: "sum").

    See Also:

    - SpOperator.OpNodeMessagePassing: Sparse tensor node-wise message passing operator.
    - MaOperator.OpSpNodeMessagePassing: Masked tensor node-wise message passing operator for sparse adjacency.
    - MaOperator.OpNodeMessagePassing: Masked tensor node-wise message passing operator for dense adjacency.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[Tensor, MaskedTensor]) -> Union[Tensor, MaskedTensor]:
      Perform node-wise message passing on the input tensors based on the specified mode and aggregation method.
    """
    def __init__(self,
                 mode: Literal["SS", "SD", "DD"] = "SS",
                 aggr: str = "sum") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpOperator.OpNodeMessagePassing(aggr)
        elif mode == "SD":
            self.mod = MaOperator.OpSpNodeMessagePassing(aggr)
        elif mode == "DD":
            assert aggr == "sum", f"aggr {aggr} is not implemented for DD"
            self.mod = MaOperator.OpNodeMessagePassing()

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[Tensor, MaskedTensor]) -> Union[Tensor, MaskedTensor]:
        """
        Perform node-wise message passing on the input tensors.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input adjacency tensor.
        - X (Union[Tensor, MaskedTensor]): The input tensor representing tuple features.

        Returns:

        - Union[Tensor, MaskedTensor]: The result of node-wise message passing.
        """
        return self.mod.forward(A, X, X)


class Op2FWL(Module):
    """
    Simulate the 2-Folklore-Weisfeiler-Lehman (FWL) test with support for both sparse and masked tensors.

    This class allows you to simulate the 2-Folklore-Weisfeiler-Lehman (FWL) test by performing message passing 
    between two input tensors, X1 and X2. It supports both sparse and masked tensors and offers flexibility in
    specifying the aggregation method.

    Args:

    - mode (Literal["SS", "DD"], optional): The mode indicating tensor types (default: "SS").
      SS means sparse adjacency and sparse X, DD means dense adjacency and dense X.
    - aggr (Literal["sum", "mean", "max"], optional): The aggregation method for message passing (default: "sum").

    See Also:

    - SpOperator.Op2FWL: Sparse tensor operator for simulating 2-FWL.
    - MaOperator.Op2FWL: Masked tensor operator for simulating 2-FWL.
    """

    def __init__(self,
                 mode: Literal["SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum",
                 optuplefeat: str="X") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpOperator.Op2FWL(aggr, optuplefeat)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaOperator.Op2FWL()
        else:
            raise NotImplementedError

    def forward(
        self,
        X1: Union[SparseTensor, MaskedTensor],
        X2: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Simulate the 2-Folklore-Weisfeiler-Lehman (FWL) test by performing message passing.

        Args:

        - X1 (Union[SparseTensor, MaskedTensor]): The first input tensor.
        - X2 (Union[SparseTensor, MaskedTensor]): The second input tensor.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).
        - tarX (Optional[Union[SparseTensor, MaskedTensor]]): The target tensor to store the result.

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of simulating the 2-Folklore-Weisfeiler-Lehman (FWL) test.

        """
        return self.mod.forward(X1, X2, datadict, tarX)


class OpMessagePassingOnSubg2D(Module):

    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum",
                 optuplefeat: str = "X", opadj: str="A") -> None:
        """
        Perform message passing on each subgraph for 2D subgraph Graph Neural Networks with support for both sparse and masked tensors.

        This class is designed for performing message passing on each subgraph within 2D subgraph Graph Neural Networks.
        It supports both sparse and masked tensors and provides flexibility in specifying the aggregation method.

        Args:

        - mode (Literal["SD", "SS", "DD"], optional): The mode indicating tensor types (default: "SS").
          SS means sparse adjacency and sparse X, SD means sparse adjacency and dense X, DD means dense adjacency and dense X.
        - aggr (Literal["sum", "mean", "max"], optional): The aggregation method for message passing (default: "sum").

        See Also:

        - SpOperator.OpMessagePassingOnSubg2D: Sparse tensor operator for message passing on 2D subgraphs.
        - MaOperator.OpSpMessagePassingOnSubg2D: Masked tensor operator for message passing on 2D subgraphs.
        - MaOperator.OpMessagePassingOnSubg2D: Masked tensor operator for message passing on 2D subgraphs with dense adjacency.
        """
        super().__init__()
        if mode == "SS":
            self.mod = SpOperator.OpMessagePassingOnSubg2D(aggr, optuplefeat, opadj)
        elif mode == "SD":
            self.mod = MaOperator.OpSpMessagePassingOnSubg2D(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaOperator.OpMessagePassingOnSubg2D()
        else:
            raise NotImplementedError

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing on each subgraph for 2D subgraph Graph Neural Networks.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input tensor representing the adjacency matrix of subgraphs.
        - X (Union[SparseTensor, MaskedTensor]): The input tensor representing 2D representations of subgraph nodes.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).
        - tarX (Optional[Union[SparseTensor, MaskedTensor]]): The target tensor to store the result.

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing on each subgraph.
        """
        return self.mod.forward(A, X, datadict, tarX)


class OpMessagePassingOnSubg3D(Module):
    """
    Perform message passing on each subgraph for 3D subgraph Graph Neural Networks with support for both sparse and masked tensors.

    This class is designed for performing message passing on each subgraph within 3D subgraph Graph Neural Networks.
    It supports both sparse and masked tensors and provides flexibility in specifying the aggregation method.

    Args:

    - mode (Literal["SD", "SS", "DD"], optional): The mode indicating tensor types (default: "SS").
      SS means sparse adjacency and sparse X, SD means sparse adjacency and dense X, DD means dense adjacency and dense X.
    - aggr (Literal["sum", "mean", "max"], optional): The aggregation method for message passing (default: "sum").

    See Also:

    - SpOperator.OpMessagePassingOnSubg3D: Sparse tensor operator for message passing on 3D subgraphs.
    - MaOperator.OpSpMessagePassingOnSubg3D: Masked tensor operator for message passing on 3D subgraphs.
    - MaOperator.OpMessagePassingOnSubg3D: Masked tensor operator for message passing on 3D subgraphs with dense adjacency.

    """
    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum",
                 optuplefeat: str = "X", opadj: str="A") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpOperator.OpMessagePassingOnSubg3D(aggr, optuplefeat, opadj)
        elif mode == "SD":
            self.mod = MaOperator.OpSpMessagePassingOnSubg3D(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaOperator.OpMessagePassingOnSubg3D()
        else:
            raise NotImplementedError

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing on each subgraph for 3D subgraph Graph Neural Networks.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input tensor representing the adjacency matrix of subgraphs.
        - X (Union[SparseTensor, MaskedTensor]): The input tensor representing 3D representations of subgraph nodes.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).
        - tarX (Optional[Union[SparseTensor, MaskedTensor]]): The target tensor to store the result.

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing on each subgraph.
        """
        return self.mod.forward(A, X, datadict, tarX)


class OpMessagePassingCrossSubg2D(Module):
    """
    Perform message passing across subgraphs within the 2D subgraph Graph Neural Network (GNN) with support for both sparse and masked tensors.

    This class is designed for performing message passing across subgraphs within the 2D subgraph Graph Neural Network (GNN).
    It supports both sparse and masked tensors and provides flexibility in specifying the aggregation method.

    Args:

    - mode (Literal["SD", "SS", "DD"], optional): The mode indicating tensor types (default: "SS").
    - aggr (Literal["sum", "mean", "max"], optional): The aggregation method for message passing (default: "sum").

    See Also:

    - SpOperator.OpMessagePassingCrossSubg2D: Sparse tensor operator for cross-subgraph message passing in 2D GNNs.
    - MaOperator.OpSpMessagePassingCrossSubg2D: Masked tensor operator for cross-subgraph message passing in 2D GNNs.
    - MaOperator.OpMessagePassingCrossSubg2D: Masked tensor operator for cross-subgraph message passing in 2D GNNs with dense adjacency.

    """
    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum",
                 optuplefeat: str = "X", opadj: str="A") -> None:
        super().__init__()
        if mode == "SS":
            self.mod = SpOperator.OpMessagePassingCrossSubg2D(aggr, optuplefeat, opadj)
        elif mode == "SD":
            self.mod = MaOperator.OpMessagePassingCrossSubg2D(aggr)
        elif mode == "DD":
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaOperator.OpMessagePassingCrossSubg2D()
        else:
            raise NotImplementedError

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
        tarX: Optional[Union[SparseTensor, MaskedTensor]] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing across subgraphs within the 2D subgraph Graph Neural Network (GNN).

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input tensor representing the adjacency matrix of subgraphs.
        - X (Union[SparseTensor, MaskedTensor]): The input tensor representing 2D representations of subgraph nodes.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).
        - tarX (Optional[Union[SparseTensor, MaskedTensor]]): The target tensor to store the result.

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing across subgraphs.

        """
        return self.mod.forward(A, X, datadict, tarX)


class OpDiag2D(Module):
    """
    Perform diagonalization operation for 2D subgraph Graph Neural Networks with support for both sparse and masked tensors.

    Args:
    
    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S").
      S means sparse, D means dense

    See Also:

    - SpOperator.OpDiag2D: Sparse tensor operator for diagonalization in 2D GNNs.
    - MaOperator.OpDiag2D: Masked tensor operator for diagonalization in 2D GNNs.

    """
    def __init__(self, mode: Literal["D", "S"] = "S") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpDiag2D()
        elif mode == "D":
            self.mod = MaOperator.OpDiag2D()
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        """
        Perform diagonalization operation for 2D subgraph Graph Neural Networks.

        Args:

        - X (Union[MaskedTensor, SparseTensor]): The input tensor for diagonalization.

        Returns:

        - Union[MaskedTensor, Tensor]: The result of the diagonalization operation.

        """
        return self.mod.forward(X)


class OpPoolingSubg2D(Module):
    """
    Perform pooling operation for subgraphs within 2D subgraph Graph Neural Networks by reducing dimensions.

    Args:

    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S"). S means sparse, D means dense
    - pool (Literal["sum", "mean", "max"], optional): The pooling method (default: "sum").

    See Also:

    - SpOperator.OpPoolingSubg2D: Sparse tensor operator for pooling in 2D GNNs.
    - MaOperator.OpPoolingSubg2D: Masked tensor operator for pooling in 2D GNNs.
    """
    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpPoolingSubg2D(pool)
        elif mode == "D":
            self.mod = MaOperator.OpPoolingSubg2D(pool)
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return self.mod(X)


class OpPoolingSubg3D(Module):
    """
    This class is designed for performing pooling operation across subgraphs within the 2D subgraph Graph Neural Network (GNN).

    Args:

    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S"). S means sparse, D means dense.
    - pool (Literal["sum", "mean", "max"], optional): The pooling method (default: "sum").

    See Also:

    - SpOperator.OpPoolingCrossSubg2D: Sparse tensor operator for cross-subgraph pooling in 2D GNNs.
    - MaOperator.OpPoolingCrossSubg2D: Masked tensor operator for cross-subgraph pooling in 2D GNNs.
    """
    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpPoolingSubg3D(pool)
        elif mode == "D":
            self.mod = MaOperator.OpPoolingSubg3D(pool)
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
            self.mod = SpOperator.OpPoolingCrossSubg2D(pool)
        elif mode == "D":
            self.mod = MaOperator.OpPoolingCrossSubg2D(pool)
        else:
            raise NotImplementedError

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return self.mod(X)


class OpUnpoolingSubgNodes2D(Module):
    """
    This class is designed for performing unpooling operation for subgraph nodes within 2D subgraph Graph Neural Networks.
    It supports both sparse and masked tensors.

    Args:

    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S"). S means sparse, D means dense.

    See Also:

    - SpOperator.OpUnpoolingSubgNodes2D: Sparse tensor operator for unpooling subgraph nodes in 2D GNNs.
    - MaOperator.OpUnpoolingSubgNodes2D: Masked tensor operator for unpooling subgraph nodes in 2D GNNs.
    """
    def __init__(self, mode: Literal["S", "D"] = "S") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpUnpoolingSubgNodes2D()
        elif mode == "D":
            self.mod = MaOperator.OpUnpoolingSubgNodes2D()

    def forward(
        self, X: Union[Tensor, MaskedTensor], tarX: Union[SparseTensor,
                                                          MaskedTensor]
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(X, tarX)


class OpUnpoolingRootNodes2D(Module):
    """
    This class is designed for performing unpooling operation for root nodes within 2D subgraph Graph Neural Networks.
    It supports both sparse and masked tensors.

    Args:

    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S").

    See Also:
    
    - SpOperator.OpUnpoolingRootNodes2D: Sparse tensor operator for unpooling
    """
    def __init__(self, mode: Literal["S", "D"] = "S") -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpUnpoolingRootNodes2D()
        elif mode == "D":
            self.mod = MaOperator.OpUnpoolingRootNodes2D()

    def forward(
        self, X: Union[Tensor, MaskedTensor], tarX: Union[SparseTensor,
                                                          MaskedTensor]
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(X, tarX)
