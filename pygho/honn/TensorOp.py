'''
Wrappers unifying operators for sparse and masked tensors
'''

from torch import Tensor
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor
from typing import Union, Tuple, List, Iterable, Literal, Dict, Optional, Callable
from . import SpOperator
from . import MaOperator
from torch.nn import Module
from pygho.backend.utils import torch_scatter_reduce
import torch

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


class OpNodePooling(Module):
    """
    Pool node representations to graph representations with support for both sparse and masked routines.

    Args:
    
    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S"). 
      S means sparse routine,  D means dense routine.
    - aggr (str, optional): The aggregation method for message passing (default: "sum").

    Methods:

    - forward(X: Union[Tensor, MaskedTensor]) -> Tensor:
      Pool node representations to graph representations.
    """

    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__()
        self.mode = mode
        self.pool = pool

    def forward(self, X: Union[Tensor, MaskedTensor],
        datadict: Optional[Dict] = None) -> Tensor:
        """
        Perform node-wise message passing on the input tensors.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input adjacency tensor.
        - X (Union[Tensor, MaskedTensor]): The input tensor representing tuple features.

        Returns:

        - Union[Tensor, MaskedTensor]: The result of node-wise message passing.
        """
        if self.mode == "S":
            h_graph: Tensor = torch_scatter_reduce(0, X, datadict["batch"],
                                        datadict["num_graphs"], self.pool)
        elif self.mode == "D":
            h_graph: Tensor = getattr(X, self.pool)(dims=[1], keepdim=False).data
        return h_graph
    
class OpNodeUnPooling(Module):
    """
    Unpool graph representations to node representations with support for both sparse and masked routines.

    Args:
    
    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S"). 
      S means sparse routine,  D means dense routine.
    - aggr (str, optional): The aggregation method for message passing (default: "sum").

    Methods:

    - forward(X: Union[Tensor, MaskedTensor]) -> Tensor:
      Pool node representations to graph representations.
    """

    def __init__(self,
                 mode: Literal["S", "D"] = "S") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, X: Tensor, datadict: Optional[Dict] = None) -> Union[Tensor, MaskedTensor]:
        """
        Perform node-wise message passing on the input tensors.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input adjacency tensor.
        - X (Union[Tensor, MaskedTensor]): The input tensor representing tuple features.

        Returns:

        - Union[Tensor, MaskedTensor]: The result of node-wise message passing.
        """
        if self.mode == "S":
            ret: Tensor = X[datadict["batch"]]
        elif self.mode == "D":
            ret: MaskedTensor = datadict["x"].tuplewiseapply(lambda x: torch.repeat_interleave(X.unsqueeze(1), x.shape[1], dim=1))
        return ret


class OpMessagePassing(Module):
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
    def __init__(self,
                 mode: Literal["SD", "SS", "DD"],
                 aggr: Literal["sum", "mean", "max"],
                 op1: str,
                 dim1: int, 
                 op2: str,
                 dim2: int,
                 op0: str,
                 broadcast_dim: int,
                 message_func: Optional[Callable] = None) -> None:
        super().__init__()
        self.mode = mode
        if mode == "SS":
            self.mod = SpOperator.OpMessagePassing(op0, op1, dim1, op2, dim2, aggr, message_func, broadcast_dim)
        elif mode == "SD":
            assert message_func is None, "general message passing with message_func is not implemented for Dense"
            self.mod = MaOperator.OpSpMessagePassing(dim1, dim2, aggr, broadcast_dim)
        elif mode == "DD":
            assert message_func is None, "general message passing with message_func is not implemented for Dense"
            assert aggr == "sum", "only sum aggragation implemented for Dense adjacency"
            self.mod = MaOperator.OpMessagePassing(dim1, dim2, broadcast_dim)
        else:
            raise NotImplementedError

    def forward(
        self,
        X1: Union[SparseTensor, MaskedTensor],
        X2: Union[SparseTensor, MaskedTensor],
        X0: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None,
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing on each subgraph for 2D subgraph Graph Neural Networks.

        Args:

        - X1 (Union[SparseTensor, MaskedTensor]): The input tensor corresponding to op1
        - X2 (Union[SparseTensor, MaskedTensor]): The input tensor corresponding to op2
        - X0 (Union[SparseTensor, MaskedTensor]): The target tensor to store the result.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data.

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing on each subgraph.
        """
        if self.mode == "SS":
            return self.mod.forward(X1, X2, datadict, X0)
        else:
            return self.mod.forward(X1, X2, X0)

class Op2FWL(OpMessagePassing):
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
                 optuplefeat: str = "X") -> None:
        super().__init__(mode, aggr, optuplefeat, 0, optuplefeat, 1, optuplefeat, 0)

    def forward(
        self,
        X1: Union[SparseTensor, MaskedTensor],
        X2: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Simulate the 2-Folklore-Weisfeiler-Lehman (FWL) test by performing message passing.

        Args:

        - X1 (Union[SparseTensor, MaskedTensor]): The first input tensor.
        - X2 (Union[SparseTensor, MaskedTensor]): The second input tensor.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of simulating the 2-Folklore-Weisfeiler-Lehman (FWL) test.

        """
        return super().forward(X1, X2, X1, datadict)


class OpMessagePassingOnSubg2D(OpMessagePassing):

    def __init__(self,
                 mode: Literal["SD", "SS", "DD"] = "SS",
                 aggr: Literal["sum", "mean", "max"] = "sum",
                 optuplefeat: str = "X",
                 opadj: str = "A",
                 message_func: Optional[Callable] = None) -> None:
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
        super().__init__(mode, aggr, optuplefeat, 1, opadj, 0, optuplefeat, 0, message_func)

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing on each subgraph for 2D subgraph Graph Neural Networks.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input tensor representing the adjacency matrix of subgraphs.
        - X (Union[SparseTensor, MaskedTensor]): The input tensor representing 2D representations of subgraph nodes.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing on each subgraph.
        """
        return super().forward(X, A, X, datadict)


class OpMessagePassingOnSubg3D(OpMessagePassing):
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
                 optuplefeat: str = "X",
                 opadj: str = "A",
                 message_func: Optional[Callable] = None) -> None:
        super().__init__(mode, aggr, optuplefeat, 2, opadj, 0, optuplefeat, 0, message_func)

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing on each subgraph for 3D subgraph Graph Neural Networks.

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input tensor representing the adjacency matrix of subgraphs.
        - X (Union[SparseTensor, MaskedTensor]): The input tensor representing 3D representations of subgraph nodes.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing on each subgraph.
        """
        return super().forward(X, A, X, datadict)


class OpMessagePassingCrossSubg2D(OpMessagePassing):
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
                 optuplefeat: str = "X",
                 opadj: str = "A",
                 message_func: Optional[Callable] = None) -> None:
        super().__init__(mode, aggr, opadj, 1, optuplefeat, 0, optuplefeat, 0, message_func)

    def forward(
        self,
        A: Union[SparseTensor, MaskedTensor],
        X: Union[SparseTensor, MaskedTensor],
        datadict: Optional[Dict] = None
    ) -> Union[SparseTensor, MaskedTensor]:
        """
        Perform message passing across subgraphs within the 2D subgraph Graph Neural Network (GNN).

        Args:

        - A (Union[SparseTensor, MaskedTensor]): The input tensor representing the adjacency matrix of subgraphs.
        - X (Union[SparseTensor, MaskedTensor]): The input tensor representing 2D representations of subgraph nodes.
        - datadict (Optional[Dict]): A dictionary for caching intermediate data (not used in this method).

        Returns:

        - Union[SparseTensor, MaskedTensor]: The result of message passing across subgraphs.

        """
        return super().forward(A, X, X, datadict)


class OpDiag(Module):

    def __init__(self, dims: Iterable[int], return_sparse: bool, mode: Literal["D", "S"]):
        super().__init__()
        if mode == "D":
            assert not return_sparse, "dense input only produce dense output"
            self.mod = MaOperator.OpDiag(dims)
        elif mode == "S":
            self.mod = SpOperator.OpDiag(dims, return_sparse=return_sparse)
        else:
            raise NotImplementedError(f"unknown mode {mode}")

    def forward(self, X: Union[SparseTensor, MaskedTensor])-> Union[SparseTensor, Tensor, MaskedTensor]:
        return self.mod.forward(X)

class OpDiag2D(OpDiag):
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
        super().__init__([0, 1], False, mode)

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
        return super().forward(X)


class OpPooling(Module):
    """
    Perform pooling operation for subgraphs within 2D subgraph Graph Neural Networks by reducing dimensions.

    Args:

    - dims (Iterable[int]): The dims to reduce
    - mode (Literal["S", "D"]): The mode indicating tensor types. S means sparse, D means dense
    - pool (Literal["sum", "mean", "max"]): The pooling method.

    See Also:

    - SpOperator.OpPoolingSubg2D: Sparse tensor operator for pooling in 2D GNNs.
    - MaOperator.OpPoolingSubg2D: Masked tensor operator for pooling in 2D GNNs.
    """
    def __init__(self,
                 dims: Iterable[int], 
                 mode: Literal["S", "D"],
                 pool: str,
                 return_sparse: bool) -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpPooling(dims, pool, return_sparse=return_sparse)
        elif mode == "D":
            assert not return_sparse, "dense input only produce dense output"
            self.mod = MaOperator.OpPooling(dims, pool)
        else:
            raise NotImplementedError
    
    def forward(self, X: Union[SparseTensor, MaskedTensor]) -> Union[SparseTensor, Tensor, MaskedTensor]:
        return self.mod.forward(X)


class OpPoolingSubg2D(OpPooling):
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
        super().__init__([1], mode, pool, False)

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return super().forward(X)


class OpPoolingSubg3D(OpPooling):
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
        super().__init__([2], mode, pool, mode=="S")

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, SparseTensor]:
        return super().forward(X)


class OpPoolingCrossSubg2D(OpPooling):

    def __init__(self,
                 mode: Literal["S", "D"] = "S",
                 pool: str = "sum") -> None:
        super().__init__([0], mode, pool, False)

    def forward(
            self, X: Union[MaskedTensor,
                           SparseTensor]) -> Union[MaskedTensor, Tensor]:
        return super().forward(X)


class OpUnpooling(Module):

    def __init__(self, mode: Literal["S", "D"], dims: Iterable[int], fromdense1dim: bool) -> None:
        super().__init__()
        if mode == "S":
            self.mod = SpOperator.OpUnpooling(dims, fromdense1dim=fromdense1dim)
        elif mode == "D":
            self.mod = MaOperator.OpUnpooling(dims)
        else:
            raise NotImplementedError(f"unknown mode {mode}")
    
    def forward(
        self, X: Union[Tensor, MaskedTensor, SparseTensor], tarX: Union[SparseTensor,
                                                          MaskedTensor]
    ) -> Union[SparseTensor, MaskedTensor]:
        return self.mod.forward(X, tarX)


class OpUnpoolingSubgNodes2D(OpUnpooling):
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
        super().__init__(mode, [1], True)
        

    def forward(
        self, X: Union[Tensor, MaskedTensor], tarX: Union[SparseTensor,
                                                          MaskedTensor]
    ) -> Union[SparseTensor, MaskedTensor]:
        return super().forward(X, tarX)


class OpUnpoolingRootNodes2D(OpUnpooling):
    """
    This class is designed for performing unpooling operation for root nodes within 2D subgraph Graph Neural Networks.
    It supports both sparse and masked tensors.

    Args:

    - mode (Literal["S", "D"], optional): The mode indicating tensor types (default: "S").

    See Also:
    
    - SpOperator.OpUnpoolingRootNodes2D: Sparse tensor operator for unpooling
    """

    def __init__(self, mode: Literal["S", "D"] = "S") -> None:
        super().__init__(mode, [0], True)
    
    def forward(self, X: Union[Tensor, MaskedTensor], tarX: Union[SparseTensor, MaskedTensor]) -> Union[SparseTensor, MaskedTensor]:
        return super().forward(X, tarX)
