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
    """
    Perform node-level message passing with an adjacency matrix A of shape (b, n, n) and node features X of shape (b, n).

    Args:

    - None

    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: MaskedTensor, X: MaskedTensor,
                tarX: MaskedTensor) -> Tensor:
        """
        Perform forward pass of node-level message passing.

        Args:

        - A (MaskedTensor): Adjacency matrix of shape (b, n, n).
        - X (MaskedTensor): Node features of shape (b, n).
        - tarX (MaskedTensor): Target node features of shape (b, n).

        Returns:

        - Tensor: The result of the message passing operation.

        """
        return mamamm(A, 2, X, 1, tarX.mask)


class OpSpNodeMessagePassing(Module):
    """
    Operator for node-level message passing.

    Args:

    - aggr (str, optional): The aggregation method for message passing (default: "sum").

    Attributes:
    
    - aggr (str): The aggregation method used for message passing.

    Methods:
    - forward(A: SparseTensor, X: Tensor, tarX: Tensor) -> Tensor: Perform node-level message passing.
    """
    def __init__(self, aggr: str = "sum") -> None:
        super().__init__()
        self.aggr = aggr

    def forward(self, A: SparseTensor, X: MaskedTensor,
                tarX: MaskedTensor) -> Tensor:
        """
        Perform forward pass of node-level message passing.

        Args:

        - A (SparseTensor): Adjacency matrix of shape (b, n, n).
        - X (MaskedTensor): Node features of shape (b, n).
        - tarX (MaskedTensor): Target node features of shape (b, n). 

        Returns:
        
        - Tensor: The result of the message passing operation.

        """
        return spmamm(A, 2, X, 1, tarX.mask, self.aggr)


class OpMessagePassing(Module):
    """
    General message passing operator for masked tensor adjacency and masked tensor tuple representation.

    This operator takes two input masked tensors 'A' and 'B' and performs message passing 
    between them to generate a new masked tensor 'tarX'. The resulting tensor has a shape of 
    (b,\* maskedshape1_dim1,\* maskedshape2_dim2,\*denseshapeshape), where 'b' represents the batch size.

    Args:

    - dim1 (int): The dimension along which message passing is applied in 'A'.
    - dim2 (int): The dimension along which message passing is applied in 'B'.
    """

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, A: MaskedTensor, B: MaskedTensor,
                tarX: MaskedTensor) -> MaskedTensor:
        """
        Perform message passing between two masked tensors.

        Args:

        - A (MaskedTensor): The first input masked tensor.
        - B (MaskedTensor): The second input masked tensor.
        - tarX (MaskedTensor): The target masked tensor. The output will use its mask

        Returns:

        - MaskedTensor: The result of message passing, represented as a masked tensor.

        Notes:

        - This method applies message passing between 'A' and 'B' to generate 'tarX'.
        - It considers the specified dimensions for message passing.
        """
        tarmask = tarX.mask
        return mamamm(A, self.dim1, B, self.dim2, tarmask, True)


class Op2FWL(OpMessagePassing):
    """
    Operator for simulating the 2-Folklore-Weisfeiler-Lehman (FWL) test. X <- X1 * X2.

    This operator is specifically designed for simulating the 2-Folklore-Weisfeiler-Lehman (FWL) test 
    by performing message passing between two masked tensors 'X1' and 'X2'. The result is masked as the
    target masked tensor 'tarX'.

    Args:
    
    - None

    See Also:
    
    - OpMessagePassing: The base class for generalized message passing.

    """
    def __init__(self) -> None:
        super().__init__(2, 1)

    def forward(self, X1: MaskedTensor, X2: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        """
        Simulate the 2-Folklore-Weisfeiler-Lehman (FWL) test by performing message passing.

        Args:
        
        - X1 (MaskedTensor): The first input masked tensor of shape (b, n, n,\*denseshapeshape1).
        - X2 (MaskedTensor): The second input masked tensor of shape (b, n, n,\*denseshapeshape2).
        - datadict (Dict): A dictionary for caching intermediate data.
        - tarX (MaskedTensor): The target masked tensor.
        """
        assert X1.masked_dim == 3, "X1 should be bxnxn adjacency matrix "
        assert X2.masked_dim == 3, "X2 should be bxnxn 2d representations"
        return super().forward(X1, X2, tarX)


class OpMessagePassingOnSubg2D(OpMessagePassing):
    """
    Operator for performing message passing on each subgraph for 2D subgraph Graph Neural Networks.

    This operator is designed for use in 2D subgraph Graph Neural Networks (GNNs). It extends the 
    base class `OpMessagePassing` to perform message passing on each subgraph represented by input tensors 
    'A' (adjacency matrix) and 'X' (2D representations). The result is stored in the target masked tensor 'tarX'.

    Args:
    
    - None

    See Also:
    
    - OpMessagePassing: The base class for generalized message passing.
    """

    def __init__(self) -> None:
        super().__init__(2, 1)

    def forward(self, A: MaskedTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        """
        Perform message passing on each subgraph for 2D subgraph Graph Neural Networks.

        Args:

        - A (MaskedTensor): The input masked tensor representing the adjacency matrix of subgraphs, of shape (b, n, n,\*denseshapeshape1).
        - X (MaskedTensor): The input masked tensor representing 2D representations of subgraph nodes, of shape (b, n, n,\*denseshapeshape2).
        - datadict (Dict): A dictionary for caching intermediate data (not used in this method).
        - tarX (MaskedTensor): The target masked tensor to mask the result.

        Returns:

        - MaskedTensor: The result of message passing on each subgraph.
        """
        assert A.masked_dim == 3, "A should be bxnxn adjacency matrix "
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X, A, tarX)


class OpMessagePassingOnSubg3D(OpMessagePassing):
    """
    Operator for performing message passing on each subgraph for 3D subgraph Graph Neural Networks.

    Args:

    - None

    See Also:

    - OpMessagePassing: The base class for generalized message passing.
    """
    def __init__(self, ) -> None:
        super().__init__(3, 1)

    def forward(self, A: MaskedTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        """
        Perform message passing on each subgraph for 3D subgraph Graph Neural Networks.

        Args:

        - A (MaskedTensor): The input masked tensor representing the adjacency matrix of subgraphs, of shape (b, n, n,\*denseshapeshape1)
        - X (MaskedTensor): The input masked tensor representing 3D representations of subgraph nodes, of shape (b, n, n, n,\*denseshapeshape2)
        - datadict (Dict): A dictionary for caching intermediate data (not used in this method).
        - tarX (MaskedTensor): The target masked tensor to mask the result,  of shape (b, n, n, n,\*denseshapeshape3).

        Notes:

        - denseshape1, denseshape2 must be broadcastable.
        """
        assert A.masked_dim == 3, "A should be bxnxn adjacency matrix " 
        assert X.masked_dim == 4, "X should be bxnxnxn 3d representations" 
        return super().forward(X, A, tarX)


class OpMessagePassingCrossSubg2D(OpMessagePassing):
    """
    Perform message passing across subgraphs within the 2D subgraph Graph Neural Network (GNN).

    Args:

    - None

    See Also:

    - OpMessagePassing: The base class for generalized message passing.

    Notes:

    - It assumes that 'A' represents the adjacency matrix of subgraphs, and 'X' represents 2D representations 
      of subgraph nodes.
    """
    def __init__(self) -> None:
        super().__init__(1, 1)

    def forward(self, A: MaskedTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        """
        Perform message passing across subgraphs within the 2D subgraph Graph Neural Network.

        Args:

        - A (MaskedTensor): The input masked tensor representing the adjacency matrix of subgraphs. of shape (b, n, n,\*denseshapeshape1).
        - X (MaskedTensor): The input masked tensor representing 2D representations of subgraph nodes. of shape (b, n, n,\*denseshapeshape2).
        - datadict (Dict): A dictionary for caching intermediate data (not used in this method).
        - tarX (MaskedTensor): The target masked tensor to store the result. of  shape (b, n, n,\*denseshapeshape3).

        Returns:

        - MaskedTensor: The result of message passing that bridges subgraphs.
        """
        assert A.masked_dim == 3, "A should be bxnxn adjacency matrix "
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(A, X, tarX)


class OpSpMessagePassing(Module):
    """
    OpMessagePassing but use sparse adjacency matrix.
    """
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
    """
    OpMessagePassingOnSubg2D but use sparse adjacency matrix.
    """
    def __init__(self, aggr: str = "sum") -> None:
        super().__init__(1, 2, aggr)

    def forward(self, A: SparseTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2D representation "
        return super().forward(A, X, tarX)


class OpSpMessagePassingOnSubg3D(OpSpMessagePassing):
    """
    OpMessagePassingOnSubg3D but use sparse adjacency matrix.
    """
    def __init__(self, aggr: str = "sum") -> None:
        super().__init__(1, 3, aggr)

    def forward(self, A: SparseTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxnxn 3D representation "
        return super().forward(A, X, tarX)


class OpSpMessagePassingCrossSubg2D(OpSpMessagePassing):
    """
    OpMessagePassingCrossSubg2D but use sparse adjacency matrix.
    """
    def __init__(self, aggr: str = "sum") -> None:
        super().__init__(1, 1, aggr)

    def forward(self, A: SparseTensor, X: MaskedTensor, datadict: Dict,
                tarX: MaskedTensor) -> MaskedTensor:
        assert X.masked_dim == 3, "X should be bxnxn 2D representation "
        return super().forward(A, X, tarX)


class OpDiag(Module):
    """
    Operator for extracting diagonal elements from a SparseTensor.

    Args:

    - dims (Iterable[int]): A list of dimensions along which to extract diagonal elements.

    """
    def __init__(self, dims: Iterable[int]) -> None:
        super().__init__()
        self.dims = sorted(list(set(dims)))

    def forward(self, A: MaskedTensor) -> MaskedTensor:
        """
        forward function

        Args:

        - A (MaskedTensor): The input masked Tensor

        Returns:

        - MaskedTensor: The diagonal elements.
        """
        return A.diag(self.dims)


class OpDiag2D(OpDiag):

    def __init__(self) -> None:
        super().__init__([1, 2])

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        """
        Extract diagonal elements from the input masked.

        Args:
        
        - A (MaskedTensor): The input MaskedTensor from which to extract diagonal elements. Be of shape (b, n, n,\*denseshapeshape)

        Returns:

        - MaskedTensor: of shape (b, n,\*denseshapeshape)

        Returns:

        - Union[Tensor, SparseTensor]: The extracted diagonal elements as either a dense or sparse tensor.
        """
        assert X
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
    """
    Operator for pooling node representations within each subgraph for 2D subgraph GNNs. 

    Parameters:
        - `pool` (str): The pooling operation to apply.
    """

    def __init__(self, pool: str = "sum") -> None:
        super().__init__([2], pool)

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        """
        Parameters:
            - `X` (MaskedTensor): The input MaskedTensor of shape(b, n, n,\*denseshapeshape) representing 2D node representations.

        Returns:
            - (Tensor): The pooled dense tensor. of shape (b, n,\*denseshapeshape)

        Raises:
            - AssertionError: If `X` is not 2D representations.
        """
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X)


class OpPoolingSubg3D(OpPooling):
    """
    Operator for pooling node representations within each subgraph for 3D subgraph GNNs. It returns sparse output only.

    Parameters:
        - `pool` (str): The pooling operation to apply.
    """
    def __init__(self, pool: str = "sum") -> None:
        super().__init__([3], pool)

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        """
        Parameters:
            - `X` (MaskedTensor): The input MaskedTensor of shape(b, n, n, n,\*denseshapeshape) representing 2D node representations.

        Returns:
            - (Tensor): The pooled dense tensor. of shape (b, n, n,\*denseshapeshape)

        Raises:
            - AssertionError: If `X` is not 2D representations.
        """
        assert X.masked_dim == 4, "X should be bxnxnxn 3d representations"
        return super().forward(X)


class OpPoolingCrossSubg2D(OpPooling):
    """
    Operator for pooling the same node representations within different subgraphsfor 2D subgraph GNNs. It returns dense output only.

    Parameters:
        - `pool` (str): The pooling operation to apply.
    """
    def __init__(self, pool: str = "sum") -> None:
        super().__init__([1], pool)

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        """
        Parameters:
            - `X` (MaskedTensor): The input MaskedTensor of shape(b, n, n,\*denseshapeshape) representing 2D node representations.

        Returns:
            - (Tensor): The pooled dense tensor. of shape (b, n,\*denseshapeshape)

        Raises:
            - AssertionError: If `X` is not 2D representations.
        """
        assert X.masked_dim == 3, "X should be bxnxn 2d representations"
        return super().forward(X)


class OpUnpooling(Module):
    """
    Operator for unpooling tensors by adding new dimensions.
    
    Parameters:
        - `dims` (int or Iterable[int]): The dimensions along which to unpool the tensor.
    """
    def __init__(self, dims: Union[int, Iterable[int]]) -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = [dims]
        self.dims = sorted(list(set(dims)))

    def forward(self, X: MaskedTensor, tarX: MaskedTensor) -> MaskedTensor:
        """
        Perform unpooling on tensors by adding new dimensions.

        Parameters:
            - `X` (MaskedTensor): The input tensor to unpool.
            - `tarX` (MaskedTensor): The target MaskedTensor to mask the output.

        Returns:
            - (MaskedTensor): The result of unpooling.
        """
        return X.unpooling(self.dims, tarX)


class OpUnpoolingSubgNodes2D(OpUnpooling):
    """
    Operator for copy node representations to the node representation of all subgraphs
    """
    def __init__(self) -> None:
        super().__init__([2])


class OpUnpoolingRootNodes2D(OpUnpooling):
    """
    Operator for copy root node representations to the subgraph rooted at i for all nodes
    """
    def __init__(self) -> None:
        super().__init__([1])
