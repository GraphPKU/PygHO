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
        return mamamm(A, 2, X, 1, tarX.mask, 1)


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

    def __init__(self, dim1: int, dim2: int, broadcast_dims: int=0) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.broadcast_dims = broadcast_dims

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
        return mamamm(A, self.dim1+1, B, self.dim2+1, tarmask, self.broadcast_dims + 1)


class OpSpMessagePassing(Module):
    """
    OpMessagePassing but use sparse adjacency matrix.
    """
    def __init__(self, dim1: int, dim2: int, aggr: str = "sum", broadcast_dim: int = 0) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.aggr = aggr
        self.broadcast_dim = broadcast_dim

    def forward(self, A: SparseTensor, X: MaskedTensor,
                tarX: MaskedTensor) -> MaskedTensor:
        return spmamm(A, self.dim1+1, X, self.dim2+1, tarX.mask, self.aggr, broadcast_dim=self.broadcast_dim+1)


class OpDiag(Module):
    """
    Operator for extracting diagonal elements from a SparseTensor.

    Args:

    - dims (Iterable[int]): A list of dimensions along which to extract diagonal elements.

    """
    def __init__(self, dims: Iterable[int]) -> None:
        super().__init__()
        self.__dims = [ _ + 1 for _ in sorted(list(set(dims)))]

    def forward(self, A: MaskedTensor) -> MaskedTensor:
        """
        forward function

        Args:

        - A (MaskedTensor): The input masked Tensor

        Returns:

        - MaskedTensor: The diagonal elements.
        """
        return A.diag(self.__dims)


class OpPooling(Module):

    def __init__(self,
                 dims: Union[int, Iterable[int]],
                 pool: str = "sum") -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = [dims]
        self.__dims = [_+1 for _ in sorted(list(set(dims)))]
        self.pool = pool

    def forward(self, X: MaskedTensor) -> MaskedTensor:
        return getattr(X, self.pool)(dims=self.__dims, keepdim=False)


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
        self.__dims = [_+1 for _ in sorted(list(set(dims)))]

    def forward(self, X: MaskedTensor, tarX: MaskedTensor) -> MaskedTensor:
        """
        Perform unpooling on tensors by adding new dimensions.

        Parameters:
            - `X` (MaskedTensor): The input tensor to unpool.
            - `tarX` (MaskedTensor): The target MaskedTensor to mask the output.

        Returns:
            - (MaskedTensor): The result of unpooling.
        """
        return X.unpooling(self.__dims, tarX)


