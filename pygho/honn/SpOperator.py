"""
Operators for SparseTensor
"""
from torch import Tensor, LongTensor
from pygho.backend.SpTensor import SparseTensor
from ..backend.Spspmm import spspmm, spspmpnn
from ..backend.Spmm import spmm
from ..backend.SpTensor import SparseTensor
from typing import Optional, Iterable, Dict, Union, List, Tuple, Callable
from torch.nn import Module

KEYSEP = "___"


def parse_precomputekey(model: Module) -> List[str]:
    """
    Parse and return precompute keys from a PyTorch model.

    Args:

    - model (Module): The PyTorch model to parse.

    Returns:
    
    - List[str]: A list of unique precompute keys found in the model.

    Example:
    
    ::

        model = MyModel()  # Initialize your PyTorch model
        precompute_keys = parse_precomputekey(model)

    Notes:
    - This function is useful for extracting precompute keys from message-passing models.
    - It iterates through the model's modules and identifies instances of OpMessagePassing modules.
    - The precompute keys associated with these modules are collected and returned as a list.

    """
    ret = []
    for mod in model.modules():
        if isinstance(mod, OpMessagePassing):
            ret.append(mod.precomputekey)
    return sorted(list(set(ret)))


class OpNodeMessagePassing(Module):
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

    def forward(self,
                A: SparseTensor,
                X: Tensor,
                tarX: Optional[Tensor] = None) -> Tensor:
        """
        Perform node-level message passing.

        Args:

        - A (SparseTensor): The adjacency matrix of the graph.
        - X (Tensor): The node feature tensor.
        - tarX (Tensor): The target node feature tensor. of no use

        Returns:

        - Tensor: The result of node-level message passing (AX).

        """
        assert A.sparse_dim == 2, "A is adjacency matrix of the whole graph of shape nxn"
        return spmm(A, 1, X, self.aggr)


class OpMessagePassing(Module):
    """
    Generalized message passing on tuple features.

    This class operates on two sparse tensors A and B and performs message passing based on specified operations and dimensions.

    Args:

    - op0 (str, optional): The operation name for the first input (default: "X"). 
      It is used to compute precomputekey and retrieve precomputation results
    - op1 (str, optional): The operation name for the second input (default: "X").
    - dim1 (int, optional): The dimension to apply op0 (default: 1).
    - op2 (str, optional): The operation name for the third input (default: "A").
    - dim2 (int, optional): The dimension to apply op2 (default: 0).
    - aggr (str, optional): The aggregation method for message passing (default: "sum").


    Attributes:

    - dim1 (int): The dimension to apply op0.
    - dim2 (int): The dimension to apply op2.
    - precomputekey (str): The precomputed key for caching intermediate data.
    - aggr (str): The aggregation method used for message passing.

    Methods:
    
    - forward(A: SparseTensor, B: SparseTensor, datadict: Dict, tarX: Optional[SparseTensor] = None) -> SparseTensor: Perform generalized message passing.

    Notes:
    
    - This class is designed for generalized message passing on tuple features.
    - It supports specifying custom operations and dimensions for message passing.
    - The `forward` method performs the message passing operation and returns the result.

    """

    def __init__(self,
                 op0: str = "X",
                 op1: str = "X",
                 dim1: int = 1,
                 op2: str = "A",
                 dim2: int = 0,
                 aggr: str = "sum",
                 message_func: Optional[Callable] = None,
                 broadcast_dims: int = 0) -> None:
        super().__init__()
        self.broadcast_dims = broadcast_dims
        self.dim1 = dim1
        self.dim2 = dim2
        self.precomputekey = f"{op0}{KEYSEP}{op1}{KEYSEP}{dim1}{KEYSEP}{op2}{KEYSEP}{dim2}{KEYSEP}{broadcast_dims}"
        self.aggr = aggr
        self.message_func = message_func
        self.use_mpnn = not (message_func is None)

    def forward(self,
                A: SparseTensor,
                B: SparseTensor,
                datadict: Dict,
                tarX: Optional[SparseTensor] = None) -> SparseTensor:
        """
        Perform generalized message passing.

        Args:

        - A (SparseTensor): The first input sparse tensor.
        - B (SparseTensor): The second input sparse tensor.
        - datadict (Dict): A dictionary for caching intermediate data. Containing precomputation results.
        - tarX (Optional[SparseTensor]): The target sparse tensor (default: None).

        Returns:

        - SparseTensor: The result of generalized message passing.

        Notes:

        - This method performs the generalized message passing operation using the provided inputs.
        - It supports caching intermediate data in the `datadict` dictionary.

        """
        if self.use_mpnn:
            assert not (
                tarX is None
            ), "target representation is a must when message func is not None"
            return spspmpnn(
                A, self.dim1, B, self.dim2, tarX,
                datadict.get(f"{self.precomputekey}{KEYSEP}acd", None),
                self.message_func, self.aggr)
        else:
            return spspmm(
                A,
                self.dim1,
                B,
                self.dim2,
                self.aggr,
                acd=datadict.get(f"{self.precomputekey}{KEYSEP}acd", None),
                bcd=datadict.get(f"{self.precomputekey}{KEYSEP}bcd", None),
                tar_ind=datadict.get(f"{self.precomputekey}{KEYSEP}tarind",
                                     None) if tarX is None else tarX.indices)


class OpDiag(Module):
    """
    Operator for extracting diagonal elements from a SparseTensor.

    Args:

    - dims (Iterable[int]): A list of dimensions along which to extract diagonal elements.
    - return_sparse (bool, optional): Whether to return the diagonal elements as a SparseTensor of a Tensor (default: False).

    Methods:

    - forward(A: SparseTensor) -> Union[Tensor, SparseTensor]: Extract diagonal elements from the input SparseTensor.

    Notes:

    - This class is used to extract diagonal elements from a SparseTensor along specified dimensions.
    - You can choose to return the diagonal elements as either a dense or sparse tensor.

    """

    def __init__(self,
                 dims: Iterable[int],
                 return_sparse: bool = False) -> None:
        super().__init__()
        self.dims = sorted(list(set(dims)))
        self.return_sparse = return_sparse

    def forward(self, A: SparseTensor) -> Union[Tensor, SparseTensor]:
        return A.diag(self.dims, return_sparse=self.return_sparse)


class OpPooling(Module):
    """
    Operator for pooling tuple representations by reducing dimensions.

    Args:

    - dims (Union[int, Iterable[int]]): The dimensions along which to apply pooling.
    - pool (str, optional): The pooling operation to apply (default: "sum").
    - return_sparse (bool, optional): Whether to return the pooled tensor as a SparseTensor (default: False).

    Methods:

    - forward(X: SparseTensor) -> Union[SparseTensor, Tensor]: Apply pooling operation to the input SparseTensor.

    """

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
        """
        Apply pooling operation to the input SparseTensor.

        Args:

        - X (SparseTensor): The input SparseTensor to which pooling is applied.

        Returns:
        
        - Union[SparseTensor, Tensor]: The pooled tensor as either a dense or sparse tensor.
        """
        return getattr(X, self.pool)(self.dims,
                                     return_sparse=self.return_sparse)



class OpUnpooling(Module):
    """
    Operator for unpooling tensors by adding new dimensions.
    
    Parameters:
        - `dims` (int or Iterable[int]): The dimensions along which to unpool the tensor.
        - `fromdense1dim` (bool, optional): Whether to perform unpooling from dense 1D. Default is True.
    """

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
        """
        Perform unpooling on tensors by adding new dimensions.

        Parameters:
            - `X` (Union[Tensor, SparseTensor]): The input tensor to unpool.
            - `tarX` (SparseTensor): The target SparseTensor.

        Returns:
            - (SparseTensor): The result of unpooling as a SparseTensor.
        """
        if isinstance(X, Tensor):
            leftdim = list(set(range(tarX.sparse_dim)) - set(self.dims))
            assert len(leftdim) == 1, "canonly pooling from 1 dim"
            return tarX.unpooling_fromdense1dim(leftdim[0], X)
        else:
            return X.unpooling(self.dims, tarX)

