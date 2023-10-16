"""
Representative GNN layers built upon message passing operations.
For all module, A means adjacency matrix, X means tuple representation 
mode SS means sparse adjacency and sparse X, SD means sparse adjacency and dense X, DD means dense adjacency and dense X.
datadict contains precomputation results.
"""

from torch import Tensor
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor
from typing import Union, Tuple, List, Iterable, Literal, Dict, Optional, Callable
from torch.nn import Module
from .utils import MLP
from . import TensorOp
from torch_geometric.nn import HeteroLinear
import torch.nn as nn


# NGNNConv: Nested Graph Neural Network Convolution Layer
class NGNNConv(Module):
    """
    Implementation of the NGNNConv layer based on the paper "Nested Graph Neural Networks" by Muhan Zhang and Pan Li, NeurIPS 2021.
    This layer performs message passing on 2D subgraph representations.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr (str): Aggregation method for message passing (e.g., "sum").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp (dict): Parameters for the MLP layer.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the NGNNConv layer.
    """

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {},
                 optuplefeat: str = "X",
                 opadj: str = "A",
                 message_func: Optional[Callable] = None):
        super().__init__()
        self.aggr = TensorOp.OpMessagePassingOnSubg2D(mode, aggr, optuplefeat,
                                                      opadj, message_func)
        self.lin = MLP(indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        tX = X.tuplewiseapply(self.lin)
        ret = self.aggr.forward(A, tX, datadict, tX)
        return ret


# SSWLConv: Subgraph Weisfeiler-Lehman Convolution Layer
class SSWLConv(Module):
    '''
    Implementation of the SSWLConv layer based on the paper "A complete expressiveness hierarchy for subgraph GNNs via subgraph Weisfeiler-Lehman tests" by Bohang Zhang et al., ICML 2023.
    This layer performs message passing on 2D subgraph representations and cross-subgraph pooling.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr (str): Aggregation method for message passing (e.g., "sum").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp (dict): Parameters for the MLP layer.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the SSWLConv layer.

    '''

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {},
                 optuplefeat: str = "X",
                 opadj: str = "A"):
        super().__init__()
        self.aggr1 = TensorOp.OpMessagePassingOnSubg2D(mode, aggr, optuplefeat,
                                                       opadj)
        self.aggr2 = TensorOp.OpMessagePassingCrossSubg2D(
            mode, aggr, optuplefeat, opadj)
        self.lin = MLP(3 * indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        tX = X
        X1 = self.aggr1.forward(A, tX, datadict, tX)
        X2 = self.aggr2.forward(A, tX, datadict, tX)
        return X.catvalue([X1, X2], True).tuplewiseapply(self.lin)


# I2Conv: I2-GNN Convolution Layer
class I2Conv(Module):
    """
    Implementation of the I2Conv layer based on the paper "Boosting the cycle counting power of graph neural networks with I2-GNNs" by Yinan Huang et al., ICLR 2023.
    This layer performs message passing on 3D subgraph representations.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr (str): Aggregation method for message passing (e.g., "sum").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp (dict): Parameters for the MLP layer.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the I2Conv layer.

    Notes:
    - This layer is based on the I2-GNN paper and performs message passing on 3D subgraph representations.
    """

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {},
                 optuplefeat: str = "X",
                 opadj: str = "A"):
        super().__init__()
        self.aggr = TensorOp.OpMessagePassingOnSubg3D(mode, aggr, optuplefeat,
                                                      opadj)
        self.lin = MLP(indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        tX = X.tuplewiseapply(self.lin)
        ret = self.aggr.forward(A, tX, datadict, tX)
        return ret


# DSSGNNConv: Equivariant Subgraph Aggregation Networks Convolution Layer
class DSSGNNConv(Module):
    """
    Implementation of the DSSGNNConv layer based on the paper "Equivariant subgraph aggregation networks" by Beatrice Bevilacqua et al., ICLR 2022.
    This layer performs message passing on 2D subgraph representations with subgraph pooling.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr_subg (str): Aggregation method for message passing within subgraphs (e.g., "sum").
    - aggr_global (str): Aggregation method for message passing in the global context (e.g., "sum").
    - pool (str): Pooling method (e.g., "mean").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp (dict): Parameters for the MLP layer.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the DSSGNNConv layer.
    """

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr_subg: str = "sum",
                 aggr_global: str = "sum",
                 pool: str = "mean",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {},
                 optuplefeat: str = "X",
                 opadj: str = "A"):
        super().__init__()
        self.aggr_subg = TensorOp.OpMessagePassingOnSubg2D(
            mode, aggr_subg, optuplefeat, opadj)
        self.pool2global = TensorOp.OpPoolingCrossSubg2D(mode[1], pool)
        self.aggr_global = TensorOp.OpNodeMessagePassing(mode, aggr_global)
        self.unpooling2subg = TensorOp.OpUnpoolingRootNodes2D(mode[1])
        self.lin = MLP(2 * indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        X1 = self.unpooling2subg.forward(
            self.aggr_global.forward(A, self.pool2global.forward(X)), X)
        X2 = self.aggr_subg.forward(A, X, datadict, X)
        return X2.catvalue(X1, True).tuplewiseapply(self.lin)


# PPGNConv: Provably Powerful Graph Networks Convolution Layer
class PPGNConv(Module):
    """
    Implementation of the PPGNConv layer based on the paper "Provably powerful graph networks" by Haggai Maron et al., NeurIPS 2019.
    This layer performs message passing with power-sum pooling on 2D subgraph representations.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr (str): Aggregation method for message passing (e.g., "sum").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp (dict): Parameters for the MLP layers.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the PPGNConv layer.

    """

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["DD", "SS"] = "SS",
                 mlp: dict = {},
                 optuplefeat: str = "X"):
        super().__init__()
        self.op = TensorOp.Op2FWL(mode, aggr, optuplefeat)
        self.lin1 = MLP(indim, outdim, **mlp)
        self.lin2 = MLP(indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        return self.op.forward(X.tuplewiseapply(self.lin1),
                               X.tuplewiseapply(self.lin2), datadict, X)


# GNNAKConv: Graph Neural Networks As Kernel Convolution layer
class GNNAKConv(Module):
    """
    Implementation of the GNNAKConv layer based on the paper "From stars to subgraphs: Uplifting any GNN with local structure awareness" by Lingxiao Zhao et al., ICLR 2022.
    This layer performs message passing on 2D subgraph representations with subgraph pooling and cross-subgraph pooling.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr (str): Aggregation method for message passing (e.g., "sum").
    - pool (str): Pooling method (e.g., "mean").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp0 (dict): Parameters for the first MLP layer.
    - mlp1 (dict): Parameters for the second MLP layer.
    - ctx (bool): Whether to include context information.

    Methods:

    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the GNNAKConv layer.

    """

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 pool: str = "mean",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp0: dict = {},
                 mlp1: dict = {},
                 ctx: bool = True,
                 optuplefeat: str = "X",
                 opadj: str = "A"):
        super().__init__()
        self.lin0 = MLP(indim, indim, **mlp0)
        self.aggr = TensorOp.OpMessagePassingOnSubg2D(mode, aggr, optuplefeat,
                                                      opadj)
        self.diag = TensorOp.OpDiag2D(mode[1])
        self.pool2subg = TensorOp.OpPoolingSubg2D(mode[1], pool)
        self.unpool4subg = TensorOp.OpUnpoolingSubgNodes2D(mode[1])
        self.ctx = ctx
        if ctx:
            self.pool2node = TensorOp.OpPoolingCrossSubg2D(mode[1], pool)
            self.unpool4rootnode = TensorOp.OpUnpoolingRootNodes2D(mode[1])
        self.lin = MLP(3 * indim if ctx else 2 * indim, outdim, **mlp1)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        X = self.aggr.forward(A, X.tuplewiseapply(self.lin0), datadict, X)
        X1 = self.unpool4subg.forward(self.diag.forward(X), X)
        X2 = self.unpool4subg.forward(self.pool2subg.forward(X), X)
        if self.ctx:
            X3 = self.unpool4rootnode.forward(self.pool2node.forward(X), X)
            return X2.catvalue([X1, X3], True).tuplewiseapply(self.lin)
        else:
            return X2.catvalue(X1, True).tuplewiseapply(self.lin)


# SUNConv: Subgraph Union Network Convolution Layer
class SUNConv(Module):
    """
    Implementation of the SUNConv layer based on the paper "Understanding and extending subgraph GNNs by rethinking their symmetries" by Fabrizio Frasca et al., NeurIPS 2022.
    This layer performs message passing on 2D subgraph representations with subgraph and cross-subgraph pooling.

    Args:

    - indim (int): Input feature dimension.
    - outdim (int): Output feature dimension.
    - aggr (str): Aggregation method for message passing (e.g., "sum").
    - pool (str): Pooling method (e.g., "mean").
    - mode (str): Mode for specifying tensor types (e.g., "SS" for sparse adjacency and sparse X).
    - mlp0 (dict): Parameters for the first MLP layer.
    - mlp1 (dict): Parameters for the second MLP layer.

    Methods:
    
    - forward(A: Union[SparseTensor, MaskedTensor], X: Union[SparseTensor, MaskedTensor], datadict: dict) -> Union[SparseTensor, MaskedTensor]:
      Forward pass of the SUNConv layer.

    Notes:
    - This layer is based on Symmetry Understanding Networks (SUN) and performs message passing on 2D subgraph representations with subgraph and cross-subgraph pooling.
    """

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 pool: str = "mean",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp0: dict = {},
                 mlp1: dict = {},
                 optuplefeat: str = "X",
                 opadj: str = "A"):
        super().__init__()
        self.lin0 = MLP(indim, indim, **mlp0)
        self.aggr = TensorOp.OpMessagePassingOnSubg2D(mode, aggr, optuplefeat,
                                                      opadj)
        self.diag = TensorOp.OpDiag2D(mode[1])
        self.pool2subg = TensorOp.OpPoolingSubg2D(mode[1], pool)
        self.unpool4subg = TensorOp.OpUnpoolingSubgNodes2D(mode[1])
        self.pool2node = TensorOp.OpPoolingCrossSubg2D(mode[1], pool)
        self.unpool4rootnode = TensorOp.OpUnpoolingRootNodes2D(mode[1])
        self.lin1_0 = HeteroLinear(7 * indim, indim, 2, False)
        self.lin1_1 = MLP(indim, outdim, **mlp1)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        X4 = self.aggr.forward(A, X.tuplewiseapply(self.lin0), datadict, X)
        Xdiag = self.diag.forward(X)
        X1 = X
        X2 = self.unpool4subg.forward(Xdiag, X)
        X3 = self.unpool4rootnode.forward(Xdiag, X)
        X5 = self.unpool4rootnode.forward(self.pool2node(X), X)
        X6 = self.unpool4subg.forward(self.pool2subg(X), X)
        X7 = self.unpool4rootnode.forward(self.pool2node(X4), X)
        X = X1.catvalue([X2, X3, X4, X5, X6, X7], True)
        X = X.diagonalapply(lambda val, ind: self.lin1_0(
            val.flatten(0, -2), ind.flatten()).unflatten(0, val.shape[0:-1]))
        X = X.tuplewiseapply(self.lin1_1)
        return X