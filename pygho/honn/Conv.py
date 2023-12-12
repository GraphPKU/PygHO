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
import torch

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
        ret = self.aggr.forward(A, tX, datadict)
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
        X1 = self.aggr1.forward(A, tX, datadict)
        X2 = self.aggr2.forward(A, tX, datadict)
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
        ret = self.aggr.forward(A, tX, datadict)
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
        X2 = self.aggr_subg.forward(A, X, datadict)
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
                               X.tuplewiseapply(self.lin2), datadict)


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
        X = self.aggr.forward(A, X.tuplewiseapply(self.lin0), datadict)
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
        self.lin1_0_0 = nn.Linear(7 * indim, indim)
        self.lin1_0_1 = nn.Linear(7 * indim, indim)
        self.lin1_1 = MLP(indim, outdim, **mlp1)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        X4 = self.aggr.forward(A, X.tuplewiseapply(self.lin0), datadict)
        Xdiag = self.diag.forward(X)
        X1 = X
        X2 = self.unpool4subg.forward(Xdiag, X)
        X3 = self.unpool4rootnode.forward(Xdiag, X)
        X5 = self.unpool4rootnode.forward(self.pool2node(X), X)
        X6 = self.unpool4subg.forward(self.pool2subg(X), X)
        X7 = self.unpool4rootnode.forward(self.pool2node(X4), X)
        X = X1.catvalue([X2, X3, X4, X5, X6, X7], True)
        X = X.diagonalapply(self.lin1_0_0, self.lin1_0_1)
        X = X.tuplewiseapply(self.lin1_1)
        return X
    

class IGN2Conv(nn.Module):
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
                 pool: str = "mean",
                 mode: Literal["S", "D"] = "S",
                 mlp: dict = {},
                 optuplefeat: str = "X"):
        super().__init__()
        self.mode = mode
        self.diag = TensorOp.OpDiag2D(mode)
        self.pool2subg = TensorOp.OpPoolingSubg2D(mode, pool)
        self.unpool4subg = TensorOp.OpUnpoolingSubgNodes2D(mode)
        self.pool2node = TensorOp.OpPoolingCrossSubg2D(mode, pool)
        self.unpool4rootnode = TensorOp.OpUnpoolingRootNodes2D(mode)
        self.poolnode2full = TensorOp.OpNodePooling(mode, pool)
        self.unpoolnode = TensorOp.OpNodeUnPooling(mode)
        self.lins = nn.ModuleList([nn.Identity()] + [nn.Linear(indim, outdim, bias=False) for _ in range(15)])
        self.lin2 = MLP(outdim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        diag_part = self.diag(X)
        sum_diag_part = self.poolnode2full.forward(diag_part, datadict)
        sum_of_rows = self.pool2subg.forward(X)
        sum_of_cols = self.pool2node.forward(X)
        sum_all = self.poolnode2full(sum_of_rows, datadict)  # N x D
        # op10 - (1234) + (14)(23) - identity
        op10 = X.tuplewiseapply(self.lins[10])
        ret = op10
        # op1 - (1234) - extract diag
        ret = ret.add(X.diagonalapply(self.lins[1], torch.zeros_like), True)
        # op2 - (1234) + (12)(34) - place sum of diag on diag
        ret = ret.add(self.unpool4subg.forward(self.unpoolnode.forward(sum_diag_part, datadict), X).diagonalapply(self.lins[2], torch.zeros_like), True)
        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        ret = ret.add(self.unpool4subg.forward(sum_of_rows, X).diagonalapply(self.lins[3], torch.zeros_like), True)
        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        ret = ret.add(self.unpool4subg.forward(sum_of_cols, X).diagonalapply(self.lins[4], torch.zeros_like), True)
        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        ret = ret.add(self.unpool4subg.forward(self.unpoolnode.forward(sum_all, datadict), X).diagonalapply(self.lins[5], torch.zeros_like), True)
        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        ret = ret.add(self.unpool4subg.forward(sum_of_rows, X).tuplewiseapply(self.lins[6]), True)
        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        ret = ret.add(self.unpool4rootnode.forward(sum_of_rows, X).tuplewiseapply(self.lins[7]), True)
        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        ret = ret.add(self.unpool4subg.forward(sum_of_cols, X).tuplewiseapply(self.lins[8]), True)
        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        ret = ret.add(self.unpool4rootnode.forward(sum_of_cols, X).tuplewiseapply(self.lins[9]), True)
        # op11 - (1234) + (13)(24) - transpose
        ret = ret.add(X.transpose(1, {"S": 0, "D": 2}[self.mode]).tuplewiseapply(self.lins[11]), True) 
        # op12 - (1234) + (234)(1) - place ii element in row i
        ret = ret.add(self.unpool4subg.forward(diag_part, X).tuplewiseapply(self.lins[12]), True)
        # op13 - (1234) + (134)(2) - place ii element in col i
        ret = ret.add(self.unpool4rootnode.forward(diag_part, X).tuplewiseapply(self.lins[13]), True)
        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        ret = ret.add(self.unpool4rootnode.forward(self.unpoolnode.forward(sum_diag_part, datadict), X).tuplewiseapply(self.lins[14]), True)
        # op15 - sum of all ops - place sum of all entries in all entries
        ret = ret.add(self.unpool4rootnode.forward(self.unpoolnode.forward(sum_diag_part, datadict), X).tuplewiseapply(self.lins[14]), True)
        return ret.tuplewiseapply(self.lin2)