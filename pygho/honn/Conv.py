from torch import Tensor
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor
from typing import Union, Tuple, List, Iterable, Literal, Dict, Optional
from torch.nn import Module
from .utils import MLP
from . import TensorOp
from torch_geometric.nn import HeteroLinear
import torch.nn as nn


class NGNNConv(Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {}):
        super().__init__()
        self.aggr = TensorOp.OpMessagePassingOnSubg2D(mode, aggr)
        self.lin = MLP(indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        tX = X.tuplewiseapply(self.lin)
        ret = self.aggr.forward(A, tX, datadict, tX)
        return ret


class SSWLConv(Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {}):
        super().__init__()
        self.aggr1 = TensorOp.OpMessagePassingOnSubg2D(mode, aggr)
        self.aggr2 = TensorOp.OpMessagePassingCrossSubg2D(mode, aggr)
        self.lin = MLP(3 * indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        tX = X
        X1 = self.aggr1.forward(A, tX, datadict, tX)
        X2 = self.aggr2.forward(A, tX, datadict, tX)
        return X.catvalue([X1, X2], True).tuplewiseapply(self.lin)


class I2Conv(Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {}):
        super().__init__()
        self.aggr = TensorOp.OpMessagePassingOnSubg3D(mode, aggr)
        self.lin = MLP(indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        tX = X.tuplewiseapply(self.lin)
        ret = self.aggr.forward(A, tX, datadict, tX)
        return ret


class DSSGNNConv(Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr_subg: str = "sum",
                 aggr_global: str = "sum",
                 pool: str = "mean",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp: dict = {}):
        super().__init__()
        self.aggr_subg = TensorOp.OpMessagePassingOnSubg2D(mode, aggr_subg)
        self.pool2global = TensorOp.OpPoolingCrossSubg2D(mode[1], pool)
        self.aggr_global = TensorOp.OpNodeMessagePassing(mode, aggr_global)
        self.unpooling2subg = TensorOp.OpUnpoolingRootNodes2D(mode[1])
        self.lin = MLP(2 * indim, outdim, **mlp)

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        X1 = self.unpooling2subg.forward(
            self.aggr_subg(self.pool2global.forward(X)), X)
        X2 = self.aggr_subg.forward(A, X, datadict, X)
        return X2.catvalue(X1, True).tuplewiseapply(self.lin)


class PPGNConv(Module):
    '''
    message passing within each subgraph
    '''

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 mode: Literal["DD", "SS"] = "SS",
                 mlp: dict = {}):
        super().__init__()
        self.op = TensorOp.Op2FWL(mode, aggr)
        self.lin1 = MLP(indim, outdim, **mlp)
        self.lin2 = MLP(indim, outdim, **mlp)

    def forward(self, X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        return self.op.forward(X.tuplewiseapply(self.lin1),
                               X.tuplewiseapply(self.lin2), datadict, X)


class GNNAKConv(Module):

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 pool: str = "mean",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp0: dict = {},
                 mlp1: dict = {},
                 ctx: bool = True):
        super().__init__()
        self.lin0 = MLP(indim, indim, **mlp0)
        self.aggr = TensorOp.OpMessagePassingOnSubg2D(mode, aggr)
        self.diag = TensorOp.OpDiag2D(mode[1])
        self.pool2subg = TensorOp.OpPoolingSubg2D(mode[1], pool)
        self.unpool4subg = TensorOp.OpUnpoolingSubgNodes2D(mode[1], pool)
        self.ctx = ctx
        if ctx:
            self.pool2node = TensorOp.OpPoolingCrossSubg2D(mode[1], pool)
            self.unpool4rootnode = TensorOp.OpUnpoolingRootNodes2D(
                mode[1], pool)
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


class SUNConv(Module):

    def __init__(self,
                 indim: int,
                 outdim: int,
                 aggr: str = "sum",
                 pool: str = "mean",
                 mode: Literal["SD", "DD", "SS"] = "SS",
                 mlp0: dict = {},
                 mlp1: dict = {}):
        super().__init__()
        self.lin0 = MLP(indim, indim, **mlp0)
        self.aggr = TensorOp.OpMessagePassingOnSubg2D(mode, aggr)
        self.diag = TensorOp.OpDiag2D(mode[1])
        self.pool2subg = TensorOp.OpPoolingSubg2D(mode[1], pool)
        self.unpool4subg = TensorOp.OpUnpoolingSubgNodes2D(mode[1], pool)
        self.pool2node = TensorOp.OpPoolingCrossSubg2D(mode[1], pool)
        self.unpool4rootnode = TensorOp.OpUnpoolingRootNodes2D(mode[1], pool)
        self.lin = nn.Sequential(HeteroLinear(7 * indim, indim, 2, False),
                                 MLP(indim, outdim, **mlp1))

    def forward(self, A: Union[SparseTensor, MaskedTensor],
                X: Union[SparseTensor, MaskedTensor],
                datadict: dict) -> Union[SparseTensor, MaskedTensor]:
        X4 = self.aggr.forward(A, X.tuplewiseapply(self.lin0), datadict)
        Xdiag = self.diag.forward(X), X
        X1 = X
        X2 = self.unpool4subg(Xdiag)
        X3 = self.unpool4rootnode(Xdiag)
        X5 = self.unpool4rootnode(self.pool2node(X))
        X6 = self.unpool4subg(self.pool2subg(X))
        X7 = self.unpool4rootnode(self.pool2node(X4))
        X = X1.catvalue([X2, X3, X4, X5, X6, X7], True)
        X.diagonalapply(self.lin)
        return X