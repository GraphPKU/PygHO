'''
A general MLP class
'''
import torch.nn as nn
from torch import Tensor
from typing import Callable
from torch import Tensor

# Norms for subgraph GNN


class NormMomentumScheduler:

    def __init__(self,
                 mfunc: Callable,
                 initmomentum: float,
                 normtype=nn.BatchNorm1d) -> None:
        super().__init__()
        self.normtype = normtype
        self.mfunc = mfunc
        self.epoch = 0
        self.initmomentum = initmomentum

    def step(self, model: nn.Module):
        ratio = self.mfunc(self.epoch)
        if 1 - 1e-6 < ratio < 1 + 1e-6:
            return self.initmomentum
        curm = self.initmomentum * ratio
        self.epoch += 1
        for mod in model.modules():
            if type(mod) is self.normtype:
                mod.momentum = curm
        return curm


class NoneNorm(nn.Module):

    def __init__(self, dim=0, normparam=0) -> None:
        super().__init__()
        self.num_features = dim

    def forward(self, x):
        return x


class BatchNorm(nn.Module):

    def __init__(self, dim, normparam=0.1) -> None:
        super().__init__()
        self.num_features = dim
        self.norm = nn.BatchNorm1d(dim, momentum=normparam)

    def forward(self, x: Tensor):
        if x.dim() == 2:
            return self.norm(x)
        elif x.dim() >= 3:
            shape = x.shape
            x = self.norm(x.flatten(0, -2)).reshape(shape)
            return x
        else:
            raise NotImplementedError


class LayerNorm(nn.Module):

    def __init__(self, dim, normparam=0.1) -> None:
        super().__init__()
        self.num_features = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        return self.norm(x)


normdict = {"bn": BatchNorm, "ln": LayerNorm, "none": NoneNorm}
act_dict = {
    "relu": nn.ReLU(inplace=True),
    "ELU": nn.ELU(inplace=True),
    "silu": nn.SiLU(inplace=True)
}


class MLP(nn.Module):

    def __init__(self,
                 hiddim: int,
                 outdim: int,
                 numlayer: int,
                 tailact: bool,
                 dp: float = 0,
                 norm: str = "bn",
                 act: str = "relu",
                 tailbias=True,
                 normparam: float = 0.1) -> None:
        super().__init__()
        assert numlayer >= 0
        if numlayer == 0:
            assert hiddim == outdim
            self.lins = NoneNorm()
        else:
            lin0 = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
            if tailact:
                lin0.append(normdict[norm](outdim, normparam))
                if dp > 0:
                    lin0.append(nn.Dropout(dp, inplace=True))
                lin0.append(act_dict[act])
            for _ in range(numlayer - 1):
                lin0.insert(0, act_dict[act])
                if dp > 0:
                    lin0.insert(0, nn.Dropout(dp, inplace=True))
                lin0.insert(0, normdict[norm](hiddim, normparam))
                lin0.insert(0, nn.Linear(hiddim, hiddim))
            self.lins = lin0

    def forward(self, x: Tensor, batch: Tensor = None):
        return self.lins(x)
