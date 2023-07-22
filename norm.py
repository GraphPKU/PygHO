import torch
from typing import List, Callable
from torch_geometric.nn.norm import GraphNorm as PygGN, InstanceNorm as PygIN
from torch import Tensor
import torch.nn as nn

class NormMomentumScheduler:
    def __init__(self, mfunc: Callable, initmomentum: float, normtype=nn.BatchNorm1d) -> None:
        super().__init__()
        self.normtype = normtype
        self.mfunc = mfunc
        self.epoch = 0
        self.initmomentum = initmomentum
    
    def step(self, model: nn.Module):
        ratio = self.mfunc(self.epoch)
        if 1-1e-6<ratio<1+1e-6:
            return self.initmomentum
        curm = self.initmomentum*ratio
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