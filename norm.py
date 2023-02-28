import torch
from typing import List, Callable
from torch_geometric.nn.norm import GraphNorm as PygGN, InstanceNorm as PygIN
from torch import Tensor
import torch.nn as nn

def expandbatch(x: Tensor, batch: Tensor):
    if batch is None:
        return x.flatten(0, 1), None
    else:
        R = x.shape[0]
        N = batch[-1] + 1
        offset = N*torch.arange(R, device=x.device).reshape(-1, 1)
        batch = batch.unsqueeze(0) + offset
        return x.flatten(0, 1), batch.flatten()


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
    
    def forward(self, x, batch):
        return x

class BatchNorm(nn.Module):
    def __init__(self, dim, normparam=0.1) -> None:
        super().__init__()
        self.num_features = dim
        self.norm = nn.BatchNorm1d(dim, momentum=normparam)
    
    def forward(self, x: Tensor, batch: Tensor):
        if x.dim() == 2:
            return self.norm(x)
        elif x.dim() == 3:
            shape = x.shape
            x = self.norm(x.flatten(0, 1)).reshape(shape)
            return x
        else:
            raise NotImplementedError

class LayerNorm(nn.Module):
    def __init__(self, dim, normparam=0.1) -> None:
        super().__init__()
        self.num_features = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, batch: Tensor):
        return self.norm(x)

class InstanceNorm(nn.Module):
    def __init__(self, dim, normparam=0.1) -> None:
        super().__init__()
        self.norm = PygIN(dim, momentum=normparam)
        self.num_features = dim

    def forward(self, x: Tensor, batch: Tensor):
        if x.dim() == 2:
            return self.norm(x, batch)
        elif x.dim() == 3:
            shape = x.shape
            x, batch = expandbatch(x, batch)
            x = self.norm(x, batch).reshape(shape)
            return x
        else:
            raise NotImplementedError

class GraphNorm(nn.Module):
    def __init__(self, dim, normparam=0.1) -> None:
        super().__init__()
        self.norm = PygGN(dim)
        self.num_features = dim

    def forward(self, x: Tensor, batch: Tensor):
        if x.dim() == 2:
            return self.norm(x, batch)
        elif x.dim() == 3:
            shape = x.shape
            x, batch = expandbatch(x, batch)
            x = self.norm(x, batch).reshape(shape)
            return x
        else:
            raise NotImplementedError

normdict = {"bn": BatchNorm, "ln": LayerNorm, "in": InstanceNorm, "gn": GraphNorm, "none": NoneNorm}
basenormdict = {"bn": nn.BatchNorm1d, "ln": None, "in": PygIN, "gn": None, "none": None}

if __name__ == "__main__":
    x = torch.randn((3,4,5))
    batch = torch.tensor((0,0,1,2))
    x, batch = expandbatch(x, batch)
    print(x.shape, batch)
    x = torch.randn((3,4,5))
    batch = None
    x, batch = expandbatch(x, batch)
    print(x.shape, batch)

    print(list(InstanceNorm(1000).modules()))