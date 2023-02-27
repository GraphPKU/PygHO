import torch.nn as nn
from torch import Tensor
import torch
from typing import List
from torch_geometric.nn.norm import GraphNorm as PygGN, InstanceNorm as PygIN
from torch_geometric.nn import Sequential as PygSeq

act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True)}

def expandbatch(x: Tensor, batch: Tensor):
    if batch is None:
        return x.flatten(0, 1), None
    else:
        R = x.shape[0]
        N = batch[-1] + 1
        offset = N*torch.arange(R, device=x.device).reshape(-1, 1)
        batch = batch.unsqueeze(0) + offset
        return x.flatten(0, 1), batch.flatten()


class NoneNorm(nn.Module):
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.num_features = dim
    
    def forward(self, x, batch):
        return x

class BatchNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.num_features = dim
        self.norm = nn.BatchNorm1d(dim, track_running_stats=False)
    
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
    def __init__(self, dim) -> None:
        super().__init__()
        self.num_features = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, batch: Tensor):
        return self.norm(x)

class InstanceNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.norm = PygIN(dim)
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
    def __init__(self, dim) -> None:
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

class MLP(nn.Module):
    def __init__(self, hiddim: int, outdim: int, numlayer: int, tailact: bool, dp: float=0, norm: str="bn", act: str="relu", tailbias=True, multiparams: int=1, sharelin: bool=True, allshare: bool=True) -> None:
        super().__init__()
        assert numlayer >= 0
        if numlayer == 0:
            assert hiddim == outdim
            self.lins = nn.ModuleList([NoneNorm() for _ in range(multiparams)])
        else:
            self.lins = nn.ModuleList()
            lin0 = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
            if tailact:
                lin0.append(normdict[norm](outdim))
                if dp > 0:
                    lin0.append(nn.Dropout(dp, inplace=True))
                lin0.append(act_dict[act])
            for _ in range(numlayer-1):
                lin0.insert(0, act_dict[act])
                if dp > 0:
                    lin0.insert(0, nn.Dropout(dp, inplace=True))
                lin0.insert(0, normdict[norm](hiddim))
                lin0.insert(0, nn.Linear(hiddim, hiddim))
            for _ in range(multiparams):
                lin = []
                for mod in lin0:
                    if type(mod) in normdict.values():
                        if allshare:
                            lin.append((mod, "x, batch -> x"))
                        else:
                            lin.append((normdict[norm](mod.num_features), "x, batch -> x"))
                    else:
                        if allshare or sharelin:
                            lin.append((mod, "x -> x"))
                        else:
                            if type(mod) is nn.Linear:
                                lin.append((nn.Linear(mod.in_features, mod.out_features, bias=not (mod.bias is None)), "x -> x"))
                            else:
                                lin.append((mod, "x -> x")) # Dropout, activation can be shared
                self.lins.append(PygSeq("x, batch", lin))
    def forward(self, x, batch=None, idx: int=0):
        return self.lins[idx](x, batch)
                
if __name__ == "__main__":
    x = torch.randn((3,4,5))
    batch = torch.tensor((0,0,1,2))
    x, batch = expandbatch(x, batch)
    print(x.shape, batch)
    x = torch.randn((3,4,5))
    batch = None
    x, batch = expandbatch(x, batch)
    print(x.shape, batch)