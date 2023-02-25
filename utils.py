import torch.nn as nn
from torch import Tensor

act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True)}

class BatchNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, x: Tensor):
        if x.dim() == 2:
            return self.bn(x)
        elif x.dim() == 3:
            shape = x.shape
            x = self.bn(x.flatten(0, 1)).reshape(shape)
            return x
        else:
            raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, hiddim: int, outdim: int, numlayer: int, tailact: bool, dp: float=0, bn: bool=True, ln: bool=False, act: str="relu", tailbias=True) -> None:
        super().__init__()
        if ln:
            bn=False
        assert numlayer >= 0
        if numlayer == 0:
            assert hiddim == outdim
            self.lin = nn.Identity()
        else:
            self.lin = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
            if tailact:
                if ln:
                    self.lin.append(nn.LayerNorm(outdim))
                if bn:
                    self.lin.append(BatchNorm(outdim))
                if dp > 0:
                    self.lin.append(nn.Dropout(dp, inplace=True))
                self.lin.append(act_dict[act])
            for _ in range(numlayer-1):
                self.lin.insert(0, act_dict[act])
                if dp > 0:
                    self.lin.insert(0, nn.Dropout(dp, inplace=True))
                if bn:
                    self.lin.insert(0, BatchNorm(hiddim))
                if ln: 
                    self.lin.insert(0, nn.LayerNorm(hiddim))
                self.lin.insert(0, nn.Linear(hiddim, hiddim))
    
    def forward(self, x, *args):
        return self.lin(x)
                
