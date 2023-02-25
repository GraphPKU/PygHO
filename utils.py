import torch.nn as nn
from torch import Tensor

act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True)}

class BatchNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, x: Tensor, batch: Tensor=None):
        if x.dim() == 2:
            return self.bn(x)
        elif x.dim() == 3:
            shape = x.shape
            x = self.bn(x.flatten(0, 1)).reshape(shape)
            return x
        else:
            raise NotImplementedError

class LayerNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)

class MLP(nn.Module):
    def __init__(self, hiddim: int, outdim: int, numlayer: int, tailact: bool, dp: float=0, bn: bool=True, ln: bool=False, act: str="relu", tailbias=True, multiparams: int=1, sharelin: bool=True, allshare: bool=True) -> None:
        super().__init__()
        if ln:
            bn=False
        assert numlayer >= 0
        if numlayer == 0:
            assert hiddim == outdim
            self.lins = nn.ModuleList([nn.Identity() for _ in range(multiparams)])
        else:
            self.lins = nn.ModuleList()
            lin0 = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
            if tailact:
                if ln:
                    lin0.append(nn.LayerNorm(outdim))
                if bn:
                    lin0.append(BatchNorm(outdim))
                if dp > 0:
                    lin0.append(nn.Dropout(dp, inplace=True))
                lin0.append(act_dict[act])
            for _ in range(numlayer-1):
                lin0.insert(0, act_dict[act])
                if dp > 0:
                    lin0.insert(0, nn.Dropout(dp, inplace=True))
                if bn:
                    lin0.insert(0, BatchNorm(hiddim))
                if ln: 
                    lin0.insert(0, nn.LayerNorm(hiddim))
                lin0.insert(0, nn.Linear(hiddim, hiddim))
            self.lins.append(lin0)
            if allshare:
                for _ in range(multiparams-1):
                    self.lins.append(lin0)
            elif sharelin:
                for _ in range(multiparams-1):
                    lin = nn.Sequential()
                    for mod in lin0:
                        if type(mod) is BatchNorm:
                            lin.append(BatchNorm(hiddim))
                        elif type(mod) is nn.LayerNorm: 
                            lin.append(nn.LayerNorm(hiddim))
                        else:
                            lin.append(mod)
                    self.lins.append(lin)
            else:
                for _ in range(multiparams-1):
                    lin0 = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
                    if tailact:
                        if ln:
                            lin0.append(nn.LayerNorm(outdim))
                        if bn:
                            lin0.append(BatchNorm(outdim))
                        if dp > 0:
                            lin0.append(nn.Dropout(dp, inplace=True))
                        lin0.append(act_dict[act])
                    for _ in range(numlayer-1):
                        lin0.insert(0, act_dict[act])
                        if dp > 0:
                            lin0.insert(0, nn.Dropout(dp, inplace=True))
                        if bn:
                            lin0.insert(0, BatchNorm(hiddim))
                        if ln: 
                            lin0.insert(0, nn.LayerNorm(hiddim))
                        lin0.insert(0, nn.Linear(hiddim, hiddim))
                    self.lins.append(lin0)

    def forward(self, x, *args, idx: int=0):
        return self.lins[idx](x)
                
