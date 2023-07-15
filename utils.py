import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Sequential as PygSeq
from norm import NoneNorm, normdict

act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True)}

class MLP(nn.Module):
    def __init__(self, hiddim: int, outdim: int, numlayer: int, tailact: bool, dp: float=0, norm: str="bn", act: str="relu", tailbias=True, normparam: float=0.1) -> None:
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
            for _ in range(numlayer-1):
                lin0.insert(0, act_dict[act])
                if dp > 0:
                    lin0.insert(0, nn.Dropout(dp, inplace=True))
                lin0.insert(0, normdict[norm](hiddim, normparam))
                lin0.insert(0, nn.Linear(hiddim, hiddim))
            self.lins = lin0
    def forward(self, x: Tensor, batch: Tensor=None):
        return self.lins(x)
