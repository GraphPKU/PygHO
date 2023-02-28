import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Sequential as PygSeq
from norm import NoneNorm, normdict

act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True)}

class MLP(nn.Module):
    def __init__(self, hiddim: int, outdim: int, numlayer: int, tailact: bool, dp: float=0, norm: str="bn", act: str="relu", tailbias=True, multiparams: int=1, sharelin: bool=True, allshare: bool=True, normparam: float=0.1) -> None:
        super().__init__()
        assert numlayer >= 0
        if numlayer == 0:
            assert hiddim == outdim
            self.lins = nn.ModuleList([NoneNorm() for _ in range(multiparams)])
        else:
            self.lins = nn.ModuleList()
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
            for _ in range(multiparams):
                lin = []
                for mod in lin0:
                    if type(mod) in normdict.values():
                        if allshare:
                            lin.append((mod, "x, batch -> x"))
                        else:
                            lin.append((normdict[norm](mod.num_features, normparam), "x, batch -> x"))
                    else:
                        if allshare or sharelin:
                            lin.append((mod, "x -> x"))
                        else:
                            if type(mod) is nn.Linear:
                                lin.append((nn.Linear(mod.in_features, mod.out_features, bias=not (mod.bias is None)), "x -> x"))
                            else:
                                lin.append((mod, "x -> x")) # Dropout, activation can be shared
                self.lins.append(PygSeq("x, batch", lin))
    def forward(self, x: Tensor, batch: Tensor=None, idx: int=0):
        return self.lins[idx](x, batch)
                

def freezeGNN(model: nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
