import torch
from torch_scatter import scatter_add
from torch import Tensor

@torch.no_grad()
def multinomial_sample_batch(prob: Tensor, batch: Tensor):
    N = batch[-1] + 1
    dim = prob.dim()
    assert dim == 1 or dim == 2
    cumprob = torch.zeros_like(prob)
    if dim == 1:
        torch.cumsum(prob[:-1], dim=-1, out=cumprob[1:])
    elif dim == 2:
        torch.cumsum(prob[:, :-1], dim=-1, out=cumprob[:, 1:])
    probsum = scatter_add(prob, batch, dim=-1)
    offset = torch.zeros_like(probsum)
    if dim == 1:
        torch.cumsum(probsum[:-1], dim=-1, out=offset[1:])
    elif dim == 2:
        torch.cumsum(probsum[:, :-1], dim=-1, out=offset[:, 1:])
    probsample = torch.rand_like(probsum) * probsum + offset
    rawsample = torch.searchsorted(cumprob, probsample, right=True) - 1
    return rawsample

if __name__ == "__main__":
    prob = torch.tensor([0.25, 0.35, 0.25, 0.15,   0, 0, 0, 0, 1,   0.9,  0, 0.1])
    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    for i in range(10):
        rawsample=multinomial_sample_batch(prob, batch)
        print(rawsample)
        tprob = prob.clone()
        tprob[rawsample] = 0
        print(tprob)
        rawsample=multinomial_sample_batch(tprob, batch)
        print(rawsample)
    