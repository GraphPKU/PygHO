import torch
from torch import Tensor, LongTensor
from typing import Tuple

def torch_scatter_reduce(dim: int, src: Tensor, ind: LongTensor, dim_size: int, aggr: str) -> Tensor:
    assert dim==0, "other dim not implemented"
    assert ind.ndim == 1, "indice must be 1-d"
    if aggr in ["min", "max"]:
        aggr = "a"+aggr
    onedim = src.ndim - 1
    dim_size = dim_size
    ret = torch.zeros_like(src[[0]].expand((dim_size,)+(-1,)*onedim))
    ret.scatter_reduce_(dim, ind.reshape((-1,)+(1,)*onedim).expand_as(src), src, aggr, include_self=False)
    return ret