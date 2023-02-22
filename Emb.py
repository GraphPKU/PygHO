import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
from torch import Tensor
from typing import Optional, List
import torch.nn as nn
from utils import BatchNorm


def x2dims(x: Tensor):
    assert x.dim() == 2
    assert x.dtype == torch.int64
    ret = torch.max(x, dim=0)[0] + 1
    return ret.tolist()

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


def createemb(dim: int, emb_dim: int, zeropad: bool=False, max_norm: Optional[float]=None, orthoinit: bool=False):
    ret =  nn.Embedding(dim, emb_dim, max_norm=max_norm, padding_idx=0 if zeropad else None)
    if orthoinit:
        nn.init.orthogonal_(ret.weight.data)
    return ret


class SingleEmbedding(nn.Module):

    def __init__(self, emb_dim: int, dim: int, lastzeropad: int = 0, orthoinit=False, max_norm: Optional[float]=None, bn: bool=False, ln: bool=False, dp: float=0.0) -> None:
        super().__init__()
        self.emb = createemb(dim, emb_dim, lastzeropad>0, max_norm, orthoinit)
        if ln:
            bn=False
        self.postemb = nn.Sequential()
        if ln:
            self.postemb.append(nn.LayerNorm(emb_dim, elementwise_affine=False))
        if bn:
            self.postemb.append(BatchNorm(emb_dim))
        if dp > 0:
            self.postemb.append(nn.Dropout(dp, inplace=True))

    def forward(self, x: Tensor):
        return self.postemb(self.emb(x))


class MultiEmbedding(nn.Module):
    def __init__(self, emb_dim: int, dims: List[int], lastzeropad: int = 0, orthoinit=False, max_norm: Optional[float]=None, bn: bool=False, ln: bool=False, dp: float=0.0):
        super().__init__()
        self.embedding_list = nn.ModuleList()

        for i, dim in enumerate(dims):
            self.embedding_list.append(createemb(dim, emb_dim, len(dims) - i <= lastzeropad, max_norm, orthoinit))
        
        if ln:
            bn=False
        self.postemb = nn.Sequential()
        if ln:
            self.postemb.append(nn.LayerNorm(emb_dim, elementwise_affine=False))
        if bn:
            self.postemb.append(BatchNorm(emb_dim))
        if dp > 0:
            self.postemb.append(nn.Dropout(dp, inplace=True))


    def forward(self, x: Tensor):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.embedding_list[i](x.select(-1, i))
        return self.postemb(x_embedding)

class AtomEncoder(nn.Module):

    def __init__(self, emb_dim, dims = None, lastzeropad = 0, orthoinit=False):
        super(AtomEncoder, self).__init__()
        if dims is None:
            dims = full_atom_feature_dims
        self.atom_embedding_list = nn.ModuleList()

        for i, dim in enumerate(dims):
            emb = nn.Embedding(dim, emb_dim, padding_idx=0 if len(dims) - i <= lastzeropad else None)
            if orthoinit:
                nn.init.orthogonal_(emb.weight.data)
            else:
                nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x: Tensor):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x.select(-1, i))

        return x_embedding


class BondEncoder(AtomEncoder):
    
    def __init__(self, emb_dim, dims = None, **kwargs):
        if dims is None:
            dims = full_bond_feature_dims
        super().__init__(emb_dim, dims, **kwargs)