import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
from torch import Tensor

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, dims = None, orthoinit=False):
        super(AtomEncoder, self).__init__()
        if dims is None:
            dims = full_atom_feature_dims
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            if orthoinit:
                torch.nn.init.orthogonal_(emb.weight.data)
            else:
                torch.nn.init.xavier_uniform_(emb.weight.data)
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