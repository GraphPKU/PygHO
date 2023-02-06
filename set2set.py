import torch.nn as nn
import torch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr import Aggregation
from typing import Optional
from torch import Tensor
from torch.nn import LayerNorm, Linear, MultiheadAttention
from torch_cluster import knn_graph


class MinDist(nn.Module):
    def __init__(self, feat: bool=False, concat: bool=False) -> None:
        super().__init__()
        self.feat = feat
        self.concat = concat

    def forward2dim(self, x, batch):
        ei = knn_graph(x, 1, batch, loop=False)
        if self.feat:
            ret = torch.zeros_like(x)
            ret[ei[1]] = x[ei[1]]-x[ei[0]]
            return ret
        else:
            ret = torch.zeros((x.shape[0]), device=x.device)
            ret[ei[1]] = torch.square(x[ei[1]]-x[ei[0]]).sum(dim=-1)
            return ret.unsqueeze(-1)

    def forward3dim(self, x, batch):
        B = x.shape[0]
        Ng = batch[-1] + 1
        Nn = x.shape[1]
        x = x.flatten(0, 1)
        batch = batch.unsqueeze(0).repeat(B, 1)
        offset = torch.arange(B, device=batch.device).reshape(-1, 1)*Ng
        batch = (batch + offset).flatten()
        ret = self.forward2dim(x, batch)
        return ret.reshape(B, Nn, -1)
    
    def forward(self, x, batch):
        if x.dim() == 2:
            ret = self.forward2dim(x, batch)
        elif x.dim() == 3:
            ret = self.forward3dim(x, batch)
        else:
            raise NotImplementedError
        if self.concat:
            ret = torch.concat((ret, x), dim=-1)
        return ret

class MaxCos(MinDist):
    def __init__(self, feat: bool=False, concat: bool=False) -> None:
        super().__init__(feat, concat)

    def forward2dim(self, x, batch):
        ei = knn_graph(x, 1, batch, loop=False, cosine=True)
        if self.feat:
            ret = torch.zeros_like(x)
            ret[ei[1]] = x[ei[1]]-x[ei[0]]
            return ret
        else:
            ret = torch.zeros((x.shape[0]), device=x.device)
            ret[ei[1]] = 1-torch.cosine_similarity(x[ei[1]], x[ei[0]], dim=-1)
            return ret.unsqueeze(-1)

# Set Transformer: copied from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/aggr/utils.py


class MultiheadAttentionBlock(torch.nn.Module):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper
    .. math::
        \mathrm{MAB}(\mathbf{x}, \mathbf{y}) &= \mathrm{LayerNorm}(\mathbf{h} +
        \mathbf{W} \mathbf{h})
        \mathbf{h} &= \mathrm{LayerNorm}(\mathbf{x} +
        \mathrm{Multihead}(\mathbf{x}, \mathbf{y}, \mathbf{y}))
    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.attn = MultiheadAttention(
            channels,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.lin = nn.Sequential(Linear(channels, channels), nn.ReLU(inplace=True))
        self.layer_norm1 = LayerNorm(channels) if layer_norm else nn.Identity()
        self.layer_norm2 = LayerNorm(channels) if layer_norm else nn.Identity()

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.lin.reset_parameters()
        if self.layer_norm1 is not None:
            self.layer_norm1.reset_parameters()
        if self.layer_norm2 is not None:
            self.layer_norm2.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, x_mask: Optional[Tensor] = None,
                y_mask: Optional[Tensor] = None) -> Tensor:
        """"""
        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, y_mask, need_weights=False)

        if x_mask is not None:
            out[~x_mask] = 0.

        out = out + x
        out = self.layer_norm1(out)
        out = out + self.lin(out)
        out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads}, '
                f'layer_norm={self.layer_norm1 is not None}, '
                f'dropout={self.dropout})')


class SetAttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm,
                                           dropout)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(x, x, mask, mask)


class SetTransformer(Aggregation):
    def __init__(
        self,
        channels: int,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.encoders = torch.nn.ModuleList([
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_encoder_blocks)
        ])

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()

    def forward(self, x, batch):

        x, mask = to_dense_batch(x, batch)

        for encoder in self.encoders:
            x = encoder(x, mask)

        x = x.flatten(0, 1)[mask.flatten()]
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.tensor([[ 0,  1, 0, 1],
                [ 2,  3, 2, 3],
                [ 4,  5, 4, 5],
                [ 6,  7, 6, 3],
                [ 8,  2, 8, 9],
                [4, 11, 10, 11]], dtype=torch.float, device=device)
    batch = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long,  device=device)
    print(MinDist().forward(x, batch))
    print(MaxCos().forward(x, batch))
    #print(SetTransformer(4, 2, 1, True, 0.0).forward(x, batch))
    x = torch.tensor([[ 0,  1, 0, 1],
                [ 2,  3, 2, 3],
                [ 4,  5, 4, 5],
                [ 6,  7, 6, 3],
                [ 8,  2, 8, 9],
                [4, 11, 10, 11]], dtype=torch.float, device=device).unsqueeze(0).repeat(4, 1, 1)
    batch = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long, device=device)
    print(MaxCos().forward(x, batch))