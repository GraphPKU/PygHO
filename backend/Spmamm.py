from .MaTensor import MaskedTensor, filterinf
import torch
from torch import BoolTensor
from typing import Optional
from torch_scatter import scatter
from .SpTensor import SparseTensor

filled_value_dict = {"sum": 0, "mean": 0, "max": -torch.inf, "min": torch.inf}
filter_inf_ops = ["max", "min"]

def spmamm(A: SparseTensor,
           B: MaskedTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    A: (B, n, m, *) SparseTensor
    B: (B, m, *) MaskedTensor
    '''
    assert A.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    b, n = A.shape[0], A.shape[1]
    bij = A.indices
    Aval = A.values
    mult = torch.einsum("z...,zm...->zm...", Aval, B.fill_masked(filled_value_dict[aggr])[bij[0], bij[2]])
    val = scatter(mult,
                  bij[0] * n + bij[1],
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n))
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else B.mask)


def maspmm(B: MaskedTensor,
           A: SparseTensor,
           mask: Optional[BoolTensor] = None,
           aggr: str = "sum") -> MaskedTensor:
    '''
    B: (B, k, m, *) MaskedTensor
    A: (B, m, n, *) SparseTensor
    '''
    assert A.sparse_dim == 3, f"A should have 3 sparse dims, but input has {A.sparse_dim}"
    b, n = A.shape[0], A.shape[2]
    bij = A.indices
    Aval = A.values
    mult = torch.einsum("z...,zm...->zm...", Aval, B.fill_masked(filled_value_dict[aggr]).transpose(1, 2)[bij[0], bij[1]])
    val = scatter(mult,
                  bij[0] * n + bij[2],
                  dim=0,
                  dim_size=b * n,
                  reduce=aggr)
    ret = val.unflatten(0, (b, n)).transpose(1, 2)
    if aggr in filter_inf_ops:
        ret = filterinf(ret)
    return MaskedTensor(ret, mask if mask is not None else B.mask)


if __name__ == "__main__":
    # for debug
    b, n, m, l, d = 2, 3, 5, 7, 11

    device = torch.device("cuda")

    A = torch.rand((b, n, m, d), device=device)
    Amask = torch.rand_like(A[:, :, :, 0]) > 0.9
    MA = MaskedTensor(A, Amask)
    ind = Amask.to_sparse_coo().indices()
    SA = SparseTensor(ind, A[ind[0], ind[1], ind[2]], shape=MA.shape)
    B = torch.rand((b, m, l, d), device=device)
    Bmask = torch.rand_like(B[:, :, :, 0]) > 0.9
    MB = MaskedTensor(B, Bmask)
    ind = Bmask.to_sparse_coo().indices()
    SB = SparseTensor(ind, B[ind[0], ind[1], ind[2]], shape=MB.shape)
    mask = torch.ones((b, n, l), dtype=torch.bool, device=device)
    print(torch.max((spmamm(SA, MB, mask).data-torch.einsum("bnmd,bmld->bnld", MA.data, MB.data)).abs()))
    print(torch.max((maspmm(MA, SB, mask).data-torch.einsum("bnmd,bmld->bnld", MA.data, MB.data)).abs()))
