import torch
from torch import LongTensor
from typing import Optional, Tuple
from torch_scatter import scatter
from .SpTensor import SparseTensor


def combineind(a: LongTensor, b: LongTensor) -> LongTensor:
    '''
    a ( M)
    b ( M)
    '''
    assert a.dtype == torch.long and b.dtype == torch.long
    assert torch.all(a < (1 << 30)) and torch.all(b < (1 << 30))
    return torch.bitwise_or(torch.bitwise_left_shift(a, 32), b)


def decomposeind(combined_ind: LongTensor) -> Tuple[LongTensor, LongTensor]:
    b = torch.bitwise_and(combined_ind, 0xFFFFFFFF)
    a = torch.bitwise_right_shift(combined_ind, 32)
    return a, b


def ptr2batch(ptr: LongTensor, dim_size: int) -> LongTensor:
    tmp = torch.arange(dim_size, device=ptr.device, dtype=ptr.dtype)
    ret = torch.searchsorted(ptr, tmp, right=True) - 1
    return ret


def spspmm_ind(ind1: LongTensor, ind2: LongTensor) -> LongTensor:
    '''
    ind1 (2, M1)
    ind2 (2, M2)
    ind1, ind2 are indices of two sparse tensors.
    ik, kj -> (i, j) (b,a1,a2), b represent id of (i, j)
    '''
    k1, k2 = ind1[1], ind2[0]
    assert torch.all(torch.diff(k2) >= 0), "ind2[0] should be sorted"
    M1, M2 = k1.shape[0], k2.shape[0]
    upperbound = torch.searchsorted(k2, k1, right=True)
    lowerbound = torch.searchsorted(k2, k1, right=False)
    # k2[lowerbound[i]-1]< k1[i] < k2[upperbound[i]]
    matched_num = torch.clamp_min_(upperbound - lowerbound, 0)
    #print(lowerbound, upperbound, matched_num)
    retptr = torch.zeros((M1 + 1),
                         dtype=matched_num.dtype,
                         device=matched_num.device)
    torch.cumsum(matched_num, dim=0, out=retptr[1:])
    retsize = retptr[-1]

    #print(retptr)

    ret = torch.zeros((3, retsize), device=ind1.device, dtype=ind1.dtype)
    ret[1] = ptr2batch(retptr, retsize)
    #print(ret[1])
    torch.arange(retsize, out=ret[2], device=ret.device, dtype=ret.dtype)
    offset = (ret[2][retptr[:-1]] - lowerbound)[ret[1]]
    ret[2] -= offset

    combinedij = combineind(ind1[0][ret[1]], ind2[1][ret[2]])
    combinedij, taridx = torch.unique(combinedij,
                                      sorted=True,
                                      return_inverse=True)
    ij = torch.stack(decomposeind(combinedij))
    ret[0] = taridx

    sorted_idx = torch.argsort(ret[0])
    return ij, ret[:, sorted_idx]


def spsphadamard_ind(tarij: LongTensor, ij: LongTensor) -> LongTensor:
    combine_tarij = combineind(tarij[0], tarij[1])
    assert torch.all(
        torch.diff(combine_tarij) > 0), "tarij should be sorted and coalesce"
    combine_ij = combineind(ij[0], ij[1])

    b2a = torch.clamp_min_(
        torch.searchsorted(combine_tarij, combine_ij, right=True) - 1, 0)
    notmatchmask = (combine_ij != combine_tarij[b2a])
    b2a[notmatchmask] = -1
    return b2a  # have no repeated elements except 0


def filterij(tarij: LongTensor, ij: LongTensor, bkl: LongTensor):
    '''
    bkl to akl
    '''
    b2a = spsphadamard_ind(tarij, ij)
    a = b2a[bkl[0]]
    retmask = a >= 0
    akl = torch.stack((a[retmask], bkl[1][retmask], bkl[2][retmask]))
    return akl


def spsphadamard(A: SparseTensor,
                 B: SparseTensor,
                 b2a: Optional[LongTensor] = None) -> SparseTensor:
    assert A.sparse_dim == 2
    assert B.sparse_dim == 2
    assert A.is_coalesced(), "A should be coalesced"
    assert B.is_coalesced(), "B should be coalesced"
    ind1, val1 = A.indices, A.values
    ind2, val2 = B.indices, B.values
    if b2a is None:
        b2a = spsphadamard_ind(ind1, ind2)
    ei = torch.stack(
        (b2a, torch.arange(b2a.shape[0], device=b2a.device, dtype=b2a.dtype)))
    mask = (b2a >= 0)
    ei = ei[:, mask]
    if val1 is None:
        retval = val2[ei[1]]
    elif val2 is None:
        retval = val1[ei[0]]
    else:
        retval = val1[ei[0]] * val2[ei[1]]
    retind = ind2[:, mask]
    return SparseTensor(retind,
                        retval,
                        shape=list(A.shape[:2]) + list(retval.shape[1:]),
                        is_coalesced=True)


def spspmm(A: SparseTensor,
           B: SparseTensor,
           aggr: str = "sum",
           bkl: Optional[LongTensor] = None,
           tar_ij: Optional[LongTensor] = None,
           akl: Optional[LongTensor] = None) -> SparseTensor:
    assert A.sparse_dim == 2
    assert B.sparse_dim == 2
    assert A.is_coalesced(), "A should be coalesced"
    assert B.is_coalesced(), "B should be coalesced"
    if akl is not None:
        assert tar_ij is not None
        if A.values is None:
            mult = B.values[akl[2]]
        elif B.values is None:
            mult = A.values[akl[1]]
        else:
            mult = A.values[akl[1]] * B.values[akl[2]]
        retval = scatter(mult,
                         akl[0],
                         dim=0,
                         dim_size=tar_ij.shape[1],
                         reduce=aggr)
        return SparseTensor(tar_ij,
                            retval,
                            shape=[A.shape[0], B.shape[1]] +
                            list(retval.shape)[1:],
                            is_coalesced=True)
    else:
        if bkl is None:
            ij, bkl = spspmm_ind(A.indices, B.indices)
        if tar_ij is not None:
            akl = filterij(tar_ij, ij, bkl)
            return spspmm(A, B, akl=akl)
        else:
            return spspmm(A, B, akl=bkl, tar_ij=ij)


if __name__ == "__main__":
    # for debug
    from torch_scatter import scatter_add
    n, m, l = 300, 200, 400
    device = torch.device("cuda")
    A = torch.rand((n, m), device=device)
    A[torch.rand_like(A) > 0.9] = 0
    A = A.to_sparse_coo()
    B = torch.rand((m, l), device=device)
    B[torch.rand_like(B) > 0.9] = 0
    B = B.to_sparse_coo()
    ind1 = A.indices()
    val1 = A.values()
    ind2 = B.indices()
    val2 = B.values()

    C = A @ B
    C = C.coalesce()

    ij, bkl = spspmm_ind(ind1, ind2)
    mult = val1[bkl[1]] * val2[bkl[2]]
    outval = scatter_add(mult, bkl[0], dim_size=ij.shape[1])
    out = torch.sparse_coo_tensor(ij, outval)
    out = out.coalesce()
    # debug spspmm
    print(torch.max(torch.abs(C.indices() - out.indices())),
          torch.max(torch.abs(C.values() - out.values())))

    tar_ij = torch.stack(
        (torch.randint_like(ind1[0], n), torch.randint_like(ind1[0], l)))
    assert torch.all(
        tar_ij == torch.stack(decomposeind(combineind(tar_ij[0], tar_ij[1]))))
    tar_ij = torch.stack(
        decomposeind(
            torch.unique(combineind(tar_ij[0], tar_ij[1]), sorted=True)))
    akl = filterij(tar_ij, ij, bkl)
    mult = val1[akl[1]] * val2[akl[2]]
    outval = scatter_add(mult, akl[0], dim_size=tar_ij.shape[1])
    maskedout = torch.sparse_coo_tensor(tar_ij, outval)
    maskedout = maskedout.coalesce()
    # debug spspmm with target filter
    print(
        torch.max(
            torch.abs(maskedout.to_dense()[tar_ij[0], tar_ij[1]] -
                      C.to_dense()[tar_ij[0], tar_ij[1]])))

    Ap = torch.rand((n, m), device=device)
    Ap[torch.rand_like(Ap) > 0.9] = 0
    Ap = Ap.to_sparse_coo()
    Bp = torch.rand((n, m), device=device)
    Bp[torch.rand_like(Bp) > 0.9] = 0
    Bp = Bp.to_sparse_coo()
    print(
        torch.max(
            torch.multiply(Ap, Bp).to_dense() -
            spsphadamard(Ap, Bp).to_dense()))
