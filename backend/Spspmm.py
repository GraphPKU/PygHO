import torch
from torch import LongTensor
from typing import Optional, Tuple
from torch_scatter import scatter
from SpTensor import SparseTensor
import warnings


def combineind(a: LongTensor, b: LongTensor) -> LongTensor:
    '''
    a ( M)
    b ( M)
    hash (a, b) into a tensor of shape (M).
    hash keeps the lexicographical order between two tuples
    '''
    assert a.dtype == torch.long and b.dtype == torch.long, "can only hash long tensor"
    assert torch.all(a < (1 << 30)) and torch.all(
        b < (1 << 30)), "this hash is not injective and may cause bug "
    return torch.bitwise_or(torch.bitwise_left_shift(a, 32), b)


def decomposeind(combined_ind: LongTensor) -> Tuple[LongTensor, LongTensor]:
    '''
    transfer hash into pairs
    '''
    b = torch.bitwise_and(combined_ind, 0xFFFFFFFF)
    a = torch.bitwise_right_shift(combined_ind, 32)
    return a, b


def ptr2batch(ptr: LongTensor, dim_size: int) -> LongTensor:
    '''
    ptr: LongTensor, ptr[0]=0, torch.all(diff(ptr)>=0) is true
    output: (dim_size) LongTensor batch, batch[ptr[i]:ptr[i+1]]=i
    '''
    assert ptr[0] == 0 and torch.all(
        torch.diff(ptr) >= 0), "should put in a ptr tensor"
    assert ptr[-1] == dim_size, "dim_size should match ptr"
    tmp = torch.arange(dim_size, device=ptr.device, dtype=ptr.dtype)
    ret = torch.searchsorted(ptr, tmp, right=True) - 1
    return ret


def spspmm_ind(ind1: LongTensor, ind2: LongTensor) -> LongTensor:
    '''
    ind1 (2, M1)
    ind2 (2, M2)
    ind1, ind2 are indices of two sparse tensors.
    (i, k), (k, j) -> (i, j) (b, c, d), b represent index of (i, j), a represent index of (i, k), b represent index of (k, j)
    '''
    k1, k2 = ind1[1], ind2[0]
    assert torch.all(torch.diff(k2) >= 0), "ind2[0] should be sorted"
    M1, M2 = k1.shape[0], k2.shape[0]

    # for each k in k1, it can match a interval of k2 as k2 is sorted
    upperbound = torch.searchsorted(k2, k1, right=True)
    lowerbound = torch.searchsorted(k2, k1, right=False)
    matched_num = torch.clamp_min_(upperbound - lowerbound, 0)

    # ptr[i] provide the offset to place pair of ind1[:, i] and the matched ind2
    retptr = torch.zeros((M1 + 1),
                         dtype=matched_num.dtype,
                         device=matched_num.device)
    torch.cumsum(matched_num, dim=0, out=retptr[1:])
    retsize = retptr[-1]

    # fill the output with ptr
    ret = torch.zeros((3, retsize), device=ind1.device, dtype=ind1.dtype)
    ret[1] = ptr2batch(retptr, retsize)
    torch.arange(retsize, out=ret[2], device=ret.device, dtype=ret.dtype)
    offset = (ret[2][retptr[:-1]] - lowerbound)[ret[1]]
    ret[2] -= offset

    # compute the ij pair index
    combinedij = combineind(ind1[0][ret[1]], ind2[1][ret[2]])
    combinedij, taridx = torch.unique(combinedij,
                                      sorted=True,
                                      return_inverse=True)
    ij = torch.stack(decomposeind(combinedij))
    ret[0] = taridx

    sorted_idx = torch.argsort(ret[0])  # sort is optional
    return ij, ret[:, sorted_idx]


def spsphadamard_ind(tar_ij: LongTensor, ij: LongTensor) -> LongTensor:
    '''
    Auxiliary function for SparseTensor-SparseTensor hadamard product
    tar_ij the indice of A, ij the indice of b
    return:
      b2a of shape ij.shape[1]. ij[:, i] matches tar_ij[:, b2a[i]]. 
        if b2a<0, ij[:, i] is not matched 
    '''
    combine_tar_ij = combineind(tar_ij[0], tar_ij[1])
    assert torch.all(
        torch.diff(combine_tar_ij) > 0), "tar_ij should be sorted and coalesce"
    combine_ij = combineind(ij[0], ij[1])

    b2a = torch.clamp_min_(
        torch.searchsorted(combine_tar_ij, combine_ij, right=True) - 1, 0)
    notmatchmask = (combine_ij != combine_tar_ij[b2a])
    b2a[notmatchmask] = -1
    return b2a


def filterij(tar_ij: LongTensor, ij: LongTensor,
             bcd: LongTensor) -> LongTensor:
    '''
    A combination of hadamard and spspmm.
    A\odot(BC). BC's ind is ij and bcd, A's ind is tar_ij.
    return acd, (A\odot(BC)).val[a] = A.val[a] * scatter(B.val[c]*C.val[d],a)
    '''
    b2a = spsphadamard_ind(tar_ij, ij)
    a = b2a[bcd[0]]
    retmask = a >= 0
    acd = torch.stack((a[retmask], bcd[1][retmask], bcd[2][retmask]))
    return acd


def spsphadamard(A: SparseTensor,
                 B: SparseTensor,
                 b2a: Optional[LongTensor] = None) -> SparseTensor:
    '''
    SparseTensor \odot SparseTensor.
    b2a is the auxiliary indice produced by spsphadamard_ind.
    return is of A's indices, may have zero value.
    '''
    assert A.sparse_dim == 2
    assert B.sparse_dim == 2
    assert A.is_coalesced(), "A should be coalesced"
    assert B.is_coalesced(), "B should be coalesced"
    assert A.shape[:A.sparse_dim] == B.shape[:B.sparse_dim]
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
           bcd: Optional[LongTensor] = None,
           tar_ij: Optional[LongTensor] = None,
           acd: Optional[LongTensor] = None) -> SparseTensor:
    '''
    SparseTensor SparseTensor matrix multiplication at sparse dim.
    tar_ij mean tuples need output.
    '''
    assert A.sparse_dim == 2
    assert B.sparse_dim == 2
    assert A.is_coalesced(), "A should be coalesced"
    assert B.is_coalesced(), "B should be coalesced"
    if acd is not None:
        assert tar_ij is not None
        if A.values is None:
            mult = B.values[acd[2]]
        elif B.values is None:
            mult = A.values[acd[1]]
        else:
            mult = A.values[acd[1]] * B.values[acd[2]]
        retval = scatter(mult,
                         acd[0],
                         dim=0,
                         dim_size=tar_ij.shape[1],
                         reduce=aggr)
        return SparseTensor(tar_ij,
                            retval,
                            shape=[A.shape[0], B.shape[1]] +
                            list(retval.shape)[1:],
                            is_coalesced=True)
    else:
        warnings.warn("acd is not found")
        if bcd is None:
            ij, bcd = spspmm_ind(A.indices, B.indices)
        if tar_ij is not None:
            acd = filterij(tar_ij, ij, bcd)
            return spspmm(A, B, acd=acd, tar_ij=tar_ij)
        else:
            warnings.warn("tar_ij is not found")
            return spspmm(A, B, acd=bcd, tar_ij=ij)


if __name__ == "__main__":
    # for debug

    ptr = torch.tensor([0, 4, 4, 7, 8, 11, 11, 11, 16], dtype=torch.long)
    print("debug ptr2batch ", ptr2batch(ptr, dim_size=16))

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

    assert torch.all(
        ind1 == torch.stack(decomposeind(combineind(
            ind1[0], ind1[1])))), "combineind or decomposeind have bug"
    assert torch.all(
        ind2 == torch.stack(decomposeind(combineind(
            ind2[0], ind2[1])))), "combineind or decomposeind have bug"

    C = A @ B
    C = C.coalesce()

    ij, bcd = spspmm_ind(ind1, ind2)
    mult = val1[bcd[1]] * val2[bcd[2]]
    outval = scatter_add(mult, bcd[0], dim_size=ij.shape[1])
    out = torch.sparse_coo_tensor(ij, outval)
    out = out.coalesce()
    print("debug spspmm_ind",
          torch.max(torch.abs(C.indices() - out.indices())),
          torch.max(torch.abs(C.values() - out.values())))

    tar_ij = torch.stack(
        (torch.randint_like(ind1[0], n), torch.randint_like(ind1[0], l)))

    tar_ij = torch.stack(
        decomposeind(
            torch.unique(combineind(tar_ij[0], tar_ij[1]), sorted=True)))
    acd = filterij(tar_ij, ij, bcd)
    mult = val1[acd[1]] * val2[acd[2]]
    outval = scatter_add(mult, acd[0], dim_size=tar_ij.shape[1])
    maskedout = torch.sparse_coo_tensor(tar_ij, outval)
    maskedout = maskedout.coalesce()
    # debug spspmm with target filter
    print(
        "debug filter_ij",
        torch.max(
            torch.abs(maskedout.to_dense()[tar_ij[0], tar_ij[1]] -
                      C.to_dense()[tar_ij[0], tar_ij[1]])))

    Ap = torch.rand((n, m), device=device)
    Ap[torch.rand_like(Ap) > 0.9] = 0
    Ap = Ap.to_sparse_coo()
    Bp = torch.rand((n, m), device=device)
    Bp[torch.rand_like(Bp) > 0.9] = 0
    Bp = Bp.to_sparse_coo()
    spsphadamardout = spsphadamard(SparseTensor.from_torch_sparse_coo(Ap),
                                   SparseTensor.from_torch_sparse_coo(Bp))
    print(
        "debug spsp_hadamard ",
        torch.max((torch.multiply(Ap, Bp) -
                   spsphadamardout.to_torch_sparse_coo()).coalesce().values().abs()))
