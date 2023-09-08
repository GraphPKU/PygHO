import torch
from torch import LongTensor, Tensor
from typing import Optional, Callable
from torch_scatter import scatter
from .SpTensor import SparseTensor, indicehash, decodehash
import warnings


def ptr2batch(ptr: LongTensor, dim_size: int) -> LongTensor:
    '''
    ptr: LongTensor, ptr[0]=0, torch.all(diff(ptr)>=0) is true
    output: (dim_size) LongTensor batch, batch[ptr[i]:ptr[i+1]]=i

    TODO: use torch_scatter gather instead?
    '''
    assert ptr.ndim == 1, "ptr should be 1-d"
    assert ptr[0] == 0 and torch.all(
        torch.diff(ptr) >= 0), "should put in a ptr tensor"
    assert ptr[-1] == dim_size, "dim_size should match ptr"
    tmp = torch.arange(dim_size, device=ptr.device, dtype=ptr.dtype)
    ret = torch.searchsorted(ptr, tmp, right=True) - 1
    return ret


def spspmm_ind(ind1: LongTensor,
               dim1: int,
               ind2: LongTensor,
               dim2: int,
               is_k2_sorted: bool = False) -> LongTensor:
    '''
    TODO: unit test
    ind1, ind2 are indices of two sparse tensors.
    ind1 (sparsedim1, M1)
    ind2 (sparsedim2, M2)
    eliminate dim1 in Tensor1 and dim2 in Tensor2, and cat dims left.
    (\vec i_1, k, \vec i_2), (\vec j_1, k, \vec j_2) -> (\vec i_1, \vec i_2, \vec j_1, \vec j_2) (b, c, d), c represent index of \vec i, d represent index of \vec j, b represent index of output
    '''
    assert 0 <= dim1 < ind1.shape[
        0], f"ind1's reduced dim {dim1} is out of range"
    assert 0 <= dim2 < ind2.shape[
        0], f"ind2's reduced dim {dim2} is out of range"
    if dim2 != 0 and not (is_k2_sorted):
        perm = torch.argsort(ind2[dim2])
        tarind, bcd = spspmm_ind(ind1, dim1, ind2[:, perm], dim2, True)
        bcd[2] = perm[bcd[2]]
        return tarind, bcd
    else:
        nnz1, nnz2, sparsedim1, sparsedim2 = ind1.shape[1], ind2.shape[
            1], ind1.shape[0], ind2.shape[0]
        k1, k2 = ind1[dim1], ind2[dim2]
        assert torch.all(torch.diff(k2) >= 0), "ind2[0] should be sorted"
        # for each k in k1, it can match a interval of k2 as k2 is sorted
        upperbound = torch.searchsorted(k2, k1, right=True)
        lowerbound = torch.searchsorted(k2, k1, right=False)
        matched_num = torch.clamp_min_(upperbound - lowerbound, 0)

        # ptr[i] provide the offset to place pair of ind1[:, i] and the matched ind2
        retptr = torch.zeros((nnz1 + 1),
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

        # compute the ind pair index

        combinedind = indicehash(
            torch.concat(
                ((torch.concat((ind1[:dim1], ind1[dim1 + 1:])))[:, ret[1]],
                 torch.concat((ind2[:dim2], ind2[dim2 + 1:]))[:, ret[2]])))
        combinedind, taridx = torch.unique(combinedind,
                                           sorted=True,
                                           return_inverse=True)
        tarind = decodehash(combinedind, sparsedim1 + sparsedim2 - 2)
        ret[0] = taridx

        sorted_idx = torch.argsort(ret[0])  # sort is optional
        return tarind, ret[:, sorted_idx]


def spsphadamard_ind(tar_ind: LongTensor, ind: LongTensor) -> LongTensor:
    '''
    Auxiliary function for SparseTensor-SparseTensor hadamard product
    tar_ind the indice of A, ind the indice of b
    return:
      b2a of shape ind.shape[1]. ind[:, i] matches tar_ind[:, b2a[i]]. 
        if b2a<0, ind[:, i] is not matched 
    '''
    assert tar_ind.shape[0] == ind.shape[0]
    combine_tar_ind = indicehash(tar_ind)
    assert torch.all(torch.diff(combine_tar_ind) >
                     0), "tar_ind should be sorted and coalesce"
    combine_ind = indicehash(ind)

    b2a = torch.clamp_min_(
        torch.searchsorted(combine_tar_ind, combine_ind, right=True) - 1, 0)
    notmatchmask = (combine_ind != combine_tar_ind[b2a])
    b2a[notmatchmask] = -1
    return b2a


def filterind(tar_ind: LongTensor, ind: LongTensor,
              bcd: LongTensor) -> LongTensor:
    '''
    A combination of hadamard and spspmm.
    A\odot(BC). BC's ind is ind and bcd, A's ind is tar_ind.
    return acd, (A\odot(BC)).val[a] = A.val[a] * scatter(B.val[c]*C.val[d],a)
    '''
    b2a = spsphadamard_ind(tar_ind, ind)
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
    Dense shapes of A, B must be broadcastable. 
    return is of A's indices, may have zero value.
    '''
    assert A.is_coalesced(), "A should be coalesced"
    assert B.is_coalesced(), "B should be coalesced"
    assert A.sparseshape == B.sparseshape, "A, B should be of the same sparse shape"
    ind1, val1 = A.indices, A.values
    ind2, val2 = B.indices, B.values
    if b2a is None:
        b2a = spsphadamard_ind(ind1, ind2)
    mask = (b2a >= 0)
    if val1 is None:
        retval = val2[mask]
    elif val2 is None:
        retval = val1[b2a[mask]]
    else:
        retval = val1[b2a[mask]] * val2[mask]
    retind = ind2[:, mask]
    return SparseTensor(retind,
                        retval,
                        shape=A.sparseshape + retval.shape[1:],
                        is_coalesced=True)


def spspmm(A: SparseTensor,
           dim1: int,
           B: SparseTensor,
           dim2: int,
           aggr: str = "sum",
           bcd: Optional[LongTensor] = None,
           tar_ind: Optional[LongTensor] = None,
           acd: Optional[LongTensor] = None) -> SparseTensor:
    '''
    SparseTensor SparseTensor matrix multiplication at sparse dim.
    tar_ind mean tuples need output.
    Dense shapes of A, B must be broadcastable. 
    '''
    assert A.is_coalesced(), "A should be coalesced"
    assert B.is_coalesced(), "B should be coalesced"
    if acd is not None:
        assert tar_ind is not None
        if A.values is None:
            mult = B.values[acd[2]]
        elif B.values is None:
            mult = A.values[acd[1]]
        else:
            mult = A.values[acd[1]] * B.values[acd[2]]
        retval = scatter(mult,
                         acd[0],
                         dim=0,
                         dim_size=tar_ind.shape[1],
                         reduce=aggr)
        return SparseTensor(tar_ind,
                            retval,
                            shape=A.sparseshape[:dim1] +
                            A.sparseshape[dim1 + 1:] + B.sparseshape[:dim2] +
                            B.sparseshape[dim2 + 1:] + retval.shape[1:],
                            is_coalesced=True)
    else:
        warnings.warn("acd is not found")
        if bcd is None:
            ind, bcd = spspmm_ind(A.indices, dim1, B.indices, dim2)
        if tar_ind is not None:
            acd = filterind(tar_ind, ind, bcd)
            return spspmm(A, dim1, B, dim2, aggr, acd=acd, tar_ind=tar_ind)
        else:
            warnings.warn("tar_ind is not found")
            return spspmm(A, dim1, B, dim2, aggr, acd=bcd, tar_ind=ind)


def spspmpnn(A: SparseTensor,
             dim1: int,
             B: SparseTensor,
             dim2: int,
             C: SparseTensor,
             acd: LongTensor,
             message_func: Callable[[Tensor, Tensor, Tensor, LongTensor],
                                    Tensor],
             aggr: str = "sum") -> SparseTensor:
    '''
    SparseTensor SparseTensor matrix multiplication at sparse dim.
    tar_ind mean tuples need output.
    Dense shapes of A, B must be broadcastable. 
    '''
    mult = message_func(None if A.values is None else A.values[acd[1]],
                        None if B.values is None else B.values[acd[2]],
                        None if C.values is None else C.values[acd[0]], acd[0])
    tar_ind = C.indices
    retval = scatter(mult,
                     acd[0],
                     dim=0,
                     dim_size=tar_ind.shape[1],
                     reduce=aggr)
    return SparseTensor(tar_ind,
                        retval,
                        shape=A.sparseshape[:dim1] + A.sparseshape[dim1 + 1:] +
                        B.sparseshape[:dim2] + B.sparseshape[dim2 + 1:] +
                        retval.shape[1:],
                        is_coalesced=True)