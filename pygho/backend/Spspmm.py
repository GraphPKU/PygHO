import torch
from torch import LongTensor, Tensor
from typing import Optional, Callable, Tuple
from .SpTensor import SparseTensor, indicehash, decodehash
import warnings
from .utils import torch_scatter_reduce

def ptr2batch(ptr: LongTensor, dim_size: int) -> LongTensor:
    """
    Converts a pointer tensor to a batch tensor. TODO: use torch_scatter gather instead?

    This function takes a pointer tensor `ptr` and a `dim_size` and converts it to a
    batch tensor where each element in the batch tensor corresponds to a range of
    indices in the original tensor.

    Args:

    - ptr (LongTensor): The pointer tensor, where `ptr[0] = 0` and `torch.all(diff(ptr) >= 0)` is true.
    - dim_size (int): The size of the target dimension.

    Returns:

    - LongTensor: A batch tensor of shape `(dim_size,)` where `batch[ptr[i]:ptr[i+1]] = i`.
    """
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
               is_k2_sorted: bool = False) -> Tuple[LongTensor, LongTensor]:   
    """
    Sparse-sparse matrix multiplication for indices.

    This function performs a sparse-sparse matrix multiplication for indices. 
    Given two sets of indices `ind1` and `ind2`, this function eliminates `dim1` in `ind1` and `dim2` in `ind2`, and concatenates the remaining dimensions. 
    
    The result represents the product of the input indices.

    Args:

    - ind1 (LongTensor): The indices of the first sparse tensor of shape `(sparsedim1, M1)`.
    - dim1 (int): The dimension to eliminate in `ind1`.
    - ind2 (LongTensor): The indices of the second sparse tensor of shape `(sparsedim2, M2)`.
    - dim2 (int): The dimension to eliminate in `ind2`.
    - is_k2_sorted (bool, optional): Whether `ind2` is sorted along `dim2`. Defaults to `False`.

    Returns:
    
    - tarind: LongTensor: The resulting indices after performing the sparse-sparse matrix    multiplication.
    - bcd: LongTensor: In tensor perspective (\*i_1, k, \*i_2), (\*j_1, k, \*j_2) -> (\*i_1, \*i_2, \*j_1, \*j_2).
      The return indice is of shape (3, nnz), (b, c, d), c represent index of \*i, d represent index of \*j, b represent index of output.For i=1,2,...,nnz,  val1[c[i]] * val2[d[i]] will be add to output val's b[i]-th element.

    Example:
    
    ::

        ind1 = torch.tensor([[0, 1, 1, 2],
                            [2, 1, 0, 2]], dtype=torch.long)
        dim1 = 0
        ind2 = torch.tensor([[2, 1, 0, 1],
                            [1, 0, 2, 2]], dtype=torch.long)
        dim2 = 1
        result = spspmm_ind(ind1, dim1, ind2, dim2)

    """
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
    """
    Auxiliary function for SparseTensor-SparseTensor Hadamard product.

    This function is an auxiliary function used in the Hadamard product of two sparse tensors. Given the indices `tar_ind` of sparse tensor A and the indices `ind` of sparse tensor B, this function returns an index array `b2a` of shape `(ind.shape[1],)` such that `ind[:, i]` matches `tar_ind[:, b2a[i]]` for each `i`. If `b2a[i]` is less than 0, it means `ind[:, i]` is not matched.

    Args:

    - tar_ind (LongTensor): The indices of sparse tensor A.
    - ind (LongTensor): The indices of sparse tensor B.

    Returns:

    - LongTensor: An index array `b2a` representing the matching indices between `tar_ind` and `ind`.
      b2a of shape ind.shape[1]. ind[:, i] matches tar_ind[:, b2a[i]]. if b2a[i]<0, ind[:, i] is not matched 
    
    Example:

    ::

        tar_ind = torch.tensor([[0, 1, 1, 2],
                                [2, 1, 0, 2]], dtype=torch.long)
        ind = torch.tensor([[2, 1, 0, 1],
                            [1, 0, 2, 2]], dtype=torch.long)
        b2a = spsphadamard_ind(tar_ind, ind)

    """
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
    """
    A combination of Hadamard and Sparse Matrix Multiplication.

    Given the indices `tar_ind` of sparse tensor A, the indices `ind` of sparse tensor BC, and the index array `bcd`, this function returns an index array `acd`, where `(A ⊙ (BC)).val[a] = A.val[a] * scatter(B.val[c] * C.val[d], a)`.

    Args:

    - tar_ind (LongTensor): The indices of sparse tensor A.
    - ind (LongTensor): The indices of sparse tensor BC.
    - bcd (LongTensor): An index array representing `(BC).val`.

    Returns:
    
    - LongTensor: An index array `acd` representing the filtered indices.

    Example:
    
    ::

        tar_ind = torch.tensor([[0, 1, 1, 2],
                                [2, 1, 0, 2]], dtype=torch.long)
        ind = torch.tensor([[2, 1, 0, 1],
                            [1, 0, 2, 2]], dtype=torch.long)
        bcd = torch.tensor([[3, 2, 1, 0],
                            [6, 5, 4, 3],
                            [9, 8, 7, 6]], dtype=torch.long)
        acd = filterind(tar_ind, ind, bcd)


    """
    b2a = spsphadamard_ind(tar_ind, ind)
    a = b2a[bcd[0]]
    retmask = a >= 0
    acd = torch.stack((a[retmask], bcd[1][retmask], bcd[2][retmask]))
    return acd


def spsphadamard(A: SparseTensor,
                 B: SparseTensor,
                 b2a: Optional[LongTensor] = None) -> SparseTensor:
    """
    Element-wise Hadamard product between two SparseTensors.

    This function performs the element-wise Hadamard product between two SparseTensors, `A` and `B`. The `b2a` parameter is an optional auxiliary index produced by the `spsphadamard_ind` function.

    Args:

    - A (SparseTensor): The first SparseTensor.
    - B (SparseTensor): The second SparseTensor.
    - b2a (LongTensor, optional): An optional index array produced by `spsphadamard_ind`. If not provided, it will be computed.

    Returns:

    - SparseTensor: A SparseTensor containing the result of the Hadamard product.


    Notes:

    - Both `A` and `B` must be coalesced SparseTensors.
    - The dense shapes of `A` and `B` must be broadcastable.
    """
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
    """
    SparseTensor SparseTensor matrix multiplication at a specified sparse dimension.

    This function performs matrix multiplication between two SparseTensors, `A` and `B`, at the specified sparse dimensions `dim1` and `dim2`. The result is a SparseTensor containing the result of the multiplication. The `aggr` parameter specifies the reduction operation used for merging the resulting values.

    Args:

    - A (SparseTensor): The first SparseTensor.
    - dim1 (int): The dimension along which `A` is multiplied.
    - B (SparseTensor): The second SparseTensor.
    - dim2 (int): The dimension along which `B` is multiplied.
    - aggr (str, optional): The reduction operation to use for merging edge features ("sum", "min", "max", "mean"). Defaults to "sum".
    - bcd (LongTensor, optional): An optional auxiliary index array produced by spspmm_ind.
    - tar_ind (LongTensor, optional): An optional target index array for the output. If not provided, it will be computed.
    - acd (LongTensor, optional): An optional auxiliary index array produced by filterind.

    Returns:

    - SparseTensor: A SparseTensor containing the result of the matrix multiplication.

    Notes:

    - Both `A` and `B` must be coalesced SparseTensors.
    - The dense shapes of `A` and `B` must be broadcastable.
    - This function allows for optional indices `bcd` and `tar_ind` for improved performance and control.

    """
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
        retval = torch_scatter_reduce(0, mult, acd[0], tar_ind.shape[1], aggr)
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
    """
    SparseTensor SparseTensor matrix multiplication at a specified sparse dimension using a message function.

    This function extend matrix multiplication between two SparseTensors, `A` and `B`, at the specified sparse dimensions `dim1` and `dim2`, while using a message function `message_func` to compute the messages sent from `A` to `B` and `C`. The result is a SparseTensor containing the result of the multiplication. The `aggr` parameter specifies the reduction operation used for merging the resulting values.

    Args:

    - A (SparseTensor): The first SparseTensor.
    - dim1 (int): The dimension along which `A` is multiplied.
    - B (SparseTensor): The second SparseTensor.
    - dim2 (int): The dimension along which `B` is multiplied.
    - C (SparseTensor): The third SparseTensor.
    - acd (LongTensor): The auxiliary index array produced by a previous operation.
    - message_func (Callable): A callable function that computes the messages between `A`, `B`, and `C`.
    - aggr (str, optional): The reduction operation to use for merging edge features ("sum", "min", "max", "mul", "any"). Defaults to "sum".

    Returns:

    - SparseTensor: A SparseTensor containing the result of the matrix multiplication.

    Notes:
    
    - Both `A` and `B` must be coalesced SparseTensors.
    - The dense shapes of `A`, `B`, and `C` must be broadcastable.
    - The `message_func` should take four arguments: `A_values`, `B_values`, `C_values`, and `acd`, and return messages based on custom logic.

    """
    mult = message_func(None if A.values is None else A.values[acd[1]],
                        None if B.values is None else B.values[acd[2]],
                        None if C.values is None else C.values[acd[0]], acd[0])
    tar_ind = C.indices
    retval = torch_scatter_reduce(0, mult, acd[0], tar_ind.shape[1], aggr)
    return SparseTensor(tar_ind,
                        retval,
                        shape=A.sparseshape[:dim1] + A.sparseshape[dim1 + 1:] +
                        B.sparseshape[:dim2] + B.sparseshape[dim2 + 1:] +
                        retval.shape[1:],
                        is_coalesced=True)