import torch
from typing import List, Optional, Tuple, Callable
from torch import LongTensor, Tensor
from typing import Iterable, Union
import numpy as np
from .utils import torch_scatter_reduce
from typing import Final


def indicehash(indice: LongTensor) -> LongTensor:
    """
    Hashes a indice of shape (sparse_dim, nnz) to a single LongTensor of shape (nnz). Keep lexicographic order.

    Parameters:
    - indice (LongTensor): The input indices tensor of shape (sparse_dim, nnz).

    Returns:
    - LongTensor: A single LongTensor representing the hashed values.

    Raises:
    - AssertionError: If the input tensor doesn't have the expected shape or if the indices are too large or if there exists negative indice.

    Example:
    
    ::
    
        indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        hashed = indicehash(indices)

    """
    assert indice.ndim == 2
    assert torch.all(indice >= 0), "indice cannot be negative"
    sparse_dim = indice.shape[0]
    if sparse_dim == 1:
        return indice[0]
    interval = (63 // sparse_dim)
    assert torch.max(indice).item() < (
        1 << interval), "too large indice, hash is not injective"

    eihash = indice[sparse_dim - 1].clone()
    for i in range(1, sparse_dim):
        eihash.bitwise_or_(indice[sparse_dim - 1 - i].bitwise_left_shift(
            interval * i))
    return eihash


def decodehash(indhash: LongTensor, sparse_dim: int) -> LongTensor:
    """
    Decodes a hashed LongTensor into tuples of indices.

    This function takes a hashed LongTensor and decodes it into pairs of indices,
    which is commonly used in sparse tensor operations.

    Parameters:

    - indhash (LongTensor): The input hashed LongTensor of shape (nnz).
    - sparse_dim (int): The number of dimensions represented by the hash.

    Returns:

    - LongTensor: A LongTensor representing pairs of indices.

    Raises:

    - AssertionError: If the input tensor doesn't have the expected shape or
      if the sparse dimension is invalid.

    Example:

    ::

        indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        hashed = indicehash(indices)
        indices = decodehash(hashed)

    """
    if sparse_dim == 1:
        return indhash.unsqueeze(0)
    assert indhash.ndim == 1, "indhash should of shape (nnz) "
    interval = (63 // sparse_dim)
    mask = eval("0b" + "1" * interval)
    offset = (sparse_dim - 1 - torch.arange(
        sparse_dim, device=indhash.device)).unsqueeze(-1) * interval
    ret = torch.bitwise_right_shift(indhash.unsqueeze(0),
                                    offset)
    ret = ret.bitwise_and_(torch.tensor(mask, device=indhash.device))
    return ret


def indicehash_tight(indice: LongTensor, dimsize: LongTensor) -> LongTensor:
    """
    Hashes a 2D LongTensor of indices tightly into a single LongTensor.
    Equivalently, it compute the indice of flattened sparse tensor with indice and dimsize

    Parameters:
    - indice (LongTensor): The input indices tensor of shape (sparse_dim, nnz).
    - dimsize (LongTensor): The sizes of each dimension in the sparse tensor of shape (sparse_dim).

    Returns:
    - LongTensor: A single LongTensor representing the tightly hashed values.

    Raises:
    - AssertionError: If the input tensors don't have the expected shapes or if the indices exceed the dimension sizes.

    Example:
    
    ::

        indices = torch.tensor([[1, 2, 0], [4, 1, 2]], dtype=torch.long)
        dim_sizes = torch.tensor([3, 5], dtype=torch.long)
        hashed = indicehash_tight(indices, dim_sizes)

    """
    assert indice.ndim == 2, "indice shoule be of shape (sparse_dim, nnz) "
    assert dimsize.ndim == 1, "dim size should be of shape (sparse_dim)"
    assert dimsize.shape[0] == indice.shape[
        0], "indice dim and dim size not match"
    assert torch.all(indice.max(dim=1)[0] < dimsize), "indice exceeds dimsize"
    assert torch.prod(dimsize) < (
        1 << 62), "total size exceeds the range that torch.long can express"
    assert torch.all(indice >= 0), "indice cannot be negative"
    if indice.shape[0] == 1:
        return indice[0]
    step = torch.ones_like(dimsize)
    step[:-1] = torch.flip(torch.cumprod(torch.flip(dimsize[1:], (0, )), 0),
                           (0, ))
    return torch.sum(step.unsqueeze(-1) * indice, dim=0)


def decodehash_tight(indhash: LongTensor, dimsize: LongTensor) -> LongTensor:
    """
    Decodes a tightly hashed LongTensor into pairs of indices considering dimension sizes.

    Parameters:
    - indhash (LongTensor): The input hashed LongTensor of shape (nnz).
    - dimsize (LongTensor): The sizes of each dimension in the sparse tensor of shape (sparse_dim).

    Returns:
    - LongTensor: A LongTensor representing pairs of indices.

    Raises:
    - AssertionError: If the input tensors don't have the expected shapes or if the total size exceeds the range that torch.long can express.

    Example:

    ::

        indices = torch.tensor([[1, 2, 0], [4, 1, 2]], dtype=torch.long)
        dim_sizes = torch.tensor([3, 5], dtype=torch.long)
        hashed = indicehash_tight(indices, dim_sizes)
        indices = decodehash_tight(hashed, dim_sizes)

    """
    assert indhash.ndim == 1, "indhash should of shape (nnz) "
    assert torch.prod(dimsize) < (
        1 << 62), "total size exceeds the range that torch.long can express"
    if dimsize.shape[0] == 1:
        return indhash.unsqueeze(0)
    step = torch.ones_like(dimsize)
    step[:-1] = torch.flip(torch.cumprod(torch.flip(dimsize[1:], (0, )), 0),
                           (0, ))
    ret = indhash.reshape(1, -1) // step.reshape(-1, 1)
    ret[1:] -= ret[:-1] * dimsize[1:].reshape(-1, 1)
    return ret


def coalesce(edge_index: LongTensor,
             edge_attr: Optional[Tensor] = None,
             reduce: str = 'sum') -> Tuple[Tensor, Optional[Tensor]]:
    """
    Coalesces and reduces duplicate entries in edge indices and attributes.
    
    Args:

    - edge_index (LongTensor): The edge indices.
    - edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-dimensional
      edge features. If given as a list, it will be reshuffled and duplicates will be
      removed for all entries. (default: None)
    - reduce (str, optional): The reduction operation to use for merging edge features.
      Options include 'sum', 'mean', 'min', 'max', 'mul'. (default: 'sum')

    Returns:

    - Tuple[Tensor, Optional[Tensor]]: A tuple containing the coalesced edge indices
      and the coalesced and reduced edge attributes (if provided). If edge_attr is
      None, the second element will be None.
    """
    sparsedim = edge_index.shape[0]
    eihash = indicehash(edge_index)
    eihash, idx = torch.unique(eihash, return_inverse=True)
    edge_index = decodehash(eihash, sparsedim)
    if edge_attr is None:
        return edge_index, None
    else:
        edge_attr = torch_scatter_reduce(0, edge_attr, idx, eihash.shape[0],
                                         reduce)
        return edge_index, edge_attr


class SparseTensor:
    """
    Represents a sparse tensor in coo format.

    This class allows you to work with sparse tensors represented by indices and
    values. It provides various operations such as sum, max, mean, unpooling,
    diagonal extraction, and more.

    Parameters:
    - indices (LongTensor): The indices of the sparse tensor, of shape (#sparsedim, #nnz).
    - values (Optional[Tensor]): The values associated with the indices, of shape (#nnz,\*denseshapeshape). Should have the same number of nnz as indices. Defaults to None.
    - shape (Optional[List[int]]): The shape of the sparse tensor. If None, it is computed from the indices and values. Defaults to None.
    - is_coalesced (bool): Indicates whether the indices and values are coalesced. Defaults to False.

    Methods:
    - is_coalesced(self): Check if the tensor is coalesced.
    - to(self, device: torch.DeviceObjType, non_blocking: bool = False): Move the tensor to the specified device.
    - diag(self, dims: Optional[Iterable[int]], return_sparse: bool = False): Extract diagonal elements from the tensor. The dimensions in dims will be take diagonal and put at dims[0]
    - sum(self, dims: Union[int, Optional[Iterable[int]]], return_sparse: bool = False): Compute the sum of tensor values along specified dimensions. return_sparse=True will return a sparse tensor, otherwise return a dense tensor.
    - max(self, dims: Union[int, Optional[Iterable[int]]], return_sparse: bool = False): Compute the maximum of tensor values along specified dimensions. return_sparse=True will return a sparse tensor, otherwise return a dense tensor.
    - mean(self, dims: Union[int, Optional[Iterable[int]]], return_sparse: bool = False): Compute the mean of tensor values along specified dimensions. return_sparse=True will return a sparse tensor, otherwise return a dense tensor.
    - unpooling(self, dims: Union[int, Iterable[int]], tarX): Perform unpooling operation along specified dimensions.
    - tuplewiseapply(self, func: Callable[[Tensor], Tensor]): Apply a function to each element of the tensor.
    - diagonalapply(self, func: Callable[[Tensor, LongTensor], Tensor]): Apply a function to diagonal elements of the tensor.
    - add(self, tarX, samesparse: bool): Add two sparse tensors together. samesparse=True means that two sparse tensor have the indice and can add values directly. 
    - catvalue(self, tarX, samesparse: bool): Concatenate values of two sparse tensors. samesparse=True means that two sparse tensor have the indice and can cat values along the first dimension directly. 
    - from_torch_sparse_coo(cls, A: torch.Tensor): Create a SparseTensor from a torch sparse COO tensor.
    - to_torch_sparse_coo(self) -> Tensor: Convert the SparseTensor to a torch sparse COO tensor.

    Attributes:
    - indices (LongTensor): The indices of the sparse tensor.
    - values (Tensor): The values associated with the indices.
    - sparse_dim (int): The number of dimensions represented by the indices.
    - nnz (int): The number of non-zero values.
    - shape (torch.Size): The shape of the tensor.
    - sparseshape (torch.Size): The shape of the tensor up to the sparse dimensions.
    - denseshape (torch.Size): The shape of the tensor after the sparse dimensions.

    """

    def __init__(self,
                 indices: LongTensor,
                 values: Optional[Tensor] = None,
                 shape: Optional[List[int]] = None,
                 is_coalesced: bool = False,
                 reduce: str = "sum"):
        assert indices.ndim == 2, "indice should of shape (#sparsedim, #nnz)"
        if values is not None:
            assert indices.shape[1] == values.shape[
                0], "indices and values should have the same number of nnz"
        self.__sparse_dim = indices.shape[0]
        if shape is not None:
            self.__shape = tuple(shape)
            # print(self.shape, self.denseshape, self.sparseshape, values.shape)
            if values is not None:
                assert self.denseshape == values.shape[
                    1:], "shape, value not match"
        else:
            self.__shape = tuple(
                list(map(lambda x: x + 1,
                         torch.max(indices, dim=1).tolist())) +
                list(values.shape[1:]))
        if is_coalesced:
            self.__indices, self.__values = indices, values
        else:
            self.__indices, self.__values = coalesce(indices, values, reduce)
        self.__nnz = self.indices.shape[1]

    def is_coalesced(self):
        return True

    def to(self, device: torch.DeviceObjType, non_blocking: bool = False):
        self.__indices = self.__indices.to(device, non_blocking=non_blocking)
        self.__values = self.__values.to(device, non_blocking=non_blocking)
        return self

    @property
    def indices(self):
        return self.__indices

    @property
    def values(self):
        return self.__values

    @property
    def sparse_dim(self):
        return self.__sparse_dim

    @property
    def nnz(self):
        return self.__nnz

    @property
    def shape(self):
        return self.__shape

    @property
    def sparseshape(self):
        return self.shape[:self.sparse_dim]

    @property
    def denseshape(self):
        return self.shape[self.sparse_dim:]

    def _diag_to_sparse(self, dims: List[int]):
        assert np.all(
            np.array(dims) < self.__sparse_dim
        ), "please use tuplewiseapply for operation on dense dims"
        assert np.all(np.array(dims) >= 0), "do not support negative dims"
        '''
        diag dims is then put at the first dims in dims list.
        '''
        mask = torch.all((self.indices[dims] - self.indices[[dims[0]]]) == 0, dim=0)
        idx = [i for i in range(self.sparse_dim) if i not in dims[1:]]
        other_shape = tuple([self.shape[i] for i in idx]) + self.denseshape
        return SparseTensor(indices=self.indices[idx][:, mask],
                            values=self.values[mask],
                            shape=other_shape,
                            is_coalesced=(idx[0] == 0)
                            and np.all(np.diff(idx) == 1))

    def _diag_to_dense(self, dims: List[int]) -> Tensor:
        '''
        diag dims is then put at the first dims in dims list.
        '''
        if len(dims) == self.sparse_dim:
            diag_idx = torch.arange(self.shape[dims[0]], device=self.indices.device)
            diag_hash = indicehash(diag_idx.reshape(1, -1).expand(len(dims), -1))
            selfhash = indicehash(self.indices[dims])
            matchidx = torch.searchsorted(selfhash, diag_hash, right=True) - 1
            notmatchmask = matchidx < 0
            matchidx.clamp_min_(0)
            ret = self.values[matchidx]
            ret[notmatchmask] = 0
            return ret
        else:
            idx = [i for i in range(self.sparse_dim) if i not in dims[1:]]
            nsparse_shape = [self.shape[i] for i in idx]
            
            diag_idx = torch.arange(self.shape[dims[0]], device=self.indices.device)
            diag_hash = indicehash(diag_idx.reshape(1, -1).expand(len(dims), -1))
            selfhash = indicehash(self.indices[dims])
            matchidx = torch.searchsorted(selfhash, diag_hash, right=True) - 1
            notmatchmask = matchidx < 0
            matchidx.clamp_min_(0)
            ret = torch.zeros(nsparse_shape + self.denseshape,
                            device=self.indices.device,
                            dtype=self.values.dtype)
            value_to_fill = self.values[matchidx]
            value_to_fill[notmatchmask] = 0
            ret.index_put_(tuple(self.indices[_][matchidx] if _ != dims[0] else diag_idx for _ in idx),  value_to_fill)
            return ret

    def diag(self, dims: Optional[Iterable[int]], return_sparse: bool = False):
        '''
        TODO: unit test ??
        '''
        if isinstance(dims, int):
            raise NotImplementedError
        if dims == None:
            dims = list(range(self.sparse_dim))
        dims = sorted(list(set(dims)))
        if return_sparse:
            return self._diag_to_sparse(dims)
        else:
            return self._diag_to_dense(dims)

    def _reduce_to_sparse(self, dims: Iterable[int], reduce: str):
        assert np.all(
            np.array(dims) < self.__sparse_dim
        ), "please use tuplewiseapply for operation on dense dims"
        assert np.all(np.array(dims) >= 0), "do not support negative dims"
        idx = [i for i in range(self.sparse_dim) if i not in list(dims)]
        other_ind = self.indices[idx]
        other_shape = tuple([self.shape[i] for i in idx]) + self.denseshape
        return SparseTensor(indices=other_ind,
                            values=self.values,
                            shape=other_shape,
                            is_coalesced=False,
                            reduce=reduce)

    def _reduce_to_dense(self, dims: Iterable[int], reduce: str) -> Tensor:
        assert np.all(
            np.array(dims) < self.__sparse_dim
        ), "please use tuplewiseapply for operation on dense dims"
        assert np.all(np.array(dims) >= 0), "do not support negative dims"
        idx = [i for i in range(self.sparse_dim) if i not in list(dims)]
        if len(idx) == 1:
            idx = idx[0]
            other_ind = self.indices[idx]
            nsparse_size = self.shape[idx]
            ret = torch_scatter_reduce(0, self.values, other_ind, nsparse_size,
                                       reduce)
            return ret
        else:
            other_ind = self.indices[idx]
            other_shape = tuple(self.shape[i] for i in idx)
            nsparse_shape = other_shape
            nsparse_size = 1
            for _ in nsparse_shape:
                nsparse_size *= _

            thash = indicehash_tight(
                other_ind,
                torch.LongTensor(nsparse_shape).to(other_ind.device))
            ret = torch_scatter_reduce(0, self.values, thash, nsparse_size,
                                       reduce)
            ret = ret.reshape(nsparse_shape + tuple(ret.shape[1:]))
            return ret

    def sum(self,
            dims: Union[int, Optional[Iterable[int]]],
            return_sparse: bool = False):
        if isinstance(dims, int):
            dims = [dims]
        if dims == None:
            return torch.sum(self.values, dims=0)
        elif return_sparse:
            return self._reduce_to_sparse(dims, "sum")
        else:
            return self._reduce_to_dense(dims, "sum")

    def max(self,
            dims: Union[int, Optional[Iterable[int]]],
            return_sparse: bool = False):
        if isinstance(dims, int):
            dims = [dims]
        if dims == None:
            return torch.max(self.values, dims=0)
        elif return_sparse:
            return self._reduce_to_sparse(dims, "max")
        else:
            return self._reduce_to_dense(dims, "max")

    def mean(self,
             dims: Union[int, Optional[Iterable[int]]],
             return_sparse: bool = False):
        if isinstance(dims, int):
            dims = [dims]
        if dims == None:
            return torch.mean(self.values, dims=0)
        elif return_sparse:
            return self._reduce_to_sparse(dims, "mean")
        else:
            return self._reduce_to_dense(dims, "mean")

    def unpooling(self, dims: Union[int, Iterable[int]], tarX):
        '''
        unpooling to of tarX indice
        dims: of tarX
        '''
        if isinstance(dims, int):
            dims = [dims]
        self_hash = indicehash(self.indices)
        assert torch.all(torch.diff(self_hash)), "self is not coalesced"
        tarX: SparseTensor = tarX
        taridx = [i for i in range(tarX.sparse_dim) if i not in list(dims)]
        tar_hash = indicehash(tarX.indices[taridx])

        b2a = torch.clamp_min_(
            torch.searchsorted(self_hash, tar_hash, right=True) - 1, 0)

        matchmask = (self_hash[b2a] == tar_hash)
        ret = torch.zeros((tar_hash.shape[0], ) + self.denseshape,
                          dtype=self.values.dtype,
                          device=self.values.device)
        ret[matchmask] = self.values[b2a[matchmask]]
        return tarX.setvalue(ret)

    def unpooling_fromdense1dim(self, dims: int, X: Tensor):
        '''
        unpooling to of self shape. Note the dims is for self to maintain, and expand other dims
        '''
        assert dims < self.sparse_dim, "only unpooling sparse dims"
        assert X.shape[0] == self.shape[dims], "shape not match"
        return self.setvalue(X[self.indices[dims]])

    @classmethod
    def from_torch_sparse_coo(cls, A: torch.Tensor):
        assert A.is_sparse, "from_torch_sparse_coo converts a torch.sparse_coo_tensor to SparseTensor"
        ret = cls(A._indices(), A._values(), A.shape, A.is_coalesced())
        return ret

    def to_torch_sparse_coo(self) -> Tensor:
        ret = torch.sparse_coo_tensor(self.indices,
                                      self.values,
                                      size=self.shape)
        ret = ret._coalesced_(self.is_coalesced())
        return ret

    def tuplewiseapply(self, func: Callable[[Tensor], Tensor]):
        nvalues = func(self.values)
        return self.setvalue(nvalues)
    
    def setvalue(self, val: Tensor):
        nvalues = val
        return SparseTensor(self.indices,
                            nvalues,
                            self.sparseshape + tuple(nvalues.shape[1:]),
                            is_coalesced=True)

    def diagonalapply(self, func_d: Callable[[Tensor], Tensor], func_nd: Callable[[Tensor], Tensor]):
        assert self.sparse_dim == 2, "only implemented for 2D"
        val_d = func_d(self.values)
        val_nd = func_nd(self.values)

        val_d, val_nd = torch.movedim(val_d, 0, -1), torch.movedim(val_nd, 0, -1)
        nvalues = torch.where((self.indices[0] == self.indices[1]), val_d, val_nd)
        nvalues = torch.movedim(nvalues, -1, 0)
        return self.setvalue(nvalues)

    def add(self, tarX, samesparse: bool):
        if not samesparse:
            return SparseTensor(
                torch.concat((self.indices, tarX.indices), dim=1),
                torch.concat((self.values, tarX.values), dim=0), self.shape,
                False)
        else:
            return self.tuplewiseapply(lambda x: x + tarX.values)

    def is_indicesymmetric(self, dim1: int, dim2: int) -> bool:
        dimension_ind = list(range(self.sparse_dim))
        dimension_ind[dim1] = dim2
        dimension_ind[dim2] = dim1
        newind = self.indices[dimension_ind]
        newindhash = indicehash(newind)
        newindhash = torch.sort(newindhash).values
        return torch.equal(newindhash, indicehash(self.indices))

    def transpose(self, dim1: int, dim2: int):
        assert self.is_indicesymmetric(dim1, dim2), "only support summetric indice now"
        dimension_ind = list(range(self.sparse_dim))
        dimension_ind[dim1] = dim2
        dimension_ind[dim2] = dim1
        newind = self.indices[dimension_ind]
        newindhash = indicehash(newind)
        retind = torch.argsort(newindhash)
        return self.tuplewiseapply(lambda x: x[retind])

    def catvalue(self, tarXs: Iterable, samesparse: bool):
        if isinstance(tarXs, SparseTensor):
            tarXs = [tarXs]
        assert samesparse == True, "must have the same sparcity to concat value"
        nvalues = torch.concat([self.values] + [_.values for _ in tarXs], dim=-1)
        return self.setvalue(nvalues)
    
    def __repr__(self):
        return f'SparseTensor(shape={self.shape}, sparse_dim={self.sparse_dim}, nnz={self.nnz})'
