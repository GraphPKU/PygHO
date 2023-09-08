import torch
from typing import List, Optional, Tuple, Callable
from torch import LongTensor, Tensor
from torch_scatter import scatter
from typing import Iterable, Union
import numpy as np


def indicehash(indice: LongTensor) -> LongTensor:
    assert indice.ndim == 2
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
    '''
    transfer hash into pairs
    '''
    if sparse_dim == 1:
        return indhash.unsqueeze(0)
    assert indhash.ndim == 1, "indhash should of shape (nnz) "
    interval = (63 // sparse_dim)
    mask = eval("0b" + "1" * interval)
    offset = (sparse_dim - 1 - torch.arange(
        sparse_dim, device=indhash.device)).unsqueeze(-1) * interval
    ret = torch.bitwise_right_shift(indhash.unsqueeze(0),
                                    offset).bitwise_and_(mask)
    return ret


def indicehash_tight(indice: LongTensor, dimsize: LongTensor) -> LongTensor:
    assert indice.ndim == 2, "indice shoule be of shape (sparse_dim, nnz) "
    assert dimsize.ndim == 1, "dim size should be of shape (sparse_dim)"
    assert dimsize.shape[0] == indice.shape[
        0], "indice dim and dim size not match"
    assert torch.all(indice.max(dim=1)[0] < dimsize), "indice exceeds dimsize"
    assert torch.prod(dimsize) < (
        1 << 62), "total size exceeds the range that torch.long can express"
    if indice.shape[0] == 1:
        return indice[0]
    step = torch.ones_like(dimsize)
    step[:-1] = torch.flip(torch.cumprod(torch.flip(dimsize[1:], (0, )), 0),
                           (0, ))
    return torch.sum(step.unsqueeze(-1) * indice, dim=0)


def decodehash_tight(indhash: LongTensor, dimsize: LongTensor) -> LongTensor:
    '''
    transfer hash into pair
    '''
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
             num_nodes: Optional[int] = None,
             reduce: str = 'add') -> Tuple[Tensor, Optional[Tensor]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"add"`)
    """
    sparsedim = edge_index.shape[0]
    eihash = indicehash(edge_index)
    eihash, idx = torch.unique(eihash, return_inverse=True)
    edge_index = decodehash(eihash, sparsedim)
    if edge_attr is None:
        return edge_index, None
    else:
        edge_attr = scatter(edge_attr,
                            idx,
                            dim=0,
                            dim_size=eihash.shape[0],
                            reduce=reduce)
        return edge_index, edge_attr


class SparseTensor:

    def __init__(self,
                 indices: LongTensor,
                 values: Optional[Tensor] = None,
                 shape: Optional[List[int]] = None,
                 is_coalesced: bool = False):
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
        self.__maxsparsesize = max(self.shape[:self.sparse_dim])
        if is_coalesced:
            self.__indices, self.__values = indices, values
        else:
            self.__indices, self.__values = coalesce(
                indices, values, num_nodes=self.__maxsparsesize)
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
    def maxsparsesize(self):
        return self.__maxsparsesize

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

    def _diag_to_sparse(self, dim: Iterable[int]):
        assert np.all(np.array(dim) < self.__sparse_dim
                      ), "please use tuplewiseapply for operation on dense dim"
        assert np.all(np.array(dim) >= 0), "do not support negative dim"
        '''
        diag dim is then put at the first dim in dim list.
        '''
        dim = sorted(list(dim))
        mask = torch.all((self.indices[dim] - self.indices[[dim[0]]]) == 0,
                         dim=0)
        idx = [i for i in range(self.sparse_dim) if i not in dim[1:]]
        other_shape = tuple([self.shape[i] for i in idx]) + self.denseshape
        return SparseTensor(indices=self.indices[idx][:, mask],
                            values=self.values[mask],
                            shape=other_shape,
                            is_coalesced=(idx[0] == 0)
                            and np.all(np.diff(idx) == 1))

    def _diag_to_dense(self, dim: Iterable[int]) -> Tensor:
        '''
        diag dim is then put at the first dim in dim list.
        '''
        assert np.all(np.array(dim) < self.__sparse_dim
                      ), "please use tuplewiseapply for operation on dense dim"
        assert np.all(np.array(dim) >= 0), "do not support negative dim"
        dim = sorted(list(dim))
        mask = torch.all((self.indices[dim] - self.indices[[dim[0]]]) == 0,
                         dim=0)
        idx = [i for i in range(self.sparse_dim) if i not in dim[1:]]
        nsparse_shape = [self.shape[i] for i in idx]
        nsparse_size = np.prod(nsparse_shape)

        thash = indicehash_tight(
            self.indices[idx][:, mask],
            torch.LongTensor(nsparse_shape).to(self.indices.device))
        ret = torch.zeros((nsparse_size, ) + self.denseshape,
                          device=thash.device,
                          dtype=self.values.dtype)
        ret[thash] = self.values[mask]
        ret = ret.unflatten(0, nsparse_shape)
        return ret

    def diag(self, dim: Optional[Iterable[int]], return_sparse: bool = False):
        '''
        TODO: unit test ??
        '''
        if isinstance(dim, int):
            raise NotImplementedError
        if dim == None:
            dim = list(range(self.sparse_dim))
        if return_sparse:
            return self._diag_to_sparse(dim)
        else:
            return self._diag_to_dense(dim)

    def _reduce_to_sparse(self, dim: Iterable[int], reduce: str):
        assert np.all(np.array(dim) < self.__sparse_dim
                      ), "please use tuplewiseapply for operation on dense dim"
        assert np.all(np.array(dim) >= 0), "do not support negative dim"
        idx = [i for i in range(self.sparse_dim) if i not in list(dim)]
        other_ind = self.indices[idx]
        other_shape = tuple([self.shape[i] for i in idx]) + self.denseshape
        return SparseTensor(indices=other_ind,
                            values=self.values,
                            shape=other_shape,
                            is_coalesced=False)

    def _reduce_to_dense(self, dim: Iterable[int], reduce: str) -> Tensor:
        assert np.all(np.array(dim) < self.__sparse_dim
                      ), "please use tuplewiseapply for operation on dense dim"
        assert np.all(np.array(dim) >= 0), "do not support negative dim"
        idx = [i for i in range(self.sparse_dim) if i not in list(dim)]
        other_ind = self.indices[idx]
        other_shape = [self.shape[i] for i in idx]
        nsparse_shape = other_shape
        nsparse_size = np.prod(nsparse_shape)

        thash = indicehash_tight(
            other_ind,
            torch.LongTensor(nsparse_shape).to(other_ind.device))
        ret = scatter(self.values,
                      thash,
                      dim=0,
                      dim_size=nsparse_size,
                      reduce=reduce)
        ret = ret.unflatten(0, nsparse_shape)
        return ret

    def sum(self,
            dim: Union[int, Optional[Iterable[int]]],
            return_sparse: bool = False):
        if isinstance(dim, int):
            dim = [dim]
        if dim == None:
            return torch.sum(self.values, dim=0)
        elif return_sparse:
            return self._reduce_to_sparse(dim, "sum")
        else:
            return self._reduce_to_dense(dim, "sum")

    def max(self,
            dim: Union[int, Optional[Iterable[int]]],
            return_sparse: bool = False):
        if isinstance(dim, int):
            dim = [dim]
        if dim == None:
            return torch.max(self.values, dim=0)
        elif return_sparse:
            return self._reduce_to_sparse(dim, "max")
        else:
            return self._reduce_to_dense(dim, "max")

    def mean(self,
             dim: Union[int, Optional[Iterable[int]]],
             return_sparse: bool = False):
        if isinstance(dim, int):
            dim = [dim]
        if dim == None:
            return torch.mean(self.values, dim=0)
        elif return_sparse:
            return self._reduce_to_sparse(dim, "mean")
        else:
            return self._reduce_to_dense(dim, "mean")

    def unpooling(self, dim: Union[int, Iterable[int]], tarX):
        '''
        unpooling to of tarX indice
        dim: of tarX
        '''
        if isinstance(dim, int):
            dim = [dim]
        self_hash = indicehash(self.indices)
        assert torch.all(torch.diff(self_hash)), "self is not coalesced"
        tarX: SparseTensor = tarX
        taridx = [i for i in range(tarX.sparse_dim) if i not in list(dim)]
        tar_hash = indicehash(tarX.indices[taridx])

        b2a = torch.clamp_min_(
            torch.searchsorted(self_hash, tar_hash, right=True) - 1, 0)

        matchmask = (self_hash[b2a] == tar_hash)
        ret = torch.zeros((tar_hash.shape[0], ) + self.denseshape,
                          dtype=self.values.dtype,
                          device=self.values.device)
        ret[matchmask] = self.values[b2a[matchmask]]
        return tarX.tuplewiseapply(lambda x: ret)

    def unpooling_fromdense1dim(self, dim: int, X: Tensor):
        '''
        unpooling to of self shape. Note the dim is for self to maintain, and expand other dims
        '''
        assert dim < self.sparse_dim, "only unpooling sparse dim"
        assert X.shape[0] == self.shape[dim], "shape not match"
        return self.tuplewiseapply(lambda _: X[self.indices[dim]])

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
        return SparseTensor(self.indices,
                            nvalues,
                            self.sparseshape + tuple(nvalues.shape[1:]),
                            is_coalesced=True)

    def diagonalapply(self, func: Callable[[Tensor, LongTensor], Tensor]):
        assert self.sparse_dim == 2, "only implemented for 2D"
        nvalues = func(self.values,
                       (self.indices[0] == self.indices[1]).to(torch.long))
        return SparseTensor(self.indices,
                            nvalues,
                            self.sparseshape + tuple(nvalues.shape[1:]),
                            is_coalesced=True)

    def add(self, tarX, samesparse: bool):
        if not samesparse:
            return SparseTensor(
                torch.concat((self.indices, tarX.indices), dim=1),
                torch.concat((self.values, tarX.values), dim=1), self.shape,
                False)
        else:
            return self.tuplewiseapply(lambda x: x + tarX.values)

    def catvalue(self, tarX, samesparse: bool):
        assert samesparse == True, "must have the same sparcity to concat value"
        if isinstance(tarX, SparseTensor):
            return self.tuplewiseapply(lambda _: torch.concat(
                (self.values, tarX.values), dim=-1))
        elif isinstance(tarX, Iterable):
            return self.tuplewiseapply(lambda _: torch.concat(
                [self.values] + [_.values for _ in tarX], dim=-1))
        else:
            raise NotImplementedError

    def __repr__(self):
        return f'SparseTensor(shape={self.shape}, sparse_dim={self.sparse_dim}, nnz={self.nnz})'
