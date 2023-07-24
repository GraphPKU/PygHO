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
        return indice.flatten()
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
        return indhash
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
        return indice
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
        return indhash
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
        if shape is not None:
            self.__shape = tuple(shape)
        else:
            self.__shape = tuple(
                list(map(lambda x: x + 1,
                         torch.max(indices, dim=1).tolist())) +
                list(values.shape[1:]))
        self.__sparse_dim = indices.shape[0]
        self.__maxsparsesize = max(self.shape[:self.sparse_dim])
        if is_coalesced:
            self.__indices, self.__values = indices, values
        else:
            self.__indices, self.__values = coalesce(
                indices, values, num_nodes=self.__maxsparsesize)
        self.__nnz = self.indices.shape[1]

    def is_coalesced(self):
        return True

    def to(self, device: torch.DeviceObjType):
        self.__indices = self.__indices.to(device)
        self.__values = self.__values.to(device)
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

    def _reduce_to_sparse(self, dim: Iterable[int], reduce: str):
        assert np.all(np.array(dim) < self.__sparse_dim
                      ), "please use tuplewiseapply for operation on dense dim"
        assert np.all(np.array(dim) >= 0), "do not support negative dim"
        idx = [i for i in range(self.sparse_dim) if i not in list(dim)]
        other_ind = self.indices[idx]
        other_shape = [self.shape[i] for i in idx]
        other_ind, other_value = coalesce(other_ind, self.values,
                                          self.maxsparsesize, reduce)
        return SparseTensor(indices=other_ind,
                            values=other_value,
                            shape=other_shape,
                            is_coalesced=True)

    def _reduce_to_dense(self, dim: Iterable[int], reduce: str) -> Tensor:
        assert np.all(np.array(dim) < self.__sparse_dim
                      ), "please use tuplewiseapply for operation on dense dim"
        assert np.all(np.array(dim) >= 0), "do not support negative dim"
        idx = [i for i in range(self.sparse_dim) if i not in list(dim)]
        nsparse_dim = len(idx)
        other_ind = self.indices[idx]
        other_shape = [self.shape[i] for i in idx]
        nsparse_shape = other_shape[:nsparse_dim]
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
        unit test TODO??
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
        ret = torch.zeros((tar_hash.shape[0], ) + self.values.shape[1:],
                          dtype=self.values.dtype,
                          device=self.values.device)
        ret[matchmask] = self.values[b2a[matchmask]]
        return tarX.tuplewiseapply(lambda x: ret)

    def unpooling_fromdense1dim(self, dim: int, X: Tensor):
        '''
        unit test TODO??
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

    def tuplewiseapply(self, func: Callable):
        nvalues = func(self.values)
        return SparseTensor(self.indices,
                            nvalues,
                            self.shape[:self.sparse_dim] +
                            tuple(nvalues.shape[1:]),
                            is_coalesced=True)

    def __repr__(self):
        return f'SparseTensor(shape={self.shape}, sparse_dim={self.sparse_dim}, nnz={self.nnz})'


if __name__ == "__main__":
    # 2D SparseTensor
    sd, sshape, nnz, d = 5, (2, 3, 7, 11, 13), 17, 7
    indices = torch.stack(
        tuple(torch.randint(sshape[i], (nnz, )) for i in range(sd)))
    values = torch.randn((nnz, d))
    tsshape = torch.LongTensor(sshape)
    thash = indicehash_tight(indices, tsshape)
    hhash = (((
        (indices[0]) * sshape[1] + indices[1]) * sshape[2] + indices[2]) *
             sshape[3] + indices[3]) * sshape[4] + indices[4]
    print(torch.all(thash == hhash))
    dthash = decodehash_tight(thash, tsshape)
    print(torch.all(dthash == indices))

    n, m, nnz, d = 5, 7, 17, 7
    indices = torch.stack(
        (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, ))))
    values = torch.randn((nnz, d))

    A1 = torch.sparse_coo_tensor(indices, values, size=(n, m, d))
    A2 = SparseTensor(indices, values, (n, m, d), False)

    A2f = SparseTensor.from_torch_sparse_coo(A1)

    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2f.indices)), torch.max((A2.values - A2f.values)))

    A1c = A1.coalesce()
    print("debug coalesce ", torch.max((A2.indices - A1c.indices())),
          torch.max((A2.values - A1c.values())))

    A2cf = SparseTensor.from_torch_sparse_coo(A1c)
    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2cf.indices)), torch.max((A2.values - A2cf.values)))

    A1t = A2.to_torch_sparse_coo()
    print("debug from_torch_sparse_coo ", (A1t - A1).coalesce())

    print("should of same shape and nnz ", A1c, A1t, A2, A2f, A2cf, sep="\n")

    # 3D SparseTensor
    n, m, l, nnz, d = 2, 3, 5, 73, 7
    indices = torch.stack(
        (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, )),
         torch.randint(0, l, (nnz, ))))
    values = torch.randn((nnz, d))

    A1 = torch.sparse_coo_tensor(indices, values, size=(n, m, l, d))
    A2 = SparseTensor(indices, values, (n, m, l, d), False)

    A2f = SparseTensor.from_torch_sparse_coo(A1)

    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2f.indices)), torch.max((A2.values - A2f.values)))

    A1c = A1.coalesce()
    print("debug coalesce ", torch.max((A2.indices - A1c.indices())),
          torch.max((A2.values - A1c.values())))

    A2cf = SparseTensor.from_torch_sparse_coo(A1c)
    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2cf.indices)), torch.max((A2.values - A2cf.values)))

    A1t = A2.to_torch_sparse_coo()
    print("debug from_torch_sparse_coo ",
          (A1t - A1).coalesce().values().abs().max())

    print("should of same shape and nnz ", A1c, A1t, A2, A2f, A2cf, sep="\n")
    A3 = A2.tuplewiseapply(lambda x: torch.ones_like(x))
    assert torch.all(A3.values == 1)
    print("debug tuplewiseapply ", A3)

    A2m = A2.max(dim=1, return_sparse=True)
    Adm = A2.max(dim=1)
    print(A2m.indices, A2m.values, Adm, A1.to_dense().max(dim=1)[0])
    A2mu = A2m.unpooling(dim=1, tarX=A2)
    print(A2mu.indices, A2mu.values)