import torch
from typing import List, Optional
from torch import LongTensor, Tensor
from torch_geometric.utils import coalesce


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
                torch.max(indices, dim=1).tolist() + list(values.shape[1:]))
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

    @classmethod
    def from_torch_sparse_coo(cls, A: torch.Tensor):
        assert A.is_sparse, "from_torch_sparse_coo converts a torch.sparse_coo_tensor to SparseTensor"
        ret = cls(A._indices(), A._values(), A.shape, A.is_coalesced())
        return ret

    def __repr__(self):
        return f'SparseTensor(shape={self.shape}, sparse_dim={self.sparse_dim}, nnz={self.nnz})'


if __name__ == "__main__":
    n, m, nnz, d = 100, 200, 50, 5
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
    print("should of same shape and nnz ", A1, A1c, A2, A2f, A2cf, sep="\n")