import torch
from typing import List, Optional, Tuple
from torch import LongTensor, Tensor
from torch_scatter import scatter
# ?? TODO add coalesce for sparse_dim >=2

def indicehash(indice: LongTensor, n: Optional[int]=None)->LongTensor:
    assert indice.ndim == 2
    sparse_dim = indice.shape[0]
    interval = (63//sparse_dim)
    n = n if not (n is None) else torch.max(indice).item()+1
    assert n < (1<<interval)

    eihash = indice[sparse_dim-1].clone()
    for i in range(1, sparse_dim):
        eihash.bitwise_or_(indice[sparse_dim-1-i].bitwise_left_shift(interval*(i))) 
    return eihash

def decodehash(indhash: LongTensor, sparse_dim: int) -> LongTensor:
    '''
    transfer hash into pairs
    '''
    interval = (63//sparse_dim)
    mask = eval("0b"+"1"*interval)
    offset = torch.range(sparse_dim, device=indhash.device).unsqueeze(-1) * interval
    ret = torch.bitwise_right_shift(indhash.unsqueeze(0), offset).bitwise_and_(mask)
    return ret

def coalesce(
    edge_index: LongTensor,
    edge_attr: Optional[Tensor]=None,
    num_nodes: Optional[int]=None,
    reduce: str = 'add') -> Tuple[Tensor, Optional[Tensor]]:
    '''
    ??TODO unittest
    '''
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
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Example:

        >>> edge_index = torch.tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = torch.tensor([1., 1., 1., 1.])
        >>> coalesce(edge_index)
        tensor([[1, 2, 3],
                [3, 1, 2]])

        >>> # Sort `edge_index` column-wise
        >>> coalesce(edge_index, sort_by_row=False)
        tensor([[2, 3, 1],
                [1, 2, 3]])

        >>> coalesce(edge_index, edge_attr)
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>> coalesce(edge_index, edge_attr, reduce='mean')
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([1., 1., 1.]))
    """
    eihash = indicehash(edge_index, num_nodes)
    eihash, idx = torch.unique(eihash, return_inverse=True)
    edge_index = decodehash(eihash)
    if edge_attr is None:
        return edge_index, None
    else:
        edge_attr = scatter(edge_attr, idx, dim=0, dim_size=eihash.shape[0], reduce=reduce)
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
    
    def to_torch_sparse_coo(self):
        ret = torch.sparse_coo_tensor(self.indices, self.values, size=self.shape)
        ret = ret._coalesced_(self.is_coalesced())
        return ret

    def __repr__(self):
        return f'SparseTensor(shape={self.shape}, sparse_dim={self.sparse_dim}, nnz={self.nnz})'


if __name__ == "__main__":
    n, m, nnz, d = 10, 20, 50, 5
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
    print("debug from_torch_sparse_coo ", (A1t-A1).coalesce())

    print("should of same shape and nnz ", A1c, A1t, A2, A2f, A2cf, sep="\n")