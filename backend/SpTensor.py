import torch
from typing import List, Optional
from torch import LongTensor, Tensor
from torch_geometric.utils import coalesce

class SparseTensor:

    def __init__(self, indices: LongTensor, values: Optional[Tensor]=None, shape: Optional[List[int]]=None, is_coalesced: bool=False):
        assert indices.ndim == 2, "indice should of shape (#sparsedim, #nnz)"
        if values is not None:
            assert indices.shape[1] == values.shape[0], "indices and values should have the same number of nnz"
        if shape is not None:
            self.shape = shape
        else:
            self.shape = torch.max(indices, dim=1).tolist() + list(values.shape[1:])
        self.__sparse_dim = indices.shape[0]
        self.__maxsparsesize = max(self.shape[:self.sparse_dim])
        self.__nnz = indices.shape[1]
        if is_coalesced:
            self.__indices, self.__values = indices, values
        else:
            self.__indices, self.__values = coalesce(indices, values, num_nodes=self.__maxsparsesize)
        
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