import torch
from torch import Tensor, BoolTensor
from typing import Optional
# merge torch.nested or torch.masked API in the long run.
# Maybe we can let inherit torch.Tensor, but seems very complex https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor


def filterinf(X: Tensor, filled_value: float = 0):
    return torch.where(torch.logical_or(X == torch.inf, X == -torch.inf),
                       filled_value, X)


class MaskedTensor:

    def __init__(self,
                 data: Tensor,
                 mask: BoolTensor,
                 padvalue: float = 0.0,
                 is_filled: bool = False):
        '''
        mask: True for valid value, False for invalid value
        '''
        assert data.ndim >= mask.ndim, "data's #dim should be larger than mask "
        assert data.shape[:mask.
                          ndim] == mask.shape, "data and mask's first dimensions should match"
        self.__data = data
        while mask.ndim < data.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(data)
        self.__mask = mask
        if not is_filled:
            self.__padvalue = torch.inf if padvalue != torch.inf else -torch.inf
            self.fill_masked_(padvalue)
        else:
            self.__padvalue = padvalue

    def fill_masked_(self, val: float = 0):
        if self.padvalue == val:
            return
        self.__padvalue = val
        self.__data = self.__data.masked_fill(torch.logical_not(self.__mask),
                                              val)  # can be inplace?

    def fill_masked(self, val: float = 0) -> Tensor:
        if self.__padvalue == val:
            return self.data
        return torch.where(self.mask, self.data, val)

    def to(self, device: torch.DeviceObjType):
        self.__data = self.__data.to(device)
        self.__mask = self.__mask.to(device)
        return self

    @property
    def padvalue(self) -> float:
        return self.__padvalue

    @property
    def data(self) -> Tensor:
        return self.__data

    @property
    def mask(self) -> BoolTensor:
        return self.__mask

    @property
    def shape(self) -> torch.Size:
        return self.__data.shape

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        '''
        mask true elements
        '''
        if dim is None:
            return torch.sum(self.fill_masked(0))
        else:
            return torch.sum(self.fill_masked(0), dim=dim, keepdim=keepdim)

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        '''
        mask true elements
        '''
        if dim is None:
            gsize = torch.clamp_min_(
                torch.sum(self.mask, dim=dim, keepdim=keepdim), 1)
            return self.sum(dim, keepdim) / gsize
        else:
            return self.sum(dim, keepdim) / torch.clamp_min_(
                torch.sum(self.mask), 1)

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        tmp = self.fill_masked(-torch.inf)
        keepdim = keepdim and dim is not None
        if dim == None:
            ret = torch.max(tmp)
        else:
            ret = torch.max(tmp, dim=dim, keepdim=keepdim)[0]
        return filterinf(ret)

    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        tmp = self.fill_masked(torch.inf)
        keepdim = keepdim and dim is not None
        if dim == None:
            ret = torch.min(tmp)
        else:
            ret = torch.min(tmp, dim=dim, keepdim=keepdim)[0]
        return filterinf(ret)


if __name__ == "__main__":
    A = torch.tensor([-torch.inf, 0, torch.inf, 1, 2, -torch.inf, 3])
    print(filterinf(A))
    B = 2
    N = 3
    M = 2
    data = torch.randn((B, N, M))
    mask = torch.zeros((B, N), dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :1] = True
    mt = MaskedTensor(data, mask, padvalue=torch.inf)
    print(mt.data)
    print(mt.mask)
    print(mt.shape)
    print(mt.max(0), mt.min(0), mt.sum(0), mt.mean(0))
    print(mt.max(), mt.min(), mt.sum(), mt.mean())
    print(mt.fill_masked(1))