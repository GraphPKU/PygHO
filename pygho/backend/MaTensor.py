import torch
from torch import Tensor, BoolTensor, LongTensor
from typing import Optional, Callable, Iterable
from typing import Union
# merge torch.nested or torch.masked API in the long run.


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
        self.__mask = mask
        self.__masked_dim = mask.ndim
        while mask.ndim < data.ndim:
            mask = mask.unsqueeze(-1)
        self.__fullmask = mask
        if not is_filled:
            self.__padvalue = torch.inf if padvalue != torch.inf else -torch.inf
            self.fill_masked_(padvalue)
        else:
            self.__padvalue = padvalue

    def fill_masked_(self, val: float = 0) -> None:
        '''
        inplace fill the masked values
        '''
        if self.padvalue == val:
            return
        self.__padvalue = val
        self.__data = torch.where(self.fullmask, self.data, val)

    def fill_masked(self, val: float = 0) -> Tensor:
        '''
        return a tensor with masked values filled with val.
        '''
        if self.__padvalue == val:
            return self.data
        return torch.where(self.fullmask, self.data, val)

    def to(self, device: torch.DeviceObjType, non_blocking: bool = True):
        self.__data = self.__data.to(device, non_blocking=non_blocking)
        self.__mask = self.__mask.to(device, non_blocking=non_blocking)
        self.__fullmask = self.__fullmask.to(device, non_blocking=non_blocking)
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
    def fullmask(self) -> BoolTensor:
        return self.__fullmask

    @property
    def shape(self) -> torch.Size:
        return self.__data.shape

    @property
    def masked_dim(self):
        return self.__masked_dim

    @property
    def dense_dim(self):
        return len(self.denseshape)

    @property
    def sparseshape(self):
        return self.shape[:self.masked_dim]

    @property
    def denseshape(self):
        return self.shape[self.masked_dim:]

    def sum(self, dims: Union[Iterable[int], int], keepdim: bool = False):
        '''
        mask true elements
        '''
        return MaskedTensor(torch.sum(self.fill_masked(0),
                                      dim=dims,
                                      keepdim=keepdim),
                            torch.amax(self.mask, dims, keepdim=keepdim),
                            padvalue=0,
                            is_filled=True)

    def mean(self, dims: Union[Iterable[int], int], keepdim: bool = False):
        '''
        mask true elements
        '''
        count = torch.clamp_min_(
            torch.sum(self.fullmask, dim=dims, keepdim=keepdim), 1)
        valsum = self.sum(dims, keepdim)
        return MaskedTensor(valsum.data / count,
                            valsum.mask,
                            padvalue=valsum.padvalue,
                            is_filled=True)

    def max(self, dims: Union[Iterable[int], int], keepdim: bool = False):
        tmp = self.fill_masked(-torch.inf)
        return MaskedTensor(filterinf(
            torch.amax(tmp, dim=dims, keepdim=keepdim), 0),
                            torch.amax(self.mask, dims, keepdim=keepdim),
                            padvalue=0,
                            is_filled=True)

    def min(self, dims: Union[Iterable[int], int], keepdim: bool = False):
        tmp = self.fill_masked(torch.inf)
        return MaskedTensor(filterinf(
            torch.amax(tmp, dim=dims, keepdim=keepdim), 0),
                            torch.amax(self.mask, dims, keepdim=keepdim),
                            padvalue=0,
                            is_filled=True)

    def diag(self, dim: Iterable[int]):
        '''
        put the output dim to dim[0]
        '''
        assert len(dim) >= 2, "must diag several dims"
        dim = sorted(list(dim))
        tdata = self.data
        tmask = self.mask
        tdata = torch.diagonal(tdata, 0, dim[0], dim[1])
        tmask = torch.diagonal(tmask, 0, dim[0], dim[1])
        for i in range(2, len(dim)):
            tdata = torch.diagonal(tdata, 0, dim[i], -1)
            tmask = torch.diagonal(tmask, 0, dim[i], -1)
        tdata = torch.movedim(tdata, -1, dim[0])
        tmask = torch.movedim(tmask, -1, dim[0])
        return MaskedTensor(tdata, tmask, self.padvalue, True)

    def unpooling(self, dim: Union[int, Iterable[int]], tarX):
        if isinstance(int):
            dim = [dim]
        dim = sorted(list(dim))
        tdata = self.data
        for _ in dim:
            tdata.unsqueeze(_)
        tdata = tdata.expand(*(-1 if i not in dim else tarX.shape[i]
                               for i in range(dim[-1] + 1)))
        return MaskedTensor(tdata, tarX.mask, self.padvalue, False)

    def tuplewiseapply(self, func: Callable[[Tensor], Tensor]):
        # it may cause nan in gradient and makes amp unable to update
        ndata = func(self.fill_masked(0))
        return MaskedTensor(ndata, self.mask)

    def diagonalapply(self, func: Callable[[Tensor, LongTensor], Tensor]):
        assert self.masked_dim == 3, "only implemented for 3D"
        diagonaltype = torch.eye(self.shape[1],
                                 self.shape[2],
                                 dtype=torch.long,
                                 device=self.data.device)
        diagonaltype = diagonaltype.unsqueeze(0).expand_as(self.mask)
        ndata = func(self.data, diagonaltype)
        return MaskedTensor(ndata, self.mask)

    def add(self, tarX, samesparse: bool):
        if samesparse:
            return MaskedTensor(tarX.data + self.data,
                                self.mask,
                                self.padvalue,
                                is_filled=self.padvalue == tarX.padvalue)
        else:
            return MaskedTensor(
                tarX.fill_masked(0) + self.fill_masked(0),
                torch.logical_or(self.mask, tarX.mask), 0, True)

    def catvalue(self, tarX, samesparse: bool):
        assert samesparse == True, "must have the same sparcity to concat value"
        if isinstance(tarX, MaskedTensor):
            return self.tuplewiseapply(lambda _: torch.concat(
                (self.data, tarX.data), dim=-1))
        elif isinstance(tarX, Iterable):
            return self.tuplewiseapply(lambda _: torch.concat(
                [self.data] + [_.data for _ in tarX], dim=-1))
        else:
            raise NotImplementedError