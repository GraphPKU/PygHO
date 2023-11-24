import torch
from torch import Tensor, BoolTensor, LongTensor
from typing import Optional, Callable, Iterable
from typing import Union
# merge torch.nested or torch.masked API in the long run.


def filterinf(X: Tensor, filled_value: float = 0):
    """
    Replaces positive and negative infinity values in a tensor with a specified value.

    Args:

    - X (Tensor): The input tensor.
    - filled_value (float, optional): The value to replace positive and negative
      infinity values with (default: 0).

    Returns:

    - Tensor: A tensor with positive and negative infinity values replaced by the
      specified `filled_value`.

    Example:
    
    ::
    
        input_tensor = torch.tensor([1.0, 2.0, torch.inf, -torch.inf, 3.0])
        result = filterinf(input_tensor, filled_value=999.0)

    """
    return X.masked_fill(torch.isinf(X), filled_value)


class MaskedTensor:
    """
    Represents a masked tensor with optional padding values.
    This class allows you to work with tensors that have a mask indicating valid and
    invalid values. You can perform various operations on the masked tensor, such as
    filling masked values, computing sums, means, maximums, minimums, and more.

    Parameters:
    
    - data (Tensor): The underlying data tensor of shape (\*maskedshape, \*denseshape)
    - mask (BoolTensor): The mask tensor of shape (\*maskedshape) 
      where `True` represents valid values, and False` represents invalid values.
    - padvalue (float, optional): The value to use for padding. Defaults to 0.
    - is_filled (bool, optional): Indicates whether the invalid values have already
      been filled to the padvalue. Defaults to False.

    Attributes:
    
    - data (Tensor): The underlying data tensor.
    - mask (BoolTensor): The mask tensor.
    - fullnegmask (BoolTensor): The mask tensor after broadcasting to match the data's
      dimensions and take logical_not.
    - padvalue (float): The padding value.
    - shape (torch.Size): The shape of the data tensor.
    - masked_dim (int): The number of dimensions in maskedshape.
    - dense_dim (int): The number of dimensions in denseshape.
    - maskedshape (torch.Size): The shape of the tensor up to the masked dimensions.
    - denseshape (torch.Size): The shape of the tensor after the masked dimensions.

    Methods:
    
    - fill_masked_(self, val: float = 0) -> None: In-place fill of masked values.
    - fill_masked(self, val: float = 0) -> Tensor: Return a tensor with masked values
      filled with the specified value.
    - to(self, device: torch.DeviceObjType, non_blocking: bool = True): Move the
      tensor to the specified device.
    - sum(self, dims: Union[Iterable[int], int], keepdim: bool = False): Compute the
      sum of masked values along specified dimensions.
    - mean(self, dims: Union[Iterable[int], int], keepdim: bool = False): Compute
      the mean of masked values along specified dimensions.
    - max(self, dims: Union[Iterable[int], int], keepdim: bool = False): Compute the
      maximum of masked values along specified dimensions.
    - min(self, dims: Union[Iterable[int], int], keepdim: bool = False): Compute the
      minimum of masked values along specified dimensions.
    - diag(self, dims: Iterable[int]): Extract diagonals from the tensor. 
      The dimensions in dims will be take diagonal and put at dims[0]
    - unpooling(self, dims: Union[int, Iterable[int]], tarX): Perform unpooling
      operation along specified dimensions.
    - tuplewiseapply(self, func: Callable[[Tensor], Tensor]): Apply a function to
      each element of the masked tensor.
    - diagonalapply(self, func: Callable[[Tensor, LongTensor], Tensor]): Apply a
      function to diagonal elements of the masked tensor.
    - add(self, tarX, samesparse: bool): Add two masked tensors together.
    - catvalue(self, tarX, samesparse: bool): Concatenate values of two masked
      tensors.
    """
    def __init__(self,
                 data: Tensor,
                 mask: BoolTensor,
                 padvalue: float = 0.0,
                 is_filled: bool = False):
        # mask: True for valid value, False for invalid value
        assert data.ndim >= mask.ndim, "data's #dim should be larger than mask "
        assert data.shape[:mask.
                          ndim] == mask.shape, "data and mask's first dimensions should match"
        self.__data = data
        self.__mask = mask
        self.__masked_dim = mask.ndim
        if self.dense_dim > 0:
          mask = mask.unsqueeze(-1)
          if self.dense_dim > 1:
              mask = mask.unflatten(-1, (self.dense_dim)*(1,))
        self.__fullnegmask = torch.logical_not(mask)
        self.__padvalue = padvalue
        if not is_filled:
            self.__data = self.__data.masked_fill(self.fullnegmask, padvalue)

    def fill_masked_(self, val: float = 0.0) -> None:
        """
        inplace fill the masked values
        """
        if self.padvalue == val:
            return
        self.__padvalue = val
        self.__data = self.__data.masked_fill(self.fullnegmask, val)

    def fill_masked(self, val: float = 0.0) -> Tensor:
        """
        return a tensor with masked values filled with val.
        """
        if self.__padvalue == val:
            return self.data
        return self.data.masked_fill(self.fullnegmask, val)

    def to(self, device: torch.DeviceObjType, non_blocking: bool = True):
        """
        move data to some device
        """
        self.__data = self.__data.to(device, non_blocking=non_blocking)
        self.__mask = self.__mask.to(device, non_blocking=non_blocking)
        self.__fullnegmask = self.__fullnegmask.to(device, non_blocking=non_blocking)
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
    def fullnegmask(self) -> BoolTensor:
        return self.__fullnegmask

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
    def maskedshape(self):
        return self.shape[:self.masked_dim]

    @property
    def denseshape(self):
        return self.shape[self.masked_dim:]

    def sum(self, dims: Union[Iterable[int], int], keepdim: bool = False):
        return MaskedTensor(torch.sum(self.fill_masked(0.),
                                      dim=dims,
                                      keepdim=keepdim),
                            torch.amax(self.mask, dims, keepdim=keepdim),
                            padvalue=0,
                            is_filled=True)

    def mean(self, dims: Union[Iterable[int], int], keepdim: bool = False):
        count = torch.clamp_min_(
            torch.sum(torch.logical_not(self.fullnegmask), dim=dims, keepdim=keepdim), 1)
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
            torch.amin(tmp, dim=dims, keepdim=keepdim), 0),
                            torch.amax(self.mask, dims, keepdim=keepdim),
                            padvalue=0,
                            is_filled=True)

    def diag(self, dims: Iterable[int]):
        """
        put the reduced output to dim[0]
        """
        assert len(dims) >= 2, "must diag several dims"
        dims = sorted(list(dims))
        tdata = self.data
        tmask = self.mask
        tdata = torch.diagonal(tdata, 0, dims[0], dims[1])
        tmask = torch.diagonal(tmask, 0, dims[0], dims[1])
        for i in range(2, len(dims)):
            tdata = torch.diagonal(tdata, 0, dims[i], -1)
            tmask = torch.diagonal(tmask, 0, dims[i], -1)
        tdata = torch.movedim(tdata, -1, dims[0])
        tmask = torch.movedim(tmask, -1, dims[0])
        return MaskedTensor(tdata, tmask, self.padvalue, True)

    def unpooling(self, dims: Union[int, Iterable[int]], tarX):
        if isinstance(dims, int):
            dims = [dims]
        dims = sorted(list(dims))
        tdata = self.data
        for _ in dims:
            tdata = tdata.unsqueeze(_)
        tdata = tdata.expand(*(-1 if i not in dims else tarX.shape[i]
                               for i in range(tdata.ndim)))
        return MaskedTensor(tdata, tarX.mask, self.padvalue, False)

    def tuplewiseapply(self, func: Callable[[Tensor], Tensor]):
        # it may cause nan in gradient and makes amp unable to update
        ndata = func(self.fill_masked(0.))
        return MaskedTensor(ndata, self.mask)

    def diagonalapply(self, func_d: Callable[[Tensor], Tensor], func_nd: Callable[[Tensor], Tensor]):
        assert self.masked_dim == 3, "only implemented for 2D"
        mask = torch.eye(self.shape[1],
                                 self.shape[2],
                                 dtype=torch.long,
                                 device=self.data.device)
        val_d = func_d(self.fill_masked(0.0))
        val_nd = func_d(self.fill_masked(0.0))
        
        val_d, val_nd = torch.movedim(val_d, (1, 2), (-2, -1)), torch.movedim(val_nd, (1, 2), (-2, -1))
        ndata = torch.where(mask, val_d, val_nd)
        ndata = torch.movedim(ndata, (-2, -1), (1, 2))
        
        return MaskedTensor(ndata, self.mask)

    def add(self, tarX, samesparse: bool):
        assert isinstance(tarX, MaskedTensor)
        tarX: MaskedTensor = tarX
        if samesparse:
            return MaskedTensor(tarX.data + self.data,
                                self.mask,
                                self.padvalue,
                                is_filled=self.padvalue == tarX.padvalue)
        else:
            return MaskedTensor(
                tarX.fill_masked(0.) + self.fill_masked(0.),
                torch.logical_or(self.mask, tarX.mask), 0, True)

    def catvalue(self, tarX: Iterable, samesparse: bool):
        assert samesparse == True, "must have the same sparcity to concat value"
        return self.tuplewiseapply(lambda _: torch.concat([self.data] + [_.data for _ in tarX], dim=-1))