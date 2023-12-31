from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch_geometric.data import Data as PygData, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader import DataLoader as PygDataLoader
import re
from typing import Any, Callable, List, Iterable, Sequence, Tuple, Union, Optional
from torch import Tensor
from functools import partial
from .SpData import sp_datapreprocess, batch2sparse
from .MaData import ma_datapreprocess, batch2dense
from torch_geometric.transforms import Compose
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor


def _repr(obj: Any) -> str:
    if obj is None:
        return 'None'
    ret = re.sub('at 0x[0-9a-fA-F]+', "", str(obj))
    ret = ret.replace("\n", " ")
    ret = ret.replace("functools.partial", " ")
    ret = ret.replace("function", " ")
    ret = ret.replace("<", " ")
    ret = ret.replace(">", " ")
    ret = ret.replace(" ", "")
    return ret


def Sppretransform(tuplesamplers: List[Callable[[PygData], SparseTensor]]
                   | Callable[[PygData], SparseTensor],
                   annotate: List[str] = [""],
                   keys: List[str] = [""]):
    """
    Create a data pre-transformation function for sparse data.

    Args:
    
    - tuplesamplers (Union[Callable[[PygData], Tuple[Tensor, Tensor, Union[List[int], int]]], List[Callable[[PygData], Tuple[Tensor, Tensor, Union[List[int], int]]]]]): A tuple sampler or a list of tuple samplers.
    - annotate (List[str], optional): A list of annotations. Defaults to [""].
    - keys (List[str], optional): A list of keys. Defaults to [""].

    Returns:
    
    - Callable: A data pre-transformation function.
    """
    if not isinstance(tuplesamplers, Iterable):
        tuplesamplers = [tuplesamplers]
    hopre_transform = partial(sp_datapreprocess,
                              tuplesamplers=tuplesamplers,
                              annotate=annotate,
                              keys=keys)
    return hopre_transform


def Mapretransform(tuplesamplers: List[Callable[[PygData], MaskedTensor]]
                   | Callable[[PygData], MaskedTensor],
                   annotate: List[str] = [""]):
    """
    Create a data pre-transformation function for dense data.

    Args:
    
    - tuplesamplers (Union[Callable[[PygData], Tuple[Tensor, List[int]]], List[Callable[[PygData], Tuple[Tensor, List[int]]]]]): A tuple sampler or a list of tuple samplers.
    - annotate (List[str], optional): A list of annotations. Defaults to [""].

    Returns:
    
    - Callable: A data pre-transformation function.
    """
    if not isinstance(tuplesamplers, Iterable):
        tuplesamplers = [tuplesamplers]
    hopre_transform = partial(ma_datapreprocess,
                              tuplesamplers=tuplesamplers,
                              annotate=annotate)
    return hopre_transform


class IterWrapper:
    """
    A wrapper for the iterator of a data loader.
    """

    def __init__(self, iterator: Iterable, batch_transform: Callable,
                 device) -> None:
        self.iterator = iterator
        self.device = device
        self.batch_transform = batch_transform

    def __next__(self):
        batch = next(self.iterator)
        if self.device is not None:
            '''
            sparse batch is usually smaller than dense batch and the to device takes less time
            '''
            batch = batch.to(self.device, non_blocking=True)
        batch = self.batch_transform(batch)
        return batch


class SpDataloader(PygDataLoader):
    """
    A data loader for sparse data that converts the inner data format to SparseTensor.

    Args:
    
    - dataset (Dataset | Sequence[BaseData] | DatasetAdapter): The input dataset or data sequence.
    - device (optional): The device to place the data on. Defaults to None.
    - \*\*kwargs: Additional keyword arguments for DataLoader. Same as Pyg Dataloader.
    """

    def __init__(self,
                 dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: List[str] | None = None,
                 exclude_keys: List[str] | None = None,
                 device=None,
                 **kwargs):
        super().__init__(dataset, batch_size, shuffle, follow_batch,
                         exclude_keys, **kwargs)
        self.device = device
        keys = [
            k.removeprefix("tupleid") for k in dataset[0].to_dict().keys()
            if k.startswith("tupleid")
        ]
        self.keys = keys

    def __iter__(self) -> _BaseDataLoaderIter:
        ret = super().__iter__()
        return IterWrapper(ret, partial(batch2sparse, keys=self.keys),
                           self.device)


class MaDataloader(PygDataLoader):
    """
    A data loader for sparse data that converts the inner data format to MaskedTensor.

    Args:
    
    - dataset (Dataset | Sequence[BaseData] | DatasetAdapter): The input dataset or data sequence.
    - device (optional): The device to place the data on. Defaults to None.
    - denseadj (bool, optional): Whether to use dense adjacency. Defaults to True.
    - other kwargs: Additional keyword arguments for DataLoader. Same as Pyg dataloader

    """

    def __init__(self,
                 dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: List[str] | None = None,
                 exclude_keys: List[str] | None = None,
                 device=None,
                 denseadj: bool = True,
                 **kwargs):
        if follow_batch is None:
            follow_batch = []
        keys = [
            k.removeprefix("tuplefeat") for k in dataset[0].to_dict().keys()
            if k.startswith("tuplefeat")
        ]
        self.keys = keys
        for i in ["edge_index"] + [f"tuplefeat{_}" for _ in keys]:
            if i not in follow_batch:
                follow_batch.append(i)
        super().__init__(dataset, batch_size, shuffle, follow_batch,
                         exclude_keys, **kwargs)
        self.device = device
        self.denseadj = denseadj

    def __iter__(self) -> _BaseDataLoaderIter:
        ret = super().__iter__()
        return IterWrapper(
            ret, partial(batch2dense, keys=self.keys, denseadj=self.denseadj),
            self.device)
