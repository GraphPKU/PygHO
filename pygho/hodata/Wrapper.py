from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch_geometric.data import Dataset as PygDataset, Data as PygData, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader import DataLoader as PygDataLoader
import os.path as osp
import re
from typing import Any, Callable, List, Iterable, Sequence, Tuple, Union
from torch import Tensor
from functools import partial
from .SpData import sp_datapreprocess
from .MaData import ma_datapreprocess, batch2dense
from torch_geometric.transforms import Compose


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


def Sppretransform(sampler: Callable[[PygData], Tuple[Tensor, Tensor, Union[List[int], int]]], keys: List[str]=[], pre_transform: Callable=None):
    subgpre_transform = partial(sp_datapreprocess,
                                subgsampler=sampler,
                                keys=keys)
    if pre_transform is not None:
        return Compose([pre_transform, subgpre_transform])
    else:
        return subgpre_transform


def Mapretransform(sampler: Callable, pre_transform: Callable=None):
    subgpre_transform = partial(ma_datapreprocess, subgsampler=sampler)
    if pre_transform is not None:
        return Compose([pre_transform, subgpre_transform])
    else:
        return subgpre_transform


def SubgDatasetClass(datasetclass, processname: str=None):

    @property
    def processed_dir(self) -> str:
        if processname is None:
            return osp.join(
                self.root,
                f'processed__{_repr(self.pre_transform)}__{_repr(self.pre_filter)}'
            )
        else:
            return osp.join(
                self.root,
                f'processed__{processname}'
            )


    setattr(datasetclass, "processed_dir", processed_dir)
    return datasetclass


SpDataloader = PygDataLoader


class IterWrapper:
    def __init__(self, iterator: Iterable, batch_transform: Callable, device) -> None:
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

    def __init__(self,
                 dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: List[str] | None = None,
                 exclude_keys: List[str] | None = None,
                 device = None,
                 **kwargs):
        if follow_batch is None:
            follow_batch = []
        for i in ["edge_index", "tuplefeat"]:
            if i not in follow_batch:
                follow_batch.append(i)
        super().__init__(dataset, batch_size, shuffle, follow_batch,
                         exclude_keys, **kwargs)
        self.device = device

    def __iter__(self) -> _BaseDataLoaderIter:
        ret = super().__iter__()
        return IterWrapper(ret, batch2dense, self.device)

class MaDataloader(PygDataLoader):

    def __init__(self,
                 dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: List[str] | None = None,
                 exclude_keys: List[str] | None = None,
                 device = None,
                 **kwargs):
        if follow_batch is None:
            follow_batch = []
        for i in ["edge_index", "tuplefeat"]:
            if i not in follow_batch:
                follow_batch.append(i)
        super().__init__(dataset, batch_size, shuffle, follow_batch,
                         exclude_keys, **kwargs)
        self.device = device

    def __iter__(self) -> _BaseDataLoaderIter:
        ret = super().__iter__()
        return IterWrapper(ret, batch2dense, self.device)
        
