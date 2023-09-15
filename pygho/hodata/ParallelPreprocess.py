import torch
from torch_geometric.data import InMemoryDataset, Data as PygData
from typing import Callable, Optional, Iterable
from multiprocessing import Pool
from pqdm.processes import pqdm
from tqdm import tqdm
from .Wrapper import _repr
import os.path as osp


class ParallelPreprocessDataset(InMemoryDataset):
    '''
    Parallelly transform a pyg dataset/
    Args:
        root: position to save processed data
        data_list: list of PygData or PygDataset. 
        pre_transform: a function maps PygData to PygData. Run only once for all data. Usually be tuple sampler. 
        num_worker: number of processes for preprocessing. Can be the number of cpu.
        processedname: name to save processed data. If none, the name will be a hash of pre_transform function
        transform: dynamically transform data during data loading.
    '''

    def __init__(self,
                 root: str,
                 data_list: Iterable[PygData],
                 pre_transform: Callable[[PygData], PygData],
                 num_worker: int,
                 processedname: Optional[str] = None,
                 transform: Optional[Callable[[PygData], PygData]] = None):
        self.tmp_data_list = list(data_list)
        self.num_worker = num_worker
        self.processedname = processedname
        super().__init__(root,
                         pre_transform=pre_transform,
                         transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def processed_dir(self) -> str:
        if self.processedname is None:
            return osp.join(
                self.root,
                f'processed__{_repr(self.pre_transform)}__{_repr(self.pre_filter)}'
            )
        else:
            return osp.join(self.root, f'processed__{self.processedname}')

    def process(self):
        if self.num_worker > 0:
            data_list = pqdm(self.tmp_data_list,
                             self.pre_transform,
                             n_jobs=self.num_worker)
        else:
            data_list = [
                self.pre_transform(_) for _ in tqdm(self.tmp_data_list)
            ]
        torch.save(self.collate(data_list), self.processed_paths[0])
