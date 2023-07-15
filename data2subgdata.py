import torch_geometric as pyg 
from torch_geometric.data import Data as PygData, Batch as PygBatch, InMemoryDataset
from torch_geometric.utils import to_networkx, k_hop_subgraph
import networkx as nx
import torch
from typing import Callable, List, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
import os.path as osp

class PairData(PygData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'subg_edge_index':
            self.subg_nodeidx.shape[0]
        if key == 'subg_nodeidx':
            return self.x.shape[0]
        if key == 'subg_edge_attr':
            return 0
        if key == 'subg_rootnode':
            return self.subg_nodeidx.shape[0]
        if key == 'subg_nodelabel':
            return 0
        if key == 'subg_nodebatch':
            return self.num_subgs
        return super().__inc__(key, value, *args, **kwargs)


def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    from pyg
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    dist = torch.empty_like(node_mask, dtype=torch.long).fill_(num_nodes+1)
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    
    for _ in range(num_hops, -1, -1):
        dist[subsets[_]] = _

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    dist = dist[subset]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask, dist


def data2subg(data: PygData, hop: int=2):
    ng = to_networkx(PygData(edge_index=data.edge_index), to_undirected=True)
    cynodes = set(sum(nx.cycle_basis(ng), start=[]))

    subgraphs = []

    for i in range(data.num_nodes):
        if i in cynodes:
            subset, subg_edge_index, rootnode, edge_mask, dist = k_hop_subgraph(i, hop, data.edge_index, relabel_nodes=True,
                                                num_nodes=data.num_nodes)
            assert subset.shape[0] > 1, "empty subgraph!"
            subg_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr
            subg_nodelabel = dist
            subgraphs.append(
                PygData(
                    x=subg_nodelabel, edge_index=subg_edge_index, edge_attr=subg_edge_attr,
                    subg_nodeidx=subset, num_nodes=subset.shape[0],
                )
            )
    if len(subgraphs) > 0:
        subgbatch = PygBatch.from_data_list(subgraphs)    
        return PairData(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, 
                        subg_edge_index=subgbatch.edge_index, subg_edge_attr=subgbatch.edge_attr, 
                        subg_nodeidx=subgbatch.subg_nodeidx, subg_rootnode=subgbatch.subg_nodeidx[subgbatch.x==0],
                        subg_nodelabel=subgbatch.x.to(torch.long), subg_nodebatch=subgbatch.batch, num_nodes=data.num_nodes, num_subgs=len(subgraphs))
    else:
        return PairData(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, 
                        subg_edge_index=torch.empty((2, 0), dtype=data.edge_index.dtype), subg_edge_attr=data.edge_attr if data.edge_attr is None else data.edge_attr[0:0], 
                        subg_nodeidx=torch.empty((0), dtype=torch.long), subg_rootnode=torch.empty((0), dtype=torch.long),
                        subg_nodelabel=torch.empty((0), dtype=torch.long), subg_nodebatch=torch.empty((0), dtype=torch.long), num_nodes=data.num_nodes, num_subgs=0)


class Subgdataset(InMemoryDataset):

    def __init__(self, dataset, num_worker=0):
        self.rawdata = dataset
        self.num_worker = num_worker
        super().__init__(osp.join(dataset.root, "subg_processed"))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'subgdata.pt'
    
    def download(self):
        pass

    def process(self):
        if self.num_worker == 0:
            data_list = list(map(data2subg, self.rawdata))
        else:
            from multiprocessing import Pool
            with Pool(self.num_worker) as pool:
                data_list = pool.map(data2subg, self.rawdata)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])