import torch_geometric as pyg
from torch_geometric.data import Data as PygData, Batch as PygBatch, InMemoryDataset
from torch_geometric.utils import to_networkx, k_hop_subgraph
import networkx as nx
import torch
from typing import Any, List, Optional, Tuple, Union, Callable, Dict
from torch import Tensor, LongTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
import os.path as osp
from .Spspmm import spspmm_ind, filterij
from typing import Tuple
from torch_geometric.utils import coalesce

class SubgData(PygData):

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key == 'tupleid':
            return self.num_nodes
        if key == 'AX_akl':
            return torch.tensor(
                [[self.num_tuples], [self.num_edges], [self.num_tuples]],
                dtype=torch.long)
        if key == 'XA_akl':
            return torch.tensor(
                [[self.num_tuples], [self.num_tuples], [self.num_edges]],
                dtype=torch.long)
        if key == 'XX_akl':
            return torch.tensor(
                [[self.num_tuples], [self.num_tuples], [self.num_tuples]],
                dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'tupleid':
            return 1
        if key == 'AX_akl':
            return 1
        if key == 'XA_akl':
            return 1
        if key == 'XX_akl':
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def k_hop_subgraph(
    node_idx: Union[int, List[int], LongTensor],
    num_hops: int,
    edge_index: LongTensor,
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
    dist = torch.empty_like(node_mask, dtype=torch.long).fill_(num_nodes + 1)
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


def CycleCenteredKhopSampler(data: PygData,
                             hop: int = 2) -> Tuple[LongTensor, LongTensor]:
    ng = to_networkx(PygData(edge_index=data.edge_index), to_undirected=True)
    cynodes = set(sum(nx.cycle_basis(ng), start=[]))

    subgraphs = []

    for i in range(data.num_nodes):
        if i in cynodes:
            subset, _, _, _, dist = k_hop_subgraph(i,
                                                   hop,
                                                   data.edge_index,
                                                   relabel_nodes=True,
                                                   num_nodes=data.num_nodes)
            assert subset.shape[0] > 1, "empty subgraph!"
            nodeidx1 = subset.clone()
            nodeidx1.fill_(i)
            subgraphs.append(
                PygData(
                    x=dist,
                    subg_nodeidx=torch.stack((nodeidx1, subset), dim=-1),
                    num_nodes=subset.shape[0],
                ))
        else:
            subgraphs.append(
                PygData(
                    x=torch.zeros((1), dtype=torch.long),
                    subg_nodeidx=torch.tensor([[i, i]], dtype=torch.long),
                    num_nodes=1,
                ))
    subgbatch = PygBatch.from_data_list(subgraphs)
    return subgbatch.subg_nodeidx.t(), subgbatch.x


def KhopSampler(data: PygData, hop: int = 2) -> Tuple[LongTensor, LongTensor]:

    subgraphs = []

    for i in range(data.num_nodes):
        if True:
            subset, _, _, _, dist = k_hop_subgraph(i,
                                                   hop,
                                                   data.edge_index,
                                                   relabel_nodes=True,
                                                   num_nodes=data.num_nodes)
            assert subset.shape[0] > 1, "empty subgraph!"
            nodeidx1 = subset.clone()
            nodeidx1.fill_(i)
            subgraphs.append(
                PygData(
                    x=dist,
                    subg_nodeidx=torch.stack((nodeidx1, subset), dim=-1),
                    num_nodes=subset.shape[0],
                ))
    subgbatch = PygBatch.from_data_list(subgraphs)
    tupleid, tuplefeat = subgbatch.subg_nodeidx.t(), subgbatch.x
    tupleid, tuplefeat = coalesce(tupleid, tuplefeat, num_nodes=data.num_nodes, reduce="min")
    return tupleid, tuplefeat


def datapreprocess(data: PygData, subgsampler: Callable,
                   keys: List[str]) -> SubgData:
    data.edge_index, data.edge_attr = coalesce(data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
    tupleid, tuplefeat = subgsampler(data)
    datadict = {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
        "tupleid": tupleid,
        "tuplefeat": tuplefeat,
        "num_tuples": tupleid.shape[0]
    }
    for key in keys:
        if key == "AX_akl":
            datadict["AX_akl"] = filterij(
                tupleid, *spspmm_ind(data.edge_index, tupleid))
        elif key == "XA_akl":
            datadict["XA_akl"] = filterij(
                tupleid, *spspmm_ind(tupleid, data.edge_index))
        elif key == "XX_akl":
            datadict["XX_akl"] = filterij(tupleid,
                                          *spspmm_ind(tupleid, tupleid))
    return SubgData(**datadict)