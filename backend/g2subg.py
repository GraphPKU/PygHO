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
        if key == 'subgid':
            raise self.num_nodes
        if key == 'AX_inda':
            return self.num_tuples
        if key == 'AX_indk':
            return self.num_edges
        if key == 'AX_indl':
            return self.num_tuples
        if key == 'XA_inda':
            return self.num_tuples
        if key == 'XA_indk':
            return self.num_tuples
        if key == 'XA_indl':
            return self.num_edges
        if key == 'XX_indl':
            return self.num_tuples
        if key == 'XX_inda':
            return self.num_tuples
        if key == 'XX_indk':
            return self.num_tuples
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

def CycleCenteredKhopSampler(data: PygData, hop: int=2):
    ng = to_networkx(PygData(edge_index=data.edge_index), to_undirected=True)
    cynodes = set(sum(nx.cycle_basis(ng), start=[]))

    subgraphs = []

    for i in range(data.num_nodes):
        if i in cynodes:
            subset, _, _, _, dist = k_hop_subgraph(i, hop, data.edge_index, relabel_nodes=True,
                                                num_nodes=data.num_nodes)
            assert subset.shape[0] > 1, "empty subgraph!"
            subg_nodelabel = dist
            nodeidx1 = subset.clone()
            nodeidx1.fill_(i)
            subgraphs.append(
                PygData(
                    x=subg_nodelabel, subg_nodeidx=torch.stack(nodeidx1), num_nodes=subset.shape[0],
                )
            )
        else:
            subgraphs.append(
                PygData(
                    x=torch.zeros((1), dtype=torch.long), subg_nodeidx=torch.tensor([[i], [i]], dtype=torch.long), num_nodes=1,
                )
            )
    subgbatch = PygBatch.from_data_list(subgraphs)    
    return subgbatch.subg_nodeidx, subgbatch.x