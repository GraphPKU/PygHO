from torch_geometric.data import Data as PygData, Batch as PygBatch
from torch_geometric.utils import to_scipy_sparse_matrix, k_hop_subgraph
import torch
from typing import List, Optional, Tuple, Union
from torch import Tensor, LongTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple
import scipy.sparse as ssp
from ..backend.SpTensor import coalesce, SparseTensor


def k_hop_subgraph(
    node_idx: Union[int, List[int], LongTensor],
    num_hops: int,
    edge_index: LongTensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the k-hop subgraph around a set of nodes in an edge list.

    Args:
    
    - node_idx (Union[int, List[int], LongTensor]): The root node(s) for the subgraph.
    - num_hops (int): The number of hops for the subgraph.
    - edge_index (LongTensor): The edge indices of the graph.
    - relabel_nodes (bool, optional): Whether to relabel node indices. Defaults to False.
    - num_nodes (Optional[int], optional): The total number of nodes. Defaults to None.
    - flow (str, optional): The direction of traversal ('source_to_target' or 'target_to_source'). Defaults to 'source_to_target'.
    - directed (bool, optional): Whether the graph is directed. Defaults to False.

    Returns:
    
        Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
            - subset (Tensor): The node indices in the subgraph.
            - edge_index (Tensor): The edge indices of the subgraph.
            - inv (Tensor): The inverse mapping of node indices in the original graph to the subgraph.
            - edge_mask (Tensor): A mask indicating which edges are part of the subgraph.
            - dist (Tensor): A distance of each node to the root node.
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


def KhopSampler(
        data: PygData,
        hop: int = 2) -> SparseTensor:
    """
    sample k-hop subgraph on a given PyG graph.

    Args:
    
    - data (PygData): The input PyG dataset.
    - hop (int, optional): The number of hops for subgraph sampling. Defaults to 2.

    Returns:
    
        SparseTensor for the precomputed tuple features.
    """

    subgraphs = []

    for i in range(data.num_nodes):
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
    return SparseTensor(tupleid, tuplefeat, shape=2*[data.num_nodes]+list(tuplefeat.shape[1:]), is_coalesced=False, reduce="min")


def I2Sampler(
        data: PygData,
        hop: int = 3) -> SparseTensor:
    """
    Perform subgraph sampling on a given graph for I2GNN.

    Args:
    
    - data (PygData): The input PyG dataset.
    - hop (int, optional): The number of hops for subgraph sampling. Defaults to 3.

    Returns:
    
        SparseTensor for the precomputed tuple features.
    """
    subgraphs = []
    spadj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    dist_matrix = torch.from_numpy(
        ssp.csgraph.shortest_path(spadj,
                                  directed=False,
                                  unweighted=True,
                                  return_predecessors=False)).to(torch.long)
    ei = data.edge_index
    for i in range(ei.shape[1]):
        nodepair = ei[:, i]
        subset, _, _, _, _ = k_hop_subgraph(nodepair,
                                            hop,
                                            data.edge_index,
                                            relabel_nodes=True,
                                            num_nodes=data.num_nodes)
        assert subset.shape[0] > 1, "empty subgraph!"
        nodeidx1 = subset.clone()
        nodeidx1.fill_(nodepair[0])
        nodeidx2 = subset.clone()
        nodeidx2.fill_(nodepair[1])
        subgraphs.append(
            PygData(
                x=torch.stack((dist_matrix[nodepair[0].item()][subset],
                               dist_matrix[nodepair[1].item()][subset]),
                              dim=-1),
                subg_nodeidx=torch.stack((nodeidx1, nodeidx2, subset), dim=-1),
                num_nodes=subset.shape[0],
            ))
    subgbatch = PygBatch.from_data_list(subgraphs)
    tupleid, tuplefeat = subgbatch.subg_nodeidx.t(), subgbatch.x
    return SparseTensor(tupleid, tuplefeat, shape=3*[data.num_nodes]+list(tuplefeat.shape[1:]), is_coalesced=False, reduce="min")