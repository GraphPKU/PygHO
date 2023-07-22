from torch_geometric.data import Data as PygData, Batch as PygBatch
from torch_geometric.utils import to_networkx, k_hop_subgraph
import networkx as nx
import torch
from typing import List, Optional, Tuple, Union
from torch import Tensor, LongTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
import scipy.sparse as ssp
import numpy as np
import scipy.linalg as spl

def spdsampler(data: PygData, hop: int = 2) -> Tensor:
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    dist_matrix = ssp.csgraph.shortest_path(adj, directed=False, unweighted=True, return_predecessors=False)
    ret = torch.LongTensor(dist_matrix).flatten()
    ret.clamp_max_(hop)
    return ret.reshape(-1)


def rdsampler(data: PygData) -> Tensor:
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    laplacian = ssp.csgraph.laplacian(adj).toarray()
    laplacian += 0.01 * np.eye(*laplacian.shape)
    assert spl.issymmetric(laplacian), "please use symmetric graph"
    L_inv = np.linalg.pinv(laplacian, hermitian=True)
    dL = np.diagonal(L_inv)
    return torch.FloatTensor((dL.reshape(-1, 1) + dL.reshape(1, -1) - L_inv - L_inv.T)).reshape(-1, 1)