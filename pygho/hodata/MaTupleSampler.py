from torch_geometric.data import Data as PygData
import torch
from torch import Tensor
from typing import Tuple, List
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as ssp
import numpy as np
import scipy.linalg as spl


def spdsampler(data: PygData, hop: int = 2) -> Tuple[Tensor, List[int]]:
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    dist_matrix = ssp.csgraph.shortest_path(adj,
                                            directed=False,
                                            unweighted=True,
                                            return_predecessors=False)
    ret = torch.LongTensor(dist_matrix).flatten()
    ret.clamp_max_(hop)
    return ret.reshape(-1), [data.num_nodes, data.num_nodes]


def rdsampler(data: PygData) -> Tuple[Tensor, List[int]]:
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    laplacian = ssp.csgraph.laplacian(adj).toarray()
    laplacian += 0.01 * np.eye(*laplacian.shape)
    assert spl.issymmetric(laplacian), "please use symmetric graph"
    L_inv = np.linalg.pinv(laplacian, hermitian=True)
    dL = np.diagonal(L_inv)
    return torch.FloatTensor(
        (dL.reshape(-1, 1) + dL.reshape(1, -1) - L_inv - L_inv.T)).reshape(
            -1, 1), [data.num_nodes, data.num_nodes]
