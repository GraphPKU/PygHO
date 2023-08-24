'''
transform for dense data
'''
from torch_geometric.data import Data as PygData, Batch as PygBatch
import torch
from torch import Tensor, LongTensor, BoolTensor
from typing import Any, Callable, Optional, Tuple
from ..backend.SpTensor import SparseTensor
from torch_geometric.utils import coalesce
import torch


class MaSubgData(PygData):

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key == 'edge_index':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def to_dense_adj(edge_index: LongTensor,
                 edge_batch: LongTensor,
                 edge_attr: Optional[Tensor] = None,
                 max_num_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 filled_value: float = 0) -> Tensor:
    '''
    edge_index: coalesced, (2, nnz)
    edge_batch: (nnz)
    edge_attr: (nnz, *)
    '''
    idx0 = edge_batch
    idx1 = edge_index[0]
    idx2 = edge_index[1]

    if max_num_nodes is None:
        max_num_nodes = edge_index.max().item() + 1

    if edge_attr is None:
        edge_attr = torch.ones(idx0.shape[0], device=edge_index.device)

    if batch_size is None:
        batch_size = torch.max(edge_batch).item() + 1

    size = [batch_size, max_num_nodes, max_num_nodes] + list(
        edge_attr.shape)[1:]
    ret = torch.empty(size, dtype=edge_attr.dtype, device=edge_attr.device)
    ret.fill_(filled_value)
    ret[idx0, idx1, idx2] = edge_attr
    return ret


def to_sparse_adj(edge_index: LongTensor,
                  edge_batch: LongTensor,
                  edge_attr: Optional[Tensor] = None,
                  max_num_nodes: Optional[int] = None,
                  batch_size: Optional[int] = None) -> SparseTensor:
    '''
    edge_index: coalesced, (2, nnz)
    edge_batch: (nnz)
    edge_attr: (nnz, *)
    '''
    if max_num_nodes is None:
        max_num_nodes = edge_index.max().item() + 1

    if batch_size is None:
        batch_size = torch.max(edge_batch).item() + 1

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    return SparseTensor(torch.concatenate(
        (edge_batch.unsqueeze(0), edge_index), dim=0),
                        edge_attr,
                        shape=size,
                        is_coalesced=False)


def to_dense_x(nodeX: Tensor,
               Xptr: LongTensor,
               max_num_nodes: Optional[int] = None,
               batch_size: Optional[int] = None,
               filled_value: float = 0) -> Tuple[Tensor, BoolTensor]:

    if batch_size is None:
        batch_size = Xptr.shape[0] - 1

    if max_num_nodes is None:
        max_num_nodes = torch.diff(Xptr).max().item()

    idx = torch.arange(max_num_nodes, device=nodeX.device).unsqueeze(0)
    idx = idx + Xptr[:-1].reshape(-1, 1)
    idx.clamp_max_(Xptr[-1] - 1)

    ret = nodeX[idx]
    mask = torch.ones((batch_size, max_num_nodes + 1),
                      dtype=torch.bool,
                      device=nodeX.device)
    mask[torch.arange(batch_size, device=nodeX.device),
         torch.diff(Xptr)] = False
    mask = mask.cummin(dim=-1)[0]
    return ret, mask[:, :-1]


def to_dense_tuplefeat(tuplefeat: Tensor,
                       Xptr: LongTensor,
                       tuplefeatptr: LongTensor,
                       max_num_nodes: Optional[int] = None,
                       batch_size: Optional[int] = None) -> Tensor:
    if batch_size is None:
        batch_size = Xptr.shape[0] - 1

    if max_num_nodes is None:
        max_num_nodes = torch.diff(Xptr).max().item()

    num_nodes = torch.diff(Xptr)

    singleidx = torch.arange((max_num_nodes), device=tuplefeat.device)
    fullidx = tuplefeatptr[:-1].reshape(-1, 1, 1) + singleidx.reshape(
        1, -1, 1) * num_nodes.reshape(-1, 1, 1) + singleidx.reshape(1, 1, -1)
    fullidx.clamp_max_(tuplefeat.shape[0] - 1)
    ret = tuplefeat[fullidx]
    return ret


def batch2dense(batch: PygBatch,
                batch_size: int = None,
                max_num_nodes: int = None,
                denseadj: bool = False):
    # print(batch[1].x, batch[1].tuplefeat)
    x, nodemask = to_dense_x(batch.x, batch.ptr, max_num_nodes,
                             batch_size)
    batch.x, batch.nodemask = x, nodemask
    batch_size, max_num_nodes = x.shape[0], x.shape[1]
    if denseadj:
        batch.A = to_dense_adj(batch.edge_index,
                                  batch.edge_index_batch,
                                  batch.edge_attr, max_num_nodes,
                                  batch_size)
    else:
        batch.A = to_sparse_adj(batch.edge_index,
                                   batch.edge_index_batch,
                                   batch.edge_attr, max_num_nodes,
                                   batch_size)
    batch.tuplefeat = to_dense_tuplefeat(batch.tuplefeat, batch.ptr,
                                            batch.tuplefeat_ptr,
                                            max_num_nodes, batch_size)
    batch.tuplemask = torch.logical_and(nodemask.unsqueeze(1),
                                           nodemask.unsqueeze(2))
    return batch


def ma_datapreprocess(data: PygData, subgsampler: Callable) -> MaSubgData:
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               data.edge_attr,
                                               num_nodes=data.num_nodes)
    tuplefeat = subgsampler(data)
    datadict = data.to_dict()
    datadict.update({
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
        "tuplefeat": tuplefeat,
    })
    return MaSubgData(**datadict)
    
