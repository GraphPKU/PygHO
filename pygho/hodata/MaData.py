'''
utilities for dense high order data
'''
from torch_geometric.data import Data as PygData, Batch as PygBatch
import torch
from torch import Tensor, LongTensor, BoolTensor
from typing import Any, Callable, Optional, Tuple, List, Union, Iterable
from ..backend.SpTensor import SparseTensor
from ..backend.MaTensor import MaskedTensor
from torch_geometric.utils import coalesce
import torch


class MaHoData(PygData):
    '''
    a data class for dense high order graph data.
    '''

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key == 'edge_index':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def to_dense_adj(edge_index: LongTensor,
                 edge_batch: LongTensor,
                 edge_attr: Optional[Tensor] = None,
                 max_num_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 filled_value: float = 0) -> MaskedTensor:
    '''
    Convert sparse adjacency to dense matrix.

    Args:

    - edge_index (LongTensor): Coalesced edge indices of shape (2, nnz).
    - edge_batch (LongTensor): Batch assignments of shape (nnz).
    - edge_attr (Optional[Tensor]): Edge attributes of shape (nnz, \*).
    - max_num_nodes (Optional[int]): Maximum number of nodes in the graph.
    - batch_size (Optional[int]): Batch size.
    - filled_value (float): Value to fill in the dense matrix.

    Returns:

    - MaskedTensor: A masked dense tensor.

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
    mask = torch.zeros([batch_size, max_num_nodes, max_num_nodes],
                       device=ret.device,
                       dtype=torch.bool)
    mask[idx0, idx1, idx2] = True
    return MaskedTensor(ret, mask, filled_value, True)


def to_sparse_adj(edge_index: LongTensor,
                  edge_batch: LongTensor,
                  edge_attr: Optional[Tensor] = None,
                  max_num_nodes: Optional[int] = None,
                  batch_size: Optional[int] = None) -> SparseTensor:
    '''
    Convert sparse edge_index and edge_attr to a SparseTensor.

    Args:

    - edge_index (LongTensor): Coalesced edge indices of shape (2, nnz).
    - edge_batch (LongTensor): Batch assignments of shape (nnz).
    - edge_attr (Optional[Tensor]): Edge attributes of shape (nnz, \*).
    - max_num_nodes (Optional[int]): Maximum number of nodes in the graph.
    - batch_size (Optional[int]): Batch size.

    Returns:
    
    - SparseTensor: A sparse tensor representation.

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
               filled_value: float = 0) -> MaskedTensor:
    '''
    Convert node features of different subgraphs to a dense matrix.

    Args:

    - nodeX (Tensor): Node features. of shape (sum of number of nodes in a batch,\*denseshapeshape).
    - Xptr (LongTensor): Pointer to subgraphs. nodeX[Xptr[i]:Xptr[i+1]] represents the node feature for subgraph i
    - max_num_nodes (Optional[int]): Maximum number of nodes in a subgraph.
    - batch_size (Optional[int]): Batch size.
    - filled_value (float): Value to fill in the dense matrix.

    Returns:

    - MaskedTensor: A masked dense tensor.  of shape (b, n,\*denseshapeshape).

    To align graphs of different sizes, padding is applied.

    '''
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
    mask = mask.cummin(dim=-1)[0][:, :-1]
    return MaskedTensor(ret, mask, filled_value, False)


def to_dense_tuplefeat(
        tuplefeat: Tensor,
        tupleshape: LongTensor,
        tuplefeatptr: LongTensor,
        max_tupleshape: Optional[LongTensor] = None,
        batch_size: Optional[int] = None,
        feat2mask: Callable[[Tensor], BoolTensor] = None) -> MaskedTensor:
    '''
    Convert tuple features of different subgraphs to a dense matrix.

    Args:

    - tuplefeat (Tensor): Tuple features. (total number of tuples in batch,\*denseshapeshape)
    - tupleshape (LongTensor): Shape of tuple features.
    - tuplefeatptr (LongTensor): Pointer to tuple features. tuplefeat[tuplefeatptr[i]:tuplefeatptr[i+1]] represents the tuple feature for subgraph i
    - max_tupleshape (Optional[LongTensor]): Maximum shape of tuple features.
    - batch_size (Optional[int]): Batch size.
    - feat2mask (Callable[[Tensor], BoolTensor]): Function to generate masks for tuple features.

    Returns:

    - MaskedTensor: A masked dense tensor. of shape (b, n1, n2,..,\*denseshapeshape), whose ret[i] is of subgraph i. (n1, n2,...) is the maximum sizes of the tuplefeat of subgraphs.

    To align tuple features of different sizes, padding is applied.

    '''
    if batch_size is None:
        batch_size = tupleshape.shape[0]

    if max_tupleshape is None:
        max_tupleshape = torch.amax(tupleshape, dim=0)

    ndim = max_tupleshape.shape[0]
    fullidx = tuplefeatptr[:-1].reshape([-1] + [1] * ndim)
    cumshape = torch.ones_like(tupleshape[:, [0]])
    # print(cumshape.shape)
    for i in range(ndim):
        tidx = (torch.arange(max_tupleshape[-i - 1], device=tuplefeat.device) *
                cumshape).reshape([batch_size] + [1] * (ndim - i - 1) + [-1] +
                                  [1] * i)
        # print(fullidx.shape, tidx.shape, max_tupleshape, ndim)
        fullidx = fullidx + tidx
        cumshape = cumshape * tupleshape[:, [-i - 1]]
    fullidx.clamp_max_(tuplefeat.shape[0] - 1)
    ret = tuplefeat[fullidx]
    if feat2mask is not None:
        mask = feat2mask(ret)
    else:
        mask = torch.ones([batch_size] + max_tupleshape.tolist(),
                          device=ret.device,
                          dtype=torch.bool)

    for i in range(ndim):
        tmask = torch.ones([batch_size] + [max_tupleshape[i] + 1] + [1] *
                           (ndim - 1),
                           dtype=torch.bool,
                           device=ret.device)
        tmask[torch.arange(batch_size, device=ret.device),
              tupleshape[:, i]] = False
        tmask = torch.cummin(tmask, dim=1)[0]
        tmask = tmask[:, :-1]
        tmask = torch.movedim(tmask, 1, i + 1)
        mask.logical_and_(tmask)
    return MaskedTensor(ret, mask, 0, False)


def batch2dense(batch: PygBatch,
                batch_size: int = None,
                max_num_nodes: int = None,
                denseadj: bool = False,
                keys: List[str] = [""]) -> PygBatch:
    '''
    A main wrapper for converting and padding data in a batch object to dense forms.

    Args:

    - batch (PygBatch): The input batch object.
    - batch_size (int): Batch size.
    - max_num_nodes (int): Maximum number of nodes in the graph.
    - denseadj (bool): Whether to convert adjacency to dense or sparse.
    - keys (List[str]): List of keys for additional attributes.

    Returns:
  
    - PygBatch: The processed batch object.

    '''

    batch.x = to_dense_x(batch.x, batch.ptr, max_num_nodes, batch_size)
    batch_size, max_num_nodes = batch.x.shape[0], batch.x.shape[1]
    if denseadj:
        batch.A = to_dense_adj(batch.edge_index, batch.edge_index_batch,
                               batch.edge_attr, max_num_nodes, batch_size)
    else:
        batch.A = to_sparse_adj(batch.edge_index, batch.edge_index_batch,
                                batch.edge_attr, max_num_nodes, batch_size)
    for key in keys:
        tuplefeat = getattr(batch, f"tuplefeat{key}")
        tupleshape = getattr(batch, f"tupleshape{key}")
        tuplefeat_ptr = getattr(batch, f"tuplefeat{key}_ptr")
        X = to_dense_tuplefeat(tuplefeat, tupleshape, tuplefeat_ptr, None,
                               batch_size, None)
        setattr(batch, f"X{key}", X)
    return batch


def ma_datapreprocess(data: PygData,
                      tuplesamplers: List[Callable[[PygData],
                                                   Tuple[Tensor, List[int]]]],
                      annotate: List[str] = [""]) -> MaHoData:
    '''
    A wrapper for preprocessing dense data.

    Args:

    - data (PygData): Input data object.
    - tuplesamplers (Union[Callable[[PygData], Tuple[Tensor, List[int]]], List[Callable[[PygData], Tuple[Tensor, List[int]]]]]): Tuple samplers for extracting data.
    - annotate (List[str]): List of annotation strings.

    Returns:
    
    - MaHoData: Preprocessed data object.

    '''
    assert len(tuplesamplers) == len(
        annotate), "each tuplesampler need a different annotate"
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               data.edge_attr,
                                               num_nodes=data.num_nodes)

    datadict = data.to_dict()
    datadict.update({
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr
    })
    for i, tuplesampler in enumerate(tuplesamplers):
        tuplefeat, tupleshape = tuplesampler(data)
        datadict.update({
            f"tuplefeat{annotate[i]}":
            tuplefeat,
            f"tupleshape{annotate[i]}":
            torch.LongTensor(tupleshape).reshape(1, -1),
        })

    return MaHoData(**datadict)
