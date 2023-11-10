'''
utilities for sparse high order data
'''
from torch_geometric.data import Data as PygData, Batch as PygBatch
import torch
from typing import Any, List, Callable, Union, Tuple, Iterable
from torch import Tensor
from ..backend.Spspmm import spspmm_ind, filterind
from ..backend.SpTensor import SparseTensor
from ..honn.SpOperator import KEYSEP
from torch_geometric.utils import coalesce


def parseop(op: str):
    '''
    Get the increment for a tensor when combining graphs.

    Args:

    - op (str): The operator string.

    Returns:
    
    - str or NotImplementedError: The increment information or NotImplementedError if the operator is not implemented.
    '''
    if op[0] == "X":
        return f"num_tuples{op[1:]}"
    elif op == "A":
        return "num_edges"
    else:
        return NotImplementedError, f"operator name {op} not implemented now"


def parsekey(key: str) -> Tuple[str, str, int, str, int, int]:
    '''
    Parse the operators in precomputation keys.

    Args:
    
    - key (str): The precomputation key.

    Returns:
    
    - Tuple[str, str, int, str, int]: A tuple containing parsed operators and dimensions.
    '''
    assert len(key.split(KEYSEP)) == 6, "key format not match"
    op0, op1, dim1, op2, dim2, broadcast_dim = key.split(KEYSEP)
    dim1 = int(dim1)
    dim2 = int(dim2)
    broadcast_dim = int(broadcast_dim)
    parseop(op0)
    parseop(op1)
    parseop(op2)
    return op0, op1, dim1, op2, dim2, broadcast_dim


class SpHoData(PygData):
    '''
    A data class for sparse high order graph data.
    '''
    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key.startswith('tupleid'):
            return getattr(self,
                           "tupleshape" + key.removeprefix("tupleid")).reshape(
                               -1, 1)
        if key.endswith(f"{KEYSEP}acd"):
            key = key.removesuffix(f"{KEYSEP}acd")
            op0, op1, _, op2, _, _ = parsekey(key)
            return torch.tensor(
                [[getattr(self, parseop(op0))], [getattr(self, parseop(op1))],
                 [getattr(self, parseop(op2))]],
                dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key.startswith('tupleid') or key.endswith(f"{KEYSEP}acd"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def batch2sparse(batch: PygBatch, keys: List[str] = [""]) -> PygBatch:
    '''
    A main wrapper for converting data in a batch object to SparseTensor.

    Args:

    - batch (PygBatch): The batch object containing graph data.
    - keys (List[str]): The list of keys to convert to SparseTensor.

    Returns:
    
    - PygBatch: The batch object with converted data.
    '''
    batch.A = SparseTensor(
        batch.edge_index,
        batch.edge_attr,
        [batch.num_nodes, batch.num_nodes] if batch.edge_attr is None else
        [batch.num_nodes, batch.num_nodes] + list(batch.edge_attr.shape[1:]),
        is_coalesced=True)
    for key in keys:
        # print("key=", key)
        totaltupleshape = getattr(batch,
                                  f"tupleshape{key}").sum(dim=0).tolist()
        tupleid = getattr(batch, f"tupleid{key}")
        tuplefeat = getattr(batch, f"tuplefeat{key}")
        X = SparseTensor(
            tupleid,
            tuplefeat,
            shape=totaltupleshape if tuplefeat is None else totaltupleshape +
            list(tuplefeat.shape[1:]),
            is_coalesced=True)
        setattr(batch, f"X{key}", X)
    return batch


def sp_datapreprocess(data: PygData,
                      tuplesamplers: List[Callable[[PygData], SparseTensor|Iterable[SparseTensor]]],
                      annotate: List[str] = [""],
                      keys: List[str] = [""]) -> SpHoData:
    '''
    A wrapper for preprocessing dense data for sparse high order graphs.

    Args:
    
    - data (PygData): The input dense data in PyG Data format.
    - tuplesamplers (List[Callable]): A single or list of tuple sampling functions.
    - annotate (List[str]): A list of annotation strings for tuple sampling.
    - keys (List[str]): A list of precomputation keys.

    Returns:
    
    - SpHoData: The preprocessed sparse high order data in SpHoData format.
    '''
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               data.edge_attr,
                                               num_nodes=data.num_nodes)
    
    assert len(tuplesamplers) == len(
        annotate
    ), "number of sparse tensors should match the number of annotate"

    datadict = data.to_dict()
    datadict.update({
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
    })

    feats = [tuplesampler(data) for tuplesampler in tuplesamplers]
    feats = [list(_) if isinstance(_, Iterable) else [_] for _ in feats]
    feats = sum(feats, [])
    assert len(feats) == len(annotate), "number of sparse tensors should match the number of annotate"
    for i, feat in enumerate(feats):
        tupleid, tuplefeat, tupleshape = feat.indices, feat.values, feat.sparseshape 
        num_tuples = tupleid.shape[1]
        datadict.update({
            f"tupleid{annotate[i]}":
            tupleid,
            f"tuplefeat{annotate[i]}":
            tuplefeat,
            f"tupleshape{annotate[i]}":
            torch.LongTensor(tupleshape).reshape(1, -1),
            f"num_tuples{annotate[i]}":
            num_tuples
        })
    for key in keys:
        op0, op1, dim1, op2, dim2, broadcast_dim = parsekey(key)
        indop1 = datadict["edge_index"] if op1[0] == "A" else datadict[f"tupleid{op1[1:]}"]
        indop2 = datadict["edge_index"] if op2[0] == "A" else datadict[f"tupleid{op2[1:]}"]
        ind, acd = spspmm_ind(indop1, dim1, indop2, dim2, False, broadcast_dim)
        tarind = datadict["edge_index"] if op0[0] == "A" else datadict[f"tupleid{op0[1:]}"]
        datadict[key + f"{KEYSEP}acd"] = filterind(tarind, ind, acd)
    return SpHoData(**datadict)