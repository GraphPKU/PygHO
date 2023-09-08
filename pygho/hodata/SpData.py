'''
transform for sparse data
'''
from torch_geometric.data import Data as PygData, Batch as PygBatch
import torch
from typing import Any, List, Callable, Union, Tuple, Iterable
from torch import Tensor
from ..backend.Spspmm import spspmm_ind, filterind
from ..backend.SpTensor import SparseTensor
from ..honn.SpXOperator import KEYSEP
from torch_geometric.utils import coalesce


def parseop(op: str):
    if op[0] == "X":
        return f"num_tuples{op[1:]}"
    elif op == "A":
        return "num_edges"
    else:
        return NotImplementedError, f"operator name {op} not implemented now"


def parsekey(key: str) -> Tuple[str, str, int, str, int]:
    assert len(key.split(KEYSEP)) == 5, "key format not match"
    op0, op1, dim1, op2, dim2 = key.split(KEYSEP)
    dim1 = int(dim1)
    dim2 = int(dim2)
    parseop(op0)
    parseop(op1)
    parseop(op2)
    return op0, op1, dim1, op2, dim2


class SpHoData(PygData):

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key.startswith('tupleid'):
            return getattr(self,
                           "tupleshape" + key.removeprefix("tupleid")).reshape(
                               -1, 1)
        if key.endswith(f"{KEYSEP}acd"):
            key = key.removesuffix(f"{KEYSEP}acd")
            op0, op1, _, op2, _ = parsekey(key)
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
                      tuplesamplers: Union[Callable[[PygData],
                                                    Tuple[Tensor, Tensor,
                                                          Union[List[int],
                                                                int]]],
                                           List[Callable[[PygData],
                                                         Tuple[Tensor, Tensor,
                                                               Union[List[int],
                                                                     int]]]]],
                      annotate: List[str] = [""],
                      keys: List[str] = [""]) -> SpHoData:
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               data.edge_attr,
                                               num_nodes=data.num_nodes)
    if not isinstance(tuplesamplers, Iterable):
        tuplesamplers = [tuplesamplers]
    assert len(tuplesamplers) == len(
        annotate
    ), "number of tuple sampler should match the number of annotate"

    datadict = data.to_dict()
    datadict.update({
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
    })
    for i, tuplesampler in enumerate(tuplesamplers):
        tupleid, tuplefeat, tupleshape = tuplesampler(data)
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
        op0, op1, dim1, op2, dim2 = parsekey(key)
        datadict[key + f"{KEYSEP}acd"] = filterind(
            datadict[f"tupleid{op0[1:]}"]
            if op0[0] == "X" else datadict["edge_index"],
            *spspmm_ind(
                datadict[f"tupleid{op1[1:]}"] if op1[0] == "X" else
                datadict["edge_index"], dim1, datadict[f"tupleid{op2[1:]}"]
                if op2[0] == "X" else datadict["edge_index"], dim2))
    return SpHoData(**datadict)