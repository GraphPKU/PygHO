'''
transform for sparse data
'''
from torch_geometric.data import Data as PygData
import torch
from typing import Any, List, Callable
from ..backend.Spspmm import spspmm_ind, filterind
from torch_geometric.utils import coalesce


def parsekey(key: str):
    if key.endswith("_acd"):
        assert len(key.split("_")) == 5, "key format not match"
        op1, dim1, op2, dim2 = key.split("_")[:4]
        dim1 = int(dim1)
        dim2 = int(dim2)
        assert op1 in ["X", "A"], f"invalid op1 {op1}"
        assert op2 in ["X", "A"], f"invalid op2 {op2}"
        return op1, dim1, op2, dim2
    else:
        raise NotImplementedError


class SpSubgData(PygData):

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key == 'tupleid':
            return self.num_nodes
        if key.endswith("_acd"):
            op1, _, op2, _ = parsekey(key)
            return torch.tensor(
                [[self.num_tuples],
                 [self.num_edges if op1 == "A" else self.num_tuples],
                 [self.num_edges if op2 == "A" else self.num_tuples]],
                dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'tupleid' or key.endswith("_acd"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def sp_datapreprocess(data: PygData, subgsampler: Callable,
                      keys: List[str]) -> SpSubgData:
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               data.edge_attr,
                                               num_nodes=data.num_nodes)
    tupleid, tuplefeat = subgsampler(data)
    '''
    (#sparsedim, #nnz), (#nnz, *)
    '''
    datadict = data.to_dict()
    datadict.update({
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
        "tupleid": tupleid,
        "tuplefeat": tuplefeat,
        "num_tuples": tupleid.shape[1]
    })
    for key in keys:
        op1, dim1, op2, dim2 = parsekey(key)
        datadict[key] = filterind(
            datadict["tupleid"],
            *spspmm_ind(datadict["tupleid"] if op1 == "X" else datadict["edge_index"],
                        dim1,
                        datadict["tupleid"] if op2 == "X" else datadict["edge_index"],
                        dim2))
    return SpSubgData(**datadict)