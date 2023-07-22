'''
transform for sparse data
'''
from torch_geometric.data import Data as PygData
import torch
from typing import Any, List, Callable
from backend.Spspmm import spspmm_ind, filterij
from torch_geometric.utils import coalesce


class SpSubgData(PygData):

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        if key == 'tupleid':
            return self.num_nodes
        if key == 'AX_acd':
            return torch.tensor(
                [[self.num_tuples], [self.num_edges], [self.num_tuples]],
                dtype=torch.long)
        if key == 'XA_acd':
            return torch.tensor(
                [[self.num_tuples], [self.num_tuples], [self.num_edges]],
                dtype=torch.long)
        if key == 'XX_acd':
            return torch.tensor(
                [[self.num_tuples], [self.num_tuples], [self.num_tuples]],
                dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'tupleid':
            return 1
        if key == 'AX_acd':
            return 1
        if key == 'XA_acd':
            return 1
        if key == 'XX_acd':
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def sp_datapreprocess(data: PygData, subgsampler: Callable,
                      keys: List[str]) -> SpSubgData:
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               data.edge_attr,
                                               num_nodes=data.num_nodes)
    tupleid, tuplefeat = subgsampler(data)
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
        if key == "AX_acd":
            datadict["AX_acd"] = filterij(
                tupleid, *spspmm_ind(data.edge_index, tupleid))
        elif key == "XA_acd":
            datadict["XA_acd"] = filterij(
                tupleid, *spspmm_ind(tupleid, data.edge_index))
        elif key == "XX_acd":
            datadict["XX_acd"] = filterij(tupleid,
                                          *spspmm_ind(tupleid, tupleid))
        else:
            raise NotImplementedError
    return SpSubgData(**datadict)