
import unittest
from pygho import MaskedTensor, SparseTensor
import pygho as MaTensor
import pygho.subgdata.MaData as MaData
import pygho.subgdata.SpData as SpData 
import torch


EPS = 1e-5

def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max((a-b).abs()).item()

def tensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.all(a==b).item()

def floattensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return maxdiff(a, b) < EPS

class MaDataTest(unittest.TestCase):
    def setUp(self) -> None:
        num_nodes1 = 2
        num_edges1 = 3
        edge_index1 = torch.randint(num_nodes1, size=(2, num_edges1))
        edge_attr1 = torch.arange(num_edges1)
        x1 = torch.arange(num_nodes1)
        tuplefeat1 = torch.ones((num_nodes1, num_nodes1)).flatten()
        data1 = MaData.MaSubgData(x=x1,
                        tuplefeat=tuplefeat1,
                        edge_index=edge_index1,
                        edge_attr=edge_attr1,
                        num_nodes=num_nodes1)
        num_nodes2 = 5
        num_edges2 = 7
        edge_index2 = torch.randint(num_nodes2, size=(2, num_edges2))
        edge_attr2 = torch.arange(num_edges2)
        x2 = torch.arange(num_nodes2)
        tuplefeat2 = 2 * torch.ones((num_nodes2, num_nodes2)).flatten()
        data2 = MaData.MaSubgData(x=x2,
                        tuplefeat=tuplefeat2,
                        edge_index=edge_index2,
                        edge_attr=edge_attr2,
                        num_nodes=num_nodes2)
        from torch_geometric.data import Batch as PygBatch
        batch = PygBatch.from_data_list([data1, data2],
                                        follow_batch=["edge_index", "tuplefeat"])
        datadict = batch.to_dict()
        datadict = MaData.batch2dense(datadict)
        print(data1, data2)
        print(datadict["tuplefeat"], datadict["x"], datadict["A"],
            datadict["nodemask"], datadict["tuplemask"])

class SpDataTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()