'''
Copied from https://github.com/JiaruiFeng/KP-GNN/tree/a127847ed8aa2955f758476225bc27c6697e7733
'''
from sklearn.metrics import accuracy_score
import torch
import pickle
import numpy as np
import scipy.io as sio
from scipy.special import comb
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset, QM9
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


class PlanarSATPairsDataset(InMemoryDataset):

    def __init__(self,
                 root="dataset/EXP",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform,
                                                    pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT" + ".pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        with open("dataset/EXP/raw/" + "newGRAPHSAT" + ".pkl", "rb") as f:
            data_list = pickle.load(f)
        data_list = [Data.from_dict(_) for _ in data_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def EXP_node_feature_transform(data):
    data.x = data.x[:, 0].to(torch.long)
    return data


class GraphCountDataset(InMemoryDataset):
    def __init__(self, root="dataset/subgraphcount", transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a = sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0])
        self.test_idx = torch.from_numpy(a['test_idx'][0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        b = self.processed_paths[0]
        a = sio.loadmat(self.raw_paths[0])  # 'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A = a['A'][0]
        # list of output
        Y = a['F']

        data_list = []
        for i in range(len(A)):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            tri = np.trace(A3) / 6
            tailed = ((np.diag(A3) / 2) * (a.sum(0) - 2)).sum()
            cyc4 = 1 / 8 * (np.trace(A3.dot(a)) + np.trace(A2) - 2 * A2.sum())
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += comb(int(deg[j]), 3)

            expy = torch.tensor([[tri, tailed, star, cyc4, cus]])

            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1).long()  # change to category
            # y=torch.tensor(Y[i:i+1,:])
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SRDataset(InMemoryDataset):
    def __init__(self, root="dataset/sr25", transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  # sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))
            data_list.append(Data(edge_index=edge_index, x=x, y=0))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class myEvaluator(Evaluator):
    def __init__(self, name):
        super().__init__(name=name)
    
    def __call__(self, y_pred, y_true):
        ret = super().eval({"y_pred": y_pred, "y_true": y_true})
        assert len(ret) == 1
        return list(ret.values())[0]

from torchmetrics import Accuracy, MeanAbsoluteError
from typing import Iterable, Callable, Optional, Tuple
from torch_geometric.data import Dataset

def loaddataset(name: str, **kwargs): #-> Iterable[Dataset], str, Callable, str
    if name == "sr":
        dataset = SRDataset(**kwargs)
        # dataset = dataset[:2]
        dataset.data.x = dataset.data.x.long()
        dataset.data.y = torch.arange(len(dataset))
        dataset.num_tasks = torch.max(dataset.data.y).item() + 1
        return (dataset, dataset, dataset), "fixed", Accuracy("multiclass", num_classes=dataset.num_tasks), "cls" # full training/valid/test??
    elif name == "EXP":
        dataset = PlanarSATPairsDataset(pre_transform=EXP_node_feature_transform, **kwargs)
        if dataset.data.x.dim() == 1:
            dataset.data.x = dataset.data.x.unsqueeze(-1)
        dataset.num_tasks = 1
        dataset.data.y = dataset.data.y.to(torch.float).reshape(-1, 1)
        return (dataset,),  "fold-8-1-1", Accuracy("binary"), "bincls"
    elif name == "CSL":
        def CSL_node_feature_transform(data):
            if "x" not in data:
                data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
            return data
        dataset = GNNBenchmarkDataset("dataset", "CSL", pre_transform=CSL_node_feature_transform,**kwargs)
        dataset.num_tasks = torch.max(dataset.data.y).item() + 1
        return (dataset,), "fold-8-1-1", Accuracy("multiclass", num_classes=10), "cls"
    elif name.startswith("subgcount"):
        y_slice = int(name[len("subgcount"):])
        dataset = GraphCountDataset(**kwargs)
        from torch_geometric.utils import degree
        dataset.data.y = dataset.data.y - dataset.data.y.mean(dim=0)
        dataset.data.y = dataset.data.y/dataset.data.y.std(dim=0) 
        # normalize as https://github.com/JiaruiFeng/KP-GNN/blob/main/train_structure_counting.py line 203
        dataset.data.y = dataset.data.y[:, [y_slice]]
        # degree feature
        # dataset.data.x.copy_(torch.cat([degree(dat.edge_index[0], num_nodes=dat.num_nodes, dtype=torch.long) for dat in dataset]).reshape(-1, 1))
        dataset.num_tasks = 1
        dataset.data.y = dataset.data.y.to(torch.float)
        return (dataset[dataset.train_idx], dataset[dataset.val_idx], dataset[dataset.test_idx]), "fixed", MeanAbsoluteError(), "l1reg" # 
    elif name in ["MUTAG", "DD", "PROTEINS", "PTC-MR", "IMDB-BINARY"]:
        dataset = TUDataset("dataset", name=name, **kwargs)
        dataset.num_tasks = 1
        dataset.data.y = dataset.data.y.to(torch.float)
        return (dataset,), "fold-9-0-1", Accuracy("binary"), "bincls"
    elif name == "zinc":
        trn_d = ZINC("dataset/ZINC", subset=True, split="train", **kwargs)
        val_d = ZINC("dataset/ZINC", subset=True, split="val")
        tst_d = ZINC("dataset/ZINC", subset=True, split="test")
        trn_d.num_tasks = 1
        trn_d.data.edge_attr = trn_d.data.edge_attr.reshape(-1, 1).to(torch.long)
        trn_d.data.y = trn_d.data.y.reshape(-1, 1)
        val_d.num_tasks = 1
        val_d.data.edge_attr = val_d.data.edge_attr.reshape(-1, 1).to(torch.long)
        val_d.data.y = val_d.data.y.reshape(-1, 1)
        tst_d.num_tasks = 1
        tst_d.data.edge_attr = tst_d.data.edge_attr.reshape(-1, 1).to(torch.long)
        tst_d.data.y = tst_d.data.y.reshape(-1, 1)
        return (trn_d, val_d, tst_d), "fixed", MeanAbsoluteError(), "reg"
    elif name == "QM9":
        raise NotImplementedError
        dataset = QM9("dataset/qm9", **kwargs)
        dataset.data.y = dataset.data.y[:, y_slice]
        dataset.num_tasks = 1
        dataset = dataset.shuffle()

        # Normalize targets to mean = 0 and std = 1.
        tenpercent = int(len(dataset) * 0.1)
        mean = dataset.data.y[tenpercent:].mean(dim=0)
        std = dataset.data.y[tenpercent:].std(dim=0)
        dataset.data.y = (dataset.data.y - mean) / std

        train_dataset = dataset[2 * tenpercent:]
        '''
        if kwargs["normalize_x"]:
            x_mean = train_dataset.data.x[:, cont_feat_start_dim:].mean(dim=0)
            x_std = train_dataset.data.x[:, cont_feat_start_dim:].std(dim=0)
            x_norm = (train_dataset.data.x[:, cont_feat_start_dim:] - x_mean) / x_std
            dataset.data.x = torch.cat([dataset.data.x[:, :cont_feat_start_dim], x_norm], 1)
        '''
        test_dataset = dataset[:tenpercent]
        val_dataset = dataset[tenpercent:2 * tenpercent]
        train_dataset = dataset[2 * tenpercent:]
        return (train_dataset, val_dataset, test_dataset), "8-1-1", MeanAbsoluteError(), "reg"
    elif name.startswith("ogbg"):
        dataset = PygGraphPropPredDataset(name=name)
        split_idx = dataset.get_idx_split()
        if "molhiv" in name:
            task = "bincls"
        elif "pcba"  in name:
            task = "bincls"
        else:
            raise NotImplementedError
        dataset.data.y = dataset.data.y.to(torch.float)
        return (dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]), "fixed", myEvaluator(name), task
    else:
        raise NotImplementedError(name)

if __name__ == "__main__":
    datalist = ["sr", "EXP", "CSL", "subgcount", "zinc", "MUTAG", "DD", "PROTEINS", "ogbg-molhiv", "ogbg-molpcba"] # "QM9",  "IMDB-BINARY",
    for ds in datalist:
        datasets = loaddataset(ds)[0]
        dataset = datasets[0]
        data = dataset[0]
        print(ds, dataset.num_tasks, data, data.x.dtype, data.y.dtype)