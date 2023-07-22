import torch
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset, QM9
from ogb.graphproppred import PygGraphPropPredDataset
from subgdata.MaData import ma_datapreprocess
from subgdata.MaSubgSampler import rdsampler, spdsampler
import torch_geometric.transforms as T
from torchmetrics import Accuracy, MeanAbsoluteError
from functools import partial
from datasets import SRDataset, PlanarSATPairsDataset, GraphCountDataset, myEvaluator

def loaddataset(name: str, **kwargs): #-> Iterable[Dataset], str, Callable, str
    kwargs["transform"] = partial(ma_datapreprocess, subgsampler=partial(spdsampler, hop=4))
    if name == "sr":
        dataset = SRDataset(**kwargs)
        # dataset = dataset[:2]
        dataset.data.x = dataset.data.x.long()
        dataset.data.y = torch.arange(len(dataset))
        dataset.num_tasks = torch.max(dataset.data.y).item() + 1
        return (dataset, dataset, dataset), "fixed", Accuracy("multiclass", num_classes=dataset.num_tasks), "cls" # full training/valid/test??
    elif name == "EXP":
        dataset = PlanarSATPairsDataset(**kwargs)
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
        kwargs["pre_transform"] = T.Compose([CSL_node_feature_transform, kwargs["pre_transform"]])
        dataset = GNNBenchmarkDataset("dataset", "CSL", **kwargs)
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
        def TU_pretransform(data):
            data.y = data.y.reshape(-1, 1).to(torch.float)
            return data
        kwargs["pre_transform"] = TU_pretransform
        dataset = TUDataset("dataset", name=name, **kwargs)
        dataset.num_tasks = 1
        return (dataset,), "fold-9-0-1", Accuracy("binary"), "bincls"
    elif name == "zinc":
        def ZINC_pretransform(data):
            data.edge_attr = data.edge_attr.reshape(-1, 1).to(torch.long)
            data.y = data.y.reshape(-1, 1)
            return data
        kwargs["pre_transform"] = ZINC_pretransform, kwargs["pre_transform"]
        trn_d = ZINC("dataset/ZINC", subset=True, split="train", **kwargs)
        val_d = ZINC("dataset/ZINC", subset=True, split="val", **kwargs)
        tst_d = ZINC("dataset/ZINC", subset=True, split="test", **kwargs)
        trn_d.num_tasks = 1
        val_d.num_tasks = 1
        tst_d.num_tasks = 1
        return (trn_d, val_d, tst_d), "fixed", MeanAbsoluteError(), "smoothl1reg" #"reg"
    elif name == "QM9":
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
        dataset = PygGraphPropPredDataset(name=name, **kwargs)
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
    datalist = ["sr", "EXP", "CSL", "subgcount0", "zinc", "MUTAG", "DD", "PROTEINS", "ogbg-molhiv", "ogbg-molpcba"] # "QM9",  "IMDB-BINARY",
    for ds in datalist:
        datasets = loaddataset(ds)[0]
        dataset = datasets[0]
        data = dataset[0]
        print(ds, dataset.num_tasks, data, data.x.dtype, data.y.dtype)