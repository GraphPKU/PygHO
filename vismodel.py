from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import UniAnchorGNN
import argparse
import time
import numpy as np
from main_uni import train, eval, parserarg, buildModel

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.MSELoss(reduction="none")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parserarg()
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
    
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    outs = []
    for rep in range(1):
        visloader = DataLoader(dataset[split_idx["train"]],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=args.num_workers)
        print(
            f"split {split_idx['train'].shape[0]} {split_idx['valid'].shape[0]} {split_idx['test'].shape[0]}"
        )
        model = buildModel(args, dataset, device)
        from torch_scatter import scatter_std, scatter_max, scatter_min, scatter_mean
        if args.load is not None:
            model.load_state_dict(torch.load(f"mod/{args.load}.{rep}.pt", map_location=device))
        model.vis = True
        model.eval()
        import igraph
        probdiff = []
        match_hnode = []
        with torch.no_grad():
            for batch in visloader:
                batch = batch.to(device)
                vis = model(batch)
                prob = vis["prob0"]
                batch = vis["input"].batch
                print((scatter_min(vis[f"match_h_node{0}"].square().sum(dim=-1, keepdim=True), batch, dim=-2)[0]<1e-15).float().mean())
                #print(scatter_std(vis[f"match_h_node{0}"], batch, dim=-2).mean())
                probdiff.append(scatter_max(prob, batch)[0] - scatter_min(prob, batch)[0])
        probdiff = torch.cat(probdiff)
        print(torch.mean(probdiff))

        


if __name__ == "__main__":
    main()