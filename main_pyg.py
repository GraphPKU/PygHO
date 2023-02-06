from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()
    losss = []
    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)

        if True:
            optimizer.zero_grad()
            pred = model(batch)
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            losss.append(loss)
    return np.average([_.item() for _ in losss])


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    ylen = len(loader.dataset)
    y_true = torch.empty((ylen, 1), dtype=torch.long)
    y_pred = torch.empty((ylen, 1), device=device)
    step = 0
    for batch in loader:
        steplen = batch.y.shape[0]
        y_true[step:step + steplen] = batch.y
        batch = batch.to(device, non_blocking=True)
        y_pred[step:step + steplen] = model(batch)
        step += steplen
    assert step == y_true.shape[0]
    y_true = y_true.numpy()
    y_pred = y_pred.cpu().numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--norm',
                        type=str,
                        default='gcn',
                        choices=["sum", "mean", "max", "gcn"],
                        help='sum, mean, max, gcn (default: gcn)')
    parser.add_argument('--dp',
                        type=float,
                        default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument(
        '--num_layer',
        type=int,
        default=5,
        help='number of GNN message passing layers (default: 5)')
    parser.add_argument(
        '--emb_dim',
        type=int,
        default=256,
        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset',
                        type=str,
                        default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature',
                        type=str,
                        default="full",
                        help='full feature or simple feature')
    parser.add_argument('--jk', type=str, choices=["sum", "last"], default="last")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pool', type=str, choices=["sum", "mean", "max"], default="mean")
    parser.add_argument('--use_elin', action="store_true")
    parser.add_argument("--mlplayer", type=int, default=1)
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]],
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]],
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
    print(
        f"split {split_idx['train'].shape[0]} {split_idx['valid'].shape[0]} {split_idx['test'].shape[0]}"
    )
    model = GNN(dataset.num_tasks,
                args.num_layer,
                args.emb_dim,
                args.norm,
                virtual_node=False,
                residual=args.res,
                dp=args.dp,
                JK=args.jk,
                graph_pooling=args.pool,
                use_elin=args.use_elin,
                mlplayer=args.mlplayer).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        t1 = time.time()
        loss = train(model, device, train_loader, optimizer, dataset.task_type)
        print(f"time : {time.time()-t1:.1f} loss: {loss:.2e}")

        print('Evaluating...')
        t1 = time.time()
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        print(f"time : {time.time()-t1:.1f}")

        print({'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(loss)
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = min(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print(
        f'Best @{best_val_epoch} validation score: {valid_curve[best_val_epoch]:.4f} Test score: {test_curve[best_val_epoch]:.4f}'
    )


if __name__ == "__main__":
    main()