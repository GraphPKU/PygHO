from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import UniAnchorGNN
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.MSELoss(reduction="none")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model, device, loader, optimizer, task_type, alpha=1e1, gamma=1e-4):
    model.train()
    losss = []
    policy_losss = []
    entropy_losss = []
    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)

        if True:
            optimizer.zero_grad()
            pred, logprob, negentropy = model(batch)
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred[is_labeled],
                                     batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred[is_labeled],
                                     batch.y.to(torch.float32)[is_labeled])
            policy_loss = (loss.detach() * logprob)
            policy_loss = torch.mean(policy_loss)
            entropy_loss = torch.mean(negentropy)
            loss = torch.mean(loss)
            totalloss = loss + alpha * policy_loss + gamma * entropy_loss
            totalloss.backward()
            optimizer.step()
            losss.append(loss)
            policy_losss.append(policy_loss)
            entropy_losss.append(entropy_loss)
    return np.average([_.item() for _ in losss]), np.average([
        _.item() for _ in policy_losss
    ]), np.average([_.item() for _ in entropy_losss])


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
        y_pred[step:step + steplen] = model(batch)[0]
        step += steplen
    assert step == y_true.shape[0]
    y_true = y_true.numpy()
    y_pred = y_pred.cpu().numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv")
    parser.add_argument('--feature', type=str, default="full")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0026)

    parser.add_argument('--dp', type=float, default=0.9)
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--mlplayer", type=int, default=1)
    parser.add_argument("--outlayer", type=int, default=1)
    parser.add_argument("--anchor_outlayer", type=int, default=1)
    parser.add_argument("--node2nodelayer", type=int, default=1)
    parser.add_argument("--use_elin", action="store_true")

    parser.add_argument('--norm',
                        type=str,
                        default='gcn',
                        choices=["sum", "mean", "max", "gcn"])
    parser.add_argument('--num_layer', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--jk',
                        type=str,
                        choices=["sum", "last"],
                        default="last")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--pool',
                        type=str,
                        choices=["sum", "mean", "max"],
                        default="sum")


    parser.add_argument('--set2set',
                        type=str,
                        choices=["id",  "mindist", "maxcos"],
                        default="id")
    parser.add_argument('--set2set_feat',
                        action="store_true")
    parser.add_argument('--set2set_concat',
                        action="store_true")
    parser.add_argument('--multi_anchor', type=int)
    parser.add_argument('--rand_sample', action="store_true")
    parser.add_argument("--num_anchor", type=int, default=0)

    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e1)
    
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
    evaluator = Evaluator(args.dataset)
    outs = []
    for rep in range(args.repeat):
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
        kwargs = {"mlp":{"dp": args.dp, "bn": args.bn, "ln": args.ln, "act": args.act}}
        model = UniAnchorGNN(dataset.num_tasks,
                          args.num_anchor,
                          args.num_layer,
                          args.emb_dim,
                          args.norm,
                          virtual_node=False,
                          residual=args.res,
                          JK=args.jk,
                          graph_pooling=args.pool,
                          set2set=args.set2set+("-feat-" if args.set2set_feat else "-") + ("concat" if args.set2set_concat else "-"),
                          use_elin=args.use_elin,
                          mlplayer=args.mlplayer,
                          rand_anchor=args.rand_sample,
                          multi_anchor=args.multi_anchor,
                          anchor_outlayer=args.anchor_outlayer,
                          outlayer=args.outlayer,
                          node2nodelayer=args.node2nodelayer,
                          **kwargs).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            loss, policyloss, entropyloss = train(model, device, train_loader,
                                                  optimizer, dataset.task_type,
                                                  args.alpha, args.gamma)
            print(
                f"Epoch {epoch} train time : {time.time()-t1:.1f} loss: {loss:.2e} {policyloss:.2e} {entropyloss:.2e}"
            )

            t1 = time.time()
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)
            print(
                f" test time : {time.time()-t1:.1f} Validation {valid_perf} Test {test_perf}"
            )
            train_curve.append(loss)
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        if 'classification' in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = min(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        print(
            f'Best @{best_val_epoch} validation score: {valid_curve[best_val_epoch]:.4f} Test score: {test_curve[best_val_epoch]:.4f}'
        )
        outs.append([
            best_val_epoch, valid_curve[best_val_epoch],
            test_curve[best_val_epoch]
        ])
    print(outs)
    print(f"all runs: ", end=" ")
    for _ in np.average(outs, axis=0):
        print(_, end=" ")
    for _ in np.std(outs, axis=0):
        print(_, end=" ")
    print()


if __name__ == "__main__":
    main()