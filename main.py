from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import UniAnchorGNN
import argparse
import time
import numpy as np
from datasets import loaddataset

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

criterion_dict = {
    "bincls": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "cls": torch.nn.CrossEntropyLoss(reduction="none"),
    "reg": torch.nn.MSELoss(reduction="none")
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(criterion, model, device, loader, optimizer, task_type, alpha=1e1, gamma=1e-4):
    model.train()
    losss = []
    policy_losss = []
    entropy_losss = []
    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)

        if True:
            optimizer.zero_grad()
            preds, logprob, negentropy, finalpred = model(batch)
            y = batch.y.to(torch.float32)
            #print(preds.shape)
            #print(logprob.shape)
            #print(negentropy.shape)
            #print(finalpred.shape, y.shape)
            value_loss = torch.mean(criterion(finalpred, y))
            y = y.unsqueeze(0).unsqueeze(0).expand(preds.shape[0], preds.shape[1], -1, -1)
            with torch.no_grad():
                loss = criterion(preds.detach(), y)
                loss = loss.sum(dim=-1)
                loss = torch.diff(loss, dim=0)
            policy_loss = 0 if logprob is None else (loss.detach() * logprob)
            policy_loss = torch.tensor(0) if logprob is None else torch.mean(policy_loss)
            entropy_loss = torch.tensor(0) if negentropy is None else torch.mean(negentropy)
            totalloss = value_loss + alpha * policy_loss + gamma * entropy_loss
            totalloss.backward()
            optimizer.step()
            losss.append(value_loss)
            policy_losss.append(policy_loss)
            entropy_losss.append(entropy_loss)
    return np.average([_.item() for _ in losss]), np.average([
        _.item() for _ in policy_losss
    ]), np.average([_.item() for _ in entropy_losss])


@torch.no_grad()
def eval(model, device, loader, evaluator, T):
    model.eval()
    ylen = len(loader.dataset)
    y_true = torch.empty((ylen, 1), dtype=torch.long)
    y_pred = torch.empty((ylen, 1), device=device)
    step = 0
    for batch in loader:
        steplen = batch.y.shape[0]
        y_true[step:step + steplen] = batch.y
        batch = batch.to(device, non_blocking=True)
        y_pred[step:step + steplen] = model(batch, T)
        step += steplen
    assert step == y_true.shape[0]
    y_pred = y_pred.cpu()
    return evaluator(y_pred, y_true)

def parserarg():
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
    parser.add_argument('--multi_anchor', type=int, default=1)
    parser.add_argument('--rand_sample', action="store_true")
    parser.add_argument("--num_anchor", type=int, default=0)
    parser.add_argument('--policy_detach',
                        action="store_true")

    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e1)

    parser.add_argument('--testT', type=float, default=1)
    
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)

    args = parser.parse_args()
    print(args)
    return args

def buildModel(args, num_tasks, device, dataset):
    kwargs = {"mlp":{"dp": args.dp, "bn": args.bn, "ln": args.ln, "act": args.act}}
    model = UniAnchorGNN(num_tasks,
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
                          policy_detach=args.policy_detach,
                          dataset=dataset,
                          **kwargs).to(device)
    return model

def main():
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parserarg()
    ### automatic dataloading and splitting
    datasets, split, evaluator, task = loaddataset(args.dataset)
    print(split, task)
    outs = []
    set_seed(0)
    if split.startswith("fold"):
        trn_ratio, val_ratio, tst_ratio = int(split.split("-")[-3]), int(split.split("-")[-2]), int(split.split("-")[-1])
        num_fold = trn_ratio + val_ratio + tst_ratio
        trn_ratio /= num_fold
        val_ratio /= num_fold
        tst_ratio /= num_fold
        num_data = len(datasets[0])
        idx = torch.randperm(num_data)
        splitsize = num_data//num_fold
        idxs = [torch.cat((idx[splitsize*_:], idx[:splitsize*_])) for _ in range(num_fold)]
        num_trn = int(trn_ratio*num_data)
        num_val = int(val_ratio*num_data)
    for rep in range(args.repeat):
        set_seed(rep)
        if "fixed" == split:
            trn_d, val_d, tst_d = datasets
        elif split.startswith("fold"):
            idx = idxs[rep]
            trn_idx, val_idx, tst_idx = idx[:num_trn], idx[num_trn:num_trn+num_val],  idx[num_trn+num_val:]
            trn_d, val_d, tst_d = datasets[0][trn_idx], datasets[0][val_idx], datasets[0][tst_idx]
        else:
            datasets, split, evaluator, task = loaddataset(args.dataset)
            trn_d, val_d, tst_d = datasets
        print(len(trn_d), len(val_d), len(tst_d))
        train_loader = DataLoader(trn_d,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(val_d,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(tst_d,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        print(f"split {len(trn_d)} {len(val_d)} {len(tst_d)}")
        model = buildModel(args, trn_d.num_tasks, device, trn_d)
        if args.load is not None:
            model.load_state_dict(torch.load(f"mod/{args.load}.{rep}.pt", map_location=device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            loss, policyloss, entropyloss = train(criterion_dict[task], model, device, train_loader,
                                                  optimizer, task,
                                                  args.alpha, args.gamma)
            print(
                f"Epoch {epoch} train time : {time.time()-t1:.1f} loss: {loss:.2e} {policyloss:.2e} {entropyloss:.2e}"
            )

            t1 = time.time()
            valid_perf = eval(model, device, valid_loader, evaluator, args.testT)
            test_perf = eval(model, device, test_loader, evaluator, args.testT)
            print(
                f" test time : {time.time()-t1:.1f} Validation {valid_perf} Test {test_perf}"
            )
            train_curve.append(loss)
            valid_curve.append(valid_perf)
            test_curve.append(test_perf)
            if valid_curve[-1] >= np.max(valid_curve):
                if args.save is not None:
                    torch.save(model.state_dict(), f"mod/{args.save}.{rep}.pt")
        
        if 'cls' in task:
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