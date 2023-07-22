import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from SSWL import SSWL
import argparse
import time
import numpy as np
from densedata import loaddataset
from norm import NormMomentumScheduler, normdict
from typing import Callable
from subgdata.MaData import batch2dense

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def get_criterion(task, args):
    if task == "smoothl1reg":
        return torch.nn.SmoothL1Loss(reduction="none", beta=args.lossparam)
    else:
        criterion_dict = {
            "bincls": torch.nn.BCEWithLogitsLoss(reduction="none"),
            "cls":  torch.nn.CrossEntropyLoss(reduction="none"),
            "reg": torch.nn.MSELoss(reduction="none"),
            "l1reg": torch.nn.L1Loss(reduction="none"),
        }
        return criterion_dict[task]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(criterion: Callable,
          model: SSWL,
          device: torch.device,
          loader: DataLoader,
          optimizer: optim.Optimizer,
          task_type: str):
    model.train()
    losss = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        datadict = batch.to_dict()
        datadict = batch2dense(datadict)
        if True:
            optimizer.zero_grad()
            finalpred = model(datadict)
            y = datadict["y"]
            if task_type != "cls":
                y = y.to(torch.float)
                if y.ndim == 1:
                    y = y.unsqueeze(-1)
            value_loss = torch.mean(criterion(finalpred, y))
            totalloss = value_loss 
            totalloss.backward()
            optimizer.step()
            losss.append(value_loss)
    loss = np.average([_.item() for _ in losss])
    return loss




@torch.no_grad()
def eval(model, device, loader: DataLoader, evaluator):
    if len(loader) == 0:
        return 0
    model.eval()
    ylen = len(loader.dataset)
    ty = loader.dataset.data.y
    if ty.dim() == 1:
        y_true = torch.zeros((ylen), dtype=ty.dtype)
    elif ty.dim() == 2:
        y_true = torch.zeros((ylen, ty.shape[1]), dtype=ty.dtype)
    else:
        raise NotImplementedError
    y_pred = torch.zeros((ylen, model.num_tasks), device=device)
    step = 0
    for batch in loader:
        steplen = batch.y.shape[0]
        y_true[step:step + steplen] = batch.y
        batch = batch.to(device, non_blocking=True)
        datadict = batch.to_dict()
        datadict = batch2dense(datadict)
        tpred = model(datadict)
        y_pred[step:step + steplen] = tpred
        step += steplen
    assert step == y_true.shape[0]
    y_pred = y_pred.cpu()
    return evaluator(y_pred, y_true)


def parserarg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default="policygrad")
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0026)
    parser.add_argument('--K', type=float, default=0.0)
    parser.add_argument('--K2', type=float, default=0.0)
    parser.add_argument('--normK', type=float, default=0.0)
    parser.add_argument('--normK2', type=float, default=0.0)
    parser.add_argument('--warmstart', type=int, default=0)

    parser.add_argument('--dp', type=float, default=0.0)
    parser.add_argument("--nnnorm", type=str, default="none")
    parser.add_argument("--normparam", type=float, default=0.1)
    parser.add_argument("--ln_out", action="store_true")
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--mlplayer", type=int, default=1)
    parser.add_argument("--outlayer", type=int, default=1)
    
    parser.add_argument('--lossparam', type=float, default=0.05)

    parser.add_argument('--embdp', type=float, default=0.0)
    parser.add_argument("--embbn", action="store_true")
    parser.add_argument("--embln", action="store_true")
    parser.add_argument("--orthoinit", action="store_true")
    parser.add_argument("--max_norm", type=float, default=None)

    parser.add_argument('--aggr',
                        type=str,
                        default='sum',
                        choices=["sum", "mean", "max"])
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--gpool',
                        type=str,
                        choices=["sum", "mean", "max"],
                        default="sum")
    parser.add_argument('--lpool',
                        type=str,
                        choices=["sum", "mean", "max"],
                        default="sum")
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)

    args = parser.parse_args()
    print(args)
    return args


def buildModel(args, num_tasks, device, dataset):
    kwargs = {
        "mlp": {
            "dp": args.dp,
            "norm": args.nnnorm,
            "act": args.act,
            "normparam": args.normparam
        },
        "emb": {
            "dp": args.embdp,
            "bn": args.embbn,
            "ln": args.embln,
            "orthoinit": args.orthoinit,
            "max_norm": args.max_norm
        },
        "conv": {
            "aggr": args.aggr,
            "mlplayer": args.mlplayer,
            "mlp": {
                "dp": args.dp,
                "norm": args.nnnorm,
                "act": args.act,
                "normparam": args.normparam
            },
        },
    }
    print("num_task", num_tasks)
    model = SSWL(dataset,
                      num_tasks,
                      args.num_layer,
                      args.emb_dim,
                      args.gpool,
                      args.lpool,
                      args.res,
                      args.outlayer,
                      args.ln_out,
                      **kwargs).to(device)
    model = model.to(device)
    print(model)
    return model #torch.compile(model)


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
        trn_ratio, val_ratio, tst_ratio = int(split.split("-")[-3]), int(
            split.split("-")[-2]), int(split.split("-")[-1])
        num_fold = trn_ratio + val_ratio + tst_ratio
        trn_ratio /= num_fold
        val_ratio /= num_fold
        tst_ratio /= num_fold
        num_data = len(datasets[0])
        idx = torch.randperm(num_data)
        splitsize = num_data // num_fold
        idxs = [
            torch.cat((idx[splitsize * _:], idx[:splitsize * _]))
            for _ in range(num_fold)
        ]
        num_trn = int(trn_ratio * num_data)
        num_val = int(val_ratio * num_data)
    for rep in range(args.repeat):
        set_seed(rep)
        if "fixed" == split:
            trn_d, val_d, tst_d = datasets
        elif split.startswith("fold"):
            idx = idxs[rep]
            trn_idx, val_idx, tst_idx = idx[:num_trn], idx[
                num_trn:num_trn + num_val], idx[num_trn + num_val:]
            trn_d, val_d, tst_d = datasets[0][trn_idx], datasets[0][
                val_idx], datasets[0][tst_idx]
        else:
            datasets, split, evaluator, task = loaddataset(args.dataset)
            trn_d, val_d, tst_d = datasets
        print(len(trn_d), len(val_d), len(tst_d))
        train_loader = DataLoader(trn_d,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers,
                                  follow_batch=["edge_index", "tuplefeat"])
        train_eval_loader = DataLoader(trn_d,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers,
                                  follow_batch=["edge_index", "tuplefeat"])
        valid_loader = DataLoader(val_d,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers,
                                  follow_batch=["edge_index", "tuplefeat"])
        test_loader = DataLoader(tst_d,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.num_workers,
                                 follow_batch=["edge_index", "tuplefeat"])
        print(f"split {len(trn_d)} {len(val_d)} {len(tst_d)}")
        model = buildModel(args, trn_d.num_tasks, device, trn_d)
        if args.load is not None:
            loadparams = torch.load(f"mod/{args.load}.{rep}.pt", map_location="cpu")
            keys2del = ["anchor_encoder.emb.weight"]
            for key in loadparams.keys():
                if key.startswith("distlin"):
                    keys2del.append(key)
            for key in keys2del:
                del loadparams[key]
            print(model.load_state_dict(loadparams, strict=False))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        schedulerwst = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.9**(args.warmstart-epoch))
        schedulerdc = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1/(1+epoch*(args.K+args.K2*epoch)))
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[schedulerwst, schedulerdc], milestones=[args.warmstart], verbose=True)
        normscd = NormMomentumScheduler(lambda x:1/(1+x*(args.normK+x*args.normK2)), args.normparam, normdict[args.nnnorm])
        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            loss = train(get_criterion(task, args),
                                                      model, device,
                                                      train_loader, optimizer,
                                                      task)
            print(
                f"Epoch {epoch} train time : {time.time()-t1:.1f} loss: {loss:.2e} "
            )

            t1 = time.time()
            train_perf = 0.0 # eval(model, device, train_eval_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)
            print(
                f" test time : {time.time()-t1:.1f} Train {train_perf} Validation {valid_perf} Test {test_perf}", flush=True
            )
            train_curve.append(loss)
            valid_curve.append(valid_perf)
            test_curve.append(test_perf)
            if args.save is not None:
                if "cls" in task:
                    if valid_curve[-1] >= np.max(valid_curve):
                        torch.save(model.state_dict(), f"mod/{args.save}.{rep}.pt")
                else:
                    if valid_curve[-1] <= np.min(valid_curve):
                        torch.save(model.state_dict(), f"mod/{args.save}.{rep}.pt")
            scheduler.step()
            normscd.step(model)
        if 'cls' in task:
            best_val_epoch = np.argmax(np.array(valid_curve)+np.arange(len(valid_curve))*1e-15)
            best_train = min(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve)-np.arange(len(valid_curve))*1e-15)
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