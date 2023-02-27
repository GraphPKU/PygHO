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
from gnn import modeldict, PPOAnchorGNN

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


def train(criterion,
          model,
          device,
          loader,
          optimizer,
          task_type,
          alpha=1e1,
          gamma=1e-4,
          T: float=1):
    model.train()
    losss = []
    policy_losss = []
    entropy_losss = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if True:
            optimizer.zero_grad()
            preds, logprob, negentropy, finalpred = model(batch, T)
            y = batch.y
            if task_type != "cls":
                y = y.to(torch.float)
            value_loss = torch.mean(criterion(finalpred, y))
            if task_type != "cls":
                y = y.unsqueeze(0).unsqueeze(0).expand(preds.shape[0],
                                                    preds.shape[1], -1, -1)
            if preds is None:
                policy_loss = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    if task_type == "cls":
                        ty = y.reshape(-1, 1, 1).expand(-1, preds.shape[0], preds.shape[1])
                        tpred = preds.detach().permute(2, 3, 0, 1)
                        loss = criterion(tpred, ty)
                        loss = loss.permute(1, 2, 0)
                    else:
                        loss = criterion(preds.detach(), y)
                        loss = loss.sum(dim=-1)
                    loss = loss[[-1]] - loss[:-1]
                policy_loss = torch.tensor(0.0) if logprob is None else (
                    loss.detach() * logprob)
                policy_loss = torch.mean(policy_loss)
            entropy_loss = torch.tensor(
                0.0) if negentropy is None else torch.mean(negentropy)
            totalloss = value_loss + alpha * policy_loss + gamma * entropy_loss
            totalloss.backward()
            optimizer.step()
            losss.append(value_loss)
            policy_losss.append(policy_loss)
            entropy_losss.append(entropy_loss)
    #from torchmetrics import Accuracy, MeanAbsoluteError
    #print(finalpred, y)
    #print(Accuracy("multiclass", num_classes=15)(finalpred.cpu(), y.cpu()))
    loss = np.average([_.item() for _ in losss])
    policy_loss = np.average([_.item() for _ in policy_losss])
    entropy_loss = np.average([_.item() for _ in entropy_losss])
    return loss, policy_loss, entropy_loss 


def train_ppo(criterion,
              model,
              device,
              loader,
              optimizer,
              task_type: str,
              alpha: float = 1e1,
              gamma: float = 1e-4,
              ppolb: float = -0.5,
              ppoub: float = 0.5,
              T: float = 1):
    model.train()
    losss = []
    policy_losss = []
    entropy_losss = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        if True:
            optimizer.zero_grad()
            preds, logprob, oldlogprob, negentropy, finalpred = model(batch, T)
            y = batch.y
            value_loss = torch.mean(criterion(finalpred, y))
            if task_type != "cls":
                y = y.unsqueeze(0).unsqueeze(0).expand(preds.shape[0],
                                                    preds.shape[1], -1, -1)
            with torch.no_grad():
                if task_type == "cls":
                    loss = criterion(preds.detach().permute(2, 3, 0, 1), y)
                else:
                    loss = criterion(preds.detach(), y)
                loss = loss.sum(dim=-1)
                loss = loss[[-1]] - loss[:-1]
            policy_loss = torch.tensor(0.0) if logprob is None else (
                loss.detach() *
                torch.exp(torch.clip(logprob - oldlogprob, ppolb, ppoub)))
            policy_loss = torch.mean(policy_loss)
            entropy_loss = torch.tensor(
                0.0) if negentropy is None else torch.mean(negentropy)
            totalloss = value_loss + alpha * policy_loss + gamma * entropy_loss
            totalloss.backward()
            optimizer.step()
            model.updateP()
            losss.append(value_loss)
            policy_losss.append(policy_loss)
            entropy_losss.append(entropy_loss)
    return np.average([_.item() for _ in losss]), np.average([
        _.item() for _ in policy_losss
    ]), np.average([_.item() for _ in entropy_losss])


@torch.no_grad()
def eval(model, device, loader: DataLoader, evaluator, T):
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
    # print(y_true.shape, y_pred.shape)
    for batch in loader:
        steplen = batch.y.shape[0]
        y_true[step:step + steplen] = batch.y
        batch = batch.to(device, non_blocking=True)
        tpred = model(batch, T)[-1]
        y_pred[step:step + steplen] = tpred
        step += steplen
    assert step == y_true.shape[0]
    y_pred = y_pred.cpu()
    #print(y_pred, y_true)
    #print("eval", evaluator(tpred.cpu(), batch.y.cpu()))
    return evaluator(y_pred, y_true)


def parserarg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        choices=modeldict.keys(),
                        default="policygrad")
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0026)
    parser.add_argument('--K', type=float, default=0.0)
    parser.add_argument('--K2', type=float, default=0.0)

    parser.add_argument('--dp', type=float, default=0.0)
    parser.add_argument("--nnnorm", type=str, default="none")
    parser.add_argument("--ln_out", action="store_true")
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--mlplayer", type=int, default=1)
    parser.add_argument("--outlayer", type=int, default=1)
    parser.add_argument("--anchor_outlayer", type=int, default=1)
    parser.add_argument("--use_elin", action="store_true")
    
    parser.add_argument('--lossparam', type=float, default=0.05)

    parser.add_argument('--embdp', type=float, default=0.0)
    parser.add_argument("--embbn", action="store_true")
    parser.add_argument("--embln", action="store_true")
    parser.add_argument("--orthoinit", action="store_true")
    parser.add_argument("--max_norm", type=float, default=None)

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

    parser.add_argument('--nosharelin', action="store_true")
    parser.add_argument("--noallshare", action="store_true")

    parser.add_argument('--set2set',
                        type=str,
                        choices=["id", "mindist", "maxcos"],
                        default="id")
    parser.add_argument('--set2set_feat', action="store_true")
    parser.add_argument('--set2set_concat', action="store_true")
    parser.add_argument('--multi_anchor', type=int, default=1)
    parser.add_argument('--rand_sample', action="store_true")
    parser.add_argument('--fullsample', action="store_true")
    parser.add_argument("--num_anchor", type=int, default=0)
    parser.add_argument('--policy_detach', action="store_true")

    parser.add_argument("--nodistlin", action="store_true")

    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e1)

    parser.add_argument('--trainT', type=float, default=1)
    parser.add_argument('--testT', type=float, default=1)

    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)

    # ppo
    parser.add_argument("--ppolb", type=float, default=-0.5)
    parser.add_argument("--ppoub", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.99)

    parser.add_argument('--randinit', action="store_true")

    args = parser.parse_args()
    print(args)
    return args


def buildModel(args, num_tasks, device, dataset):
    kwargs = {
        "mlp": {
            "dp": args.dp,
            "norm": args.nnnorm,
            "act": args.act,
            "sharelin": not args.nosharelin,
            "allshare": not args.noallshare
        },
        "emb": {
            "dp": args.embdp,
            "bn": args.embbn,
            "ln": args.embln,
            "orthoinit": args.orthoinit,
            "max_norm": args.max_norm
        }
    }
    print("num_task", num_tasks)
    if args.model == "policygrad":
        model = UniAnchorGNN(num_tasks,
                             args.num_anchor,
                             args.num_layer,
                             args.emb_dim,
                             args.norm,
                             virtual_node=False,
                             residual=args.res,
                             JK=args.jk,
                             graph_pooling=args.pool,
                             set2set=args.set2set +
                             ("-feat-" if args.set2set_feat else "-") +
                             ("concat" if args.set2set_concat else "-"),
                             use_elin=args.use_elin,
                             mlplayer=args.mlplayer,
                             rand_anchor=args.rand_sample,
                             multi_anchor=args.multi_anchor,
                             anchor_outlayer=args.anchor_outlayer,
                             outlayer=args.outlayer,
                             policy_detach=args.policy_detach,
                             dataset=dataset,
                             randinit=args.randinit,
                             ln_out=args.ln_out,
                             fullsample=args.fullsample,
                             nodistlin=args.nodistlin,
                             **kwargs).to(device)
    elif args.model == "ppo":
        raise NotImplementedError
        model = PPOAnchorGNN(num_tasks,
                             args.num_anchor,
                             args.num_layer,
                             args.emb_dim,
                             args.norm,
                             virtual_node=False,
                             residual=args.res,
                             JK=args.jk,
                             graph_pooling=args.pool,
                             set2set=args.set2set +
                             ("-feat-" if args.set2set_feat else "-") +
                             ("concat" if args.set2set_concat else "-"),
                             use_elin=args.use_elin,
                             mlplayer=args.mlplayer,
                             rand_anchor=args.rand_sample,
                             multi_anchor=args.multi_anchor,
                             anchor_outlayer=args.anchor_outlayer,
                             outlayer=args.outlayer,
                             policy_detach=args.policy_detach,
                             dataset=dataset,
                             tau=args.tau,
                             ln_out=args.ln_out,
                             nodistlin=args.nodistlin,
                             **kwargs).to(device)
    else:
        raise NotImplementedError(f"unknown model {args.model}")
    model = model.to(device)
    print(model)
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
                                  num_workers=args.num_workers)
        train_eval_loader = DataLoader(trn_d,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(val_d,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(tst_d,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.num_workers)
        print(f"split {len(trn_d)} {len(val_d)} {len(tst_d)}")
        model = buildModel(args, trn_d.num_tasks, device, trn_d)
        if args.load is not None:
            model.load_state_dict(
                torch.load(f"mod/{args.load}.{rep}.pt", map_location=device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1/(1+epoch*(args.K+args.K2*epoch)))
        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            if args.model == "ppo":
                loss, policyloss, entropyloss = train_ppo(
                    get_criterion(task, args), model, device, train_loader,
                    optimizer, task, args.alpha, args.gamma, args.ppolb, args.ppoub, args.trainT)
            else:
                loss, policyloss, entropyloss = train(get_criterion(task, args),
                                                      model, device,
                                                      train_loader, optimizer,
                                                      task, args.alpha,
                                                      args.gamma, args.trainT)
            print(
                f"Epoch {epoch} train time : {time.time()-t1:.1f} loss: {loss:.2e} {policyloss:.2e} {entropyloss:.2e}"
            )

            t1 = time.time()
            train_perf = 0.0 # eval(model, device, train_eval_loader, evaluator, args.testT)
            valid_perf = eval(model, device, valid_loader, evaluator,
                              args.testT)
            test_perf = eval(model, device, test_loader, evaluator, args.testT)
            print(
                f" test time : {time.time()-t1:.1f} Train {train_perf} Validation {valid_perf} Test {test_perf}", flush=True
            )
            train_curve.append(loss)
            valid_curve.append(valid_perf)
            test_curve.append(test_perf)
            if valid_curve[-1] >= np.max(valid_curve):
                if args.save is not None:
                    torch.save(model.state_dict(), f"mod/{args.save}.{rep}.pt")
            scheduler.step()
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