import optuna
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, choices=["ogbg-molhiv"])
parser.add_argument("num_anchor", type=int)
parser.add_argument("dev", type=int)
args = parser.parse_args()

stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.db", study_name=args.num_anchor, load_if_exists=True, direction="maximize")

def obj(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main_anchor.py --num_anchor {args.num_anchor} --dataset {dataset} "
    dp = trial.suggest_float("dp", 0, 0.9, step=0.05)
    layer = trial.suggest_int("layer", 4, 4)
    dim = trial.suggest_int("dim", 64, 64, step=16)
    bs = trial.suggest_int("bs", 1024, 1024, step=64)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    lr = trial.suggest_float("lr", 2e-3, 5e-3, step=1e-4)
    pool = "sum" #trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = "gcn" #trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 1, 1)
    res = trial.suggest_categorical("res", [True, False])
    cmd += f" --dp {dp} --num_layer {layer} --emb_dim {dim} --batch_size {bs} --jk {jk} "
    cmd += f" --norm {norm} --lr {lr} --pool {pool} --mlplayer {mlplayer} "
    if res:
        cmd += " --res "
    cmd += f"|grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-4]) - float(ret.split()[-1]) 
    return out


def obj2(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main_anchor.py --num_anchor {args.num_anchor} --dataset {dataset} "
    dp = trial.suggest_float("dp", 0, 0.9, step=0.05)
    layer = trial.suggest_int("layer", 4, 4)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    set2set = trial.suggest_categorical("set2set", ["id", "mindist", "maxcos"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    res = trial.suggest_categorical("res", [True, False])
    alpha = trial.suggest_float("alpha", 1e-3, 1e1, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e1, log=True)
    cmd += f" --an_dp {dp} --an_num_layer {layer} --an_jk {jk} "
    cmd += f" --an_norm {norm} --an_set2set {set2set} --alpha {alpha} --gamma {gamma} "
    if res:
        cmd += " --an_res "
    cmd += f"--repeat 3 |grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-4]) - float(ret.split()[-1]) 
    return out

if args.num_anchor == 1:
    stu.optimize(obj, 100)
else:
    stu.optimize(obj2, 100)