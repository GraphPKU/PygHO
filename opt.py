import optuna
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("num_anchor", type=int)
parser.add_argument("dev", type=int)
args = parser.parse_args()

stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.db", study_name=args.num_anchor, load_if_exists=True, direction="maximize")

def obj(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {args.num_anchor} --dataset {dataset} "
    dp = trial.suggest_float("dp", 0, 0.9, step=0.05)
    layer = trial.suggest_int("layer", 2, 5)
    dim = trial.suggest_int("dim", 16, 64, step=16)
    bs = trial.suggest_int("bs", 16, 64, step=16)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    lr = trial.suggest_float("lr", 2e-3, 5e-3, step=1e-4)
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
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
      --batch_size 1200 --bn --norm sum --pool sum --mlplayer 2 --repeat 3 --set2set_feat --epochs 1000
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {args.num_anchor} --dataset {dataset} "
    dp = trial.suggest_float("dp", 0, 0, step=0.05)
    layer = trial.suggest_int("layer", 2, 6)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    set2set = trial.suggest_categorical("set2set", ["id", "mindist", "maxcos"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    res = trial.suggest_categorical("res", [True, False])
    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e0, log=True)
    bs = trial.suggest_int("bs", 16, 64, step=16)
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    cmd += f" --dp {dp} --num_layer {layer} --jk {jk} "
    cmd += f"  --set2set {set2set} --alpha {alpha} --gamma {gamma} --batch_size {bs} "
    cmd += f" --norm {norm} --pool {pool} --mlplayer {mlplayer} "
    if res:
        cmd += " --res "
    cmd += f"--repeat 3 |grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-4]) - float(ret.split()[-1]) 
    return out

if args.num_anchor < 1:
    stu.optimize(obj, 100)
else:
    stu.optimize(obj2, 100)