import optuna
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("num_anchor", type=int)
parser.add_argument("dev", type=int)
parser.add_argument("model", type=str, choices=["ppo", "policygrad", "debug", "randanchor"])
args = parser.parse_args()

isreg = args.dataset in ["QM9", "subgcount0", "subgcount1", "subgcount2", "subgcount3", "zinc"]
print(args, isreg)

stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.db", study_name=f"{args.model}_{args.num_anchor}", load_if_exists=True, direction="maximize" if not isreg else "minimize")

def debug(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor 0 --repeat 3 --randinit --dataset {dataset} --epochs 1000 "
    dp = trial.suggest_float("dp", 0, 0.0, step=0.05)
    layer = trial.suggest_int("layer", 2, 6)
    dim = trial.suggest_int("dim", 1500, 1500, step=100)
    bs = trial.suggest_int("bs", 15, 15, step=1)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, step=1e-4)
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    res = trial.suggest_categorical("res", [True, False])
    bn = trial.suggest_categorical("bn", [True, False])
    ln = trial.suggest_categorical("ln", [True, False])
    ln_out = trial.suggest_categorical("ln_out", [True, False])
    outlayer = trial.suggest_int("outlayer", 1, 3)
    cmd += f" --dp {dp} --num_layer {layer} --emb_dim {dim} --batch_size {bs} --jk {jk} "
    cmd += f" --norm {norm} --lr {lr} --pool {pool} --mlplayer {mlplayer}  --outlayer {outlayer} "
    if res:
        cmd += " --res "
    if bn:
        cmd += " --bn "
    if ln:
        cmd += " --ln "
    if ln_out:
        cmd += " --ln_out "
    cmd += f"|grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


def randanchor(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    num_anchor = trial.suggest_int("num_anchor", 0, 10)
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {num_anchor} --repeat 3 --rand_sample --dataset {dataset} --epochs 2000 "
    dp = trial.suggest_float("dp", 0, 0.0, step=0.05)
    layer = trial.suggest_int("layer", 2, 6)
    dim = trial.suggest_int("dim", 16, 128, step=16)
    bs = trial.suggest_int("bs", 1200, 1200, step=16)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, step=1e-4)
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    res = trial.suggest_categorical("res", [True, False])
    bn = trial.suggest_categorical("bn", [True, False])
    ln = trial.suggest_categorical("ln", [True, False])
    ln_out = trial.suggest_categorical("ln_out", [True, False])
    outlayer = trial.suggest_int("outlayer", 1, 3)
    cmd += f" --dp {dp} --num_layer {layer} --emb_dim {dim} --batch_size {bs} --jk {jk} "
    cmd += f" --norm {norm} --lr {lr} --pool {pool} --mlplayer {mlplayer}  --outlayer {outlayer} "
    if res:
        cmd += " --res "
    if bn:
        cmd += " --bn "
    if ln:
        cmd += " --ln "
    if ln_out:
        cmd += " --ln_out "
    cmd += f"|grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


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
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


def obj2(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {args.num_anchor} --repeat 3 --dataset {dataset} --epochs 500  --dp 0.0 --num_layer 5 --emb_dim 32 --batch_size 16 --jk sum  --norm gcn --lr 0.0023 --pool max --mlplayer 1  --outlayer 1  --bn  --ln  --ln_out "
    set2set = trial.suggest_categorical("set2set", ["id", "mindist", "maxcos"])
    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e0, log=True)
    lr = trial.suggest_float("lr", 2e-3, 5e-3, step=1e-4)
    s2sfeat = trial.suggest_categorical("s2sfeat", [True, False])
    s2scat = trial.suggest_categorical("s2scat", [True, False])
    T = trial.suggest_float("T", 1, 1e3, log=True)
    cmd += f"  --set2set {set2set} --alpha {alpha} --gamma {gamma} "
    cmd += f" --lr {lr}  --testT {T} "
    if s2scat:
        cmd += " --set2set_concat "
    if s2sfeat:
        cmd += " --set2set_feat "
    cmd += f"--repeat 3 |grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


def objppo(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {args.num_anchor} --dataset {dataset} --epochs 1000 "
    cmd += f" --dp 0.0 --num_layer 4 --jk sum  " 

    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e2, log=True)
    T = trial.suggest_float("T", 1e-1, 1e3, log=True)
    tau = trial.suggest_float("tau", 0.01, 0.99, step=0.01)
    ppolb = trial.suggest_float("ppolb", -2, 0, step=0.1)
    ppoub = trial.suggest_float("ppoub", 0, 2, step=0.1)
    lr = trial.suggest_float("lr", 2e-3, 5e-3, step=1e-4)
    cmd += f" --set2set id --alpha {alpha} --gamma {gamma} --batch_size 960  --norm mean --pool mean "
    cmd += f" --mlplayer 1 --bn --lr {lr}  --testT {T}  --set2set_concat  --set2set_feat "
    cmd += f" --repeat 10 --model ppo --tau {tau} --ppolb {ppolb} --ppoub {ppoub} "
    
    cmd += f"--repeat 3 |grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


if args.model == "debug":
    stu.optimize(debug, 100)
elif args.model == "randanchor":
    stu.optimize(randanchor, 100)
else:
    if args.num_anchor < 1:
        stu.optimize(obj, 100)
    else:
        if args.model == "ppo":
            stu.optimize(objppo, 100)
        else:
            stu.optimize(obj2, 100)