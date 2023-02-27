import optuna
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("num_anchor", type=int)
parser.add_argument("dev", type=int)
parser.add_argument("model", type=str, choices=["ppo", "policygrad", "debug", "randanchor", "fullsample"])
parser.add_argument("delrec", type=int, default=0)
args = parser.parse_args()

isreg = args.dataset in ["QM9", "subgcount0", "subgcount1", "subgcount2", "subgcount3", "zinc"]
print(args, isreg)

stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.db", study_name=f"{args.model}_{args.num_anchor}", load_if_exists=True, direction="maximize" if not isreg else "minimize")

def debug(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor 0 --repeat 2 --randinit --dataset {dataset} --epochs 400 "
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
    K = trial.suggest_float("K", 1e-5, 1e1, log=True)
    K2 = trial.suggest_float("K2", 1e-5, 1e1, log=True)
    lossparam = trial.suggest_float("lossparam", 1e-3, 5e-1, log=True)
    cmd += f" --dp {dp} --num_layer {layer} --emb_dim {dim} --batch_size {bs} --jk {jk} --lossparam {lossparam}"
    cmd += f" --norm {norm} --lr {lr} --pool {pool} --mlplayer {mlplayer}  --outlayer {outlayer} --K {K} --K2 {K2} "
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
    num_anchor = trial.suggest_int("num_anchor", 0, 6)
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {num_anchor} --repeat 3 --rand_sample --dataset {dataset} --epochs 100 "
    dp = trial.suggest_float("dp", 0, 0.3, step=0.05)
    layer = trial.suggest_int("layer", 1, 6)
    dim = trial.suggest_int("dim", 16, 64, step=16)
    bs = trial.suggest_int("bs", 1500, 1500, step=16)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, step=3e-4)
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    res = trial.suggest_categorical("res", [True, False])
    bn = trial.suggest_categorical("bn", [True, False])
    ln = trial.suggest_categorical("ln", [True, False])
    embln = trial.suggest_categorical("embln", [True, False])
    orthoinit = trial.suggest_categorical("orthoinit", [True, False])
    ln_out = False #trial.suggest_categorical("ln_out", [True, False])
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
    if orthoinit:
        cmd += " --orthoinit "
    if embln:
        cmd += " --embln "
    cmd += f"|grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out

def fullsample(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor 1 --repeat 2 --fullsample --multi_anchor 30 --dataset {dataset} --epochs 1000 "
    dp = trial.suggest_float("dp", 0, 0.3, step=0.05)
    layer = trial.suggest_int("layer", 1, 3)
    dim = trial.suggest_int("dim", 16, 32, step=16)
    bs = trial.suggest_int("bs", 1500, 1500, step=256)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, step=3e-4)
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    res = trial.suggest_categorical("res", [True, False])
    nnnorm = trial.suggest_categorical("nnnorm", ["none", "ln", "bn", "gn", "in"])
    ln_out = False #trial.suggest_categorical("ln_out", [True, False])
    outlayer = trial.suggest_int("outlayer", 1, 3)
    cmd += f" --dp {dp} --num_layer {layer} --emb_dim {dim} --batch_size {bs} --jk {jk} --nnnorm {nnnorm} "
    cmd += f" --norm {norm} --lr {lr} --pool {pool} --mlplayer {mlplayer}  --outlayer {outlayer} "
    if res:
        cmd += " --res "
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
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {args.num_anchor} --dataset {dataset} --epochs 400  --dp 0.0 --batch_size 1024 --repeat 2 "
    layer = trial.suggest_int("layer", 9, 12)
    dim = 256#trial.suggest_int("dim", 256, 256, step=16)
    jk = "sum" #trial.suggest_categorical("jk", ["sum", "last"])
    pool = "sum" #trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = "sum" #trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 2, 3)
    res = True #trial.suggest_categorical("res", [True, False])
    nnnorm = "bn"#trial.suggest_categorical("nnnorm", ["none", "ln", "bn", "gn", "in"])
    orthoinit = False #trial.suggest_categorical("orthoinit", [True, False])
    outlayer = trial.suggest_int("outlayer", 2, 3)
    lr = trial.suggest_float("lr", 1e-2, 6e-2, step=1e-3)
    ln_out = False #trial.suggest_categorical("ln_out", [True, False])
    K = trial.suggest_float("K", 1e-6, 5e-3, log=True)
    K2 = trial.suggest_float("K2", 1e-6, 5e-3, log=True)
    lossparam = trial.suggest_float("lossparam", 1e-2, 5e-1, log=True)
    warmstart = trial.suggest_int("warmstart", 0, 20, step=5)
    normparam = trial.suggest_float("normparam", 1e-2, 2e-1, log=True)
    cmd += f" --lr {lr}  --nnnorm {nnnorm}  --K {K} --K2 {K2} --lossparam {lossparam} "
    cmd += f" --num_layer {layer} --emb_dim {dim} --jk {jk} --warmstart {warmstart} "
    cmd += f" --norm {norm} --pool {pool} --mlplayer {mlplayer}  --outlayer {outlayer} --normparam {normparam} "
    if ln_out:
        cmd += " --ln_out "
    if orthoinit:
        cmd += " --orthoinit "
    if res:
        cmd += " --res "
    cmd += f" |grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


def obj2(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main.py --num_anchor {args.num_anchor} --dataset {dataset} --epochs 100  --dp 0.0 --batch_size 1024 --repeat 3 "
    layer = trial.suggest_int("layer", 1, 5)
    dim = trial.suggest_int("dim", 32, 128, step=16)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    res = trial.suggest_categorical("res", [True, False])
    nnnorm = trial.suggest_categorical("nnnorm", ["none", "ln", "bn", "gn", "in"])
    orthoinit = trial.suggest_categorical("orthoinit", [True, False])
    outlayer = trial.suggest_int("outlayer", 1, 2)
    anchor_outlayer = trial.suggest_int("anchor_outlayer", 1, 2)
    set2set = trial.suggest_categorical("set2set", ["id", "mindist", "maxcos"])
    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, step=1e-4)
    s2sfeat = trial.suggest_categorical("s2sfeat", [True, False])
    s2scat = trial.suggest_categorical("s2scat", [True, False])
    testT = trial.suggest_float("testT", 1e-1, 1e3, log=True)
    trainT = trial.suggest_float("trainT", 1e-1, 1e3, log=True)
    ln_out = False #trial.suggest_categorical("ln_out", [True, False])
    multi_anchor = trial.suggest_int("multi_anchor", 1, 1, step=3)
    noallshare = trial.suggest_categorical("noallshare", [True, False])
    nosharelin = trial.suggest_categorical("nosharelin", [True, False])
    cmd += f" --anchor_outlayer {anchor_outlayer} --set2set {set2set} --alpha {alpha} --gamma {gamma} --multi_anchor {multi_anchor} "
    cmd += f" --lr {lr}  --testT {testT} --trainT {trainT} --nnnorm {nnnorm} "
    cmd += f" --num_layer {layer} --emb_dim {dim} --jk {jk} "
    cmd += f" --norm {norm} --lr {lr} --pool {pool} --mlplayer {mlplayer}  --outlayer {outlayer} "
    if noallshare:
        cmd += " --noallshare "
    if nosharelin:
        cmd += " --nosharelin "
    if ln_out:
        cmd += " --ln_out "
    if s2scat:
        cmd += " --set2set_concat "
    if s2sfeat:
        cmd += " --set2set_feat "
    if orthoinit:
        cmd += " --orthoinit "
    if res:
        cmd += " --res "
    cmd += f" |grep runs:"
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
    cmd += f" --model ppo --tau {tau} --ppolb {ppolb} --ppoub {ppoub} "
    
    cmd += f"--repeat 3 |grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = (float(ret.split()[-4]) - float(ret.split()[-1])) if not isreg else (float(ret.split()[-4]) + float(ret.split()[-1]))
    return out


if args.delrec >0:
    stu = optuna.delete_study(storage=f"sqlite:///{args.dataset}.db", study_name=f"{args.model}_{args.num_anchor}")
    exit()
if args.model == "debug":
    stu.optimize(debug, 100)
elif args.model == "randanchor":
    stu.optimize(randanchor, 100)
elif args.model == "fullsample":
    stu.optimize(fullsample, 100)
else:
    if args.num_anchor < 1:
        stu.optimize(obj, 100)
    else:
        if args.model == "ppo":
            stu.optimize(objppo, 100)
        else:
            stu.optimize(obj2, 100)