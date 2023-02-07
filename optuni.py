import optuna
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, choices=["ogbg-molhiv"])
parser.add_argument("num_anchor", type=int)
parser.add_argument("dev", type=int)
parser.add_argument("repeat", type=int)
args = parser.parse_args()

stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.{args.num_anchor}.db", study_name=args.num_anchor, load_if_exists=True, direction="maximize")

def obj(trial: optuna.Trial, dev: int =args.dev, dataset=args.dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main_uni.py --num_anchor {args.num_anchor} --dataset {dataset} --repeat {args.repeat} --epochs 100 "
    
    bs = trial.suggest_int("bs", 512, 512, step=64)
    lr = trial.suggest_float("lr", 3e-4, 5e-3, step=3e-4)

    dp = trial.suggest_float("dp", 0, 0.9, step=0.05)
    bn = trial.suggest_categorical("bn", [True, False])
    ln = trial.suggest_categorical("ln", [True, False])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    outlayer = trial.suggest_int("outlayer", 1, 3)
    anchor_outlayer = trial.suggest_int("aoutlayer", 1, 3)
    node2nodelayer = trial.suggest_int("n2nlayer", 1 , 2)

    norm = trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    layer = trial.suggest_int("layer", 2, 7)
    dim = trial.suggest_int("dim", 32, 256, step=32)
    jk = trial.suggest_categorical("jk", ["sum", "last"])
    res = trial.suggest_categorical("res", [True, False])
    pool = trial.suggest_categorical("pool", ["sum", "mean", "max"])
    
    set2set = trial.suggest_categorical("set2set", ["id", "mindist", "maxcos"])
    set2set_feat = trial.suggest_categorical("set2set_feat", [True, False])
    set2set_concat = trial.suggest_categorical("set2set_concat", [True, False])

    multi_anchor = trial.suggest_int("multi_anchor", 1, 4)
    
    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e2, log=True)
    
    cmd += f" --lr {lr} --batch_size {bs} --mlplayer {mlplayer} --outlayer {outlayer} --anchor_outlayer {anchor_outlayer}  --node2nodelayer {node2nodelayer} " 
    cmd += f" --emb_dim {dim} --pool {pool} --dp {dp} --num_layer {layer} --jk {jk} --multi_anchor {multi_anchor} "
    cmd += f" --norm {norm} --set2set {set2set} --alpha {alpha} --gamma {gamma} "
    if res:
        cmd += " --res "
    if bn:
        cmd += " --bn "
    if ln:
        cmd += " --ln "
    if set2set_feat:
        cmd += " --set2set_feat "
    if set2set_concat:
        cmd += " --set2set_concat "
    cmd += f"|grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-4]) - float(ret.split()[-1])*(args.repeat/(args.repeat-1))**0.5
    return out

stu.optimize(obj, 200)