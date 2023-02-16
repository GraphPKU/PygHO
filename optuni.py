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
    cmd = f"CUDA_VISIBLE_DEVICES={dev} python main_uni.py --num_anchor {args.num_anchor} --dataset {dataset} --repeat {args.repeat} --epochs 200 "
    
    bs = trial.suggest_int("bs", 1024, 1024, step=64)
    lr = trial.suggest_float("lr", 3e-3, 3e-3, step=1e-3)

    dp = trial.suggest_float("dp", 0, 0.9, step=0.1)
    bn = True #trial.suggest_categorical("bn", [True, False])
    ln = False #trial.suggest_categorical("ln", [True, False])
    mlplayer = trial.suggest_int("mlplayer", 1, 2)
    outlayer = trial.suggest_int("outlayer", 1, 3)
    anchor_outlayer = trial.suggest_int("aoutlayer", 1, 3)
    node2nodelayer = trial.suggest_int("n2nlayer", 1 , 2)

    norm = "gcn"#trial.suggest_categorical("norm", ["sum", "mean", "max", "gcn"])
    layer = trial.suggest_int("layer", 6, 6)
    dim = trial.suggest_int("dim", 64, 128, step=64)
    jk = "last"#trial.suggest_categorical("jk", ["sum", "last"])
    res = True #trial.suggest_categorical("res", [True, False])
    pool = "sum" #trial.suggest_categorical("pool", ["sum", "mean", "max"])
    
    set2set = "mindist" #trial.suggest_categorical("set2set", ["mindist", "maxcos"])
    set2set_feat = trial.suggest_categorical("set2set_feat", [True, False])
    set2set_concat = trial.suggest_categorical("set2set_concat", [True, False])

    multi_anchor = trial.suggest_int("multi_anchor", 1, 4)
    
    alpha = trial.suggest_float("alpha", 1e-1, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e-3, log=True)
    policy_detach = trial.suggest_categorical("pd", [True, False])
    policy_eval = trial.suggest_categorical("pe", [True, False])
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
    if policy_detach:
        cmd += " --policy_detach "
    if policy_eval:
        cmd += " --policy_eval "
    cmd += f"|grep runs:"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-4]) - float(ret.split()[-1])*(args.repeat/(args.repeat-1))**0.5
    return out

stu.optimize(obj, 200)