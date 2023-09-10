import numpy as np
import optuna
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("dev", type=int)
args = parser.parse_args()
def parseout(outs: list[str])->float:
    ret = 10
    for line in outs:
        if "tst MAE" in line:
            ret = float(line.split()[-1])
    if np.isnan(ret):
        return 10
    return ret

def obj(trial: optuna.Trial):
    aggr = trial.suggest_categorical("aggr", ["sum", "mean", "max"])
    npool = trial.suggest_categorical("npool", ["sum", "mean", "max"])
    lpool = trial.suggest_categorical("lpool", ["sum", "mean", "max"])
    mlplayer = trial.suggest_int("mlplayer", 0, 3)
    outlayer = trial.suggest_int("outlayer", 1, 4)
    dp = trial.suggest_float("dp", 0, 0.2, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, step=3e-4)
    wd = trial.suggest_float("wd", 1e-6, 1e-1, log=True)
    norm = trial.suggest_categorical("norm", ["ln", "bn", "none"])
    cmd = f"CUDA_VISIBLE_DEVICES={args.dev} python -O example/zinc.py --sparse  --aggr {aggr} --conv NGNN --npool {npool} --lpool {lpool} --cpool mean --mlplayer {mlplayer} --norm {norm} --lr {lr} --wd {wd:.1e} --cosT 0 --dp {dp:.2f} --outlayer {outlayer} "
    out = subprocess.check_output(cmd, shell=True)
    out = str(out, encoding="utf-8")
    out = out.splitlines()
    return parseout(out)

def objfine(trial: optuna.Trial):
    outlayer = 4#trial.suggest_int("outlayer", 1, 4)
    lr = trial.suggest_float("lr", 3e-3, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-7, 1e-1, log=True)
    aggr = "sum"#trial.suggest_categorical("aggr", ["sum", "mean"])
    npool = "sum"#trial.suggest_categorical("npool", ["sum", "mean"])
    lpool = "mean"#trial.suggest_categorical("lpool", ["sum", "mean"])
    minlr = trial.suggest_float("minlr", 1e-7, 3e-3, log=True)
    normparam = trial.suggest_float("normparam", 1e-2, 1-1e-2)
    cosT = trial.suggest_int("cosT", 20, 50, log=True)
    K = trial.suggest_float("K", 1e-7, 1, log=True)
    K2 = trial.suggest_float("K2", 1e-7, 1, log=True)
    cmd = f"CUDA_VISIBLE_DEVICES={args.dev} python -O example/zinc.py --epochs 400 --sparse  --aggr {aggr} --conv NGNN --npool {npool} --lpool {lpool} --cpool mean --mlplayer 2 --norm bn --lr {lr:.1e} --wd {wd:.1e} --cosT {cosT} --dp 0.0 --outlayer {outlayer} --normparam {normparam}  --minlr {minlr} --K {K:.1e} --K2 {K2:.2e} "
    out = subprocess.check_output(cmd, shell=True)
    out = str(out, encoding="utf-8")
    out = out.splitlines()
    return parseout(out)

stu = optuna.create_study(storage="sqlite:///NGNNSp.db", study_name="finef", direction="minimize", load_if_exists=True)
stu.optimize(objfine, 200)
