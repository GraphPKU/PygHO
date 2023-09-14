CUDA_VISIBLE_DEVICES=0 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv NGNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > SpNGNN.time.out &
CUDA_VISIBLE_DEVICES=1 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv GNNAK --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > SpGNNAK.time.out &
CUDA_VISIBLE_DEVICES=2 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv DSSGNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > SpDSSGNN.time.out &
CUDA_VISIBLE_DEVICES=3 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv SSWL --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > SpSSWL.time.out &
CUDA_VISIBLE_DEVICES=4 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv PPGN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > SpPPGN.time.out &
CUDA_VISIBLE_DEVICES=5 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv SUN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > SpSUN.time.out &
CUDA_VISIBLE_DEVICES=6 nohup python -O example/zinc.py --epochs 10 --repeat 10  --aggr sum --conv SSWL --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > MaSSWL.time.out &
CUDA_VISIBLE_DEVICES=7 nohup python -O example/zinc.py --epochs 10 --repeat 10  --aggr sum --conv PPGN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > MaPPGN.time.out &

CUDA_VISIBLE_DEVICES=6 nohup python -O example/zinc.py --epochs 10 --repeat 1  --aggr sum --conv I2GNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 3.4e-3 --wd 3.7e-2 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 2.03e-5 --K 0.011 --K2 0.0073  > I2GNN.time.out &


{'lr': 0.0034065612285146232, 'wd': 0.03722265158992254, 'aggr': 'sum', 'npool': 'sum', 'lpool': 'sum', 'minlr': 2.0341235269027242e-05, 'normparam': 0.3130753368607271, 'cosT': 26, 'K': 0.011016896208476656, 'K2': 0.007270837470201833}

CUDA_VISIBLE_DEVICES=6 nohup python -O example/zinc.py --epochs 10 --repeat 10  --aggr sum --conv I2GNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 3.4e-3 --wd 3.7e-2 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 2.03e-5 --K 0.011 --K2 0.0073  > I2GNN.time.out &

{'lr': 0.004543542861001459, 'wd': 6.477760912476973e-06, 'aggr': 'mean', 'npool': 'sum', 'lpool': 'sum', 'minlr': 7.030999869053724e-05, 'normparam': 0.18535052628942864, 'cosT': 32, 'K': 0.00010376840392440702, 'K2': 8.242613862862107e-05}
CUDA_VISIBLE_DEVICES=4 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv PPGN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 4.5e-3 --wd 6.5e-6 --cosT 32 --dp 0.0 --outlayer 4 --normparam  1.85e-1 --minlr 7.0e-5 --K 1.04e-4 --K2 8.24e-5  > SpPPGN.2.time.out &


{'lr': 0.008917818793847022, 'wd': 6.478937487304309e-07, 'aggr': 'sum', 'npool': 'sum', 'lpool': 'mean', 'minlr': 1.5994362893084637e-06, 'normparam': 0.21991685589063176, 'cosT': 40, 'K': 0.014100560322057884, 'K2': 1.0347911649315998e-07}
CUDA_VISIBLE_DEVICES=6 nohup python -O example/zinc.py --epochs 600 --repeat 10  --aggr sum --conv SSWL --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9e-3 --wd 6.5e-7 --cosT 40 --dp 0.0 --outlayer 4 --normparam  0.22 --minlr 8.4e-5 --K 1.4e-2 --K2 1.0e-7  > MaSSWL.2.time.out &

{'lr': 0.0086, 'wd': 0.012, 'aggr': 'sum', 'npool': 'sum', 'lpool': 'sum', 'minlr': 8.9e-06, 'normparam': 0.31, 'cosT': 42, 'K': 0.0013, 'K2': 0.00028
CUDA_VISIBLE_DEVICES=0 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv DSSGNN --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 0.0086 --wd 0.012 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 8.9e-6 --K 1.3e-3 --K2 2.8e-4  > SpDSSGNN.2.time.out &

CUDA_VISIBLE_DEVICES=2 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv GNNAK --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 0.0086 --wd 0.012 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 8.9e-6 --K 1.3e-3 --K2 2.8e-4  > SpGNNAK.2.time.out &

{'lr': 0.0086, 'wd': 0.0064, 'aggr': 'sum', 'npool': 'sum', 'lpool': 'sum', 'minlr': 2.36e-05, 'normparam': 0.57, 'cosT': 35, 'K': 5.7e-07, 'K2': 0.00028}
CUDA_VISIBLE_DEVICES=1 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv SUN --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 0.0086 --wd 0.0064 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.57 --minlr 2.4e-5 --K 5.7e-7 --K2 2.8e-4  > SpSUN.2.time.out &

{'lr': 0.0037306061580411167, 'wd': 0.0001365758890619353, 'aggr': 'sum', 'npool': 'sum', 'lpool': 'mean', 'minlr': 5.902538747285543e-07, 'normparam': 0.21560077784065987, 'cosT': 43, 'K': 0.036279045710338645, 'K2': 0.006338929618591591}
CUDA_VISIBLE_DEVICES=3 nohup python -O example/zinc.py --epochs 10 --repeat 10 --sparse  --aggr sum --conv GNNAK --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 3.7e-3 --wd 1.4e-4 --cosT 43 --dp 0.0 --outlayer 4 --normparam  2.2e-1 --minlr 5.9e-7 --K 3.6e-2 --K2 6.3e-3  > SpGNNAK.2.time.out &



CUDA_VISIBLE_DEVICES=6 nohup python example/zinc.py --sparse --aggr sum --conv SUN --npool mean --lpool mean --cpool max > SpSun.debug.time.out &
CUDA_VISIBLE_DEVICES=5 nohup python example/zinc.py --sparse --aggr sum --conv SSWL --npool mean --lpool mean --cpool max > SpSSWL.debug.time.out &
CUDA_VISIBLE_DEVICES=6 nohup python example/zinc.py --sparse --aggr sum --conv NGNN --npool mean --lpool mean --cpool max > SpNGNN.debug.time.out &
CUDA_VISIBLE_DEVICES=7 nohup python example/zinc.py --sparse --aggr sum --conv GNNAK --npool mean --lpool mean --cpool max > SpGNNAK.debug.time.out &
CUDA_VISIBLE_DEVICES=3 nohup python example/zinc.py --sparse --aggr sum --conv DSSGNN --npool mean --lpool mean --cpool max > SpDSSGNN.debug.time.out &


CUDA_VISIBLE_DEVICES=0 nohup python example/zinc.py --aggr sum --conv SSWL --npool mean --lpool mean --cpool max > MaSSWL.debug.time.out &
CUDA_VISIBLE_DEVICES=1 nohup python example/zinc.py --aggr sum --conv NGNN --npool mean --lpool mean --cpool max > MaNGNN.debug.time.out &
CUDA_VISIBLE_DEVICES=2 nohup python example/zinc.py --aggr sum --conv GNNAK --npool mean --lpool mean --cpool max > MaGNNAK.debug.time.out &
CUDA_VISIBLE_DEVICES=4 nohup python example/zinc.py --aggr sum --conv SUN --npool mean --lpool mean --cpool max > MaSun.debug.time.out &
CUDA_VISIBLE_DEVICES=3 nohup python example/zinc.py --aggr sum --conv DSSGNN --npool mean --lpool mean --cpool max > MaDSSGNN.debug.time.out &

CUDA_VISIBLE_DEVICES=2 nohup python -O example/zinc.py  --aggr sum --conv PPGN --npool mean --lpool mean --cpool max > MaPPGN.debug.time.out &
CUDA_VISIBLE_DEVICES=5 nohup python -O example/zinc.py --sparse  --aggr sum --conv PPGN --npool mean --lpool mean --cpool max > SpPPGN.debug.time.out &
