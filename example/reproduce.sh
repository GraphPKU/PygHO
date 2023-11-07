CUDA_VISIBLE_DEVICES=0 nohup python -O example/zinc.py --epochs 10 --repeat 1 --sparse  --aggr sum --conv NGNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6  > NGNN.compiled.out &
CUDA_VISIBLE_DEVICES=2 nohup python -O example/zinc.py --sparse --epochs 10 --repeat 1  --aggr sum --conv I2GNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 3.4e-3 --wd 3.7e-2 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 2.03e-5 --K 0.011 --K2 0.0073  > I2GNN.compiled.out &
CUDA_VISIBLE_DEVICES=3 nohup python -O example/zinc.py --epochs 10 --repeat 1 --sparse  --aggr sum --conv PPGN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 4.5e-3 --wd 6.5e-6 --cosT 32 --dp 0.0 --outlayer 4 --normparam  1.85e-1 --minlr 7.0e-5 --K 1.04e-4 --K2 8.24e-5  > PPGN.compiled.out &
CUDA_VISIBLE_DEVICES=4 nohup python -O example/zinc.py --epochs 10 --repeat 1  --aggr sum --conv SSWL --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9e-3 --wd 6.5e-7 --cosT 40 --dp 0.0 --outlayer 4 --normparam  0.22 --minlr 8.4e-5 --K 1.4e-2 --K2 1.0e-7  > SSWL.compiled.out &
CUDA_VISIBLE_DEVICES=5 nohup python -O example/zinc.py --epochs 10 --repeat 1 --sparse  --aggr sum --conv DSSGNN --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 0.0086 --wd 0.012 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 8.9e-6 --K 1.3e-3 --K2 2.8e-4  > DSSGNN.compiled.out &
CUDA_VISIBLE_DEVICES=6 nohup python -O example/zinc.py --epochs 10 --repeat 1 --sparse  --aggr sum --conv GNNAK --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 0.0086 --wd 0.012 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.31 --minlr 8.9e-6 --K 1.3e-3 --K2 2.8e-4  > GNNAK.compiled.out &
CUDA_VISIBLE_DEVICES=7 nohup python -O example/zinc.py --epochs 10 --repeat 1 --sparse  --aggr sum --conv SUN --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 0.0086 --wd 0.0064 --cosT 26 --dp 0.0 --outlayer 4 --normparam  0.57 --minlr 2.4e-5 --K 5.7e-7 --K2 2.8e-4  > SUN.compiled.out &
CUDA_VISIBLE_DEVICES=1 nohup python -O example/NGAT.py --epochs 1000 --repeat 10 --sparse  --aggr sum --conv NGNN --npool sum --lpool sum --cpool mean --mlplayer 2 --norm bn --lr 1e-2 --wd 4.9e-5 --cosT 26 --dp 0.0 --outlayer 4 --normparam  1.94e-1 --minlr 8.4e-5 --K 4.9e-3 --K2 4.33e-6 > NGAT.compiled.out &