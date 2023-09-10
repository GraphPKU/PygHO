CUDA_VISIBLE_DEVICES=0 nohup python -O example/zinc.py --epochs 400 --sparse  --aggr sum --conv NGNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > SpNGNN.out &
CUDA_VISIBLE_DEVICES=1 nohup python -O example/zinc.py --epochs 400 --sparse  --aggr sum --conv PPGN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > SpPPGN.out &
CUDA_VISIBLE_DEVICES=2 nohup python -O example/zinc.py --epochs 400 --sparse  --aggr sum --conv SSWL --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > SpSSWL.out &
CUDA_VISIBLE_DEVICES=3 nohup python -O example/zinc.py --epochs 400 --sparse  --aggr sum --conv GNNAK --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > SpGNNAK.out &
CUDA_VISIBLE_DEVICES=4 nohup python -O example/zinc.py --epochs 400 --sparse  --aggr sum --conv SUN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > SpSUN.out &
CUDA_VISIBLE_DEVICES=5 nohup python -O example/zinc.py --epochs 400 --sparse  --aggr sum --conv DSSGNN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > SpDSSGNN.out &
CUDA_VISIBLE_DEVICES=6 nohup python -O example/zinc.py --epochs 400  --aggr sum --conv PPGN --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > MaPPGN.out &
CUDA_VISIBLE_DEVICES=7 nohup python -O example/zinc.py --epochs 400  --aggr sum --conv SSWL --npool sum --lpool mean --cpool mean --mlplayer 2 --norm bn --lr 9.4e-3 --wd 7.5e-5 --cosT 40 --dp 0.0 --outlayer 4 --normparam 0.244  --minlr 1e-4 --repeat 10 > MaSSWL.out &
CUDA_VISIBLE_DEVICES=6 nohup python example/zinc.py --sparse --aggr sum --conv SUN --npool mean --lpool mean --cpool max > SpSun.debug.out &
CUDA_VISIBLE_DEVICES=5 nohup python example/zinc.py --sparse --aggr sum --conv SSWL --npool mean --lpool mean --cpool max > SpSSWL.debug.out &
CUDA_VISIBLE_DEVICES=6 nohup python example/zinc.py --sparse --aggr sum --conv NGNN --npool mean --lpool mean --cpool max > SpNGNN.debug.out &
CUDA_VISIBLE_DEVICES=7 nohup python example/zinc.py --sparse --aggr sum --conv GNNAK --npool mean --lpool mean --cpool max > SpGNNAK.debug.out &
CUDA_VISIBLE_DEVICES=3 nohup python example/zinc.py --sparse --aggr sum --conv DSSGNN --npool mean --lpool mean --cpool max > SpDSSGNN.debug.out &


CUDA_VISIBLE_DEVICES=0 nohup python example/zinc.py --aggr sum --conv SSWL --npool mean --lpool mean --cpool max > MaSSWL.debug.out &
CUDA_VISIBLE_DEVICES=1 nohup python example/zinc.py --aggr sum --conv NGNN --npool mean --lpool mean --cpool max > MaNGNN.debug.out &
CUDA_VISIBLE_DEVICES=2 nohup python example/zinc.py --aggr sum --conv GNNAK --npool mean --lpool mean --cpool max > MaGNNAK.debug.out &
CUDA_VISIBLE_DEVICES=4 nohup python example/zinc.py --aggr sum --conv SUN --npool mean --lpool mean --cpool max > MaSun.debug.out &
CUDA_VISIBLE_DEVICES=3 nohup python example/zinc.py --aggr sum --conv DSSGNN --npool mean --lpool mean --cpool max > MaDSSGNN.debug.out &

CUDA_VISIBLE_DEVICES=2 nohup python -O example/zinc.py  --aggr sum --conv PPGN --npool mean --lpool mean --cpool max > MaPPGN.debug.out &
CUDA_VISIBLE_DEVICES=5 nohup python -O example/zinc.py --sparse  --aggr sum --conv PPGN --npool mean --lpool mean --cpool max > SpPPGN.debug.out &
