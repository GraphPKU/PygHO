CUDA_VISIBLE_DEVICES=4 nohup python example/zinc.py --sparse --aggr sum --conv SUN --npool mean --lpool mean --cpool max > SpSun.debug.out &
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
