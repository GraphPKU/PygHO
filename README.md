
 python main.py --num_anchor 3 --dataset EXP --epochs 1000  --dp 0.0 --num_layer 4 --jk sum   --set2set id --alpha 8 --gamma 0 --batch_size 960  --norm mean --pool mean --mlplayer 1 --bn --lr 0.0022  --testT 135  --set2set_concat  --set2set_feat --repeat 10

 CUDA_VISIBLE_DEVICES=1 python main.py --num_anchor 1 --repeat 10 --rand_sample --dataset CSL --epochs 500  --dp 0.0 --num_layer 5 --emb_dim 32 --batch_size 16 --jk sum  --norm gcn --lr 0.0023 --pool max --mlplayer 1  --outlayer 1  --bn  --ln  --ln_out

  CUDA_VISIBLE_DEVICES=1 python main.py --num_anchor 1 --dataset CSL --epochs 500 --dp 0.0 --num_layer 5 --emb_dim 32 --batch_size 16 --jk sum --norm gcn --lr 0.0023 --pool max --mlplayer 1 --outlayer 1 --bn --ln --ln_out --set2set mindist --alpha 35 --gamma 0.036 --lr 0.0028 --testT 60 --repeat 10 > test/CSL.anchor1.out


        
CUDA_VISIBLE_DEVICES=0 python main.py --num_anchor 3 --repeat 3 --dataset sr --epochs 500 --dp 0.0 --num_layer 3 --emb_dim 64 --multi_anchor 100 --batch_size 15 --jk sum --norm sum --set2set mindist --lr 3e-4 --pool sum --mlplayer 1 --outlayer 2 --alpha 0.01 --gamma 0 --ln_out --orthoinit --trainT 100 --testT 100 --nnnorm in --nodistlin  > sr.debug.out &
