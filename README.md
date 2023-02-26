
 python main.py --num_anchor 3 --dataset EXP --epochs 1000  --dp 0.0 --num_layer 4 --jk sum   --set2set id --alpha 8 --gamma 0 --batch_size 960  --norm mean --pool mean --mlplayer 1 --bn --lr 0.0022  --testT 135  --set2set_concat  --set2set_feat --repeat 10

 CUDA_VISIBLE_DEVICES=1 python main.py --num_anchor 1 --repeat 10 --rand_sample --dataset CSL --epochs 500  --dp 0.0 --num_layer 5 --emb_dim 32 --batch_size 16 --jk sum  --norm gcn --lr 0.0023 --pool max --mlplayer 1  --outlayer 1  --bn  --ln  --ln_out

  CUDA_VISIBLE_DEVICES=1 python main.py --num_anchor 1 --dataset CSL --epochs 500 --dp 0.0 --num_layer 5 --emb_dim 32 --batch_size 16 --jk sum --norm gcn --lr 0.0023 --pool max --mlplayer 1 --outlayer 1 --bn --ln --ln_out --set2set mindist --alpha 35 --gamma 0.036 --lr 0.0028 --testT 60 --repeat 10 > test/CSL.anchor1.out


        
CUDA_VISIBLE_DEVICES=0 python main.py --num_anchor 3 --dataset sr --epochs 500 --dp 0.0 --batch_size 15 --set2set id --alpha 6.1 --gamma 3e-4 --multi_anchor 300 --lr 0.004 --testT 2.1 --trainT 14.4 --nnnorm in --num_layer 5 --emb_dim 64 --jk last --norm sum --lr 0.004 --pool sum --mlplayer 1 --outlayer 2 --ln_out --set2set_feat --repeat 10 > test/sr.anchor3.out
