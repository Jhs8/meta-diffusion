#!/bin/bash

# 设置预训练模型路径
for i in {1..10}
do
    j=$(($i*10))
    PRETRAINED_CKPT=/data/housen/meta/big/UNetBig_meta-epoch_100-timesteps_1000-class_condn_-1_${j}.pt 

# for i in {0..99}
# do
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8112 main.py \
        --arch UNetBig --dataset meta --class-cond "-1" --sampling-only --sampling-steps 250 \
        --num-sampled-images 10000 --pretrained-ckpt $PRETRAINED_CKPT --ddim --batch-size 2600 --save-dir "/data/housen/meta/big/sample"
done
# done