# CUDA_VISIBLE_DEVICES=0,4,6,7 python -m torch.distributed.launch --nproc_per_node=4 main.py \
#   --arch UNet --dataset cifar100 --class-cond -1 --epochs 500 --batch-size 512 --ddim

# CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --nproc_per_node=4 main.py \
#   --arch UNetSmall --dataset cifar100 --class-cond -1 --epochs 500 --batch-size 1920 --ddim

CUDA_VISIBLE_DEVICES=3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=6 --master_port 8111 main.py \
  --arch UNetBig --dataset meta --class-cond -1 --epochs 200 --batch-size 384 --ddim --save-dir /data/housen/meta/big/1
