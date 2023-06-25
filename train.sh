#!/bin/sh
### for vitonhd
### LFGP warping module
if [ $1 == 1 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=7129 train_warping.py \
        --name gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027 \
        --resize_or_crop None --verbose --tf_log \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch  \
        --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
        --image_pairs_txt train_pairs_1018.txt \
        --display_freq 320 --print_freq 160 --save_epoch_freq 10 --write_loss_frep 320 \
        --niter_decay 50 --niter 70  --mask_epoch 70 \
        --lr 0.00005
fi
