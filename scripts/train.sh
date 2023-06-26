#!/bin/sh
### for vitonhd
## LFGP warping module
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
elif [ $1 == 2 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 train_tryon.py \
        --name gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029 \
        --resize_or_crop None --verbose --tf_log --batchSize 10 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
        --image_pairs_txt train_pairs_1018.txt \
        --warproot sample/test_gpvton_lrarms_for_training_1029 \
        --display_freq 50 --print_freq 25 --save_epoch_freq 10 --write_loss_frep 25 \
        --niter_decay 0 --niter 200 \
        --lr 0.0005
fi
