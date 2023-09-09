#!/bin/sh
### for vitonhd 512
## train LFGP warping module
if [ $1 == 1 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=7129 train_warping.py \
        --name gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027 \
        --resize_or_crop None --verbose --tf_log \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch  \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD-512 \
        --image_pairs_txt train_pairs_1018.txt \
        --display_freq 320 --print_freq 160 --save_epoch_freq 10 --write_loss_frep 320 \
        --niter_decay 50 --niter 70  --mask_epoch 70 \
        --lr 0.00005
## prepare the warped garment for the training of try-on generator
elif [ $1 == 2 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4739 test_warping.py \
        --name test_gpvton_lrarms_for_training_1029 \
        --PBAFN_warp_checkpoint 'checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
        --resize_or_crop None --verbose --tf_log \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD-512 \
        --image_pairs_txt train_pairs_1018.txt
## train try-on generator
elif [ $1 == 3 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 train_tryon.py \
        --name gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029 \
        --resize_or_crop None --verbose --tf_log --batchSize 10 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD-512 \
        --image_pairs_txt train_pairs_1018.txt \
        --warproot sample/test_gpvton_lrarms_for_training_1029 \
        --display_freq 50 --print_freq 25 --save_epoch_freq 10 --write_loss_frep 25 \
        --niter_decay 0 --niter 200 \
        --lr 0.0005
### for vitonhd 1024
## prepare the warped garment for the training of try-on generator
elif [ $1 == 4 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4739 test_warping.py \
        --name test_gpvton_lrarms_for_training_1024_230721 \
        --PBAFN_warp_checkpoint 'checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
        --resize_or_crop None --verbose --tf_log  --resolution 1024 \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD_ori \
        --image_pairs_txt train_pairs_1018.txt
## train try-on generator
elif [ $1 == 5 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 train_tryon.py \
        --name gp-vton_gen_vitonhd_wskin_wgan_lrarms_1024_230722 \
        --resize_or_crop None --verbose --tf_log --resolution 1024 \
        --batchSize 3 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD_ori \
        --image_pairs_txt train_pairs_1018.txt \
        --warproot sample/test_gpvton_lrarms_for_training_1024_230721 \
        --display_freq 150 --print_freq 150 --save_epoch_freq 10 --write_loss_frep 50 \
        --niter_decay 0 --niter 200 \
        --lr 0.0005
fi
