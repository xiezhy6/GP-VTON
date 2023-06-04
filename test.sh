#!/bin/sh
### for vitonhd
### warping
if [ $1 == 1 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4739 test_warping.py \
        --name test_partflow_vitonhd_unpaired_1109 \
        --PBAFN_warp_checkpoint 'checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
        --resize_or_crop None --verbose --tf_log \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
        --image_pairs_txt test_pairs_unpaired_1018.txt
### for gen
elif [ $1 == 2 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 test_tryon.py \
        --name test_gpvtongen_vitonhd_unpaired_1109 \
        --resize_or_crop None --verbose --tf_log --batchSize 12 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --PBAFN_gen_checkpoint 'checkpoints/gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029/PBAFN_gen_epoch_201.pth' \
        --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
        --image_pairs_txt test_pairs_unpaired_1018.txt \
        --warproot /home/tiger/datazy/CVPR2023/GP-VTON-partflow_final_kaiwu/sample/test_partflow_vitonhd_unpaired_1109
fi
