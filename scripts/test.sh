#!/bin/sh
### for vitonhd 512
### for garment warping
if [ $1 == 1 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4739 test_warping.py \
        --name test_partflow_vitonhd_unpaired_1109 \
        --PBAFN_warp_checkpoint 'checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
        --resize_or_crop None --verbose --tf_log \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD-512 \
        --image_pairs_txt test_pairs_unpaired_1018.txt
### for try-on generation
elif [ $1 == 2 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 test_tryon.py \
        --name test_gpvtongen_vitonhd_unpaired_1109 \
        --resize_or_crop None --verbose --tf_log --batchSize 12 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --PBAFN_gen_checkpoint 'checkpoints/gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029/PBAFN_gen_epoch_201.pth' \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD-512 \
        --image_pairs_txt test_pairs_unpaired_1018.txt \
        --warproot sample/test_partflow_vitonhd_unpaired_1109
### for vitonhd 1024
### for garment warping
elif [ $1 == 3 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4739 test_warping.py \
        --name test_gpvton_lrarms_for_testing_1024_230724 \
        --PBAFN_warp_checkpoint 'checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
        --resize_or_crop None --verbose --tf_log --resolution 1024 \
        --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD_ori \
        --image_pairs_txt test_pairs_unpaired_1018.txt
### for try-on generation
elif [ $1 == 4 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 test_tryon.py \
        --name test_gpvton_gen_vitonhd_wskin_lrarms_unpaired_1024_230719 \
        --resize_or_crop None --verbose --tf_log --resolution 1024 \
        --batchSize 3 --num_gpus 8 --label_nc 14 --launcher pytorch \
        --PBAFN_gen_checkpoint 'checkpoints/gp-vton_gen_vitonhd_wskin_wgan_lrarms_1024_230722/PBAFN_gen_epoch_201.pth' \
        --dataroot /home/tiger/datazy/Datasets/VITON-HD_ori \
        --image_pairs_txt test_pairs_unpaired_1018.txt \
        --warproot sample/test_gpvton_lrarms_for_testing_1024_230724
fi
