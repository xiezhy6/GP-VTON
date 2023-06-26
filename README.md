<div align="center">

<h1>GP-VTON: Towards General Purpose Virtual Try-on via Collaborative Local-Flow Global-Parsing Learning</h1>

<div>
    <a href="https://xiezhy6.github.io/" target="_blank">Zhenyu Xie</a><sup>1</sup>,
    <a href="https://github.com/xiezhy6/GP-VTON" target="_blank">Zaiyu Huang</a><sup>1</sup>,
    <a href="https://github.com/xiezhy6/GP-VTON">Xin Dong</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=XSf0hP4AAAAJ&hl=en" target="_blank">Fuwei Zhao</a><sup>1</sup>,
    <a href="https://sites.google.com/view/hydong?pli=1" target="_blank">Haoye Dong</a><sup>3</sup>,
</div>
<div>
    <a href="https://github.com/xiezhy6/GP-VTON" target="_blank">Xijin Zhang</a><sup>2</sup>
    <a href="https://github.com/xiezhy6/GP-VTON" target="_blank">Feida Zhu</a><sup>2</sup>
    <a href="https://lemondan.github.io/" target="_blank">Xiaodan Liang</a><sup>1,4</sup>
</div>
<div>
    <sup>1</sup>Shenzhen Campus of Sun Yat-Sen University&emsp; <sup>2</sup>ByteDance
</div>
<div>
    <sup>3</sup>Carnegie Mellon University&emsp; <sup>4</sup>Peng Cheng Laboratory
</div>

[Paper](https://arxiv.org/pdf/2303.13756.pdf) | [Project Page](https://github.com/xiezhy6/GP-VTON)
</br>

<strong>GP-VTON aims to transfer an in-shop garment onto a specific person.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="figures/worldcup_vton.png">
</div>

</div>

## Fine-grained Parsing
We provide the fine-grained parsing result of the model images and in-shop garment images from two existing high-resolution (1024 x 768) virtual try-on benchmarks, namely, VITON-HD and DressCode.

We provide two version of the parsing results. One is with the original resolution (1024 x 768). Another is with the resolution of 512 x 384, on which our experiment are conducted.

**VITON-HD**

|Resolution|Google Cloud|Baidu Yun|
|--------|--------------|-----------|
| 1024 x 768 | Available soon | Available soon |
| 512 x 384 | [Download](https://drive.google.com/file/d/1PS2cReM8uRDg9Rg51RH1cVQ7jkiAekAo/view?usp=sharing) | [Download](https://pan.baidu.com/s/1lFO2DcqCVurP2XxYbujnOg?pwd=b39q) |


**DressCode**

|Resolution|Google Cloud|Baidu Yun|
|--------|--------------|-----------|
| 1024 x 768 | [Download](https://drive.google.com/file/d/14-md1SlLI-TQdi7tB9R9XwjryrRf8G-Q/view?usp=sharing) | [Download](https://pan.baidu.com/s/1T_O-xgfYkTfH0dSFZOEcdw?pwd=up9i) |
| 512 x 384 | [Download](https://drive.google.com/file/d/1spTLgPjx1_qbmNa2Or8MzNDVawoNWXFO/view?usp=sharing) | [Download](https://pan.baidu.com/s/1hFkOvnOtWOp7UIplTMCPOw?pwd=c4dy) |

## Environment Setup
Install required packages:

```
pip3 install -r requirements.txt
```

## Dataset
We conduct experiments on the publicly available [VITON-HD](https://github.com/shadow2496/VITON-HD) and [DressCode](https://github.com/aimagelab/dress-code) datasets. For convenience, we provide all of the conditions used in our experiments in the following links.

|Resolution|Google Cloud|Baidu Yun|
|--------|--------------|-----------|
|VITON-HD(512 x 384)|[Download](https://drive.google.com/file/d/1zK3yZaqwIN8P933WwmDFVX8sUi1nVgbR/view?usp=sharing)|[Download](https://pan.baidu.com/s/1XlVunNM1mHsHN3HIoLCPLA?pwd=hu7h)|
|DressCode(512 x 384)|[Download (coming soon)]()|[Download (coming soon)]()|

## Inference

**VITON-HD**

Please download the pre-trained model from [Google Link](https://drive.google.com/file/d/1vrUJf8n0nJdzX76gRI1x2u07chqPKBRf/view?usp=sharing) or [Baidu Yun Link](https://pan.baidu.com/s/1ABlcBOSHEiKscYExiXVuew?pwd=cjdd), and put the downloaded directory under root directory of this project.

To test the first stage (i.e., the LFGP warping module), run the following command:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4739 test_warping.py \
    --name test_partflow_vitonhd_unpaired_1109 \
    --PBAFN_warp_checkpoint 'checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth' \
    --resize_or_crop None --verbose --tf_log \
    --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch \
    --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
    --image_pairs_txt test_pairs_unpaired_1018.txt
```

To test the second stage (i.e., the try-on generator), run the following command:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 test_tryon.py \
    --name test_gpvtongen_vitonhd_unpaired_1109 \
    --resize_or_crop None --verbose --tf_log --batchSize 12 --num_gpus 8 --label_nc 14 --launcher pytorch \
    --PBAFN_gen_checkpoint 'checkpoints/gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029/PBAFN_gen_epoch_201.pth' \
    --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
    --image_pairs_txt test_pairs_unpaired_1018.txt \
    --warproot /home/tiger/datazy/CVPR2023/GP-VTON-partflow_final_kaiwu/sample/test_partflow_vitonhd_unpaired_1109
```

Note that, in the above two commands, parameter `--dataroot` refers to the root directory of VITON-HD dataset, parameter `--image_pairs_txt` refers to the test list, which is put under the root directory of VITON-HD dataset, parameter `--warproot` in the second command refers to the directory of the warped results generated by the first command. Both of the generated results from the two commands are saved under the directory `./sample/exp_name`, in which `exp_name` is defined by the parameter `--name` in each command.

## Training

**VITON-HD**

To train the first stage (i.e., the LFGP warping module), run the following command:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=7129 train_warping.py \
    --name gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027 \
    --resize_or_crop None --verbose --tf_log \
    --batchSize 2 --num_gpus 8 --label_nc 14 --launcher pytorch  \
    --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
    --image_pairs_txt train_pairs_1018.txt \
    --display_freq 320 --print_freq 160 --save_epoch_freq 10 --write_loss_frep 320 \
    --niter_decay 50 --niter 70  --mask_epoch 70 \
    --lr 0.00005
```

To train the second stage (i.e., the try-on generator), run the following command:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4736 train_tryon.py \
    --name gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029 \
    --resize_or_crop None --verbose --tf_log --batchSize 10 --num_gpus 8 --label_nc 14 --launcher pytorch \
    --dataroot /home/tiger/datazy/CVPR2023/Datasets/VITONHD \
    --image_pairs_txt train_pairs_1018.txt \
    --warproot sample/test_gpvton_lrarms_for_training_1029 \
    --display_freq 50 --print_freq 25 --save_epoch_freq 10 --write_loss_frep 25 \
    --niter_decay 0 --niter 200 \
    --lr 0.0005
```


## Todo
- [x] Release the ground truth of the garment parsing and human parsing for two public benchmarks (VITON-HD and DressesCode) used in the paper
- [x] Release the the pretrained model and the inference script for VITON-HD dataset.
- [ ] Release the the pretrained model and the inference script for DressCode dataset.
- [x] Release the training script for VITON-HD dataset.
- [ ] Release the training script for DressCode dataset.

## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@inproceedings{xie2023gpvton,
  title     = {GP-VTON: Towards General Purpose Virtual Try-on via Collaborative Local-Flow Global-Parsing Learning},
  author    = {Zhenyu, Xie and Zaiyu, Huang and Xin, Dong and Fuwei, Zhao and Haoye, Dong and Xijin, Zhang and Feida, Zhu and Xiaodan, Liang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
}
```

## Acknowledgments

Thanks to [PF-AFN](https://github.com/geyuying/PF-AFN), our code is based on it.


## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.
