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

[Paper](https://arxiv.org/pdf/2303.13756.pdf) | [Project Page]()
</br>

<strong>GP-VTON aims to transfer an in-shop garment onto a specific person.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="figures/worldcup_vton.png">
</div>

</div>

## Dataset
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



## Todo
- [x] Release the ground truth of the garment parsing and human parsing for two public benchmarks (VITON-HD and DressesCode) used in the paper
- [ ] Release the the pretrained model and the inference script.
- [ ] Release the training script.

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
