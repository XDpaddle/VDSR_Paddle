# VDSR_Paddle

### Overview

This repository contains an op-for-op Paddle reimplementation of [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587).
It is modified from the original source code of Pytorch implementation(https://github.com/Lornatang/VDSR-PyTorch)


## About Accelerating the Super-Resolution Convolutional Neural Network

If you're new to VDSR, here's an abstract straight from the paper:

We present a highly accurate single-image superresolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for
ImageNet classification. We find increasing our network depth shows a significant improvement in accuracy. Our finalmodel uses 20 weight layers. By
cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With
very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We
learn residuals onlyb and use extremely high learning rates
(104 times higher than SRCNN) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy and
visual improvements in our results are easily noticeable.


## Download datasets

Contains T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

In the config.py
- line 31: `mode` change  to 'valid' 
- line 70: `hr_dir` change to the image address you want to test
- line 69: `sr_dir` change to the image address you want to save

In the validate.py
- line 13: `model_path` change to the weight address you use 

python validate.py

## Result

Source of original paper results: https://arxiv.org/pdf/1511.04587.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |

|  Set5   |   4   | 25.52(**31.05**) |



### Credit

#### Accurate Image Super-Resolution Using Very Deep Convolutional Networks

_Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee_ <br>

**Abstract** <br>
We present a highly accurate single-image superresolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for
ImageNet classification. We find increasing our network depth shows a significant improvement in accuracy. Our finalmodel uses 20 weight layers. By
cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With
very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We
learn residuals onlyb and use extremely high learning rates
(104 times higher than SRCNN) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy and
visual improvements in our results are easily noticeable.

[[Paper]](https://arxiv.org/pdf/1511.04587) [[Author's implements(MATLAB)]](https://cv.snu.ac.kr/research/VDSR/VDSR_code.zip)

```
@inproceedings{vedaldi15matconvnet,
  author    = {A. Vedaldi and K. Lenc},
  title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
  booktitle = {Proceeding of the {ACM} Int. Conf. on Multimedia},
  year      = {2015},
}
```
