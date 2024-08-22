# Cumulative Rain Density Sensing Network for Single Image Derain

[![Project](https://img.shields.io/badge/Project-Page-blue.svg)]() 
[![IEEE](https://img.shields.io/badge/Project-Page-blue.svg)](https://ieeexplore.ieee.org/document/9001158) 


## Abstract

This paper focuses on single image derain, which aims to restore clear image from single rain image. Through full consideration of different frequency information preservation and the complicated interactions between rain-streaks and background, a novel end-to-end cumulative rain-density sensing network (CRDNet) is proposed for adaptive rain-streaks removal. An effective W-Net with powerful learning ability is proposed as a key component to recover rain-invariant low-frequency signals. A cumulative rain-density classifier with a novel cost-sensitive label encoding strategy is proposed as an auxiliary network to improve discriminative power of extracted high-frequency rain-streaks through multi-task training. The proposed CRDNet has been compared with state-of-the-art methods on two public datasets. The quantitative and visual experimental results demonstrate that it can achieve excellent performance with great improvement. Related source code and models are available on github https://github.com/peylnog/CRDNet.


## :sparkles: Getting Start
### Installation

**Framework**
1. Python 3.7
2. Pytorch1.0 (with ubuntu 16)
3. Torchvision

**Python Dependencies**

1. skimage
2. numpy
3. visdom : `pip install visdom`

## Train
Download Weights  [BaiDuYunLink](https://pan.baidu.com/s/1tIMv2snc0E93Btu9YA5TIw)  passwd:ncf3 

```
python3.7 derain.py 
```

## Test
```
python3.7 derain_test.py
ps:make sure data root is right
```

## Citation
Please cite us if this work is helpful to you.

```
    InProceedings{
    author = {L. Peng, A. Jiang, Q. Yi and M. Wang},
    title = {Cumulative Rain Density Sensing Network for Single Image Derain(CRDNet)},
    booktitle = {IEEE Signal Processing Letters(SPL)},
    month = {February},
    year = {2020}
    }
    
```
