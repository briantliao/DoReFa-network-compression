![Maintenance](https://img.shields.io/maintenance/yes/2019.svg?color=red&style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/csyhhu/Awesome-Deep-Neural-Network-Compression.svg?style=flat-square)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/csyhhu/Awesome-Deep-Neural-Network-Compression.svg?style=flat-square)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg?style=flat-square)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg?style=flat-square)](https://awesome.re)

# Re-implemented Codes for Network Compression

A repo for **my** re-implementation of state-of-the-art methods. Mostly written in `PyTorch`.

This re-implementation is based on the published paper and released codes (if available). Details and tricks may be 
different, which leads to the differences in final performance.

Leave me an issue if you found anything wrong or want to contribute it.
## Methods

Finished:

- [X] [DoReFa](./DoReFa): [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
- [X] [BWN](./BWN): [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
- [X] [TTQ](./TTQ): [Trained Ternary Quantization](https://arxiv.org/pdf/1612.01064v1.pdf)
- [X] [FBS](./FBS): [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/abs/1810.05331)

Unfinished:

- [ ] [ExNN](./ExNN): [Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/abs/1707.09870)
- [ ] [INQ](./INQ): [Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights](https://arxiv.org/abs/1702.03044)
- [ ] [QIL](./QIL): [Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss](https://arxiv.org/abs/1808.05779) 

The following are my work:
- [L-OBS](https://github.com/csyhhu/L-OBS) [NIPS 2017] First author: [Xing Dong](http://www.simonxin.com/)
- [L-DNQ](https://github.com/csyhhu/L-DNQ) [AAAI 2019] Co-work with: [Wenya Wang](https://www.ntu.edu.sg/home/wangwy/), [Sinno Jialin Pan](https://www.ntu.edu.sg/home/sinnopan/)
- [Co-Prune](https://github.com/csyhhu/Co-Prune) [IJCAI 2019] Co-work with: [Wenya Wang](https://www.ntu.edu.sg/home/wangwy/), [Sinno Jialin Pan](https://www.ntu.edu.sg/home/sinnopan/)
- [Meta-Quant](https://github.com/csyhhu/Meta-Quant)[NeurIPS 2019] Co-work with: [Wenya Wang](https://www.ntu.edu.sg/home/wangwy/), [Sinno Jialin Pan](https://www.ntu.edu.sg/home/sinnopan/)

## How to use them
Please check the detailed description in each folder. However, here is some setting that most codes will use:

### Specify Your Dataset Root
Check `utils/dataset.py` to see whether it incorporates the dataset you want to use and whether you specify the right path.

For example, if you want to use `CIFAR10`, you need to specify your dataset root in Line 90 in `utils/dataset.py` as:
```python
data_root_list = ['/data/CIFAR10', '/home/datasets/CIFAR10']
```
Code will search in the data root list to find `CIFAR10`, it will terminate once it find it and use `CIFAR10` from that path.
Otherwise, it will download a new `CIFAR10` in the last folder in `data_root_list`. 

This setting is designed for deploying codes in various machine such as your own PC and server.

Currently, the following dataset is supported:
- MNIST
- SVHN
- CIFAR10
- CIFAR100
- STL10
- ImageNet

#### Special Things for ImageNet
For ImageNet, I implement two ways of loading it:

- `utils.dataset.get_dataloader`

The first way is from `PyTorch`'s [official release](https://github.com/pytorch/examples/tree/master/imagenet). 
Your need to go through that link before use it. However, `PyTorch`'s dataloader will go through all the images 
in folder to generate a list of all images' path, which is time-consuming. I modified it a little bit by saving 
that path collection at the first image traverse as .pkl and load that .pkl for time saving.

- `utils.dataset.get_lmdb_imagenet`

If your machine is too slow in loading image. My recommendation is to generate a `lmdb` ImageNet at the first place
and use this interface for image reading. Here are the steps:

Generate lmdb `ImageNet` as in `caffe` using its default setting.

Or you can download a finish one from [here](https://github.com/jiecaoyu/XNOR-Net-PyTorch).

Then you can use `utils.dataset.get_lmdb_imagenet` to read in `ImageNet` without the effort to
load in single image using CPU every time.

However, since there is some difference caused by this conversion, using `lmdb ImageNet` in `PyTorch`'s
release pretrained model will show a slight drop in performance (2% drop in my case).

## Results

### Quantization

#### ResNet20 using CIFAR10

| Method | W | A | G | E | Quantized Acc | FP Acc |
| :-------:|:---:|:---:|:---:|:---:|:-------------:|:--------:|
| TTQ    |1.5| 32| 32| 32|     90.3      | 91.5  |
| DoReFa | 1 | 32|32|32| 89.69 | 91.5|
| BWN    | 1 | 32|32|32| 90.3 | 91.5|


### Acknowledgement

Some of my utils code is copied and modified from:

- [PyTorch with CIFAR10](https://github.com/kuangliu/pytorch-cifar): But its architecture is not quite situable for CIFAR10.
- [XNOR using PyTorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch): Thanks for providing the efficient way to load ImageNet.



# Awesome Deep Neural Network Compression
Paper collection, Summary, Code for Deep Neural Network Compression, including:
 - Quantization, 
 - Pruning (Unstructure, structure), 
 - Distillation 
 
 and so on.

## Paper:
 +  By Topic:
    
    - [Quantization](./Paper/Quantization.md)
    - [Pruning](./Paper/Pruning.md)
    - [Efficient Model Design](./Paper/Efficient-Model-Design.md)
    - [Network Architecture Search (NAS) for Model Compression](./Paper/NAS.md)
    - [Compression Meets Robustness (Adversarial)](./Paper/Robust-Compression.md)
    - [NLP Compression](./Paper/NLP-Compression.md)
    
 +  [By Conference](./Paper/PaperByConference.md):
 
	- [2020](./Paper/Conference/2020.md)
    - [2019](./Paper/Conference/2019.md)
    - [2018](./Paper/Conference/2018.md)
    
 +  Survey
 
 + Related Topic:
   - [Optimization](./)
   - [Meta Learning](./Paper/Meta-Learning.md)

## Codes / Tools:
 + [My Implementation](./Codes): My re-implementation of state-of-the-art compression methods.
## Summary: 
My summary (slides) for network compression. Some papers are chosen to be represented.
* [Quantization Summary](./Summary/Quantization-Summary.pdf)
* [Pruning Summary](./Summary/Prunning-Summary.pdf)
* Theory: From basic convex optimization to quantization
## Compression System:
* [Distiller](https://nervanasystems.github.io/distiller/)
* [PocketFlow](https://github.com/Tencent/PocketFlow)