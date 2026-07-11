# <p align=center> Edge and Flow Guided Iterative CNN for Remote Sensing Image Change Detection <p>

## Getting Started
### 1. Dataset Download & Pre-processing
Change detection benchmarks:
- LEVIR-CD: https://justchenhao.github.io/LEVIR
- WHU-CD: Official version: http://gpcv.whu.edu.cn/data/building_dataset.html, Pre-processed version: https://drive.google.com/file/d/1c93Y0ioe16rxEkVIJJyUpJdKkMmPOTxR/view
- SYSU-CD: https://github.com/liumency/SYSU-CD
Please ensure that each image in the dataset is cropped to patches of 256×256 pixels.
### 2. Dataset Organization
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
### 3. Environment Setup
Create a virtual environment and install all dependencies:
`pip install -r requirements.txt`
### 4. Training & Testing
Please download the VGG16 pretrained weight file [vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and place it in the `model/` directory of the EFICNN project.

Training:
`bash train.sh`
\
Testing:
`bash test.sh`

## Acknowledgement
This repository is built upon [ChangeViT](https://github.com/zhuduowang/ChangeViT) and [A2Net](https://github.com/guanyuezhen/A2Net)

We sincerely thank the authors for their well-organized and open-sourced codebases.
