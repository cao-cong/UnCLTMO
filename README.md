# Unsupervised HDR Image and Video Tone Mapping via Contrastive Learning (UnCLTMO)

This repository contains official implementation of Unsupervised HDR Image and Video Tone Mapping via Contrastive Learning in TCSVT 2023, by Cong Cao, Huanjing Yue, Xin Liu, and Jingyu Yang.

<p align="center">
  <img width="800" src="https://github.com/cao-cong/UnCLTMO/blob/main/images/ContrastiveLearningLoss.png">
</p>

## Dataset

### Unsupervised Video Tone Mapping Dataset (UVTM Dataset)

You can download our dataset from [Baidu Netdisk](https://pan.baidu.com/s/1X-FRzSMqYc97nlKXdJce7Q) (key: 6jl2).

## Code

### Dependencies and Installation

- Python >= 3.5
- Pytorch >= 1.10

### Prepare Data

For image tome mapping, you can download training data from [Baidu Netdisk](https://pan.baidu.com/s/10AC_UpjAtttD1EJBc_wVpg) (key: hesn), and download HDR Survey, HDRI Haven, and LVZ-HDR dataset as test data. For video tone mapping, you need to add UVTM dataset for training and testing.

### Test

You can download pretrained weights from [Baidu Netdisk](https://pan.baidu.com/s/1LJwoanmPY0AqUafNqlCX_g) (key: b6jm), then run the following commands for image and video TMO testing:
  ```
  cd activate_trained_model
  sh run_imageTMO_test_on_HDRSurveyDataset.sh
  sh run_videoTMO_test_on_UVTMTestDataset.sh
  ```


### Train

Run the following commands for image and video TMO training
  ```
  bash run_imageTMO_train.sh
  bash run_videoTMO_train.sh
  ```


