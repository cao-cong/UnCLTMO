# Unsupervised HDR Image and Video Tone Mapping via Contrastive Learning (UnCLTMO)

This repository contains official implementation of Unsupervised HDR Image and Video Tone Mapping via Contrastive Learning in TCSVT 2023, by Cong Cao, Huanjing Yue, Xin Liu, and Jingyu Yang. [[arxiv]](https://arxiv.org/abs/2303.07327) [[journal]](https://ieeexplore.ieee.org/document/10167696)

<p align="center">
  <img width="800" src="https://github.com/cao-cong/UnCLTMO/blob/main/images/ContrastiveLearningLoss.png">
</p>

## Demo Video

[https://youtu.be/rzXfqiCZtIQ](https://youtu.be/rzXfqiCZtIQ)

## Dataset

### Unsupervised Video Tone Mapping Dataset (UVTM Dataset)

You can download our dataset from [Google Drive](https://drive.google.com/file/d/1IkcfGDlJOAWBRIqSYT71HPxWBGiuXKwl/view?usp=sharing) or [MEGA](https://mega.nz/file/RAgWnTpY#QjIbA_Xs07EZrVIn9qEqO_1LLLOiXgJana6LWTSz-d0) or [Baidu Netdisk](https://pan.baidu.com/s/1X-FRzSMqYc97nlKXdJce7Q) (key: 6jl2).

## Code

### Dependencies and Installation

- Python >= 3.5
- Pytorch >= 1.10

### Prepare Data

For image tome mapping, you can download training data from [Google Drive](https://drive.google.com/drive/folders/1ECjaZenzVx2xDwURziPaI5QNOcEsrCjK?usp=sharing) or [MEGA](https://mega.nz/folder/5JhUGCYS#VyCDrLnNxs4-8j0_PAFqLQ) or [Baidu Netdisk](https://pan.baidu.com/s/10AC_UpjAtttD1EJBc_wVpg) (key: hesn), and download [HDR Survey](http://markfairchild.org/HDR.html), [HDRI Haven](https://zenodo.org/record/1285800#.Yd_d7mhBw2w), and [LVZ-HDR](https://www.kaggle.com/datasets/landrykezebou/lvzhdr-tone-mapping-benchmark-dataset-tmonet) dataset as test data. We also provide preprocessed test data from [Google Drive](https://drive.google.com/drive/folders/1SWRJ5jQrzOywMbwzlVmmZFNxSialT5wS?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/15ml85l4y_L1MPTmtAXYCOA) (key: xrhp), the LVZ-HDR data has been multiplied by dgain 100. For video tone mapping, you need to add UVTM dataset for training and testing.

### Test

You can download pretrained weights from [Google Drive](https://drive.google.com/drive/folders/17MpuVAcQWmZI_ar5Hr0x3d6e_gN4in8K?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1LJwoanmPY0AqUafNqlCX_g) (key: b6jm), then run the following commands for image and video TMO testing:
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

## Citation

If you find our dataset or code helpful in your research or work, please cite our paper:
```
@article{cao2023unsupervised,
  title={Unsupervised HDR Image and Video Tone Mapping via Contrastive Learning},
  author={Cao, Cong and Yue, Huanjing and Liu, Xin and Yang, Jingyu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```


