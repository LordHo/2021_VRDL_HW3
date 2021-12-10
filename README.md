# 2021_VRDL_HW3

2021 VRDL HW3 Instance Segmentation - Nuclei segmentation  
Codalab Link: <https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5>

## Introduction

Train an instance segmentation model to detect and segment all the nuclei in the image.

## Dataset

Nuclear segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear.  
Download: <https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view>  
Discussion on kaggle: <https://www.kaggle.com/c/data-science-bowl-2018/discussion>

## Environment

### Hardware

OS: Window 10  
GPU: NVIDIA GeForce RTX 2080 Ti

### Installation

## Methodology

Candidate List

- Mask R-CNN (BaseLine, with proper configuration)
- SOLOv2
- Swin-Transformer
- Focal-Transformer

Liberary

- Mask R-CNN
  - [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-your-model)
- SOLOv2
  - [SOLO: Segmenting Objects by Locations](https://github.com/WXinlong/SOLO)
- Swin-Transformer - Swin-L (HTC++, multi scale)
  - [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
  - [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- Focal-Transformer - Focal-L (HTC++, multi-scale)
  - [Focal-Transformer](https://github.com/microsoft/Focal-Transformer)

Dataset

- [Instance Segmentation on COCO test-dev](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=deep-occlusion-aware-instance-segmentation)

### Data Pre-process

### Model Architecture

### Hyperparameters

### Data Post-process

#### Answer Format

Format is same as [COCO Dataset](https://cocodataset.org/#format-results) and set the category_id = 1.  
![answer format](readme_img/answer_format.png)

Segmentation result need to encode by [RLE](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py#L80).  
![mask encoded by RLE flow](readme_img/mask_encoded_by_RLE.png)

## Table of Experiment Results

Using AP@0.5, IoU=.50 means set .50 as threshold of IoU.

## Model Weights

## Analyze

## Summary

## Reference

## Template Disscussion

- 5th place solution (based only on Mask-RCNN)
  - <https://www.kaggle.com/c/data-science-bowl-2018/discussion/56326>

- mask-rcnn and instance segmentation network
  - <https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690>

- Top 5, also with the weights
  - <https://www.kaggle.com/c/data-science-bowl-2018/discussion/56316>

- 10th place Code+Datasets (LB: 0.591) Mask R-CNN single model
  - <https://www.kaggle.com/c/data-science-bowl-2018/discussion/56238>

- [ods ai topcoders, 1st place solution]
  - <https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741>
  - <https://github.com/selimsef/dsb2018_topcoders/>

## Problem & Answer

1. using model.eval() vs torch.no_grad()  
   ![model.eval() vs torch.no_grad()](readme_img/eval%20vs%20no_grad.png)
