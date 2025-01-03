<!-- # SVFR: A Unified Framework for Generalized Video Face Restoration -->

<div>
<h1>SVFR: A Unified Framework for Generalized Video Face Restoration</h1>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/pdf/2501.01235)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://wangzhiyaoo.github.io/SVFR/)

## 🔥 Overview

SVFR is a unified framework for face video restoration that supports tasks such as **BFR, Colorization, Inpainting**, and **their combinations** within one cohesive system.

<img src="assert/method.png">

## 🎬 Demo

### BFR
<!-- 
<div style="display: flex; gap: 10px;">
  <video controls width="360">
    <source src="https://wangzhiyaoo.github.io/SVFR/static/videos/wild-test/case1_bfr.mp4" type="video/mp4">
    
  </video>
  
  <video controls width="360">
    <source src="https://wangzhiyaoo.github.io/SVFR/static/videos/wild-test/case4_bfr.mp4" type="video/mp4">
    
  </video>
</div> -->


<!-- <div style="display: flex; gap: 10px;">
  <video src="https://github.com/user-attachments/assets/49f985f3-a2db-4b9f-aed0-e9943bae9c17" controls width=45%></video>
  <video src="https://github.com/user-attachments/assets/8fcd1dd9-79d3-4e57-b98e-a80ae2badfb5" controls width="45%"></video>
</div> -->

| Case1                                                                                                                        | Case2                                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
|<video src="https://github.com/user-attachments/assets/49f985f3-a2db-4b9f-aed0-e9943bae9c17" /> | <video src="https://github.com/user-attachments/assets/8fcd1dd9-79d3-4e57-b98e-a80ae2badfb5" /> |


<!-- <video src="https://wangzhiyaoo.github.io/SVFR/bfr"> -->



<!-- https://github.com/user-attachments/assets/49f985f3-a2db-4b9f-aed0-e9943bae9c17
  
https://github.com/user-attachments/assets/8fcd1dd9-79d3-4e57-b98e-a80ae2badfb5 -->





### BFR+Colorization
<!-- <div style="display: flex; gap: 10px;">
  <video controls width="360">
    <source src="https://wangzhiyaoo.github.io/SVFR/static/videos/wild-test/case10_bfr_colorization.mp4" type="video/mp4">
    
  </video>
  
  <video controls width="360">
    <source src="https://wangzhiyaoo.github.io/SVFR/static/videos/wild-test/case12_bfr_colorization.mp4" type="video/mp4">
    
  </video>
</div> -->


<!-- https://github.com/user-attachments/assets/795f4cb1-a7c9-41c5-9486-26e64a96bcf0

https://github.com/user-attachments/assets/6ccf2267-30be-4553-9ecc-f3e7e0ca1d6f -->

| Case3                                                                                                                        | Case4                                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
|<video src="https://github.com/user-attachments/assets/795f4cb1-a7c9-41c5-9486-26e64a96bcf0" /> | <video src="https://github.com/user-attachments/assets/6ccf2267-30be-4553-9ecc-f3e7e0ca1d6f" /> |


### BFR+Colorization+Inpainting
<!-- <div style="display: flex; gap: 10px;">
  <video controls width="360">
    <source src="https://wangzhiyaoo.github.io/SVFR/static/videos/wild-test/case14_bfr+colorization+inpainting.mp4" type="video/mp4">
    
  </video>
  
  <video controls width="360">
    <source src="https://wangzhiyaoo.github.io/SVFR/static/videos/wild-test/case15_bfr+colorization+inpainting.mp4" type="video/mp4">
    
  </video>
</div> -->



<!-- https://github.com/user-attachments/assets/6113819f-142b-4faa-b1c3-a2b669fd0786

https://github.com/user-attachments/assets/efdac23c-0ba5-4dad-ab8c-48904af5dd89
 -->


| Case5                                                                                                                        | Case6                                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
|<video src="https://github.com/user-attachments/assets/6113819f-142b-4faa-b1c3-a2b669fd0786" /> | <video src="https://github.com/user-attachments/assets/efdac23c-0ba5-4dad-ab8c-48904af5dd89" /> |


## 🎙️ News

- **[2025.01.02]**: We released the initial version of the [inference code](#inference) and [models](#download-checkpoints). Stay tuned for continuous updates!
- **[2024.12.17]**: This repo is created!

## 🚀 Getting Started

## Setup

Use the following command to install a conda environment for SVFR from scratch:

```bash
conda create -n svfr python=3.9 -y
conda activate svfr
```

Install PyTorch:  make sure to select the appropriate CUDA version based on your hardware, for example,

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

## Download checkpoints

<li>Download the Stable Video Diffusion</li>

```
conda install git-lfs
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt models/stable-video-diffusion-img2vid-xt
```

<li>Download SVFR</li>

You can download checkpoints manually through link on [Google Drive](https://drive.google.com/drive/folders/1nzy9Vk-yA_DwXm1Pm4dyE2o0r7V6_5mn?usp=share_link).

Put checkpoints as follows:

```
└── models
    ├── face_align
    │   ├── yoloface_v5m.pt
    ├── face_restoration
    │   ├── unet.pth
    │   ├── id_linear.pth
    │   ├── insightface_glint360k.pth
    └── stable-video-diffusion-img2vid-xt
        ├── vae
        ├── scheduler
        └── ...
```

## Inference

### Inference single or multi task

```
python3 infer.py \
 --config config/infer.yaml \
 --task_ids 0 \
 --input_path ./assert/lq/lq1.mp4 \
 --output_dir ./results/ 
```

<li>task_id:</li>

> 0 -- bfr  
> 1 -- colorization  
> 2 -- inpainting  
> 0,1 -- bfr and colorization  
> 0,1,2 -- bfr and colorization and inpainting  
> ...

### Inference with additional inpainting mask

```
# For Inference with Inpainting
# Add '--mask_path' if you need to specify the mask file.

python3 infer.py \
 --config config/infer.yaml \
 --task_ids 0,1,2 \
 --input_path ./assert/lq/lq3.mp4 \
 --output_dir ./results/ 
 --mask_path ./assert/mask/lq3.png
```


## BibTex
```
@misc{wang2025svfrunifiedframeworkgeneralized,
      title={SVFR: A Unified Framework for Generalized Video Face Restoration}, 
      author={Zhiyao Wang and Xu Chen and Chengming Xu and Junwei Zhu and Xiaobin Hu and Jiangning Zhang and Chengjie Wang and Yuqi Liu and Yiyi Zhou and Rongrong Ji},
      year={2025},
      eprint={2501.01235},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01235}, 
}
```
