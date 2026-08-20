# SoccerMaster: A Vision Foundation Model for Soccer Understanding
Official implementation of **SoccerMaster: A Vision Foundation Model for Soccer Understanding** (CVPR 2026 Oral)

[*Haolin Yang*](https://haolinyang-hlyang.github.io/),
[*Jiayuan Rao*](https://jyrao.github.io/),
[*Haoning Wu*](https://haoningwu3639.github.io/),
[*Weidi Xie*](https://weidixie.github.io/)


<div style="line-height: 1;">
  <a href="https://haolinyang-hlyang.github.io/SoccerMaster/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website🌐-SoccerMaster-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/pdf/2512.11016" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv📄-SoccerMaster-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center">
   <img src="./images/teaser.jpg">
</div>
<p><em><strong>SoccerMaster</strong> is a unified soccer-specific vision foundation model that leverages diverse soccer content, including images and videos, to support a wide range of soccer understanding tasks, such as commentary generation, detection, tracking, classification, <em>etc</em>.</em></p>
<div align="center">
   <img src="./images/arch.jpg">
</div>
<p><em><strong>SoccerMaster Architecture.</strong> (a) The architecture of SoccerMaster, which encodes both soccer videos and images through spatial and temporal attention modules to generate semantically rich representations. (b) The pretraining tasks and downstream adaptations of SoccerMaster across both spatial perception and semantic understanding tasks.</em></p>

## Quick Start

### 1. Environment Setup

Create a conda environment and install dependencies:

```bash
conda create -n tracklab_release python=3.10.16
conda activate tracklab_release

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install setuptools==78.1.1
cd ./codes/sn-gamestate
pip install -e .
cd ../tracklab
pip install -e .
cd ../sam2
pip install -e .

pip install albumentations==1.4.19
pip install git+https://github.com/huggingface/transformers
pip install accelerate==1.8.1
pip install "qwen-vl-utils[decord]==0.0.8"
pip install flash-attn --no-build-isolation
```

### 2. Download Pretrained Models

First, download the SigLIP2 backbone model to the pretrained models directory:

```bash
pip install huggingface_hub
hf download google/siglip2-large-patch16-512 --local-dir ./codes/SoccerMaster/pretrained_models/google/siglip2-large-patch16-512
```

### 3. Download SoccerMaster Checkpoints

Clone the SoccerMaster model checkpoints from Hugging Face:

```bash
cd SoccerMaster/codes/SoccerMaster/pretrained_models
mkdir SoccerMaster
hf download xleprime/SoccerMaster --local-dir SoccerMaster
```

After completing these steps, you should have the following directory structure:
```
root/codes/SoccerMaster/pretrained_models/
├── google/
│   └── siglip2-large-patch16-512/
└── SoccerMaster/
```

### 4. Prepare Pretrained Models

Please organize `codes/sn-gamestate/pretrained_models/` as follows:

```
codes/sn-gamestate/pretrained_models/
├── calibration
│   ├── mean.npy
│   ├── pnl_SV_kp
│   ├── pnl_SV_lines
│   ├── Radar.png
│   ├── std.npy
│   ├── SV_kp
│   └── SV_lines
├── google
│   └── siglip2-large-patch16-512
├── jn
│   ├── Qwen2.5-VL-72B-Instruct
│   └── Qwen2.5-VL-7B-Instruct
├── legibility
│   └── legibility_resnet34_soccer_20240215.pth
├── reid
│   ├── hrnetv2_w32_imagenet_pretrained.pth
│   └── prtreid-soccernet-baseline.pth.tar
├── SoccerMaster
│   └── ...
└── yolo
    └── yolo_v8x6_finetuned.pt
```

**Download instructions:**

- **calibration & reid**: Follow the instructions from [SoccerNet Game State Recognition](https://github.com/soccernet/sn-gamestate) to obtain the calibration and ReID pretrained models.
- **google**: Download `siglip2-large-patch16-512` from [Hugging Face](https://huggingface.co/google/siglip2-large-patch16-512).
- **legibility**: Download `legibility_resnet34_soccer_20240215.pth` from [Google Drive](https://drive.google.com/file/d/18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw/view?usp=sharing), following the [jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) project.
- **jn**: Download Qwen2.5-VL models from the [Qwen2.5-VL Collection](https://huggingface.co/collections/Qwen/qwen25-vl) on Hugging Face.
- **yolo**: Move `yolo_v8x6_finetuned.pt` to `codes/sn-gamestate/pretrained_models/yolo/`. This file can be obtained from [https://huggingface.co/xleprime/SoccerMaster](https://huggingface.co/xleprime/SoccerMaster).

After preparing all pretrained models, create a symbolic link so that `codes/SoccerMaster/pretrained_models` points to `codes/sn-gamestate/pretrained_models`:

```bash
cd codes/SoccerMaster
ln -s ../sn-gamestate/pretrained_models pretrained_models
```

## Data Pipeline

### Input Data Format

The data pipeline follows the [SoccerNet Game State Recognition (GSR)](https://github.com/soccernet/sn-gamestate) format. You need to organize your input video frames into the following directory structure. Here we use two video sequences (`SNGS-10001` with 750 frames and `SNGS-10002` with 722 frames) as an example:

```
codes/sn-gamestate/datasets/SoccerNetGS/
├── sequences_info.json
└── sn500/
    ├── SNGS-10001/
    │   └── img1/
    │       ├── 000001.jpg
    │       ├── 000002.jpg
    │       ├── ...
    │       └── 000750.jpg
    └── SNGS-10002/
        └── img1/
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── 000722.jpg
```

Each video sequence should be extracted into individual frames (`.jpg`), placed under `<sequence_name>/img1/`, and named with zero-padded 6-digit indices starting from `000001.jpg`.

You also need to prepare a `sequences_info.json` file that registers all sequences. The format is as follows:

```json
{
    "sn500": [
        {
            "id": 0,
            "name": "SNGS-10001",
            "n_frames": 750
        },
        {
            "id": 1,
            "name": "SNGS-10002",
            "n_frames": 722
        }
    ]
}
```

Each entry contains:
- `id`: a unique integer index for the sequence.
- `name`: the directory name of the sequence (must match the folder name under `sn500/`).
- `n_frames`: the total number of frames in the sequence.

### Pipeline Steps

The data pipeline consists of three steps.

### Step 1: Detection & Tracking

```bash
cd codes/sn-gamestate
CUDA_VISIBLE_DEVICES=0 python -m tracklab.main -cn gsr_step_1_example
```

### Step 2: SAM2 Segmentation Refinement

Use SAM2 to refine the tracking results with precise segmentation masks, then merge the refined results back into the tracker state.

```bash
cd codes/sam2/step_2

# Run SAM2 segmentation inference (supports multi-GPU)
bash gsr_step2_example.sh

# Merge segmentation results back into the tracker state
bash merge_example.sh
```

### Step 3: Remaining Modules

Run the remaining pipeline modules: camera calibration, jersey number recognition, role classification, team assignment, etc.

```bash
cd codes/sn-gamestate
CUDA_VISIBLE_DEVICES=0 python -m tracklab.main -cn gsr_step_3_example_accelerate
```

> **Model selection for Step 3:**
>
> For best results, use `Qwen2.5-VL-72B-Instruct` for jersey number (jn) recognition and `Qwen2.5-VL-7B-Instruct` for role classification (config: `gsr_step_3_example`). However, for faster inference speed and lower GPU memory consumption, you can use the accelerated config (`gsr_step_3_example_accelerate`) which merges both tasks into a single module and uses `Qwen2.5-VL-7B-Instruct` for both, with a minor trade-off in accuracy (Recommended).

## TODO
- [x] Add pretraining code.
- [] Refine pretraining code.
- [x] Release SoccerMaster checkpoints.
- [x] Add instructions for quick start.
- [ ] Release datasets.
- [x] Release data pipeline.

## Citations
If you find our work useful, please cite:
```bibtex
@inproceedings{yang2025soccermaster,
  title={SoccerMaster: A Vision Foundation Model for Soccer Understanding},
  author={Yang, Haolin and Rao, Jiayuan and Wu, Haoning and Xie, Weidi},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
