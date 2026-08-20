#!/bin/bash

# SAM2 memory inference script - Multi-GPU multi-process version
# This script runs memory_inference.py to process player segmentation in soccer videos

# Set base paths
SAM_CHECKPOINT="../checkpoints/sam2.1_hiera_large.pt"  # SAM2 model checkpoint path
SAM_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"       # SAM2 config file path
INPUT_PKLZ="../../sn-gamestate/outputs/gsr/step_1_sn500_10001_10002/states/sn-gamestate.pklz"
DATASET_ROOT="../datasets/SoccerNetGS"  # Video frames directory
OUTPUT_DIR="../outputs/gsr_step2_sn500_10001_10002"                              # Output directory
SPLIT="sn500"                                         # Data split (train, valid, test, challenge, sn500)
FPS=25                                               # Output video frame rate

# Threshold parameters
BEST_IOU_THRESHOLD=0.5
BEST_SEG_BBOX_BE_OVERLAPPED_RATIO_THRESHOLD=0.7
MASK_IOU_THRESHOLD=0.6
SEG_MASK_BE_OVERLAPPED_RATIO_THRESHOLD=0.6

# Bbox constraint parameters
MAX_EXPANSION_RATIO=1.0
MAX_WIDTH_OFFSET=30
MAX_HEIGHT_OFFSET=60

# Morphological operation parameters
KERNEL_SIZE=10

# Multi-GPU multi-process parameters
GPU_LIST="0,1,2,3,4,5,6,7"  # GPU list to use, adjust according to your actual GPU count
MAX_PROCESSES_PER_GPU=1        # Max processes per GPU, start from 1 and adjust based on GPU memory

# Video ID range
VIDEO_ID_START=10001
VIDEO_ID_END=10002

# Create output directory
mkdir -p $OUTPUT_DIR

echo "GPU list: ${GPU_LIST}"
echo "Max processes per GPU: ${MAX_PROCESSES_PER_GPU}"

python inference.py \
  --sam_checkpoint $SAM_CHECKPOINT \
  --sam_config $SAM_CONFIG \
  --input_pklz $INPUT_PKLZ \
  --dataset_root $DATASET_ROOT \
  --output_dir $OUTPUT_DIR \
  --split $SPLIT \
  --fps $FPS \
  --best_iou_threshold $BEST_IOU_THRESHOLD \
  --best_seg_bbox_be_overlapped_ratio_threshold $BEST_SEG_BBOX_BE_OVERLAPPED_RATIO_THRESHOLD \
  --mask_iou_threshold $MASK_IOU_THRESHOLD \
  --seg_mask_be_overlapped_ratio_threshold $SEG_MASK_BE_OVERLAPPED_RATIO_THRESHOLD \
  --max_expansion_ratio $MAX_EXPANSION_RATIO \
  --max_width_offset $MAX_WIDTH_OFFSET \
  --max_height_offset $MAX_HEIGHT_OFFSET \
  --kernel_size $KERNEL_SIZE \
  --fix_duplicate_track_ids \
  --gpu_list $GPU_LIST \
  --max_processes_per_gpu $MAX_PROCESSES_PER_GPU \
  --video_id_start $VIDEO_ID_START \
  --video_id_end $VIDEO_ID_END