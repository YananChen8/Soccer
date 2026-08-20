#!/bin/bash
set -euo pipefail

SAM_CHECKPOINT="../checkpoints/sam2.1_hiera_large.pt"
SAM_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
INPUT_PKLZ="../../sn-gamestate/outputs/gsr/repro_official_test/step1_smoke/states/sn-gamestate.pklz"
DATASET_ROOT="/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS"
OUTPUT_DIR="../outputs/repro_official_test_step2_smoke"
SPLIT="test"
FPS=25
BEST_IOU_THRESHOLD=0.5
BEST_SEG_BBOX_BE_OVERLAPPED_RATIO_THRESHOLD=0.7
MASK_IOU_THRESHOLD=0.6
SEG_MASK_BE_OVERLAPPED_RATIO_THRESHOLD=0.6
MAX_EXPANSION_RATIO=1.0
MAX_WIDTH_OFFSET=30
MAX_HEIGHT_OFFSET=60
KERNEL_SIZE=10
GPU_LIST="${GPU_LIST:-0}"
MAX_PROCESSES_PER_GPU="${MAX_PROCESSES_PER_GPU:-1}"
VIDEO_ID_LIST="116,117,118"
mkdir -p "$OUTPUT_DIR"
python inference.py \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_config "$SAM_CONFIG" \
  --input_pklz "$INPUT_PKLZ" \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --fps "$FPS" \
  --best_iou_threshold "$BEST_IOU_THRESHOLD" \
  --best_seg_bbox_be_overlapped_ratio_threshold "$BEST_SEG_BBOX_BE_OVERLAPPED_RATIO_THRESHOLD" \
  --mask_iou_threshold "$MASK_IOU_THRESHOLD" \
  --seg_mask_be_overlapped_ratio_threshold "$SEG_MASK_BE_OVERLAPPED_RATIO_THRESHOLD" \
  --max_expansion_ratio "$MAX_EXPANSION_RATIO" \
  --max_width_offset "$MAX_WIDTH_OFFSET" \
  --max_height_offset "$MAX_HEIGHT_OFFSET" \
  --kernel_size "$KERNEL_SIZE" \
  --fix_duplicate_track_ids \
  --gpu_list "$GPU_LIST" \
  --max_processes_per_gpu "$MAX_PROCESSES_PER_GPU" \
  --video_id_list "$VIDEO_ID_LIST"
