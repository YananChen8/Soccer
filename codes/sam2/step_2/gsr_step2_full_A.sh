#!/bin/bash
set -euo pipefail
SAM_CHECKPOINT="../checkpoints/sam2.1_hiera_large.pt"
SAM_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
INPUT_PKLZ="../../sn-gamestate/outputs/gsr/repro_official_test/step1_full/states/sn-gamestate.pklz"
DATASET_ROOT="/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS"
OUTPUT_DIR="../outputs/repro_official_test_step2_full"
SPLIT="test"
FPS=25
mkdir -p "$OUTPUT_DIR"
python inference.py \
  --sam_checkpoint "$SAM_CHECKPOINT" --sam_config "$SAM_CONFIG" \
  --input_pklz "$INPUT_PKLZ" --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" --split "$SPLIT" --fps "$FPS" \
  --best_iou_threshold 0.5 --best_seg_bbox_be_overlapped_ratio_threshold 0.7 \
  --mask_iou_threshold 0.6 --seg_mask_be_overlapped_ratio_threshold 0.6 \
  --max_expansion_ratio 1.0 --max_width_offset 30 --max_height_offset 60 --kernel_size 10 \
  --fix_duplicate_track_ids --gpu_list 0 --max_processes_per_gpu 1 \
  --video_id_list 116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140
