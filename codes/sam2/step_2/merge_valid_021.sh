#!/bin/bash
set -euo pipefail

INPUT_PKLZ="../../sn-gamestate/outputs/gsr/step_1_valid_021/states/sn-gamestate.pklz"
DATASET_ROOT="/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS"
OUTPUT_DIR="../outputs/gsr_step2_valid_021"
SPLIT="valid"
OUTPUT_PKL="${OUTPUT_DIR}/results.pkl"
VIDEO_ID_LIST="021"
mkdir -p "$OUTPUT_DIR"
SAVE_PKLZ_PATH="${OUTPUT_DIR}/refined_sn-gamestate.pklz"
python merge_pkl.py \
  --input_pklz "$INPUT_PKLZ" \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --fix_duplicate_track_ids \
  --save_refined_pklz \
  --save_pklz_path "$SAVE_PKLZ_PATH" \
  --output_pkl "$OUTPUT_PKL" \
  --include_unmatched_segments \
  --video_id_list "$VIDEO_ID_LIST"
