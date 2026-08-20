#!/bin/bash
set -euo pipefail
INPUT_PKLZ="../../sn-gamestate/outputs/gsr/repro_official_test/step1_full/states/sn-gamestate.pklz"
DATASET_ROOT="/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS"
OUTPUT_DIR="../outputs/repro_official_test_step2_full"
SPLIT="test"
mkdir -p "$OUTPUT_DIR"
python merge_pkl.py \
  --input_pklz "$INPUT_PKLZ" --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" --split "$SPLIT" \
  --fix_duplicate_track_ids --save_refined_pklz \
  --save_pklz_path "$OUTPUT_DIR/refined_sn-gamestate.pklz" \
  --output_pkl "$OUTPUT_DIR/results.pkl" --include_unmatched_segments \
  --video_id_list 116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,187,188,189,190,191,192,193,194,195,196,197,198,199,200
