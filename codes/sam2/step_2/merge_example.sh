#!/bin/bash

# Set base paths
INPUT_PKLZ="../../sn-gamestate/outputs/gsr/step_1_sn500_10001_10002/states/sn-gamestate.pklz"  # Input pklz file path
DATASET_ROOT="../datasets/SoccerNetGS"  # Video frames directory
OUTPUT_DIR="../outputs/gsr_step2_sn500_10001_10002"                              # Output directory
SPLIT="sn500"                                         # Data split (train, valid, test)
OUTPUT_PKL="${OUTPUT_DIR}/results.pkl"     # Result output pkl file path

# Video ID range
VIDEO_ID_START=10001
VIDEO_ID_END=10002

# Create output directory
mkdir -p $OUTPUT_DIR

SAVE_PKLZ_PATH="${OUTPUT_DIR}/refined_sn-gamestate.pklz"   # Custom save pklz path
python merge_pkl.py \
  --input_pklz $INPUT_PKLZ \
  --dataset_root $DATASET_ROOT \
  --output_dir $OUTPUT_DIR \
  --split $SPLIT \
  --fix_duplicate_track_ids \
  --save_refined_pklz \
  --save_pklz_path $SAVE_PKLZ_PATH \
  --output_pkl $OUTPUT_PKL \
  --include_unmatched_segments \
  --video_id_start $VIDEO_ID_START \
  --video_id_end $VIDEO_ID_END