#!/bin/bash
set -euo pipefail
mkdir -p ../outputs/baseline_valid10_step2
python merge_pkl.py --input_pklz ../../sn-gamestate/outputs/gsr/baseline_valid10/step1/states/sn-gamestate.pklz --dataset_root /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS --output_dir ../outputs/baseline_valid10_step2 --split valid --fix_duplicate_track_ids --save_refined_pklz --save_pklz_path ../outputs/baseline_valid10_step2/refined_sn-gamestate.pklz --output_pkl ../outputs/baseline_valid10_step2/results.pkl --include_unmatched_segments --video_id_list 021,023,034,040,041,051,052,085,091,093
