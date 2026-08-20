#!/bin/bash
set -euo pipefail
mkdir -p "../outputs/repro_official_test_step2_15"
python merge_pkl.py --input_pklz "../../sn-gamestate/outputs/gsr/repro_official_test/step1_full/states/sn-gamestate.pklz" --dataset_root "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS" --output_dir "../outputs/repro_official_test_step2_15" --split "test" --fix_duplicate_track_ids --save_refined_pklz --save_pklz_path "../outputs/repro_official_test_step2_15/refined_sn-gamestate.pklz" --output_pkl "../outputs/repro_official_test_step2_15/results.pkl" --include_unmatched_segments --video_id_list 116,117,118,119,120,121,122,123,124,125,126,127,128,129,130
