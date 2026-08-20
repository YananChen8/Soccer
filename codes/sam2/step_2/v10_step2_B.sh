#!/bin/bash
set -euo pipefail
mkdir -p ../outputs/baseline_valid10_step2
python inference.py --sam_checkpoint ../checkpoints/sam2.1_hiera_large.pt --sam_config configs/sam2.1/sam2.1_hiera_l.yaml --input_pklz ../../sn-gamestate/outputs/gsr/baseline_valid10/step1/states/sn-gamestate.pklz --dataset_root /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS --output_dir ../outputs/baseline_valid10_step2 --split valid --fps 25 --best_iou_threshold 0.5 --best_seg_bbox_be_overlapped_ratio_threshold 0.7 --mask_iou_threshold 0.6 --seg_mask_be_overlapped_ratio_threshold 0.6 --max_expansion_ratio 1.0 --max_width_offset 30 --max_height_offset 60 --kernel_size 10 --fix_duplicate_track_ids --gpu_list 7 --max_processes_per_gpu 1 --video_id_list 051,052,085,091,093
