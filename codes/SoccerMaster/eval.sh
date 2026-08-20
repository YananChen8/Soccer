#!/bin/bash

CONFIG_PATH="configs/pretrain.yaml"

CHECKPOINT_PATH="./pretrained_models/SoccerMaster"

LOG_DIR="outputs/pretrain/eval_logs"

CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes=1 eval.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --log_dir $LOG_DIR

echo "Evaluation completed! Check the results in $LOG_DIR" 