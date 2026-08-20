#!/bin/bash
set -euo pipefail

ROOT="/remote-home/jiayuanrao/yishan/SoccerMaster"
CONDA_SH="/remote-home/jiayuanrao/tools/anaconda/anaconda3/etc/profile.d/conda.sh"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
source "$CONDA_SH"
conda activate wys_soccermaster
cd "$ROOT"

echo "[$(date)] SoccerMaster smoke pipeline valid/SNGS-021"
echo "Python: $(which python)"

cd "$ROOT/codes/sn-gamestate"
echo "[$(date)] Step 1: detection and tracking"
SLURM_JOBID="$(date +%s)" CUDA_VISIBLE_DEVICES="${STEP1_GPUS:-6,7}" python -m tracklab.main -cn gsr_step_1_valid_021 2>&1 | tee "$LOG_DIR/step1_valid_021.log"
test -f "$ROOT/codes/sn-gamestate/outputs/gsr/step_1_valid_021/states/sn-gamestate.pklz"

cd "$ROOT/codes/sam2/step_2"
echo "[$(date)] Step 2: SAM2 segmentation refinement"
GPU_LIST="${STEP2_GPUS:-6,7}" bash gsr_step2_valid_021.sh 2>&1 | tee "$LOG_DIR/step2_inference_valid_021.log"
bash merge_valid_021.sh 2>&1 | tee "$LOG_DIR/step2_merge_valid_021.log"
test -f "$ROOT/codes/sam2/outputs/gsr_step2_valid_021/refined_sn-gamestate.pklz"

cd "$ROOT/codes/sn-gamestate"
echo "[$(date)] Step 3: remaining modules"
SLURM_JOBID="$(date +%s)" CUDA_VISIBLE_DEVICES="${STEP3_GPUS:-6}" python -m tracklab.main -cn gsr_step_3_valid_021_accelerate 2>&1 | tee "$LOG_DIR/step3_valid_021.log"

echo "[$(date)] Done. Outputs:"
echo "  $ROOT/codes/sn-gamestate/outputs/gsr/step_1_valid_021"
echo "  $ROOT/codes/sam2/outputs/gsr_step2_valid_021"
echo "  $ROOT/codes/sn-gamestate/outputs/gsr/step_3_valid_021"
