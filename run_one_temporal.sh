#!/bin/bash
# Launch wrapper: runs a single temporal adapter experiment on server 202.
# Used by the batch runner to avoid SSH quoting hell.
set -euo pipefail

source /remote-home/jiayuanrao/tools/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate wys_soccermaster

SNGSR=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
TRACKLAB=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/tracklab

# Args
ADAPTER_CKPT="${1:-}"       # kp_adapter_ckpt path (empty = baseline)
ADAPTER_WINDOW="${2:-3}"
RESIDUAL_SCALE="${3:-0}"
EXP_NAME="${4:-test}"
OUT_DIR="$SNGSR/outputs/gsr/temporal_hrnet/quick_subset12_smoke_test/${EXP_NAME}"
VID_LIST="['SNGS-116','SNGS-117','SNGS-118']"
DET_PKLZ=/remote-home/jiayuanrao/yishan/SoccerMaster/experiments/detection_benchmark/runs/eval_sam3_ft12ep_4p_test/states/sam3_4p_test_roles.pklz

mkdir -p "$OUT_DIR/states"

# PRTReId directory conflict fix
rm -rf "$SNGSR/reid/0" 2>/dev/null || true

cd "$SNGSR"

KP_ADAPTER_CKPT="$ADAPTER_CKPT" \
ADAPTER_RESIDUAL_SCALE="$RESIDUAL_SCALE" \
CUDA_VISIBLE_DEVICES=0 \
SLURM_JOBID="$(date +%s)" \
python -m tracklab.main \
    -cn gsr_step_3_sam3_4p_test \
    "experiment_subname=temporal_hrnet/quick_subset12_smoke_test/${EXP_NAME}" \
    "dataset.vids_dict.test=$VID_LIST" \
    "modules/pitch=temporal_nbjw_calib" \
    "state.load_file=$DET_PKLZ" \
    "state.save_file=$OUT_DIR/states/sn-gamestate.pklz" \
    "visualization.cfg.save_videos=False" \
    eval_tracking=True \
    test_tracking=True \
    2>&1 | tee "$OUT_DIR/main.log"

echo "DONE: $EXP_NAME  exit=$?"
