#!/bin/bash
# Per-experiment bash wrapper: activates conda, then runs the tracklab pipeline.
# Args: <name> <ckpt_path_or_empty> <residual_scale> <gpu_id>
set -euo pipefail

source /remote-home/jiayuanrao/tools/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate wys_soccermaster

EXP_NAME="$1"
ADAPTER_CKPT="$2"
SCALE="$3"
GPU="$4"

SNGSR=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
TRACKLAB=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/tracklab
CKPT_BASE="$SNGSR/outputs/gsr/temporal_hrnet/quick_subset12"
OUT_BASE="$SNGSR/outputs/gsr/temporal_hrnet/quick_subset12_smoke_test_parallel"
DET_PKLZ=/remote-home/jiayuanrao/yishan/SoccerMaster/experiments/detection_benchmark/runs/eval_sam3_ft12ep_4p_test/states/sam3_4p_test_roles.pklz

OUT_DIR="$OUT_BASE/$EXP_NAME"
mkdir -p "$OUT_DIR/states"

# Reuse existing pklz if already built
PKLZ="$OUT_DIR/states/sn-gamestate.pklz"
if [ -f "$PKLZ" ] && [ "$(stat -c%s "$PKLZ" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    echo "[$(date '+%F %T')] SKIP $EXP_NAME: pklz already exists ($(du -sh "$PKLZ" | cut -f1))" | tee "$OUT_DIR/main.log"
    exit 0
fi

# Clean reid cache
rm -rf "$SNGSR/reid/0" 2>/dev/null || true
cd "$SNGSR"

export KP_ADAPTER_CKPT="$ADAPTER_CKPT"
export ADAPTER_RESIDUAL_SCALE="$SCALE"
export CUDA_VISIBLE_DEVICES="$GPU"
export SLURM_JOBID="$(date +%s)"
export PYTHONPATH="$TRACKLAB:$SNGSR:"

echo "[$(date '+%F %T')] START $EXP_NAME gpu=$GPU scale=$SCALE ckpt=$ADAPTER_CKPT" | tee "$OUT_DIR/main.log"

python -m tracklab.main \
    -cn gsr_step_3_sam3_4p_test \
    "experiment_subname=temporal_hrnet/quick_subset12_smoke_test_parallel/${EXP_NAME}" \
    "dataset.vids_dict.test=['SNGS-116','SNGS-117','SNGS-118']" \
    "modules/pitch=temporal_nbjw_calib" \
    "state.load_file=$DET_PKLZ" \
    "state.save_file=$PKLZ" \
    "visualization.cfg.save_videos=False" \
    "eval_tracking=False" \
    "test_tracking=True" \
    >> "$OUT_DIR/main.log" 2>&1

EC=$?
echo "[$(date '+%F %T')] DONE $EXP_NAME exit=$EC" | tee -a "$OUT_DIR/main.log"
exit $EC
