#!/bin/bash
# Run 9 temporal adapter smoke tests on SNGS-116/117/118 (test split).
# Reuses SAM3-ft12ep-4prompt detection pklz + full pipeline.
# Only replacing pitch calibration via "modules/pitch=temporal_nbjw_calib".
#
# Config:   gsr_step_3_test_15A (same as prior 3dcnn gshota_test)
# Adapter:  KP_ADAPTER_CKPT env var → kp_adapter_ckpt in temporal_nbjw_calib.yaml
# Scale:    ADAPTER_RESIDUAL_SCALE env var → residual_scale override in _load_adapter
set -euo pipefail

source /remote-home/jiayuanrao/tools/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate wys_soccermaster

SNGSR=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
TRACKLAB=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/tracklab

BASE_OUT=$SNGSR/outputs/gsr/temporal_hrnet/quick_subset12_smoke_test
CKPT_BASE=$SNGSR/outputs/gsr/temporal_hrnet/quick_subset12
GPU=${GPU:-0}
VID_LIST="['SNGS-116','SNGS-117','SNGS-118']"
DET_PKLZ=/remote-home/jiayuanrao/yishan/SoccerMaster/experiments/detection_benchmark/runs/eval_sam3_ft12ep_4p_test/states/sam3_4p_test_roles.pklz

mkdir -p "$BASE_OUT"

run_one() {
    local name=$1
    local ckpt=$2
    local residual_scale=$3

    local out_dir="$BASE_OUT/${name}_rs${residual_scale}"
    local log_file="$out_dir/main.log"

    echo "[$(date '+%F %T')] >>> START ${name}_rs${residual_scale} <<<" | tee -a /remote-home/jiayuanrao/yishan/temporal_smoke_progress.log

    mkdir -p "$out_dir/states"

    # PRTReId writes to reid/0 relative to cwd (project root) — shared across
    # experiments.  Delete before each run to avoid FileExistsError.
    rm -rf "$SNGSR/reid/0" 2>/dev/null || true

    cd "$SNGSR"

    OVERRIDES=(
        "experiment_subname=temporal_hrnet/quick_subset12_smoke_test/${name}_rs${residual_scale}"
        "dataset.vids_dict.test=$VID_LIST"
        "modules/pitch=temporal_nbjw_calib"
        "state.load_file=$DET_PKLZ"
        "state.save_file=$out_dir/states/sn-gamestate.pklz"
        "visualization.cfg.save_videos=False"
        "eval_tracking=True"
        "test_tracking=True"
    )

    KP_ADAPTER_CKPT="$ckpt" \
    ADAPTER_RESIDUAL_SCALE="$residual_scale" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    SLURM_JOBID="$(date +%s)" \
    PYTHONPATH="$TRACKLAB:$SNGSR:" \
    python -m tracklab.main \
        -cn gsr_step_3_test_15A \
        "${OVERRIDES[@]}" \
        2>&1 | tee "$log_file"

    echo "[$(date '+%F %T')] >>> DONE ${name}_rs${residual_scale} <<<" | tee -a /remote-home/jiayuanrao/yishan/temporal_smoke_progress.log
}

# ---------------------------------------------------------------------------
# 1) BASELINE — pass-through (adapter=None → byte-identical to nbjw default)
# ---------------------------------------------------------------------------
run_one "baseline" "" 0

# ---------------------------------------------------------------------------
# 2) 3DCNN K15 — TemporalHeatmapAdapter (dense depthwise Conv3D, K=15)
# ---------------------------------------------------------------------------
CKPT_3DCNN="$CKPT_BASE/3dcnn_k15/kp_adapter_3dcnn_k15.pt"
run_one "3dcnn_k15" "$CKPT_3DCNN" 0.5
run_one "3dcnn_k15" "$CKPT_3DCNN" 1.0

# ---------------------------------------------------------------------------
# 3) TCN K50 — KeypointTokenTemporalAdapter (sparse tokens, TCN, K=50)
# ---------------------------------------------------------------------------
CKPT_TCN="$CKPT_BASE/tcn_k50/kp_adapter_tcn_k50.pt"
run_one "tcn_k50" "$CKPT_TCN" 0.5
run_one "tcn_k50" "$CKPT_TCN" 1.0

# ---------------------------------------------------------------------------
# 4) STGCN K50 — KeypointTokenTemporalAdapter (sparse tokens + graph conv, K=50)
# ---------------------------------------------------------------------------
CKPT_STGCN="$CKPT_BASE/stgcn_k50/kp_adapter_stgcn_k50.pt"
run_one "stgcn_k50" "$CKPT_STGCN" 0.5
run_one "stgcn_k50" "$CKPT_STGCN" 1.0

# ---------------------------------------------------------------------------
# 5) Transformer K50 — KeypointTokenTemporalAdapter (sparse tokens + self-attn, K=50)
# ---------------------------------------------------------------------------
CKPT_TF="$CKPT_BASE/transformer_k50/kp_adapter_transformer_k50.pt"
run_one "transformer_k50" "$CKPT_TF" 0.5
run_one "transformer_k50" "$CKPT_TF" 1.0

echo "[$(date '+%F %T')] ===== ALL 9 EXPERIMENTS DONE =====" | tee -a /remote-home/jiayuanrao/yishan/temporal_smoke_progress.log
