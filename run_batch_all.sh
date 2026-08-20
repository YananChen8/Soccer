#!/bin/bash
# Master batch launcher: starts all 9 calibration-only experiments in BACKGROUND
# across the given GPUs, then waits for them all to finish.
# Usage: bash run_batch.sh <gpu0> <gpu1> <gpu2> ...
set -euo pipefail

RUNNER=/remote-home/jiayuanrao/yishan/run_one_exp.sh
CKPT_BASE=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/quick_subset12
LOG_DIR=/remote-home/jiayuanrao/yishan/temporal_calib_logs
mkdir -p "$LOG_DIR"

GPUS=("$@")
N=${#GPUS[@]}
echo "Launching 9 experiments across $N GPUs: ${GPUS[*]}"

declare -A PIDS
i=0

launch_one() {
    local name=$1 ckpt=$2 scale=$3
    local gpu=${GPUS[$((i % N))]}
    local log="$LOG_DIR/${name}.log"
    i=$((i+1))
    echo "[$(date '+%H:%M:%S')] LAUNCH $name gpu=$gpu scale=$scale"
    bash "$RUNNER" "$name" "$ckpt" "$scale" "$gpu" >> "$log" 2>&1 &
    PIDS[$name]=$!
}

launch_one "baseline_rs0"         ""                                                     "0"
launch_one "3dcnn_k15_rs0.5"      "$CKPT_BASE/3dcnn_k15/kp_adapter_3dcnn_k15.pt"         "0.5"
launch_one "3dcnn_k15_rs1.0"      "$CKPT_BASE/3dcnn_k15/kp_adapter_3dcnn_k15.pt"         "1.0"
launch_one "tcn_k50_rs0.5"        "$CKPT_BASE/tcn_k50/kp_adapter_tcn_k50.pt"             "0.5"
launch_one "tcn_k50_rs1.0"        "$CKPT_BASE/tcn_k50/kp_adapter_tcn_k50.pt"             "1.0"
launch_one "stgcn_k50_rs0.5"      "$CKPT_BASE/stgcn_k50/kp_adapter_stgcn_k50.pt"         "0.5"
launch_one "stgcn_k50_rs1.0"      "$CKPT_BASE/stgcn_k50/kp_adapter_stgcn_k50.pt"         "1.0"
launch_one "transformer_k50_rs0.5" "$CKPT_BASE/transformer_k50/kp_adapter_transformer_k50.pt" "0.5"
launch_one "transformer_k50_rs1.0" "$CKPT_BASE/transformer_k50/kp_adapter_transformer_k50.pt" "1.0"

echo ""
echo "All 9 launched. Waiting for completion..."
for name in "${!PIDS[@]}"; do
    wait "${PIDS[$name]}"
    echo "[$(date '+%H:%M:%S')] DONE $name (exit=$?)"
done

echo ""
echo "===== ALL 9 CALIBRATION JOBS DONE ====="
echo "Output dirs:"
ls -d /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/quick_subset12_smoke_test_parallel/*/
