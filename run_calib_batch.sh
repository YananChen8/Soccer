#!/bin/bash
# Quick batch launcher for calib_only.py on 9 experiments across given GPUs.
set -euo pipefail
source /remote-home/jiayuanrao/tools/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate wys_soccermaster

GPUS=("$@")
N=${#GPUS[@]}
CKPT_BASE=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/quick_subset12

EXPS=(
    "baseline_rs0||0"
    "3dcnn_k15_rs0.5|3dcnn_k15/kp_adapter_3dcnn_k15.pt|0.5"
    "3dcnn_k15_rs1.0|3dcnn_k15/kp_adapter_3dcnn_k15.pt|1.0"
    "tcn_k50_rs0.5|tcn_k50/kp_adapter_tcn_k50.pt|0.5"
    "tcn_k50_rs1.0|tcn_k50/kp_adapter_tcn_k50.pt|1.0"
    "stgcn_k50_rs0.5|stgcn_k50/kp_adapter_stgcn_k50.pt|0.5"
    "stgcn_k50_rs1.0|stgcn_k50/kp_adapter_stgcn_k50.pt|1.0"
    "transformer_k50_rs0.5|transformer_k50/kp_adapter_transformer_k50.pt|0.5"
    "transformer_k50_rs1.0|transformer_k50/kp_adapter_transformer_k50.pt|1.0"
)

PIDS=()
for i in "${!EXPS[@]}"; do
    IFS='|' read -r name ckpt scale <<< "${EXPS[$i]}"
    gpu=${GPUS[$((i % N))]}
    log="/remote-home/jiayuanrao/yishan/calib_${name}.log"
    echo "[$(date '+%H:%M:%S')] LAUNCH $name gpu=$gpu scale=$scale"
    CUDA_VISIBLE_DEVICES=$gpu python3 /remote-home/jiayuanrao/yishan/calib_only.py \
        "$name" "$ckpt" "$scale" "$gpu" >> "$log" 2>&1 &
    PIDS+=($!)
    sleep 2  # stagger GPU memory allocation
done

echo "All launched. Waiting..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "[$(date '+%H:%M:%S')] PID=$pid done"
done
echo "===== ALL DONE ====="
