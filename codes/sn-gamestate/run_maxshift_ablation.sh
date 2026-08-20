#!/usr/bin/env bash
set -euo pipefail

SNG=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
cd "$SNG"

GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0 2 3)
fi
for gpu in "${GPUS[@]}"; do
  if [ "$gpu" = "6" ]; then
    echo "Refusing to use banned GPU 202:6" >&2
    exit 2
  fi
done

export PYTHONPATH=plugins/calibration:.
export PYTHONUNBUFFERED=1
VIDEOS=(
  SNGS-060 SNGS-065 SNGS-070 SNGS-075 SNGS-099 SNGS-104
  SNGS-110 SNGS-115 SNGS-155 SNGS-160 SNGS-165 SNGS-170
)
MS_VALUES=(5 10 20)
LOG_DIR=outputs/gsr/temporal_hrnet/logs/maxshift_ablation
mkdir -p "$LOG_DIR"

echo "START max_shift ablation $(date '+%F %T')"
for idx in "${!MS_VALUES[@]}"; do
  ms="${MS_VALUES[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  out="outputs/gsr/temporal_hrnet/quick_subset12/stgcn_k50_ms${ms}"
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "START ms=$ms gpu=$gpu $(date '+%F %T')"
    "$PY" experiments/detection_benchmark/train_temporal_token_adapter_online.py \
      --split train \
      --videos "${VIDEOS[@]}" \
      --architecture stgcn \
      --window-size 50 \
      --epochs 1 \
      --hrnet-batch-size 32 \
      --hidden 64 \
      --residual-scale 1.0 \
      --max-shift-px "$ms" \
      --out "$out/kp_adapter_stgcn_k50_ms${ms}.pt"
    echo "DONE ms=$ms gpu=$gpu $(date '+%F %T')"
  ) > "$LOG_DIR/stgcn_k50_ms${ms}.log" 2>&1 &
done
wait
echo "DONE max_shift ablation $(date '+%F %T')"
