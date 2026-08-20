#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate

PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701
CACHE=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_memmap_cache_20260701/train
if [ "${RESET_OUTPUTS:-0}" = "1" ]; then
  rm -rf "$ROOT"
fi
mkdir -p "$ROOT"

start_one() {
  local gpu="$1"
  local run="$2"
  local level="$3"
  local k="$4"
  local bs="$5"
  local scale="$6"
  local out="$ROOT/$run"
  mkdir -p "$out"
  cat > "$out/launch_config.txt" <<EOF
host=$(hostname)
gpu=$gpu
run=$run
fusion_level=$level
window_size=$k
batch_size=$bs
epochs=5
workers=4
loss=last_pair_motion_residual
auto_balance_steps=100
peak_target_ratio=0.3
motion_target_ratio=0.5
residual_scale=$scale
cache=$CACHE
EOF
  if pgrep -f "tmp_train_temporal_hrnet_cached_fullft_20260701.py.*$out" >/dev/null; then
    echo "already running $run"
    return 0
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=plugins/calibration:. "$PY" -u tmp_train_temporal_hrnet_cached_fullft_20260701.py \
    --cache-dir "$CACHE" \
    --out-dir "$out" \
    --fusion-level "$level" \
    --window-size "$k" \
    --epochs 5 \
    --batch-size "$bs" \
    --workers 4 \
    --hrnet-lr 3e-6 \
    --adapter-lr 3e-5 \
    --auto-balance-steps 100 \
    --peak-target-ratio 0.3 \
    --motion-target-ratio 0.5 \
    --residual-scale "$scale" \
    --resume \
    --log-every 100 \
    > "$out/train.log" 2>&1 &
  echo $! > "$out/train.pid"
  echo "started $run pid=$(cat "$out/train.pid") gpu=$gpu"
}

# Do not use host200 GPUs 1, 5, 6.
# K15 last is too slow for this branch (~0.8 samples/s); use feature caching before revisiting it.
start_one 2 fullft_cached_k15_stage1_motion_lastpair_fast_e5 stage1 15 2 0.02
start_one 3 fullft_cached_k5_last_motion_lastpair_fast_e5 last 5 4 0.05
start_one 4 fullft_cached_k5_stage1_motion_lastpair_fast_e5 stage1 5 8 0.02

echo "root=$ROOT"
