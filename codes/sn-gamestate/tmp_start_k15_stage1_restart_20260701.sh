#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate

PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k15_stage1_restart_20260701
CACHE=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_memmap_cache_20260701/train
RUN=fullft_cached_k15_stage1_motion_lastpair_fast_e5_restart_stepckpt
OUT="$ROOT/$RUN"
mkdir -p "$OUT"

cat > "$OUT/launch_config.txt" <<EOF
host=$(hostname)
gpu=7
run=$RUN
fusion_level=stage1
window_size=15
batch_size=1
epochs=5
workers=4
loss=last_pair_motion_residual
save_every_steps=2000
auto_balance_steps=100
peak_target_ratio=0.3
motion_target_ratio=0.5
residual_scale=0.02
cache=$CACHE
EOF

nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH=plugins/calibration:. "$PY" -u tmp_train_temporal_hrnet_cached_fullft_20260701.py \
  --cache-dir "$CACHE" \
  --out-dir "$OUT" \
  --fusion-level stage1 \
  --window-size 15 \
  --epochs 5 \
  --batch-size 1 \
  --workers 4 \
  --hrnet-lr 3e-6 \
  --adapter-lr 3e-5 \
  --auto-balance-steps 100 \
  --peak-target-ratio 0.3 \
  --motion-target-ratio 0.5 \
  --residual-scale 0.02 \
  --save-every-steps 2000 \
  --resume \
  --log-every 100 \
  > "$OUT/train.log" 2>&1 &
echo $! > "$OUT/train.pid"
echo "started $RUN pid=$(cat "$OUT/train.pid") gpu=7"

nohup env PYTHONPATH=plugins/calibration:. "$PY" -u tmp_wait_eval_k15_stage1_restart_20260701.py \
  > "$ROOT/wait_eval_stdout.log" 2>&1 &
echo $! > "$ROOT/wait_eval.pid"
echo "started wait_eval pid=$(cat "$ROOT/wait_eval.pid")"
