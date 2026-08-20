#!/usr/bin/env bash
set -u

REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
ABS_HUB="$REPO/$HUB"
SCRIPT="$REPO/tmp_watch_official_aux_fullft_20260701.sh"
TRAIN="$REPO/tmp_train_temporal_hrnet_full_finetune_official_aux_20260701.py"
WATCH="$ABS_HUB/_watcher"
CLAIMS="$WATCH/claims"

cd "$REPO" || exit 1
mkdir -p "$WATCH" "$CLAIMS"

runs=(
  "fullft_offaux_last_nomotion_k3:last:0:0.05"
  "fullft_offaux_stage1_nomotion_k3:stage1:0:0.02"
  "fullft_offaux_last_motion_k3:last:0.2:0.05"
  "fullft_offaux_stage1_motion_k3:stage1:0.2:0.02"
)

is_forbidden_gpu() {
  local host="$1"
  local gpu="$2"
  if [ "$host" = "200" ]; then
    [ "$gpu" = "1" ] || [ "$gpu" = "5" ] || [ "$gpu" = "6" ]
    return
  fi
  if [ "$host" = "202" ]; then
    [ "$gpu" = "6" ]
    return
  fi
  return 1
}

gpu_util() {
  nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'
}

gpu_compute_apps() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' | wc -l
}

claimed_count() {
  find "$CLAIMS" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
}

claim_next_run() {
  local host="$1"
  local gpu="$2"
  local item name level motion scale claim
  for item in "${runs[@]}"; do
    IFS=: read -r name level motion scale <<< "$item"
    claim="$CLAIMS/$name"
    if mkdir "$claim" 2>/dev/null; then
      {
        echo "claimed_at=$(date '+%F %T')"
        echo "host=$host"
        echo "gpu=$gpu"
        echo "name=$name"
        echo "level=$level"
        echo "motion_ratio=$motion"
        echo "scale=$scale"
      } > "$claim/claim.txt"
      echo "$name:$level:$motion:$scale"
      return 0
    fi
  done
  return 1
}

write_config() {
  local run_dir="$1"
  local host="$2"
  local gpu="$3"
  local name="$4"
  local level="$5"
  local motion="$6"
  local scale="$7"
  cat > "$run_dir/run_config.json" <<JSON
{
  "run": "$name",
  "host": "$host",
  "gpu": "$gpu",
  "output_root": "$HUB",
  "train_script": "tmp_train_temporal_hrnet_full_finetune_official_aux_20260701.py",
  "eval_script": "experiments/detection_benchmark/eval_temporal_feature_fusion_calib.py",
  "model": {
    "fusion_level": "$level",
    "window_size": 3,
    "residual_scale": $scale,
    "full_finetune": true,
    "line_hrnet": "frozen"
  },
  "train": {
    "split": "train",
    "epochs": 1,
    "batch_size": 4,
    "grad_accum_steps": 2,
    "effective_batch_size": 8,
    "hrnet_lr": 0.000003,
    "adapter_lr": 0.00003,
    "shuffle_chunk_frames": 30,
    "grad_clip_norm": 1.0,
    "amp": true
  },
  "loss": {
    "heat": "official NBJW masked MSE: ((pred-target)^2 * mask).mean()",
    "fg_weight": 0,
    "peak_ratio": 0.2,
    "motion_ratio": $motion,
    "aux_scaling": "detached scale so each active aux term targets ratio * heat, capped by aux_max_scale",
    "aux_max_scale": 200000,
    "presence_masking_removed": true
  },
  "eval": {
    "videos": [116,117,118,119,120,121,122,123],
    "stride": 20,
    "line_threshold_env": 0.1
  }
}
JSON
}

run_one() {
  local host="$1"
  local gpu="$2"
  local name="$3"
  local level="$4"
  local motion="$5"
  local scale="$6"
  local run_dir="$ABS_HUB/$name"
  local ckpt="$run_dir/checkpoints/$name.pt"
  local eval_dir="$run_dir/eval_test116_123_stride20"

  rm -rf "$run_dir"
  mkdir -p "$run_dir/logs" "$run_dir/checkpoints" "$eval_dir"
  write_config "$run_dir" "$host" "$gpu" "$name" "$level" "$motion" "$scale"
  {
    echo "START $(date '+%F %T') host=$host gpu=$gpu name=$name level=$level motion_ratio=$motion scale=$scale"
    echo "batch_size=4 grad_accum=2 effective_batch=8 chunk=30 hrnet_lr=3e-6 adapter_lr=3e-5"
    echo "pid=$$ cwd=$(pwd)"
  } > "$run_dir/status.txt"

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$TRAIN" \
    --split train \
    --fusion-level "$level" \
    --window-size 3 \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --shuffle-chunk-frames 30 \
    --grad-clip-norm 1.0 \
    --log-every 100 \
    --hrnet-lr 3e-6 \
    --adapter-lr 3e-5 \
    --fg-weight 0 \
    --peak-ratio 0.2 \
    --motion-ratio "$motion" \
    --peak-radius-px 5 \
    --peak-beta 20 \
    --residual-scale "$scale" \
    --out "$ckpt" \
    > "$run_dir/logs/train.log" 2>&1
  train_status=$?
  if [ "$train_status" -ne 0 ]; then
    echo "FAILED_TRAIN $(date '+%F %T') status=$train_status" >> "$run_dir/status.txt"
    touch "$run_dir/FAILED_TRAIN"
    exit 0
  fi

  echo "TRAIN_DONE $(date '+%F %T')" >> "$run_dir/status.txt"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  NBJW_LINE_THRESHOLD=0.1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=plugins/calibration:. "$PY" experiments/detection_benchmark/eval_temporal_feature_fusion_calib.py \
    --checkpoint "$ckpt" \
    --videos 116 117 118 119 120 121 122 123 \
    --stride 20 \
    --out "$eval_dir/results.json" \
    > "$eval_dir/eval.log" 2>&1
  eval_status=$?
  if [ "$eval_status" -ne 0 ]; then
    echo "FAILED_EVAL $(date '+%F %T') status=$eval_status" >> "$run_dir/status.txt"
    touch "$run_dir/FAILED_EVAL"
    exit 0
  fi

  echo "DONE $(date '+%F %T')" >> "$run_dir/status.txt"
  touch "$run_dir/DONE"
}

watch_host() {
  local host="$1"
  local max_seconds="${2:-36000}"
  local poll=30
  local need_zero=180
  local start now gpu util apps free_for spec name level motion scale
  declare -A zero_since

  start=$(date +%s)
  echo "WATCH_START $(date '+%F %T') host=$host max_seconds=$max_seconds" >> "$WATCH/watch_${host}.log"
  while true; do
    if [ "$(claimed_count)" -ge 4 ]; then
      echo "WATCH_STOP all_claimed $(date '+%F %T') host=$host" >> "$WATCH/watch_${host}.log"
      exit 0
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$max_seconds" ]; then
      echo "WATCH_STOP timeout $(date '+%F %T') host=$host claimed=$(claimed_count)" >> "$WATCH/watch_${host}.log"
      exit 0
    fi
    for gpu in 0 1 2 3 4 5 6 7; do
      if is_forbidden_gpu "$host" "$gpu"; then
        continue
      fi
      util="$(gpu_util "$gpu")"
      apps="$(gpu_compute_apps "$gpu")"
      if [ "${util:-999}" = "0" ] && [ "${apps:-1}" = "0" ]; then
        if [ -z "${zero_since[$gpu]+x}" ]; then
          zero_since[$gpu]="$now"
        fi
        free_for=$((now - zero_since[$gpu]))
        echo "$(date '+%F %T') host=$host gpu=$gpu util=$util apps=$apps free_for=${free_for}s claimed=$(claimed_count)" >> "$WATCH/watch_${host}.log"
        if [ "$free_for" -ge "$need_zero" ]; then
          if spec="$(claim_next_run "$host" "$gpu")"; then
            IFS=: read -r name level motion scale <<< "$spec"
            echo "LAUNCH $(date '+%F %T') host=$host gpu=$gpu name=$name" >> "$WATCH/watch_${host}.log"
            nohup bash "$SCRIPT" run "$host" "$gpu" "$name" "$level" "$motion" "$scale" \
              > "$WATCH/run_${name}_${host}_${gpu}.log" 2>&1 &
            echo $! > "$WATCH/run_${name}_${host}_${gpu}.pid"
            unset "zero_since[$gpu]"
          fi
        fi
      else
        unset "zero_since[$gpu]"
      fi
    done
    sleep "$poll"
  done
}

case "${1:-}" in
  watch)
    watch_host "$2" "${3:-36000}"
    ;;
  run)
    run_one "$2" "$3" "$4" "$5" "$6" "$7"
    ;;
  *)
    echo "usage: $0 watch <200|202> [max_seconds] | run <host> <gpu> <name> <level> <motion_ratio> <scale>" >&2
    exit 2
    ;;
esac
