#!/usr/bin/env bash
set -euo pipefail

SNG=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0 1 2 3 4 5 7)
fi
for gpu in "${GPUS[@]}"; do
  if [ "$gpu" = "6" ]; then
    echo "Refusing to use banned GPU 202:6" >&2
    exit 2
  fi
done

cd "$SNG"
RUN_ID="fast_full_test_$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="outputs/gsr/temporal_hrnet/${RUN_ID}"
PARAM_DIR="$OUT_DIR/params"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$PARAM_DIR" "$LOG_DIR"
printf '%s\n' "$SNG/$OUT_DIR" > outputs/gsr/temporal_hrnet/fast_full_test_latest.txt
printf 'running\n' > "$OUT_DIR/status.txt"

export PYTHONPATH=plugins/calibration:.:experiments/detection_benchmark
export PYTHONUNBUFFERED=1

mapfile -t VIDEOS < <(find datasets/SoccerNetGS/test -maxdepth 1 -type d -name 'SNGS-*' -printf '%f\n' | sed 's/^SNGS-//' | sort)
N=${#GPUS[@]}
echo "OUT_DIR=$SNG/$OUT_DIR"
echo "GPUS=${GPUS[*]}"
echo "VIDEOS=${VIDEOS[*]}"
echo "START_PARAMS $(date '+%F %T')"

pids=()
for slot in "${!GPUS[@]}"; do
  gpu="${GPUS[$slot]}"
  chunk=()
  for i in "${!VIDEOS[@]}"; do
    if [ $((i % N)) -eq "$slot" ]; then
      chunk+=("${VIDEOS[$i]}")
    fi
  done
  if [ "${#chunk[@]}" -eq 0 ]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" experiments/detection_benchmark/fast_full_test.py \
      --mode params \
      --videos "${chunk[@]}" \
      --out "$PARAM_DIR/g${gpu}.json" \
      --stride 20
  ) > "$LOG_DIR/params_g${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "launched gpu=$gpu pid=${pids[-1]} videos=${chunk[*]}"
done

fail=0
while [ "${#pids[@]}" -gt 0 ]; do
  remaining=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      remaining+=("$pid")
    else
      if ! wait "$pid"; then
        fail=1
      fi
    fi
  done
  pids=("${remaining[@]}")
  echo "PARAMS_RUNNING remaining=${#pids[@]} $(date '+%F %T')"
  if [ "${#pids[@]}" -gt 0 ]; then
    sleep 60
  fi
done
if [ "$fail" -ne 0 ]; then
  printf 'params_failed\n' > "$OUT_DIR/status.txt"
  echo "PARAMS_FAILED; see $SNG/$LOG_DIR"
  exit 1
fi

echo "START_SCORE $(date '+%F %T')"
"$PY" experiments/detection_benchmark/fast_full_test.py \
  --mode score \
  --params-glob "$PARAM_DIR/g*.json" \
  --out "$OUT_DIR/result.json" \
  --nproc 48 \
  > "$LOG_DIR/score.log" 2>&1
cat "$LOG_DIR/score.log"
printf 'done\n' > "$OUT_DIR/status.txt"
echo "DONE $(date '+%F %T')"
echo "RESULT=$SNG/$OUT_DIR/result.json"
