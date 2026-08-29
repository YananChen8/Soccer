#!/usr/bin/env bash
set -euo pipefail

# Run SAM3 periodic100 on the valid10 split, then run full Step 3 and metrics.
# The new global team-side mapping is explicitly disabled by default.
#
# Typical remote use:
#   cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
#   bash tools/run_sam3_periodic100_valid10_step3_eval.sh
#
# Useful overrides:
#   OVERWRITE=1 bash tools/run_sam3_periodic100_valid10_step3_eval.sh
#   RAW_STATE=/path/to/existing/valid10_p100.pklz RUN_BUILD=0 bash tools/run_sam3_periodic100_valid10_step3_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 STEP3_CUDA_VISIBLE_DEVICES=3 bash tools/run_sam3_periodic100_valid10_step3_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS}"
SAM3_ROOT="${SAM3_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sam3_official}"
CHECKPOINT="${CHECKPOINT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sam3/sam3.pt}"
OUT_ROOT="${OUT_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/sam3_periodic100_valid10}"

SPLIT="${SPLIT:-valid}"
VALID10_VIDEOS="${VALID10_VIDEOS:-SNGS-021 SNGS-023 SNGS-034 SNGS-040 SNGS-041 SNGS-051 SNGS-052 SNGS-085 SNGS-091 SNGS-093}"
VERSION="${VERSION:-sam3}"
PROMPTS="${PROMPTS:-player}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#VISIBLE_GPU_ARRAY[@]} -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; please set at least one GPU." >&2
  exit 2
fi

if [[ -z "${SAM3_GPUS:-}" ]]; then
  SAM3_GPUS=""
  for idx in "${!VISIBLE_GPU_ARRAY[@]}"; do
    if [[ -n "$SAM3_GPUS" ]]; then
      SAM3_GPUS+=","
    fi
    SAM3_GPUS+="$idx"
  done
fi

LAST_GPU_INDEX=$((${#VISIBLE_GPU_ARRAY[@]} - 1))
STEP3_CUDA_VISIBLE_DEVICES="${STEP3_CUDA_VISIBLE_DEVICES:-${VISIBLE_GPU_ARRAY[$LAST_GPU_INDEX]}}"

RECONDITION_EVERY="${RECONDITION_EVERY:-100}"
COLLECTIVE_TIMEOUT_SEC="${COLLECTIVE_TIMEOUT_SEC:-1800}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-48}"
OFFLOAD_VIDEO_TO_CPU="${OFFLOAD_VIDEO_TO_CPU:-1}"
OFFLOAD_STATE_TO_CPU="${OFFLOAD_STATE_TO_CPU:-1}"
NFRAMES="${NFRAMES:--1}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_RAW_ATOMIC="${RUN_RAW_ATOMIC:-1}"
RUN_STEP3="${RUN_STEP3:-1}"
RUN_STEP3_ATOMIC="${RUN_STEP3_ATOMIC:-1}"
RUN_OFFICIAL_EVAL="${RUN_OFFICIAL_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"

STEP3_CONFIG="${STEP3_CONFIG:-gsr_step_3_sam3_concept_valid_021}"
STEP3_BATCH_JERSEY_ROLE="${STEP3_BATCH_JERSEY_ROLE:-8}"
STEP3_BATCH_LEGIBILITY="${STEP3_BATCH_LEGIBILITY:-16}"
STEP3_BATCH_REID="${STEP3_BATCH_REID:-32}"
STEP3_FRAME_STRIDE="${STEP3_FRAME_STRIDE:-1}"
STEP3_SKIP_ROLE_INFERENCE="${STEP3_SKIP_ROLE_INFERENCE:-False}"
STEP3_DOWNSAMPLE_FACTOR="${STEP3_DOWNSAMPLE_FACTOR:-4}"
STEP3_SAVE_VIDEOS="${STEP3_SAVE_VIDEOS:-False}"
GLOBAL_TEAM_MAPPING="${GLOBAL_TEAM_MAPPING:-0}"

RAW_STATE="${RAW_STATE:-$OUT_ROOT/raw/states/sn-gamestate.pklz}"
STEP3_DIR="${STEP3_DIR:-$OUT_ROOT/step3}"
STEP3_STATE="$STEP3_DIR/states/sn-gamestate.pklz"
RAW_ATOMIC_JSON="$OUT_ROOT/raw/atomic_metrics.json"
STEP3_ATOMIC_JSON="$STEP3_DIR/atomic_metrics.json"
OFFICIAL_EVAL_DIR="$OUT_ROOT/official_eval"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
LOG_DIR="$OUT_ROOT/logs"

IFS=' ' read -r -a VIDEO_ARRAY <<< "$VALID10_VIDEOS"
IFS='|' read -r -a PROMPT_ARRAY <<< "$PROMPTS"

hydra_video_list() {
  local out="["
  local video
  for video in "$@"; do
    out+="\"$video\","
  done
  out="${out%,}]"
  echo "$out"
}

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo
  echo ">>> $*"
  echo ">>> log: $log_file"
  "$@" 2>&1 | tee "$log_file"
}

bool_from_flag() {
  local value="$1"
  if [[ "$value" == "1" || "$value" == "true" || "$value" == "True" || "$value" == "yes" ]]; then
    echo "True"
  else
    echo "False"
  fi
}

print_config() {
  echo "=== SAM3 periodic${RECONDITION_EVERY} valid10 Step3 eval ==="
  echo "repo:          $REPO_DIR"
  echo "dataset:       $DATASET_ROOT"
  echo "videos:        ${VIDEO_ARRAY[*]}"
  echo "sam3 GPUs:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ; --gpus $SAM3_GPUS"
  echo "step3 GPU:     CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES"
  echo "raw state:     $RAW_STATE"
  echo "step3 state:   $STEP3_STATE"
  echo "out root:      $OUT_ROOT"
  echo "global map:    $(bool_from_flag "$GLOBAL_TEAM_MAPPING")"
  echo "overwrite:     $OVERWRITE"
  echo
}

build_raw_state() {
  if [[ "$RUN_BUILD" != "1" ]]; then
    echo "skip SAM3 build because RUN_BUILD=$RUN_BUILD"
    if [[ ! -f "$RAW_STATE" ]]; then
      echo "RAW_STATE does not exist: $RAW_STATE" >&2
      exit 3
    fi
    return
  fi
  if [[ -f "$RAW_STATE" && "$OVERWRITE" != "1" ]]; then
    echo "skip SAM3 build; exists: $RAW_STATE"
    return
  fi

  local cmd=(
    "$PYTHON_BIN"
    tools/build_sam3_gsr_state.py
    --dataset-root "$DATASET_ROOT"
    --split "$SPLIT"
    --videos "${VIDEO_ARRAY[@]}"
    --sam3-root "$SAM3_ROOT"
    --version "$VERSION"
    --checkpoint "$CHECKPOINT"
    --mode periodic
    --gpus "$SAM3_GPUS"
    --collective-timeout-sec "$COLLECTIVE_TIMEOUT_SEC"
    --recondition-every "$RECONDITION_EVERY"
    --max-num-objects "$MAX_NUM_OBJECTS"
    --prompts "${PROMPT_ARRAY[@]}"
    --out "$RAW_STATE"
  )
  if [[ "$OFFLOAD_VIDEO_TO_CPU" == "1" ]]; then
    cmd+=(--offload-video-to-cpu)
  fi
  if [[ "$OFFLOAD_STATE_TO_CPU" == "1" ]]; then
    cmd+=(--offload-state-to-cpu)
  fi
  if [[ "$NFRAMES" != "-1" && "$NFRAMES" != "0" ]]; then
    cmd+=(--nframes "$NFRAMES")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  run_with_log "$LOG_DIR/01_build_periodic100_raw.log" \
    env \
      "SAM3_COLLECTIVE_OP_TIMEOUT_SEC=$COLLECTIVE_TIMEOUT_SEC" \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      "${cmd[@]}"
}

eval_raw_atomic() {
  if [[ "$RUN_RAW_ATOMIC" != "1" ]]; then
    echo "skip raw atomic eval because RUN_RAW_ATOMIC=$RUN_RAW_ATOMIC"
    return
  fi
  if [[ ! -f "$RAW_STATE" ]]; then
    echo "missing raw state: $RAW_STATE" >&2
    exit 3
  fi

  run_with_log "$LOG_DIR/02_eval_raw_atomic.log" \
    "$PYTHON_BIN" tools/eval_atomic_gsr.py \
      --dataset-root "$DATASET_ROOT" \
      --split "$SPLIT" \
      --videos "${VIDEO_ARRAY[@]}" \
      --state-pklz "$RAW_STATE" \
      --out "$RAW_ATOMIC_JSON"
}

run_step3() {
  if [[ "$RUN_STEP3" != "1" ]]; then
    echo "skip Step 3 because RUN_STEP3=$RUN_STEP3"
    if [[ ! -f "$STEP3_STATE" ]]; then
      echo "STEP3_STATE does not exist: $STEP3_STATE" >&2
      exit 3
    fi
    return
  fi
  if [[ ! -f "$RAW_STATE" ]]; then
    echo "missing raw state for Step 3: $RAW_STATE" >&2
    exit 3
  fi
  if [[ -f "$STEP3_STATE" && "$OVERWRITE" != "1" ]]; then
    echo "skip Step 3; exists: $STEP3_STATE"
    return
  fi

  local vids_override
  vids_override="$(hydra_video_list "${VIDEO_ARRAY[@]}")"
  local global_mapping
  global_mapping="$(bool_from_flag "$GLOBAL_TEAM_MAPPING")"

  local cmd=(
    "$PYTHON_BIN"
    -m tracklab.main
    -cn "$STEP3_CONFIG"
    "experiment_subname=sam3_periodic${RECONDITION_EVERY}_valid10_step3"
    "hydra.run.dir=$STEP3_DIR"
    "state.load_file=$RAW_STATE"
    "dataset.dataset_path=$DATASET_ROOT"
    "dataset.eval_set=$SPLIT"
    "dataset.vids_dict.${SPLIT}=$vids_override"
    "eval_tracking=False"
    "visualization.cfg.save_videos=$STEP3_SAVE_VIDEOS"
    "modules.reid.batch_size=$STEP3_BATCH_REID"
    "modules.legibility.batch_size=$STEP3_BATCH_LEGIBILITY"
    "modules.jersey_and_role.batch_size=$STEP3_BATCH_JERSEY_ROLE"
    "modules.jersey_and_role.cfg.frame_stride=$STEP3_FRAME_STRIDE"
    "modules.jersey_and_role.cfg.skip_role_inference=$STEP3_SKIP_ROLE_INFERENCE"
    "modules.jersey_and_role.cfg.use_legibility_filter=True"
    "modules.jersey_and_role.cfg.downsample_factor=$STEP3_DOWNSAMPLE_FACTOR"
    "modules.team_side.cfg.use_global_pitch_mapping=$global_mapping"
  )

  run_with_log "$LOG_DIR/03_step3.log" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
      "CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES" \
      "${cmd[@]}"
}

eval_step3_atomic() {
  if [[ "$RUN_STEP3_ATOMIC" != "1" ]]; then
    echo "skip Step 3 atomic eval because RUN_STEP3_ATOMIC=$RUN_STEP3_ATOMIC"
    return
  fi
  if [[ ! -f "$STEP3_STATE" ]]; then
    echo "missing Step 3 state: $STEP3_STATE" >&2
    exit 3
  fi

  run_with_log "$LOG_DIR/04_eval_step3_atomic.log" \
    "$PYTHON_BIN" tools/eval_atomic_gsr.py \
      --dataset-root "$DATASET_ROOT" \
      --split "$SPLIT" \
      --videos "${VIDEO_ARRAY[@]}" \
      --state-pklz "$STEP3_STATE" \
      --out "$STEP3_ATOMIC_JSON"
}

eval_step3_official() {
  if [[ "$RUN_OFFICIAL_EVAL" != "1" ]]; then
    echo "skip official TrackEval because RUN_OFFICIAL_EVAL=$RUN_OFFICIAL_EVAL"
    return
  fi
  if [[ ! -f "$STEP3_STATE" ]]; then
    echo "missing Step 3 state for official eval: $STEP3_STATE" >&2
    exit 3
  fi

  local vids_override
  vids_override="$(hydra_video_list "${VIDEO_ARRAY[@]}")"

  run_with_log "$LOG_DIR/05_eval_step3_official.log" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
      "CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES" \
      "$PYTHON_BIN" -m tracklab.main \
        -cn gsr_eval_v10 \
        "experiment_subname=sam3_periodic${RECONDITION_EVERY}_valid10_official_eval" \
        "hydra.run.dir=$OFFICIAL_EVAL_DIR" \
        "state.load_file=$STEP3_STATE" \
        "dataset.dataset_path=$DATASET_ROOT" \
        "dataset.eval_set=$SPLIT" \
        "dataset.vids_dict.${SPLIT}=$vids_override"
}

write_summary() {
  "$PYTHON_BIN" - "$RAW_STATE" "$STEP3_STATE" "$RAW_ATOMIC_JSON" "$STEP3_ATOMIC_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

raw_state, step3_state, raw_json, step3_json, summary_csv = map(Path, sys.argv[1:])

def load(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get(data, *keys):
    cur = data
    for key in keys:
        if cur is None or not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def pct(value):
    if value is None:
        return ""
    return f"{float(value) * 100:.3f}"

raw = load(raw_json)
step3 = load(step3_json)
attrs = get(step3, "attributes", "summary") or {}

row = {
    "variant": "periodic100_valid10",
    "raw_state": str(raw_state) if raw_state.exists() else "",
    "raw_Image_HOTA": pct(get(raw, "image_hota", "summary", "HOTA")),
    "raw_Image_DetA": pct(get(raw, "image_hota", "summary", "DetA")),
    "raw_Image_AssA": pct(get(raw, "image_hota", "summary", "AssA")),
    "raw_Image_LocA": pct(get(raw, "image_hota", "summary", "LocA")),
    "raw_Pitch_LocA": pct(get(raw, "pitch_hota", "summary", "LocA")),
    "raw_pred_detections": get(raw, "diagnostics", "pred_detections") or "",
    "step3_state": str(step3_state) if step3_state.exists() else "",
    "step3_Image_HOTA": pct(get(step3, "image_hota", "summary", "HOTA")),
    "step3_Image_DetA": pct(get(step3, "image_hota", "summary", "DetA")),
    "step3_Image_AssA": pct(get(step3, "image_hota", "summary", "AssA")),
    "step3_Image_LocA": pct(get(step3, "image_hota", "summary", "LocA")),
    "step3_Pitch_LocA": pct(get(step3, "pitch_hota", "summary", "LocA")),
    "RoleMacroF1": pct(attrs.get("RoleMacroF1")),
    "TeamTrackAccuracy": pct(attrs.get("TeamTrackAccuracy")),
    "JerseyTrackExactAccuracy": pct(attrs.get("JerseyTrackExactAccuracy")),
    "matched_tracks": attrs.get("matched_tracks", ""),
    "step3_pred_detections": get(step3, "diagnostics", "pred_detections") or "",
}

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

print(f"Wrote {summary_csv}")
PY
}

mkdir -p "$OUT_ROOT" "$LOG_DIR"
print_config
build_raw_state
eval_raw_atomic
run_step3
eval_step3_atomic
eval_step3_official
write_summary

echo
echo "Done."
echo "Raw metrics:   $RAW_ATOMIC_JSON"
echo "Step3 metrics: $STEP3_ATOMIC_JSON"
echo "Summary:       $SUMMARY_CSV"
