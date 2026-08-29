#!/usr/bin/env bash
set -euo pipefail

# One-command SAM3 sweep for SNGS-021:
#   native, periodic50, periodic60, periodic70
# For each variant this script builds the raw SAM3 TrackLab state, evaluates
# raw atomic metrics, runs Step 3, evaluates Step 3 atomic metrics, then writes
# a CSV summary.
#
# Typical remote use:
#   cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
#   bash tools/run_sam3_native_periodic_step3_eval.sh
#
# Useful overrides:
#   OVERWRITE=1 bash tools/run_sam3_native_periodic_step3_eval.sh
#   NFRAMES=300 bash tools/run_sam3_native_periodic_step3_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 STEP3_CUDA_VISIBLE_DEVICES=3 bash tools/run_sam3_native_periodic_step3_eval.sh
#   NATIVE_STATE=/path/to/existing/sn-gamestate.pklz bash tools/run_sam3_native_periodic_step3_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS}"
SAM3_ROOT="${SAM3_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sam3_official}"
CHECKPOINT="${CHECKPOINT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sam3/sam3.pt}"
OUT_ROOT="${OUT_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/sam3_native_periodic_021_full_sweep}"

SPLIT="${SPLIT:-valid}"
VIDEO="${VIDEO:-SNGS-021}"
VARIANTS="${VARIANTS:-native periodic50 periodic60 periodic70}"
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

COLLECTIVE_TIMEOUT_SEC="${COLLECTIVE_TIMEOUT_SEC:-1800}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-48}"
OFFLOAD_VIDEO_TO_CPU="${OFFLOAD_VIDEO_TO_CPU:-1}"
OFFLOAD_STATE_TO_CPU="${OFFLOAD_STATE_TO_CPU:-1}"
NFRAMES="${NFRAMES:--1}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_RAW_ATOMIC="${RUN_RAW_ATOMIC:-1}"
RUN_STEP3="${RUN_STEP3:-1}"
RUN_STEP3_ATOMIC="${RUN_STEP3_ATOMIC:-1}"
RUN_OFFICIAL_EVAL="${RUN_OFFICIAL_EVAL:-0}"
OVERWRITE="${OVERWRITE:-0}"

STEP3_CONFIG="${STEP3_CONFIG:-gsr_step_3_sam3_concept_valid_021}"
STEP3_BATCH_JERSEY_ROLE="${STEP3_BATCH_JERSEY_ROLE:-8}"
STEP3_BATCH_LEGIBILITY="${STEP3_BATCH_LEGIBILITY:-16}"
STEP3_BATCH_REID="${STEP3_BATCH_REID:-32}"
STEP3_FRAME_STRIDE="${STEP3_FRAME_STRIDE:-1}"
STEP3_SKIP_ROLE_INFERENCE="${STEP3_SKIP_ROLE_INFERENCE:-False}"
STEP3_DOWNSAMPLE_FACTOR="${STEP3_DOWNSAMPLE_FACTOR:-4}"
STEP3_SAVE_VIDEOS="${STEP3_SAVE_VIDEOS:-False}"

mkdir -p "$OUT_ROOT"

echo "=== SAM3 native/periodic Step3 sweep ==="
echo "repo:        $REPO_DIR"
echo "dataset:     $DATASET_ROOT"
echo "video:       $SPLIT/$VIDEO"
echo "variants:    $VARIANTS"
echo "sam3 GPUs:   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ; --gpus $SAM3_GPUS"
echo "step3 GPU:   CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES"
echo "out root:    $OUT_ROOT"
echo "overwrite:   $OVERWRITE"
echo

IFS=' ' read -r -a VARIANT_ARRAY <<< "$VARIANTS"
IFS='|' read -r -a PROMPT_ARRAY <<< "$PROMPTS"

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo
  echo ">>> $*"
  echo ">>> log: $log_file"
  "$@" 2>&1 | tee "$log_file"
}

external_state_for_variant() {
  local variant="$1"
  local env_name
  env_name="$(printf '%s_STATE' "$variant" | tr '[:lower:]' '[:upper:]')"
  echo "${!env_name-}"
}

raw_state_for_variant() {
  local variant="$1"
  local external_state
  external_state="$(external_state_for_variant "$variant")"
  if [[ -n "$external_state" ]]; then
    echo "$external_state"
  else
    echo "$OUT_ROOT/$variant/states/sn-gamestate.pklz"
  fi
}

record_raw_state_path() {
  local variant="$1"
  local variant_dir="$OUT_ROOT/$variant"
  local raw_state
  raw_state="$(raw_state_for_variant "$variant")"
  mkdir -p "$variant_dir"
  printf '%s\n' "$raw_state" > "$variant_dir/raw_state_path.txt"
}

variant_mode() {
  local variant="$1"
  if [[ "$variant" == "native" || "$variant" == "naive" ]]; then
    echo "native"
  elif [[ "$variant" =~ ^periodic[0-9]+$ ]]; then
    echo "periodic"
  else
    echo "Unknown variant '$variant'. Use native periodic50 periodic60 periodic70." >&2
    exit 2
  fi
}

variant_recondition_every() {
  local variant="$1"
  if [[ "$variant" == "native" || "$variant" == "naive" ]]; then
    echo "0"
  else
    echo "${variant#periodic}"
  fi
}

build_raw_state() {
  local variant="$1"
  local mode="$2"
  local recondition_every="$3"
  local variant_dir="$OUT_ROOT/$variant"
  local raw_state
  raw_state="$(raw_state_for_variant "$variant")"
  local log_file="$variant_dir/logs/01_build_raw.log"
  record_raw_state_path "$variant"

  if [[ -n "$(external_state_for_variant "$variant")" ]]; then
    if [[ ! -f "$raw_state" ]]; then
      echo "external state for $variant does not exist: $raw_state" >&2
      exit 3
    fi
    echo "skip build for $variant; using external state: $raw_state"
    return
  fi

  if [[ "$RUN_BUILD" != "1" ]]; then
    echo "skip build for $variant because RUN_BUILD=$RUN_BUILD"
    return
  fi
  if [[ -f "$raw_state" && "$OVERWRITE" != "1" ]]; then
    echo "skip build for $variant; exists: $raw_state"
    return
  fi

  local cmd=(
    "$PYTHON_BIN"
    tools/build_sam3_gsr_state.py
    --dataset-root "$DATASET_ROOT"
    --split "$SPLIT"
    --videos "$VIDEO"
    --sam3-root "$SAM3_ROOT"
    --version "$VERSION"
    --checkpoint "$CHECKPOINT"
    --mode "$mode"
    --gpus "$SAM3_GPUS"
    --collective-timeout-sec "$COLLECTIVE_TIMEOUT_SEC"
    --recondition-every "$recondition_every"
    --max-num-objects "$MAX_NUM_OBJECTS"
    --prompts "${PROMPT_ARRAY[@]}"
    --out "$raw_state"
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

  run_with_log "$log_file" \
    env \
      "SAM3_COLLECTIVE_OP_TIMEOUT_SEC=$COLLECTIVE_TIMEOUT_SEC" \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      "${cmd[@]}"
}

eval_atomic_raw() {
  local variant="$1"
  local variant_dir="$OUT_ROOT/$variant"
  local raw_state
  raw_state="$(raw_state_for_variant "$variant")"
  local metrics="$variant_dir/atomic_raw_metrics.json"
  local log_file="$variant_dir/logs/02_eval_atomic_raw.log"
  record_raw_state_path "$variant"

  if [[ "$RUN_RAW_ATOMIC" != "1" ]]; then
    echo "skip raw atomic eval for $variant because RUN_RAW_ATOMIC=$RUN_RAW_ATOMIC"
    return
  fi
  if [[ ! -f "$raw_state" ]]; then
    echo "missing raw state for $variant: $raw_state" >&2
    exit 3
  fi

  run_with_log "$log_file" \
    "$PYTHON_BIN" tools/eval_atomic_gsr.py \
      --dataset-root "$DATASET_ROOT" \
      --split "$SPLIT" \
      --videos "$VIDEO" \
      --state-pklz "$raw_state" \
      --out "$metrics"
}

run_step3() {
  local variant="$1"
  local variant_dir="$OUT_ROOT/$variant"
  local raw_state
  raw_state="$(raw_state_for_variant "$variant")"
  local step3_dir="$variant_dir/step3"
  local step3_state="$step3_dir/states/sn-gamestate.pklz"
  local log_file="$variant_dir/logs/03_step3.log"
  local vids_override
  vids_override="[\"$VIDEO\"]"
  record_raw_state_path "$variant"

  if [[ "$RUN_STEP3" != "1" ]]; then
    echo "skip Step 3 for $variant because RUN_STEP3=$RUN_STEP3"
    return
  fi
  if [[ ! -f "$raw_state" ]]; then
    echo "missing raw state for Step 3 $variant: $raw_state" >&2
    exit 3
  fi
  if [[ -f "$step3_state" && "$OVERWRITE" != "1" ]]; then
    echo "skip Step 3 for $variant; exists: $step3_state"
    return
  fi

  local cmd=(
    "$PYTHON_BIN"
    -m tracklab.main
    -cn "$STEP3_CONFIG"
    "experiment_subname=${variant}_step3"
    "hydra.run.dir=$step3_dir"
    "state.load_file=$raw_state"
    "dataset.dataset_path=$DATASET_ROOT"
    "dataset.eval_set=$SPLIT"
    "dataset.vids_dict.${SPLIT}=$vids_override"
    "visualization.cfg.save_videos=$STEP3_SAVE_VIDEOS"
    "modules.reid.batch_size=$STEP3_BATCH_REID"
    "modules.legibility.batch_size=$STEP3_BATCH_LEGIBILITY"
    "modules.jersey_and_role.batch_size=$STEP3_BATCH_JERSEY_ROLE"
    "modules.jersey_and_role.cfg.frame_stride=$STEP3_FRAME_STRIDE"
    "modules.jersey_and_role.cfg.skip_role_inference=$STEP3_SKIP_ROLE_INFERENCE"
    "modules.jersey_and_role.cfg.use_legibility_filter=True"
    "modules.jersey_and_role.cfg.downsample_factor=$STEP3_DOWNSAMPLE_FACTOR"
  )

  run_with_log "$log_file" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
      "CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES" \
      "${cmd[@]}"
}

eval_atomic_step3() {
  local variant="$1"
  local variant_dir="$OUT_ROOT/$variant"
  local step3_state="$variant_dir/step3/states/sn-gamestate.pklz"
  local metrics="$variant_dir/step3/atomic_metrics.json"
  local log_file="$variant_dir/logs/04_eval_atomic_step3.log"

  if [[ "$RUN_STEP3_ATOMIC" != "1" ]]; then
    echo "skip Step 3 atomic eval for $variant because RUN_STEP3_ATOMIC=$RUN_STEP3_ATOMIC"
    return
  fi
  if [[ ! -f "$step3_state" ]]; then
    echo "missing Step 3 state for $variant: $step3_state" >&2
    exit 3
  fi

  run_with_log "$log_file" \
    "$PYTHON_BIN" tools/eval_atomic_gsr.py \
      --dataset-root "$DATASET_ROOT" \
      --split "$SPLIT" \
      --videos "$VIDEO" \
      --state-pklz "$step3_state" \
      --out "$metrics"
}

eval_official_step3() {
  local variant="$1"
  local variant_dir="$OUT_ROOT/$variant"
  local step3_state="$variant_dir/step3/states/sn-gamestate.pklz"
  local official_dir="$variant_dir/official_eval"
  local log_file="$variant_dir/logs/05_eval_official_step3.log"
  local vids_override
  vids_override="[\"$VIDEO\"]"

  if [[ "$RUN_OFFICIAL_EVAL" != "1" ]]; then
    return
  fi
  if [[ ! -f "$step3_state" ]]; then
    echo "missing Step 3 state for official eval $variant: $step3_state" >&2
    exit 3
  fi

  run_with_log "$log_file" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
    "$PYTHON_BIN" -m tracklab.main \
      -cn gsr_eval_sam3_concept_valid_021 \
      "experiment_subname=${variant}_official_eval" \
      "hydra.run.dir=$official_dir" \
      "state.load_file=$step3_state" \
      "dataset.dataset_path=$DATASET_ROOT" \
      "dataset.eval_set=$SPLIT" \
      "dataset.vids_dict.${SPLIT}=$vids_override"
}

write_summary() {
  "$PYTHON_BIN" - "$OUT_ROOT" "${VARIANT_ARRAY[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
variants = sys.argv[2:]

columns = [
    "variant",
    "raw_state",
    "raw_Image_HOTA",
    "raw_Image_DetA",
    "raw_Image_AssA",
    "raw_Image_LocA",
    "raw_Pitch_LocA",
    "raw_pred_detections",
    "step3_state",
    "step3_Image_HOTA",
    "step3_Image_DetA",
    "step3_Image_AssA",
    "step3_Image_LocA",
    "step3_Pitch_LocA",
    "RoleMacroF1",
    "TeamTrackAccuracy",
    "JerseyTrackExactAccuracy",
    "matched_tracks",
    "step3_pred_detections",
]

def load_json(path):
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

def count(value):
    if value is None:
        return ""
    return str(value)

def row_for(variant):
    variant_dir = out_root / variant
    raw_state_path_txt = variant_dir / "raw_state_path.txt"
    if raw_state_path_txt.exists():
        raw_state = Path(raw_state_path_txt.read_text(encoding="utf-8").strip())
    else:
        raw_state = variant_dir / "states" / "sn-gamestate.pklz"
    step3_state = variant_dir / "step3" / "states" / "sn-gamestate.pklz"
    raw = load_json(variant_dir / "atomic_raw_metrics.json")
    step3 = load_json(variant_dir / "step3" / "atomic_metrics.json")

    raw_pitch = get(raw, "pitch_hota", "summary", "LocA")
    step3_pitch = get(step3, "pitch_hota", "summary", "LocA")
    attrs = get(step3, "attributes", "summary") or {}

    return {
        "variant": variant,
        "raw_state": str(raw_state) if raw_state.exists() else "",
        "raw_Image_HOTA": pct(get(raw, "image_hota", "summary", "HOTA")),
        "raw_Image_DetA": pct(get(raw, "image_hota", "summary", "DetA")),
        "raw_Image_AssA": pct(get(raw, "image_hota", "summary", "AssA")),
        "raw_Image_LocA": pct(get(raw, "image_hota", "summary", "LocA")),
        "raw_Pitch_LocA": pct(raw_pitch),
        "raw_pred_detections": count(get(raw, "diagnostics", "pred_detections")),
        "step3_state": str(step3_state) if step3_state.exists() else "",
        "step3_Image_HOTA": pct(get(step3, "image_hota", "summary", "HOTA")),
        "step3_Image_DetA": pct(get(step3, "image_hota", "summary", "DetA")),
        "step3_Image_AssA": pct(get(step3, "image_hota", "summary", "AssA")),
        "step3_Image_LocA": pct(get(step3, "image_hota", "summary", "LocA")),
        "step3_Pitch_LocA": pct(step3_pitch),
        "RoleMacroF1": pct(attrs.get("RoleMacroF1")),
        "TeamTrackAccuracy": pct(attrs.get("TeamTrackAccuracy")),
        "JerseyTrackExactAccuracy": pct(attrs.get("JerseyTrackExactAccuracy")),
        "matched_tracks": count(attrs.get("matched_tracks")),
        "step3_pred_detections": count(get(step3, "diagnostics", "pred_detections")),
    }

summary_csv = out_root / "summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for variant in variants:
        writer.writerow(row_for(variant))

print(f"\nWrote {summary_csv}")
PY
}

for variant in "${VARIANT_ARRAY[@]}"; do
  mode="$(variant_mode "$variant")"
  recondition_every="$(variant_recondition_every "$variant")"

  echo
  echo "=============================="
  echo "Variant: $variant"
  echo "mode=$mode recondition_every=$recondition_every"
  echo "=============================="

  build_raw_state "$variant" "$mode" "$recondition_every"
  eval_atomic_raw "$variant"
  run_step3 "$variant"
  eval_atomic_step3 "$variant"
  eval_official_step3 "$variant"
  write_summary
done

echo
echo "All requested variants are done."
echo "Summary: $OUT_ROOT/summary.csv"
