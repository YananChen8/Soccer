#!/usr/bin/env bash
set -euo pipefail

# Run SAM3 periodic100 on valid10 one video at a time, then run Step 3,
# atomic metrics, and official GS-HOTA for each finished video.
#
# This script is intentionally per-video:
#   - a failed/OOM video writes a failed row and, by default, continues;
#   - summary.csv is updated immediately after each video;
#   - the new global team-side pitch mapping stays disabled by default.
#
# Typical remote use:
#   cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
#   bash tools/run_sam3_periodic100_valid10_per_video_step3_eval.sh
#
# Useful overrides:
#   OVERWRITE=1 bash tools/run_sam3_periodic100_valid10_per_video_step3_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 STEP3_CUDA_VISIBLE_DEVICES=3 bash tools/run_sam3_periodic100_valid10_per_video_step3_eval.sh
#   VALID10_VIDEOS="SNGS-021 SNGS-023" bash tools/run_sam3_periodic100_valid10_per_video_step3_eval.sh
#   CONTINUE_ON_ERROR=0 bash tools/run_sam3_periodic100_valid10_per_video_step3_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS}"
SAM3_ROOT="${SAM3_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sam3_official}"
CHECKPOINT="${CHECKPOINT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sam3/sam3.pt}"
OUT_ROOT="${OUT_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/sam3_periodic100_valid10_per_video}"

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
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

STEP3_CONFIG="${STEP3_CONFIG:-gsr_step_3_sam3_concept_valid_021}"
OFFICIAL_EVAL_CONFIG="${OFFICIAL_EVAL_CONFIG:-gsr_eval_sam3_concept_valid_021}"
STEP3_BATCH_JERSEY_ROLE="${STEP3_BATCH_JERSEY_ROLE:-8}"
STEP3_BATCH_LEGIBILITY="${STEP3_BATCH_LEGIBILITY:-16}"
STEP3_BATCH_REID="${STEP3_BATCH_REID:-32}"
STEP3_FRAME_STRIDE="${STEP3_FRAME_STRIDE:-1}"
STEP3_SKIP_ROLE_INFERENCE="${STEP3_SKIP_ROLE_INFERENCE:-False}"
STEP3_DOWNSAMPLE_FACTOR="${STEP3_DOWNSAMPLE_FACTOR:-4}"
STEP3_SAVE_VIDEOS="${STEP3_SAVE_VIDEOS:-False}"
GLOBAL_TEAM_MAPPING="${GLOBAL_TEAM_MAPPING:-0}"

SUMMARY_CSV="$OUT_ROOT/summary.csv"

IFS=' ' read -r -a VIDEO_ARRAY <<< "$VALID10_VIDEOS"
IFS='|' read -r -a PROMPT_ARRAY <<< "$PROMPTS"

bool_from_flag() {
  local value="$1"
  if [[ "$value" == "1" || "$value" == "true" || "$value" == "True" || "$value" == "yes" ]]; then
    echo "True"
  else
    echo "False"
  fi
}

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  set +e
  {
    echo
    echo ">>> $*"
    echo ">>> log: $log_file"
    "$@"
  } 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

hydra_video_list() {
  local out="["
  local video
  for video in "$@"; do
    out+="\"$video\","
  done
  out="${out%,}]"
  echo "$out"
}

set_video_paths() {
  VIDEO="$1"
  VIDEO_DIR="$OUT_ROOT/$VIDEO"
  RAW_DIR="$VIDEO_DIR/raw"
  STEP3_DIR="$VIDEO_DIR/step3"
  OFFICIAL_EVAL_DIR="$VIDEO_DIR/official_eval"
  LOG_DIR="$VIDEO_DIR/logs"

  RAW_STATE="$RAW_DIR/states/sn-gamestate.pklz"
  STEP3_STATE="$STEP3_DIR/states/sn-gamestate.pklz"
  RAW_ATOMIC_JSON="$RAW_DIR/atomic_metrics.json"
  STEP3_ATOMIC_JSON="$STEP3_DIR/atomic_metrics.json"
  BUILD_LOG="$LOG_DIR/01_build_periodic100_raw.log"
  RAW_ATOMIC_LOG="$LOG_DIR/02_eval_raw_atomic.log"
  STEP3_LOG="$LOG_DIR/03_step3.log"
  STEP3_ATOMIC_LOG="$LOG_DIR/04_eval_step3_atomic.log"
  OFFICIAL_EVAL_LOG="$LOG_DIR/05_eval_step3_gs_hota.log"
}

print_config() {
  echo "=== SAM3 periodic${RECONDITION_EVERY} valid10 per-video Step3 + metrics ==="
  echo "repo:          $REPO_DIR"
  echo "dataset:       $DATASET_ROOT"
  echo "videos:        ${VIDEO_ARRAY[*]}"
  echo "sam3 GPUs:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ; --gpus $SAM3_GPUS"
  echo "step3 GPU:     CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES"
  echo "out root:      $OUT_ROOT"
  echo "summary:       $SUMMARY_CSV"
  echo "global map:    $(bool_from_flag "$GLOBAL_TEAM_MAPPING")"
  echo "continue err:  $CONTINUE_ON_ERROR"
  echo "overwrite:     $OVERWRITE"
  echo
}

build_raw_state() {
  if [[ "$RUN_BUILD" != "1" ]]; then
    echo "skip SAM3 build for $VIDEO because RUN_BUILD=$RUN_BUILD"
    if [[ ! -f "$RAW_STATE" ]]; then
      echo "missing raw state: $RAW_STATE" >&2
      return 3
    fi
    return 0
  fi
  if [[ -f "$RAW_STATE" && "$OVERWRITE" != "1" ]]; then
    echo "skip SAM3 build for $VIDEO; exists: $RAW_STATE"
    return 0
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

  run_with_log "$BUILD_LOG" \
    env \
      "SAM3_COLLECTIVE_OP_TIMEOUT_SEC=$COLLECTIVE_TIMEOUT_SEC" \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      "${cmd[@]}"
}

eval_raw_atomic() {
  if [[ "$RUN_RAW_ATOMIC" != "1" ]]; then
    echo "skip raw atomic eval for $VIDEO because RUN_RAW_ATOMIC=$RUN_RAW_ATOMIC"
    return 0
  fi
  if [[ ! -f "$RAW_STATE" ]]; then
    echo "missing raw state for raw atomic eval: $RAW_STATE" >&2
    return 3
  fi

  run_with_log "$RAW_ATOMIC_LOG" \
    "$PYTHON_BIN" tools/eval_atomic_gsr.py \
      --dataset-root "$DATASET_ROOT" \
      --split "$SPLIT" \
      --videos "$VIDEO" \
      --state-pklz "$RAW_STATE" \
      --out "$RAW_ATOMIC_JSON"
}

run_step3() {
  if [[ "$RUN_STEP3" != "1" ]]; then
    echo "skip Step 3 for $VIDEO because RUN_STEP3=$RUN_STEP3"
    if [[ ! -f "$STEP3_STATE" ]]; then
      echo "missing Step 3 state: $STEP3_STATE" >&2
      return 3
    fi
    return 0
  fi
  if [[ ! -f "$RAW_STATE" ]]; then
    echo "missing raw state for Step 3: $RAW_STATE" >&2
    return 3
  fi
  if [[ -f "$STEP3_STATE" && "$OVERWRITE" != "1" ]]; then
    echo "skip Step 3 for $VIDEO; exists: $STEP3_STATE"
    return 0
  fi

  local vids_override
  vids_override="$(hydra_video_list "$VIDEO")"
  local global_mapping
  global_mapping="$(bool_from_flag "$GLOBAL_TEAM_MAPPING")"

  local cmd=(
    "$PYTHON_BIN"
    -m tracklab.main
    -cn "$STEP3_CONFIG"
    "experiment_subname=sam3_periodic${RECONDITION_EVERY}_${VIDEO}_step3"
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

  run_with_log "$STEP3_LOG" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
      "CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES" \
      "${cmd[@]}"
}

eval_step3_atomic() {
  if [[ "$RUN_STEP3_ATOMIC" != "1" ]]; then
    echo "skip Step 3 atomic eval for $VIDEO because RUN_STEP3_ATOMIC=$RUN_STEP3_ATOMIC"
    return 0
  fi
  if [[ ! -f "$STEP3_STATE" ]]; then
    echo "missing Step 3 state for atomic eval: $STEP3_STATE" >&2
    return 3
  fi

  run_with_log "$STEP3_ATOMIC_LOG" \
    "$PYTHON_BIN" tools/eval_atomic_gsr.py \
      --dataset-root "$DATASET_ROOT" \
      --split "$SPLIT" \
      --videos "$VIDEO" \
      --state-pklz "$STEP3_STATE" \
      --out "$STEP3_ATOMIC_JSON"
}

eval_step3_gs_hota() {
  if [[ "$RUN_OFFICIAL_EVAL" != "1" ]]; then
    echo "skip official GS-HOTA for $VIDEO because RUN_OFFICIAL_EVAL=$RUN_OFFICIAL_EVAL"
    return 0
  fi
  if [[ ! -f "$STEP3_STATE" ]]; then
    echo "missing Step 3 state for official GS-HOTA: $STEP3_STATE" >&2
    return 3
  fi

  local vids_override
  vids_override="$(hydra_video_list "$VIDEO")"

  run_with_log "$OFFICIAL_EVAL_LOG" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
      "CUDA_VISIBLE_DEVICES=$STEP3_CUDA_VISIBLE_DEVICES" \
      "$PYTHON_BIN" -m tracklab.main \
        -cn "$OFFICIAL_EVAL_CONFIG" \
        "experiment_subname=sam3_periodic${RECONDITION_EVERY}_${VIDEO}_gs_hota" \
        "hydra.run.dir=$OFFICIAL_EVAL_DIR" \
        "state.load_file=$STEP3_STATE" \
        "dataset.dataset_path=$DATASET_ROOT" \
        "dataset.eval_set=$SPLIT" \
        "dataset.vids_dict.${SPLIT}=$vids_override"
}

write_summary_row() {
  local status="$1"
  "$PYTHON_BIN" - \
    "$SUMMARY_CSV" \
    "$VIDEO" \
    "periodic${RECONDITION_EVERY}" \
    "$status" \
    "$RAW_STATE" \
    "$STEP3_STATE" \
    "$RAW_ATOMIC_JSON" \
    "$STEP3_ATOMIC_JSON" \
    "$OFFICIAL_EVAL_DIR" \
    "$OFFICIAL_EVAL_LOG" \
    "$LOG_DIR" \
    "${VIDEO_ARRAY[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
video = sys.argv[2]
variant = sys.argv[3]
status = sys.argv[4]
raw_state = Path(sys.argv[5])
step3_state = Path(sys.argv[6])
raw_json = Path(sys.argv[7])
step3_json = Path(sys.argv[8])
official_dir = Path(sys.argv[9])
official_log = Path(sys.argv[10])
logs_dir = Path(sys.argv[11])
video_order = sys.argv[12:]

columns = [
    "video",
    "variant",
    "status",
    "raw_state",
    "raw_Image_HOTA",
    "raw_Image_DetA",
    "raw_Image_AssA",
    "raw_Image_LocA",
    "raw_Pitch_LocA",
    "raw_pred_detections",
    "step3_state",
    "step3_GS_HOTA",
    "step3_GS_DetA",
    "step3_GS_AssA",
    "step3_GS_LocA",
    "step3_GS_IDF1",
    "step3_official_summary",
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
    "logs_dir",
]

metric_fields = {
    "HOTA": "step3_GS_HOTA",
    "DetA": "step3_GS_DetA",
    "AssA": "step3_GS_AssA",
    "LocA": "step3_GS_LocA",
    "IDF1": "step3_GS_IDF1",
}

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
    if value is None or value == "":
        return ""
    return f"{float(value) * 100:.3f}"

def percentish(value):
    if value is None or value == "":
        return ""
    try:
        val = float(str(value).strip().rstrip("%"))
    except ValueError:
        return ""
    if abs(val) <= 1.0:
        val *= 100.0
    return f"{val:.3f}"

def count(value):
    if value is None:
        return ""
    return str(value)

def parse_key_value_summary(path):
    out = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in metric_fields:
            continue
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if match:
            out[key] = match.group(0)
    return out

def parse_detailed_csv(path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("seq") or "").strip().upper() not in {"COMBINED", "COMBINED_SEQ"}:
                    continue
                return {key: row[key] for key in metric_fields if key in row and row[key] not in (None, "")}
    except (OSError, csv.Error):
        return {}
    return {}

def split_tokens(line):
    line = line.replace("|", " ").replace(",", " ")
    return [tok.strip().rstrip(":") for tok in line.split() if tok.strip()]

def parse_table_text(path):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    for idx, line in enumerate(lines):
        tokens = split_tokens(line)
        starts = [i for i, tok in enumerate(tokens) if tok == "HOTA"]
        for start in reversed(starts):
            header = tokens[start:]
            if "DetA" not in header and "AssA" not in header:
                continue
            for next_line in lines[idx + 1 : idx + 8]:
                row_tokens = split_tokens(next_line)
                if not row_tokens:
                    continue
                numeric = [
                    tok for tok in row_tokens
                    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", tok)
                ]
                if len(numeric) < min(3, len(header)):
                    continue
                values = numeric[-len(header):]
                return {
                    key: value
                    for key, value in zip(header, values)
                    if key in metric_fields
                }
    return {}

def summary_priority(path):
    name = path.name.lower()
    if name == "all_summary.txt":
        return 0
    if name == "cls_comb_det_av_summary.txt":
        return 1
    if name == "cls_comb_cls_av_summary.txt":
        return 2
    if name == "pedestrian_summary.txt":
        return 3
    if name.endswith("_summary.txt"):
        return 4
    return 9

def parse_official_metrics():
    candidates = []
    if official_dir.exists():
        candidates.extend(sorted(official_dir.rglob("*_summary.txt"), key=lambda p: (summary_priority(p), str(p))))
        candidates.extend(sorted(official_dir.rglob("*_detailed.csv")))
    if official_log.exists():
        candidates.append(official_log)

    for path in candidates:
        if path.name.endswith("_summary.txt"):
            parsed = parse_key_value_summary(path)
        elif path.name.endswith("_detailed.csv"):
            parsed = parse_detailed_csv(path)
        else:
            parsed = parse_table_text(path)
        if parsed.get("HOTA"):
            return parsed, path
    return {}, None

raw = load_json(raw_json)
step3 = load_json(step3_json)
attrs = get(step3, "attributes", "summary") or {}
official, official_source = parse_official_metrics()

row = {
    "video": video,
    "variant": variant,
    "status": status,
    "raw_state": str(raw_state) if raw_state.exists() else "",
    "raw_Image_HOTA": pct(get(raw, "image_hota", "summary", "HOTA")),
    "raw_Image_DetA": pct(get(raw, "image_hota", "summary", "DetA")),
    "raw_Image_AssA": pct(get(raw, "image_hota", "summary", "AssA")),
    "raw_Image_LocA": pct(get(raw, "image_hota", "summary", "LocA")),
    "raw_Pitch_LocA": pct(get(raw, "pitch_hota", "summary", "LocA")),
    "raw_pred_detections": count(get(raw, "diagnostics", "pred_detections")),
    "step3_state": str(step3_state) if step3_state.exists() else "",
    "step3_GS_HOTA": percentish(official.get("HOTA")),
    "step3_GS_DetA": percentish(official.get("DetA")),
    "step3_GS_AssA": percentish(official.get("AssA")),
    "step3_GS_LocA": percentish(official.get("LocA")),
    "step3_GS_IDF1": percentish(official.get("IDF1")),
    "step3_official_summary": str(official_source) if official_source else "",
    "step3_Image_HOTA": pct(get(step3, "image_hota", "summary", "HOTA")),
    "step3_Image_DetA": pct(get(step3, "image_hota", "summary", "DetA")),
    "step3_Image_AssA": pct(get(step3, "image_hota", "summary", "AssA")),
    "step3_Image_LocA": pct(get(step3, "image_hota", "summary", "LocA")),
    "step3_Pitch_LocA": pct(get(step3, "pitch_hota", "summary", "LocA")),
    "RoleMacroF1": pct(attrs.get("RoleMacroF1")),
    "TeamTrackAccuracy": pct(attrs.get("TeamTrackAccuracy")),
    "JerseyTrackExactAccuracy": pct(attrs.get("JerseyTrackExactAccuracy")),
    "matched_tracks": count(attrs.get("matched_tracks")),
    "step3_pred_detections": count(get(step3, "diagnostics", "pred_detections")),
    "logs_dir": str(logs_dir),
}

if status == "ok" and official_log.exists() and not row["step3_GS_HOTA"]:
    row["status"] = "ok:gs_hota_unparsed"

rows = {}
if summary_csv.exists():
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for old_row in reader:
            old_video = old_row.get("video")
            if old_video:
                rows[old_video] = {col: old_row.get(col, "") for col in columns}
rows[video] = row

order_index = {name: idx for idx, name in enumerate(video_order)}
ordered_videos = sorted(rows, key=lambda name: (order_index.get(name, 10_000), name))

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for name in ordered_videos:
        writer.writerow({col: rows[name].get(col, "") for col in columns})

print(f"\nUpdated {summary_csv}")
print(
    "RESULT "
    f"{video} "
    f"status={row['status']} "
    f"GS_HOTA={row['step3_GS_HOTA'] or 'NA'} "
    f"Image_HOTA={row['step3_Image_HOTA'] or 'NA'} "
    f"DetA={row['step3_Image_DetA'] or 'NA'} "
    f"AssA={row['step3_Image_AssA'] or 'NA'} "
    f"Pitch_LocA={row['step3_Pitch_LocA'] or 'NA'} "
    f"RoleMacroF1={row['RoleMacroF1'] or 'NA'} "
    f"TeamAcc={row['TeamTrackAccuracy'] or 'NA'} "
    f"JerseyAcc={row['JerseyTrackExactAccuracy'] or 'NA'}"
)
PY
}

process_video() {
  local video="$1"
  local rc
  local status="ok"
  set_video_paths "$video"
  mkdir -p "$VIDEO_DIR" "$RAW_DIR" "$STEP3_DIR" "$OFFICIAL_EVAL_DIR" "$LOG_DIR"

  echo
  echo "=============================="
  echo "Video: $VIDEO"
  echo "=============================="

  if build_raw_state; then
    :
  else
    rc=$?
    status="failed:build_raw:$rc"
    write_summary_row "$status"
    return "$rc"
  fi

  if eval_raw_atomic; then
    :
  else
    rc=$?
    status="failed:eval_raw_atomic:$rc"
    write_summary_row "$status"
    return "$rc"
  fi

  if run_step3; then
    :
  else
    rc=$?
    status="failed:step3:$rc"
    write_summary_row "$status"
    return "$rc"
  fi

  if eval_step3_atomic; then
    :
  else
    rc=$?
    status="failed:eval_step3_atomic:$rc"
    write_summary_row "$status"
    return "$rc"
  fi

  if eval_step3_gs_hota; then
    :
  else
    rc=$?
    status="failed:gs_hota:$rc"
    write_summary_row "$status"
    return "$rc"
  fi

  write_summary_row "$status"
}

mkdir -p "$OUT_ROOT"
print_config

FAILED_VIDEOS=()
for video in "${VIDEO_ARRAY[@]}"; do
  if process_video "$video"; then
    echo "finished $video"
  else
    rc=$?
    FAILED_VIDEOS+=("$video")
    echo "video $video failed with rc=$rc" >&2
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "stopping because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR" >&2
      exit "$rc"
    fi
  fi
done

echo
echo "Done. Summary: $SUMMARY_CSV"
if [[ ${#FAILED_VIDEOS[@]} -gt 0 ]]; then
  echo "Failed videos: ${FAILED_VIDEOS[*]}" >&2
  exit 1
fi
