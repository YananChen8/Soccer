#!/usr/bin/env bash
set -euo pipefail

# Compare original YOLO+SAM2 refined/IoU-tracking Step3 and SAM3 periodic100
# with the standalone atomic metrics:
#   Image HOTA/DetA/AssA/LocA, Pitch LocA, RoleMacroF1,
#   TeamTrackAccuracy, JerseyTrackExactAccuracy.
#
# This script does not rerun Step3. It evaluates two final Step3 states with the
# same tools/eval_atomic_gsr.py implementation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS}"
VALID10_VIDEOS="${VALID10_VIDEOS:-SNGS-021 SNGS-023 SNGS-034 SNGS-040 SNGS-041 SNGS-051 SNGS-052 SNGS-085 SNGS-091 SNGS-093}"

SAM3_OUT_ROOT="${SAM3_OUT_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/sam3_periodic100_valid10_per_video}"
SAM3_MERGED_STATE="${SAM3_MERGED_STATE:-$SAM3_OUT_ROOT/step3_merged/states/sn-gamestate.pklz}"
SAM3_ATOMIC_JSON="${SAM3_ATOMIC_JSON:-$SAM3_OUT_ROOT/atomic_valid10_metrics.json}"

BASELINE_STATE="${BASELINE_STATE:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/baseline_valid10/step3_merged/states/sn-gamestate.pklz}"
BASELINE_ATOMIC_JSON="${BASELINE_ATOMIC_JSON:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/baseline_valid10/atomic_valid10_metrics.json}"

SUMMARY_CSV="${SUMMARY_CSV:-$SAM3_OUT_ROOT/atomic_valid10_compare.csv}"
LOG_DIR="${LOG_DIR:-$SAM3_OUT_ROOT/logs}"

OVERWRITE_MERGE="${OVERWRITE_MERGE:-0}"
RUN_SAM3_ATOMIC="${RUN_SAM3_ATOMIC:-1}"
RUN_BASELINE_ATOMIC="${RUN_BASELINE_ATOMIC:-auto}"
MIN_TRACK_MATCHES="${MIN_TRACK_MATCHES:-1}"
INCLUDE_MISSING_GT_JERSEY="${INCLUDE_MISSING_GT_JERSEY:-0}"
FORCE_PITCH="${FORCE_PITCH:-0}"

IFS=' ' read -r -a VIDEO_ARRAY <<< "$VALID10_VIDEOS"

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo
  echo ">>> $*"
  echo ">>> log: $log_file"
  "$@" 2>&1 | tee "$log_file"
}

collect_sam3_step3_states() {
  SAM3_STEP3_STATES=()
  local video
  local state
  for video in "${VIDEO_ARRAY[@]}"; do
    state="$SAM3_OUT_ROOT/$video/step3/states/sn-gamestate.pklz"
    if [[ ! -f "$state" ]]; then
      echo "missing SAM3 Step3 state for $video: $state" >&2
      exit 3
    fi
    SAM3_STEP3_STATES+=("$state")
  done
}

merge_sam3_states() {
  collect_sam3_step3_states
  if [[ -f "$SAM3_MERGED_STATE" && "$OVERWRITE_MERGE" != "1" ]]; then
    echo "skip merge; exists: $SAM3_MERGED_STATE"
    return
  fi
  run_with_log "$LOG_DIR/merge_sam3_p100_step3_states_for_atomic.log" \
    "$PYTHON_BIN" tools/merge_tracker_states.py \
      --inputs "${SAM3_STEP3_STATES[@]}" \
      --output "$SAM3_MERGED_STATE" \
      --overwrite
}

run_atomic_eval() {
  local label="$1"
  local state="$2"
  local out_json="$3"
  local log_file="$4"

  if [[ ! -f "$state" ]]; then
    echo "missing state for $label: $state" >&2
    exit 3
  fi

  local cmd=(
    "$PYTHON_BIN" tools/eval_atomic_gsr.py
    --dataset-root "$DATASET_ROOT"
    --split valid
    --videos "${VIDEO_ARRAY[@]}"
    --state-pklz "$state"
    --min-track-matches "$MIN_TRACK_MATCHES"
    --out "$out_json"
  )
  if [[ "$INCLUDE_MISSING_GT_JERSEY" == "1" ]]; then
    cmd+=(--include-missing-gt-jersey)
  fi
  if [[ "$FORCE_PITCH" == "1" ]]; then
    cmd+=(--force-pitch)
  fi

  run_with_log "$log_file" "${cmd[@]}"
}

write_compare_csv() {
  "$PYTHON_BIN" - "$SUMMARY_CSV" "$BASELINE_STATE" "$BASELINE_ATOMIC_JSON" "$SAM3_MERGED_STATE" "$SAM3_ATOMIC_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
baseline_state = Path(sys.argv[2])
baseline_json = Path(sys.argv[3])
sam3_state = Path(sys.argv[4])
sam3_json = Path(sys.argv[5])

columns = [
    "variant",
    "state",
    "metrics_json",
    "Image_HOTA",
    "Image_DetA",
    "Image_AssA",
    "Image_LocA_IoU",
    "Pitch_HOTA",
    "Pitch_DetA",
    "Pitch_AssA",
    "Pitch_LocA",
    "Pitch_status",
    "RoleMacroF1",
    "TeamTrackAccuracy",
    "JerseyTrackExactAccuracy",
    "matched_tracks",
    "frame_matches",
    "gt_detections",
    "pred_detections",
    "pred_duplicate_rows_fixed",
]

def load(path: Path) -> dict:
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

def text(value):
    if value is None:
        return ""
    return str(value)

def row_for(variant: str, state: Path, metrics_json: Path) -> dict[str, str]:
    data = load(metrics_json)
    attrs = get(data, "attributes", "summary") or {}
    pitch = data.get("pitch_hota")
    pitch_summary = pitch.get("summary") if isinstance(pitch, dict) else {}
    pitch_status = "ok" if pitch_summary else f"skipped: {data.get('pitch_skip_reason', '')}".strip()
    return {
        "variant": variant,
        "state": str(state),
        "metrics_json": str(metrics_json),
        "Image_HOTA": pct(get(data, "image_hota", "summary", "HOTA")),
        "Image_DetA": pct(get(data, "image_hota", "summary", "DetA")),
        "Image_AssA": pct(get(data, "image_hota", "summary", "AssA")),
        "Image_LocA_IoU": pct(get(data, "image_hota", "summary", "LocA")),
        "Pitch_HOTA": pct(pitch_summary.get("HOTA")),
        "Pitch_DetA": pct(pitch_summary.get("DetA")),
        "Pitch_AssA": pct(pitch_summary.get("AssA")),
        "Pitch_LocA": pct(pitch_summary.get("LocA")),
        "Pitch_status": pitch_status,
        "RoleMacroF1": pct(attrs.get("RoleMacroF1")),
        "TeamTrackAccuracy": pct(attrs.get("TeamTrackAccuracy")),
        "JerseyTrackExactAccuracy": pct(attrs.get("JerseyTrackExactAccuracy")),
        "matched_tracks": text(attrs.get("matched_tracks")),
        "frame_matches": text(attrs.get("frame_matches")),
        "gt_detections": text(get(data, "diagnostics", "gt_detections")),
        "pred_detections": text(get(data, "diagnostics", "pred_detections")),
        "pred_duplicate_rows_fixed": text(
            get(data, "diagnostics", "pred_duplicates_before_dedupe", "duplicate_frame_track_rows")
        ),
    }

rows = [
    row_for("yolo_sam2_refined_iou_step3_atomic_valid10", baseline_state, baseline_json),
    row_for("sam3_periodic100_step3_atomic_valid10", sam3_state, sam3_json),
]

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {summary_csv}")
for row in rows:
    print(
        f"{row['variant']}: "
        f"Image_HOTA={row['Image_HOTA'] or 'NA'} "
        f"Image_DetA={row['Image_DetA'] or 'NA'} "
        f"Image_AssA={row['Image_AssA'] or 'NA'} "
        f"Pitch_LocA={row['Pitch_LocA'] or 'NA'} "
        f"RoleMacroF1={row['RoleMacroF1'] or 'NA'} "
        f"TeamAcc={row['TeamTrackAccuracy'] or 'NA'} "
        f"JerseyAcc={row['JerseyTrackExactAccuracy'] or 'NA'}"
    )
PY
}

mkdir -p "$LOG_DIR"

echo "=== Atomic valid10 comparison ==="
echo "baseline state:  $BASELINE_STATE"
echo "baseline json:   $BASELINE_ATOMIC_JSON"
echo "SAM3 root:       $SAM3_OUT_ROOT"
echo "SAM3 merged:     $SAM3_MERGED_STATE"
echo "SAM3 json:       $SAM3_ATOMIC_JSON"
echo "summary CSV:     $SUMMARY_CSV"

merge_sam3_states

if [[ "$RUN_SAM3_ATOMIC" == "1" ]]; then
  run_atomic_eval "sam3_periodic100_valid10" "$SAM3_MERGED_STATE" "$SAM3_ATOMIC_JSON" "$LOG_DIR/eval_sam3_periodic100_valid10_atomic.log"
fi

if [[ "$RUN_BASELINE_ATOMIC" == "1" ]] || { [[ "$RUN_BASELINE_ATOMIC" == "auto" ]] && [[ ! -f "$BASELINE_ATOMIC_JSON" ]]; }; then
  run_atomic_eval "baseline_valid10" "$BASELINE_STATE" "$BASELINE_ATOMIC_JSON" "$LOG_DIR/eval_baseline_valid10_atomic.log"
else
  echo "skip baseline atomic eval; exists: $BASELINE_ATOMIC_JSON"
fi

write_compare_csv
