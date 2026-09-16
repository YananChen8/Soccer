#!/usr/bin/env bash
set -euo pipefail

# Compare original YOLO+SAM2 refined/IoU-tracking Step3 against SAM3 periodic100
# with the same official valid10 GS-HOTA evaluator.
#
# This script does not rerun Step3. It:
#   1. merges the per-video SAM3 periodic100 Step3 states;
#   2. runs official TrackEval through gsr_eval_v10 on the merged SAM3 state;
#   3. reuses or runs the original baseline official eval;
#   4. writes one comparable CSV.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS}"
VALID10_VIDEOS="${VALID10_VIDEOS:-SNGS-021 SNGS-023 SNGS-034 SNGS-040 SNGS-041 SNGS-051 SNGS-052 SNGS-085 SNGS-091 SNGS-093}"

SAM3_OUT_ROOT="${SAM3_OUT_ROOT:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/sam3_periodic100_valid10_per_video}"
SAM3_MERGED_STATE="${SAM3_MERGED_STATE:-$SAM3_OUT_ROOT/step3_merged/states/sn-gamestate.pklz}"
SAM3_EVAL_DIR="${SAM3_EVAL_DIR:-$SAM3_OUT_ROOT/official_valid10_eval}"

BASELINE_STATE="${BASELINE_STATE:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/baseline_valid10/step3_merged/states/sn-gamestate.pklz}"
BASELINE_EVAL_DIR="${BASELINE_EVAL_DIR:-/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/baseline_valid10/eval}"

SUMMARY_CSV="${SUMMARY_CSV:-$SAM3_OUT_ROOT/official_valid10_compare.csv}"
LOG_DIR="${LOG_DIR:-$SAM3_OUT_ROOT/logs}"

OVERWRITE_MERGE="${OVERWRITE_MERGE:-0}"
RUN_SAM3_EVAL="${RUN_SAM3_EVAL:-1}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

IFS=' ' read -r -a VIDEO_ARRAY <<< "$VALID10_VIDEOS"

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

summary_exists() {
  local dir="$1"
  [[ -n "$(find "$dir" -name 'cls_comb_det_av_summary.txt' -print -quit 2>/dev/null || true)" ]]
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
  run_with_log "$LOG_DIR/merge_sam3_p100_step3_states.log" \
    "$PYTHON_BIN" tools/merge_tracker_states.py \
      --inputs "${SAM3_STEP3_STATES[@]}" \
      --output "$SAM3_MERGED_STATE" \
      --overwrite
}

run_official_eval() {
  local label="$1"
  local state="$2"
  local out_dir="$3"
  local log_file="$4"
  local vids_override
  vids_override="$(hydra_video_list "${VIDEO_ARRAY[@]}")"

  if [[ ! -f "$state" ]]; then
    echo "missing state for $label: $state" >&2
    exit 3
  fi

  run_with_log "$log_file" \
    env \
      HYDRA_FULL_ERROR=1 \
      PYTHONPATH=plugins/calibration:. \
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      "$PYTHON_BIN" -m tracklab.main \
        -cn gsr_eval_v10 \
        "experiment_subname=${label}/official_valid10_eval" \
        "hydra.run.dir=$out_dir" \
        "state.load_file=$state" \
        "dataset.dataset_path=$DATASET_ROOT" \
        "dataset.eval_set=valid" \
        "dataset.vids_dict.valid=$vids_override"
}

write_compare_csv() {
  "$PYTHON_BIN" - "$SUMMARY_CSV" "$BASELINE_STATE" "$BASELINE_EVAL_DIR" "$SAM3_MERGED_STATE" "$SAM3_EVAL_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
baseline_state = Path(sys.argv[2])
baseline_eval_dir = Path(sys.argv[3])
sam3_state = Path(sys.argv[4])
sam3_eval_dir = Path(sys.argv[5])

fields = [
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA",
    "HOTA(0)", "LocA(0)", "HOTALocA(0)", "MOTA", "MOTP", "MODA", "CLR_Re",
    "CLR_Pr", "IDF1", "IDR", "IDP", "Dets", "GT_Dets", "IDs", "GT_IDs",
]

columns = ["variant", "state", "eval_dir", "summary_file"] + fields

def numeric_token(token: str) -> bool:
    return re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", token.strip()) is not None

def split_tokens(line: str) -> list[str]:
    return [token.strip() for token in line.replace(",", " ").replace("|", " ").split() if token.strip()]

def parse_key_value_summary(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in fields:
            continue
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if match:
            result[key] = match.group(0)
    return result

def parse_table_summary(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        header = split_tokens(line)
        if "HOTA" not in header or "DetA" not in header:
            continue
        for data_line in lines[idx + 1 : idx + 6]:
            tokens = split_tokens(data_line)
            numbers = [token for token in tokens if numeric_token(token)]
            if len(numbers) < 3:
                continue
            values = numbers[-len(header):]
            return {key: value for key, value in zip(header, values) if key in fields}
    return {}

def parse_detailed_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = (row.get("seq") or "").strip().upper()
            if seq not in {"COMBINED", "COMBINED_SEQ"}:
                continue
            return {key: row[key] for key in fields if key in row and row[key] not in (None, "")}
    return {}

def parse_metric_file(path: Path) -> dict[str, str]:
    if path.name.endswith("_detailed.csv"):
        return parse_detailed_csv(path)
    parsed = parse_key_value_summary(path)
    if parsed.get("HOTA"):
        return parsed
    return parse_table_summary(path)

def candidate_metric_files(eval_dir: Path) -> list[Path]:
    if not eval_dir.exists():
        return []
    candidates = []
    priorities = [
        "cls_comb_det_av_summary.txt",
        "all_summary.txt",
        "cls_comb_cls_av_summary.txt",
    ]
    for name in priorities:
        candidates.extend(sorted(eval_dir.rglob(name)))
    candidates.extend(sorted(path for path in eval_dir.rglob("*_summary.txt") if path not in candidates))
    candidates.extend(sorted(eval_dir.rglob("*_detailed.csv")))
    return candidates

def find_metrics(eval_dir: Path) -> tuple[Path, dict[str, str]]:
    candidates = candidate_metric_files(eval_dir)
    if not candidates:
        raise FileNotFoundError(f"No TrackEval summary found under {eval_dir}")
    for path in candidates:
        parsed = parse_metric_file(path)
        if parsed.get("HOTA"):
            return path, parsed
    return candidates[0], {}

def row_for(variant: str, state: Path, eval_dir: Path) -> dict[str, str]:
    summary, parsed = find_metrics(eval_dir)
    row = {
        "variant": variant,
        "state": str(state),
        "eval_dir": str(eval_dir),
        "summary_file": str(summary),
    }
    for field in fields:
        row[field] = parsed.get(field, "")
    return row

rows = [
    row_for("yolo_sam2_refined_iou_step3_official_valid10", baseline_state, baseline_eval_dir),
    row_for("sam3_periodic100_step3_official_valid10", sam3_state, sam3_eval_dir),
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
        f"GS-HOTA={row.get('HOTA', 'NA')} "
        f"DetA={row.get('DetA', 'NA')} "
        f"AssA={row.get('AssA', 'NA')} "
        f"LocA={row.get('LocA', 'NA')} "
        f"IDF1={row.get('IDF1', 'NA')}"
    )
PY
}

mkdir -p "$LOG_DIR"

echo "=== Official valid10 GS-HOTA comparison ==="
echo "baseline state: $BASELINE_STATE"
echo "baseline eval:  $BASELINE_EVAL_DIR"
echo "SAM3 root:      $SAM3_OUT_ROOT"
echo "SAM3 merged:    $SAM3_MERGED_STATE"
echo "SAM3 eval:      $SAM3_EVAL_DIR"
echo "summary CSV:    $SUMMARY_CSV"

merge_sam3_states

if [[ "$RUN_SAM3_EVAL" == "1" ]]; then
  run_official_eval "sam3_periodic100_valid10" "$SAM3_MERGED_STATE" "$SAM3_EVAL_DIR" "$LOG_DIR/eval_sam3_periodic100_valid10_official.log"
fi

if [[ "$RUN_BASELINE_EVAL" == "1" ]] || { [[ "$RUN_BASELINE_EVAL" == "auto" ]] && ! summary_exists "$BASELINE_EVAL_DIR"; }; then
  run_official_eval "baseline_valid10" "$BASELINE_STATE" "$BASELINE_EVAL_DIR" "$LOG_DIR/eval_baseline_valid10_official.log"
else
  echo "skip baseline eval; summary already exists under $BASELINE_EVAL_DIR"
fi

write_compare_csv
