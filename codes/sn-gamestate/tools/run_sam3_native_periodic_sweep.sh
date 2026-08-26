#!/usr/bin/env bash
set -euo pipefail

# Run the SNGS-021 SAM3 native / periodic state builders.
# Execute from codes/sn-gamestate on the remote machine with the SAM3 env active.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-datasets/SoccerNetGS}"
SAM3_ROOT="${SAM3_ROOT:-../sam3_official}"
OUT_ROOT="${OUT_ROOT:-outputs/gsr/sam3_native_periodic_021}"
VERSION="${VERSION:-sam3}"
PROMPTS="${PROMPTS:-player}"
PROMPT_TEAM_COLORS="${PROMPT_TEAM_COLORS:-}"
VIDEO="${VIDEO:-SNGS-021}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-48}"
CHUNK_FRAMES="${CHUNK_FRAMES:-0}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-10}"
STITCH_IOU="${STITCH_IOU:-0.5}"

IFS='|' read -r -a PROMPT_ARGS <<< "$PROMPTS"

COMMON_ARGS=(
  tools/build_sam3_gsr_state.py
  --dataset-root "$DATASET_ROOT"
  --split valid
  --videos "$VIDEO"
  --sam3-root "$SAM3_ROOT"
  --version "$VERSION"
  --prompts "${PROMPT_ARGS[@]}"
  --max-num-objects "$MAX_NUM_OBJECTS"
  --overwrite
)

if [[ "${OFFLOAD_VIDEO_TO_CPU:-0}" == "1" ]]; then
  COMMON_ARGS+=(--offload-video-to-cpu)
fi

if [[ "${OFFLOAD_STATE_TO_CPU:-0}" == "1" ]]; then
  COMMON_ARGS+=(--offload-state-to-cpu)
fi

if [[ "$CHUNK_FRAMES" != "0" ]]; then
  COMMON_ARGS+=(
    --chunk-frames "$CHUNK_FRAMES"
    --chunk-overlap "$CHUNK_OVERLAP"
    --stitch-chunks
    --stitch-iou "$STITCH_IOU"
  )
fi

if [[ -n "${PROMPT_TEAM_COLORS}" ]]; then
  IFS='|' read -r -a COLOR_ARGS <<< "$PROMPT_TEAM_COLORS"
  COMMON_ARGS+=(--prompt-team-colors "${COLOR_ARGS[@]}")
fi

if [[ -n "${METADATA_STATE:-}" ]]; then
  COMMON_ARGS+=(--metadata-state "$METADATA_STATE")
fi

if [[ -n "${VIDEO_ID_MAP:-}" ]]; then
  COMMON_ARGS+=(--video-id-map "$VIDEO_ID_MAP")
fi

"$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --mode native \
  --recondition-every 0 \
  --out "$OUT_ROOT/native/states/sn-gamestate.pklz"

for N in 50 60 70 75; do
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --mode periodic \
    --recondition-every "$N" \
    --out "$OUT_ROOT/periodic${N}/states/sn-gamestate.pklz"
done
