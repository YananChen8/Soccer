#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
REPORT=$ROOT/report_eval_visual_20260701
LOGDIR=$REPORT/_logs/stride5_scatter_reproj25_retry
RUN=fullft_offaux_last_nomotion_k3
OUT=$REPORT/test_stride5_scatter_reproj25_$RUN
mkdir -p "$LOGDIR" "$OUT"
rm -f "$LOGDIR/$RUN.log"
nohup env CUDA_VISIBLE_DEVICES=1 "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
  --mode eval \
  --split test \
  --videos SNGS-116 SNGS-117 SNGS-118 SNGS-119 SNGS-120 SNGS-121 SNGS-122 SNGS-123 \
  --runs "$RUN" \
  --stride 5 \
  --scatter-max-reproj 25 \
  --ckpt-root "$ROOT" \
  --out-dir "$OUT" \
  --device cuda \
  > "$LOGDIR/$RUN.log" 2>&1 && \
nohup env CUDA_VISIBLE_DEVICES=1 "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
  --mode scatter \
  --split test \
  --runs "$RUN" \
  --stride 5 \
  --scatter-max-reproj 25 \
  --ckpt-root "$ROOT" \
  --out-dir "$OUT" \
  --device cuda \
  >> "$LOGDIR/$RUN.log" 2>&1 &
echo "$!" > "$LOGDIR/$RUN.pid"
echo "launched retry run=$RUN pid=$(cat "$LOGDIR/$RUN.pid") out=$OUT"
