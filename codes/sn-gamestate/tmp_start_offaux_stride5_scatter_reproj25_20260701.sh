#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate

PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
REPORT=$ROOT/report_eval_visual_20260701
LOGDIR=$REPORT/_logs/stride5_scatter_reproj25
mkdir -p "$LOGDIR"

runs=(
  fullft_offaux_last_motion_k3
  fullft_offaux_last_nomotion_k3
  fullft_offaux_stage1_motion_k3
  fullft_offaux_stage1_nomotion_k3
)
gpus=(0 1 2 3)

for i in "${!runs[@]}"; do
  run="${runs[$i]}"
  gpu="${gpus[$i]}"
  out="$REPORT/test_stride5_scatter_reproj25_${run}"
  mkdir -p "$out"
  runner="$LOGDIR/run_${run}.sh"
  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
    echo "START $(date -Is) run=$run gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
      --mode eval \
      --split test \
      --videos SNGS-116 SNGS-117 SNGS-118 SNGS-119 SNGS-120 SNGS-121 SNGS-122 SNGS-123 \
      --runs "$run" \
      --stride 5 \
      --scatter-max-reproj 25 \
      --ckpt-root "$ROOT" \
      --out-dir "$out" \
      --device cuda
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
      --mode scatter \
      --split test \
      --runs "$run" \
      --stride 5 \
      --scatter-max-reproj 25 \
      --ckpt-root "$ROOT" \
      --out-dir "$out" \
      --device cuda
    echo "DONE $(date -Is) run=$run"
EOF
  chmod +x "$runner"
  nohup bash "$runner" > "$LOGDIR/${run}.log" 2>&1 &
  echo "$!" > "$LOGDIR/${run}.pid"
  echo "launched run=$run gpu=$gpu pid=$(cat "$LOGDIR/${run}.pid") out=$out"
done
