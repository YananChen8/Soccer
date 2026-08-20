set -e
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
export PYTHONPATH=plugins/calibration:.:experiments/detection_benchmark
export CUDA_VISIBLE_DEVICES=0

ROOT=outputs/tta_calib/true_tta_nbjw_20260702/full49_pseudo_best_chunks_s20_20260704
mkdir -p "$ROOT"

"$PY" -u tmp_true_tta_nbjw_20260702.py \
  --videos 140 141 142 143 144 145 146 147 148 149 150 187 188 189 190 191 192 193 194 195 196 197 198 199 200 \
  --stride 20 \
  --methods pseudo_label \
  --steps 5 \
  --lr 3e-5 \
  --anchor-weight 0.05 \
  --peak-weight 0.0 \
  --pseudo-conf-threshold 0.05 \
  --pseudo-ransac-px 12 \
  --pseudo-sigma 2.0 \
  --outlier-weight 0.001 \
  --out-dir "$ROOT/chunk_b_140_150_187_200" \
  --device cuda
