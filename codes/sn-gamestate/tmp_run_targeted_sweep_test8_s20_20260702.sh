set -e
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
export PYTHONPATH=plugins/calibration:.:experiments/detection_benchmark
export CUDA_VISIBLE_DEVICES=7
ROOT=outputs/tta_calib/true_tta_nbjw_20260702/targeted_sweep_test8_s20_20260702
mkdir -p "$ROOT"

run_one() {
  name="$1"
  shift
  echo "=== $name ==="
  "$PY" -u tmp_true_tta_nbjw_20260702.py \
    --videos 116 117 118 119 120 121 122 123 \
    --stride 20 \
    --out-dir "$ROOT/$name" \
    --device cuda \
    "$@"
}

run_one flip_bn_s5_lr5e-5_anchor0_smooth08 \
  --methods flip_consistency \
  --steps 5 \
  --lr 5e-5 \
  --anchor-weight 0.0 \
  --peak-weight 0.0 \
  --smooth-gate \
  --smooth-gate-ratio 0.8

run_one flip_bn_s7_lr1e-4_anchor0_smooth08 \
  --methods flip_consistency \
  --steps 7 \
  --lr 1e-4 \
  --anchor-weight 0.0 \
  --peak-weight 0.0 \
  --smooth-gate \
  --smooth-gate-ratio 0.8

run_one temporal_s5_lr3e-5_tw005_gate12_anchor005_out001 \
  --methods pseudo_label_temporal \
  --steps 5 \
  --lr 3e-5 \
  --anchor-weight 0.05 \
  --peak-weight 0.0 \
  --pseudo-conf-threshold 0.05 \
  --pseudo-ransac-px 12 \
  --pseudo-sigma 2.0 \
  --outlier-weight 0.001 \
  --temporal-weight 0.05 \
  --temporal-gate-px 30

run_one temporal_s5_lr3e-5_tw01_gate12_anchor005_out001 \
  --methods pseudo_label_temporal \
  --steps 5 \
  --lr 3e-5 \
  --anchor-weight 0.05 \
  --peak-weight 0.0 \
  --pseudo-conf-threshold 0.05 \
  --pseudo-ransac-px 12 \
  --pseudo-sigma 2.0 \
  --outlier-weight 0.001 \
  --temporal-weight 0.1 \
  --temporal-gate-px 30

run_one temporal_s5_lr3e-5_tw02_gate12_anchor005_out001_smooth08 \
  --methods pseudo_label_temporal \
  --steps 5 \
  --lr 3e-5 \
  --anchor-weight 0.05 \
  --peak-weight 0.0 \
  --pseudo-conf-threshold 0.05 \
  --pseudo-ransac-px 12 \
  --pseudo-sigma 2.0 \
  --outlier-weight 0.001 \
  --temporal-weight 0.2 \
  --temporal-gate-px 30 \
  --smooth-gate \
  --smooth-gate-ratio 0.8
