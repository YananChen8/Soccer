set -e
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
export PYTHONPATH=plugins/calibration:.:experiments/detection_benchmark
export CUDA_VISIBLE_DEVICES=7
ROOT=outputs/tta_calib/true_tta_nbjw_20260702/weighted_pseudo_sweep_test8_s20_20260704
mkdir -p "$ROOT"

run_one() {
  name="$1"
  shift
  echo "=== $name ==="
  "$PY" -u tmp_true_tta_nbjw_20260702.py \
    --videos 116 117 118 119 120 121 122 123 \
    --stride 20 \
    --methods pseudo_label_weighted \
    --steps 5 \
    --lr 3e-5 \
    --anchor-weight 0.05 \
    --peak-weight 0.0 \
    --pseudo-conf-threshold 0.05 \
    --pseudo-ransac-px 12 \
    --pseudo-sigma 2.0 \
    --outlier-weight 0.001 \
    --out-dir "$ROOT/$name" \
    --device cuda \
    "$@"
}

run_one weighted_tau6_conf05 --pseudo-residual-tau 6 --pseudo-conf-gamma 0.5
run_one weighted_tau10_conf05 --pseudo-residual-tau 10 --pseudo-conf-gamma 0.5
run_one weighted_tau10_conf10 --pseudo-residual-tau 10 --pseudo-conf-gamma 1.0
