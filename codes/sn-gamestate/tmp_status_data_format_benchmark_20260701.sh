#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
OUT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/data_format_benchmark_20260701
echo PROCS
ps -ef | grep tmp_benchmark_temporal_data_formats_20260701.py | grep -v grep || true
echo SIZE
du -sh "$OUT" 2>/dev/null || true
echo LOG
tail -40 "$OUT/benchmark.log" 2>/dev/null || true
echo RESULTS
ls -lh "$OUT" 2>/dev/null || true
