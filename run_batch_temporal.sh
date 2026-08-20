#!/bin/bash
# Batch launcher for 9 temporal adapter smoke test experiments.
# Calls run_one_temporal.sh sequentially.
set -euo pipefail

LAUNCHER=/remote-home/jiayuanrao/yishan/run_one_temporal.sh
CKPT_BASE=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/quick_subset12
PROGRESS=/remote-home/jiayuanrao/yishan/temporal_smoke_progress_$(date +%Y%m%d_%H%M%S).log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

log "========== STARTING 9 EXPERIMENTS =========="

# 1) Baseline
log ">>> 1/9: baseline_rs0"
bash "$LAUNCHER" "" 3 0 "baseline_rs0"
log "<<< 1/9 done"

# 2-3) 3DCNN K15
CKPT="$CKPT_BASE/3dcnn_k15/kp_adapter_3dcnn_k15.pt"
log ">>> 2/9: 3dcnn_k15_rs0.5"
bash "$LAUNCHER" "$CKPT" 15 0.5 "3dcnn_k15_rs0.5"
log "<<< 2/9 done"

log ">>> 3/9: 3dcnn_k15_rs1.0"
bash "$LAUNCHER" "$CKPT" 15 1.0 "3dcnn_k15_rs1.0"
log "<<< 3/9 done"

# 4-5) TCN K50
CKPT="$CKPT_BASE/tcn_k50/kp_adapter_tcn_k50.pt"
log ">>> 4/9: tcn_k50_rs0.5"
bash "$LAUNCHER" "$CKPT" 50 0.5 "tcn_k50_rs0.5"
log "<<< 4/9 done"

log ">>> 5/9: tcn_k50_rs1.0"
bash "$LAUNCHER" "$CKPT" 50 1.0 "tcn_k50_rs1.0"
log "<<< 5/9 done"

# 6-7) STGCN K50
CKPT="$CKPT_BASE/stgcn_k50/kp_adapter_stgcn_k50.pt"
log ">>> 6/9: stgcn_k50_rs0.5"
bash "$LAUNCHER" "$CKPT" 50 0.5 "stgcn_k50_rs0.5"
log "<<< 6/9 done"

log ">>> 7/9: stgcn_k50_rs1.0"
bash "$LAUNCHER" "$CKPT" 50 1.0 "stgcn_k50_rs1.0"
log "<<< 7/9 done"

# 8-9) Transformer K50
CKPT="$CKPT_BASE/transformer_k50/kp_adapter_transformer_k50.pt"
log ">>> 8/9: transformer_k50_rs0.5"
bash "$LAUNCHER" "$CKPT" 50 0.5 "transformer_k50_rs0.5"
log "<<< 8/9 done"

log ">>> 9/9: transformer_k50_rs1.0"
bash "$LAUNCHER" "$CKPT" 50 1.0 "transformer_k50_rs1.0"
log "<<< 9/9 done"

log "========== ALL 9 DONE =========="
