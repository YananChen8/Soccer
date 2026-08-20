from pathlib import Path

root = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701")
for path in sorted(root.glob("fullft_offaux_*_k3/logs/train.log")):
    print(f"=== {path} ===")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[:12]:
        print(line)
    print("--- tail ---")
    for line in lines[-12:]:
        print(line)
