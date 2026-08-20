import json
import pathlib
import subprocess

print("tmux:")
print(subprocess.run(["tmux", "ls"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout)
print("processes:")
out = subprocess.run(["ps", "-eo", "pid,ppid,stat,etime,pcpu,cmd"], stdout=subprocess.PIPE, text=True).stdout
for line in out.splitlines():
    if "eval_temporal_feature_fusion" in line or "reeval_lora_fixed" in line:
        print(line)

hub = pathlib.Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub")
for run in [
    "feature_fusion_last_lora_k10_train12",
    "feature_fusion_mass5_k10_lora_train12",
    "feature_fusion_peak_sharp_k10_lora_train12",
]:
    p = hub / run / "eval_test116_123_stride20_fixedbaseline/results.json"
    print(run, "exists", p.exists(), "size", p.stat().st_size if p.exists() else None)
    if p.exists():
        try:
            d = json.load(open(p))
            print(" videos", list(d.get("videos", {}).keys()), "aggregate", "aggregate" in d)
            if "aggregate" in d:
                print(" aggregate", json.dumps(d["aggregate"], ensure_ascii=False))
        except Exception as exc:
            print(" json_error", repr(exc))
