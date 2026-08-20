#!/usr/bin/env python3
"""Master scheduler: 9 experiments on given GPUs."""
import subprocess, sys, time, os
GPUS = sys.argv[1:]
LAUNCHER = "/remote-home/jiayuanrao/yishan/run_one_experiment.py"
EXPS = [
    ("baseline_rs0", "", "0"),
    ("3dcnn_k15_rs0.5", "3dcnn_k15/kp_adapter_3dcnn_k15.pt", "0.5"),
    ("3dcnn_k15_rs1.0", "3dcnn_k15/kp_adapter_3dcnn_k15.pt", "1.0"),
    ("tcn_k50_rs0.5", "tcn_k50/kp_adapter_tcn_k50.pt", "0.5"),
    ("tcn_k50_rs1.0", "tcn_k50/kp_adapter_tcn_k50.pt", "1.0"),
    ("stgcn_k50_rs0.5", "stgcn_k50/kp_adapter_stgcn_k50.pt", "0.5"),
    ("stgcn_k50_rs1.0", "stgcn_k50/kp_adapter_stgcn_k50.pt", "1.0"),
    ("transformer_k50_rs0.5", "transformer_k50/kp_adapter_transformer_k50.pt", "0.5"),
    ("transformer_k50_rs1.0", "transformer_k50/kp_adapter_transformer_k50.pt", "1.0"),
]
procs = {}
for i, (name, ckpt, scale) in enumerate(EXPS):
    gpu = GPUS[i % len(GPUS)]
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {name} gpu={gpu}", flush=True)
    p = subprocess.Popen([sys.executable, LAUNCHER, name, ckpt, scale, gpu])
    procs[name] = (p, gpu)
for name, (p, gpu) in procs.items():
    ec = p.wait()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {name} gpu={gpu} exit={ec}", flush=True)
print("ALL DONE", flush=True)

