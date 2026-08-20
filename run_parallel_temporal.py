#!/usr/bin/env python3
"""Parallel scheduler: run multiple temporal adapter smoke tests on different GPUs.

Each experiment runs:
  tracklab.main -cn gsr_step_3_sam3_4p_test
  modules/pitch=temporal_nbjw_calib
  KP_ADAPTER_CKPT + ADAPTER_RESIDUAL_SCALE via env vars

All 9 experiments can run in parallel on separate GPUs.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

SNGSR = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
TRACKLAB = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/tracklab"
CKPT_BASE = SNGSR / "outputs/gsr/temporal_hrnet/quick_subset12"
OUT_BASE = SNGSR / "outputs/gsr/temporal_hrnet/quick_subset12_smoke_test_parallel"
DET_PKLZ = "/remote-home/jiayuanrao/yishan/SoccerMaster/experiments/detection_benchmark/runs/eval_sam3_ft12ep_4p_test/states/sam3_4p_test_roles.pklz"
VID_LIST = "['SNGS-116','SNGS-117','SNGS-118']"

# Available GPUs (in-use ones filtered out)
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 7]

EXPERIMENTS = [
    # (name, ckpt_path, residual_scale)
    ("baseline_rs0", "", 0),
    ("3dcnn_k15_rs0.5", "3dcnn_k15/kp_adapter_3dcnn_k15.pt", 0.5),
    ("3dcnn_k15_rs1.0", "3dcnn_k15/kp_adapter_3dcnn_k15.pt", 1.0),
    ("tcn_k50_rs0.5", "tcn_k50/kp_adapter_tcn_k50.pt", 0.5),
    ("tcn_k50_rs1.0", "tcn_k50/kp_adapter_tcn_k50.pt", 1.0),
    ("stgcn_k50_rs0.5", "stgcn_k50/kp_adapter_stgcn_k50.pt", 0.5),
    ("stgcn_k50_rs1.0", "stgcn_k50/kp_adapter_stgcn_k50.pt", 1.0),
    ("transformer_k50_rs0.5", "transformer_k50/kp_adapter_transformer_k50.pt", 0.5),
    ("transformer_k50_rs1.0", "transformer_k50/kp_adapter_transformer_k50.pt", 1.0),
]


def run_one(gpu, name, ckpt_rel, scale):
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "states").mkdir(exist_ok=True)

    ckpt = str(CKPT_BASE / ckpt_rel) if ckpt_rel else ""

    # Clean reid cache
    reid_cache = SNGSR / "reid" / "0"
    if reid_cache.exists():
        subprocess.run(["rm", "-rf", str(reid_cache)])

    env = os.environ.copy()
    env["KP_ADAPTER_CKPT"] = ckpt
    env["ADAPTER_RESIDUAL_SCALE"] = str(scale)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SLURM_JOBID"] = str(int(time.time()))
    env["PYTHONPATH"] = f"{TRACKLAB}:{SNGSR}:"

    cmd = [
        sys.executable, "-m", "tracklab.main",
        "-cn", "gsr_step_3_sam3_4p_test",
        f"experiment_subname=temporal_hrnet/quick_subset12_smoke_test_parallel/{name}",
        f"dataset.vids_dict.test={VID_LIST}",
        "modules/pitch=temporal_nbjw_calib",
        f"state.load_file={DET_PKLZ}",
        f"state.save_file={out_dir}/states/sn-gamestate.pklz",
        "visualization.cfg.save_videos=False",
        "eval_tracking=True",
        "test_tracking=True",
    ]

    log = out_dir / "main.log"
    print(f"[{time.strftime('%H:%M:%S')}] GPU={gpu} START {name}", flush=True)
    with open(log, "w") as f:
        f.write(f"[{time.strftime('%F %T')}] START {name} gpu={gpu} ckpt={ckpt} scale={scale}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(SNGSR), env=env, stdout=f, stderr=subprocess.STDOUT
        )
        proc.wait()
    ec = proc.returncode
    print(f"[{time.strftime('%H:%M:%S')}] GPU={gpu} DONE {name} exit={ec}", flush=True)
    return ec


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gpu_iter = iter(AVAILABLE_GPUS)
    futures = {}
    with ThreadPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as pool:
        for exp in EXPERIMENTS:
            gpu = next(gpu_iter)
            name, ckpt, scale = exp
            f = pool.submit(run_one, gpu, name, ckpt, scale)
            futures[f] = name

        for f in as_completed(futures):
            name = futures[f]
            ec = f.result()
            print(f"DONE {name} exit={ec}")
    print("ALL DONE")
