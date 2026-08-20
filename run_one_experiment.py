#!/usr/bin/env python3
"""Run one temporal adapter smoke test. Args: <name> <ckpt_rel_or_empty> <scale> <gpu>"""
import os, sys, subprocess, time, shutil
from pathlib import Path
name, ckpt_rel, scale, gpu = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SNGSR = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
TRACKLAB = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/tracklab"
CKPT_BASE = SNGSR / "outputs/gsr/temporal_hrnet/quick_subset12"
OUT_BASE = SNGSR / "outputs/gsr/temporal_hrnet/quick_subset12_smoke_test_parallel"
DET_PKLZ = "/remote-home/jiayuanrao/yishan/SoccerMaster/experiments/detection_benchmark/runs/eval_sam3_ft12ep_4p_test/states/sam3_4p_test_roles.pklz"
ckpt = str(CKPT_BASE / ckpt_rel) if ckpt_rel else ""
out_dir = OUT_BASE / name
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "states").mkdir(exist_ok=True)
reid0 = SNGSR / "reid" / "0"
if reid0.exists(): shutil.rmtree(str(reid0), ignore_errors=True)
env = os.environ.copy()
env["KP_ADAPTER_CKPT"] = ckpt
env["ADAPTER_RESIDUAL_SCALE"] = scale
env["CUDA_VISIBLE_DEVICES"] = gpu
env["SLURM_JOBID"] = str(int(time.time()))
env["PYTHONPATH"] = f"{TRACKLAB}:{SNGSR}:"
log = out_dir / "main.log"
ts = time.strftime("%F %T")
print(f"[{ts}] GPU={gpu} START {name}", flush=True)
with open(log, "w") as fh:
    fh.write(f"[{ts}] START {name} gpu={gpu} ckpt={ckpt} scale={scale}\n"); fh.flush()
    proc = subprocess.Popen([sys.executable, "-m", "tracklab.main", "-cn", "gsr_step_3_sam3_4p_test",
        f"experiment_subname=temporal_hrnet/quick_subset12_smoke_test_parallel/{name}",
        "dataset.vids_dict.test=['SNGS-116','SNGS-117','SNGS-118']",
        "modules/pitch=temporal_nbjw_calib", f"state.load_file={DET_PKLZ}",
        f"state.save_file={out_dir}/states/sn-gamestate.pklz",
        "visualization.cfg.save_videos=False", "eval_tracking=True", "test_tracking=True"],
        cwd=str(SNGSR), env=env, stdout=fh, stderr=subprocess.STDOUT)
    ec = proc.wait()
ts2 = time.strftime("%F %T")
print(f"[{ts2}] GPU={gpu} DONE {name} exit={ec}", flush=True)

