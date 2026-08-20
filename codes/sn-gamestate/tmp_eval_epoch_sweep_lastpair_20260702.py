import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
PY = "/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python"

JOBS = [
    {
        "root": Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701"),
        "out": Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701/report_eval_epoch_sweep_test_stride20"),
        "runs": [
            "fullft_cached_k5_last_motion_lastpair_fast_e5",
            "fullft_cached_k5_stage1_motion_lastpair_fast_e5",
        ],
    },
    {
        "root": Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k15_stage1_restart_20260701"),
        "out": Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k15_stage1_restart_20260701/report_eval_epoch_sweep_test_stride20"),
        "runs": [
            "fullft_cached_k15_stage1_motion_lastpair_fast_e5_restart_stepckpt",
        ],
    },
]


def log(msg):
    for job in JOBS:
        job["out"].mkdir(parents=True, exist_ok=True)
    with open(REPO / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/eval_epoch_sweep_lastpair_20260702.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()


def pending_tags(job):
    out_dir = job["out"]
    todo = []
    for run_name in job["runs"]:
        for epoch in range(1, 6):
            tag = f"{run_name}_epoch{epoch}"
            if (out_dir / tag / "DONE.json").exists():
                continue
            ckpt = job["root"] / run_name / f"epoch{epoch}.pt"
            if ckpt.exists():
                todo.append((tag, run_name, epoch, f"{run_name}/epoch{epoch}.pt"))
    return todo


def eval_batch(job, batch):
    if not batch:
        return False
    root = job["root"]
    out_dir = job["out"]
    batch_id = "batch_" + time.strftime("%Y%m%d_%H%M%S")
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    wrapper = batch_dir / "tmp_eval_wrapper.py"
    wrapper.write_text(
        "import tmp_official_aux_report_eval_visual_20260701 as m\n"
        f"m.RUNS={[('baseline', None)] + [(tag, rel) for tag, _run, _epoch, rel in batch]!r}\n"
        "m.main()\n",
        encoding="utf-8",
    )
    run_args = ["baseline"] + [tag for tag, _run, _epoch, _rel in batch]
    cmd = [
        PY, "-u", str(wrapper),
        "--mode", "eval",
        "--split", "test",
        "--videos", "116", "117", "118", "119", "120", "121", "122", "123",
        "--stride", "20",
        "--runs", *run_args,
        "--ckpt-root", str(root),
        "--out-dir", str(batch_dir),
        "--device", "cuda",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "plugins/calibration:."
    log("eval_batch_start " + ",".join(tag for tag, _run, _epoch, _rel in batch))
    with open(batch_dir / "eval.log", "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    for tag, run_name, epoch, rel in batch:
        tag_dir = out_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        (tag_dir / "DONE.json").write_text(
            json.dumps({"run": run_name, "epoch": epoch, "returncode": rc, "batch": batch_id, "rel": rel}, indent=2),
            encoding="utf-8",
        )
    log(f"eval_batch_done n={len(batch)} rc={rc} batch={batch_id}")
    return True


def all_epochs_done():
    for job in JOBS:
        for run in job["runs"]:
            for epoch in range(1, 6):
                if not (job["out"] / f"{run}_epoch{epoch}" / "DONE.json").exists():
                    return False
    return True


def main():
    os.chdir(REPO)
    log("epoch_sweep_start")
    while not all_epochs_done():
        did_work = False
        for job in JOBS:
            did_work = eval_batch(job, pending_tags(job)) or did_work
        if not did_work:
            log("waiting_for_more_epoch_checkpoints")
            time.sleep(600)
    log("epoch_sweep_all_done")


if __name__ == "__main__":
    main()
