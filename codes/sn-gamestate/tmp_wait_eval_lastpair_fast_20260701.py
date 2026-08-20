import os
import subprocess
import time
from pathlib import Path

ROOT = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701")
REPORT = ROOT / "report_eval_test_stride20"
RUNS = [
    "fullft_cached_k15_stage1_motion_lastpair_fast_e5",
    "fullft_cached_k5_last_motion_lastpair_fast_e5",
    "fullft_cached_k5_stage1_motion_lastpair_fast_e5",
]
PY = "/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python"


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def log(msg):
    REPORT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "wait_eval.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()


def main():
    os.chdir("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
    log("wait_eval_start")
    while True:
        running = []
        for run in RUNS:
            pid_file = ROOT / run / "train.pid"
            if pid_file.exists():
                pid = pid_file.read_text().strip()
                if pid and alive(pid):
                    running.append((run, pid))
        if not running:
            break
        log("waiting " + ", ".join(f"{r}:{p}" for r, p in running))
        time.sleep(300)

    eval_runs = [run for run in RUNS if (ROOT / run / "latest.pt").exists()]
    skipped = [run for run in RUNS if run not in eval_runs]
    if skipped:
        log("skip_missing_checkpoint " + ", ".join(skipped))
    if not eval_runs:
        log("eval_skip no latest.pt found")
        return

    REPORT.mkdir(parents=True, exist_ok=True)
    log("eval_start")
    cmd = [
        PY,
        "-u",
        "tmp_official_aux_report_eval_visual_20260701.py",
        "--mode",
        "eval",
        "--split",
        "test",
        "--videos",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "--stride",
        "20",
        "--runs",
        "baseline",
        *eval_runs,
        "--ckpt-root",
        str(ROOT),
        "--out-dir",
        str(REPORT),
        "--device",
        "cuda",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "plugins/calibration:."
    with open(REPORT / "eval.log", "a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    log(f"eval_done rc={proc.returncode}")


if __name__ == "__main__":
    main()
