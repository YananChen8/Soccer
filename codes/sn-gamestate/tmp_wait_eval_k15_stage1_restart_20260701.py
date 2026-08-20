import os
import subprocess
import time
from pathlib import Path

REPO = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
ROOT = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k15_stage1_restart_20260701")
RUN = "fullft_cached_k15_stage1_motion_lastpair_fast_e5_restart_stepckpt"
REPORT = ROOT / "report_eval_test_stride20"
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


def main():
    os.chdir(REPO)
    pid_file = ROOT / RUN / "train.pid"
    log("wait_eval_start")
    while pid_file.exists():
        pid = pid_file.read_text().strip()
        if not pid or not alive(pid):
            break
        log(f"waiting {RUN}:{pid}")
        time.sleep(300)
    if not (ROOT / RUN / "latest.pt").exists():
        log("eval_skip no latest.pt")
        return
    cmd = [
        PY, "-u", "tmp_official_aux_report_eval_visual_20260701.py",
        "--mode", "eval",
        "--split", "test",
        "--videos", "116", "117", "118", "119", "120", "121", "122", "123",
        "--stride", "20",
        "--runs", "baseline", RUN,
        "--ckpt-root", str(ROOT),
        "--out-dir", str(REPORT),
        "--device", "cuda",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "plugins/calibration:."
    log("eval_start")
    with open(REPORT / "eval.log", "a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    log(f"eval_done rc={proc.returncode}")


if __name__ == "__main__":
    main()
