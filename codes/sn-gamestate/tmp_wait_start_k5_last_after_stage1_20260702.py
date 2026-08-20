import os
import subprocess
import time
from pathlib import Path

REPO = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
PY = "/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python"
ROOT = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701")
STAGE1 = ROOT / "fullft_cached_k5_stage1_motion_lastpair_fast_e5"
OUT = ROOT / "fullft_cached_k5_last_motion_lastpair_fast_e5"
CACHE = "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_memmap_cache_20260701/train"
LOG = ROOT / "wait_start_k5_last_after_stage1.log"


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def read_pid(path):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""


def main():
    os.chdir(REPO)
    log("wait_start")
    stage_pid_file = STAGE1 / "train.pid"
    while True:
        pid = read_pid(stage_pid_file)
        if not pid or not alive(pid):
            break
        log(f"waiting_stage1 pid={pid}")
        time.sleep(300)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "launch_config.txt").write_text(
        "\n".join([
            "host=host200",
            "gpu=4",
            "run=fullft_cached_k5_last_motion_lastpair_fast_e5",
            "fusion_level=last",
            "window_size=5",
            "batch_size=4",
            "epochs=5",
            "workers=4",
            "loss=last_pair_motion_residual_nan_guard",
            "save_every_steps=2000",
            "auto_balance_steps=100",
            "peak_target_ratio=0.3",
            "motion_target_ratio=0.5",
            "residual_scale=0.05",
            f"cache={CACHE}",
            "",
        ]),
        encoding="utf-8",
    )
    cmd = [
        PY, "-u", "tmp_train_temporal_hrnet_cached_fullft_20260701.py",
        "--cache-dir", CACHE,
        "--out-dir", str(OUT),
        "--fusion-level", "last",
        "--window-size", "5",
        "--epochs", "5",
        "--batch-size", "4",
        "--workers", "4",
        "--hrnet-lr", "3e-6",
        "--adapter-lr", "3e-5",
        "--auto-balance-steps", "100",
        "--peak-target-ratio", "0.3",
        "--motion-target-ratio", "0.5",
        "--residual-scale", "0.05",
        "--save-every-steps", "2000",
        "--resume",
        "--log-every", "100",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    env["PYTHONPATH"] = "plugins/calibration:."
    log("starting_k5_last_gpu4")
    with (OUT / "train.log").open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    (OUT / "train.pid").write_text(str(proc.pid), encoding="utf-8")
    log(f"started_k5_last pid={proc.pid}")


if __name__ == "__main__":
    main()
