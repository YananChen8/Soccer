import argparse
import csv
import json
import socket
import subprocess
import sys
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from nbjw_calib.model.cls_hrnet import get_cls_net
from sn_gamestate.temporal_hrnet import TemporalHRNetFeatureFusion


CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration/SV_kp"
H, W = 540, 960
HM_H, HM_W = 270, 480
KPTS = 58


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def start_tensorboard(out_dir, args):
    writer = None
    tb_dir = Path(out_dir) / "tensorboard"
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        class NativeSummaryWriter:
            def __init__(self, log_dir):
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self.writer = EventFileWriter(str(log_dir))

            def add_scalar(self, tag, scalar_value, global_step):
                summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(scalar_value))])
                self.writer.add_event(Event(wall_time=time.time(), step=int(global_step), summary=summary))

            def flush(self):
                self.writer.flush()

            def close(self):
                self.writer.close()

        writer = NativeSummaryWriter(tb_dir)
    except Exception as exc:
        print(f"TENSORBOARD_WRITER_DISABLED reason={exc!r}", flush=True)
        return None

    if not args.tensorboard:
        print(f"TENSORBOARD_EVENTS_ONLY logdir={tb_dir}", flush=True)
        return writer

    port = int(args.tensorboard_port) if int(args.tensorboard_port) > 0 else find_free_port()
    log_path = Path(out_dir) / "tensorboard_server.log"
    pid_path = Path(out_dir) / "tensorboard_server.json"
    tb_code = (
        "import importlib.metadata as md, pkg_resources, sys\n"
        "if not hasattr(pkg_resources, 'iter_entry_points'):\n"
        "    class EntryPointCompat:\n"
        "        def __init__(self, ep):\n"
        "            self._ep = ep\n"
        "            self.name = ep.name\n"
        "        def load(self):\n"
        "            return self._ep.load()\n"
        "        def resolve(self):\n"
        "            return self._ep.load()\n"
        "    def iter_entry_points(group, name=None):\n"
        "        eps = md.entry_points()\n"
        "        eps = eps.select(group=group) if hasattr(eps, 'select') else eps.get(group, [])\n"
        "        for ep in eps:\n"
        "            if name is None or ep.name == name:\n"
        "                yield EntryPointCompat(ep)\n"
        "    pkg_resources.iter_entry_points = iter_entry_points\n"
        "from tensorboard import main\n"
        "main.run_main()\n"
    )
    cmd = [
        sys.executable,
        "-c",
        tb_code,
        "--logdir",
        str(Path(out_dir).resolve()),
        "--host",
        str(args.tensorboard_host),
        "--port",
        str(port),
    ]
    try:
        log = open(log_path, "a", buffering=1)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid_path.write_text(json.dumps({"pid": proc.pid, "host": args.tensorboard_host, "port": port, "logdir": str(Path(out_dir).resolve()), "cmd": cmd}, indent=2))
        print(f"TENSORBOARD_STARTED pid={proc.pid} host={args.tensorboard_host} port={port} logdir={Path(out_dir).resolve()}", flush=True)
    except Exception as exc:
        print(f"TENSORBOARD_SERVER_DISABLED reason={exc!r} logdir={tb_dir}", flush=True)
    return writer


class MemmapTemporalDataset(Dataset):
    def __init__(self, cache_dir, window_size):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "cache_meta.json").read_text())
        self.shape = tuple(meta["image_shape"])
        self.n = int(meta["n_frames"])
        self.window_size = int(window_size)
        self.img_path = self.cache_dir / f"images_u8_chw_{H}x{W}.dat"
        lab = np.load(self.cache_dir / "compact_labels.npz")
        self.coords = lab["coords"]
        self.masks = lab["masks"]
        self.rows = list(csv.DictReader((self.cache_dir / "manifest.csv").open(newline="")))
        self.video_start = {}
        for i, row in enumerate(self.rows):
            self.video_start.setdefault(row["video"], i)
        self.images = None

    def _images(self):
        if self.images is None:
            self.images = np.memmap(self.img_path, mode="r", dtype=np.uint8, shape=self.shape)
        return self.images

    def __len__(self):
        return self.n

    def _window(self, images, idx, first):
        frame_ids = [max(first, idx - off) for off in range(self.window_size - 1, -1, -1)]
        return [torch.from_numpy(np.asarray(images[j]).copy()).float().div_(255.0) for j in frame_ids]

    def __getitem__(self, idx):
        images = self._images()
        first = self.video_start[self.rows[idx]["video"]]
        prev_idx = max(first, idx - 1)
        return {
            "window": torch.stack(self._window(images, idx, first)),
            "prev_window": torch.stack(self._window(images, prev_idx, first)),
            "coords": torch.from_numpy(self.coords[idx].astype(np.int64)),
            "mask": torch.from_numpy(self.masks[idx].astype(np.float32)),
            "prev_coords": torch.from_numpy(self.coords[prev_idx].astype(np.int64)),
            "prev_mask": torch.from_numpy(self.masks[prev_idx].astype(np.float32)),
            "has_prev": torch.tensor(float(idx > first), dtype=torch.float32),
        }


def load_hrnet(device):
    cfg = yaml.safe_load(open(CFG))
    model = get_cls_net(cfg["cfg"])
    model.load_state_dict(torch.load(CKPT, map_location=device))
    return model.to(device)


def coords_to_heatmap(coords, mask, sigma=2.0):
    b = coords.shape[0]
    yy, xx = torch.meshgrid(
        torch.arange(HM_H, device=coords.device),
        torch.arange(HM_W, device=coords.device),
        indexing="ij",
    )
    kp_coords = coords[:, : KPTS - 1]
    kp_mask = mask[:, : KPTS - 1]
    x = kp_coords[:, :, 0].view(b, KPTS - 1, 1, 1).float()
    y = kp_coords[:, :, 1].view(b, KPTS - 1, 1, 1).float()
    kp_valid = (kp_mask > 0.5) & (kp_coords[:, :, 0] >= 0) & (kp_coords[:, :, 1] >= 0)
    dist2 = (xx.view(1, 1, HM_H, HM_W) - x).square() + (yy.view(1, 1, HM_H, HM_W) - y).square()
    kp_heat = torch.exp(-dist2 / (2.0 * sigma * sigma)) * kp_valid[:, :, None, None].float()
    bg_heat = torch.clamp(1.0 - kp_heat.sum(dim=1, keepdim=True), 0.0, 1.0)
    heat = torch.cat([kp_heat, bg_heat], dim=1)
    valid = torch.cat([kp_valid, mask[:, KPTS - 1 : KPTS] > 0.5], dim=1)
    return heat, valid.float()


def masked_mse(pred, gt, mask):
    pred = torch.nan_to_num(pred[:, : gt.shape[1]].float(), nan=0.0, posinf=1.0, neginf=0.0)
    raw = (pred - gt).square()
    keep = mask[:, :, None, None] > 0
    return torch.where(keep, raw, torch.zeros_like(raw)).mean()


def local_softargmax_xy_from_coords(pred, coords, mask, radius_px=5.0, beta=20.0):
    pred = torch.nan_to_num(pred[:, : KPTS - 1].float(), nan=0.0, posinf=1.0, neginf=0.0)
    coords = coords[:, : KPTS - 1]
    mask = mask[:, : KPTS - 1]
    b, c, h, w = pred.shape
    valid = (mask > 0.5) & (coords[:, :, 0] >= 0) & (coords[:, :, 1] >= 0)
    yy, xx = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij")
    gx = coords[:, :, 0].view(b, c, 1, 1).float()
    gy = coords[:, :, 1].view(b, c, 1, 1).float()
    keep = (xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()
    keep = keep <= radius_px * radius_px
    keep = keep & valid[:, :, None, None]
    logits = (pred * beta).flatten(2).masked_fill(~keep.flatten(2), -1e4)
    prob = torch.softmax(logits, dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=pred.device, dtype=pred.dtype),
        torch.arange(w, device=pred.device, dtype=pred.dtype),
        indexing="ij",
    )
    pred_xy = torch.stack([(prob * grid_x.flatten()).sum(-1), (prob * grid_y.flatten()).sum(-1)], dim=-1)
    gt_xy = torch.stack([coords[:, :, 0].float(), coords[:, :, 1].float()], dim=-1)
    return pred_xy, gt_xy, valid


def peak_location_loss(pred, coords, mask):
    pred_xy, gt_xy, valid = local_softargmax_xy_from_coords(pred, coords, mask)
    raw = torch.nn.functional.smooth_l1_loss(pred_xy, gt_xy, reduction="none").sum(-1)
    raw = torch.where(valid & torch.isfinite(raw), raw, torch.zeros_like(raw))
    return raw.sum() / (valid.float().sum() + 1e-6)


def peak_margin_loss(pred, coords, mask, inner_radius=2.0, outer_radius=8.0, margin=0.03, min_peak=0.05, fp_threshold=0.03):
    pred = torch.nan_to_num(pred[:, : KPTS - 1].float(), nan=0.0, posinf=1.0, neginf=0.0)
    coords = coords[:, : KPTS - 1]
    mask = mask[:, : KPTS - 1]
    b, c, h, w = pred.shape
    valid = (mask > 0.5) & (coords[:, :, 0] >= 0) & (coords[:, :, 1] >= 0)
    yy, xx = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij")
    gx = coords[:, :, 0].view(b, c, 1, 1).float()
    gy = coords[:, :, 1].view(b, c, 1, 1).float()
    dist2 = (xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()
    inner = (dist2 <= inner_radius * inner_radius) & valid[:, :, None, None]
    outer = (dist2 > outer_radius * outer_radius) & valid[:, :, None, None]

    neg_large = torch.full_like(pred, -1.0)
    in_max = torch.where(inner, pred, neg_large).flatten(2).amax(-1)
    out_max = torch.where(outer, pred, neg_large).flatten(2).amax(-1)
    margin_loss = torch.relu(out_max + margin - in_max)
    peak_floor = torch.relu(min_peak - in_max)
    valid_f = valid.float()
    valid_loss = ((margin_loss + 0.5 * peak_floor) * valid_f).sum() / (valid_f.sum() + 1e-6)

    invalid = ~valid
    invalid_max = torch.where(invalid[:, :, None, None], pred, neg_large).flatten(2).amax(-1)
    fp_loss = torch.relu(invalid_max - fp_threshold).square()
    fp_loss = (fp_loss * invalid.float()).sum() / (invalid.float().sum() + 1e-6)
    return valid_loss + 0.25 * fp_loss


def temporal_motion_residual_loss(pred_xy, gt_xy, valid, beta=5.0):
    pred_motion = pred_xy[:, 1:] - pred_xy[:, :-1]
    gt_motion = gt_xy[:, 1:] - gt_xy[:, :-1]
    valid_pair = (valid[:, 1:] > 0) & (valid[:, :-1] > 0)
    residual_raw = pred_motion - gt_motion
    gt_norm = torch.norm(gt_motion, dim=-1, keepdim=True)
    residual = residual_raw / (beta + gt_norm)
    per_point = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none").sum(dim=-1)
    finite_pair = valid_pair & torch.isfinite(per_point) & torch.isfinite(residual_raw).all(dim=-1)
    weighted = torch.where(finite_pair, per_point, torch.zeros_like(per_point))
    denom = finite_pair.float().sum()
    if denom < 1:
        zero = weighted.sum() * 0.0
        return zero, {
            "motion_valid_pair_ratio": 0.0,
            "mean_gt_motion_norm": 0.0,
            "mean_pred_motion_norm": 0.0,
            "mean_motion_residual_norm": 0.0,
        }
    stats = {
        "motion_valid_pair_ratio": float(valid_pair.float().mean().detach().item()),
        "mean_gt_motion_norm": float(torch.norm(gt_motion, dim=-1)[finite_pair].mean().detach().item()),
        "mean_pred_motion_norm": float(torch.norm(pred_motion, dim=-1)[finite_pair].mean().detach().item()),
        "mean_motion_residual_norm": float(torch.norm(residual_raw, dim=-1)[finite_pair].mean().detach().item()),
    }
    return weighted.sum() / denom, stats


def compute_losses(model, teacher_model, batch, device, use_amp, args):
    win = batch["window"].to(device, non_blocking=True)
    prev_win = batch["prev_window"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    mask0 = batch["mask"].to(device, non_blocking=True)
    prev_coords = batch["prev_coords"].to(device, non_blocking=True)
    prev_mask = batch["prev_mask"].to(device, non_blocking=True)
    has_prev = batch["has_prev"].to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred = model(win)
        with torch.no_grad():
            teacher = teacher_model(win[:, -1])
        gt_heat, gt_valid = coords_to_heatmap(coords, mask0)
        heat = masked_mse(pred, gt_heat, gt_valid)
        peak_raw = peak_margin_loss(
            pred,
            coords,
            mask0,
            inner_radius=args.peak_inner_radius,
            outer_radius=args.peak_outer_radius,
            margin=args.peak_margin,
            min_peak=args.peak_min_value,
            fp_threshold=args.false_peak_threshold,
        )
        teacher_raw = torch.nn.functional.mse_loss(
            torch.nan_to_num(pred.float(), nan=0.0, posinf=1.0, neginf=0.0),
            torch.nan_to_num(teacher.float(), nan=0.0, posinf=1.0, neginf=0.0),
        )
        pred_xy_t, gt_xy_t, valid_t = local_softargmax_xy_from_coords(pred, coords, mask0)
        with torch.no_grad():
            was_training = model.training
            model.eval()
            prev_pred = model(prev_win)
            model.train(was_training)
            pred_xy_prev, gt_xy_prev, valid_prev = local_softargmax_xy_from_coords(prev_pred, prev_coords, prev_mask)
        pred_xy = torch.stack([pred_xy_prev.detach(), pred_xy_t], dim=1)
        gt_xy = torch.stack([gt_xy_prev, gt_xy_t], dim=1)
        valid = torch.stack([valid_prev, valid_t], dim=1)
        valid[:, 0] = valid[:, 0] & (has_prev[:, None] > 0.5)
        motion_raw, motion_stats = temporal_motion_residual_loss(pred_xy, gt_xy, valid, beta=args.motion_beta)
        peak_weighted = peak_raw * args.peak_weight
        motion_weighted = motion_raw * args.motion_weight
        teacher_weighted = teacher_raw * args.teacher_weight
        loss = heat + peak_weighted + motion_weighted + teacher_weighted
    return loss, heat, peak_raw, motion_raw, teacher_raw, peak_weighted, motion_weighted, teacher_weighted, motion_stats


def finite_median(vals, default):
    vals = [float(v) for v in vals if np.isfinite(float(v)) and float(v) > 0]
    return float(np.median(vals)) if vals else float(default)


def auto_balance_weights(model, teacher_model, loader, device, use_amp, args, out_dir):
    heats, peaks, motions = [], [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            _loss, heat, peak, motion, _teacher, _pw, _mw, _tw, _stats = compute_losses(model, teacher_model, batch, device, use_amp, args)
            heats.append(float(heat.detach().item()))
            peaks.append(float(peak.detach().item()))
            motions.append(float(motion.detach().item()))
            if i >= args.auto_balance_steps:
                break
    model.train()
    mh = finite_median(heats, 1e-4)
    mp = finite_median(peaks, 1e-8)
    mm = finite_median(motions, 1e-8)
    if args.peak_weight <= 0:
        args.peak_weight = min(max(args.peak_target_ratio * mh / max(mp, 1e-12), args.min_aux_weight), args.max_aux_weight)
    if args.motion_weight <= 0:
        args.motion_weight = min(max(args.motion_target_ratio * mh / max(mm, 1e-12), args.min_aux_weight), args.max_aux_weight)
    info = {
        "median_heat": mh,
        "median_peak_raw": mp,
        "median_motion_raw": mm,
        "peak_weight": args.peak_weight,
        "motion_weight": args.motion_weight,
        "peak_target_ratio": args.peak_target_ratio,
        "motion_target_ratio": args.motion_target_ratio,
    }
    (out_dir / "auto_balance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("AUTO_BALANCE", json.dumps(info), flush=True)


def self_test():
    coords = torch.full((1, KPTS, 2), -1, dtype=torch.long)
    mask = torch.zeros(1, KPTS)
    coords[0, 0] = torch.tensor([10, 10])
    mask[0, 0] = 1
    mask[0, KPTS - 1] = 1
    heat, valid = coords_to_heatmap(coords, mask)
    assert heat.shape == (1, KPTS, HM_H, HM_W)
    assert valid[0, 0] == 1 and valid[0, KPTS - 1] == 1
    assert float(heat[0, 0, 10, 10]) > 0.99
    assert float(heat[0, KPTS - 1, 10, 10]) < 0.01
    assert float(heat[0, KPTS - 1, 200, 400]) > 0.99
    pred = torch.zeros(2, 3, 4, 2)
    gt = torch.zeros(2, 3, 4, 2)
    valid = torch.zeros(2, 3, 4)
    loss, stats = temporal_motion_residual_loss(pred, gt, valid)
    assert torch.isfinite(loss) and float(loss) == 0.0
    valid[:, :, :] = 1
    gt[:, 1:, :, 0] = 2.0
    pred[:, 1:, :, 0] = 2.0
    loss, stats = temporal_motion_residual_loss(pred, gt, valid)
    assert torch.isfinite(loss) and float(loss) < 1e-8
    print("SELF_TEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=False)
    ap.add_argument("--out-dir", required=False)
    ap.add_argument("--fusion-level", choices=["last", "stage1"], default="last")
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hrnet-lr", type=float, default=3e-6)
    ap.add_argument("--adapter-lr", type=float, default=3e-5)
    ap.add_argument("--peak-weight", type=float, default=-1.0)
    ap.add_argument("--motion-weight", type=float, default=-1.0)
    ap.add_argument("--teacher-weight", type=float, default=0.5)
    ap.add_argument("--peak-target-ratio", type=float, default=0.25)
    ap.add_argument("--motion-target-ratio", type=float, default=0.1)
    ap.add_argument("--max-aux-weight", type=float, default=100000.0)
    ap.add_argument("--min-aux-weight", type=float, default=1e-6)
    ap.add_argument("--auto-balance-steps", type=int, default=100)
    ap.add_argument("--motion-beta", type=float, default=5.0)
    ap.add_argument("--peak-inner-radius", type=float, default=2.0)
    ap.add_argument("--peak-outer-radius", type=float, default=8.0)
    ap.add_argument("--peak-margin", type=float, default=0.03)
    ap.add_argument("--peak-min-value", type=float, default=0.05)
    ap.add_argument("--false-peak-threshold", type=float, default=0.03)
    ap.add_argument("--residual-scale", type=float, default=-1)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    ap.add_argument("--tensorboard-port", type=int, default=0)
    ap.add_argument("--tensorboard-host", default="0.0.0.0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--device", default="cuda")
    ap.set_defaults(tensorboard=True)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.cache_dir or not args.out_dir:
        raise SystemExit("--cache-dir and --out-dir are required unless --self-test is used")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"
    args.residual_scale = args.residual_scale if args.residual_scale > 0 else (0.02 if args.fusion_level == "stage1" else 0.05)

    ds = MemmapTemporalDataset(args.cache_dir, args.window_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    model = TemporalHRNetFeatureFusion(load_hrnet(device), level=args.fusion_level, window_size=args.window_size, residual_scale=args.residual_scale, freeze_hrnet=False).to(device)
    teacher_model = load_hrnet(device).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.Adam([
        {"params": model.hrnet.parameters(), "lr": args.hrnet_lr},
        {"params": model.fusion.parameters(), "lr": args.adapter_lr},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, global_step = 0, 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck.get("global_step", 0))
        args.peak_weight = float(ck.get("config", {}).get("peak_weight", args.peak_weight))
        args.motion_weight = float(ck.get("config", {}).get("motion_weight", args.motion_weight))
        print(f"RESUME epoch={start_epoch} global_step={global_step}", flush=True)

    use_amp = device.type == "cuda"
    if start_epoch == 0 and args.auto_balance_steps > 0:
        auto_balance_weights(model, teacher_model, dl, device, use_amp, args, out_dir)
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    fields = [
        "epoch", "step", "frames", "total_loss", "heat_loss",
        "peak_raw", "motion_raw", "teacher_raw", "peak_weight", "motion_weight", "teacher_weight",
        "peak_weighted", "motion_weighted", "teacher_weighted", "motion_valid_pair_ratio",
        "mean_gt_motion_norm", "mean_pred_motion_norm", "mean_motion_residual_norm", "fps",
        "total_loss_x1e6", "heat_loss_x1e6", "peak_weighted_x1e6", "motion_weighted_x1e6", "teacher_weighted_x1e6",
    ]
    csv_path = out_dir / "step_losses.csv"
    loss_file = csv_path.open("a" if args.resume and csv_path.exists() else "w", newline="")
    writer = csv.DictWriter(loss_file, fieldnames=fields)
    if loss_file.tell() == 0:
        writer.writeheader()
    tb_writer = start_tensorboard(out_dir, args)

    model.train()
    run_start = time.perf_counter()

    def save_checkpoint(path, epoch_value):
        payload = {
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch_value,
            "global_step": global_step,
            "fusion_level": args.fusion_level,
            "window_size": args.window_size,
            "residual_scale": args.residual_scale,
            "full_finetune": True,
            "config": vars(args),
        }
        torch.save(payload, path)

    for epoch in range(start_epoch, args.epochs):
        totals = defaultdict(float)
        for batch in dl:
            loss, heat, peak_raw, motion_raw, teacher_raw, peak_weighted, motion_weighted, teacher_weighted, stats = compute_losses(model, teacher_model, batch, device, use_amp, args)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(
                    f"NONFINITE_SKIP epoch={epoch} next_step={global_step + 1} "
                    f"loss={float(loss.detach().item())} heat={float(heat.detach().item())} "
                    f"peak={float(peak_raw.detach().item())} motion={float(motion_raw.detach().item())} "
                    f"teacher={float(teacher_raw.detach().item())}",
                    flush=True,
                )
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            frames = int(batch["window"].size(0))
            vals = {
                "epoch": epoch,
                "step": global_step,
                "frames": frames,
                "total_loss": float(loss.detach().item()),
                "heat_loss": float(heat.detach().item()),
                "peak_raw": float(peak_raw.detach().item()),
                "motion_raw": float(motion_raw.detach().item()),
                "teacher_raw": float(teacher_raw.detach().item()),
                "peak_weight": float(args.peak_weight),
                "motion_weight": float(args.motion_weight),
                "teacher_weight": float(args.teacher_weight),
                "peak_weighted": float(peak_weighted.detach().item()),
                "motion_weighted": float(motion_weighted.detach().item()),
                "teacher_weighted": float(teacher_weighted.detach().item()),
                "fps": frames * global_step / max(time.perf_counter() - run_start, 1e-6),
                **stats,
            }
            vals.update({
                "total_loss_x1e6": vals["total_loss"] * 1e6,
                "heat_loss_x1e6": vals["heat_loss"] * 1e6,
                "peak_weighted_x1e6": vals["peak_weighted"] * 1e6,
                "motion_weighted_x1e6": vals["motion_weighted"] * 1e6,
                "teacher_weighted_x1e6": vals["teacher_weighted"] * 1e6,
            })
            writer.writerow(vals)
            if tb_writer is not None:
                for key in [
                    "total_loss", "heat_loss", "peak_raw", "motion_raw", "teacher_raw",
                    "peak_weighted", "motion_weighted", "teacher_weighted",
                    "total_loss_x1e6", "heat_loss_x1e6", "peak_weighted_x1e6",
                    "motion_weighted_x1e6", "teacher_weighted_x1e6",
                    "motion_valid_pair_ratio", "mean_gt_motion_norm",
                    "mean_pred_motion_norm", "mean_motion_residual_norm", "fps",
                ]:
                    tb_writer.add_scalar(f"train/{key}", vals[key], global_step)
            for k, v in vals.items():
                if k not in ("epoch", "step"):
                    totals[k] += float(v)
            totals["batches"] += 1
            if global_step % 50 == 0:
                loss_file.flush()
                if tb_writer is not None:
                    tb_writer.flush()
            if args.log_every and global_step % args.log_every == 0:
                den = max(totals["batches"], 1)
                print(
                    f"epoch={epoch} step={global_step} loss={totals['total_loss']/den:.6e} "
                    f"heat={totals['heat_loss']/den:.6e} peak_w={totals['peak_weighted']/den:.6e} "
                    f"motion_w={totals['motion_weighted']/den:.6e} teacher_w={totals['teacher_weighted']/den:.6e} "
                    f"loss_x1e6={totals['total_loss_x1e6']/den:.2f} heat_x1e6={totals['heat_loss_x1e6']/den:.2f} "
                    f"peak_w_x1e6={totals['peak_weighted_x1e6']/den:.2f} motion_w_x1e6={totals['motion_weighted_x1e6']/den:.2f} "
                    f"teacher_w_x1e6={totals['teacher_weighted_x1e6']/den:.2f} "
                    f"valid_pair={totals['motion_valid_pair_ratio']/den:.3f} "
                    f"fps={vals['fps']:.2f}",
                    flush=True,
                )
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                # Mid-epoch resume repeats the current epoch, but preserves learned weights.
                save_checkpoint(ckpt_path, epoch - 1)
                print(f"STEP_CHECKPOINT step={global_step} saved={ckpt_path}", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        save_checkpoint(out_dir / f"epoch{epoch + 1}.pt", epoch)
        save_checkpoint(ckpt_path, epoch)
        den = max(totals["batches"], 1)
        print(f"EPOCH_DONE epoch={epoch} step={global_step} loss={totals['total_loss']/den:.6e} loss_x1e6={totals['total_loss_x1e6']/den:.2f} saved={out_dir / f'epoch{epoch + 1}.pt'}", flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
    loss_file.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
