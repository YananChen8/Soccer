#!/usr/bin/env python3
"""Minimal calibration-only runner — NO tracklab pipeline, NO QWen, NO PRTReId.

For each experiment:
  1. Load frozen HRNet + optional temporal adapter
  2. Run NBJW calib on SNGS-116/117/118 frame by frame (read from images)
  3. Write per-frame pklz with keypoints + camera parameters
Output: {out_dir}/states/sn-gamestate.pklz (zip of per-video .pkl files)
"""
import os, sys, time, zipfile, pickle, io, json
from pathlib import Path
from collections import deque

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

SNGSR = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
CKPT_DIR = "/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/lib/python3.10/site-packages"
sys.path.insert(0, str(SNGSR / "plugins/calibration"))
sys.path.insert(0, str(SNGSR))

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, coords_to_dict, complete_keypoints
from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from nbjw_calib.nbjw_calib import NBJW_Calib_Keypoints
from sn_gamestate.temporal_hrnet import (
    TemporalHeatmapAdapter, KeypointTokenTemporalAdapter, heatmaps_to_tokens, pad_window
)

# ---- args ----
name     = sys.argv[1]  # e.g. baseline_rs0
ckpt_rel = sys.argv[2]  # "" = baseline
scale    = float(sys.argv[3])
gpu      = sys.argv[4]

CKPT_BASE = SNGSR / "outputs/gsr/temporal_hrnet/quick_subset12"
OUT_BASE  = SNGSR / "outputs/gsr/temporal_hrnet/quick_subset12_calib_only"
VIDEOS    = ["SNGS-116", "SNGS-117", "SNGS-118"]
IMG_DIR   = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS/test")

ckpt_path = str(CKPT_BASE / ckpt_rel) if ckpt_rel else ""
out_dir   = OUT_BASE / name
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(f"cuda:{gpu}")
resize = T.Resize((540, 960))
to_tensor = T.ToTensor()

# ---- load HRNets ----
print(f"[{time.strftime('%H:%M:%S')}] Loading HRNets...", flush=True)
cfg = {
    'MODEL': {'IMAGE_SIZE': [960, 540], 'NUM_JOINTS': 58, 'PRETRAIN': '',
              'EXTRA': {'FINAL_CONV_KERNEL': 1,
                        'STAGE1': {'NUM_MODULES':1,'NUM_BRANCHES':1,'BLOCK':'BOTTLENECK','NUM_BLOCKS':[4],'NUM_CHANNELS':[64],'FUSE_METHOD':'SUM'},
                        'STAGE2': {'NUM_MODULES':1,'NUM_BRANCHES':2,'BLOCK':'BASIC','NUM_BLOCKS':[4,4],'NUM_CHANNELS':[48,96],'FUSE_METHOD':'SUM'},
                        'STAGE3': {'NUM_MODULES':4,'NUM_BRANCHES':3,'BLOCK':'BASIC','NUM_BLOCKS':[4,4,4],'NUM_CHANNELS':[48,96,192],'FUSE_METHOD':'SUM'},
                        'STAGE4': {'NUM_MODULES':3,'NUM_BRANCHES':4,'BLOCK':'BASIC','NUM_BLOCKS':[4,4,4,4],'NUM_CHANNELS':[48,96,192,384],'FUSE_METHOD':'SUM'}}}}
cfg_l = {k: dict(v) for k, v in cfg.items()}
cfg_l['MODEL']['NUM_JOINTS'] = 24

kp_model = get_cls_net(cfg).to(device).eval()
line_model = get_cls_net_l(cfg_l).to(device).eval()

CKPT_ROOT = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"
kp_model.load_state_dict(torch.load(f"{CKPT_ROOT}/SV_kp", map_location=device))
line_model.load_state_dict(torch.load(f"{CKPT_ROOT}/SV_lines", map_location=device))
for p in kp_model.parameters(): p.requires_grad_(False)
for p in line_model.parameters(): p.requires_grad_(False)

# ---- load adapter if any ----
class _TokenWrapper(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
    def forward(self, h_seq):
        b, k, c, hw, ww = h_seq.shape
        tokens = torch.stack([heatmaps_to_tokens(h_seq[:, t]) for t in range(k)], dim=1)
        return self.adapter(tokens, h_seq[:, -1])

adapter = None
window = 3
if ckpt_path:
    ck = torch.load(ckpt_path, map_location=device)
    which = ck.get("which", "")
    if which == "kp":
        a = TemporalHeatmapAdapter(
            ck["channels"], ck["window_size"],
            residual_scale=scale,
            adapter_type=ck.get("adapter_type", "depthwise_conv3d"),
            mix_hidden=ck.get("mix_hidden", 128),
        )
    else:
        a = KeypointTokenTemporalAdapter(
            channels=ck["channels"], window_size=ck["window_size"],
            architecture=ck["architecture"], hidden=ck.get("hidden", 64),
            residual_scale=scale, max_shift_px=ck.get("max_shift_px", 12.0),
        )
        a = _TokenWrapper(a)
    a.load_state_dict(ck["state_dict"])
    a.to(device).eval()
    adapter = a
    window = ck["window_size"]
    print(f"[{time.strftime('%H:%M:%S')}] Adapter loaded: {which} window={window} scale={scale}", flush=True)
else:
    print(f"[{time.strftime('%H:%M:%S')}] No adapter (baseline)", flush=True)

# ---- solvers ----
calib_kp = FramebyFrameCalib()
calib_line = FramebyFrameCalib()
calib_kp.image_width, calib_kp.image_height = 1920, 1080
calib_line.image_width, calib_line.image_height = 1920, 1080

# ---- process each video ----
states_dir = out_dir / "states"
states_dir.mkdir(exist_ok=True)

all_frames = {}  # {video_id: [frame_dicts]}

for vid in VIDEOS:
    vid_dir = IMG_DIR / vid / "img1"
    if not vid_dir.exists():
        print(f"MISSING: {vid_dir}", flush=True)
        continue
    frames_list = sorted(vid_dir.glob("*.jpg"))
    print(f"[{time.strftime('%H:%M:%S')}] {vid}: {len(frames_list)} frames", flush=True)

    buf = deque(maxlen=window)
    video_frames = []

    for fi, img_path in enumerate(frames_list):
        img = Image.open(img_path).convert("RGB")
        img_540 = resize(img)
        x = to_tensor(img_540).unsqueeze(0).to(device)  # [1,3,540,960]

        with torch.no_grad():
            hm = kp_model(x)          # [1,58,270,480]
            hm_l = line_model(x)      # [1,24,270,480]

            if adapter is not None:
                buf.append(hm)
                win = pad_window(torch.stack(list(buf), dim=1), window)
                hm, _ = adapter(win)

        # Decode keypoints
        hm_np = hm.float().cpu().numpy()[0]          # [58,270,480]
        hm_l_np = hm_l.float().cpu().numpy()[0]      # [24,270,480]

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(hm_np[None, :-1])[0]
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(hm_l_np[None, :-1])[0]

        kp_dict = coords_to_dict(kp_coords, threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)
        final_kp = complete_keypoints(kp_dict, lines_dict, 1920, 1080, normalize=True)

        # Solve camera
        success, cam_params = calib_kp.calibrate(final_kp, lines_dict)
        if not success:
            success, cam_params = calib_kp.calibrate(final_kp, lines_dict, use_prev_homography=True)

        if success:
            params = {
                "pan": float(cam_params.pan), "tilt": float(cam_params.tilt),
                "roll": float(cam_params.roll),
                "x_focal": float(cam_params.x_focal), "y_focal": float(cam_params.y_focal),
                "principal_point": [float(cam_params.pp_x), float(cam_params.pp_y)],
                "position_meters": [float(cam_params.pos[0]), float(cam_params.pos[1]), float(cam_params.pos[2])],
                "rotation_matrix": cam_params.rot_m.tolist() if hasattr(cam_params, 'rot_m') else None,
                "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        else:
            params = {}

        frame_data = {
            "image_id": fi,
            "keypoints": {int(k): {"x": float(v[0]), "y": float(v[1]), "p": float(v[2])}
                          for k, v in final_kp.items()} if success else {},
            "parameters": params,
        }
        video_frames.append(frame_data)

    all_frames[vid] = video_frames
    print(f"[{time.strftime('%H:%M:%S')}] {vid}: done, {len(video_frames)} frames, success_rate={sum(1 for f in video_frames if f['parameters'])/len(video_frames):.1%}", flush=True)

# ---- write pklz ----
pklz_path = states_dir / "sn-gamestate.pklz"
with zipfile.ZipFile(pklz_path, 'w', zipfile.ZIP_STORED) as zf:
    summary = {"video_ids": list(all_frames.keys()), "columns": ["keypoints", "parameters"]}
    zf.writestr("summary.json", json.dumps(summary))
    for vid, frames in all_frames.items():
        zf.writestr(f"{vid}.pkl", pickle.dumps(frames))
        zf.writestr(f"{vid}_image.pkl", pickle.dumps(frames))

sz_mb = pklz_path.stat().st_size / 1e6
print(f"[{time.strftime('%H:%M:%S')}] DONE {name}: {pklz_path} ({sz_mb:.1f} MB)", flush=True)
