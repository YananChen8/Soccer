"""Temporal NBJW keypoints module — non-destructive wiring of Plan A.

Subclasses NBJW_Calib_Keypoints and transparently wraps its two HRNets so the
INHERITED process() runs unchanged: self.model(batch) now returns temporally
refined heatmaps before the existing decode. The original nbjw_calib.py is never
touched.

  * adapter ckpt = None  -> wrapper is pass-through -> byte-identical to baseline.
  * reset() (called per video by OfflineTrackingEngine) clears the buffers.

Config: point pitch module _target_ at this class and add
  kp_adapter_ckpt / line_adapter_ckpt / adapter_window
(see configs/modules/pitch/temporal_nbjw_calib.yaml). Omit them -> baseline.

ponytail: assumes batch_size=1 (the nbjw pitch config) so the per-frame rolling
buffer aligns frame-to-frame. Higher batch would need per-sequence buffers.
"""
import os
import torch
import torch.nn as nn

from sn_gamestate.calibration.nbjw_calib import NBJW_Calib_Keypoints
from sn_gamestate.temporal_hrnet import (
    KeypointTokenTemporalAdapter,
    SparseTemporalKeypointAdapter,
    TemporalHeatmapAdapter,
    heatmaps_to_tokens,
    pad_window,
)


class _TokenAdapterWrapper(nn.Module):
    """Wraps KeypointTokenTemporalAdapter so _TemporalHRNetWrapper sees
    forward(heatmap_window [B,K,C,h,w]) -> (refined [B,C,h,w], delta)."""

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, h_seq):
        # h_seq: [B, K, C, h, w]
        b, k, c, h, w = h_seq.shape
        tokens = []
        for t in range(k):
            tokens.append(heatmaps_to_tokens(h_seq[:, t]))
        tokens = torch.stack(tokens, dim=1)  # [B, K, C, 3]
        return self.adapter(tokens, h_seq[:, -1])


class _TemporalHRNetWrapper(nn.Module):
    """Wraps one HRNet: forward = HRNet(x) then optional temporal refine."""

    def __init__(self, hrnet, adapter, window):
        super().__init__()
        self.hrnet = hrnet
        self.adapter = adapter        # None -> pass-through
        self.window = window
        self.buf = []

    def reset(self):
        self.buf = []

    def forward(self, x):
        hm = self.hrnet(x)            # [B,C,h,w]
        if self.adapter is None:
            return hm
        self.buf.append(hm.detach())
        if len(self.buf) > self.window:
            self.buf.pop(0)
        win = pad_window(torch.stack(self.buf, dim=1), self.window)  # [B,K,C,h,w]
        with torch.no_grad():
            refined, _ = self.adapter(win)
        return refined


def _load_adapter(path, device):
    if not path or not os.path.isfile(path):
        return None
    ck = torch.load(path, map_location=device)
    which = ck.get("which", "")

    if ck.get("model_family") == "sparse_keypoint":
        a = SparseTemporalKeypointAdapter(
            ck["channels"],
            ck["window_size"],
            architecture=ck["architecture"],
            hidden=ck.get("hidden", 64),
            residual_scale=ck.get("residual_scale", 1.0),
            max_shift_px=ck.get("max_shift_px", 8.0),
        )
        a.load_state_dict(ck["state_dict"])
    elif which == "kp_token":
        raw_adapter = KeypointTokenTemporalAdapter(
            channels=ck["channels"],
            window_size=ck["window_size"],
            architecture=ck["architecture"],
            hidden=ck.get("hidden", 64),
            residual_scale=ck.get("residual_scale", 1.0),
            max_shift_px=ck.get("max_shift_px", 12.0),
        )
        raw_adapter.load_state_dict(ck["state_dict"])
        a = _TokenAdapterWrapper(raw_adapter)
    elif which == "kp":
        a = TemporalHeatmapAdapter(
            ck["channels"],
            ck["window_size"],
            residual_scale=ck.get("residual_scale", 1.0),
            adapter_type=ck.get("adapter_type", "depthwise_conv3d"),
            mix_hidden=ck.get("mix_hidden", 128),
        )
        a.load_state_dict(ck["state_dict"])
    else:
        a = TemporalHeatmapAdapter(
            ck["channels"],
            ck["window_size"],
            residual_scale=ck.get("residual_scale", 1.0),
            adapter_type=ck.get("adapter_type", "depthwise_conv3d"),
            mix_hidden=ck.get("mix_hidden", 128),
        )
        a.load_state_dict(ck["state_dict"])
    # inference-time strength override: gentler refinement keeps keypoints in the
    # calibration solver's stable regime. ADAPTER_RESIDUAL_SCALE=0 => baseline.
    ov = os.environ.get("ADAPTER_RESIDUAL_SCALE")
    if ov is not None:
        scale_override = float(ov)
        # Reach into wrapper if present
        if hasattr(a, "adapter"):
            a.adapter.residual_scale = scale_override
        elif hasattr(a, "residual_scale"):
            a.residual_scale = scale_override
        print(f"[adapter] residual_scale override -> {scale_override}")
    return a.to(device).eval()


class TemporalNBJWKeypoints(NBJW_Calib_Keypoints):
    def __init__(self, *args, kp_adapter_ckpt=None, line_adapter_ckpt=None,
                 adapter_window=3, **kwargs):
        super().__init__(*args, **kwargs)
        kp_a = _load_adapter(kp_adapter_ckpt, self.device)
        ln_a = _load_adapter(line_adapter_ckpt, self.device)
        win_kp = kp_a.adapter.window_size if (kp_a is not None and hasattr(kp_a, 'adapter')) else (
            kp_a.window_size if kp_a is not None else adapter_window
        )
        win_ln = ln_a.adapter.window_size if (ln_a is not None and hasattr(ln_a, 'adapter')) else (
            ln_a.window_size if ln_a is not None else adapter_window
        )
        self.model = _TemporalHRNetWrapper(self.model, kp_a, win_kp).to(self.device)
        self.model_l = _TemporalHRNetWrapper(self.model_l, ln_a, win_ln).to(self.device)
        n = sum(a is not None for a in (kp_a, ln_a))
        active_windows = [
            wrapper.window for wrapper in (self.model, self.model_l)
            if wrapper.adapter is not None
        ]
        print(f"[TemporalNBJWKeypoints] {n}/2 adapters active (windows={active_windows})")

    @property
    def level(self):
        # tracklab derives level from __bases__[0].__name__; since we subclass
        # NBJW_Calib_Keypoints (not ImageLevelModule directly) it would wrongly
        # become "nbjw". Pin it to "image".
        return "image"

    def reset(self):
        # OfflineTrackingEngine calls this per video; clear temporal buffers.
        self.model.reset()
        self.model_l.reset()


if __name__ == "__main__":
    # ponytail self-check: wrapper buffer + reset + pass-through (no HRNet load).
    class FakeHRNet(nn.Module):
        def forward(self, x):
            return torch.randn(1, 58, 270, 480)
    w = _TemporalHRNetWrapper(FakeHRNet(), None, 3)
    out = w(torch.randn(1, 3, 540, 960))
    assert out.shape == (1, 58, 270, 480) and len(w.buf) == 0  # pass-through
    a = TemporalHeatmapAdapter(58, 3)
    w2 = _TemporalHRNetWrapper(FakeHRNet(), a, 3)
    for i in range(5):
        w2(torch.randn(1, 3, 540, 960))
    assert len(w2.buf) == 3                 # buffer capped at window
    w2.reset(); assert len(w2.buf) == 0     # reset clears
    print("ok: wrapper pass-through, buffer cap, reset")
