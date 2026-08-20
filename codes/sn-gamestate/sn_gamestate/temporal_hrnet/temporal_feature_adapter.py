"""Temporal Feature Adapter (Plan B) — pluggable SKELETON only.

Plan B inserts temporal fusion at the HRNet *feature* level instead of the
heatmap level. For NBJW's HRNet the natural tap is the fused multi-resolution
feature right before the head:

    cls_hrnet.HighResolutionNet.forward(): the 4 branches are interpolated to
    branch-0 resolution (135x240) and concatenated -> x  [B, 720, 135, 240]
    (channels = 48+96+192+384). Then self.upsample(x2) + self.head(x) -> heatmap.

So the clean single-scale tap is that concatenated 720-ch feature at 135x240.
A feature adapter would refine x' = x + delta(x_seq) and feed x' to head.

RISK (why this is skeleton-only in v1):
  * Requires splitting HRNet forward into encoder() / head() — cls_hrnet.py
    today does both in one forward(). That edit touches the frozen model file.
  * 720 channels at 135x240 is ~23M floats/frame -> caching feature windows is
    far heavier than heatmap windows. Online K-frame forward is the only sane
    route, which means running the backbone K times per frame.
  * Plan A already gives a strictly easier path with identical post-processing.

==> Implement Plan A first. This file gives the module so the wiring exists,
    but the encoder/head split is left as a documented TODO.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalFeatureAdapter(nn.Module):
    """Single-scale feature refinement. Same shape contract as Plan A but on
    feature channels D instead of heatmap channels C.

    F_seq [B, K, D, h, w] -> F_refined [B, D, h, w].

    For NBJW: D=720, h=135, w=240 (the pre-head concatenated feature).
    Multi-scale handling: v1 only refines this single highest-resolution
    branch-0 fused feature (the spec's "only highest-resolution branch"). Adding
    per-scale adapters is a later extension and would tap inside the stage loop.
    """

    def __init__(self, channels: int, window_size: int = 3,
                 residual_scale: float = 1.0, hidden: int = 256):
        super().__init__()
        self.window_size = window_size
        self.residual_scale = residual_scale
        self.temporal = nn.Conv3d(
            channels, channels, kernel_size=(3, 3, 3),
            padding=(1, 1, 1), groups=channels,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        if self.proj[-1].bias is not None:
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, f_seq: torch.Tensor):
        f_t = f_seq[:, -1]
        if self.residual_scale == 0:
            return f_t
        x = f_seq.permute(0, 2, 1, 3, 4)   # [B, D, K, h, w]
        x = self.temporal(x)[:, :, -1]     # [B, D, h, w]
        delta = self.proj(x)
        return f_t + self.residual_scale * delta


# TODO(plan-b-wiring): to actually use this, refactor
# cls_hrnet.HighResolutionNet.forward into:
#     def encode(self, x) -> feat_720x135x240   # everything up to concat
#     def head_forward(self, feat) -> heatmap    # upsample + self.head
# then in NBJW_Calib_Keypoints buffer K encoded features and call
#     feat = adapter(stack(feats)); heatmap = model.head_forward(feat)
# Keep the original forward() intact (forward = head_forward(encode(x))) so
# nothing downstream changes when the adapter is disabled.


if __name__ == "__main__":
    B, K, D, h, w = 2, 3, 720, 135, 240
    a = TemporalFeatureAdapter(channels=D, window_size=K)
    out = a(torch.randn(B, K, D, h, w))
    assert out.shape == (B, D, h, w)
    # zero-init identity
    x = torch.randn(B, K, D, h, w)
    assert torch.allclose(a(x), x[:, -1], atol=1e-6)
    print("ok: feature adapter skeleton shape + zero-init identity")
