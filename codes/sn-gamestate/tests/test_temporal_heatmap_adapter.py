"""Tests 1-3 for TemporalHeatmapAdapter. Pure torch, no GPU, no checkpoint.

Run:  python -m pytest tests/test_temporal_heatmap_adapter.py -q
  or:  python tests/test_temporal_heatmap_adapter.py   (asserts, no pytest)

Test 4 (valid10 tiny smoke) lives with cache_hrnet_heatmaps.py since it needs
the remote HRNet + frames.
"""
import torch

from sn_gamestate.temporal_hrnet import TemporalHeatmapAdapter, pad_window

# NBJW real shapes (verified on 202): kp [B,58,270,480], line [B,24,270,480]
KP_C, LN_C, H, W = 58, 24, 270, 480


def test_shape_and_backward():
    B, K = 2, 3
    a = TemporalHeatmapAdapter(channels=KP_C, window_size=K)
    h_seq = torch.randn(B, K, KP_C, H, W)
    refined, delta = a(h_seq)
    assert refined.shape == (B, KP_C, H, W)
    assert delta.shape == (B, KP_C, H, W)
    refined.sum().backward()  # backward must run
    assert a.decoder[-1].weight.grad is not None


def test_zero_init_identity():
    B, K = 1, 4
    a = TemporalHeatmapAdapter(channels=LN_C, window_size=K)
    h_seq = torch.randn(B, K, LN_C, H, W)
    refined, delta = a(h_seq)
    max_abs = delta.abs().max().item()
    mean_abs = delta.abs().mean().item()
    print(f"zero-init delta: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}")
    assert max_abs < 1e-6
    assert torch.allclose(refined, h_seq[:, -1], atol=1e-6)


def test_residual_scale_zero_passthrough():
    a = TemporalHeatmapAdapter(channels=LN_C, window_size=3, residual_scale=0.0)
    h_seq = torch.randn(1, 3, LN_C, H, W)
    refined, delta = a(h_seq)
    assert torch.equal(refined, h_seq[:, -1])
    assert torch.count_nonzero(delta) == 0


def test_full_window_adapter():
    a = TemporalHeatmapAdapter(
        channels=LN_C,
        window_size=15,
        adapter_type="depthwise_full_window",
        mix_hidden=16,
    )
    h_seq = torch.randn(1, 15, LN_C, 16, 24)
    refined, delta = a(h_seq)
    assert refined.shape == (1, LN_C, 16, 24)
    assert delta.shape == refined.shape
    refined.sum().backward()
    assert a.temporal.weight.grad is not None


def test_short_window_padding():
    # video start: only 1 frame available, must not crash
    one = torch.randn(2, 1, KP_C, H, W)
    padded = pad_window(one, k=3)
    assert padded.shape == (2, 3, KP_C, H, W)
    assert torch.equal(padded[:, 0], padded[:, -1])  # repeat-pad oldest==current


def test_checkpoint_compat_simulation():
    """Test 3 proxy without the real checkpoint: a tiny stand-in 'HRNet' whose
    output is wrapped by the adapter. Disabled / zero-init => identical output."""
    torch.manual_seed(0)
    fake_hrnet_out = torch.randn(1, KP_C, H, W)          # H_t from frozen HRNet
    h_seq = torch.stack([fake_hrnet_out] * 3, dim=1)      # repeat-padded window
    a = TemporalHeatmapAdapter(channels=KP_C, window_size=3)  # zero-init
    refined, _ = a(h_seq)
    diff = (refined - fake_hrnet_out).abs().max().item()
    print(f"checkpoint-compat max heatmap diff (zero-init): {diff:.3e}")
    assert diff < 1e-6  # frozen-HRNet output unchanged => coords/camera unchanged


if __name__ == "__main__":
    test_shape_and_backward()
    test_zero_init_identity()
    test_residual_scale_zero_passthrough()
    test_full_window_adapter()
    test_short_window_padding()
    test_checkpoint_compat_simulation()
    print("ALL OK")
