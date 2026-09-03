"""Tests for NBJW + Broadcast2Pitch localization refinement.

Run:
  python tests/test_refine_nbjw_with_b2p.py
  python -m pytest tests/test_refine_nbjw_with_b2p.py -q
"""

from __future__ import annotations

import argparse
import pickle
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import refine_nbjw_with_b2p as m  # noqa: E402


def make_template(root: Path) -> None:
    template_dir = root / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    pts = np.zeros((97, 3), dtype=np.float32)
    xs = np.linspace(10.0, 115.0, 97)
    ys = np.linspace(5.0, 73.0, 97)
    pts[:, 0] = xs
    pts[:, 1] = ys[::-1]
    pts[:, 2] = 1.0
    np.save(template_dir / "soccernet_template_97.npy", pts)


def make_state(path: Path, frames: int = 1, with_h: bool = True) -> np.ndarray:
    h = np.array([[0.08, 0.0, -50.0], [0.0, 0.08, -30.0], [0.0, 0.0, 1.0]], dtype=float)
    image_rows = []
    det_rows = []
    for frame in range(1, frames + 1):
        image_row = {
            "video_id": "021",
            "video_name": "SNGS-021",
            "file_path": f"SNGS-021/img1/{frame:06d}.jpg",
            "frame": frame,
            "parameters": {"synthetic": True},
        }
        if with_h:
            image_row["h"] = h.copy()
        image_rows.append(image_row)
        bbox = np.array([100.0 + frame, 200.0, 40.0, 80.0], dtype=np.float32)
        det_rows.append(
            {
                "id": frame,
                "video_id": "021",
                "image_id": frame,
                "category_id": 1,
                "bbox_ltwh": bbox,
                "bbox_conf": 1.0,
                "bbox_pitch": {"x_bottom_middle": 999.0, "y_bottom_middle": 999.0},
                "track_id": frame,
                "role": "player",
                "team": "left",
                "jersey": str(frame),
            }
        )
    image_df = pd.DataFrame(image_rows, index=list(range(1, frames + 1)))
    det_df = pd.DataFrame(det_rows)
    summary = {"columns": {"detection": list(det_df.columns), "image": list(image_df.columns)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("summary.json", __import__("json").dumps(summary))
        with zf.open("021.pkl", "w", force_zip64=True) as fh:
            pickle.dump(det_df, fh, protocol=pickle.DEFAULT_PROTOCOL)
        with zf.open("021_image.pkl", "w", force_zip64=True) as fh:
            pickle.dump(image_df, fh, protocol=pickle.DEFAULT_PROTOCOL)
    return h


def make_empty_caches(cache_dir: Path, frames: int) -> None:
    for frame in range(1, frames + 1):
        path = m.cache_path_for_frame(cache_dir, "SNGS-021", frame)
        m.save_observation_npz(path, m.empty_observation(path))


def make_args(tmp: Path, frames: int) -> argparse.Namespace:
    source = tmp / "source" / "sn-gamestate.pklz"
    out = tmp / "out" / "states" / "sn-gamestate.pklz"
    cache = tmp / "cache"
    b2p_root = tmp / "b2p"
    make_template(b2p_root)
    make_state(source, frames=frames)
    make_empty_caches(cache, frames)
    return argparse.Namespace(
        source_state=source,
        dataset_root=tmp / "datasets" / "SoccerNetGS",
        b2p_root=b2p_root,
        checkpoint=tmp / "b2p" / "checkpoints" / "dummy.pth",
        split="valid",
        videos=["SNGS-021"],
        cache_dir=cache,
        out_state=out,
        metrics_out=tmp / "out" / "frame_metrics.csv",
        b2p_python=None,
        skip_inference=True,
        b2p_device=None,
        max_frames=None,
        wl=1.0,
        wc=1.0,
        wk=1.0,
        trust_weight=1.0,
        trust_pixel_scale=75.0,
        min_keypoints=4,
        min_lines=2,
        min_line_points=8,
        min_circle_points=8,
        keypoint_conf=0.4,
        line_conf=0.8,
        max_line_points_per_class=40,
        max_lm_nfev=20,
        max_anchor_shift_px=120.0,
        max_pitch_abs_x=80.0,
        max_pitch_abs_y=60.0,
        overlay_dir=None,
        num_overlays=0,
        no_overlays=True,
        eval_out=None,
        eval_threshold=5.0,
        nproc=1,
    )


def test_homography_direction_and_coordinate_conversion():
    h_img_to_template = np.array(
        [[0.25, 0.0, 10.0], [0.0, 0.20, 5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    assert m.verify_b2p_conversion_with_synthetic_points(h_img_to_template, "image_to_template")
    h_template_to_img = np.linalg.inv(h_img_to_template)
    assert m.verify_b2p_conversion_with_synthetic_points(h_template_to_img, "template_to_image")

    template_pts = np.array([[10.0, 5.0], [115.0, 5.0], [115.0, 73.0], [10.0, 73.0]], dtype=float)
    image_pts, ok_img = m.project_points(h_template_to_img, template_pts)
    assert np.all(ok_img)
    h_soccer = m.convert_b2p_homography_to_soccer(h_img_to_template, "image_to_template")
    soccer_pts, ok_soccer = m.project_points(h_soccer, image_pts)
    assert np.all(ok_soccer)
    np.testing.assert_allclose(soccer_pts, template_pts - np.array([62.5, 39.0]), atol=1e-7)


def test_bbox_bottom_center_projection():
    h = np.array([[0.1, 0.0, -1.0], [0.0, 0.2, -2.0], [0.0, 0.0, 1.0]], dtype=float)
    pitch = m.bbox_ltwh_to_pitch(h, np.array([10.0, 20.0, 30.0, 40.0]))
    assert pitch is not None
    assert pitch["x_bottom_left"] == 0.0
    assert pitch["y_bottom_left"] == 10.0
    assert pitch["x_bottom_right"] == 3.0
    assert pitch["y_bottom_right"] == 10.0
    assert pitch["x_bottom_middle"] == 1.5
    assert pitch["y_bottom_middle"] == 10.0


def test_lm_failure_falls_back_to_nbjw():
    h0 = np.array([[0.08, 0.0, -50.0], [0.0, 0.08, -30.0], [0.0, 0.0, 1.0]], dtype=float)
    inv_h0 = np.linalg.inv(h0)
    template = np.zeros((97, 3), dtype=np.float32)
    template[:4, :2] = np.array([[10.0, 5.0], [115.0, 5.0], [115.0, 73.0], [10.0, 73.0]], dtype=float)
    template[:, 2] = 1.0
    template_soccer = m.template_keypoints_soccer(template)
    image_kpts, _ = m.project_points(inv_h0, template_soccer[:4])
    keypoints = np.zeros((97, 3), dtype=np.float32)
    keypoints[:4, :2] = image_kpts
    keypoints[:4, 2] = 0.95
    line_points = {}
    for line_name in ["Side line top", "Side line bottom"]:
        endpoints = np.asarray(m.PITCH_LINE_ENDPOINTS_TEMPLATE[line_name], dtype=float)[:, :2]
        soccer_line = m.transform_template_points_to_soccer(
            endpoints[0][None, :] * (1 - np.linspace(0, 1, 10)[:, None])
            + endpoints[1][None, :] * np.linspace(0, 1, 10)[:, None]
        )
        img_line, _ = m.project_points(inv_h0, soccer_line)
        line_points[line_name] = np.c_[img_line, np.ones((len(img_line),), dtype=float)]
    obs = m.FrameObservation(
        keypoints=keypoints,
        line_points=line_points,
        line_confidences={name: 1.0 for name in m.LINE_NAMES},
        circle_points=np.zeros((0, 3), dtype=np.float32),
        cache_path=Path("synthetic.npz"),
    )

    class Failed:
        success = False
        x = m.homography_to_params(h0)

    result = m.refine_homography_with_observations(
        h0,
        obs,
        template_soccer,
        m.RefinementConfig(min_keypoints=4, min_lines=2, min_line_points=8),
        least_squares_fn=lambda *args, **kwargs: Failed(),
    )
    assert not result.accepted
    assert result.fallback_reason == "solver_failed"
    np.testing.assert_allclose(result.final_h, h0)


def test_state_field_invariance_and_h_diagnostics():
    with TemporaryDirectory() as d:
        tmp = Path(d)
        args = make_args(tmp, frames=1)
        summary = m.refine_state(args)
        assert summary["state_invariance"]["passed"] is True
        with zipfile.ZipFile(args.out_state, "r") as zf:
            det = pickle.load(zf.open("021.pkl"))
            img = pickle.load(zf.open("021_image.pkl"))
        assert len(det) == 1
        np.testing.assert_array_equal(det.iloc[0]["bbox_ltwh"], np.array([101.0, 200.0, 40.0, 80.0], dtype=np.float32))
        assert det.iloc[0]["track_id"] == 1
        assert det.iloc[0]["role"] == "player"
        assert det.iloc[0]["team"] == "left"
        assert det.iloc[0]["jersey"] == "1"
        assert "h_nbjw" in img.columns
        assert "h_refined" in img.columns
        assert img.iloc[0]["b2p_fallback_reason"] == "too_few_keypoints"


def test_five_frame_smoke_test():
    with TemporaryDirectory() as d:
        tmp = Path(d)
        args = make_args(tmp, frames=5)
        summary = m.refine_state(args)
        assert summary["frames"] == 5
        assert summary["fallback"] == 5
        metrics = pd.read_csv(args.metrics_out)
        assert len(metrics) == 5
        assert set(metrics["fallback_reason"]) == {"too_few_keypoints"}


def test_missing_h_state_is_diagnostic_only():
    with TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source" / "sn-gamestate.pklz"
        make_state(source, frames=1, with_h=False)
        ok, diagnosis = m.inspect_state_has_h(source, ["SNGS-021"])
        assert not ok
        assert "missing h column" in diagnosis


if __name__ == "__main__":
    test_homography_direction_and_coordinate_conversion()
    test_bbox_bottom_center_projection()
    test_lm_failure_falls_back_to_nbjw()
    test_state_field_invariance_and_h_diagnostics()
    test_five_frame_smoke_test()
    test_missing_h_state_is_diagnostic_only()
    print("ALL OK")
