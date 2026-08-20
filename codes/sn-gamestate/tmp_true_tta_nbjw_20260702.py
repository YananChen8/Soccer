import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import tmp_official_aux_report_eval_visual_20260701 as ref
from nbjw_calib.utils.utils_calib import keypoint_world_coords_2D
from nbjw_calib.utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, coords_to_dict
from nbjw_calib.utils.utils_keypoints import KeypointsDB
from nbjw_calib.utils.utils_linesWC import LineKeypointsWCDB


def _mirror_index(coords, width=105.0, tol=0.2):
    arr = np.asarray(coords, dtype=float)
    out = []
    for x, y in coords:
        target = np.asarray([width - float(x), float(y)], dtype=float)
        d = np.linalg.norm(arr - target[None, :], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) > tol:
            raise RuntimeError(f"missing mirror keypoint mapping for {(x, y)}")
        out.append(j)
    return out


def keypoint_swap():
    db = KeypointsDB({}, torch.zeros(3, 540, 960))
    return _mirror_index(db.keypoint_world_coords_2D) + [57]


def line_swap():
    obj = LineKeypointsWCDB(Image.new("RGB", (960, 540)), np.eye(3), (960, 540))
    names = [n.strip() for n in obj.lines_list]
    name_to_idx = {n: i for i, n in enumerate(names)}
    manual = {
        "Goal left crossbar": "Goal right crossbar",
        "Goal right crossbar": "Goal left crossbar",
        "Goal left post left": "Goal right post right",
        "Goal right post right": "Goal left post left",
        "Goal left post right": "Goal right post left",
        "Goal right post left": "Goal left post right",
        "Side line left": "Side line right",
        "Side line right": "Side line left",
    }

    def mirror_name(name):
        if name in manual:
            return manual[name]
        if ". left " in name:
            return name.replace(". left ", ". right ")
        if ". right " in name:
            return name.replace(". right ", ". left ")
        return name

    return [name_to_idx[mirror_name(n)] for n in names] + [23]


def align_flipped_heatmap(hm, swap):
    h = torch.flip(hm, dims=[-1])
    if h.shape[1] != len(swap):
        raise RuntimeError(f"heatmap channels={h.shape[1]} swap={len(swap)}")
    return h[:, swap, :, :]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def select_trainable(model, scope):
    freeze_all(model)
    selected = []
    if scope == "bn":
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    selected.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    selected.append(module.bias)
    elif scope == "head":
        for name, param in model.named_parameters():
            if name.startswith("head.") or ".head." in name:
                param.requires_grad_(True)
                selected.append(param)
    else:
        raise ValueError(f"unknown train scope: {scope}")
    if not selected:
        raise RuntimeError(f"no trainable parameters selected for scope={scope}")
    return selected


def peak_sharpness_loss(hm):
    semantic = hm[:, :-1].clamp_min(1e-8)
    flat = semantic.flatten(2)
    peak = flat.amax(dim=-1).mean()
    mass = flat.mean(dim=-1).mean()
    # Minimize negative peak while discouraging high response everywhere.
    return -peak + 0.05 * mass


def flip_consistency_loss(model, img, kp_swap, anchor=None, anchor_weight=0.05, peak_weight=0.0):
    raw = model(img)
    flipped = align_flipped_heatmap(model(torch.flip(img, dims=[-1])), kp_swap)
    loss = F.mse_loss(raw, flipped)
    if anchor is not None and anchor_weight > 0:
        loss = loss + anchor_weight * F.mse_loss(raw, anchor)
    if peak_weight > 0:
        loss = loss + peak_weight * peak_sharpness_loss(raw)
    return loss


def decode_anchor_keypoints(anchor, threshold):
    coords = get_keypoints_from_heatmap_batch_maxpool(anchor[:, :-1], max_keypoints=1)
    return coords_to_dict(coords, threshold=threshold)[0]


def ransac_inlier_ids(keypoints, ransac_px):
    world, image, ids = [], [], []
    for kid, kp in keypoints.items():
        if kid in [12, 15, 16, 19] or kid > 57:
            continue
        world.append(keypoint_world_coords_2D[kid - 1])
        image.append([kp["x"], kp["y"]])
        ids.append(kid)
    if len(ids) < 4:
        return set(), set(ids)
    _, status = cv2.findHomography(
        np.asarray(world, dtype=np.float32),
        np.asarray(image, dtype=np.float32),
        cv2.RANSAC,
        float(ransac_px),
    )
    if status is None:
        return set(), set(ids)
    inliers = {kid for kid, ok in zip(ids, status.reshape(-1)) if int(ok) == 1}
    return inliers, set(ids) - inliers


def gaussian_pseudo_from_keypoints(anchor, keypoints, inliers, sigma, weights=None):
    pseudo = torch.zeros_like(anchor)
    _, _, h, w = pseudo.shape
    yy = torch.arange(h, device=anchor.device, dtype=anchor.dtype).view(h, 1)
    xx = torch.arange(w, device=anchor.device, dtype=anchor.dtype).view(1, w)
    weights = weights or {}
    for kid in inliers:
        kp = keypoints.get(kid)
        if not kp:
            continue
        x = float(kp["x"]) / 2.0
        y = float(kp["y"]) / 2.0
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        weight = float(weights.get(kid, 1.0))
        pseudo[0, kid - 1] = weight * torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    return pseudo


def ransac_pseudo_from_keypoints(anchor, keypoints, ransac_px, sigma):
    inliers, outliers = ransac_inlier_ids(keypoints, ransac_px)
    pseudo = gaussian_pseudo_from_keypoints(anchor, keypoints, inliers, sigma)
    return pseudo, sorted(inliers), sorted(outliers), len(keypoints), {kid: 1.0 for kid in inliers}


def _keypoint_conf(kp):
    for key in ("confidence", "score", "value", "conf"):
        if key in kp:
            try:
                return max(0.0, float(kp[key]))
            except Exception:
                pass
    return 1.0


def weighted_ransac_pseudo_from_keypoints(anchor, keypoints, ransac_px, sigma, residual_tau=8.0, conf_gamma=0.5):
    world, image, ids = [], [], []
    for kid, kp in keypoints.items():
        if kid in [12, 15, 16, 19] or kid > 57:
            continue
        world.append(keypoint_world_coords_2D[kid - 1])
        image.append([kp["x"], kp["y"]])
        ids.append(kid)
    if len(ids) < 4:
        return torch.zeros_like(anchor), [], sorted(ids), len(keypoints), {}
    hmat, status = cv2.findHomography(
        np.asarray(world, dtype=np.float32),
        np.asarray(image, dtype=np.float32),
        cv2.RANSAC,
        float(ransac_px),
    )
    if hmat is None or status is None:
        return torch.zeros_like(anchor), [], sorted(ids), len(keypoints), {}
    world_h = np.concatenate([np.asarray(world, dtype=np.float64), np.ones((len(world), 1))], axis=1)
    proj = (np.asarray(hmat, dtype=np.float64) @ world_h.T).T
    ok = np.abs(proj[:, 2]) > 1e-6
    proj_xy = np.full((len(world), 2), np.nan, dtype=np.float64)
    proj_xy[ok] = proj[ok, :2] / proj[ok, 2:3]
    residual = np.linalg.norm(proj_xy - np.asarray(image, dtype=np.float64), axis=1)
    inliers = [kid for kid, keep in zip(ids, status.reshape(-1)) if int(keep) == 1]
    outliers = sorted(set(ids) - set(inliers))
    weights = {}
    tau = max(float(residual_tau), 1e-6)
    confs = np.asarray([_keypoint_conf(keypoints[kid]) for kid in ids], dtype=np.float64)
    cmax = float(np.nanmax(confs)) if np.isfinite(confs).any() else 1.0
    for kid, res, conf, keep in zip(ids, residual, confs, status.reshape(-1)):
        if int(keep) != 1 or not np.isfinite(res):
            continue
        conf_w = (float(conf) / max(cmax, 1e-6)) ** float(conf_gamma)
        geom_w = float(np.exp(-0.5 * (float(res) / tau) ** 2))
        weights[kid] = float(np.clip(conf_w * geom_w, 0.05, 1.0))
    pseudo = gaussian_pseudo_from_keypoints(anchor, keypoints, sorted(weights), sigma, weights=weights)
    return pseudo, sorted(weights), outliers, len(keypoints), weights


def temporal_pseudo_from_history(anchor, current, history, sigma, gate_px):
    if len(history) < 2:
        return torch.zeros_like(anchor), []
    prev2, prev1 = history[-2], history[-1]
    targets = {}
    _, _, h, w = anchor.shape
    for kid, kp in current.items():
        if kid not in prev1 or kid not in prev2:
            continue
        x2, y2 = float(prev2[kid]["x"]), float(prev2[kid]["y"])
        x1, y1 = float(prev1[kid]["x"]), float(prev1[kid]["y"])
        pred = {"x": 2.0 * x1 - x2, "y": 2.0 * y1 - y2}
        if pred["x"] < 0 or pred["y"] < 0 or pred["x"] >= w * 2 or pred["y"] >= h * 2:
            continue
        dist = ((float(kp["x"]) - pred["x"]) ** 2 + (float(kp["y"]) - pred["y"]) ** 2) ** 0.5
        if gate_px > 0 and dist > gate_px:
            continue
        targets[kid] = pred
    ids = sorted(targets)
    return gaussian_pseudo_from_keypoints(anchor, targets, ids, sigma), ids


def pseudo_label_loss(
    model, img, pseudo, inlier_ids, outlier_ids,
    temporal_pseudo=None, temporal_ids=None,
    anchor=None, anchor_weight=0.02, peak_weight=0.001,
    outlier_weight=0.0, temporal_weight=0.0, inlier_weights=None,
):
    pred = model(img)
    if inlier_ids:
        inlier_idx = torch.as_tensor([i - 1 for i in inlier_ids], device=pred.device, dtype=torch.long)
        per_ch = (pred[:, inlier_idx] - pseudo[:, inlier_idx]).pow(2).flatten(2).mean(dim=2).squeeze(0)
        if inlier_weights:
            w = torch.as_tensor([float(inlier_weights.get(i, 1.0)) for i in inlier_ids], device=pred.device, dtype=pred.dtype)
            loss = (per_ch * w).sum() / w.sum().clamp_min(1e-6)
        else:
            loss = per_ch.mean()
    else:
        loss = pred[:, :-1].mean() * 0.0
    if temporal_weight > 0 and temporal_ids:
        temporal_idx = torch.as_tensor([i - 1 for i in temporal_ids], device=pred.device, dtype=torch.long)
        loss = loss + temporal_weight * F.mse_loss(pred[:, temporal_idx], temporal_pseudo[:, temporal_idx])
    if outlier_weight > 0 and outlier_ids:
        outlier_idx = torch.as_tensor([i - 1 for i in outlier_ids], device=pred.device, dtype=torch.long)
        loss = loss + outlier_weight * pred[:, outlier_idx].clamp_min(0).mean()
    if anchor is not None and anchor_weight > 0:
        loss = loss + anchor_weight * F.mse_loss(pred, anchor)
    if peak_weight > 0:
        loss = loss + peak_weight * peak_sharpness_loss(pred)
    return loss


def adapt_model(model, img, kp_swap, method, args, temporal_history=None):
    if method in ("bn_flip", "flip_consistency"):
        params = select_trainable(model, "bn")
        actual_peak_weight = 0.0
    elif method == "head_flip_peak":
        params = select_trainable(model, "head")
        actual_peak_weight = args.peak_weight
    elif method in ("pseudo_label", "pseudo_label_weighted", "pseudo_label_temporal", "head_pseudo_label"):
        params = select_trainable(model, "head")
        actual_peak_weight = args.peak_weight
    else:
        raise ValueError(f"unsupported true TTA method: {method}")

    model.train()
    with torch.no_grad():
        anchor = model(img).detach()
        current_keypoints = decode_anchor_keypoints(anchor, args.pseudo_conf_threshold)
        if method == "pseudo_label_weighted":
            pseudo, inliers, outliers, n_decoded, inlier_weights = weighted_ransac_pseudo_from_keypoints(
                anchor,
                current_keypoints,
                ransac_px=args.pseudo_ransac_px,
                sigma=args.pseudo_sigma,
                residual_tau=args.pseudo_residual_tau,
                conf_gamma=args.pseudo_conf_gamma,
            )
        else:
            pseudo, inliers, outliers, n_decoded, inlier_weights = ransac_pseudo_from_keypoints(
                anchor,
                current_keypoints,
                ransac_px=args.pseudo_ransac_px,
                sigma=args.pseudo_sigma,
            )
        temporal_pseudo, temporal_ids = temporal_pseudo_from_history(
            anchor,
            current_keypoints,
            temporal_history or [],
            sigma=args.temporal_sigma,
            gate_px=args.temporal_gate_px,
        )

    def build_loss():
        if method in ("pseudo_label", "pseudo_label_weighted", "pseudo_label_temporal", "head_pseudo_label"):
            return pseudo_label_loss(
                model, img, pseudo,
                inlier_ids=inliers,
                outlier_ids=outliers,
                temporal_pseudo=temporal_pseudo,
                temporal_ids=temporal_ids if method == "pseudo_label_temporal" else [],
                anchor=anchor,
                anchor_weight=args.anchor_weight,
                peak_weight=actual_peak_weight,
                outlier_weight=args.outlier_weight,
                temporal_weight=args.temporal_weight,
                inlier_weights=inlier_weights if method == "pseudo_label_weighted" else None,
            )
        return flip_consistency_loss(
                model, img, kp_swap,
                anchor=anchor,
                anchor_weight=args.anchor_weight,
                peak_weight=actual_peak_weight,
            )

    opt = torch.optim.Adam(params, lr=args.lr)
    with torch.no_grad():
        loss_start = float(build_loss().detach().cpu())
    losses = [loss_start]
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = build_loss()
        loss.backward()
        opt.step()
    with torch.no_grad():
        losses.append(float(build_loss().detach().cpu()))
    model.eval()
    return losses, {
        "pseudo_decoded": n_decoded,
        "pseudo_inliers": len(inliers),
        "pseudo_outliers": len(outliers),
        "pseudo_weight_mean": float(np.mean(list(inlier_weights.values()))) if inlier_weights else 0.0,
        "temporal_keypoints": len(temporal_ids) if method == "pseudo_label_temporal" else 0,
    }


def mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def flatten_params(params):
    out = {}

    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(f"{prefix}.{i}", v)
        else:
            try:
                f = float(value)
                if np.isfinite(f):
                    out[prefix] = f
            except Exception:
                pass

    walk("", params or {})
    return out


def param_l2(a, b):
    av, bv = flatten_params(a), flatten_params(b)
    keys = sorted(set(av) | set(bv))
    if not keys:
        return None
    return float(np.linalg.norm([bv.get(k, 0.0) - av.get(k, 0.0) for k in keys]))


def load_memmap_source(cache_dir, videos, limit_videos):
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    image_path = Path(meta["image_path"])
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    shape = tuple(meta["image_shape"])
    mm = np.memmap(image_path, dtype=np.uint8, mode="r", shape=shape)
    by_video = {}
    with (cache_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = row["video"].replace("SNGS-", "")
            by_video.setdefault(video, []).append({
                "idx": int(row["idx"]),
                "frame": row["frame_index"],
                "gid": row["image_id"],
                "image_path": row.get("src_image") or row.get("dst_image") or "",
            })
    for rows in by_video.values():
        rows.sort(key=lambda r: int(r["frame"]))
    if len(videos) == 1 and videos[0].lower() == "all":
        videos = sorted(by_video, key=lambda x: int(x))
    if limit_videos:
        videos = videos[:limit_videos]
    return mm, by_video, videos


def evaluate(args):
    device = torch.device(args.device)
    frames_root, data_root = ref.get_split_paths("test")
    videos = [str(v).replace("SNGS-", "") for v in args.videos]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_swap = keypoint_swap()
    ln_swap = line_swap()
    (out_dir / "semantic_flip_mappings.json").write_text(
        json.dumps({"keypoint_swap": kp_swap, "line_swap": ln_swap}, indent=2),
        encoding="utf-8",
    )

    kp_model, line_model = ref.base.load_hrnets(device)
    line_model.eval()
    base_state = copy.deepcopy(kp_model.state_dict())
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    memmap = None
    memmap_rows = None
    if args.memmap_cache:
        memmap, memmap_rows, videos = load_memmap_source(args.memmap_cache, videos, args.video_limit)
    elif args.video_limit:
        videos = videos[:args.video_limit]

    gate_modes = [f"{m}_smoothgate" for m in args.methods] if args.smooth_gate else []
    modes = ["baseline_raw"] + args.methods + gate_modes
    rows_by_mode = {m: [] for m in modes}
    frame_rows = []
    loss_rows = []
    smooth_gate_rows = []

    for video in videos:
        if memmap_rows is None:
            files = [
                {"frame": p.stem, "gid": None, "image_path": str(p), "path": p}
                for p in sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
            ]
        else:
            files = memmap_rows.get(video, [])
        gt = ref.base.load_gt_lines_for_video(str(data_root), video)
        id_map = ref.image_id_map(data_root, video)
        keypoint_history = {m: [] for m in modes}
        smooth_prev = {m: None for m in args.methods}
        start = time.perf_counter()
        for idx, item in enumerate(files):
            if idx % args.stride != 0:
                continue
            frame = item["frame"]
            image_path = item["image_path"]
            gid = item.get("gid") or id_map.get(frame, f"3{video}{frame}")
            if gid not in gt:
                continue
            if memmap is None:
                img = tfm(Image.open(item["path"]).convert("RGB")).unsqueeze(0).to(device)
            else:
                img = torch.from_numpy(np.array(memmap[item["idx"]], copy=False)).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
            with torch.no_grad():
                line_hm = line_model(img)
                kp_model.load_state_dict(base_state)
                kp_model.eval()
                kp_raw = kp_model(img)
            outputs = {"baseline_raw": kp_raw}

            for method in args.methods:
                kp_model.load_state_dict(base_state)
                losses, meta = adapt_model(
                    kp_model, img, kp_swap,
                    method=method,
                    args=args,
                    temporal_history=keypoint_history.get(method, []),
                )
                with torch.no_grad():
                    outputs[method] = kp_model(img)
                loss_row = {
                    "method": method,
                    "video": video,
                    "frame": frame,
                    "loss_start": losses[0] if losses else None,
                    "loss_end": losses[-1] if losses else None,
                    "loss_delta": (losses[-1] - losses[0]) if len(losses) >= 2 else 0.0,
                }
                loss_row.update(meta)
                loss_rows.append(loss_row)

            scored_by_mode = {}
            for mode, kp_hm in outputs.items():
                s = ref.score_hm(kp_hm, line_hm, gt[gid])
                row = {
                    "run": mode,
                    "split": "test",
                    "video": video,
                    "frame": frame,
                    "image_path": str(image_path),
                    "point_acc": s["point_acc"],
                    "line_acc": s["line_acc"],
                    "reproj_mean": s["reproj_mean"],
                    "angle_to_midline_deg": ref.signed_angle_to_midline(s["params"]),
                    "signed_angle_to_midline_deg": ref.signed_angle_to_midline(s["params"]),
                    "folded_angle_to_midline_deg": ref.folded_angle_to_midline(s["params"]),
                    "reproj": s["reproj"],
                    "params": s["params"],
                }
                scored_by_mode[mode] = row
                rows_by_mode[mode].append(row)
                frame_rows.append({k: v for k, v in row.items() if k not in ("params", "reproj")})
                keypoint_history.setdefault(mode, []).append(
                    decode_anchor_keypoints(kp_hm.detach(), args.pseudo_conf_threshold)
                )
                keypoint_history[mode] = keypoint_history[mode][-2:]
            if args.smooth_gate:
                raw_row = scored_by_mode["baseline_raw"]
                for method in args.methods:
                    tta_row = scored_by_mode[method]
                    gate_name = f"{method}_smoothgate"
                    prev = smooth_prev[method]
                    if prev is None:
                        use_tta = False
                        raw_jump = None
                        tta_jump = None
                    else:
                        raw_jump = param_l2(prev, raw_row["params"])
                        tta_jump = param_l2(prev, tta_row["params"])
                        use_tta = (
                            tta_jump is not None and raw_jump is not None
                            and tta_jump <= raw_jump * args.smooth_gate_ratio + args.smooth_gate_tol
                        )
                    chosen = copy.deepcopy(tta_row if use_tta else raw_row)
                    chosen["run"] = gate_name
                    rows_by_mode[gate_name].append(chosen)
                    frame_rows.append({k: v for k, v in chosen.items() if k not in ("params", "reproj")})
                    smooth_gate_rows.append({
                        "method": method,
                        "video": video,
                        "frame": frame,
                        "use_tta": int(use_tta),
                        "raw_jump": raw_jump,
                        "tta_jump": tta_jump,
                    })
                    smooth_prev[method] = chosen["params"]
        print(f"video={video} seconds={time.perf_counter() - start:.1f}", flush=True)

    results = summarize_results(rows_by_mode, videos)
    loss_rows.extend(smooth_gate_rows)
    write_outputs(out_dir, results, frame_rows, loss_rows, args)


def summarize_results(rows_by_mode, videos):
    results = {}
    for mode, rows in rows_by_mode.items():
        results[mode] = {"videos": {}, "aggregate": None}
        for video in videos:
            vr = [r for r in rows if r["video"] == video]
            results[mode]["videos"][video] = ref.summarize(vr)
        vids = results[mode]["videos"]
        results[mode]["aggregate"] = {
            "point_acc": mean([v["point_acc"] for v in vids.values()]),
            "line_acc": mean([v["line_acc"] for v in vids.values()]),
            "reproj_mean": mean([v["reproj_mean"] for v in vids.values()]),
            "smooth_mean": mean([v["smooth_mean"] for v in vids.values()]),
            "JaC@5": mean([v["JaC@5"] for v in vids.values()]),
            "JaC@10": mean([v["JaC@10"] for v in vids.values()]),
            "JaC@15": mean([v["JaC@15"] for v in vids.values()]),
            "JaC@20": mean([v["JaC@20"] for v in vids.values()]),
            "MRE": mean([v["MRE"] for v in vids.values()]),
            "CR": mean([v["CR"] for v in vids.values()]),
            "Final Score": mean([v["Final Score"] for v in vids.values()]),
            "camera_smooth_l2_mean": mean([v["camera_smooth_l2_mean"] for v in vids.values()]),
            "camera_smooth_l2_p95": mean([v["camera_smooth_l2_p95"] for v in vids.values()]),
            "n_total": int(sum(v["n_total"] for v in vids.values())),
            "n_scored": int(sum(v["n_scored"] for v in vids.values())),
        }
    return results


def fmt(v, nd=6):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def write_outputs(out_dir, results, frame_rows, loss_rows, args):
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "test_frame_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "run", "split", "video", "frame", "image_path", "point_acc", "line_acc",
            "reproj_mean", "angle_to_midline_deg", "signed_angle_to_midline_deg",
            "folded_angle_to_midline_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)
    if loss_rows:
        with (out_dir / "tta_losses.csv").open("w", newline="", encoding="utf-8") as f:
            fields = sorted({k for r in loss_rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(loss_rows)
    ref.write_tables(results, out_dir, "test")

    raw = results["baseline_raw"]["aggregate"]
    keys = ["point_acc", "line_acc", "reproj_mean", "JaC@5", "JaC@10", "JaC@15", "JaC@20", "MRE", "CR", "Final Score", "camera_smooth_l2_mean", "camera_smooth_l2_p95"]
    lines = [
        "# True NBJW Test-Time Adaptation",
        "",
        f"- videos: {' '.join(args.videos)}",
        f"- stride: {args.stride}",
        f"- methods: {' '.join(args.methods)}",
        f"- steps: {args.steps}",
        f"- lr: {args.lr}",
        f"- anchor_weight: {args.anchor_weight}",
        f"- peak_weight: {args.peak_weight}",
        "- Adaptation loss uses no test GT. GT is used only by final evaluator.",
        "",
    ]
    for mode in [m for m in results if m != "baseline_raw"]:
        cur = results[mode]["aggregate"]
        lines += [
            f"## {mode}",
            "",
            "| metric | raw | tta | delta |",
            "|---|---:|---:|---:|",
        ]
        for k in keys:
            rv, tv = raw.get(k), cur.get(k)
            delta = None if rv is None or tv is None else tv - rv
            nd = 3 if k in ("MRE", "reproj_mean", "camera_smooth_l2_mean", "camera_smooth_l2_p95") else 6
            lines.append(f"| {k} | {fmt(rv, nd)} | {fmt(tv, nd)} | {fmt(delta, nd)} |")
        lines.append("")
    (out_dir / "true_tta_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["116"])
    ap.add_argument("--stride", type=int, default=80)
    ap.add_argument(
        "--methods", nargs="+",
        choices=["flip_consistency", "pseudo_label", "pseudo_label_weighted", "pseudo_label_temporal", "bn_flip", "head_flip_peak", "head_pseudo_label"],
        default=["flip_consistency"],
    )
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--anchor-weight", type=float, default=0.05)
    ap.add_argument("--peak-weight", type=float, default=0.001)
    ap.add_argument("--pseudo-conf-threshold", type=float, default=0.05)
    ap.add_argument("--pseudo-ransac-px", type=float, default=20.0)
    ap.add_argument("--pseudo-sigma", type=float, default=2.0)
    ap.add_argument("--pseudo-residual-tau", type=float, default=8.0)
    ap.add_argument("--pseudo-conf-gamma", type=float, default=0.5)
    ap.add_argument("--outlier-weight", type=float, default=0.001)
    ap.add_argument("--temporal-weight", type=float, default=0.1)
    ap.add_argument("--temporal-gate-px", type=float, default=30.0)
    ap.add_argument("--temporal-sigma", type=float, default=2.0)
    ap.add_argument("--smooth-gate", action="store_true")
    ap.add_argument("--smooth-gate-ratio", type=float, default=1.0)
    ap.add_argument("--smooth-gate-tol", type=float, default=0.0)
    ap.add_argument("--memmap-cache", default=None)
    ap.add_argument("--video-limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
