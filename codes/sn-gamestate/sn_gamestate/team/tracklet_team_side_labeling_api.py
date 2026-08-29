import pandas as pd
import torch
import numpy as np
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging


log = logging.getLogger(__name__)


class TrackletTeamSideLabeling(VideoLevelModule):
    """
    This module labels the team side (left, right) of the detections with role = {'player', 'goalkeeper'} based on the team clustering.
    Team 'left'/'right' means the team with its goal on the left/right side of the image (from the camera perspective).
    """

    input_columns = ["track_id", "team_cluster", "bbox_pitch", "role"]
    output_columns = ["team"]
    
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.use_global_pitch_mapping = self._cfg_bool("use_global_pitch_mapping", False)
        self.min_track_x_gap = self._cfg_float("min_track_x_gap", 3.0)
        self.min_tracks_per_team = self._cfg_int("min_tracks_per_team", 2)
        self.window_sizes = self._cfg_list("window_sizes", [150, 300, 600, -1])

    def _cfg_get(self, name, default):
        if self.cfg is None:
            return default
        if isinstance(self.cfg, dict):
            return self.cfg.get(name, default)
        return getattr(self.cfg, name, default)

    def _cfg_bool(self, name, default):
        value = self._cfg_get(name, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _cfg_float(self, name, default):
        value = self._cfg_get(name, default)
        return float(default if value is None else value)

    def _cfg_int(self, name, default):
        value = self._cfg_get(name, default)
        return int(default if value is None else value)

    def _cfg_list(self, name, default):
        value = self._cfg_get(name, default)
        if value is None:
            return list(default)
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return [int(item) for item in value]

    @staticmethod
    def _pitch_x(bbox):
        if isinstance(bbox, dict):
            value = bbox.get("x_bottom_middle")
            try:
                return float(value)
            except (TypeError, ValueError):
                return np.nan
        return np.nan

    @staticmethod
    def _apply_mapping(detections, mapping):
        for cluster_id, team_side in mapping.items():
            detections.loc[detections.team_cluster == cluster_id, "team"] = team_side

    def _legacy_cluster_mapping(self, detections):
        team_a = detections[detections.team_cluster == 0]
        team_b = detections[detections.team_cluster == 1]
        xa_coordinates = np.asarray([self._pitch_x(bbox) for bbox in team_a.bbox_pitch], dtype=float)
        xb_coordinates = np.asarray([self._pitch_x(bbox) for bbox in team_b.bbox_pitch], dtype=float)

        finite_a = xa_coordinates[np.isfinite(xa_coordinates)]
        finite_b = xb_coordinates[np.isfinite(xb_coordinates)]
        avg_a = float(np.mean(finite_a)) if len(finite_a) else np.nan
        avg_b = float(np.mean(finite_b)) if len(finite_b) else np.nan
        if not np.isfinite(avg_a) or not np.isfinite(avg_b):
            log.warning(
                "Cannot infer team side from pitch positions: cluster0=%s cluster1=%s",
                avg_a,
                avg_b,
            )
            return {}

        if avg_a > avg_b:
            return {0: "right", 1: "left"}
        return {0: "left", 1: "right"}

    def _window_candidates(self, detections, window_size):
        if window_size <= 0 or "image_id" not in detections.columns:
            return [("all", detections)]
        image_ids = np.asarray(sorted(pd.unique(detections.image_id.dropna())), dtype=float)
        if len(image_ids) <= window_size:
            return [("all", detections)]

        step = max(1, int(window_size // 2))
        starts = list(range(0, max(1, len(image_ids) - window_size + 1), step))
        final_start = max(0, len(image_ids) - window_size)
        if starts[-1] != final_start:
            starts.append(final_start)

        windows = []
        for start in starts:
            left = image_ids[start]
            right = image_ids[min(start + window_size - 1, len(image_ids) - 1)]
            window = detections[(detections.image_id >= left) & (detections.image_id <= right)]
            windows.append((f"{int(left)}..{int(right)}", window))
        return windows

    def _track_median_stats(self, window):
        track_rows = (
            window.groupby(["team_cluster", "track_id"], dropna=True)["_pitch_x"]
            .median()
            .reset_index()
        )
        cluster_counts = track_rows.groupby("team_cluster")["track_id"].nunique()
        if any(cluster_counts.get(cluster_id, 0) < self.min_tracks_per_team for cluster_id in (0, 1)):
            return None

        cluster_medians = track_rows.groupby("team_cluster")["_pitch_x"].median()
        x0 = float(cluster_medians.get(0, np.nan))
        x1 = float(cluster_medians.get(1, np.nan))
        if not np.isfinite(x0) or not np.isfinite(x1):
            return None
        return x0, x1, abs(x0 - x1), cluster_counts.to_dict()

    def _mapping_from_track_medians(self, detections):
        player_detections = detections[
            (detections.role == "player")
            & (detections.team_cluster.isin([0, 1]))
            & detections.bbox_pitch.apply(lambda bbox: isinstance(bbox, dict))
            & detections.track_id.notna()
        ].copy()
        if player_detections.empty:
            log.warning("Global team-side mapping has no player detections with bbox_pitch; falling back.")
            return None

        player_detections["_pitch_x"] = player_detections.bbox_pitch.apply(self._pitch_x)
        player_detections = player_detections[np.isfinite(player_detections["_pitch_x"])]
        if player_detections.empty:
            log.warning("Global team-side mapping has no finite pitch x values; falling back.")
            return None

        for window_size in self.window_sizes:
            best = None
            for window_label, window in self._window_candidates(player_detections, window_size):
                if window.empty:
                    continue
                stats = self._track_median_stats(window)
                if stats is None:
                    continue
                x0, x1, gap, cluster_counts = stats
                if best is None or gap > best["gap"]:
                    best = {
                        "window_label": window_label,
                        "x0": x0,
                        "x1": x1,
                        "gap": gap,
                        "cluster_counts": cluster_counts,
                    }

            if best is None:
                continue

            x0 = best["x0"]
            x1 = best["x1"]
            gap = best["gap"]
            if gap < self.min_track_x_gap:
                log.info(
                    "Team-side global mapping window=%s undecided: cluster0_x=%.3f cluster1_x=%.3f gap=%.3f < %.3f",
                    best["window_label"],
                    x0,
                    x1,
                    gap,
                    self.min_track_x_gap,
                )
                continue

            if x0 < x1:
                mapping = {0: "left", 1: "right"}
            else:
                mapping = {0: "right", 1: "left"}
            log.info(
                "Team-side global mapping locked with window=%s: cluster0_x=%.3f cluster1_x=%.3f gap=%.3f tracks=%s mapping=%s",
                best["window_label"],
                x0,
                x1,
                gap,
                best["cluster_counts"],
                mapping,
            )
            return mapping

        log.warning(
            "Global team-side mapping stayed undecided after windows=%s; falling back to legacy mean-position mapping.",
            self.window_sizes,
        )
        return None
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        if "track_id" not in detections.columns:
            return detections
        if "team" not in detections.columns:
            detections["team"] = pd.Series([None] * len(detections), index=detections.index, dtype=object)
        else:
            detections["team"] = detections["team"].astype(object)

        if self.use_global_pitch_mapping:
            mapping = self._mapping_from_track_medians(detections)
            if mapping is None:
                mapping = self._legacy_cluster_mapping(detections)
        else:
            mapping = self._legacy_cluster_mapping(detections)

        self._apply_mapping(detections, mapping)

        # Goalkeeper labeling
        goalkeepers = detections[detections.role == "goalkeeper"]
        gk_x = pd.to_numeric(goalkeepers.bbox_pitch.apply(self._pitch_x), errors="coerce")
        valid_gk = gk_x[np.isfinite(gk_x)]
        if len(valid_gk):
            detections.loc[valid_gk.index, "team"] = valid_gk.apply(
                lambda x: "right" if x > 0 else "left"
            )

        return detections
