import pandas as pd
import numpy as np
import logging
from collections import Counter
from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)

# ROBUST per-video two-color split. Replaces the old hardcoded COLOR_TO_BINARY map
# (which assumed each video had exactly one {blue,yellow}=0 color and one
# {white,green,stripes}=1 color -- this broke for white-vs-green or white-vs-red videos
# where both dominant colors mapped to the same binary). Instead, for EACH video we pick
# the two most frequent player color hints and assign them to clusters 0 and 1 dynamically.
# TrackletTeamSideLabeling downstream converts {0,1} -> left/right by average pitch x.


class TrackletTeamColorPrompt(VideoLevelModule):
    """
    Team assignment for role="player" derived from the SAM3 detection-time text-prompt color
    (injected as 'team_color_hint'), instead of KMeans on reid embeddings. Majority-votes the
    hint per FINAL track_id, then dynamically maps the two dominant per-video colors to
    cluster {0,1}. Robust to any color pairing (white/red, white/green, blue/stripes, ...).
    Same output contract (team_cluster in {0,1}) as TrackletTeamClustering.
    """
    input_columns = ["track_id", "team_color_hint", "role"]
    output_columns = ["team_cluster"]

    def __init__(self, **kwargs):
        super().__init__()

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections['team_cluster'] = np.nan

        if "track_id" not in detections.columns or "team_color_hint" not in detections.columns:
            log.warning("TrackletTeamColorPrompt: missing track_id or team_color_hint column, "
                        "all team_cluster left as NaN")
            return detections

        player_detections = detections[detections.role == "player"]

        # 1) per-track majority color + track size (weight)
        track_color = {}
        track_size = {}
        for track_id in player_detections.track_id.unique():
            if pd.isna(track_id):
                continue
            tracklet = player_detections[player_detections.track_id == track_id]
            hints = tracklet['team_color_hint'].dropna()
            if len(hints) == 0:
                continue
            track_color[track_id] = hints.mode()[0]
            track_size[track_id] = len(tracklet)

        if not track_color:
            return detections

        # 2) two dominant colors in THIS video, weighted by #detections (robust to noise)
        color_weight = Counter()
        for track_id, color in track_color.items():
            color_weight[color] += track_size[track_id]
        top2 = [c for c, _ in color_weight.most_common(2)]
        color_to_bin = {}
        if len(top2) >= 1:
            color_to_bin[top2[0]] = 0
        if len(top2) >= 2:
            color_to_bin[top2[1]] = 1
        log.info(f"TrackletTeamColorPrompt(robust): video color weights={dict(color_weight)} "
                 f"-> top2={top2} mapped to clusters {color_to_bin}")

        # 3) assign cluster per track (rare 3rd+ color -> NaN)
        for track_id, color in track_color.items():
            binary = color_to_bin.get(color, np.nan)
            detections.loc[detections.track_id == track_id, 'team_cluster'] = binary

        return detections
