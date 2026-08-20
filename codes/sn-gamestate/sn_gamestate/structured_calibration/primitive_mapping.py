"""point -> line -> primitive -> field 层级映射配置（创新点1的几何骨架）。

这是 Primitive-aware Keypoint Reliability Weighting 的基础：把 nbjw/PnLCalib 的
关键点按足球场标准几何组织成层级。本文件是**纯配置 + 反向索引**，不依赖 nbjw；
3D 模板坐标在需要时由 `attach_world_coords()` 从 nbjw 加载附上。

约定：keypoint id 1..57 = 主点（nbjw keypoint_world_coords_2D[id-1]），
58..73 = 辅助点（keypoint_aux_world_coords_2D[id-58]）。
elevated 点 {12,15,16,19} 的世界 z = -2.44m（横梁/柱顶，非地面平面）。
"""
from __future__ import annotations

# ── 语义线 -> 关键点 id（取自 nbjw_calib.py 的 kp_to_line）──────────────────
KP_TO_LINE = {
    "Big rect. left bottom":  [24, 68, 25],
    "Big rect. left main":    [5, 64, 31, 46, 34, 66, 25],
    "Big rect. left top":     [4, 62, 5],
    "Big rect. right bottom": [26, 69, 27],
    "Big rect. right main":   [6, 65, 33, 56, 36, 67, 26],
    "Big rect. right top":    [6, 63, 7],
    "Circle central":         [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
    "Circle left":            [31, 37, 47, 41, 34],
    "Circle right":           [33, 40, 55, 44, 36],
    "Goal left crossbar":     [16, 12],
    "Goal left post left":    [16, 17],
    "Goal left post right":   [12, 13],
    "Goal right crossbar":    [15, 19],
    "Goal right post left":   [15, 14],
    "Goal right post right":  [19, 18],
    "Middle line":            [2, 32, 51, 35, 29],
    "Side line bottom":       [28, 70, 71, 29, 72, 73, 30],
    "Side line left":         [1, 4, 8, 13, 17, 20, 24, 28],
    "Side line right":        [3, 7, 11, 14, 18, 23, 27, 30],
    "Side line top":          [1, 58, 59, 2, 60, 61, 3],
    "Small rect. left bottom":  [20, 21],
    "Small rect. left main":    [9, 21],
    "Small rect. left top":     [8, 9],
    "Small rect. right bottom": [22, 23],
    "Small rect. right main":   [10, 22],
    "Small rect. right top":    [10, 11],
}

# ── 语义线 -> primitive（一条线可属多个 primitive）──────────────────────────
LINE_TO_PRIMITIVE = {
    "Circle central": ["center_circle"],
    "Circle left":    ["center_circle"],
    "Circle right":   ["center_circle"],
    "Middle line":    ["halfway_line"],
    "Big rect. left bottom": ["left_penalty_box"],
    "Big rect. left main":   ["left_penalty_box"],
    "Big rect. left top":    ["left_penalty_box"],
    "Big rect. right bottom": ["right_penalty_box"],
    "Big rect. right main":   ["right_penalty_box"],
    "Big rect. right top":    ["right_penalty_box"],
    "Small rect. left bottom": ["left_goal_area"],
    "Small rect. left main":   ["left_goal_area"],
    "Small rect. left top":    ["left_goal_area"],
    "Small rect. right bottom": ["right_goal_area"],
    "Small rect. right main":   ["right_goal_area"],
    "Small rect. right top":    ["right_goal_area"],
    "Goal left crossbar":   ["left_goal_frame"],
    "Goal left post left":  ["left_goal_frame"],
    "Goal left post right": ["left_goal_frame"],
    "Goal right crossbar":   ["right_goal_frame"],
    "Goal right post left":  ["right_goal_frame"],
    "Goal right post right": ["right_goal_frame"],
    "Side line bottom": ["field_boundary"],
    "Side line left":   ["field_boundary"],
    "Side line right":  ["field_boundary"],
    "Side line top":    ["field_boundary"],
}

PRIMITIVES = [
    "center_circle", "halfway_line",
    "left_penalty_box", "right_penalty_box",
    "left_goal_area", "right_goal_area",
    "left_goal_frame", "right_goal_frame",
    "field_boundary",
]

ELEVATED_IDS = {12, 15, 16, 19}   # world z = -2.44 m (crossbar / post tops)
GOAL_IDS = {12, 13, 14, 15, 16, 17, 18, 19}

# point-only features that are not on any line (penalty marks) -> primitive membership
# id45 = left penalty mark (-41.5, 0); id57 = right penalty mark (41.5, 0)
POINT_ONLY_PRIMITIVE = {45: "left_penalty_box", 57: "right_penalty_box"}


# ── 反向索引 ────────────────────────────────────────────────────────────────
def _build_reverse():
    kp_to_lines, kp_to_prims = {}, {}
    prim_to_kps = {p: set() for p in PRIMITIVES}
    prim_to_lines = {p: set() for p in PRIMITIVES}
    for line, ids in KP_TO_LINE.items():
        prims = LINE_TO_PRIMITIVE[line]
        for p in prims:
            prim_to_lines[p].add(line)
        for kid in ids:
            kp_to_lines.setdefault(kid, set()).add(line)
            for p in prims:
                kp_to_prims.setdefault(kid, set()).add(p)
                prim_to_kps[p].add(kid)
    # point-only features (no line membership) -> primitive only
    for kid, p in POINT_ONLY_PRIMITIVE.items():
        kp_to_prims.setdefault(kid, set()).add(p)
        prim_to_kps[p].add(kid)
    return kp_to_lines, kp_to_prims, prim_to_kps, prim_to_lines


KP_TO_LINES, KP_TO_PRIMS, PRIM_TO_KPS, PRIM_TO_LINES = _build_reverse()


def line_orientation(line, world):
    """'x'(沿球场长边/底线方向) or 'y'(沿宽边/中线方向) or 'mixed'.
    world: {id:(xw,yw)}. 用该线上点的坐标极差判断主方向。"""
    pts = [world[i] for i in KP_TO_LINE[line] if i in world]
    if len(pts) < 2:
        return "unknown"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    if dx > 3 * max(dy, 1e-6):
        return "x"
    if dy > 3 * max(dx, 1e-6):
        return "y"
    return "mixed"


def attach_world_coords(W, A):
    """W=keypoint_world_coords_2D(57), A=keypoint_aux_world_coords_2D(16).
    Returns {id:(xw,yw,zw)} for every id present in KP_TO_LINE."""
    world = {}
    allids = set(i for ids in KP_TO_LINE.values() for i in ids) | set(POINT_ONLY_PRIMITIVE)
    for kid in allids:
        if kid <= 57:
            xw, yw = W[kid - 1]
        else:
            xw, yw = A[kid - 1 - 57]
        zw = -2.44 if kid in ELEVATED_IDS else 0.0
        world[kid] = (float(xw), float(yw), float(zw))
    return world
