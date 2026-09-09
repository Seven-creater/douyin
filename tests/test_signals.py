"""signals 单测：指标纯函数 + 类型化事件候选推导（合成序列，零 cv2/ffmpeg）。"""
from __future__ import annotations

import numpy as np

from src.library.signals import (compute_series, derive_event_candidates,
                                 frame_stats, pair_grid_diff)


def _series(n_frames: int = 30, *, luma=None, diff=None, grid=None,
            flow_mag=None, flow_topbin=None, flow_radial=None) -> dict:
    """合成指标序列：未给的字段用平稳默认值（时间轴 0.1s/帧）。"""
    n = n_frames - 1
    return {
        "t": [round(i * 0.1, 3) for i in range(n_frames)],
        "luma_mean": luma if luma is not None else [100.0] * n_frames,
        "luma_std": [20.0] * n_frames,
        "diff_global": diff if diff is not None else [1.0] * n,
        "grid": grid if grid is not None else [[2.0] * 36 for _ in range(n)],
        "flow_mag": flow_mag if flow_mag is not None else [0.5] * n,
        "flow_topbin": flow_topbin if flow_topbin is not None else [0.3] * n,
        "flow_radial": flow_radial if flow_radial is not None else [0.0] * n,
    }


def test_frame_stats_and_grid_diff():
    a = np.full((12, 12), 100, np.uint8)
    b = a.copy()
    b[:2, :2] = 250                                     # 左上角 4 像素变亮
    m, s = frame_stats(a)
    assert m == 100.0 and s == 0.0
    d, cells = pair_grid_diff(a, b, rows=3, cols=3)     # 3x3 网格，每格 4x4 像素
    assert d > 0 and len(cells) == 9
    assert cells[0] > 30 and cells[1] < 1               # 只有(0,0)格活跃（均值 37.5）


def test_detect_luma_flash():
    luma = [100.0] * 10 + [190.0, 104.0] + [100.0] * 10  # t=1.0 一帧闪白回落
    cands = derive_event_candidates(_series(luma=luma))
    flashes = [c for c in cands if "luma_flash" in c["type_hypotheses"]]
    assert len(flashes) == 1 and abs(flashes[0]["t_s"] - 1.0) < 0.15


def test_detect_hard_cut_confidence_uses_scene_bounds():
    diff = [1.0] * 5 + [50.0] + [1.0] * 23             # t=0.6 切点
    plain = derive_event_candidates(_series(diff=diff))
    cuts = [c for c in plain if "hard_cut" in c["type_hypotheses"]]
    assert len(cuts) == 1 and cuts[0]["confidence"] == 0.55   # 不在场景边界
    anchored = derive_event_candidates(_series(diff=diff), shot_boundaries=[0.6])
    cuts2 = [c for c in anchored if "hard_cut" in c["type_hypotheses"]]
    assert cuts2[0]["confidence"] == 0.85                     # 与 scene 边界重合


def test_detect_region_change_mask_family():
    grid = [[2.0] * 36 for _ in range(29)]
    active = [2.0] * 36
    for k in (7, 8, 9, 13, 14, 15, 19, 20):             # 8 个连续区域格活跃
        active[k] = 40.0
    grid[10] = active                                   # t=1.1 局部剧变但全局低
    cands = derive_event_candidates(_series(grid=grid))
    region = [c for c in cands if "tracked_mask_fill" in c["type_hypotheses"]]
    assert len(region) == 1 and region[0]["signature"]["cells_active"] == 8


def test_detect_whip_pan_and_freeze():
    flow = [0.5] * 5 + [5.0, 6.5, 5.5, 4.0] + [0.5] * 20  # t=0.6 附近甩镜（29 边界）
    topbin = [0.3] * 29
    for k in range(5, 9):
        topbin[k] = 0.8
    cands = derive_event_candidates(_series(flow_mag=flow, flow_topbin=topbin))
    assert any("whip_pan" in c["type_hypotheses"] for c in cands)
    # 定格：光流与差分同时近零 ≥3 边界
    flat = derive_event_candidates(_series(flow_mag=[0.05] * 29, diff=[0.5] * 29))
    assert any("beat_freeze" in c["type_hypotheses"] for c in flat)


def test_detect_montage_burst_groups_cuts():
    diff = [1.0] * 29
    for k in (5, 8, 11):                                # 0.6/0.9/1.2s 三连切
        diff[k] = 45.0
    cands = derive_event_candidates(_series(diff=diff))
    burst = [c for c in cands if c["type_hypotheses"] == ["montage_burst"]]
    assert len(burst) == 1 and burst[0]["signature"]["n_cuts"] == 3


def test_max_events_cap_keeps_type_diversity():
    diff = [1.0] * 59
    for k in range(5, 55, 3):                           # 17 个切点
        diff[k] = 45.0
    series = _series(60, diff=diff)
    cands = derive_event_candidates(series, thr={"max_events": 3})
    assert len(cands) == 3
    types = {h for c in cands for h in c["type_hypotheses"]}
    assert {"hard_cut", "montage_burst"} <= types       # 低置信类型不被挤光


def test_compute_series_shapes():
    frames = [np.full((12, 12), v, np.uint8) for v in (100, 130, 100)]
    s = compute_series(frames, grid_rows=3, grid_cols=3)
    assert len(s["luma_mean"]) == 3 and len(s["diff_global"]) == 2
    assert len(s["grid"][0]) == 9
    assert s["luma_mean"] == [100.0, 130.0, 100.0]
