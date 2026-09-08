"""beat_cut 单测：切点计划 / 半拍细分 / 镜头挑选去重（纯函数，零模型）。"""
from __future__ import annotations

import numpy as np

from src.library.beat_cut import build_cut_plan, pick_shots


def test_build_cut_plan_covers_full_duration():
    plan = build_cut_plan([1.0, 2.0, 3.0], 4.2)
    spans = [(round(a, 2), round(b, 2)) for a, b in plan]
    assert spans[0][0] == 0.0 and abs(plan[-1][1] - 4.2) < 1e-6
    assert all(b - a >= 0.3 for a, b in plan)


def test_build_cut_plan_half_beat_subdivides():
    plan = build_cut_plan([1.0, 2.0], 2.0, half_beat=True)
    bounds = [a for a, _ in plan] + [plan[-1][1]]
    assert 1.5 in [round(b, 2) for b in bounds]        # 拍间中点出现


def test_build_cut_plan_caps_and_thins():
    beats = [i * 0.47 for i in range(1, 100)]           # ~46s 高密度
    plan = build_cut_plan(beats, 47.0, max_cuts=20)
    assert len(plan) == 20
    assert all(b > a for a, b in plan)


def test_pick_shots_dedupe_and_duration():
    rows = [{"duration_s": d} for d in (3.0, 0.2, 2.0, 3.0)]
    emb = np.array([[1, 0], [0.9, 0.1], [0.1, 0.9], [0.8, 0.2]], "float32")
    q = np.array([1, 0], "float32")
    picked = pick_shots(rows, emb, [q, q], [0.5, 0.5], used=set(), min_len_s=0.45)
    assert picked[0] == 0                                # 最相似且时长够
    assert picked[1] == 3                                # 0 已用，跳过 1（太短）取 3


def test_pick_shots_fallback_when_all_used():
    rows = [{"duration_s": 1.0}, {"duration_s": 1.0}]
    emb = np.array([[1, 0], [0.9, 0.1]], "float32")
    q = np.array([1, 0], "float32")
    picked = pick_shots(rows, emb, [q, q, q], [0.5] * 3, used=set(), min_len_s=0.45)
    assert picked[:2] == [0, 1] and picked[2] is None    # 用尽 → None 占位
