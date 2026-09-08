"""critic 单测：问题清单解析归一 / 动作映射（换策略阈值+定位切序号）。"""
from __future__ import annotations

import numpy as np

from src.library.critic import ASPECTS, derive_actions, parse_critique


def test_parse_critique_normalizes_bad_aspect():
    raw = ('{"score": 4, "verdict": "差", "issues": [{"aspect": "画面好怪", "t_s": 3.0, '
           '"severity": "high", "note": "x", "suggest": "y"}], "good_points": ["节奏好"]}')
    c = parse_critique(raw)
    assert c["score"] == 4
    assert c["issues"][0]["aspect"] == "quality"          # 越界分类归一不拒收
    assert all(i["aspect"] in ASPECTS for i in c["issues"])


def test_parse_critique_rejects_garbage():
    assert parse_critique("这不是JSON") is None
    assert parse_critique('{"no_issues_key": 1}') is None


def test_derive_actions_switches_crop_on_high_or_multi():
    plan = [(0, 1), (1, 2), (2, 3)]
    hi = {"issues": [{"aspect": "aspect", "t_s": 0.5, "severity": "high"}]}
    a = derive_actions(hi, plan, "center")
    assert a["switch_crop"] == "blurpad"
    multi = {"issues": [{"aspect": "crop_weird", "t_s": 0.5, "severity": "low"},
                        {"aspect": "crop_weird", "t_s": 2.5, "severity": "low"}]}
    assert derive_actions(multi, plan, "center")["switch_crop"] == "blurpad"
    assert derive_actions(multi, plan, "blurpad")["switch_crop"] == "center"
    single_low = {"issues": [{"aspect": "crop_weird", "t_s": 0.5, "severity": "low"}]}
    assert derive_actions(single_low, plan, "center")["switch_crop"] is None


def test_derive_actions_locates_relevance_swaps():
    plan = [(0, 1), (1, 2.5), (2.5, 4)]
    crit = {"issues": [{"aspect": "relevance", "t_s": 2.9, "severity": "high"},
                       {"aspect": "relevance", "t_s": 0.2, "severity": "medium"},
                       {"aspect": "rhythm", "t_s": 3.0, "severity": "high"}]}
    a = derive_actions(crit, plan, "center")
    assert a["reswap_cut_indices"] == [0, 2]              # 0.2→切0，2.9→切2；rhythm 不算


def test_apply_swaps_picks_unused_row(tmp_path):
    from src.library.critic import apply_swaps
    rows = [{"duration_s": 1.0}, {"duration_s": 1.0}]
    emb = np.array([[0.9, 0.1], [1.0, 0.0]], "float32")

    class FakeEmb:
        def embed(self, texts):
            return np.array([[1.0, 0.0]] * len(texts), "float32")

    issues = [{"aspect": "relevance", "t_s": 0.5, "severity": "high"}]
    swaps = apply_swaps(rows, emb, FakeEmb(), [(0, 1.0)], issues, used=set())
    assert swaps == {0: 1}                                 # row0 已被原版用过？不——used 为空则取最相似 row1? 排序:cos[1]=1.0>row0
    swaps2 = apply_swaps(rows, emb, FakeEmb(), [(0, 1.0)], issues, used={1})
    assert swaps2 == {0: 0}
