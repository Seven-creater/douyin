"""ranking 单测：spec 四级优先规则逐级验证 + 确定性。"""
from __future__ import annotations

from src.trend.deduplicate import merge_videos
from src.trend.ranking import rank_videos, sort_key
from tests.test_deduplicate import make_record


def merged(*recs):
    return merge_videos(list(recs))


def test_multi_list_beats_single_list():
    # 单榜第 1 名 < 双榜（哪怕双榜名次都靠后）
    m = merged(
        make_record("SOLO", "1001", 1, likes=10**9),
        make_record("DUO", "1003", 15),
        make_record("DUO", "1004", 15),
    )
    ranked = rank_videos(m, top_n=2)
    assert ranked[0].merged.aweme_id == "DUO"


def test_same_list_count_total_rank_first():
    m = merged(
        make_record("HIGH", "1001", 2),
        make_record("LOW", "1001", 10),
    )
    ranked = rank_videos(m, top_n=2)
    assert [r.merged.aweme_id for r in ranked] == ["HIGH", "LOW"]


def test_low_fan_viral_then_high_likes_priority():
    # 都只上 1002/1005 之一时：1002 名次优先于 1005
    m = merged(
        make_record("LF", "1002", 1),   # 低粉爆款第1
        make_record("HL", "1005", 1),   # 高点赞第1
    )
    ranked = rank_videos(m, top_n=2)
    assert ranked[0].merged.aweme_id == "LF"


def test_likes_tiebreak():
    m = merged(
        make_record("MORE", "1001", 5, likes=999),
        make_record("LESS", "1001", 5, likes=111),
    )
    ranked = rank_videos(m, top_n=2)
    assert ranked[0].merged.aweme_id == "MORE"


def test_missing_likes_treated_as_minus_one():
    m = merged(make_record("NOLIKES", "1001", 5))  # likes=None
    assert sort_key(m[0])[4] == -1


def test_top_n_truncation_and_final_rank():
    recs = [make_record(f"id{i:02d}", "1001", i + 1) for i in range(30)]
    ranked = rank_videos(merged(*recs), top_n=20)
    assert len(ranked) == 20
    assert [r.final_rank for r in ranked] == list(range(1, 21))


def test_deterministic_same_input_same_order():
    recs = [make_record(f"id{i}", "1001", (i % 15) + 1, likes=1000 - i) for i in range(25)]
    m = merged(*recs)
    r1 = [r.merged.aweme_id for r in rank_videos(m, top_n=25)]
    r2 = [r.merged.aweme_id for r in rank_videos(m, top_n=25)]
    assert r1 == r2
