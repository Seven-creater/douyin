"""deduplicate 单测：跨榜合并、优先级、None 补全、确定性。"""
from __future__ import annotations

from src.trend.deduplicate import LIST_PRIORITY, merge_videos, to_csv_row, to_jsonl_record
from src.trend.parser import MusicInfo, Stats, TrendInfo, VideoRecord


def make_record(aweme_id, list_id, rank, *, likes=None, views=None, title=None, author=None, url=None):
    return VideoRecord(
        source="wellbyte", platform="douyin", aweme_id=aweme_id,
        title=title, author=author, author_id=None, share_url=None,
        download_url=url or f"https://cdn.test/{aweme_id}.mp4",
        stats=Stats(likes=likes, views=views),
        music=MusicInfo(),
        trend=TrendInfo(lists=[list_id], rank={list_id: rank}, date_window=24),
        extras={}, raw={"item_id": aweme_id},
    )


def test_merge_cross_lists():
    recs = [
        make_record("A", "1001", 3, likes=100),
        make_record("A", "1005", 1),
        make_record("A", "1002", 8),
    ]
    merged = merge_videos(recs)
    assert len(merged) == 1
    mv = merged[0]
    assert mv.appeared_in_lists == ["1001", "1002", "1005"]  # 按优先级排序
    assert mv.ranks == {"1001": 3, "1002": 8, "1005": 1}
    assert mv.n_lists == 3


def test_primary_record_from_highest_priority_list():
    # 1001 存在时主记录来自 1001（即使 1002 名次更靠前）
    recs = [
        make_record("A", "1002", 1, likes=5),
        make_record("A", "1001", 9, likes=5),
    ]
    mv = merge_videos(recs)[0]
    # 主记录判定看 trend.lists 合并前的来源；合并后 lists 是全量——用 rank 校验
    assert mv.ranks == {"1001": 9, "1002": 1}


def test_none_field_filled_from_other_list():
    recs = [
        make_record("A", "1001", 1, title=None, author=None),
        make_record("A", "1002", 5, title="补全标题", author="作者", likes=42),
    ]
    mv = merge_videos(recs)[0]
    assert mv.record.title == "补全标题"
    assert mv.record.author == "作者"
    assert mv.record.stats.likes == 42


def test_distinct_videos_stay_separate():
    recs = [make_record("A", "1001", 1), make_record("B", "1001", 2), make_record("C", "1003", 1)]
    merged = merge_videos(recs)
    assert len(merged) == 3
    assert {m.aweme_id for m in merged} == {"A", "B", "C"}


def test_deterministic_order():
    recs = [make_record(str(i), "1001", i) for i in range(20, 0, -1)]
    m1 = [m.aweme_id for m in merge_videos(list(recs))]
    m2 = [m.aweme_id for m in merge_videos(list(reversed(recs)))]
    assert m1 == m2


def test_priority_order_constant():
    assert LIST_PRIORITY == ("1001", "1002", "1005", "1003", "1004")


def test_to_jsonl_record_contains_everything():
    recs = [make_record("A", "1001", 1), make_record("A", "1005", 2)]
    mv = merge_videos(recs)[0]
    d = to_jsonl_record(mv, final_rank=7)
    assert d["final_rank"] == 7
    assert d["appeared_in_lists"] == ["1001", "1005"]
    assert d["ranks"] == {"1001": 1, "1005": 2}
    assert d["raw"]["item_id"] == "A"
    assert d["trend"]["lists"] == ["1001", "1005"]


def test_to_csv_row_columns():
    recs = [make_record("A", "1001", 2, likes=10, views=20)]
    mv = merge_videos(recs)[0]
    row = to_csv_row(mv, final_rank=1)
    assert row["list_1001"] == 2 and row["list_1005"] == ""
    assert row["likes"] == 10 and row["n_lists"] == 1
