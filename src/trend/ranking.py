"""热点排序：简单透明的规则排序，不做 AI 评分（spec 第五节）。

sort_key（全降序，元组字典序，无隐式权重）：
    (上榜榜单数, 总榜名次分, 低粉爆款名次分, 高点赞名次分, likes, aweme_id数值)
    名次分 s(x) = 21 - rank（page_size=20，未上榜=0）
    likes 缺失按 -1；tiebreak 用 aweme_id 数值取负保证确定性
"""
from __future__ import annotations

from dataclasses import dataclass

from src.trend.deduplicate import MergedVideo

_PAGE_SIZE = 20


def _score(rank: int | None) -> int:
    return (_PAGE_SIZE + 1 - rank) if rank else 0


@dataclass
class ScoredVideo:
    merged: MergedVideo
    final_rank: int
    sort_key: tuple


def sort_key(mv: MergedVideo) -> tuple:
    likes = mv.record.stats.likes if mv.record.stats.likes is not None else -1
    try:
        tiebreak = -int(mv.record.aweme_id)
    except ValueError:
        tiebreak = 0
    return (
        mv.n_lists,
        _score(mv.ranks.get("1001")),
        _score(mv.ranks.get("1002")),
        _score(mv.ranks.get("1005")),
        likes,
        tiebreak,
    )


def rank_videos(merged: list[MergedVideo], top_n: int = 20) -> list[ScoredVideo]:
    """排序并取 Top N。返回 final_rank = 1..N。"""
    ordered = sorted(merged, key=sort_key, reverse=True)
    return [ScoredVideo(merged=mv, final_rank=i, sort_key=sort_key(mv)) for i, mv in enumerate(ordered[:top_n], start=1)]
