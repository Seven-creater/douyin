"""多榜单视频按 aweme_id 去重合并。

同一视频可能同时出现在总榜 + 高点赞榜 + 低粉爆款榜；合并后保留：
- appeared_in_lists：上榜榜单号（升序字符串）
- ranks：各榜单名次 {sub_type: rank}
- 主记录：来自最高优先级榜单；其 None 字段用其他榜单记录补全
- raw：主记录的完整原始数据（各榜 raw 相同，因为字段字典完全一致）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.trend.parser import VideoRecord

# 主记录选取与 None 补全优先级：总榜 > 低粉爆款 > 高点赞 > 高完播 > 高涨粉
LIST_PRIORITY: tuple[str, ...] = ("1001", "1002", "1005", "1003", "1004")


@dataclass
class MergedVideo:
    record: VideoRecord                 # 主记录（来自最高优先级榜单）
    appeared_in_lists: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def aweme_id(self) -> str:
        return self.record.aweme_id

    @property
    def n_lists(self) -> int:
        return len(self.appeared_in_lists)


def _priority(list_id: str) -> int:
    try:
        return LIST_PRIORITY.index(list_id)
    except ValueError:
        return len(LIST_PRIORITY)  # 未知榜单排最后


def merge_videos(records: list[VideoRecord]) -> list[MergedVideo]:
    """按 aweme_id 合并。输入顺序无关；输出按主榜单优先级稳定排列。"""
    by_id: dict[str, list[VideoRecord]] = {}
    for rec in records:
        by_id.setdefault(rec.aweme_id, []).append(rec)

    merged: list[MergedVideo] = []
    for aweme_id, recs in by_id.items():
        # 按（榜单优先级, 榜内名次）排序选主记录
        recs.sort(key=lambda r: (
            _priority(r.trend.lists[0]) if r.trend.lists else len(LIST_PRIORITY),
            min(r.trend.rank.values()) if r.trend.rank else 10**9,
        ))
        primary = recs[0]

        appeared = sorted({lst for r in recs for lst in r.trend.lists}, key=_priority)
        ranks: dict[str, int] = {}
        for r in recs:
            for lst, rk in r.trend.rank.items():
                ranks.setdefault(lst, rk)

        # 主记录 None 字段用其他榜单补全（真实数据各榜字段一致，此为防御性逻辑）
        for other in recs[1:]:
            if primary.title is None and other.title is not None:
                primary.title = other.title
            if primary.author is None and other.author is not None:
                primary.author = other.author
            if primary.download_url is None and other.download_url is not None:
                primary.download_url = other.download_url
            if primary.stats.likes is None and other.stats.likes is not None:
                primary.stats.likes = other.stats.likes
            if primary.stats.views is None and other.stats.views is not None:
                primary.stats.views = other.stats.views

        # trend 汇总为主记录上的全量信息（写 jsonl 时用）
        primary.trend.lists = appeared
        primary.trend.rank = ranks

        merged.append(MergedVideo(record=primary, appeared_in_lists=appeared, ranks=ranks))

    merged.sort(key=lambda m: m.aweme_id)  # 确定性输出顺序
    return merged


def to_jsonl_record(mv: MergedVideo, final_rank: int | None = None) -> dict:
    d = mv.record.to_dict()
    d["appeared_in_lists"] = mv.appeared_in_lists
    d["ranks"] = mv.ranks
    if final_rank is not None:
        d["final_rank"] = final_rank
    return d


CSV_COLUMNS = [
    "final_rank", "aweme_id", "title", "author", "n_lists",
    "list_1001", "list_1002", "list_1003", "list_1004", "list_1005",
    "likes", "views", "duration_ms", "media_type", "publish_time", "download_url",
]


def to_csv_row(mv: MergedVideo, final_rank: int | None = None) -> dict:
    """不含 raw（人工检查用，Excel 可读）。utf-8-sig 写盘。"""
    ex = mv.record.extras
    return {
        "final_rank": final_rank if final_rank is not None else "",
        "aweme_id": mv.record.aweme_id,
        "title": mv.record.title or "",
        "author": mv.record.author or "",
        "n_lists": mv.n_lists,
        "list_1001": mv.ranks.get("1001", ""),
        "list_1002": mv.ranks.get("1002", ""),
        "list_1003": mv.ranks.get("1003", ""),
        "list_1004": mv.ranks.get("1004", ""),
        "list_1005": mv.ranks.get("1005", ""),
        "likes": mv.record.stats.likes if mv.record.stats.likes is not None else "",
        "views": mv.record.stats.views if mv.record.stats.views is not None else "",
        "duration_ms": ex.get("duration_ms", ""),
        "media_type": ex.get("media_type", ""),
        "publish_time": ex.get("publish_time", ""),
        "download_url": mv.record.download_url or "",
    }


def write_outputs(
    merged_ranked: list[MergedVideo],
    processed_dir: Path,
    *,
    jsonl_name: str = "trending_videos.jsonl",
    csv_name: str = "trending_videos.csv",
) -> tuple[Path, Path]:
    """全量（含 final_rank，仅排序后的 TopN 有值）写 jsonl + csv。"""
    import csv
    import json

    processed_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = processed_dir / jsonl_name
    csv_path = processed_dir / csv_name

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, mv in enumerate(merged_ranked, start=1):
            f.write(json.dumps(to_jsonl_record(mv, final_rank=i), ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, mv in enumerate(merged_ranked, start=1):
            writer.writerow(to_csv_row(mv, final_rank=i))
    return jsonl_path, csv_path
