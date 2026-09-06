"""Wellbyte 抖音榜单 raw JSON → 统一 VideoRecord。

═══ 实测字段字典（2026-09-06，5 榜单 80 条真实响应，各榜字段完全一致）═══
raw 字段             → 目标字段                    说明
─────────────────────────────────────────────────────────────────
item_id             → aweme_id                    str，稳定作品 ID，去重主键
item_title          → title                       可为空串
nick_name           → author
avatar_url          → extras.avatar_url
item_url            → download_url                视频=CDN mp4 直链（带签名，有时效）；
                                                  图集(media_type=2)=douyinstatic 静态对象
item_cover_url      → extras.cover_url            封面图直链（带签名）
like_cnt            → stats.likes
play_cnt            → stats.views
item_duration       → extras.duration_ms          毫秒；图集=0
media_type          → extras.media_type           实测 4=视频，2=图集
image_cnt           → extras.image_cnt            >0 即图集
publish_time        → extras.publish_time         epoch 秒
score               → extras.score                榜单热度分（如 1432514）
fans_cnt            → extras.fans_cnt
follow_cnt          → extras.follow_cnt
like_rate           → extras.like_rate
follow_rate         → extras.follow_rate
favorite_id / is_favorite → 仅保留于 raw           与下载无关

响应中不存在（目标 schema 置 null，绝不编造）：
    author_id / share_url / stats.comments / stats.shares / music.*（全部字段）

rank 说明：榜单响应没有显式排名字段；objs 数组顺序即榜单顺序，
rank = index + 1；score 为热度参考分（保留在 extras）。

CLI 人工核对：python -m src.trend.parser data/raw/wellbyte/2026-09-06_1001.json
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Stats:
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    views: int | None = None


@dataclass
class MusicInfo:
    """榜单响应不含任何音乐信息，全部字段恒为 None（BGM 是 Phase 2 独立课题）。"""

    id: str | None = None
    title: str | None = None
    author: str | None = None
    url: str | None = None


@dataclass
class TrendInfo:
    lists: list[str] = field(default_factory=list)
    rank: dict[str, int] = field(default_factory=dict)
    date_window: int = 24


@dataclass
class VideoRecord:
    source: str
    platform: str
    aweme_id: str
    title: str | None
    author: str | None
    author_id: str | None
    share_url: str | None
    download_url: str | None
    stats: Stats
    music: MusicInfo
    trend: TrendInfo
    extras: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "platform": self.platform,
            "aweme_id": self.aweme_id,
            "title": self.title,
            "author": self.author,
            "author_id": self.author_id,
            "share_url": self.share_url,
            "download_url": self.download_url,
            "stats": {
                "likes": self.stats.likes,
                "comments": self.stats.comments,
                "shares": self.stats.shares,
                "views": self.stats.views,
            },
            "music": {
                "id": self.music.id,
                "title": self.music.title,
                "author": self.music.author,
                "url": self.music.url,
            },
            "trend": {
                "lists": self.trend.lists,
                "rank": self.trend.rank,
                "date_window": self.trend.date_window,
            },
            "extras": self.extras,
            "raw": self.raw,
        }


def _s(v) -> str | None:
    """字符串化；None/空串 → None。"""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _i(v) -> int | None:
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _map_item(item: dict, sub_type: int, rank: int, date_window: int) -> VideoRecord | None:
    """单条 raw obj → VideoRecord。item_id 缺失返回 None（调用方丢弃并告警）。"""
    aweme_id = _s(item.get("item_id"))
    if not aweme_id:
        return None
    return VideoRecord(
        source="wellbyte",
        platform="douyin",
        aweme_id=aweme_id,
        title=_s(item.get("item_title")),
        author=_s(item.get("nick_name")),
        author_id=None,          # 响应无此字段
        share_url=None,          # 响应无分享页 URL，只有 CDN 直链
        download_url=_s(item.get("item_url")),
        stats=Stats(
            likes=_i(item.get("like_cnt")),
            comments=None,       # 响应无此字段
            shares=None,         # 响应无此字段
            views=_i(item.get("play_cnt")),
        ),
        music=MusicInfo(),       # 响应无任何音乐字段
        trend=TrendInfo(
            lists=[str(sub_type)],
            rank={str(sub_type): rank},
            date_window=date_window,
        ),
        extras={
            "duration_ms": _i(item.get("item_duration")),
            "media_type": _i(item.get("media_type")),
            "image_cnt": _i(item.get("image_cnt")),
            "publish_time": _i(item.get("publish_time")),
            "score": _i(item.get("score")),
            "fans_cnt": _i(item.get("fans_cnt")),
            "follow_cnt": _i(item.get("follow_cnt")),
            "like_rate": item.get("like_rate"),
            "follow_rate": item.get("follow_rate"),
            "cover_url": _s(item.get("item_cover_url")),
            "avatar_url": _s(item.get("avatar_url")),
        },
        raw=dict(item),
    )


def parse_payload(payload: dict, *, sub_type: int, date_window: int = 24) -> list[VideoRecord]:
    """解析一份已加载的榜单信封。坏条目（无 item_id）丢弃并告警，不抛异常。"""
    data = payload.get("data")
    objs = data.get("objs") if isinstance(data, dict) else None
    if not isinstance(objs, list):
        logger.warning("[parser %s] data.objs 缺失或非数组，返回空", sub_type)
        return []
    records: list[VideoRecord] = []
    dropped = 0
    for idx, item in enumerate(objs):
        if not isinstance(item, dict):
            dropped += 1
            continue
        rec = _map_item(item, sub_type, rank=idx + 1, date_window=date_window)
        if rec is None:
            dropped += 1
            logger.warning("[parser %s] 第 %d 条缺 item_id，丢弃", sub_type, idx + 1)
            continue
        records.append(rec)
    if dropped:
        logger.warning("[parser %s] 共丢弃 %d 条坏数据", sub_type, dropped)
    return records


def parse_file(path: Path, *, date_window: int = 24) -> list[VideoRecord]:
    """解析一个 raw 文件；sub_type 取自文件名最后一段（YYYY-MM-DD_<sub_type>.json）。"""
    path = Path(path)
    m = re.search(r"_(\d+)\.json$", path.name)
    if not m:
        raise ValueError(f"文件名无法解析出 sub_type: {path.name}")
    sub_type = int(m.group(1))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_payload(payload, sub_type=sub_type, date_window=date_window)


def _main(argv: list[str]) -> int:
    """人工核对入口：python -m src.trend.parser <raw.json> [...]"""
    if not argv:
        print(__doc__.split("CLI")[0])
        print("用法: python -m src.trend.parser <raw.json> [...]")
        return 2
    for raw_path in argv:
        records = parse_file(Path(raw_path))
        print(f"\n=== {raw_path} → {len(records)} 条 ===")
        print(f"{'rank':>4} {'aweme_id':<20} {'mt':>2} {'dur_s':>7} {'likes':>9} {'views':>10}  title")
        for r in records:
            dur = (r.extras.get("duration_ms") or 0) / 1000
            print(
                f"{r.trend.rank.get(r.trend.lists[0], 0):>4} {r.aweme_id:<20} "
                f"{r.extras.get('media_type'):>2} {dur:>7.1f} {(r.stats.likes or 0):>9} "
                f"{(r.stats.views or 0):>10}  {(r.title or '')[:40]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
