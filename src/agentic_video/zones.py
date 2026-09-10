"""片头/片尾排除区：长片的 OP/ED 职员表素材不得进入检索池。

2026-09-10 叙事库首跑成片帧级审核（用户复核确认）：60 秒成片全部来自
w034 窗口（电影 9060-9105s，总长 9295.8s），即片尾 ED 职员表段——选窗与
检索此前都没有"片头片尾"概念。本模块把排除规则集中一处，供
narrative_index 选窗、story_planner 叙事检索、planner 编辑检索共用：
长片（≥ min_movie_s）的片头 head_s 秒与片尾 tail_s 秒内的镜头一律剔除。
"""
from __future__ import annotations

import json
from pathlib import Path

ZONE_DEFAULTS = {"head_s": 90.0, "tail_s": 360.0, "min_movie_s": 1800.0}


def zone_config(cfg) -> dict:
    config = dict(ZONE_DEFAULTS)
    config.update(cfg.library.get("source_zones") or {})
    return config


def load_source_durations(library_dir: Path) -> dict[str, float]:
    """从 sources/*/*.json（窗口扫描信封）读各源视频真实时长。"""
    durations: dict[str, float] = {}
    for path in sorted(Path(library_dir).glob("sources/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        video = payload.get("video")
        duration = payload.get("duration_s")
        if not isinstance(video, str) or not isinstance(duration, (int, float)):
            continue
        durations[video] = max(float(duration), durations.get(video, 0.0))
    return durations


def row_video_durations(rows: list[dict]) -> dict[str, float]:
    """兜底时长：无扫描信封的源用已索引镜头的最大 end_s 估计（略偏小）。"""
    durations: dict[str, float] = {}
    for row in rows:
        video = str(row.get("video") or "")
        try:
            end = float(row.get("end_s") or row.get("source_end_s") or 0)
        except (TypeError, ValueError):
            continue
        if video and end > durations.get(video, 0.0):
            durations[video] = end
    return durations


def exclusion_reason(row: dict, durations: dict[str, float], config: dict) -> str:
    """返回排除原因（head_credits/tail_credits），合格返回空串。

    只对长片生效：预告片/短切片（< min_movie_s）整条不做区排除。
    """
    duration = durations.get(str(row.get("video") or ""))
    if duration is None or duration < float(config.get("min_movie_s", 1800.0)):
        return ""
    try:
        start = float(row.get("start_s") or row.get("source_start_s") or 0.0)
        end = float(row.get("end_s") or row.get("source_end_s") or start)
    except (TypeError, ValueError):
        return ""
    if end < start:
        end = start
    if start < float(config.get("head_s", 90.0)):
        return "head_credits"
    if end > duration - float(config.get("tail_s", 360.0)):
        return "tail_credits"
    return ""


def excluded_row_indices(rows: list[dict], cfg) -> set[int]:
    """检索侧统一入口：返回被排除的行号集合（相对传入 rows 的下标）。"""
    config = zone_config(cfg)
    durations = load_source_durations(cfg.paths.library_dir)
    for video, end in row_video_durations(rows).items():
        if end > durations.get(video, 0.0):
            durations[video] = end
    return {idx for idx, row in enumerate(rows)
            if exclusion_reason(row, durations, config)}
