"""单命令入口：抓榜 → 存 raw → parse → 去重 → 排序 → TopN →（Step 6 起）下载 → summary。

用法：
    python -m src.pipeline.collect_trends                 # 全流程（下载阶段 Step 6 接入）
    python -m src.pipeline.collect_trends --no-download   # 只做榜单/解析/排序（服务器模式）
    python -m src.pipeline.collect_trends --refetch       # 当日 raw 已存在也强制重拉
    python -m src.pipeline.collect_trends --top-n 10 --max-downloads 3

运行端分工（2026-09-06 Gate A 实测结论）：
    服务器：本命令 --no-download（Wellbyte 可达；字节系 CDN 被防火墙屏蔽，无法下载）
    本地：  全流程（CDN mp4 直链 requests 直下，无需 Cookie/TikTokDownloader）
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig, ensure_utf8_stdio, load_config, require_api_key, setup_logging
from src.trend.deduplicate import MergedVideo, merge_videos, write_outputs
from src.trend.parser import parse_file
from src.trend.ranking import rank_videos
from src.trend.wellbyte_client import FetchOutcome, WellbyteClient

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="抖音热点采集流水线（Phase 1）")
    ap.add_argument("--no-download", action="store_true", help="只抓榜/解析/排序，不下载")
    ap.add_argument("--refetch", action="store_true", help="当日 raw 已存在也强制重拉（花 credits）")
    ap.add_argument("--top-n", type=int, default=None, help="Top N 候选数（默认取 config）")
    ap.add_argument("--max-downloads", type=int, default=None, help="本次最多下载条数（阶梯验证用）")
    ap.add_argument("--sub-types", default="", help="逗号分隔榜单号；默认 config 全部 5 榜")
    ap.add_argument("--date", default=None, help="处理哪天的 raw（YYYY-MM-DD，默认今天）")
    return ap.parse_args(argv)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def run_trend_stage(
    cfg: AppConfig,
    *,
    sub_types: list[int],
    date: str,
    refetch: bool,
) -> tuple[list[MergedVideo], list[MergedVideo], dict[str, Any]]:
    """抓榜（或复用当日 raw）→ parse → 去重 → 排序。

    返回 (全部去重排序后的视频, TopN 候选, 统计)。
    """
    raw_paths = {st: cfg.paths.raw_dir / f"{date}_{st}.json" for st in sub_types}
    missing = [st for st, p in raw_paths.items() if not p.exists()]

    outcomes: list[FetchOutcome] = []
    if missing or refetch:
        api_key = require_api_key()  # 只有真要发请求才需要 key
        client = WellbyteClient(
            api_key,
            base_url=cfg.wellbyte.base_url,
            endpoint=cfg.wellbyte.endpoint,
            timeout=cfg.wellbyte.timeout_seconds,
            max_attempts=cfg.wellbyte.retry_max_attempts,
            backoff_seconds=cfg.wellbyte.retry_backoff_seconds,
        )
        for st in sub_types:
            outcomes.append(
                client.fetch_and_save(st, cfg.wellbyte.request_params, raw_paths[st], overwrite=refetch)
            )
    else:
        logger.info("当日 %d 个 raw 全部存在，跳过所有请求（0 credits）", len(sub_types))

    # 解析所有存在的 raw（失败的榜单用已有文件；全缺则该榜无记录）
    records = []
    per_billboard: dict[int, dict[str, Any]] = {}
    for st in sub_types:
        p = raw_paths[st]
        if p.exists():
            recs = parse_file(p, date_window=int(cfg.wellbyte.request_params.get("date_window", 24)))
            records.extend(recs)
            per_billboard[st] = {"count": len(recs), "raw": str(p)}
        else:
            per_billboard[st] = {"count": 0, "raw": None}

    merged = merge_videos(records)
    top_n = cfg.ranking.get("top_n", 20)
    ranked = rank_videos(merged, top_n=top_n)  # 全量已排序，取 TopN
    full_ordered = rank_videos(merged, top_n=len(merged))

    jsonl_path, csv_path = write_outputs([sv.merged for sv in full_ordered], cfg.paths.processed_dir)

    fetched = [o for o in outcomes if o.status == "fetched"]
    stats = {
        "date": date,
        "requests": len(outcomes),
        "fetched": len(fetched),
        "skipped": sum(1 for o in outcomes if o.status == "skipped_existing"),
        "failed": [o.error for o in outcomes if o.status == "failed"],
        "credits_charged": sum(int(o.meta.get("credits_charged") or 0) for o in fetched),
        "per_billboard": {str(st): per_billboard[st]["count"] for st in sub_types},
        "total_records": len(records),
        "unique_videos": len(merged),
        "top_n": top_n,
        "outputs": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }
    return [sv.merged for sv in full_ordered], [sv.merged for sv in ranked], stats


def print_summary(trend_stats: dict[str, Any], download_stats: dict[str, Any] | None, elapsed_s: float) -> None:
    print()
    print("=" * 62)
    print(f"抖音热点采集 Summary  {trend_stats.get('date', '')}")
    print("=" * 62)
    n_req = trend_stats["requests"]
    print(
        f"榜单请求: {n_req}"
        f"（成功 {trend_stats['fetched']} / 跳过 {trend_stats['skipped']} / 失败 {len(trend_stats['failed'])}）"
        f" · credits={trend_stats['credits_charged']}"
    )
    per = "  ".join(f"{st}:{cnt}" for st, cnt in trend_stats["per_billboard"].items())
    print(f"各榜视频数: {per}")
    print(f"条目: {trend_stats['total_records']} → 去重后 {trend_stats['unique_videos']} · TopN {trend_stats['top_n']}")
    if download_stats:
        skipped_total = download_stats["skipped"] + download_stats.get("skipped_image", 0)
        print(
            f"下载: 成功 {download_stats['success']} / 失败 {download_stats['failed']}"
            f" / 跳过 {skipped_total}"
            f"（已下载 {download_stats['skipped']} + 图集 {download_stats.get('skipped_image', 0)}）"
            f"（明细见 data/videos/manifest.json）"
        )
        if download_stats.get("sync"):
            sc = download_stats["sync"]
            print(f"回传服务器: 推送 {sc['pushed']} 个视频目录，失败 {len(sc['errors'])} 项")
    else:
        print("下载: 未启用（--no-download）")
    print(f"总耗时: {elapsed_s:.1f}s")
    print(f"产物: {trend_stats['outputs']['jsonl']}")
    print(f"      {trend_stats['outputs']['csv']}")
    print("=" * 62)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    args = parse_args(argv)
    t0 = time.time()
    cfg = load_config()
    setup_logging(cfg.paths.logs_dir, cfg.logging_level)

    sub_types = [int(s) for s in args.sub_types.split(",") if s.strip()] or cfg.wellbyte.sub_types
    date = args.date or _today()
    if args.top_n:
        cfg = AppConfig(  # dataclass 不可变，覆盖 top_n 重建
            wellbyte=cfg.wellbyte, paths=cfg.paths,
            ranking={**cfg.ranking, "top_n": args.top_n},
            download=cfg.download, perception=cfg.perception,
            template=cfg.template, generation=cfg.generation, logging_level=cfg.logging_level,
        )

    try:
        _, top, trend_stats = run_trend_stage(cfg, sub_types=sub_types, date=date, refetch=args.refetch)
    except Exception:
        logger.exception("trend 阶段失败")
        return 1

    download_stats: dict[str, Any] | None = None
    if not args.no_download:
        # 本地 CDN 直下 + 回传服务器；下载阶段的意外异常不掩盖已完成的榜单结果
        from src.pipeline.run_downloads import run_download_stage  # 延迟导入（服务器无需）

        try:
            download_stats = run_download_stage(cfg, top, max_downloads=args.max_downloads)
        except Exception:  # noqa: BLE001
            logger.exception("download 阶段意外失败（榜单结果已保留）")
            download_stats = {"success": 0, "failed": 0, "skipped": 0, "skipped_image": 0,
                              "attempted": 0, "errors": [{"id": "-", "error": "download stage crashed, see log"}], "sync": None}

    print_summary(trend_stats, download_stats, time.time() - t0)

    # run_summary 存档
    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "trend": trend_stats,
        "download": download_stats,
    }
    out = cfg.paths.processed_dir / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("run_summary → %s", out)

    if trend_stats["failed"]:
        return 2
    if download_stats and download_stats["failed"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
