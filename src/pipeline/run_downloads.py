"""下载阶段编排：TopN → 过滤图集 → manifest 跳过 → CDN 直下（重试）→ metadata → 回传服务器。

统计口径（summary 用）：
    success / failed / skipped(已下载跳过) / skipped_image(图集跳过) / attempted
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.download.direct_downloader import DirectCDNDownloader
from src.download.manifest import Manifest
from src.download.models import DownloadItem, DownloadResult
from src.download.sync import push_videos

logger = logging.getLogger(__name__)

MEDIA_TYPE_VIDEO = 4  # 实测：4=视频，2=图集（图集的 item_url 是其 BGM mp3，不当视频下）


def _write_metadata(videos_dir: Path, item: DownloadItem, merged_raw: dict) -> Path:
    """合并 wellbyte 榜单信息为每视频 metadata.json（Phase 2 perception 直接可读）。"""
    meta_path = videos_dir / item.aweme_id / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aweme_id": item.aweme_id,
        "url": item.url,
        "title": item.title,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "wellbyte": merged_raw,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def run_download_stage(
    cfg: AppConfig,
    top_merged: list,
    *,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    dl_cfg = cfg.download
    retry_cfg = dl_cfg.get("retry") or {}
    max_attempts = int(retry_cfg.get("max_attempts", 2))
    backoff = float(retry_cfg.get("backoff_seconds", 10))
    direct_cfg = dl_cfg.get("direct") or {}

    manifest = Manifest(cfg.paths.videos_dir / "manifest.json", cfg.paths.videos_dir)
    downloader = DirectCDNDownloader(
        cfg.paths.videos_dir,
        user_agent=direct_cfg.get("user_agent", "Mozilla/5.0"),
        referer=direct_cfg.get("referer", "https://www.douyin.com/"),
        min_valid_bytes=int(dl_cfg.get("min_valid_bytes", 10240)),
        timeout=float(dl_cfg.get("per_video_timeout_seconds", 600)),
    )

    stats: dict[str, Any] = {
        "success": 0, "failed": 0, "skipped": 0, "skipped_image": 0,
        "attempted": 0, "errors": [], "sync": None,
    }
    new_success_ids: list[str] = []
    attempted_videos = 0

    for mv in top_merged:
        media_type = (mv.record.extras or {}).get("media_type")
        if media_type != MEDIA_TYPE_VIDEO:
            stats["skipped_image"] += 1
            logger.info("[download] %s 为图集(media_type=%s)，跳过（其 item_url 是 BGM mp3，已留存于数据）", mv.aweme_id, media_type)
            continue
        if max_downloads is not None and attempted_videos >= max_downloads:
            break

        item = DownloadItem(
            aweme_id=mv.aweme_id,
            url=mv.record.download_url or "",
            title=mv.record.title,
            media_type=media_type,
        )

        if manifest.is_done(mv.aweme_id):
            stats["skipped"] += 1
            attempted_videos += 1
            logger.info("[download %s] manifest 记录已成功且文件在，跳过", mv.aweme_id)
            continue
        if not item.url:
            manifest.record_failure(mv.aweme_id, url="", title=item.title, stage="no_url", error="download_url 为空")
            manifest.save()
            stats["failed"] += 1
            stats["errors"].append({"id": mv.aweme_id, "error": "no_url"})
            attempted_videos += 1
            continue

        stats["attempted"] += 1
        attempted_videos += 1
        result = None
        try:
            for attempt in range(1, max_attempts + 1):
                result = downloader.download_video(item)
                if result.success:
                    break
                logger.warning("[download %s] 第 %d/%d 次失败", mv.aweme_id, attempt, max_attempts)
                if attempt < max_attempts:
                    time.sleep(backoff)
        except Exception as exc:  # noqa: BLE001 - 单条意外错误只记失败，不杀整个 batch
            logger.exception("[download %s] 意外异常", mv.aweme_id)
            result = DownloadResult(
                success=False, aweme_id=mv.aweme_id, url=item.url,
                error=f"unexpected: {type(exc).__name__}: {exc}",
            )

        if result and result.success:
            meta_path = _write_metadata(cfg.paths.videos_dir, item, mv.record.to_dict())
            manifest.record_success(
                mv.aweme_id, url=item.url, title=item.title,
                video_path=result.video_path or cfg.paths.videos_dir / mv.aweme_id / "video.mp4",
                file_size=result.file_size_bytes or 0,
            )
            manifest.save()
            stats["success"] += 1
            new_success_ids.append(mv.aweme_id)
        else:
            error = result.error if result else "unknown"
            manifest.record_failure(mv.aweme_id, url=item.url, title=item.title, stage="download", error=error)
            manifest.save()
            stats["failed"] += 1
            stats["errors"].append({"id": mv.aweme_id, "error": error})

    # 回传服务器（Phase 2 感知在服务器；失败不致命）
    sync_cfg = _load_sync_cfg()
    if sync_cfg.get("enabled") and new_success_ids:
        stats["sync"] = push_videos(
            sync_cfg["ssh_target"], sync_cfg["remote_root"],
            cfg.paths.videos_dir, cfg.paths.processed_dir,
            new_ids=new_success_ids, manifest_path=cfg.paths.videos_dir / "manifest.json",
        )

    return stats


def _load_sync_cfg() -> dict:
    """sync 配置放在 config/default.yaml 顶层 sync: 段（AppConfig 未显式携带）。"""
    import yaml

    from src.config import repo_root

    path = repo_root() / "config" / "default.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw.get("sync") or {}
    except (OSError, yaml.YAMLError):
        return {}
