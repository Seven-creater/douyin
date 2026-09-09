"""ingest：外部视频入库（D4）——参考片/GLM 重剪版 → data/videos/<id>/ + 可选推送服务器。

与抖音下载区分开：这类视频没有 manifest/感知链路，只有文件本身。
metadata.json 记录来源与引用关系（reedit 引用 reference 的 aweme_id，report 依赖）。

CLI：python -m src.library.ingest --file <path> --id <id> --kind reference|reedit
     [--ref <aweme_id>] [--title <t>] [--push] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def run_ingest(cfg, file: Path, vid: str, *, kind: str, ref: str | None = None,
               title: str = "", force: bool = False, push: bool = False) -> Path:
    if kind not in ("reference", "reedit"):
        raise ValueError(f"kind 只能 reference|reedit，得到 {kind!r}")
    if not file.exists():
        raise FileNotFoundError(file)
    dst_dir = cfg.paths.videos_dir / vid
    dst = dst_dir / "video.mp4"
    if dst.exists() and not force:
        logger.info("[ingest %s] 已存在，跳过（--force 覆盖）", vid)
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dst)
        logger.info("[ingest %s] %s → %s（%.1fMB）", vid, file.name, dst,
                    dst.stat().st_size / 2**20)
    (dst_dir / "metadata.json").write_text(json.dumps({
        "aweme_id": vid, "title": title or file.stem, "kind": kind,
        "source": "manual_ingest", "references": [ref] if ref else [],
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin_path": str(file)}, ensure_ascii=False, indent=1), encoding="utf-8")
    if push:
        from src.download.sync import _push_one_dir
        from src.pipeline.run_downloads import _load_sync_cfg

        sync_cfg = _load_sync_cfg()
        err = _push_one_dir(sync_cfg["ssh_target"],
                            f"{sync_cfg['remote_root']}/data/videos", dst_dir)
        logger.info("[ingest %s] 推送 %s", vid, "失败" if err else "成功")
    common.emit_status_line("ok", vid=vid, kind=kind)
    return dst


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="外部视频入库（参考/重剪对照）")
    ap.add_argument("--file", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--kind", choices=["reference", "reedit"], required=True)
    ap.add_argument("--ref", default=None, help="reedit 引用的 reference aweme_id")
    ap.add_argument("--title", default="")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_ingest")
    try:
        run_ingest(cfg, Path(args.file), args.id, kind=args.kind, ref=args.ref,
                   title=args.title, force=args.force, push=args.push)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
