"""run_library：素材库管线编排（B 索引 + C 三步），阶段闸门制。

CLI：python -m src.library.run_library --stage shots|captions|index|story|retrieve|assemble|all
        --template-id <id> [--source trailers] [--force]
一条命令（story 起）：python -m src.library.run_library --template-id <id> --stage all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="素材库管线")
    ap.add_argument("--stage", default="all",
                    choices=["shots", "captions", "index", "story", "retrieve", "assemble", "all"])
    ap.add_argument("--template-id", default=None, help="story 起必填")
    ap.add_argument("--source", default="trailers")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_run")

    need_tid = args.stage in ("story", "retrieve", "assemble", "all")
    if need_tid and not args.template_id:
        ap.error(f"--stage {args.stage} 需要 --template-id")

    def _run(module: str, fn: str, **kw):
        import importlib

        m = importlib.import_module(f"src.library.{module}")
        return getattr(m, fn)(cfg, **kw)

    try:
        if args.stage in ("shots", "all"):
            _run("index_shots", "index_all", source=args.source, force=args.force) \
                if hasattr(__import__("src.library.index_shots", fromlist=["x"]), "index_all") \
                else _shots_all(cfg, args.source, args.force)
        if args.stage in ("captions", "all"):
            _captions_all(cfg, args.force)
        if args.stage in ("index", "all"):
            from src.library.build_index import build
            build(cfg)
        if args.stage in ("story", "all"):
            from src.library.story import run_story
            run_story(cfg, args.template_id, force=args.force)
        if args.stage in ("retrieve", "all"):
            from src.library.retrieve import run_retrieve
            run_retrieve(cfg, args.template_id, force=args.force)
        if args.stage in ("assemble", "all"):
            from src.library.assemble_lib import run_assemble
            run_assemble(cfg, args.template_id, force=args.force)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_library 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1
    common.emit_status_line("ok", stage=args.stage)
    return 0


def _shots_all(cfg, source: str, force: bool):
    from src.library.index_shots import index_video

    raw_dir = cfg.paths.library_dir / "raw" / source
    videos = sorted(p for p in raw_dir.glob("*.mp4") if p.is_file())
    for v in videos:
        index_video(cfg, v, force=force)


def _captions_all(cfg, force: bool):
    from src.library.caption_shots import Qwen2VLRunner, caption_video

    runner = Qwen2VLRunner(cfg.library.get("captions") or {})
    for r in sorted((cfg.paths.library_dir / "shots").glob("*/result.json")):
        caption_video(cfg, r, force=force, runner=runner)


if __name__ == "__main__":
    sys.exit(main())
