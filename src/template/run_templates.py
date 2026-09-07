"""run_templates：模板抽取批跑（模型加载一次；镜像 run_baseline 模式）。

CLI：python -m src.template.run_templates [--limit N] [--ids a,b] [--force] [--gpus 0,1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common
from src.template import extract_template

logger = logging.getLogger(__name__)


def ordered_ids(cfg: AppConfig) -> list[str]:
    """按时长升序（与 run_baseline 一致，先短后长）。"""
    pairs = []
    for aweme_id in common.list_aweme_ids(cfg.paths.videos_dir):
        try:
            env = common.read_result_json(Path(cfg.paths.perception_dir) / aweme_id / "inspect")
            dur = float((env or {}).get("output", {}).get("duration_s") or 10**9)
        except (OSError, ValueError, TypeError):
            dur = 10**9
        pairs.append((aweme_id, dur))
    return [a for a, _ in sorted(pairs, key=lambda p: p[1])]


def run_templates(cfg: AppConfig, *, ids: list[str] | None = None, force: bool = False,
                  limit: int | None = None) -> dict:
    all_ids = ids or ordered_ids(cfg)
    if limit:
        all_ids = all_ids[:limit]

    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    gpus = cfg.perception.get("omni", {}).get("visible_gpus", "0,1")
    set_visible_gpus(gpus)
    runner = OmniRunner(cfg.perception.get("omni") or {})
    runner.load()

    n_ok = n_skip = n_fail = 0
    per_video = []
    t_all = time.time()
    for i, aweme_id in enumerate(all_ids, 1):
        t0 = time.time()
        try:
            path = extract_template.synthesize(cfg, aweme_id, force=force, runner=runner)
            if path is None:
                n_fail += 1
                per_video.append({"id": aweme_id, "status": "error", "error": "parse failed"})
            elif time.time() - t0 < 0.05:
                n_skip += 1
                per_video.append({"id": aweme_id, "status": "skip"})
            else:
                n_ok += 1
                usage = {}
                try:
                    usage = json.loads(Path(path).read_text(encoding="utf-8"))["output"]["usage"]
                except (OSError, ValueError, KeyError):
                    pass
                per_video.append({"id": aweme_id, "status": "ok",
                                  "total_s": round(time.time() - t0, 1), "usage": usage})
            logger.info("[templates %d/%d] %s %s", i, len(all_ids), aweme_id, per_video[-1]["status"])
        except Exception as exc:  # noqa: BLE001 - 单条失败不杀批
            n_fail += 1
            per_video.append({"id": aweme_id, "status": "error", "error": str(exc)[:300]})
            logger.exception("[templates %d/%d] %s 失败", i, len(all_ids), aweme_id)
            common.append_metric(
                common.metrics_path_for(cfg.paths.perception_dir),
                tool="extract_template", aweme_id=aweme_id, status="error",
                extra={"error": str(exc)[:300]},
            )

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": gpus, "load_elapsed_s": runner.load_elapsed_s,
        "total_elapsed_s": round(time.time() - t_all, 1),
        "n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail, "per_video": per_video,
    }
    out = cfg.paths.processed_dir / f"templates_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(out)}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="模板抽取批跑")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="template_run")
    if args.gpus:
        cfg.perception.setdefault("omni", {})["visible_gpus"] = args.gpus

    result = run_templates(cfg, ids=[i for i in args.ids.split(",") if i.strip()],
                           force=args.force, limit=args.limit)
    print()
    print("=" * 60)
    print(f"模板抽取 Summary（gpus={result['gpus']}）")
    print(f"加载 {result['load_elapsed_s']:.0f}s · 总耗时 {result['total_elapsed_s']:.0f}s · "
          f"ok={result['n_ok']} skip={result['n_skip']} fail={result['n_fail']}")
    for v in result["per_video"]:
        line = f"  {v['id']} {v['status']:5s}"
        if v.get("usage"):
            u = v["usage"]
            line += f" {v.get('total_s', 0):6.1f}s in_tok={u.get('input_tokens')}"
        if v.get("error"):
            line += f" ERR={v['error'][:60]}"
        print(line)
    print(f"产物: {result['summary_path']}")
    print("=" * 60)
    return 0 if result["n_fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
