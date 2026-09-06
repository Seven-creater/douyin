"""run_baseline：whole-video Qwen3-Omni baseline 批跑（模型只加载一次）。

CLI：python -m src.perception.run_baseline [--limit N] [--ids a,b] [--force] [--gpus 0,1]
排序：按视频时长升序（先短后长，风险早暴露）。
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
from src.perception import common, omni_watch_full

logger = logging.getLogger(__name__)


def ordered_ids(cfg: AppConfig) -> list[str]:
    """读 metadata.json 的 wellbyte.extras.duration_ms 升序；缺省回退字典序。"""
    pairs = []
    for aweme_id in common.list_aweme_ids(cfg.paths.videos_dir):
        meta_path = cfg.paths.videos_dir / aweme_id / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dur_ms = meta["wellbyte"]["extras"]["duration_ms"] or 10**9
        except (OSError, ValueError, KeyError):
            dur_ms = 10**9
        pairs.append((aweme_id, int(dur_ms)))
    return [a for a, _ in sorted(pairs, key=lambda p: p[1])]


def run_baseline(cfg: AppConfig, *, ids: list[str] | None = None, force: bool = False,
                 limit: int | None = None) -> dict:
    p_cfg = common.perception_cfg(cfg)
    all_ids = ids or ordered_ids(cfg)
    if limit:
        all_ids = all_ids[:limit]
    if not all_ids:
        raise SystemExit("没有待处理视频")

    from src.perception.omni_runner import OmniRunner

    gpus = p_cfg.get("omni", {}).get("visible_gpus", "0,1")
    from src.perception.omni_runner import set_visible_gpus

    set_visible_gpus(gpus)
    runner = OmniRunner(p_cfg.get("omni") or {}, ffmpeg_bin=p_cfg.get("ffmpeg_bin", "ffmpeg"))
    runner.load()  # 只加载一次

    n_ok = n_skip = n_fail = 0
    per_video = []
    t_all = time.time()
    for i, aweme_id in enumerate(all_ids, 1):
        t0 = time.time()
        try:
            path = omni_watch_full.run_for_video(cfg, aweme_id, force=force, runner=runner)
            skipped = time.time() - t0 < 0.05
            n_skip += skipped
            n_ok += not skipped
            usage = None
            try:
                usage = json.loads(Path(path).read_text(encoding="utf-8"))["output"]["usage"]
            except (OSError, ValueError, KeyError):
                pass
            per_video.append({"id": aweme_id, "status": "skip" if skipped else "ok",
                              "total_s": round(time.time() - t0, 1), "usage": usage})
            logger.info("[baseline %d/%d] %s %s", i, len(all_ids), aweme_id,
                        "skip" if skipped else f"ok {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001 - 单条失败不杀批
            n_fail += 1
            per_video.append({"id": aweme_id, "status": "error", "error": str(exc)[:300]})
            logger.exception("[baseline %d/%d] %s 失败", i, len(all_ids), aweme_id)
            common.append_metric(
                common.metrics_path_for(cfg.paths.perception_dir),
                tool="omni_watch_full", aweme_id=aweme_id, status="error",
                extra={"error": str(exc)[:300]},
            )

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": gpus, "load_elapsed_s": runner.load_elapsed_s,
        "total_elapsed_s": round(time.time() - t_all, 1),
        "n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail,
        "per_video": per_video,
    }
    out = cfg.paths.processed_dir / f"perception_baseline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(out)}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="whole-video baseline 批跑")
    ap.add_argument("--limit", type=int, default=None, help="只跑时长最短的前 N 条")
    ap.add_argument("--ids", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_baseline")
    if args.gpus:
        cfg.perception.setdefault("omni", {})["visible_gpus"] = args.gpus

    result = run_baseline(cfg, ids=[i for i in args.ids.split(",") if i.strip()],
                          force=args.force, limit=args.limit)
    print()
    print("=" * 60)
    print(f"Whole-video Baseline Summary（gpus={result['gpus']}）")
    print(f"模型加载: {result['load_elapsed_s']:.0f}s · 总耗时: {result['total_elapsed_s']:.0f}s")
    print(f"结果: ok={result['n_ok']} skip={result['n_skip']} fail={result['n_fail']}")
    for v in result["per_video"]:
        line = f"  {v['id']} {v['status']:5s} {v.get('total_s', 0):7.1f}s"
        if v.get("usage"):
            u = v["usage"]
            line += f"  tokens={u['input_tokens']}/{u['output_tokens']} vram={u['vram_peak_gb']}"
        if v.get("error"):
            line += f"  ERR={v['error'][:80]}"
        print(line)
    print(f"产物: {result['summary_path']}")
    print("=" * 60)
    return 0 if result["n_fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
