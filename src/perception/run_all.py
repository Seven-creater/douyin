"""run_all：轻/中感知工具一条命令批跑全部视频（模型各加载一次复用）。

CLI：python -m src.perception.run_all [--tools inspect,frames,shots,ocr,transcribe] [--force] [--ids a,b]
顺序固定（ocr 依赖 frames 产物）；每个工具每条视频幂等（已有产物跳过）。
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
from src.perception import detect_beats, detect_shots, extract_frames, inspect_video, ocr_frames, transcribe_audio

logger = logging.getLogger(__name__)

DEFAULT_TOOLS = ("inspect", "frames", "shots", "ocr", "transcribe", "beats")


def run_all(cfg: AppConfig, *, ids: list[str] | None = None, force: bool = False,
            tools: tuple[str, ...] = DEFAULT_TOOLS) -> dict:
    ids = ids or common.list_aweme_ids(cfg.paths.videos_dir)
    if not ids:
        raise SystemExit("data/videos/ 下没有已下载视频（先跑 Phase 1 collect_trends）")

    stats = {t: {"ok": 0, "skip": 0, "error": 0, "errors": []} for t in tools}
    t_models: dict[str, object] = {}

    for tool in tools:
        for aweme_id in ids:
            t0 = time.time()
            try:
                if tool == "inspect":
                    path = inspect_video.run_for_video(cfg, aweme_id, force=force)
                elif tool == "frames":
                    path = extract_frames.run_for_video(cfg, aweme_id, force=force)
                elif tool == "shots":
                    path = detect_shots.run_for_video(cfg, aweme_id, force=force)
                elif tool == "ocr":
                    if "ocr" not in t_models:
                        t_models["ocr"] = ocr_frames.load_ocr()
                    path = ocr_frames.run_for_video(cfg, aweme_id, force=force, ocr=t_models["ocr"])
                elif tool == "beats":
                    path = detect_beats.run_for_video(cfg, aweme_id, force=force)
                elif tool == "transcribe":
                    if "asr" not in t_models:
                        t_models["asr"] = transcribe_audio.load_transcriber(
                            model=cfg.perception.get("transcribe", {}).get("model", "iic/SenseVoiceSmall"),
                            vad_model=cfg.perception.get("transcribe", {}).get("vad_model", "fsmn-vad"),
                            vad_max_segment_ms=int(cfg.perception.get("transcribe", {}).get("vad_max_single_segment_ms", 30000)),
                            device=cfg.perception.get("transcribe", {}).get("device", "cuda:0"),
                        )
                    path = transcribe_audio.run_for_video(cfg, aweme_id, force=force, model=t_models["asr"])
                else:
                    raise ValueError(f"未知工具: {tool}")
                # run_for_video 返回路径：区分新跑还是跳过没有直接信号，用耗时近似（<0.05s 视为跳过）
                stats[tool]["skip" if time.time() - t0 < 0.05 else "ok"] += 1
            except Exception as exc:  # noqa: BLE001 - 单条失败不杀批
                logger.exception("[run_all %s %s] 失败", tool, aweme_id)
                stats[tool]["error"] += 1
                stats[tool]["errors"].append({"id": aweme_id, "error": str(exc)[:300]})
                common.append_metric(
                    common.metrics_path_for(cfg.paths.perception_dir),
                    tool=f"run_all:{tool}", aweme_id=aweme_id, status="error",
                    elapsed_s=round(time.time() - t0, 2), extra={"error": str(exc)[:300]},
                )
        logger.info("[run_all] %s 完成: %s", tool, {k: v for k, v in stats[tool].items() if k != "errors"})

    summary_path = cfg.paths.processed_dir / f"perception_run_all_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(
        {"finished_at": datetime.now().isoformat(timespec="seconds"), "n_videos": len(ids),
         "tools": {t: {k: v for k, v in s.items() if k != "errors"} for t, s in stats.items()},
         "errors": {t: s["errors"] for t, s in stats.items() if s["errors"]}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stats": stats, "summary": str(summary_path)}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="轻/中感知工具批跑")
    ap.add_argument("--tools", default=",".join(DEFAULT_TOOLS))
    ap.add_argument("--ids", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_run_all")

    tools = tuple(t.strip() for t in args.tools.split(",") if t.strip())
    unknown = [t for t in tools if t not in DEFAULT_TOOLS]
    if unknown:
        raise SystemExit(f"未知工具: {unknown}（可选 {DEFAULT_TOOLS}）")
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or None

    result = run_all(cfg, ids=ids, force=args.force, tools=tools)
    stats = result["stats"]
    print()
    print("=" * 56)
    print(f"感知工具批跑 Summary（{sum(s['ok'] for s in stats.values())} ok / "
          f"{sum(s['skip'] for s in stats.values())} skip / "
          f"{sum(s['error'] for s in stats.values())} error）")
    for t, s in stats.items():
        print(f"  {t:10s} ok={s['ok']:2d} skip={s['skip']:2d} error={s['error']}")
    print(f"产物: {result['summary']}")
    print("=" * 56)
    return 2 if any(s["error"] for s in stats.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
