"""omni_watch_clip：视频片段理解（ffmpeg 切片 → Omni，低成本 Action）。

CLI：python -m src.perception.omni_watch_clip --aweme-id <id> --start 3 --end 9.5 --question "..." [--gpus 0,1]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common
from src.perception.omni_prompts import build_clip_prompt

logger = logging.getLogger(__name__)


def run_for_video(cfg: AppConfig, aweme_id: str, *, start_s: float, end_s: float,
                  question: str, force: bool = False, runner=None):
    p_cfg = common.perception_cfg(cfg)
    video = common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    duration_s = common.video_duration_s(p_cfg.get("ffprobe_bin", "ffprobe"), video)
    if not (0 <= start_s < end_s <= duration_s + 0.01):
        raise ValueError(f"片段区间非法: [{start_s}, {end_s}]，视频时长 {duration_s:.1f}s")

    clip_dir = common.clip_dir_for(cfg.paths.perception_dir, aweme_id, start_s, end_s)
    params = {"start_s": start_s, "end_s": end_s, "question": question,
              "fps": p_cfg.get("omni", {}).get("fps", 2.0)}
    if common.done_or_skip(clip_dir, params, force=force):
        logger.info("[omni_clip %s %s-%s] 已有产物，跳过", aweme_id, start_s, end_s)
        return clip_dir / "result.json"

    if runner is None:
        from src.perception.omni_runner import OmniRunner

        runner = OmniRunner(p_cfg.get("omni") or {}, ffmpeg_bin=p_cfg.get("ffmpeg_bin", "ffmpeg"))

    answer = runner.watch(video, build_clip_prompt(start_s, end_s, question),
                          start_s=start_s, end_s=end_s, clip_dir=clip_dir,
                          max_new_tokens=p_cfg.get("omni", {}).get("max_new_tokens", 2048),
                          duration_s=end_s - start_s)

    output = {
        "video": {"path": str(video), "duration_s": duration_s},
        "clip": {"start_s": start_s, "end_s": end_s,
                 "duration_s": round(end_s - start_s, 2),
                 "path": str(answer.clip_path) if answer.clip_path else None},
        "question": question,
        "answer_text": answer.text,
        "usage": {
            "elapsed_s": answer.elapsed_s, "input_build_s": answer.input_build_s,
            "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
            "vram_peak_gb": answer.vram_peak_gb,
            "frames_estimate": answer.frames_estimate,
            "load_elapsed_s": runner.load_elapsed_s,
        },
    }
    path = common.write_result_json(clip_dir, tool="omni_watch_clip", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="omni_watch_clip", aweme_id=aweme_id, status="ok",
        elapsed_s=answer.elapsed_s, input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens, vram_peak_gb=answer.vram_peak_gb,
        extra={"clip": [start_s, end_s], "fps": params["fps"]},
    )
    logger.info("[omni_clip %s %.1f-%.1f] %.1fs tokens=%d/%d → %s", aweme_id, start_s, end_s,
                answer.elapsed_s, answer.input_tokens, answer.output_tokens, path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="片段理解（切片 → Qwen3-Omni）")
    common.add_common_cli_args(ap)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--gpus", default=None)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_omni_clip")

    from src.perception.omni_runner import set_visible_gpus

    set_visible_gpus(args.gpus or cfg.perception.get("omni", {}).get("visible_gpus", "0,1"))

    try:
        aweme_id, _ = common.resolve_target(cfg, args)
        path = run_for_video(cfg, aweme_id, start_s=args.start, end_s=args.end,
                             question=args.question, force=args.force)
        common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("omni_watch_clip 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
