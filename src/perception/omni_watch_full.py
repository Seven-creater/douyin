"""omni_watch_full：整段视频理解（whole-video baseline，高成本 Action）。

CLI：python -m src.perception.omni_watch_full --aweme-id <id> [--gpus 0,1] [--question ...] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common
from src.perception.omni_prompts import BASELINE_PROMPT, dedupe_repetition, parse_baseline_sections

logger = logging.getLogger(__name__)


def build_prompt(question: str | None) -> str:
    if not question:
        return BASELINE_PROMPT
    return BASELINE_PROMPT + "\n\n在完成上述分析后，最后单独一行回答：\n" + question


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False,
                  question: str | None = None, prompt_name: str = "baseline_v1",
                  runner=None, video_override: Path | None = None):
    p_cfg = common.perception_cfg(cfg)
    video = video_override or common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "omni_full")
    params = {"prompt_name": prompt_name, "fps": p_cfg.get("omni", {}).get("fps", 2.0),
              "question": question}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[omni_full %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    if runner is None:
        from src.perception.omni_runner import OmniRunner

        runner = OmniRunner(p_cfg.get("omni") or {}, ffmpeg_bin=p_cfg.get("ffmpeg_bin", "ffmpeg"))

    duration_s = common.video_duration_s(p_cfg.get("ffprobe_bin", "ffprobe"), video)
    t0 = time.time()
    answer = runner.watch(video, build_prompt(question),
                          max_new_tokens=p_cfg.get("omni", {}).get("max_new_tokens", 2048),
                          duration_s=duration_s)
    answer.text = dedupe_repetition(answer.text)  # 复读保险
    total_s = time.time() - t0

    output = {
        "video": {"path": str(video), "duration_s": duration_s},
        "prompt_name": prompt_name,
        "answer_text": answer.text,
        "sections": parse_baseline_sections(answer.text),
        "usage": {
            "elapsed_s": answer.elapsed_s, "total_s": round(total_s, 2),
            "input_build_s": answer.input_build_s,
            "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
            "vram_peak_gb": answer.vram_peak_gb, "vram_reserved_gb": answer.vram_reserved_gb,
            "frames_estimate": answer.frames_estimate,
            "load_elapsed_s": runner.load_elapsed_s,
        },
    }
    path = common.write_result_json(tdir, tool="omni_watch_full", aweme_id=aweme_id,
                                    params=params, output=output)
    (tdir / "answer.md").write_text(answer.text, encoding="utf-8")
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="omni_watch_full", aweme_id=aweme_id, status="ok",
        elapsed_s=answer.elapsed_s, input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens, vram_peak_gb=answer.vram_peak_gb,
        extra={"prompt_name": prompt_name, "fps": params["fps"], "clip": None,
               "model": p_cfg.get("omni", {}).get("model_path"), "frames_estimate": answer.frames_estimate},
    )
    logger.info("[omni_full %s] %.1fs tokens=%d/%d vram=%s → %s", aweme_id,
                answer.elapsed_s, answer.input_tokens, answer.output_tokens,
                answer.vram_peak_gb, path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="整段视频理解（Qwen3-Omni thinker-only）")
    common.add_common_cli_args(ap)
    ap.add_argument("--question", default=None, help="在 baseline 六小节之外追加的问题")
    ap.add_argument("--prompt-name", default="baseline_v1")
    ap.add_argument("--gpus", default=None, help="覆盖 config 的 visible_gpus，如 0,1")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_omni_full")

    gpus = args.gpus or cfg.perception.get("omni", {}).get("visible_gpus", "0,1")
    from src.perception.omni_runner import set_visible_gpus

    set_visible_gpus(gpus)  # 必须在 import torch 前

    try:
        aweme_id, video = common.resolve_target(cfg, args)
        path = run_for_video(cfg, aweme_id, force=args.force, question=args.question,
                             prompt_name=args.prompt_name,
                             video_override=video if args.video else None)
        common.emit_status_line("ok", output=str(path), gpus=gpus)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("omni_watch_full 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
