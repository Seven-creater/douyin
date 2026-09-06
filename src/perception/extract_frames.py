"""extract_frames：均匀抽帧（ffmpeg，无 GPU）。

CLI：python -m src.perception.extract_frames --aweme-id <id> [--fps 1.0] [--max-frames 64] [--force]
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def expected_frame_count(duration_s: float, fps: float, max_frames: int) -> tuple[int, float]:
    """返回 (期望帧数, effective_fps)。超过 max_frames 时降 fps 截断。"""
    n = math.ceil(duration_s * fps)
    if n <= max_frames:
        return n, fps
    effective = (max_frames - 0.5) / duration_s  # 保证不超上限
    return max_frames, effective


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    ffmpeg_bin: str,
    duration_s: float,
    fps: float = 1.0,
    max_frames: int = 64,
    jpeg_quality: int = 2,
) -> dict:
    n_expected, effective_fps = expected_frame_count(duration_s, fps, max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-i", str(video_path),
        "-vf", f"fps={effective_fps}",
        "-qscale:v", str(jpeg_quality),
        "-start_number", "0",
        str(out_dir / "frame_%06d.jpg"),
    ])
    files = sorted(out_dir.glob("frame_*.jpg"))
    # fps 滤镜对小数 fps 有量化（实测 0.685 → 实际 ~0.5），以实际产出反推：
    # 首帧 t=0、之后均匀分布是 fps 滤镜的确定行为 → t_s = index / actual_fps 自洽
    n_actual = len(files)
    if n_actual < min(n_expected, 4):  # 连 4 帧都没有基本是抽帧失败
        raise RuntimeError(f"抽帧异常：期望≈{n_expected} 实得 {n_actual}")
    actual_fps = (n_actual - 1) / duration_s if n_actual >= 2 and duration_s > 0 else 0.0
    frames = [
        {"index": i, "t_s": round(i / actual_fps, 3) if actual_fps else 0.0,
         "file": f.name, "size_bytes": f.stat().st_size}
        for i, f in enumerate(files)
    ]
    return {
        "requested_fps": fps,
        "fps": round(actual_fps, 4),
        "planned_fps": round(effective_fps, 4),
        "count": n_actual,
        "max_frames": max_frames,
        "frames": frames,
    }


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False,
                  fps: float | None = None, max_frames: int | None = None):
    p_cfg = common.perception_cfg(cfg)
    f_cfg = p_cfg.get("frames") or {}
    fps = float(fps if fps is not None else f_cfg.get("fps", 1.0))
    max_frames = int(max_frames if max_frames is not None else f_cfg.get("max_frames", 64))
    jpeg_quality = int(f_cfg.get("jpeg_quality", 2))
    ffmpeg_bin = p_cfg.get("ffmpeg_bin", "ffmpeg")

    video = common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "frames")
    params = {"fps": fps, "max_frames": max_frames, "jpeg_quality": jpeg_quality}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[frames %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    duration_s = common.video_duration_s(p_cfg.get("ffprobe_bin", "ffprobe"), video)
    t0 = time.time()
    output = extract_frames(video, tdir, ffmpeg_bin=ffmpeg_bin, duration_s=duration_s,
                            fps=fps, max_frames=max_frames, jpeg_quality=jpeg_quality)
    path = common.write_result_json(tdir, tool="extract_frames", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="extract_frames", aweme_id=aweme_id, status="ok",
        elapsed_s=round(time.time() - t0, 2),
        extra={"count": output["count"], "fps": output["fps"]},
    )
    logger.info("[frames %s] %d 帧 @%.3ffps → %s", aweme_id, output["count"], output["fps"], path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="均匀抽帧")
    common.add_common_cli_args(ap)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_frames")
    try:
        aweme_id, video = common.resolve_target(cfg, args)
        if args.video:
            out = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "frames")
            dur = common.video_duration_s(cfg.perception.get("ffprobe_bin", "ffprobe"), video)
            output = extract_frames(video, out, ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                                    duration_s=dur, fps=args.fps or 1.0,
                                    max_frames=args.max_frames or 64)
            common.emit_status_line("ok", aweme_id=aweme_id, count=output["count"])
        else:
            path = run_for_video(cfg, aweme_id, force=args.force, fps=args.fps, max_frames=args.max_frames)
            common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract_frames 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
