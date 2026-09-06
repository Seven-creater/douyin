"""detect_shots：镜头切分（ffmpeg scene 阈值 + showinfo，无 GPU、零新依赖）。

CLI：python -m src.perception.detect_shots --aweme-id <id> [--threshold 0.3] [--force]
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)

_PTS_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def parse_showinfo_times(stderr: str) -> list[float]:
    """从 showinfo 的 stderr 里解析全部 pts_time（即切换帧时刻）。"""
    return [float(m) for m in _PTS_TIME_RE.findall(stderr)]


def boundaries_to_shots(boundaries: list[float], duration_s: float, *, min_shot_len_s: float) -> dict:
    cuts = sorted({round(b, 3) for b in boundaries if 0.05 < b < duration_s - 0.05})
    edges = [0.0, *cuts, duration_s]
    shots: list[dict] = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if end - start < min_shot_len_s and shots:
            shots[-1]["end_s"] = round(end, 3)             # 过短并入前一镜头
            shots[-1]["duration_s"] = round(end - shots[-1]["start_s"], 3)
            shots[-1]["mid_s"] = round((shots[-1]["start_s"] + end) / 2, 3)
            continue
        shots.append({
            "index": len(shots), "start_s": round(start, 3), "end_s": round(end, 3),
            "duration_s": round(end - start, 3), "mid_s": round((start + end) / 2, 3),
        })
    # boundaries 从最终 shots 推导（被合并的切点不再出现，两口径一致）
    final_edges = [s["start_s"] for s in shots] + [shots[-1]["end_s"] if shots else round(duration_s, 3)]
    return {"shot_count": len(shots), "boundaries_s": final_edges, "shots": shots}


def detect_shots(
    video_path: Path,
    *,
    ffmpeg_bin: str,
    threshold: float,
    min_shot_len_s: float,
    duration_s: float,
) -> dict:
    """解码丢弃，仅解析 showinfo。"""
    proc = subprocess.run(
        [ffmpeg_bin, "-i", str(video_path),
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise common.FFmpegError(f"ffmpeg 退出码 {proc.returncode}: {(proc.stderr or '')[-300:]}")
    cuts = parse_showinfo_times(proc.stderr or "")
    result = boundaries_to_shots(cuts, duration_s, min_shot_len_s=min_shot_len_s)
    result["threshold"] = threshold
    result["min_shot_len_s"] = min_shot_len_s
    return result


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False, threshold: float | None = None):
    p_cfg = common.perception_cfg(cfg)
    s_cfg = p_cfg.get("shots") or {}
    threshold = float(threshold if threshold is not None else s_cfg.get("threshold", 0.3))
    min_shot_len_s = float(s_cfg.get("min_shot_len_s", 0.5))

    video = common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "shots")
    params = {"threshold": threshold, "min_shot_len_s": min_shot_len_s}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[shots %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    duration_s = common.video_duration_s(p_cfg.get("ffprobe_bin", "ffprobe"), video)
    t0 = time.time()
    output = detect_shots(video, ffmpeg_bin=p_cfg.get("ffmpeg_bin", "ffmpeg"),
                          threshold=threshold, min_shot_len_s=min_shot_len_s, duration_s=duration_s)
    path = common.write_result_json(tdir, tool="detect_shots", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="detect_shots", aweme_id=aweme_id, status="ok",
        elapsed_s=round(time.time() - t0, 2),
        extra={"shot_count": output["shot_count"]},
    )
    logger.info("[shots %s] %d 个镜头 → %s", aweme_id, output["shot_count"], path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="镜头切分")
    common.add_common_cli_args(ap)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_shots")
    try:
        aweme_id, video = common.resolve_target(cfg, args)
        if args.video:
            dur = common.video_duration_s(cfg.perception.get("ffprobe_bin", "ffprobe"), video)
            out = detect_shots(video, ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                               threshold=args.threshold or 0.3, min_shot_len_s=0.5, duration_s=dur)
            common.emit_status_line("ok", aweme_id=aweme_id, shot_count=out["shot_count"])
        else:
            path = run_for_video(cfg, aweme_id, force=args.force, threshold=args.threshold)
            common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("detect_shots 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
