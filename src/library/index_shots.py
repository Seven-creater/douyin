"""index_shots：素材库 B1——把 raw/ 下的视频切成镜头 + 抽关键帧。

CLI：python -m src.library.index_shots [--source trailers] [--force]
产物：data/library/shots/<video_stem>/result.json（信封）+ kf/ 关键帧 jpg
镜头记录：{video, start_s, end_s, duration_s, kf_path}（绝对秒，含片头黑帧过滤）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def parse_scene_log(stdout: str) -> list[float]:
    """解析 ffmpeg select=scene 的 pts_time 行 → 升序去重边界列表（含 0.0）。"""
    import re

    times = [float(m.group(1)) for m in re.finditer(r"pts_time:([\d.]+)", stdout)]
    times = sorted(set(round(t, 3) for t in times if t >= 0))
    return [0.0] + times


def build_shots(boundaries: list[float], duration_s: float, *,
                min_len_s: float, max_shots: int) -> list[dict]:
    """边界 → 镜头区间；过滤过短镜头；超量时按长度降序保留（长镜头信息量大）。"""
    shots = []
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else duration_s
        if end - b >= min_len_s:
            shots.append({"start_s": round(b, 2), "end_s": round(min(end, duration_s), 2)})
    if len(shots) > max_shots:
        shots = sorted(shots, key=lambda s: s["end_s"] - s["start_s"], reverse=True)[:max_shots]
        shots.sort(key=lambda s: s["start_s"])
    return shots


def index_video(cfg: AppConfig, video: Path, *, force: bool = False) -> Path | None:
    lib = cfg.library.get("shots") or {}
    threshold = float(lib.get("threshold", 0.3))
    min_len = float(lib.get("min_shot_len_s", 0.4))
    max_shots = int(lib.get("max_shots_per_video", 400))

    stem = video.stem
    tdir = cfg.paths.library_dir / "shots" / stem
    env = common.read_result_json(tdir)
    if env is not None and not force:
        logger.info("[shots %s] 已有产物，跳过", stem)
        return tdir / "result.json"

    ffprobe = cfg.perception.get("ffprobe_bin", "ffprobe")
    duration = common.video_duration_s(ffprobe, video)
    if not duration:
        logger.warning("[shots %s] ffprobe 无时长，跳过", stem)
        return None

    # scene 检测（detected_shots 同款 filter，log 级别更静）
    cmd = ["ffmpeg", "-i", str(video), "-filter:v",
           f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    boundaries = parse_scene_log(proc.stderr)
    shots = build_shots(boundaries, duration, min_len_s=min_len, max_shots=max_shots)

    # 每镜头中点抽 1 关键帧
    kf_dir = tdir / "kf"
    kf_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, s in enumerate(shots):
        mid = (s["start_s"] + s["end_s"]) / 2
        kf = kf_dir / f"s{i:04d}.jpg"
        if not kf.exists():
            common.run_ffmpeg("ffmpeg", [
                "-y", "-loglevel", "error", "-ss", f"{mid:g}", "-i", str(video),
                "-frames:v", "1", "-q:v", "3", str(kf)])
        records.append({"video": str(video), "video_stem": stem, "shot_idx": i,
                        "start_s": s["start_s"], "end_s": s["end_s"],
                        "duration_s": round(s["end_s"] - s["start_s"], 2),
                        "kf": str(kf)})
        if i % 50 == 0:
            logger.info("[shots %s] %d/%d", stem, i, len(shots))

    path = common.write_result_json(
        tdir, tool="index_shots", aweme_id=stem,
        params={"threshold": threshold, "min_len_s": min_len},
        output={"duration_s": duration, "n_shots": len(records), "shots": records})
    logger.info("[shots %s] %d 镜头 → %s", stem, len(records), path)
    common.emit_status_line("ok", video=stem, shots=len(records))
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="素材库镜头切分")
    ap.add_argument("--source", default="trailers", help="raw/ 下的子目录名")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_shots")

    raw_dir = cfg.paths.library_dir / "raw" / args.source
    videos = sorted(p for p in raw_dir.glob("*.mp4") if p.is_file())
    if not videos:
        logger.error("无视频：%s", raw_dir)
        return 1
    ok = 0
    for v in videos:
        if index_video(cfg, v, force=args.force):
            ok += 1
    logger.info("[shots] 完成 %d/%d 个视频", ok, len(videos))
    common.emit_status_line("ok", done=ok, total=len(videos))
    return 0 if ok == len(videos) else 1


if __name__ == "__main__":
    sys.exit(main())
