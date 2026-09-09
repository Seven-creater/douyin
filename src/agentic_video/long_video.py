"""Two-pass high-action indexing for long source videos."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path

from src.config import AppConfig, repo_root
from src.library.index_shots import build_shots, parse_scene_log
from src.perception import common

logger = logging.getLogger(__name__)


def resolve_source_video(cfg: AppConfig, source: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
    else:
        entry = ((cfg.library.get("sources") or {}).get(source) or {})
        raw = entry.get("video")
        if not raw:
            raise KeyError(f"library.sources.{source}.video is not configured")
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"source video does not exist: {path}")
    return path.resolve()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return float(ordered[index])


def robust_normalize(values: list[float]) -> list[float]:
    low, high = _percentile(values, 0.10), _percentile(values, 0.90)
    if high <= low + 1e-12:
        return [0.0 for _ in values]
    return [min(1.0, max(0.0, (float(value) - low) / (high - low))) for value in values]


def aggregate_action_windows(samples: list[dict], duration_s: float, *,
                             window_s: float = 45.0, stride_s: float = 15.0) -> list[dict]:
    """Aggregate sparse per-second features into overlapping candidate windows."""
    windows = []
    start = 0.0
    while start < duration_s:
        end = min(duration_s, start + window_s)
        selected = [row for row in samples if start <= float(row["t_s"]) < end]
        if selected:
            windows.append({
                "start_s": round(start, 3), "end_s": round(end, 3),
                "motion": sum(float(r.get("motion", 0)) for r in selected) / len(selected),
                "cut_density": sum(float(r.get("cut", 0)) for r in selected) / len(selected),
                "audio_energy": sum(float(r.get("audio_energy", 0)) for r in selected) / len(selected),
                "audio_onset": sum(float(r.get("audio_onset", 0)) for r in selected) / len(selected),
                "n_samples": len(selected),
            })
        if end >= duration_s:
            break
        start += stride_s
    for key in ("motion", "cut_density", "audio_energy", "audio_onset"):
        norm = robust_normalize([float(row[key]) for row in windows])
        for row, value in zip(windows, norm):
            row[f"{key}_norm"] = round(value, 6)
    for row in windows:
        row["action_score"] = round(
            0.35 * row["motion_norm"] + 0.25 * row["cut_density_norm"]
            + 0.20 * row["audio_energy_norm"] + 0.20 * row["audio_onset_norm"], 6)
    return windows


def _overlap_ratio(left: dict, right: dict) -> float:
    intersection = max(0.0, min(left["end_s"], right["end_s"])
                       - max(left["start_s"], right["start_s"]))
    shorter = min(left["end_s"] - left["start_s"], right["end_s"] - right["start_s"])
    return intersection / shorter if shorter > 0 else 0.0


def select_action_windows(windows: list[dict], duration_s: float, *, max_windows: int = 24,
                          partitions: int = 4, max_overlap: float = 0.20) -> list[dict]:
    """Score-ordered NMS with an equal per-partition cap for full-film diversity."""
    per_partition = max(1, math.ceil(max_windows / partitions))
    counts = [0] * partitions
    selected: list[dict] = []
    for row in sorted(windows, key=lambda item: (-item["action_score"], item["start_s"])):
        midpoint = (row["start_s"] + row["end_s"]) / 2
        partition = min(partitions - 1, int(midpoint / max(duration_s, 1) * partitions))
        if counts[partition] >= per_partition:
            continue
        if any(_overlap_ratio(row, old) > max_overlap for old in selected):
            continue
        chosen = dict(row)
        chosen["partition"] = partition
        chosen["rank"] = len(selected)
        selected.append(chosen)
        counts[partition] += 1
        if len(selected) >= max_windows:
            break
    selected.sort(key=lambda item: item["start_s"])
    for index, row in enumerate(selected):
        row["window_idx"] = index
    return selected


def scan_sparse_features(video: Path, *, ffmpeg_bin: str = "ffmpeg",
                         ffprobe_bin: str = "ffprobe", fps: float = 2.0,
                         width: int = 160, audio_sr: int = 8000,
                         cut_delta: float = 25.0) -> tuple[list[dict], float]:
    """Stream low-resolution frames and mono audio; never hold the movie in memory."""
    import numpy as np

    probe = common.run_ffprobe_json(ffprobe_bin, video)
    stream = next((s for s in probe.get("streams") or [] if s.get("codec_type") == "video"), {})
    src_w, src_h = int(stream.get("width") or 1920), int(stream.get("height") or 1080)
    height = max(2, int(round(width * src_h / max(1, src_w) / 2)) * 2)
    duration = float((probe.get("format") or {}).get("duration") or 0)
    proc = subprocess.Popen(
        [ffmpeg_bin, "-loglevel", "error", "-i", str(video), "-vf",
         f"fps={fps:g},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height
    samples: dict[int, dict] = {}
    previous = None
    index = 0
    while True:
        chunk = proc.stdout.read(frame_size)
        if len(chunk) != frame_size:
            break
        frame = np.frombuffer(chunk, np.uint8).reshape(height, width)
        diff = float(np.mean(np.abs(frame.astype(np.int16) - previous.astype(np.int16)))) \
            if previous is not None else 0.0
        second = int(index / fps)
        row = samples.setdefault(second, {"t_s": float(second), "motion": 0.0, "cut": 0.0,
                                          "audio_energy": 0.0, "audio_onset": 0.0})
        row["motion"] = max(row["motion"], diff)
        row["cut"] += float(diff >= cut_delta)
        previous = frame.copy()
        index += 1
    stderr = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"sparse video scan failed: {stderr[-300:]}")

    audio = subprocess.Popen(
        [ffmpeg_bin, "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(audio_sr), "-f", "f32le", "-"], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    previous_rms = 0.0
    second = 0
    bytes_per_second = audio_sr * 4
    while True:
        chunk = audio.stdout.read(bytes_per_second)
        if not chunk:
            break
        pcm = np.frombuffer(chunk, np.float32)
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        row = samples.setdefault(second, {"t_s": float(second), "motion": 0.0, "cut": 0.0,
                                          "audio_energy": 0.0, "audio_onset": 0.0})
        row["audio_energy"] = rms
        row["audio_onset"] = max(0.0, rms - previous_rms)
        previous_rms = rms
        second += 1
    audio_err = audio.stderr.read().decode(errors="replace")
    if audio.wait() != 0:
        raise RuntimeError(f"sparse audio scan failed: {audio_err[-300:]}")
    return [samples[key] for key in sorted(samples)], duration


def _window_boundaries(video: Path, start_s: float, duration_s: float, *,
                       ffmpeg_bin: str, threshold: float) -> list[float]:
    proc = subprocess.run(
        [ffmpeg_bin, "-ss", f"{start_s:g}", "-i", str(video), "-t", f"{duration_s:g}",
         "-filter:v", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, timeout=max(600, int(duration_s * 8)))
    if proc.returncode != 0:
        raise common.FFmpegError(f"window scene detection failed: {(proc.stderr or '')[-300:]}")
    boundaries = parse_scene_log(proc.stderr or "")
    if boundaries and max(boundaries) > duration_s + 1:
        boundaries = [max(0.0, value - start_s) for value in boundaries]
    return sorted(set(round(value, 3) for value in boundaries if 0 <= value < duration_s))


def index_selected_windows(cfg: AppConfig, source: str, video: Path, windows: list[dict], *,
                           force: bool = False) -> Path:
    shot_cfg = cfg.library.get("shots") or {}
    threshold = float(shot_cfg.get("threshold", 0.3))
    min_len = float(shot_cfg.get("min_shot_len_s", 0.4))
    stem = f"{source}__high_action"
    target = cfg.paths.library_dir / "shots" / stem
    result_path = target / "result.json"
    if result_path.exists() and not force:
        return result_path
    keyframes = target / "kf"
    keyframes.mkdir(parents=True, exist_ok=True)
    records = []
    for window in windows:
        start = float(window["start_s"])
        duration = float(window["end_s"]) - start
        boundaries = _window_boundaries(video, start, duration,
                                        ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                                        threshold=threshold)
        local_shots = build_shots(boundaries, duration, min_len_s=min_len, max_shots=100000)
        for shot in local_shots:
            original_start = start + float(shot["start_s"])
            original_end = start + float(shot["end_s"])
            shot_idx = len(records)
            fractions = (0.12, 0.50, 0.88)
            frame_paths = []
            for key_idx, fraction in enumerate(fractions):
                timestamp = original_start + (original_end - original_start) * fraction
                frame = keyframes / f"s{shot_idx:05d}_{key_idx}.jpg"
                if force or not frame.exists():
                    common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), [
                        "-y", "-loglevel", "error", "-ss", f"{timestamp:g}", "-i", str(video),
                        "-frames:v", "1", "-q:v", "3", str(frame)])
                frame_paths.append(str(frame))
            records.append({
                "video": str(video), "video_stem": stem, "source": source,
                "shot_idx": shot_idx, "window_idx": window["window_idx"],
                "start_s": round(original_start, 3), "end_s": round(original_end, 3),
                "duration_s": round(original_end - original_start, 3),
                "original_start_s": round(original_start, 3),
                "original_end_s": round(original_end, 3), "kf": frame_paths[1],
                "kfs": frame_paths, "action_score": window["action_score"],
                "motion_score": window["motion_norm"],
                "cut_density": window["cut_density_norm"],
                "audio_energy": window["audio_energy_norm"],
            })
    return common.write_result_json(
        target, tool="index_high_action_windows", aweme_id=stem,
        params={"source": source, "threshold": threshold, "min_len_s": min_len,
                "video_sha256": hashlib.sha256(str(video).encode()).hexdigest()},
        output={"source": source, "video": str(video), "n_windows": len(windows),
                "n_shots": len(records), "windows": windows, "shots": records})


def run_high_action_index(cfg: AppConfig, source: str, *, video: Path | None = None,
                          force: bool = False) -> Path:
    video = resolve_source_video(cfg, source, video)
    action_cfg = cfg.library.get("high_action") or {}
    output_dir = cfg.paths.library_dir / "sources" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = output_dir / "high_action_windows.json"
    if scan_path.exists() and not force:
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        windows = scan["selected_windows"]
    else:
        samples, duration = scan_sparse_features(
            video, ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            ffprobe_bin=cfg.perception.get("ffprobe_bin", "ffprobe"),
            fps=float(action_cfg.get("scan_fps", 2.0)),
            width=int(action_cfg.get("scan_width", 160)),
            audio_sr=int(action_cfg.get("audio_sr", 8000)))
        candidates = aggregate_action_windows(
            samples, duration, window_s=float(action_cfg.get("window_s", 45.0)),
            stride_s=float(action_cfg.get("stride_s", 15.0)))
        windows = select_action_windows(
            candidates, duration, max_windows=int(action_cfg.get("max_windows", 24)),
            partitions=int(action_cfg.get("partitions", 4)),
            max_overlap=float(action_cfg.get("max_overlap", 0.20)))
        scan_path.write_text(json.dumps({
            "source": source, "video": str(video), "duration_s": duration,
            "configuration": action_cfg, "selected_windows": windows,
            "candidate_count": len(candidates), "sample_count": len(samples),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_selected_windows(cfg, source, video, windows, force=force)
