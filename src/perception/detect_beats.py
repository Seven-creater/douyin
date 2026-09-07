"""detect_beats：BGM 节拍点检测（librosa onset，CPU，无 GPU）。

CLI：python -m src.perception.detect_beats --aweme-id <id> [--force]
产物：data/perception/<aweme_id>/beats/result.json → beat_points_s（秒列表）
"""
from __future__ import annotations

import argparse
import io
import logging
import subprocess
import sys
import time
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def extract_audio_pcm(ffmpeg_bin: str, video_path: Path, *, sr: int = 16000):
    """ffmpeg 抽单声道 PCM（pipe，不落盘）→ (np.ndarray, sr)。"""
    import soundfile as sf

    proc = subprocess.run(
        [ffmpeg_bin, "-loglevel", "error", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "pipe:1"],
        capture_output=True, timeout=120,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise common.FFmpegError(f"音频抽取失败: {(proc.stderr or b'')[-200:]!r}")
    pcm, sr_actual = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return pcm, sr_actual


def postprocess_onsets(times: list[float], *, duration_s: float,
                       min_interval_s: float, max_beats: int) -> list[float]:
    """纯函数：越界剔除 → 最小间隔过滤 → 上限截断（保序）。"""
    cleaned = [round(t, 3) for t in times if 0.0 <= t <= duration_s]
    result: list[float] = []
    for t in cleaned:
        if result and t - result[-1] < min_interval_s:
            continue
        result.append(t)
        if len(result) >= max_beats:
            break
    return result


def detect_beats(pcm, sr: int, *, hop_ms: int = 50, min_interval_s: float = 0.3,
                 max_beats: int = 160) -> dict:
    import librosa
    import numpy as np

    hop_length = max(1, int(sr * hop_ms / 1000))
    onset_env = librosa.onset.onset_strength(y=np.asarray(pcm), sr=sr, hop_length=hop_length)
    times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        units="time", backtrack=True,
    )
    duration_s = float(len(pcm) / sr)
    beats = postprocess_onsets(list(times), duration_s=duration_s,
                               min_interval_s=min_interval_s, max_beats=max_beats)
    tempo = None
    try:
        t = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        tempo = float(t[0]) if not hasattr(t[0], "item") else float(t[0].item())
        tempo = round(tempo, 1)
    except Exception:  # noqa: BLE001 - tempo 仅参考
        pass
    return {
        "beat_points_s": beats,
        "n_beats": len(beats),
        "tempo_bpm_est": tempo,
        "method": "librosa_onset_strength",
        "hop_ms": hop_ms,
        "min_interval_s": min_interval_s,
    }


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False):
    p_cfg = common.perception_cfg(cfg)
    b_cfg = p_cfg.get("beats") or {}
    hop_ms = int(b_cfg.get("hop_ms", 50))
    min_interval_s = float(b_cfg.get("min_interval_s", 0.3))
    max_beats = int(b_cfg.get("max_beats", 160))
    sr = int(b_cfg.get("sr", 16000))

    video = common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "beats")
    params = {"sr": sr, "hop_ms": hop_ms, "min_interval_s": min_interval_s, "max_beats": max_beats}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[beats %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    t0 = time.time()
    pcm, sr_actual = extract_audio_pcm(p_cfg.get("ffmpeg_bin", "ffmpeg"), video, sr=sr)
    output = detect_beats(pcm, sr_actual, hop_ms=hop_ms,
                          min_interval_s=min_interval_s, max_beats=max_beats)
    path = common.write_result_json(tdir, tool="detect_beats", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="detect_beats", aweme_id=aweme_id, status="ok",
        elapsed_s=round(time.time() - t0, 2),
        extra={"n_beats": output["n_beats"], "tempo": output["tempo_bpm_est"]},
    )
    logger.info("[beats %s] %d 拍 tempo≈%s → %s", aweme_id, output["n_beats"],
                output["tempo_bpm_est"], path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="BGM 节拍检测（librosa）")
    common.add_common_cli_args(ap)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_beats")
    try:
        aweme_id, _ = common.resolve_target(cfg, args)
        path = run_for_video(cfg, aweme_id, force=args.force)
        common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("detect_beats 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
