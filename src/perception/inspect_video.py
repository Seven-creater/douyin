"""inspect_video：ffprobe 元数据 → 结构化 JSON（无 GPU）。

CLI：python -m src.perception.inspect_video --aweme-id <id> [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def _parse_fps(rate: str | None) -> float | None:
    """'30/1' → 30.0。"""
    if not rate or "/" not in rate:
        return float(rate) if rate else None
    num, den = rate.split("/", 1)
    try:
        den_f = float(den)
        return float(num) / den_f if den_f else None
    except ValueError:
        return None


def _rotation_deg(stream: dict) -> int:
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            r = int(sd["rotation"]) % 360
            return 360 - r if r < 0 else r  # ffprobe 旋转为逆时针负值约定
    return 0


def inspect_video(ffprobe_json: dict) -> dict:
    """纯解析（便于 mock 单测）。输入 run_ffprobe_json 的输出。"""
    vstream = astream = None
    for s in ffprobe_json.get("streams") or []:
        if s.get("codec_type") == "video" and vstream is None:
            vstream = s
        elif s.get("codec_type") == "audio" and astream is None:
            astream = s
    if vstream is None:
        raise ValueError("无视频流")
    fmt = ffprobe_json.get("format") or {}
    return {
        "duration_s": float(fmt.get("duration") or 0.0),
        "width": int(vstream.get("width") or 0),
        "height": int(vstream.get("height") or 0),
        "fps": _parse_fps(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate")),
        "video_codec": vstream.get("codec_name"),
        "has_audio": astream is not None,
        "audio_codec": astream.get("codec_name") if astream else None,
        "audio_sample_rate": int(astream["sample_rate"]) if astream and astream.get("sample_rate") else None,
        "audio_channels": int(astream["channels"]) if astream and astream.get("channels") else None,
        "bit_rate_bps": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
        "rotation": _rotation_deg(vstream),
    }


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False):
    p_cfg = common.perception_cfg(cfg)
    video = common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "inspect")
    params = {"ffprobe": p_cfg.get("ffprobe_bin", "ffprobe")}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[inspect %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    t0 = time.time()
    raw = common.run_ffprobe_json(p_cfg.get("ffprobe_bin", "ffprobe"), video)
    output = inspect_video(raw)
    path = common.write_result_json(tdir, tool="inspect_video", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="inspect_video", aweme_id=aweme_id, status="ok",
        elapsed_s=round(time.time() - t0, 2),
    )
    logger.info("[inspect %s] %ss %sx%s %sfps audio=%s → %s", aweme_id,
                output["duration_s"], output["width"], output["height"],
                output["fps"], output["has_audio"], path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="ffprobe 视频元数据")
    common.add_common_cli_args(ap)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_inspect")
    aweme_id, video = common.resolve_target(cfg, args)
    try:
        if args.video:
            output = inspect_video(common.run_ffprobe_json(
                cfg.perception.get("ffprobe_bin", "ffprobe"), video))
            common.emit_status_line("ok", aweme_id=aweme_id, **{k: output[k] for k in
                ("duration_s", "width", "height", "has_audio")})
        else:
            path = run_for_video(cfg, aweme_id, force=args.force)
            common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("inspect_video 失败")
        common.emit_status_line("error", aweme_id=aweme_id, error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
