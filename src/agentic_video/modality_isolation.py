"""Media-level modality isolation and auditable request contracts.

Prompt wording is not an isolation boundary.  This module verifies the media
streams and the complete serialized request immediately before an inference
call, then records only hashes and policy results in the audit log.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file
from src.perception.common import run_ffmpeg, run_ffprobe_json

ISOLATION_SCHEMA = "modality_request_audit_v1"
MASK_SCHEMA = "visual_mask_v1"
CHANNELS = {"V", "A", "T", "FUSION", "AV"}

_ALLOWED_PAYLOAD_KEYS = {
    "V": {"request_id", "channel", "interval", "anonymous_subject_ids",
          "observation_dimensions", "sampling", "mask_artifact_sha"},
    "A": {"request_id", "channel", "interval", "audio_tasks"},
    "T": {"request_id", "channel", "interval", "text_region_refs",
          "text_tasks"},
    "FUSION": {"request_id", "channel", "claim_ids", "event_ids"},
    "AV": {"request_id", "channel", "interval", "claim_ids", "event_ids",
           "observation_dimensions", "sampling"},
}


class IsolationError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _stream_types(path: Path, *, ffprobe_bin: str = "ffprobe") -> list[str]:
    data = run_ffprobe_json(ffprobe_bin, Path(path))
    return [str(row.get("codec_type")) for row in data.get("streams") or []]


def _validate_normalized_region(region: dict[str, Any]) -> None:
    for key in ("x", "y", "width", "height"):
        value = float(region.get(key, -1))
        if value < 0 or value > 1:
            raise IsolationError("mask_region_invalid", key)
    if float(region["x"]) + float(region["width"]) > 1.000001:
        raise IsolationError("mask_region_invalid", "x+width")
    if float(region["y"]) + float(region["height"]) > 1.000001:
        raise IsolationError("mask_region_invalid", "y+height")
    interval = region.get("interval")
    if interval is not None and (len(interval) != 2 or
                                 float(interval[1]) <= float(interval[0])):
        raise IsolationError("mask_interval_invalid")


def _drawbox(region: dict[str, Any]) -> str:
    x = f"iw*{float(region['x']):.8f}"
    y = f"ih*{float(region['y']):.8f}"
    w = f"iw*{float(region['width']):.8f}"
    h = f"ih*{float(region['height']):.8f}"
    value = f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill"
    interval = region.get("interval")
    if interval is not None:
        value += (f":enable='between(t,{float(interval[0]):.6f},"
                  f"{float(interval[1]):.6f})'")
    return value


def create_visual_only_copy(
        source: Path, output_dir: Path, *, mask_regions: list[dict[str, Any]],
        mask_version: str = MASK_SCHEMA, ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Create a muted, text-masked visual artifact and verify its streams."""
    source = Path(source)
    output_dir = Path(output_dir)
    if not source.is_file():
        raise IsolationError("source_media_missing", str(source))
    for region in mask_regions:
        _validate_normalized_region(region)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "visual_only.mp4"
    filters = ",".join(_drawbox(row) for row in mask_regions)
    args = ["-y", "-loglevel", "error", "-i", str(source), "-map", "0:v:0"]
    if filters:
        args.extend(["-vf", filters])
    args.extend(["-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "18", "-movflags", "+faststart", str(destination)])
    run_ffmpeg(ffmpeg_bin, args, timeout_s=600)
    streams = _stream_types(destination, ffprobe_bin=ffprobe_bin)
    if "audio" in streams or streams.count("video") != 1:
        raise IsolationError("visual_media_contract_failed")
    audit = {
        "schema_version": MASK_SCHEMA,
        "mask_version": str(mask_version),
        "source_sha": sha256_file(source),
        "artifact_path": str(destination),
        "artifact_sha": sha256_file(destination),
        "streams": streams,
        "mask_regions": mask_regions,
        "timestamp_mapping": {
            "kind": "identity", "source_time_origin_s": 0.0,
            "artifact_time_origin_s": 0.0, "time_scale": 1.0},
        "coverage_policy": (
            "Pixels inside a mask are unavailable. A mask overlap cannot "
            "support absence or a negative action claim."),
    }
    (output_dir / "visual_mask_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def mask_overlaps(region: dict[str, Any], observation_box: dict[str, float],
                  interval: list[float] | None = None) -> bool:
    """Return whether a normalized mask overlaps a relevant observed region."""
    if interval is not None and region.get("interval") is not None:
        a, b = map(float, region["interval"])
        if b <= float(interval[0]) or float(interval[1]) <= a:
            return False
    ax1, ay1 = float(region["x"]), float(region["y"])
    ax2 = ax1 + float(region["width"])
    ay2 = ay1 + float(region["height"])
    bx1, by1 = float(observation_box["x"]), float(observation_box["y"])
    bx2 = bx1 + float(observation_box["width"])
    by2 = by1 + float(observation_box["height"])
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def audit_request(
        *, channel: str, prompt: str, payload: dict[str, Any],
        media_paths: Iterable[Path] = (), forbidden_markers: Iterable[str] = (),
        ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Validate a complete request and return a non-leaking audit record."""
    channel = str(channel).upper()
    if channel not in CHANNELS:
        raise IsolationError("unknown_channel", channel)
    if str(payload.get("channel") or "").upper() != channel:
        raise IsolationError("payload_channel_mismatch")
    extra = set(payload) - _ALLOWED_PAYLOAD_KEYS[channel]
    if extra:
        raise IsolationError("payload_field_forbidden", ",".join(sorted(extra)))
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    marker_hashes = []
    for marker in forbidden_markers:
        marker = str(marker).strip()
        if marker and marker.casefold() in (prompt + serialized).casefold():
            marker_hashes.append(json_hash(marker))
    if marker_hashes:
        # The error and log intentionally omit the source phrase.
        raise IsolationError("reference_specific_content_detected",
                             str(len(marker_hashes)))

    media = []
    for value in media_paths:
        path = Path(value)
        streams = _stream_types(path, ffprobe_bin=ffprobe_bin)
        media.append({"sha": sha256_file(path), "streams": streams})
    if channel == "V":
        if not media or any("audio" in row["streams"] for row in media):
            raise IsolationError("visual_request_contains_audio")
    elif channel == "A":
        if not media or any("video" in row["streams"] for row in media):
            raise IsolationError("audio_request_contains_visual")
    elif channel == "AV":
        if not media or any(
                "video" not in row["streams"] or "audio" not in row["streams"]
                for row in media):
            raise IsolationError("audiovisual_request_missing_stream")
    elif channel in {"T", "FUSION"} and media:
        raise IsolationError("non_media_channel_contains_media")

    subjects = payload.get("anonymous_subject_ids") or []
    if channel == "V" and any(not re.fullmatch(r"E\d+", str(item))
                              for item in subjects):
        raise IsolationError("visual_subject_not_anonymous")
    return {
        "schema_version": ISOLATION_SCHEMA,
        "request_id": payload.get("request_id"),
        "channel": channel,
        "prompt_sha": json_hash(prompt),
        "payload_sha": json_hash(payload),
        "payload_keys": sorted(payload),
        "media": media,
        "forbidden_marker_count": 0,
        "contract_passed": True,
    }


def write_request_audit(path: Path, audit: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def require_audio_only_capability(runner: Any) -> None:
    """Fail closed; a video-bearing fallback is not an audio-only call."""
    if not callable(getattr(runner, "inspect_audio", None)):
        raise IsolationError("audio_only_backend_unavailable")
