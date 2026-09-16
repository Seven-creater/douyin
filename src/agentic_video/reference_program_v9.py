"""V9 reference-first program induction with evidence-bound timing.

This module intentionally stops at three reviewed reference programs.  It does
not search a target movie, select target material, or render a new video.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from array import array
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.manifest import git_sha, json_hash
from src.agentic_video.recipe_v2 import sha256_file
from src.perception import common
from src.perception.detect_shots import detect_shots
from src.perception.inspect_video import inspect_video
from src.perception.omni_runner import cut_clip


V9_VERSION = "reference_program_v9_p0"
CONTENT_VERSION = "reference_content_program_v9_p0"
EDIT_VERSION = "reference_edit_program_v9_p0"
REQUIREMENTS_VERSION = "material_requirements_v9_p0"
CONTINUITY_LEVELS = {"required", "preferred", "not_required", "unknown"}
CONTINUITY_DIMENSIONS = (
    "subject", "opponent", "scene", "event", "actor_role", "spatial_orientation",
)
EVIDENCE_TYPES = {
    "observed_visual", "observed_textual_claim", "observed_audio_claim",
    "inferred", "unsupported",
}
PROBE_TYPES = {
    "native_frames", "dense_video", "ocr_context", "audio_asr_context",
    "beat_audio", "cross_modal_check",
}
COMPOSITION_MODES = {
    "continuous_clip", "micro_montage", "event_compression_montage",
    "evidence_montage", "dialogue_compression", "reaction_result_pair",
}
OPERATION_TYPES = {
    "hard_cut", "speed_change", "freeze", "text_overlay", "text_animation",
    "bgm", "transition", "crop_reframe", "other",
}
_TARGET_LEAK_TERMS = {"小黑", "罗小黑", "无限", "风息"}
_REFERENCE_LEAK_TERMS = {
    "跆拳道", "没有双手", "失去双手", "全国冠军", "taekwondo", "handless",
    "no hands", "national champion",
}
_TOOL_LEAK_TERMS = {"flashvid", "omni", "qwen", "yolo", "reid", "tracker",
                    "subject_harvest", "luoxiaohei"}
_HUMAN_SECTION_CHECKS = {
    "visible_content_correct", "event_phases_correct", "composition_mode_correct",
    "text_audio_rhythm_role_correct", "material_requirement_searchable",
}


class V9Blocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = ""):
        super().__init__(detail or reason_code)
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail or reason_code


def _write_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)
    return path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _snapshot_stage(output_dir: Path, stage: str, input_hash: str,
                    paths: list[Path]) -> dict[str, Any]:
    """Keep immutable, input-addressed copies of each P0 gate's evidence."""
    available = [path for path in paths if path.is_file()]
    revision = json_hash({"input": input_hash, "outputs": [
        [str(path.relative_to(output_dir)), sha256_file(path)] for path in available]})
    destination = output_dir / "stages" / stage / revision[:16]
    records = []
    for path in available:
        relative = path.relative_to(output_dir)
        copy = destination / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.is_file() and sha256_file(copy) != sha256_file(path):
            raise V9Blocked(stage, "stage_snapshot_conflict", str(copy))
        if not copy.is_file():
            shutil.copy2(path, copy)
        records.append({"source": str(path), "snapshot": str(copy),
                        "sha256": sha256_file(copy)})
    manifest = {"stage": stage, "input_sha256": input_hash,
                "revision_sha256": revision,
                "artifacts": records}
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _parse_one_object(raw: str, *, stage: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw or ""),
                     flags=re.DOTALL).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned:
        raise V9Blocked(stage, "model_response_not_one_json_object")
    try:
        value = json.loads(cleaned)
    except (TypeError, ValueError) as exc:
        raise V9Blocked(stage, "model_response_invalid_json", str(exc)) from exc
    if not isinstance(value, dict):
        raise V9Blocked(stage, "model_response_not_object")
    return value


def _answer_text(answer: Any) -> str:
    return str(getattr(answer, "text", answer) or "")


def _answer_audit(answer: Any) -> dict[str, Any]:
    return {
        key: getattr(answer, key, None)
        for key in (
            "input_tokens", "output_tokens", "elapsed_s", "input_build_s",
            "frames_estimate", "sampling", "gpu_pair",
        )
    }


def _native_frame_pts(ffprobe_bin: str, video: Path) -> list[dict[str, Any]]:
    command = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,pkt_duration_time,key_frame,pict_type",
        "-of", "json", str(video),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise common.FFmpegError(f"native frame PTS probe failed: {(proc.stderr or '')[-300:]}")
    rows = json.loads(proc.stdout).get("frames") or []
    frames = []
    for row in rows:
        raw_pts = row.get("best_effort_timestamp_time", row.get("pkt_pts_time"))
        if raw_pts is None:
            raise V9Blocked("evidence", "native_frame_pts_missing")
        pts = float(raw_pts)
        if not math.isfinite(pts):
            raise V9Blocked("evidence", "native_frame_pts_invalid")
        frames.append({
            "frame_id": f"f{len(frames):06d}",
            "pts_s": round(pts, 6),
            "duration_s": (round(float(row["pkt_duration_time"]), 6)
                           if row.get("pkt_duration_time") is not None else None),
            "key_frame": bool(int(row.get("key_frame") or 0)),
            "pict_type": row.get("pict_type"),
        })
    if not frames:
        raise V9Blocked("evidence", "native_frame_pts_empty")
    if any(right["pts_s"] < left["pts_s"]
           for left, right in zip(frames, frames[1:])):
        raise V9Blocked("evidence", "native_frame_pts_not_monotonic")
    return frames


def _nearest_frame(frames: list[dict[str, Any]], timestamp: float) -> dict[str, Any]:
    return min(frames, key=lambda row: abs(float(row["pts_s"]) - float(timestamp)))


def merge_cut_candidates(candidates: Iterable[dict[str, Any]],
                         frames: list[dict[str, Any]], *,
                         within_frames: int = 2) -> list[dict[str, Any]]:
    """Snap deterministic cut candidates to PTS and merge within N source frames."""
    frame_index = {row["frame_id"]: index for index, row in enumerate(frames)}
    snapped = []
    for row in candidates:
        frame = _nearest_frame(frames, float(row["time_s"]))
        snapped.append({
            "frame_id": frame["frame_id"], "pts_s": frame["pts_s"],
            "signals": [str(row.get("signal") or "scene")],
            "thresholds": ([float(row["threshold"])]
                           if row.get("threshold") is not None else []),
        })
    snapped.sort(key=lambda row: frame_index[row["frame_id"]])
    merged: list[dict[str, Any]] = []
    for row in snapped:
        if (merged and frame_index[row["frame_id"]] -
                frame_index[merged[-1]["frame_id"]] <= within_frames):
            merged[-1]["signals"] = sorted(set(merged[-1]["signals"] + row["signals"]))
            merged[-1]["thresholds"] = sorted(set(
                merged[-1]["thresholds"] + row["thresholds"]))
            continue
        merged.append(dict(row))
    for index, row in enumerate(merged, 1):
        row["boundary_id"] = f"cut_{index:03d}"
    return merged


def _luma_signals(ffmpeg_bin: str, video: Path, *, source_width: int,
                  source_height: int,
                  native_frames: list[dict[str, Any]]) -> dict[str, Any]:
    sample_width = min(160, max(2, source_width))
    sample_height = max(2, int(round(source_height * sample_width / source_width)))
    if sample_height % 2:
        sample_height += 1
    proc = subprocess.run([
        ffmpeg_bin, "-v", "error", "-i", str(video), "-an", "-vf",
        f"scale={sample_width}:{sample_height},format=gray",
        "-f", "rawvideo", "pipe:1",
    ], capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise common.FFmpegError(f"luma signal extraction failed: {(proc.stderr or b'')[-300:]!r}")
    frame_size = sample_width * sample_height
    count = len(proc.stdout) // frame_size
    raw = memoryview(proc.stdout)
    diffs: list[float] = []
    histogram_changes: list[float] = []
    intensity: list[float] = []
    previous = None
    for index in range(count):
        current = raw[index * frame_size:(index + 1) * frame_size]
        intensity.append(sum(current) / frame_size)
        if previous is not None:
            diffs.append(sum(abs(a - b) for a, b in zip(previous, current)) / frame_size)
            previous_hist = [0] * 16
            current_hist = [0] * 16
            for value in previous:
                previous_hist[value // 16] += 1
            for value in current:
                current_hist[value // 16] += 1
            histogram_changes.append(
                sum(abs(a - b) for a, b in zip(previous_hist, current_hist)) /
                (2 * frame_size))
        previous = current
    if abs(count - len(native_frames)) > 1:
        raise V9Blocked("evidence", "native_signal_frame_count_mismatch",
                        f"decoded={count} pts={len(native_frames)}")
    signal_count = min(count, len(native_frames))
    ranked = sorted(range(1, signal_count),
                    key=lambda idx: diffs[idx - 1], reverse=True)
    histogram_ranked = sorted(
        range(1, signal_count),
        key=lambda idx: histogram_changes[idx - 1], reverse=True)
    peak_count = min(12, max(1, count // 12)) if count else 0
    peaks = []
    for idx in sorted(ranked[:peak_count]):
        native = native_frames[idx]
        peaks.append({
            "frame_id": native["frame_id"], "pts_s": native["pts_s"],
            "mean_abs_luma_delta": round(diffs[idx - 1], 4),
            "histogram_l1": round(histogram_changes[idx - 1], 6),
        })
    change_candidates = []
    for signal, ranked_indices in (
            ("adjacent_luma_change", ranked[:peak_count]),
            ("histogram_change", histogram_ranked[:peak_count])):
        for idx in ranked_indices:
            native = native_frames[idx]
            change_candidates.append({
                "time_s": native["pts_s"], "signal": signal,
                "score": round(diffs[idx - 1], 4) if signal == "adjacent_luma_change"
                else round(histogram_changes[idx - 1], 6),
            })
    return {
        "analysis_scope": "all_native_frames", "sample_count": count,
        "sample_size": [sample_width, sample_height],
        "motion_peaks": peaks,
        "mean_luma": round(statistics.fmean(intensity), 4) if intensity else None,
        "max_luma_delta": round(max(diffs), 4) if diffs else None,
        "max_histogram_l1": (round(max(histogram_changes), 6)
                              if histogram_changes else None),
        "change_candidates": change_candidates,
    }


def _silence_events(ffmpeg_bin: str, video: Path) -> list[dict[str, Any]]:
    proc = subprocess.run([
        ffmpeg_bin, "-v", "info", "-i", str(video), "-af",
        "silencedetect=noise=-35dB:d=0.15", "-f", "null", "-",
    ], capture_output=True, text=True, timeout=180)
    text = proc.stderr or ""
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    return [{"start_s": round(start, 6), "end_s": round(end, 6)}
            for start, end in zip(starts, ends) if end >= start]


def _audio_intensity(ffmpeg_bin: str, video: Path, *, sample_rate: int = 16000,
                     window_ms: int = 50) -> dict[str, Any]:
    proc = subprocess.run([
        ffmpeg_bin, "-v", "error", "-i", str(video), "-vn", "-ac", "1",
        "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ], capture_output=True, timeout=180)
    if proc.returncode != 0:
        raise common.FFmpegError(f"audio intensity extraction failed: {(proc.stderr or b'')[-300:]!r}")
    samples = array("h")
    samples.frombytes(proc.stdout)
    window_size = max(1, round(sample_rate * window_ms / 1000))
    levels = []
    for start in range(0, len(samples), window_size):
        chunk = samples[start:start + window_size]
        if not chunk:
            continue
        rms = math.sqrt(sum(float(value) ** 2 for value in chunk) / len(chunk)) / 32768
        levels.append(rms)
    changes = [abs(right - left) for left, right in zip(levels, levels[1:])]
    ranked = sorted(range(1, len(levels)), key=lambda idx: changes[idx - 1], reverse=True)
    peaks = [{
        "pts_s": round(index * window_ms / 1000, 6),
        "normalized_rms_change": round(changes[index - 1], 6),
    } for index in sorted(ranked[:12])]
    return {
        "sample_rate": sample_rate, "window_ms": window_ms,
        "window_count": len(levels), "intensity_change_peaks": peaks,
    }


def _normalize_ocr_events(ocr: dict[str, Any],
                          frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(ocr.get("text_events") or [], 1):
        start = _nearest_frame(frames, float(row.get("t_start_s") or 0.0))
        end = _nearest_frame(frames, float(row.get("t_end_s") or start["pts_s"]))
        result.append({
            **row, "claim_id": f"ocr_{index:03d}",
            "evidence_type": "observed_textual_claim",
            "start_frame_id": start["frame_id"], "end_frame_id": end["frame_id"],
            "interval": [start["pts_s"], end["pts_s"]],
            "time_source": "nearest_native_frame_pts",
        })
    return result


def _read_cached_output(path: Path, *, reference_sha256: str,
                        tool_version: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not path.is_file():
        return None, {"status": "unavailable", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, {"status": "invalid", "path": str(path), "error": str(exc)}
    if not isinstance(payload, dict) or (payload.get("source_sha256") != reference_sha256 or
                                         payload.get("tool_version") != tool_version):
        return None, {"status": "unverified_source", "path": str(path),
                      "sha256": sha256_file(path)}
    output = payload.get("output")
    if not isinstance(output, dict):
        return None, {"status": "invalid", "path": str(path),
                      "error": "output must be an object"}
    return output, {"status": "reused_verified_artifact", "path": str(path),
                    "sha256": sha256_file(path), "source_sha256": reference_sha256,
                    "tool_version": tool_version}


def _evidence_cache_dir(reference: Path, repo: Path) -> Path:
    return repo / "data" / "agentic_narrative" / reference.parent.name / "evidence"


def _boundary_registry(duration_s: float, frames: list[dict[str, Any]],
                       cuts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [{"boundary_id": "video_start", "frame_id": frames[0]["frame_id"],
               "pts_s": frames[0]["pts_s"], "kind": "video_start"}]
    result.extend({"boundary_id": row["boundary_id"], "frame_id": row["frame_id"],
                   "pts_s": row["pts_s"], "kind": "cut_candidate"}
                  for row in cuts)
    result.append({"boundary_id": "video_end", "frame_id": frames[-1]["frame_id"],
                   "pts_s": round(duration_s, 6), "kind": "video_end"})
    return result


def build_reference_evidence_ledger(
        reference: Path, output_dir: Path, *, ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe", repo_root: Path | None = None,
        scene_thresholds: tuple[float, ...] = (0.08, 0.15, 0.25, 0.4),
        force: bool = False) -> dict[str, Any]:
    """Build deterministic, PTS-owned evidence for a short reference video."""
    reference = Path(reference).resolve()
    output_dir = Path(output_dir).resolve()
    output_path = output_dir / "reference_evidence.json"
    reference_sha = sha256_file(reference) if reference.is_file() else None
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    cache = _evidence_cache_dir(reference, root)
    tool_sources = {
        "ocr": root / "src" / "perception" / "ocr_frames.py",
        "transcribe": root / "src" / "perception" / "transcribe_audio.py",
        "beats": root / "src" / "perception" / "detect_beats.py",
    }
    evidence_input_sha = json_hash({
        "reference_sha256": reference_sha,
        "builder_sha256": sha256_file(Path(__file__)),
        "tool_sha256": {key: sha256_file(path)
                        for key, path in tool_sources.items()},
        "cache_sha256": {key: (sha256_file(cache / f"{key}.json")
                               if (cache / f"{key}.json").is_file() else None)
                         for key in tool_sources},
        "scene_thresholds": scene_thresholds,
        "ffmpeg_bin": ffmpeg_bin, "ffprobe_bin": ffprobe_bin,
    })
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if (cached.get("schema_version") == "reference_evidence_v9_p0" and
                (cached.get("reference") or {}).get("sha256") == reference_sha and
                cached.get("input_sha256") == evidence_input_sha):
            return cached
    if not reference.is_file():
        raise FileNotFoundError(reference)
    raw_probe = common.run_ffprobe_json(ffprobe_bin, reference)
    media = inspect_video(raw_probe)
    native_frames = _native_frame_pts(ffprobe_bin, reference)
    cut_rows = []
    per_threshold = []
    for threshold in scene_thresholds:
        detected = detect_shots(
            reference, ffmpeg_bin=ffmpeg_bin, threshold=threshold,
            min_shot_len_s=0.05, duration_s=float(media["duration_s"]))
        inner = [value for value in detected["boundaries_s"]
                 if 0.05 < float(value) < float(media["duration_s"]) - 0.05]
        per_threshold.append({"threshold": threshold, "cut_count": len(inner),
                              "boundaries_s": inner})
        cut_rows.extend({"time_s": value, "signal": "scene_change",
                         "threshold": threshold} for value in inner)
    signals = _luma_signals(
        ffmpeg_bin, reference, source_width=int(media["width"]),
        source_height=int(media["height"]),
        native_frames=native_frames)
    cut_rows.extend(signals["change_candidates"])
    merged_cuts = merge_cut_candidates(cut_rows, native_frames, within_frames=2)
    ocr, ocr_provenance = _read_cached_output(
        cache / "ocr.json", reference_sha256=reference_sha,
        tool_version=sha256_file(tool_sources["ocr"]))
    transcript, transcript_provenance = _read_cached_output(
        cache / "transcribe.json", reference_sha256=reference_sha,
        tool_version=sha256_file(tool_sources["transcribe"]))
    beats, beat_provenance = _read_cached_output(
        cache / "beats.json", reference_sha256=reference_sha,
        tool_version=sha256_file(tool_sources["beats"]))
    ledger = {
        "schema_version": "reference_evidence_v9_p0",
        "input_sha256": evidence_input_sha,
        "reference": {"path": str(reference), "sha256": reference_sha, **media},
        "timing_owner": "deterministic_native_pts",
        "native_frame_count": len(native_frames),
        "native_frames": native_frames,
        "scene_detection": {
            "thresholds": list(scene_thresholds), "per_threshold": per_threshold,
            "merge_policy": "within_two_native_frames", "cut_candidates": merged_cuts,
        },
        "boundaries": _boundary_registry(float(media["duration_s"]), native_frames,
                                          merged_cuts),
        "motion": signals,
        "ocr": {**(ocr or {"text_events": [], "full_text": ""}),
                "normalized_claim_events": _normalize_ocr_events(
                    ocr or {}, native_frames)},
        "ocr_provenance": ocr_provenance,
        "transcript": transcript or {"segments": [], "full_text": ""},
        "transcript_provenance": transcript_provenance,
        "audio": {
            "beat_analysis": beats or {"beat_points_s": []},
            "beat_provenance": beat_provenance,
            "silence_intervals": _silence_events(ffmpeg_bin, reference),
            "intensity": _audio_intensity(ffmpeg_bin, reference),
        },
        "vlm_frame_policy": {
            "native_frames_used_for_deterministic_analysis": len(native_frames),
            "native_frames_sent_in_one_global_vlm_call": 0,
            "global_semantic_watch_fps": 4.0,
            "flashvid_loaded": False,
        },
    }
    _write_json(output_path, ledger)
    return ledger


def _compact_evidence(ledger: dict[str, Any]) -> dict[str, Any]:
    ocr = ledger.get("ocr") or {}
    transcript = ledger.get("transcript") or {}
    audio = ledger.get("audio") or {}
    return {
        "reference_duration_s": (ledger.get("reference") or {}).get("duration_s"),
        "boundary_registry": ledger.get("boundaries") or [],
        "scene_cut_support": (ledger.get("scene_detection") or {}).get(
            "cut_candidates") or [],
        "motion_peaks": (ledger.get("motion") or {}).get("motion_peaks") or [],
        "ocr_text_events": ocr.get("normalized_claim_events") or [],
        "asr_segments": [
            {"segment_id": f"asr_{index:03d}", **row}
            for index, row in enumerate(transcript.get("segments") or [], 1)
        ],
        "asr_full_text": transcript.get("full_text") or "",
        "beat_points_s": (audio.get("beat_analysis") or {}).get("beat_points_s") or [],
        "silence_intervals": audio.get("silence_intervals") or [],
        "audio_intensity_change_peaks": (audio.get("intensity") or {}).get(
            "intensity_change_peaks") or [],
    }


GLOBAL_WATCH_PROMPT = """你是参考短视频理解 Agent。观看完整参考视频，结合下方确定性证据，
只输出一个 JSON 对象。不要套 Hook/Struggle/Reversal/Payoff 模板，不要猜秒数；时间只引用
boundary_registry 中已有 boundary_id。观察、文字/声音声明、推断必须分层。

输出：
{
  "core_expression_draft": {
    "reference_specific": "参考片实际表达",
    "transferable_structure": "去除人物身份后的抽象表达"
  },
  "reference_specific_terms": ["仅参考片成立、迁移需求中必须禁止的词"],
  "observations": [{
    "evidence_id": "model_ev_001",
    "evidence_type": "observed_visual|observed_textual_claim|observed_audio_claim|inferred|unsupported",
    "description": "具体描述",
    "start_boundary_id": "video_start或cut_...",
    "end_boundary_id": "cut_...或video_end",
    "supporting_deterministic_ids": []
  }],
  "meaningful_units": [{
    "unit_id": "unit_01", "purpose": "该视觉单元增加了什么新信息",
    "start_boundary_id": "...", "end_boundary_id": "...", "evidence_ids": []
  }],
  "section_drafts": [{
    "section_id": "section_01", "start_boundary_id": "...", "end_boundary_id": "...",
    "audience_takeaway": "", "content_function": "", "evidence_ids": [],
    "unit_ids": []
  }],
  "unresolved_questions": [{
    "id": "q_001", "question": "尚不确定的具体问题", "importance": "required|optional",
    "gap_type": "visual_detail|event_structure|text_claim|audio_claim|rhythm|cross_modal",
    "selected_probe": "native_frames|dense_video|ocr_context|audio_asr_context|beat_audio|cross_modal_check",
    "start_boundary_id": "...", "end_boundary_id": "...", "status": "open"
  }]
}

只描述实际可见、可听的信息变化；不预设人物、事件类型或段落功能。
OCR/ASR 的内容只能写成 claim，除非画面另有支持。
确定性证据：
"""


def _boundary_map(ledger: dict[str, Any]) -> dict[str, float]:
    return {str(row["boundary_id"]): float(row["pts_s"])
            for row in ledger.get("boundaries") or []}


def _require_boundary_pair(row: dict[str, Any], boundaries: dict[str, float],
                           *, stage: str) -> tuple[float, float]:
    start_id = str(row.get("start_boundary_id") or "")
    end_id = str(row.get("end_boundary_id") or "")
    if start_id not in boundaries or end_id not in boundaries:
        raise V9Blocked(stage, "unknown_boundary_reference",
                        f"{start_id!r} -> {end_id!r}")
    start, end = boundaries[start_id], boundaries[end_id]
    if end <= start:
        raise V9Blocked(stage, "invalid_boundary_order",
                        f"{start_id!r} -> {end_id!r}")
    return start, end


def _attach_intervals(value: dict[str, Any], ledger: dict[str, Any], *,
                      keys: tuple[str, ...], stage: str) -> dict[str, Any]:
    boundaries = _boundary_map(ledger)
    for key in keys:
        for row in value.get(key) or []:
            start, end = _require_boundary_pair(row, boundaries, stage=stage)
            row["interval"] = [round(start, 6), round(end, 6)]
            row["time_source"] = "deterministic_boundary_registry"
    return value


def build_reference_understanding_draft(
        reference: Path, ledger: dict[str, Any], output_dir: Path, *, runner,
        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_path = output_dir / "reference_understanding_draft.json"
    prompt = GLOBAL_WATCH_PROMPT + json.dumps(
        _compact_evidence(ledger), ensure_ascii=False, separators=(",", ":"))
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if (cached.get("reference_sha256") == ledger["reference"]["sha256"] and
                cached.get("evidence_ledger_sha256") == json_hash(ledger) and
                cached.get("prompt_sha256") == prompt_sha):
            return cached
    if runner is None:
        raise V9Blocked("global_watch", "runner_required")
    answer = runner.watch(
        Path(reference), prompt, duration_s=float(ledger["reference"]["duration_s"]),
        fps=4.0, max_new_tokens=4096, stop_after_json_object=True)
    raw = _answer_text(answer)
    raw_path = output_dir / "raw_responses" / "global_watch.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")
    draft = _parse_one_object(raw, stage="global_watch")
    _attach_intervals(
        draft, ledger,
        keys=("observations", "meaningful_units", "section_drafts",
              "unresolved_questions"), stage="global_watch")
    draft.update({
        "schema_version": "reference_understanding_draft_v9",
        "reference_sha256": ledger["reference"]["sha256"],
        "evidence_ledger_sha256": json_hash(ledger),
        "prompt_sha256": prompt_sha,
        "raw_response_sha256": sha256_file(raw_path),
        "model_audit": _answer_audit(answer),
        "global_watch": {"fps": 4.0, "flashvid_loaded": False,
                         "whole_reference": True},
    })
    _write_json(output_path, draft)
    return draft


SECTION_WATCH_PROMPT = """直接观看这一段参考原视频，只输出一个 JSON 对象。切点是候选信号，
不能仅凭切点数量推断剪辑模式。先说明每个可辨画面单元新增的信息，再解释镜头之间是否
属于同一事件；看不清的地方明确写成未决问题。时间只能引用输入中的 boundary_id。
{
  "section_id":"...",
  "shots":[{"start_boundary_id":"...","end_boundary_id":"...",
            "information_added":"画面具体新增信息","edit_function":"为何保留此画面",
            "event_relation":"same_event|different_event|uncertain",
            "supporting_deterministic_ids":[]}],
  "cut_assessments":[{"boundary_id":"...",
                      "status":"real_cut|not_cut|uncertain","reason":"画面依据"}],
  "rhythm_claimed":false,"rhythm_evidence_ids":[],
  "unresolved_questions":[{"id":"...","question":"...",
    "importance":"required|optional","gap_type":"visual_detail|event_structure|text_claim|audio_claim|rhythm|cross_modal",
    "selected_probe":"native_frames|dense_video|ocr_context|audio_asr_context|beat_audio|cross_modal_check",
    "status":"open"}]
}
输入："""


def _section_evidence(ledger: dict[str, Any], interval: list[float]) -> dict[str, Any]:
    start, end = map(float, interval)
    compact = _compact_evidence(ledger)
    cuts = [row for row in compact["scene_cut_support"]
            if start < float(row["pts_s"]) < end]
    ocr = [row for row in compact["ocr_text_events"]
           if float(row["interval"][0]) <= end and
           float(row["interval"][1]) >= start]
    def asr_time(row: dict[str, Any], key: str) -> float:
        if f"{key}_s" in row:
            return float(row[f"{key}_s"])
        return float(row.get(f"{key}_ms", 0)) / 1000

    asr = [row for row in compact["asr_segments"]
           if asr_time(row, "start") <= end and asr_time(row, "end") >= start]
    return {
        "interval": interval,
        "boundaries": [row for row in compact["boundary_registry"]
                       if start <= float(row["pts_s"]) <= end],
        "cut_candidates": cuts,
        "ocr_text_events": ocr,
        "asr_segments": asr,
        "beat_points_s": [value for value in compact["beat_points_s"]
                          if start <= float(value) <= end],
        "audio_intensity_change_peaks": [
            row for row in compact["audio_intensity_change_peaks"]
            if start <= float(row.get("pts_s", -1)) <= end],
    }


def build_section_observations(
        reference: Path, draft: dict[str, Any], ledger: dict[str, Any],
        output_dir: Path, *, runner, force: bool = False) -> dict[str, Any]:
    """Rewatch each draft Section in source media before compiling either Program."""
    output_dir = Path(output_dir)
    output_path = output_dir / "section_observations.json"
    sections = draft.get("section_drafts") or []
    if not sections:
        raise V9Blocked("section_watch", "draft_sections_missing")
    source_hash = ledger["reference"]["sha256"]
    input_hash = json_hash({"draft": draft, "evidence": _compact_evidence(ledger),
                            "prompt": SECTION_WATCH_PROMPT})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if (cached.get("reference_sha256") == source_hash and
                cached.get("input_sha256") == input_hash):
            return cached
    rows = []
    for section in sections:
        section_id = str(section.get("section_id") or "")
        interval = section.get("interval")
        if not section_id or not isinstance(interval, list) or len(interval) != 2:
            raise V9Blocked("section_watch", "section_interval_missing", section_id)
        evidence = _section_evidence(ledger, interval)
        prompt = SECTION_WATCH_PROMPT + json.dumps({
            "section_id": section_id,
            "draft": {key: section.get(key) for key in (
                "start_boundary_id", "end_boundary_id", "evidence_ids", "unit_ids")},
            "deterministic_evidence": evidence,
        }, ensure_ascii=False, separators=(",", ":"))
        answer = runner.watch(
            Path(reference), prompt, start_s=interval[0], end_s=interval[1],
            clip_dir=output_dir / "section_clips" / _safe_id(section_id),
            duration_s=interval[1] - interval[0], fps=4.0,
            use_audio_in_video=True, max_new_tokens=2048,
            stop_after_json_object=True)
        raw = _answer_text(answer)
        raw_path = output_dir / "raw_responses" / f"section_{_safe_id(section_id)}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        value = _parse_one_object(raw, stage="section_watch")
        if value.get("section_id") != section_id or not isinstance(value.get("shots"), list):
            raise V9Blocked("section_watch", "section_observation_invalid", section_id)
        if not value["shots"]:
            raise V9Blocked("section_watch", "section_shots_missing", section_id)
        _attach_intervals(value, ledger, keys=("shots",), stage="section_watch")
        for shot in value["shots"]:
            if (shot["interval"][0] < interval[0] - 1e-6 or
                    shot["interval"][1] > interval[1] + 1e-6):
                raise V9Blocked("section_watch", "shot_outside_section", section_id)
            if (not str(shot.get("information_added") or "").strip() or
                    not str(shot.get("edit_function") or "").strip() or
                    shot.get("event_relation") not in {
                        "same_event", "different_event", "uncertain"}):
                raise V9Blocked("section_watch", "shot_explanation_incomplete", section_id)
        valid_cuts = {row["boundary_id"] for row in evidence["cut_candidates"]}
        for assessment in value.get("cut_assessments") or []:
            if (assessment.get("boundary_id") not in valid_cuts or
                    assessment.get("status") not in {"real_cut", "not_cut", "uncertain"} or
                    not str(assessment.get("reason") or "").strip()):
                raise V9Blocked("section_watch", "cut_assessment_invalid", section_id)
        value.update({
            "source_interval": interval, "source_sha256": source_hash,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_response": str(raw_path), "raw_response_sha256": sha256_file(raw_path),
            "model_audit": _answer_audit(answer),
        })
        rows.append(value)
    result = {"schema_version": "section_observations_v9_p0",
              "reference_sha256": source_hash, "input_sha256": input_hash,
              "sections": rows}
    _write_json(output_path, result)
    return result


def collect_reference_gaps(draft: dict[str, Any], section_observations: dict[str, Any],
                           ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Ask only about signals that the current explanation leaves unaccounted for."""
    questions = [dict(row) for row in draft.get("unresolved_questions") or []]

    def add(question: dict[str, Any]) -> None:
        interval = question["interval"]
        for known in questions:
            prior = known.get("interval") or []
            if (known.get("gap_type") == question["gap_type"] and len(prior) == 2 and
                    max(float(prior[0]), float(interval[0])) <
                    min(float(prior[1]), float(interval[1]))):
                return
        questions.append(question)

    for section in section_observations.get("sections") or []:
        section_id = str(section["section_id"])
        interval = section["source_interval"]
        for index, row in enumerate(section.get("unresolved_questions") or [], 1):
            if not isinstance(row, dict) or not row.get("gap_type"):
                continue
            question = dict(row)
            question.setdefault("id", f"{section_id}_model_{index}")
            question.setdefault("importance", "optional")
            question.setdefault("status", "open")
            question.setdefault("interval", interval)
            add(question)
        evidence = _section_evidence(ledger, interval)
        cuts = evidence["cut_candidates"]
        assessments = {str(row.get("boundary_id")): row
                       for row in section.get("cut_assessments") or []}
        unexplained_cuts = [row["boundary_id"] for row in cuts
                            if assessments.get(row["boundary_id"], {}).get("status")
                            not in {"real_cut", "not_cut"}]
        shot_edges = {str(shot.get(key)) for shot in section.get("shots") or []
                      for key in ("start_boundary_id", "end_boundary_id")}
        unexplained_cuts.extend(
            cut_id for cut_id, assessment in assessments.items()
            if assessment.get("status") == "real_cut" and cut_id not in shot_edges
            and cut_id not in unexplained_cuts)
        if (unexplained_cuts or (len(cuts) >= 2 and not section.get("shots"))):
            add({"id": f"{section_id}_cut_structure",
                 "question": "这些候选切点分别对应什么画面信息，哪些是真切镜？",
                 "importance": "required", "gap_type": "event_structure",
                 "selected_probe": "dense_video", "status": "open",
                 "interval": interval, "gap_source": "signal_audit",
                 "trigger_ids": unexplained_cuts or
                                [row["boundary_id"] for row in cuts]})
        cited = {str(value) for shot in section.get("shots") or []
                 for value in shot.get("supporting_deterministic_ids") or []}
        if (len({str(row.get("text") or "") for row in evidence["ocr_text_events"]}) >= 2 and
                not any(row["claim_id"] in cited
                        for row in evidence["ocr_text_events"])):
            add({"id": f"{section_id}_text_role", "question": "变化的画面文字承担什么信息？",
                 "importance": "optional", "gap_type": "text_claim",
                 "selected_probe": "ocr_context", "status": "open",
                 "interval": interval, "gap_source": "signal_audit",
                 "trigger_ids": [row["claim_id"] for row in evidence["ocr_text_events"]]})
        if (any(str(row.get("text") or "").strip() for row in evidence["asr_segments"]) and
                not any(row["segment_id"] in cited for row in evidence["asr_segments"])):
            add({"id": f"{section_id}_audio_role", "question": "这段声音是否承担语义？",
                 "importance": "optional", "gap_type": "audio_claim",
                 "selected_probe": "audio_asr_context", "status": "open",
                 "interval": interval, "gap_source": "signal_audit"})
        if (section.get("rhythm_claimed") is True and
                evidence["beat_points_s"] and not section.get("rhythm_evidence_ids")):
            add({"id": f"{section_id}_rhythm", "question": "声称的切镜卡点有节拍依据吗？",
                 "importance": "optional", "gap_type": "rhythm",
                 "selected_probe": "beat_audio", "status": "open",
                 "interval": interval, "gap_source": "signal_audit"})
    return questions


PROBE_PROMPT = """你正在解决参考视频分析中的一个具体未决问题。只输出一个 JSON 对象：
{
  "question_id": "...", "status": "resolved|open",
  "answer": "只陈述当前输入支持的结论",
  "evidence": [{"evidence_type": "observed_visual|observed_textual_claim|observed_audio_claim|inferred|unsupported", "description": ""}],
  "new_questions": []
}
不要猜秒数，不要扩展到未问的问题；文字/声音声明不能自动升级为视觉事实。
问题："""

_GAP_PROBES = {
    "visual_detail": {"native_frames", "dense_video"},
    "event_structure": {"dense_video", "cross_modal_check"},
    "text_claim": {"ocr_context", "cross_modal_check"},
    "audio_claim": {"audio_asr_context", "cross_modal_check"},
    "rhythm": {"beat_audio", "cross_modal_check"},
    "cross_modal": {"cross_modal_check"},
}


def _validate_probe_selection(question: dict[str, Any]) -> None:
    gap_type = str(question.get("gap_type") or "")
    selected = str(question.get("selected_probe") or "")
    allowed = _GAP_PROBES.get(gap_type)
    if selected not in PROBE_TYPES or allowed is None or selected not in allowed:
        raise V9Blocked("probe", "probe_does_not_match_gap",
                        f"gap={gap_type!r} probe={selected!r}")


def _native_probe_images(reference: Path, interval: list[float], output: Path, *,
                         ffmpeg_bin: str, max_frames: int = 60) -> list[Path]:
    start, end = map(float, interval)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("f*.jpg"):
        stale.unlink()
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-loglevel", "error", "-ss", f"{start:g}", "-to", f"{end:g}",
        "-i", str(reference), "-vsync", "0", "-q:v", "2",
        str(output / "f%03d.jpg"),
    ], timeout_s=180)
    images = sorted(output.glob("f*.jpg"))
    if len(images) > max_frames:
        raise V9Blocked("probe", "native_frame_probe_too_wide",
                        f"{len(images)} frames exceeds {max_frames}")
    return images


def _ocr_probe_frames(reference: Path, events: list[dict[str, Any]],
                      native_frames: list[dict[str, Any]], output: Path, *,
                      ffmpeg_bin: str) -> tuple[list[Path], list[dict[str, Any]]]:
    """Pair each OCR claim with a source frame instead of hoping 4fps hits it."""
    if not events:
        raise V9Blocked("probe", "ocr_claim_frames_unavailable")
    if len(events) > 12:
        raise V9Blocked("probe", "ocr_probe_needs_narrower_interval")
    output.mkdir(parents=True, exist_ok=True)
    images = []
    refs = []
    for index, event in enumerate(events, 1):
        start, end = map(float, event["interval"])
        frame = _nearest_frame(native_frames, (start + end) / 2)
        image = output / f"ocr_{index:03d}.jpg"
        common.run_ffmpeg(ffmpeg_bin, [
            "-y", "-loglevel", "error", "-ss", f"{frame['pts_s']:.6f}",
            "-i", str(reference), "-frames:v", "1", "-q:v", "2", str(image),
        ], timeout_s=120)
        images.append(image)
        refs.append({"claim_id": event["claim_id"], "frame_id": frame["frame_id"],
                     "pts_s": frame["pts_s"], "path": str(image),
                     "sha256": sha256_file(image)})
    return images, refs


def _run_reference_probe(reference: Path, question: dict[str, Any],
                         ledger: dict[str, Any], output: Path, *,
                         runner, ffmpeg_bin: str) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = str(question.get("selected_probe") or "")
    _validate_probe_selection(question)
    interval = [float(value) for value in question["interval"]]
    section_evidence = _section_evidence(ledger, interval)
    provenance = {
        "ocr_context": ledger.get("ocr_provenance") or {},
        "audio_asr_context": ledger.get("transcript_provenance") or {},
        "beat_audio": (ledger.get("audio") or {}).get("beat_provenance") or {},
    }
    if (selected in provenance and provenance[selected].get("status") !=
            "reused_verified_artifact"):
        raise V9Blocked("probe", "probe_evidence_unavailable", selected)
    modality_fields = {
        "ocr_context": ("ocr_text_events", "boundaries"),
        "audio_asr_context": ("asr_segments", "boundaries"),
        "beat_audio": ("beat_points_s", "cut_candidates",
                       "audio_intensity_change_peaks"),
        "cross_modal_check": ("ocr_text_events", "asr_segments", "beat_points_s",
                              "cut_candidates", "boundaries"),
        "dense_video": ("cut_candidates", "boundaries"),
        "native_frames": ("boundaries",),
    }
    evidence = {key: section_evidence[key] for key in modality_fields[selected]}
    if selected == "audio_asr_context" and not evidence["asr_segments"]:
        raise V9Blocked("probe", "asr_segments_unavailable")
    if selected == "beat_audio" and not evidence["beat_points_s"]:
        raise V9Blocked("probe", "beat_points_unavailable")
    probe_input = {
        "question_id": question.get("id"), "question": question.get("question"),
        "probe_type": selected, "source_interval": interval,
        "source_sha256": ledger["reference"]["sha256"],
        "deterministic_evidence": evidence,
        "evidence_provenance": provenance.get(selected, {}),
    }
    ocr_images: list[Path] = []
    if selected == "ocr_context":
        ocr_images, probe_input["ocr_frames"] = _ocr_probe_frames(
            reference, evidence["ocr_text_events"], ledger["native_frames"],
            output / "ocr_frames", ffmpeg_bin=ffmpeg_bin)
    input_path = _write_json(output / "input.json", probe_input)
    prompt = PROBE_PROMPT + json.dumps(probe_input, ensure_ascii=False)
    if selected == "native_frames":
        images = _native_probe_images(
            reference, interval, output / "native_frames",
            ffmpeg_bin=ffmpeg_bin)
        if not images or not hasattr(runner, "inspect_media"):
            raise V9Blocked("probe", "native_frame_probe_unavailable")
        answer = runner.inspect_media(images, prompt, max_new_tokens=1536)
    elif selected == "ocr_context":
        if not hasattr(runner, "inspect_media"):
            raise V9Blocked("probe", "ocr_frame_probe_unavailable")
        clip = cut_clip(ffmpeg_bin, reference, output / "clip", start_s=interval[0],
                        end_s=interval[1])
        answer = runner.inspect_media(
            ocr_images, prompt, video_path=clip, fps=4.0,
            source_origin_s=interval[0], max_new_tokens=1536)
    else:
        fps = 12.0 if selected == "dense_video" else 4.0
        answer = runner.watch(
            reference, prompt, start_s=interval[0], end_s=interval[1],
            clip_dir=output / "clip", duration_s=interval[1] - interval[0],
            fps=fps, use_audio_in_video=True, max_new_tokens=1536,
            stop_after_json_object=True)
    raw = _answer_text(answer)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    result = _parse_one_object(raw, stage="probe")
    if str(result.get("question_id")) != str(question.get("id")):
        raise V9Blocked("probe", "probe_question_id_mismatch")
    if result.get("status") not in {"resolved", "open"}:
        raise V9Blocked("probe", "probe_status_invalid")
    for row in result.get("evidence") or []:
        if row.get("evidence_type") not in EVIDENCE_TYPES:
            raise V9Blocked("probe", "probe_evidence_type_invalid")
    if result["status"] == "resolved" and not result.get("evidence"):
        raise V9Blocked("probe", "probe_resolution_without_evidence")
    audit = {
        "probe": selected, "question_id": question.get("id"),
        "interval": interval, "raw_response": str(raw_path),
        "raw_response_sha256": sha256_file(raw_path),
        "input_path": str(input_path), "input_sha256": sha256_file(input_path),
        "evidence_fields": list(evidence), "evidence_provenance": provenance.get(selected),
        **_answer_audit(answer),
    }
    return result, audit


def resolve_reference_questions(
        reference: Path, draft: dict[str, Any], ledger: dict[str, Any],
        output_dir: Path, *, runner, ffmpeg_bin: str = "ffmpeg",
        max_rounds: int = 2, max_probes_per_round: int = 8,
        section_observations: dict[str, Any] | None = None,
        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_path = output_dir / "resolved_reference_understanding.json"
    questions = (collect_reference_gaps(draft, section_observations, ledger)
                 if section_observations is not None else
                 [dict(row) for row in draft.get("unresolved_questions") or []])
    input_hash = json_hash({"questions": questions, "ledger": json_hash(ledger),
                            "section_observations": section_observations})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    known_ids = {str(row.get("id")) for row in questions}
    probe_log: list[dict[str, Any]] = []
    for round_index in range(1, max_rounds + 1):
        pending = [row for row in questions if row.get("status") == "open"]
        if not pending:
            break
        for question in pending[:max_probes_per_round]:
            result, audit = _run_reference_probe(
                Path(reference), question, ledger,
                output_dir / "probes" / f"round_{round_index}" / str(question["id"]),
                runner=runner, ffmpeg_bin=ffmpeg_bin)
            question["status"] = result["status"]
            question["resolution"] = result.get("answer") or ""
            question["resolution_evidence"] = result.get("evidence") or []
            question["resolved_in_round"] = (round_index
                                                if result["status"] == "resolved" else None)
            probe_log.append({"round": round_index, **audit,
                              "result_status": result["status"]})
            for offset, new in enumerate(result.get("new_questions") or [], 1):
                if not isinstance(new, dict):
                    continue
                new_id = str(new.get("id") or f"{question['id']}_followup_{offset}")
                if new_id in known_ids:
                    continue
                new["id"] = new_id
                new.setdefault("importance", "optional")
                new.setdefault("status", "open")
                if "interval" not in new:
                    new["interval"] = list(question["interval"])
                questions.append(new)
                known_ids.add(new_id)
    required_open = [str(row.get("id")) for row in questions
                     if row.get("importance") == "required" and
                     row.get("status") != "resolved"]
    resolved = {
        "schema_version": "resolved_reference_understanding_v9",
        "draft_sha256": json_hash(draft), "input_sha256": input_hash,
        "questions": questions,
        "probe_history": probe_log, "required_unresolved_ids": required_open,
        "probe_budget": {"max_rounds": max_rounds,
                         "max_probes_per_round": max_probes_per_round},
    }
    _write_json(output_path, resolved)
    log_path = output_dir / "probe_log.jsonl"
    log_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in probe_log), encoding="utf-8")
    return resolved


CONTENT_PROGRAM_PROMPT = """根据参考理解初稿、确定性证据与 Probe 结果，生成 Content Program。
只输出一个 JSON 对象，不要写目标电影、目标角色或感知工具。所有 Section 时间只能引用给定
boundary_id；参考片事实和可迁移结构必须分层。
{
  "core_expression": {"reference_specific": "", "transferable_structure": ""},
  "reference_specific_terms": [],
  "perception_focus": {
    "primary_focus": {"type": "subject|event|theme|place|contrast|mixed"},
    "continuity_requirements": [], "important_evidence_types": [],
    "important_relations": []
  },
  "meaningful_units": [{"unit_id":"unit_01","purpose":"新增信息","start_boundary_id":"...","end_boundary_id":"...","evidence_ids":[]}],
  "sections": [{
    "section_id": "section_01", "start_boundary_id": "...", "end_boundary_id": "...",
    "reference_specific_fact": "", "transferable_structure": "",
    "audience_takeaway": "", "cognition_change": {"before":"","after":""},
    "content_function": "", "relation_to_previous": "", "evidence_ids": [],
    "unit_ids": [],
    "continuity": {
      "subject":"required|preferred|not_required|unknown",
      "opponent":"required|preferred|not_required|unknown",
      "scene":"required|preferred|not_required|unknown",
      "event":"required|preferred|not_required|unknown",
      "actor_role":"required|preferred|not_required|unknown",
      "spatial_orientation":"required|preferred|not_required|unknown"
    },
    "continuity_basis": {"subject":{"reason":"画面依据","evidence_ids":[]}}
  }]
}
每个连续性维度都要填写；unknown 表示尚不能判断，不得写成 not_required。
不要自动补固定故事槽；Section 必须来自实际画面信息变化。
输入："""


EDIT_PROGRAM_PROMPT = """根据 Content Program、确定性时间线和逐 Section 直接视频复看的观察生成 Edit Program。
只输出一个 JSON 对象。区分局部 operation 与可迁移 editorial pattern；不能猜秒数，操作边界
只能引用 boundary_id。不能只根据切点数量推断 montage；要引用画面新增信息与事件关系。
长事件可以先完整理解，再选多个不连续瞬间组成短 Section。
{
  "operations": [{
    "operation_id":"op_001", "section_id":"section_01",
    "operation_type":"hard_cut|speed_change|freeze|text_overlay|text_animation|bgm|transition|crop_reframe|other",
    "start_boundary_id":"...", "end_boundary_id":"...", "evidence_ids":[],
    "support_status":"supported|uncertain|unsupported", "purpose":""
  }],
  "editorial_patterns": [{
    "section_id":"section_01",
    "composition_mode":"continuous_clip|micro_montage|event_compression_montage|evidence_montage|dialogue_compression|reaction_result_pair",
    "duration_budget_s":0.0, "source_semantics":"one_long_event|multiple_events|dialogue|continuous_moment|paired_moments",
    "semantic_phases":[], "snippet_count_range":[1,1],
    "source_continuity":"continuous_required|non_contiguous_allowed",
    "semantic_continuity":"required", "ordering_constraint":"preserve_event_progression|claim_consistency|source_order|verified_relation",
    "individual_duration_policy":"minimum_sufficient_duration",
    "audience_requirement":"", "evidence_ids":[]
  }],
  "measured_style": {
    "reference_duration_s":0.0, "meaningful_unit_count":0,
    "section_duration_distribution":[], "shot_duration_distribution":[],
    "information_interval_distribution":[]
  }
}
不同事件不能被编造成单一事件因果；对白压缩不得改变原意。
输入："""


def _ask_object(runner: Any, prompt: str, output: Path, *, stage: str,
                max_new_tokens: int = 4096) -> tuple[dict[str, Any], dict[str, Any]]:
    answer = runner.ask(prompt, max_new_tokens=max_new_tokens)
    raw = _answer_text(answer)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw, encoding="utf-8")
    return _parse_one_object(raw, stage=stage), {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": str(output), "raw_response_sha256": sha256_file(output),
        "model_audit": _answer_audit(answer),
    }


def build_reference_content_program(
        draft: dict[str, Any], resolved: dict[str, Any], ledger: dict[str, Any],
        output_dir: Path, *, runner,
        section_observations: dict[str, Any] | None = None,
        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_path = output_dir / "reference_content_program.json"
    input_hash = json_hash({"draft": draft, "resolved": resolved,
                            "ledger": json_hash(ledger),
                            "section_observations": section_observations,
                            "prompt": CONTENT_PROGRAM_PROMPT})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    if resolved.get("required_unresolved_ids"):
        raise V9Blocked("content_program", "required_questions_unresolved",
                        ",".join(resolved["required_unresolved_ids"]))
    payload = {
        "deterministic_evidence": _compact_evidence(ledger),
        "draft": draft, "question_resolutions": resolved,
        "section_observations": section_observations or {},
    }
    value, audit = _ask_object(
        runner, CONTENT_PROGRAM_PROMPT + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")),
        output_dir / "raw_responses" / "content_program.txt",
        stage="content_program")
    _attach_intervals(value, ledger, keys=("meaningful_units", "sections"),
                      stage="content_program")
    value.update({
        "schema_version": CONTENT_VERSION,
        "input_sha256": input_hash,
        "reference_sha256": ledger["reference"]["sha256"],
        "evidence_ledger_sha256": json_hash(ledger),
        "reference_observations": draft.get("observations") or [],
        "unresolved_questions": resolved.get("questions") or [],
        "probe_history": resolved.get("probe_history") or [],
        "provenance": audit,
    })
    _write_json(output_path, value)
    return value


def _quantiles(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return round(float(p50), 6), round(float(ordered[p90_index]), 6)


def build_reference_edit_program(
        content: dict[str, Any], ledger: dict[str, Any], output_dir: Path, *,
        runner, section_observations: dict[str, Any] | None = None,
        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_path = output_dir / "reference_edit_program.json"
    if not section_observations or not section_observations.get("sections"):
        raise V9Blocked("edit_program", "direct_section_watch_required")
    input_hash = json_hash({"content": content, "ledger": json_hash(ledger),
                            "section_observations": section_observations,
                            "prompt": EDIT_PROGRAM_PROMPT})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    payload = {"content_program": content,
               "deterministic_evidence": _compact_evidence(ledger),
               "direct_section_watches": section_observations}
    value, audit = _ask_object(
        runner, EDIT_PROGRAM_PROMPT + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")),
        output_dir / "raw_responses" / "edit_program.txt", stage="edit_program")
    boundaries = _boundary_map(ledger)
    for operation in value.get("operations") or []:
        start_id = operation.get("start_boundary_id")
        end_id = operation.get("end_boundary_id")
        if start_id is not None or end_id is not None:
            start, end = _require_boundary_pair(operation, boundaries,
                                                stage="edit_program")
            operation["interval"] = [round(start, 6), round(end, 6)]
            operation["time_source"] = "deterministic_boundary_registry"
    section_durations = [float(row["interval"][1]) - float(row["interval"][0])
                         for row in content.get("sections") or []]
    cut_times = [float(row["pts_s"]) for row in
                 (ledger.get("scene_detection") or {}).get("cut_candidates") or []]
    duration = float(ledger["reference"]["duration_s"])
    shot_edges = [0.0, *cut_times, duration]
    shot_durations = [round(right - left, 6)
                      for left, right in zip(shot_edges, shot_edges[1:])]
    information_times = sorted({
        float(row["interval"][0]) for row in content.get("meaningful_units") or []
    })
    information_intervals = [round(right - left, 6)
                             for left, right in zip(information_times,
                                                    information_times[1:])]
    section_p50, section_p90 = _quantiles(section_durations)
    shot_p50, shot_p90 = _quantiles(shot_durations)
    value["measured_style"] = {
        "reference_duration_s": round(duration, 6),
        "meaningful_unit_count": len(content.get("meaningful_units") or []),
        "section_duration_distribution": section_durations,
        "section_duration_p50_s": section_p50,
        "section_duration_p90_s": section_p90,
        "shot_duration_distribution": shot_durations,
        "shot_duration_p50_s": shot_p50,
        "shot_duration_p90_s": shot_p90,
        "information_interval_distribution": information_intervals,
        "time_source": "deterministic_native_pts",
    }
    value.update({
        "schema_version": EDIT_VERSION,
        "input_sha256": input_hash,
        "direct_section_watch_sha256": json_hash(section_observations),
        "reference_sha256": ledger["reference"]["sha256"],
        "content_program_sha256": json_hash(content), "provenance": audit,
    })
    _write_json(output_path, value)
    return value


def _semantic_phases(pattern: dict[str, Any]) -> list[str]:
    phases = [str(value).strip() for value in pattern.get("semantic_phases") or []
              if str(value).strip()]
    return phases or ["minimum_sufficient_meaning"]


def compile_material_requirements(content: dict[str, Any], edit: dict[str, Any],
                                  output_dir: Path) -> dict[str, Any]:
    """Compile source-agnostic requirements; never copy reference-specific facts."""
    patterns = {str(row.get("section_id")): row
                for row in edit.get("editorial_patterns") or []}
    requirements = []
    for section in content.get("sections") or []:
        section_id = str(section.get("section_id") or "")
        pattern = patterns.get(section_id) or {}
        mode = str(pattern.get("composition_mode") or "continuous_clip")
        phases = _semantic_phases(pattern)
        requirement = {
            "requirement_id": f"req_{len(requirements) + 1:02d}",
            "section_id": section_id,
            "semantic_requirement": {
                "meaning_to_prove": str(section.get("transferable_structure") or
                                        section.get("audience_takeaway") or ""),
                "required_event_understanding": (
                    "understand_full_source_event_before_selecting_moments"
                    if mode in {"event_compression_montage", "dialogue_compression",
                                "reaction_result_pair"}
                    else "understand_each_source_evidence_in_context"),
                "required_semantic_phases": phases,
            },
            "presentation_requirement": {
                "target_duration_s": float(pattern.get("duration_budget_s") or
                                           (section["interval"][1] -
                                            section["interval"][0])),
                "composition_mode": mode,
                "snippet_count_range": pattern.get("snippet_count_range") or [1, 1],
                "individual_duration_policy": "minimum_sufficient_duration",
                "source_continuity": pattern.get("source_continuity") or
                                     "continuous_required",
                "semantic_continuity": pattern.get("semantic_continuity") or
                                       "required",
                "ordering_constraint": pattern.get("ordering_constraint") or
                                       "source_order",
                "audience_requirement": pattern.get("audience_requirement") or
                                        "blind_viewer_understands_section_meaning",
            },
            "continuity_requirement": {
                "levels": {key: (section.get("continuity") or {}).get(key)
                           for key in CONTINUITY_DIMENSIONS},
                "basis_evidence_ids": {
                    key: list(((section.get("continuity_basis") or {}).get(key) or
                               {}).get("evidence_ids") or [])
                    for key in CONTINUITY_DIMENSIONS},
            },
            "evidence_requirement": {
                "minimum_sufficient_evidence_set": phases,
                "disallowed_substitutes": [
                    "result_without_required_context",
                    "long_explanation_without_required_evidence",
                    "unverified_claim_presented_as_visual_fact",
                    "filler_used_only_to_reach_duration",
                ],
            },
            "traceability": {
                "content_section_id": section_id,
                "edit_pattern_section_id": (section_id if pattern else None),
                "content_evidence_ids": section.get("evidence_ids") or [],
            },
        }
        if mode == "dialogue_compression":
            requirement["evidence_requirement"]["dialogue_integrity"] = {
                "complete_utterances_required": True,
                "meaning_and_causality_must_not_change": True,
                "synthetic_statement_from_unrelated_sentences_disallowed": True,
                "audio_bridge_allowed": True,
            }
        requirements.append(requirement)
    result = {
        "schema_version": REQUIREMENTS_VERSION,
        "source_agnostic": True,
        "content_program_sha256": json_hash(content),
        "edit_program_sha256": json_hash(edit),
        "requirements": requirements,
    }
    _write_json(Path(output_dir) / "material_requirements.json", result)
    return result


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _validate_evidence_layering(content: dict[str, Any], errors: list[str]) -> None:
    for index, row in enumerate(content.get("reference_observations") or []):
        evidence_type = row.get("evidence_type")
        if evidence_type not in EVIDENCE_TYPES:
            errors.append(f"reference_observations[{index}].evidence_type invalid")
        description = str(row.get("description") or "").strip()
        if not description:
            errors.append(f"reference_observations[{index}].description empty")
        if (evidence_type in {"observed_textual_claim", "observed_audio_claim"} and
                row.get("verified_as_visual_fact") is True):
            errors.append(f"reference_observations[{index}] claim upgraded to visual fact")


def _validate_content(content: dict[str, Any], ledger: dict[str, Any],
                      errors: list[str], warnings: list[str]) -> None:
    if content.get("schema_version") != CONTENT_VERSION:
        errors.append("content schema_version invalid")
    core = content.get("core_expression")
    if not isinstance(core, dict) or not str(core.get("transferable_structure") or "").strip():
        errors.append("content core_expression.transferable_structure missing")
    focus = content.get("perception_focus")
    if not isinstance(focus, dict) or not isinstance(focus.get("primary_focus"), dict):
        errors.append("content perception_focus missing")
    elif (not (focus.get("important_evidence_types") or []) or
          not (focus.get("important_relations") or [])):
        errors.append("content perception_focus is not evidence-operational")
    sections = content.get("sections")
    if not isinstance(sections, list) or not sections:
        errors.append("content sections missing")
        sections = []
    boundaries = _boundary_map(ledger)
    section_ids: set[str] = set()
    known_evidence = {
        str(row.get("evidence_id")) for row in content.get("reference_observations") or []
        if row.get("evidence_id")
    }
    known_evidence.update(boundaries)
    known_units = {str(row.get("unit_id")) for row in content.get("meaningful_units") or []
                   if row.get("unit_id")}
    assigned_units: set[str] = set()
    for index, section in enumerate(sections):
        section_id = str(section.get("section_id") or "")
        if not section_id or section_id in section_ids:
            errors.append(f"content sections[{index}].section_id invalid")
        section_ids.add(section_id)
        try:
            expected = _require_boundary_pair(section, boundaries, stage="validation")
        except V9Blocked as exc:
            errors.append(f"content sections[{index}] {exc.reason_code}")
            continue
        interval = section.get("interval")
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(abs(float(actual) - expected[pos]) > 1e-6
                    for pos, actual in enumerate(interval))):
            errors.append(f"content sections[{index}] interval not derived from boundary PTS")
        cognition = section.get("cognition_change")
        if (not isinstance(cognition, dict) or
                not str(cognition.get("before") or "").strip() or
                not str(cognition.get("after") or "").strip()):
            errors.append(f"content sections[{index}].cognition_change incomplete")
        if not str(section.get("audience_takeaway") or "").strip():
            errors.append(f"content sections[{index}].audience_takeaway empty")
        if not str(section.get("transferable_structure") or "").strip():
            errors.append(f"content sections[{index}].transferable_structure empty")
        if not (section.get("evidence_ids") or []):
            errors.append(f"content sections[{index}].evidence_ids empty")
        for evidence_id in section.get("evidence_ids") or []:
            if str(evidence_id) not in known_evidence:
                errors.append(f"content sections[{index}] unknown evidence {evidence_id}")
        for unit_id in section.get("unit_ids") or []:
            if str(unit_id) not in known_units:
                errors.append(f"content sections[{index}] unknown unit {unit_id}")
            assigned_units.add(str(unit_id))
        levels = section.get("continuity") or {}
        basis = section.get("continuity_basis") or {}
        for dimension in CONTINUITY_DIMENSIONS:
            level = levels.get(dimension)
            if level not in CONTINUITY_LEVELS:
                errors.append(f"content sections[{index}].continuity.{dimension} invalid")
                continue
            support = basis.get(dimension) or {}
            if not str(support.get("reason") or "").strip():
                errors.append(f"content sections[{index}].continuity_basis.{dimension} reason missing")
            ids = support.get("evidence_ids") or []
            if level in {"required", "preferred"} and not ids:
                errors.append(f"content sections[{index}].continuity_basis.{dimension} evidence missing")
            for evidence_id in ids:
                if str(evidence_id) not in known_evidence:
                    errors.append(f"content sections[{index}] unknown continuity evidence {evidence_id}")
            if level == "unknown":
                warnings.append(f"content sections[{index}].continuity.{dimension} unknown")
    missing_units = known_units - assigned_units
    if missing_units:
        errors.append(f"meaningful units swallowed by coarse sections: {sorted(missing_units)}")
    if len(sections) <= 2 and len(known_units) >= 4:
        errors.append("two-window collapse: meaningful units were not structurally separated")
    unresolved = [str(row.get("id")) for row in content.get("unresolved_questions") or []
                  if row.get("importance") == "required" and row.get("status") != "resolved"]
    if unresolved:
        errors.append(f"required unresolved questions remain: {unresolved}")
    _validate_evidence_layering(content, errors)
    duration = float((ledger.get("reference") or {}).get("duration_s") or 0.0)
    if sections:
        covered_end = max(float(row.get("interval", [0.0, 0.0])[1]) for row in sections)
        if duration - covered_end > 0.1:
            warnings.append("content sections do not describe the reference tail")


def _validate_edit(edit: dict[str, Any], content: dict[str, Any],
                   ledger: dict[str, Any], errors: list[str]) -> None:
    if edit.get("schema_version") != EDIT_VERSION:
        errors.append("edit schema_version invalid")
    section_ids = {str(row.get("section_id")) for row in content.get("sections") or []}
    covered_sections: set[str] = set()
    boundaries = _boundary_map(ledger)
    for index, operation in enumerate(edit.get("operations") or []):
        operation_type = operation.get("operation_type")
        if operation_type not in OPERATION_TYPES:
            errors.append(f"operations[{index}].operation_type invalid")
        section_id = str(operation.get("section_id") or "")
        if section_id not in section_ids:
            errors.append(f"operations[{index}] unknown section")
        covered_sections.add(section_id)
        if operation.get("start_boundary_id") is not None:
            try:
                expected = _require_boundary_pair(operation, boundaries,
                                                  stage="validation")
            except V9Blocked as exc:
                errors.append(f"operations[{index}] {exc.reason_code}")
            else:
                interval = operation.get("interval") or []
                if len(interval) != 2 or any(
                        abs(float(interval[pos]) - expected[pos]) > 1e-6
                        for pos in (0, 1)):
                    errors.append(f"operations[{index}] time is not deterministic PTS")
    for index, pattern in enumerate(edit.get("editorial_patterns") or []):
        section_id = str(pattern.get("section_id") or "")
        if section_id not in section_ids:
            errors.append(f"editorial_patterns[{index}] unknown section")
        covered_sections.add(section_id)
        mode = pattern.get("composition_mode")
        if mode not in COMPOSITION_MODES:
            errors.append(f"editorial_patterns[{index}].composition_mode invalid")
        count_range = pattern.get("snippet_count_range")
        if (not isinstance(count_range, list) or len(count_range) != 2 or
                not all(isinstance(value, int) and value >= 1 for value in count_range) or
                count_range[1] < count_range[0]):
            errors.append(f"editorial_patterns[{index}].snippet_count_range invalid")
        if mode == "event_compression_montage":
            if pattern.get("source_continuity") != "non_contiguous_allowed":
                errors.append(f"editorial_patterns[{index}] event compression must allow non-contiguous source")
            if len(pattern.get("semantic_phases") or []) < 2:
                errors.append(f"editorial_patterns[{index}] event compression lacks phases")
            if pattern.get("ordering_constraint") != "preserve_event_progression":
                errors.append(f"editorial_patterns[{index}] event progression not preserved")
        if mode == "evidence_montage" and pattern.get("source_semantics") == "one_long_event":
            errors.append(f"editorial_patterns[{index}] evidence montage fabricates one event")
        if mode == "dialogue_compression" and pattern.get("semantic_continuity") != "required":
            errors.append(f"editorial_patterns[{index}] dialogue continuity not required")
    missing = section_ids - covered_sections
    if missing:
        errors.append(f"edit sections lack operation or editorial pattern: {sorted(missing)}")
    style = edit.get("measured_style") or {}
    if style.get("time_source") != "deterministic_native_pts":
        errors.append("measured style does not use deterministic native PTS")
    if int(style.get("meaningful_unit_count") or 0) != len(
            content.get("meaningful_units") or []):
        errors.append("measured style meaningful unit count mismatch")
    if (edit.get("content_program_sha256") is not None and
            edit["content_program_sha256"] != json_hash(content)):
        errors.append("edit content program hash mismatch")


def _validate_requirements(requirements: dict[str, Any], content: dict[str, Any],
                           edit: dict[str, Any], errors: list[str]) -> None:
    if requirements.get("schema_version") != REQUIREMENTS_VERSION:
        errors.append("requirements schema_version invalid")
    rows = requirements.get("requirements")
    if not isinstance(rows, list) or not rows:
        errors.append("requirements missing")
        return
    content_by_section = {str(row.get("section_id")): row
                          for row in content.get("sections") or []}
    content_sections = set(content_by_section)
    forbidden = set(_TARGET_LEAK_TERMS) | set(_REFERENCE_LEAK_TERMS) | set(_TOOL_LEAK_TERMS)
    forbidden.update(str(value).strip() for value in
                     content.get("reference_specific_terms") or [] if str(value).strip())
    text = "\n".join(_all_strings(rows)).lower()
    leaked = sorted(term for term in forbidden if term and term.lower() in text)
    if leaked:
        errors.append(f"reference or target fact leaked into requirements: {leaked}")
    for index, row in enumerate(rows):
        for key in ("semantic_requirement", "presentation_requirement",
                    "continuity_requirement", "evidence_requirement"):
            if not isinstance(row.get(key), dict) or not row[key]:
                errors.append(f"requirements[{index}].{key} missing")
        if not str((row.get("semantic_requirement") or {}).get(
                "meaning_to_prove") or "").strip():
            errors.append(f"requirements[{index}].meaning_to_prove empty")
        section_id = str(row.get("section_id") or "")
        if section_id not in content_sections:
            errors.append(f"requirements[{index}] unknown content section")
        trace = row.get("traceability") or {}
        if trace.get("content_section_id") != section_id:
            errors.append(f"requirements[{index}] content trace broken")
        mode = (row.get("presentation_requirement") or {}).get("composition_mode")
        if mode not in COMPOSITION_MODES:
            errors.append(f"requirements[{index}] composition_mode invalid")
        continuity = (row.get("continuity_requirement") or {}).get("levels") or {}
        if set(continuity) != set(CONTINUITY_DIMENSIONS) or any(
                value not in CONTINUITY_LEVELS for value in continuity.values()):
            errors.append(f"requirements[{index}] continuity levels invalid")
        elif continuity != (content_by_section.get(section_id) or {}).get("continuity"):
            errors.append(f"requirements[{index}] continuity changed during compilation")
        if mode == "evidence_montage" and continuity.get("event") == "required":
            errors.append(f"requirements[{index}] evidence montage invents same-event causality")
        if (mode in {"event_compression_montage", "reaction_result_pair"} and
                continuity.get("event") == "not_required"):
            errors.append(f"requirements[{index}] event continuity contradicts composition")
        if mode == "dialogue_compression":
            dialogue = (row.get("evidence_requirement") or {}).get("dialogue_integrity")
            if not isinstance(dialogue, dict) or not all((
                    dialogue.get("complete_utterances_required"),
                    dialogue.get("meaning_and_causality_must_not_change"),
                    dialogue.get("synthetic_statement_from_unrelated_sentences_disallowed"))):
                errors.append(f"requirements[{index}] dialogue integrity incomplete")
    pattern_sections = {str(row.get("section_id"))
                        for row in edit.get("editorial_patterns") or []}
    for index, row in enumerate(rows):
        trace = row.get("traceability") or {}
        expected = str(row.get("section_id")) if str(row.get("section_id")) in pattern_sections else None
        if trace.get("edit_pattern_section_id") != expected:
            errors.append(f"requirements[{index}] edit trace broken")


def validate_reference_programs(content: dict[str, Any], edit: dict[str, Any],
                                requirements: dict[str, Any],
                                ledger: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    _validate_content(content, ledger, errors, warnings)
    _validate_edit(edit, content, ledger, errors)
    _validate_requirements(requirements, content, edit, errors)
    return {
        "schema_version": "reference_program_validation_v9",
        "passed": not errors, "errors": errors, "warnings": warnings,
    }


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "section"


def _build_review_assets(reference: Path, content: dict[str, Any], output_dir: Path,
                         *, ffmpeg_bin: str) -> list[dict[str, Any]]:
    assets_dir = Path(output_dir) / "review_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for section in content.get("sections") or []:
        section_id = str(section["section_id"])
        start, end = map(float, section["interval"])
        name = _safe_id(section_id)
        clip = assets_dir / f"{name}.mp4"
        common.run_ffmpeg(ffmpeg_bin, [
            "-y", "-loglevel", "error", "-ss", f"{start:g}", "-to", f"{end:g}",
            "-i", str(reference), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-ac", "2", "-c:a", "aac", str(clip),
        ], timeout_s=180)
        keyframes = []
        for index, ratio in enumerate((0.15, 0.5, 0.85), 1):
            frame = assets_dir / f"{name}_keyframe_{index}.jpg"
            timestamp = start + (end - start) * ratio
            common.run_ffmpeg(ffmpeg_bin, [
                "-y", "-loglevel", "error", "-ss", f"{timestamp:g}", "-i",
                str(reference), "-frames:v", "1", "-q:v", "2", str(frame),
            ], timeout_s=120)
            keyframes.append({"path": str(frame), "source_time_s": round(timestamp, 6),
                              "sha256": sha256_file(frame)})
        rows.append({
            "section_id": section_id, "interval": [start, end],
            "clip": str(clip), "keyframes": keyframes,
            "clip_sha256": sha256_file(clip),
        })
    _write_json(assets_dir / "manifest.json", {"sections": rows})
    return rows


def _write_review_pack(output_dir: Path, content: dict[str, Any],
                       edit: dict[str, Any], requirements: dict[str, Any],
                       assets: list[dict[str, Any]]) -> None:
    patterns = {str(row.get("section_id")): row
                for row in edit.get("editorial_patterns") or []}
    reqs = {str(row.get("section_id")): row
            for row in requirements.get("requirements") or []}
    assets_by_id = {str(row.get("section_id")): row for row in assets}
    lines = [
        "# V9 Reference Program 人工审核表", "",
        "自动验证通过不等于交付通过。请逐段观看片段与关键帧，再填写 human_review.template.json。",
        "",
    ]
    review_rows = []
    for section in content.get("sections") or []:
        section_id = str(section["section_id"])
        pattern = patterns.get(section_id) or {}
        asset = assets_by_id.get(section_id) or {}
        lines.extend([
            f"## {section_id}", "",
            f"- 区间：{section.get('interval')}",
            f"- 观众所得：{section.get('audience_takeaway', '')}",
            f"- 认知变化：{json.dumps(section.get('cognition_change') or {}, ensure_ascii=False)}",
            f"- 连续性：{json.dumps(section.get('continuity') or {}, ensure_ascii=False)}",
            f"- 连续性依据：{json.dumps(section.get('continuity_basis') or {}, ensure_ascii=False)}",
            f"- Composition：{pattern.get('composition_mode', 'missing')}",
            f"- 片段：{asset.get('clip', 'missing')}",
            f"- 关键帧：{json.dumps(asset.get('keyframes') or [], ensure_ascii=False)}",
            f"- 匿名素材需求：{json.dumps(reqs.get(section_id) or {}, ensure_ascii=False)}",
            "",
        ])
        review_rows.append({
            "section_id": section_id, "passed": None,
            "checks": {
                "visible_content_correct": None,
                "event_phases_correct": None,
                "composition_mode_correct": None,
                "text_audio_rhythm_role_correct": None,
                "material_requirement_searchable": None,
            },
            "notes": "",
        })
    lines.extend([
        "## Requirement Transfer Test", "",
        "只阅读匿名化 material_requirements.json。在不知道参考人物身份的情况下，",
        "是否仍能明确知道每个 Section 应寻找和验证什么素材？", "",
    ])
    Path(output_dir, "review_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(Path(output_dir) / "human_review.template.json", {
        "schema_version": "reference_program_human_review_v9",
        "reviewer": "", "sections": review_rows,
        "continuity_resolutions": [],
        "requirement_transfer_test": {"passed": None, "notes": ""},
        "overall_notes": "",
    })


def _write_acceptance(output_dir: Path, *, automated_passed: bool,
                      decision: str, failure_stage: str | None = None,
                      reason_code: str | None = None,
                      validation: dict[str, Any] | None = None,
                      human_passed: bool | None = None,
                      detail: str = "", failure_class: str | None = None) -> dict[str, Any]:
    acceptance = {
        "schema_version": "reference_program_acceptance_v9",
        "automated_passed": bool(automated_passed),
        "human_passed": human_passed,
        "passed": bool(automated_passed and human_passed is True),
        "decision": decision,
        "delivery": "passed" if automated_passed and human_passed is True else "blocked",
        "failure_class": failure_class,
        "failure_stage": failure_stage,
        "reason_code": reason_code,
        "detail": detail,
        "validation": validation,
        "long_video_perception_authorized": bool(
            automated_passed and human_passed is True),
    }
    _write_json(Path(output_dir) / "acceptance.json", acceptance)
    return acceptance


def accept_reference_programs(output_dir: Path, human_review: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    automatic = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    if not automatic.get("automated_passed"):
        raise V9Blocked("human_acceptance", "automatic_validation_not_passed")
    review = json.loads(Path(human_review).read_text(encoding="utf-8"))
    content = json.loads((output_dir / "reference_content_program.json").read_text(
        encoding="utf-8"))
    amendments = review.get("continuity_resolutions") or []
    amendment_errors = []
    amended = False
    updated_validation = automatic.get("validation")
    if amendments:
        if not str(review.get("reviewer") or "").strip():
            amendment_errors.append("continuity_reviewer_missing")
        before_hash = sha256_file(output_dir / "reference_content_program.json")
        by_id = {str(row.get("section_id")): row
                 for row in content.get("sections") or []}
        seen_amendments = set()
        for amendment in amendments:
            section_id = str(amendment.get("section_id") or "")
            dimension = str(amendment.get("dimension") or "")
            key = (section_id, dimension)
            section = by_id.get(section_id)
            level = amendment.get("level")
            if (key in seen_amendments or section is None or
                    dimension not in CONTINUITY_DIMENSIONS or
                    (section.get("continuity") or {}).get(dimension) != "unknown" or
                    level not in CONTINUITY_LEVELS - {"unknown"} or
                    not str(amendment.get("reason") or "").strip()):
                amendment_errors.append(f"invalid_continuity_resolution:{section_id}/{dimension}")
                continue
            seen_amendments.add(key)
            section["continuity"][dimension] = level
            section.setdefault("continuity_basis", {})[dimension] = {
                "reason": str(amendment["reason"]),
                "evidence_ids": list(amendment.get("evidence_ids") or []),
                "reviewed_by": str(review.get("reviewer") or ""),
            }
        content["human_amendments"] = {
            "model_content_sha256": before_hash,
            "human_review_sha256": sha256_file(Path(human_review)),
            "resolutions": amendments,
        }
        if not amendment_errors:
            edit = json.loads((output_dir / "reference_edit_program.json").read_text(
                encoding="utf-8"))
            edit["content_program_sha256"] = json_hash(content)
            ledger = json.loads((output_dir / "reference_evidence.json").read_text(
                encoding="utf-8"))
            proposed = compile_material_requirements(
                content, edit, output_dir / "human_review_draft")
            checked = validate_reference_programs(content, edit, proposed, ledger)
            if checked["passed"]:
                _write_json(output_dir / "reference_content_program.json", content)
                _write_json(output_dir / "reference_edit_program.json", edit)
                _write_json(output_dir / "material_requirements.json", proposed)
                _write_json(output_dir / "validation.json", checked)
                updated_validation = checked
                amended = True
            else:
                amendment_errors.extend(checked["errors"])
    expected = {str(row["section_id"]) for row in content.get("sections") or []}
    rows = review.get("sections") or []
    reviewed = {str(row.get("section_id")): row for row in rows}
    failures = list(amendment_errors)
    if not str(review.get("reviewer") or "").strip():
        failures.append("human_reviewer_missing")
    if not amendment_errors:
        for section in content.get("sections") or []:
            for dimension, level in (section.get("continuity") or {}).items():
                if level == "unknown":
                    failures.append(
                        f"continuity_unknown:{section['section_id']}/{dimension}")
    for section_id in sorted(expected):
        row = reviewed.get(section_id)
        if not row or row.get("passed") is not True:
            failures.append(f"section_not_passed:{section_id}")
            continue
        checks = row.get("checks") or {}
        if set(checks) != _HUMAN_SECTION_CHECKS or any(
                checks.get(key) is not True for key in _HUMAN_SECTION_CHECKS):
            failures.append(f"section_check_not_passed:{section_id}")
    transfer = review.get("requirement_transfer_test") or {}
    if transfer.get("passed") is not True:
        failures.append("requirement_transfer_test_failed")
    if amended:
        failures.append("continuity_amendment_requires_rereview")
    passed = not failures
    review_copy = output_dir / "human_review.json"
    review_copy.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    acceptance = _write_acceptance(
        output_dir, automated_passed=True,
        decision="PASS" if passed else "BLOCKED",
        failure_stage=None if passed else "human",
        reason_code=None if passed else "required_human_review_failed",
        validation=updated_validation, human_passed=passed,
        detail=";".join(failures))
    if passed:
        frozen = {name: sha256_file(output_dir / name) for name in (
            "reference_content_program.json", "reference_edit_program.json",
            "material_requirements.json")}
        _write_json(output_dir / "frozen_program_hashes.json", {
            "schema_version": "frozen_reference_programs_v9",
            "created_at": _now(), "program_sha256": frozen,
            "human_review_sha256": sha256_file(review_copy),
        })
        acceptance["frozen_program_sha256"] = frozen
        _write_json(output_dir / "acceptance.json", acceptance)
    return acceptance


def run_reference_program_v9(cfg: Any, reference: Path, output_dir: Path, *,
                             runner, force: bool = False) -> dict[str, Any]:
    """Run Evidence -> Gap -> Probe -> three Programs; stop pending human review."""
    reference = Path(reference).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "frozen_program_hashes.json").exists():
        raise V9Blocked("input", "frozen_run_is_immutable",
                        "Choose a new output directory for another reference analysis")
    ffmpeg_bin = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    ffprobe_bin = cfg.perception.get("ffprobe_bin", "ffprobe")
    manifest = {
        "schema_version": "reference_program_run_manifest_v9",
        "program_version": V9_VERSION,
        "created_at": _now(), "git_sha": git_sha(Path(__file__).resolve().parents[2]),
        "reference": str(reference), "reference_sha256": sha256_file(reference),
        "scope": {
            "reference_only": True, "target_movie_search": False,
            "character_selection": False, "target_material_planning": False,
            "rendering": False, "flashvid": False,
        },
        "prompt_hashes": {
            "global_watch": hashlib.sha256(GLOBAL_WATCH_PROMPT.encode()).hexdigest(),
            "section_watch": hashlib.sha256(SECTION_WATCH_PROMPT.encode()).hexdigest(),
            "probe": hashlib.sha256(PROBE_PROMPT.encode()).hexdigest(),
            "content_program": hashlib.sha256(CONTENT_PROGRAM_PROMPT.encode()).hexdigest(),
            "edit_program": hashlib.sha256(EDIT_PROGRAM_PROMPT.encode()).hexdigest(),
        },
        "stages": {},
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    validation: dict[str, Any] | None = None
    try:
        ledger = build_reference_evidence_ledger(
            reference, output_dir, ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin, force=force)
        manifest["stages"]["evidence"] = {
            "status": "complete", "native_frame_count": ledger["native_frame_count"],
            "sha256": json_hash(ledger),
        }
        draft = build_reference_understanding_draft(
            reference, ledger, output_dir, runner=runner, force=force)
        manifest["stages"]["global_watch"] = {"status": "complete",
                                                  "sha256": json_hash(draft)}
        _snapshot_stage(output_dir, "p0_a", json_hash({
            "reference": ledger["reference"]["sha256"],
            "draft_input": draft["prompt_sha256"],
            "ledger": json_hash(ledger),
        }), [output_dir / "reference_evidence.json",
             output_dir / "reference_understanding_draft.json",
             output_dir / "raw_responses" / "global_watch.txt"])
        section_observations = build_section_observations(
            reference, draft, ledger, output_dir, runner=runner, force=force)
        manifest["stages"]["section_watch"] = {
            "status": "complete", "sha256": json_hash(section_observations)}
        _snapshot_stage(output_dir, "p0_b", section_observations["input_sha256"],
                        [output_dir / "section_observations.json", *sorted(
                            (output_dir / "raw_responses").glob("section_*.txt"))])
        resolved = resolve_reference_questions(
            reference, draft, ledger, output_dir, runner=runner,
            ffmpeg_bin=ffmpeg_bin, section_observations=section_observations,
            force=force)
        manifest["stages"]["probes"] = {
            "status": ("blocked" if resolved["required_unresolved_ids"] else "complete"),
            "probe_count": len(resolved["probe_history"]),
            "required_unresolved_ids": resolved["required_unresolved_ids"],
        }
        _snapshot_stage(output_dir, "p0_c", resolved["input_sha256"],
                        [output_dir / "resolved_reference_understanding.json",
                         output_dir / "probe_log.jsonl", *sorted(
                             (output_dir / "probes").rglob("raw.txt")), *sorted(
                             (output_dir / "probes").rglob("input.json"))])
        if resolved["required_unresolved_ids"]:
            raise V9Blocked("probe", "required_questions_unresolved",
                            ",".join(resolved["required_unresolved_ids"]))
        content = build_reference_content_program(
            draft, resolved, ledger, output_dir, runner=runner,
            section_observations=section_observations, force=force)
        edit = build_reference_edit_program(
            content, ledger, output_dir, runner=runner,
            section_observations=section_observations, force=force)
        requirements = compile_material_requirements(content, edit, output_dir)
        validation = validate_reference_programs(content, edit, requirements, ledger)
        _write_json(output_dir / "validation.json", validation)
        manifest["stages"]["programs"] = {
            "status": "complete" if validation["passed"] else "blocked",
            "content_sha256": sha256_file(output_dir / "reference_content_program.json"),
            "edit_sha256": sha256_file(output_dir / "reference_edit_program.json"),
            "requirements_sha256": sha256_file(output_dir / "material_requirements.json"),
        }
        _snapshot_stage(output_dir, "p0_d", json_hash({
            "content": json_hash(content), "edit": json_hash(edit),
            "requirements": json_hash(requirements),
        }), [output_dir / "reference_content_program.json",
             output_dir / "reference_edit_program.json",
             output_dir / "material_requirements.json",
             output_dir / "validation.json",
             output_dir / "raw_responses" / "content_program.txt",
             output_dir / "raw_responses" / "edit_program.txt"])
        if not validation["passed"]:
            raise V9Blocked("validation", "reference_program_contract_failed",
                            "; ".join(validation["errors"]))
        assets = _build_review_assets(reference, content, output_dir,
                                      ffmpeg_bin=ffmpeg_bin)
        _write_review_pack(output_dir, content, edit, requirements, assets)
        acceptance = _write_acceptance(
            output_dir, automated_passed=True, decision="PENDING_HUMAN",
            reason_code="human_section_and_transfer_review_required",
            validation=validation, human_passed=None)
        manifest["stages"]["human_review"] = {"status": "pending"}
        _write_json(output_dir / "run_manifest.json", manifest)
        return {"output": str(output_dir), **acceptance,
                "section_count": len(content.get("sections") or [])}
    except V9Blocked as exc:
        manifest["stages"].setdefault(exc.stage, {})["status"] = "blocked"
        manifest["stages"][exc.stage].update({
            "reason_code": exc.reason_code, "detail": exc.detail,
        })
        _write_json(output_dir / "run_manifest.json", manifest)
        acceptance = _write_acceptance(
            output_dir, automated_passed=False, decision="BLOCKED",
            failure_stage=exc.stage, reason_code=exc.reason_code,
            detail=exc.detail, failure_class="verification",
            validation=validation)
        return {"output": str(output_dir), **acceptance}
    except Exception as exc:  # preserve a stable blocked artifact for runner/FFmpeg failures
        detail = f"{type(exc).__name__}: {exc}"
        manifest["stages"]["infrastructure"] = {
            "status": "blocked", "reason_code": "reference_runtime_failure",
            "detail": detail,
        }
        _write_json(output_dir / "run_manifest.json", manifest)
        acceptance = _write_acceptance(
            output_dir, automated_passed=False, decision="BLOCKED",
            failure_stage="infrastructure", reason_code="reference_runtime_failure",
            detail=detail, failure_class="infrastructure")
        return {"output": str(output_dir), **acceptance}
