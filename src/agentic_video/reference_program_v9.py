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
from src.agentic_video import narrative_boundary
from src.agentic_video.recipe_v2 import sha256_file
from src.perception import common
from src.perception.detect_shots import detect_shots
from src.perception.inspect_video import inspect_video
from src.perception.omni_runner import cut_clip


V9_VERSION = "reference_program_v9_p04"
CONTENT_VERSION = "reference_content_program_v9_p03"
EDIT_VERSION = "reference_edit_program_v9_p03"
SECTION_OBSERVATIONS_VERSION = "section_observations_v9_p03"
NORMALIZATION_VERSION = "shot_normalization_v9_p03"
RECONCILIATION_VERSION = "boundary_reconciliation_v9_p04"
CONFLICT_GATE_VERSION = "semantic_conflicts_v9_p03"
MONTAGE_WATCH_VERSION = "montage_shot_observations_v9_p03"
REQUIREMENTS_VERSION = "material_requirements_v9_p0"
HUMAN_REVIEW_VERSION = "reference_program_human_review_v9_p01"
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
    "multi_angle_action", "contrast_montage", "text_led_montage",
}
TRANSITION_TYPES = {
    "hard_cut", "zoom_blur", "whip_pan", "flash", "crossfade", "match_cut",
    "other",
}
# P0.2：短于该时长且两侧均为内容镜头的边界段按转场段处理，不再要求信息增量解释。
TRANSITION_MAX_S = 0.30
# P0.2：快速蒙太奇 Section 的逐镜头观察触发阈值。
MONTAGE_SHOT_TRIGGER_COUNT = 4
MONTAGE_MEDIAN_SHOT_MAX_S = 0.8
MONTAGE_TRANSITION_TRIGGER_COUNT = 2
# P0.3：边界对账最大迭代轮数（move 后必须复查新边界，直到真实语义变化）。
RECONCILE_MAX_ITERATIONS = 6
# P0.3：蒙太奇类模式的 snippet 数与参考镜头结构对齐（下限≈0.6n）。
MONTAGE_LIKE_MODES = {
    "micro_montage", "event_compression_montage", "evidence_montage",
    "multi_angle_action", "contrast_montage", "text_led_montage",
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
    opening = re.match(r"^```(?:json)?[ \t]*\r?\n", cleaned)
    if opening:
        cleaned = cleaned[opening.end():].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
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
首个 shot 的 event_relation 表示它与本 Section 开始之前的画面是否属于同一连续事件；
若本段开头仍在延续上一段的事件，必须如实写 same_event，不得为了配合分段而改写。
cut_assessments 必须逐一评估输入 cut_candidates_to_assess 列出的**每一个**内部切点
（real_cut/not_cut/uncertain 三选一，附画面依据），一个都不能遗漏；Section 起止
boundary 只是时间锚点，未列入该数组时不得当作本段可验证切点。数组为空则输出空数组。
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


def _section_watch_request(section_id: str, interval: list[float],
                           ledger: dict[str, Any]) -> dict[str, Any]:
    evidence = _section_evidence(ledger, interval)
    prompt = SECTION_WATCH_PROMPT + json.dumps({
        "section_id": section_id,
        "deterministic_evidence": evidence,
        "cut_candidates_to_assess": [
            row["boundary_id"] for row in evidence["cut_candidates"]],
    }, ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "evidence": evidence}


def _validate_section_answer(raw: str, answer: Any, section_id: str,
                             interval: list[float],
                             evidence: dict[str, Any],
                             ledger: dict[str, Any], output_dir: Path,
                             *, raw_name: str, prompt: str) -> dict[str, Any]:
    """校验一段 Section 复看应答（单发与批量路径共用）。"""
    raw_path = output_dir / "raw_responses" / raw_name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")
    value = _parse_one_object(raw, stage="section_watch")
    if value.get("section_id") != section_id or not isinstance(value.get("shots"), list):
        raise V9Blocked("section_watch", "section_observation_invalid", section_id)
    if not value["shots"]:
        raise V9Blocked("section_watch", "section_shots_missing", section_id)
    _attach_intervals(value, ledger, keys=("shots",), stage="section_watch")
    for shot_index, shot in enumerate(value["shots"], 1):
        shot["shot_id"] = f"{section_id}.shot_{shot_index:03d}"
        if (shot["interval"][0] < interval[0] - 1e-6 or
                shot["interval"][1] > interval[1] + 1e-6):
            raise V9Blocked("section_watch", "shot_outside_section", section_id)
        if (not str(shot.get("information_added") or "").strip() or
                not str(shot.get("edit_function") or "").strip() or
                shot.get("event_relation") not in {
                    "same_event", "different_event", "uncertain"}):
            raise V9Blocked("section_watch", "shot_explanation_incomplete",
                            section_id)
    valid_cuts = {row["boundary_id"] for row in evidence["cut_candidates"]}
    for assessment in value.get("cut_assessments") or []:
        if (assessment.get("boundary_id") not in valid_cuts or
                assessment.get("status") not in {"real_cut", "not_cut", "uncertain"} or
                not str(assessment.get("reason") or "").strip()):
            raise V9Blocked("section_watch", "cut_assessment_invalid", section_id)
    assessed = {str(row.get("boundary_id"))
                for row in value.get("cut_assessments") or []}
    missing_cuts = sorted(valid_cuts - assessed)
    if missing_cuts:
        value["pending_cut_assessments"] = missing_cuts
    value.update({
        "source_interval": interval,
        "source_sha256": ledger["reference"]["sha256"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": str(raw_path), "raw_response_sha256": sha256_file(raw_path),
        "model_audit": _answer_audit(answer),
    })
    return value


def _observe_section(reference: Path, section_id: str, interval: list[float],
                     ledger: dict[str, Any], output_dir: Path, *, runner,
                     raw_name: str | None = None) -> dict[str, Any]:
    """Watch one Section interval in source media; shared by first pass and reconciliation."""
    request = _section_watch_request(section_id, interval, ledger)
    answer = runner.watch(
        Path(reference), request["prompt"], start_s=interval[0], end_s=interval[1],
        clip_dir=output_dir / "section_clips" / _safe_id(section_id),
        duration_s=interval[1] - interval[0], fps=4.0,
        use_audio_in_video=True, max_new_tokens=4096,
        stop_after_json_object=True)
    return _validate_section_answer(
        _answer_text(answer), answer, section_id, interval, request["evidence"],
        ledger, output_dir,
        raw_name=raw_name or f"section_{_safe_id(section_id)}.txt",
        prompt=request["prompt"])


DENSE_CUT_PROMPT = """每两帧一组（before/after）是一个候选切点两侧的画面。
逐个判断该切点：real_cut（画面内容真实切换到新内容）、not_cut（同一画面内的
微小变化/噪声）、uncertain（看不清）。只输出一个 JSON 对象：
{"assessments":[{"boundary_id":"...","status":"real_cut|not_cut|uncertain",
  "reason":"画面依据"}]}
输入："""


def _split_shots_at_assessed_real_cuts(row: dict[str, Any],
                                       ledger: dict[str, Any]) -> dict[str, Any]:
    """P0.4：评估为 real_cut 的切点由代码直接拆 shot（继承父镜头描述）。

    p04c 实测：移动边界后的重看段，模型 9 个 shot 只覆盖到 cut_018，而密集
    补判把 cut_019-022 判为 real_cut——"detector 知道、模型没逐刀交作业"
    再度触发必需 gap 并卡死 probe。确定性信号（real_cut 评估）有权威：
    直接按切点拆 shot，子镜头标记 split_basis，不赌模型逐刀枚举。
    """
    pts_by_id = _boundary_map(ledger)
    assessments = {str(item.get("boundary_id")): str(item.get("status"))
                   for item in row.get("cut_assessments") or []}
    real_cuts = sorted(
        (bid for bid, status in assessments.items() if status == "real_cut"
         and bid in pts_by_id), key=lambda bid: pts_by_id[bid])
    new_shots: list[dict[str, Any]] = []
    changed = False
    for shot in row.get("shots") or []:
        start_pts = pts_by_id.get(str(shot.get("start_boundary_id")))
        end_pts = pts_by_id.get(str(shot.get("end_boundary_id")))
        if start_pts is None or end_pts is None:
            new_shots.append(shot)
            continue
        interior = [bid for bid in real_cuts
                    if start_pts < pts_by_id[bid] < end_pts]
        if not interior:
            new_shots.append(shot)
            continue
        changed = True
        edges = [str(shot["start_boundary_id"]), *interior,
                 str(shot["end_boundary_id"])]
        for index, (left, right) in enumerate(zip(edges, edges[1:]), 1):
            child = dict(shot)
            child["shot_id"] = f"{shot.get('shot_id')}_s{index:02d}"
            child["start_boundary_id"] = left
            child["end_boundary_id"] = right
            child["interval"] = [pts_by_id[left], pts_by_id[right]]
            child["split_basis"] = "assessed_real_cut"
            new_shots.append(child)
    if changed:
        row["shots"] = new_shots
    # 补齐覆盖（p04b/p04c 实测）：模型 shot 可能没铺满 Section（尾部/头部
    # 缺 shot）。沿 real_cut 链程序化补齐到 Section 首末边界；子镜头继承
    # 相邻 shot 的描述并标记来源，逐镜头内容随后由蒙太奇逐镜头补看覆盖。
    interval = row.get("source_interval") or row.get("interval")
    shots = row.get("shots") or []
    if interval and shots:
        span_start, span_end = float(interval[0]), float(interval[1])
        start_id = min(
            pts_by_id, key=lambda bid: abs(pts_by_id[bid] - span_start))
        end_id = min(pts_by_id, key=lambda bid: abs(pts_by_id[bid] - span_end))
        first = shots[0]
        first_start = pts_by_id.get(str(first.get("start_boundary_id")))
        if first_start is not None and span_start < first_start - 0.05:
            head_cuts = [bid for bid in real_cuts
                         if span_start - 0.05 < pts_by_id[bid] < first_start]
            chain = [start_id, *head_cuts, str(first["start_boundary_id"])]
            for index, (left, right) in enumerate(zip(chain, chain[1:]), 1):
                child = dict(first)
                child["shot_id"] = f"{first.get('shot_id')}_t{index:02d}"
                child["start_boundary_id"] = left
                child["end_boundary_id"] = right
                child["interval"] = [pts_by_id[left], pts_by_id[right]]
                child["split_basis"] = "assessed_real_cut_tiling"
                shots.insert(0, child)
        last = shots[-1]
        last_end = pts_by_id.get(str(last.get("end_boundary_id")))
        if last_end is not None and last_end < span_end - 0.05:
            tail_cuts = [bid for bid in real_cuts
                         if last_end < pts_by_id[bid] < span_end + 0.05]
            chain = [str(last["end_boundary_id"]), *tail_cuts, end_id]
            for index, (left, right) in enumerate(zip(chain, chain[1:]), 1):
                child = dict(last)
                child["shot_id"] = f"{last.get('shot_id')}_t{index:02d}"
                child["start_boundary_id"] = left
                child["end_boundary_id"] = right
                child["interval"] = [pts_by_id[left], pts_by_id[right]]
                child["split_basis"] = "assessed_real_cut_tiling"
                shots.append(child)
        row["shots"] = shots
    return row


def _complete_dense_cut_assessments(
        reference: Path, ledger: dict[str, Any], row: dict[str, Any],
        output_dir: Path, *, runner, ffmpeg_bin: str) -> dict[str, Any]:
    """P0.3：整段 watch 未覆盖的密集候选切点，逐对帧（±0.25s）分批补判。"""
    missing = [str(item) for item in row.pop("pending_cut_assessments", []) or []]
    if not missing:
        return _split_shots_at_assessed_real_cuts(row, ledger)
    if not hasattr(runner, "inspect_media"):
        raise V9Blocked("section_watch", "cut_assessment_missing",
                        f"{row.get('section_id')}:{missing}")
    boundary_pts = _boundary_map(ledger)
    duration = float(ledger["reference"]["duration_s"])
    batches = [missing[index:index + 3] for index in range(0, len(missing), 3)]
    for batch_index, batch in enumerate(batches):
        images: list[Path] = []
        for boundary_id in batch:
            pts = boundary_pts[boundary_id]
            pair_dir = (output_dir / "dense_cut_frames" /
                        _safe_id(str(row.get("section_id"))) /
                        _safe_id(boundary_id))
            pair_dir.mkdir(parents=True, exist_ok=True)
            for tag, offset in (("before", -0.25), ("after", 0.25)):
                timestamp = min(max(pts + offset, 0.02), duration - 0.02)
                jpg = pair_dir / f"{tag}.jpg"
                if not jpg.is_file():
                    common.run_ffmpeg(ffmpeg_bin, [
                        "-y", "-ss", f"{timestamp:.3f}", "-i", str(reference),
                        "-frames:v", "1", "-q:v", "3", str(jpg)], timeout_s=60)
                images.append(jpg)
        payload = [{"boundary_id": boundary_id} for boundary_id in batch]
        answer = runner.inspect_media(
            images, DENSE_CUT_PROMPT + json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")),
            max_new_tokens=1024, stop_after_json_object=True)
        raw = _answer_text(answer)
        raw_path = (output_dir / "raw_responses" /
                    f"dense_cuts_{_safe_id(str(row.get('section_id')))}"
                    f"_{batch_index}.txt")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        value = _parse_one_object(raw, stage="section_watch")
        covered = set()
        for assessment in value.get("assessments") or []:
            boundary_id = str(assessment.get("boundary_id") or "")
            if (boundary_id not in batch or
                    assessment.get("status") not in {
                        "real_cut", "not_cut", "uncertain"} or
                    not str(assessment.get("reason") or "").strip()):
                continue
            row.setdefault("cut_assessments", []).append(
                {"boundary_id": boundary_id,
                 "status": assessment["status"],
                 "reason": f"[dense-frame-check] {assessment['reason']}"})
            covered.add(boundary_id)
        if set(batch) - covered:
            raise V9Blocked("section_watch", "cut_assessment_missing",
                            f"{row.get('section_id')}:{sorted(set(batch) - covered)}")
    return _split_shots_at_assessed_real_cuts(row, ledger)


def build_section_observations(
        reference: Path, draft: dict[str, Any], ledger: dict[str, Any],
        output_dir: Path, *, runner, ffmpeg_bin: str = "ffmpeg",
        force: bool = False) -> dict[str, Any]:
    """Rewatch each draft Section in source media before compiling either Program.

    多 worker 池可用时整段复看并行下发（watch_many），密集切点补判也批量并行。
    """
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
    specs = []
    for section in sections:
        section_id = str(section.get("section_id") or "")
        interval = section.get("interval")
        if not section_id or not isinstance(interval, list) or len(interval) != 2:
            raise V9Blocked("section_watch", "section_interval_missing", section_id)
        specs.append((section_id, [float(interval[0]), float(interval[1])]))
    requests = {section_id: _section_watch_request(section_id, interval, ledger)
                for section_id, interval in specs}
    rows: list[dict[str, Any]] = []
    if hasattr(runner, "watch_many") and len(specs) > 1:
        watch_requests = [{
            "video_path": str(Path(reference)),
            "prompt": requests[section_id]["prompt"],
            "kwargs": {
                "start_s": interval[0], "end_s": interval[1],
                "clip_dir": output_dir / "section_clips" / _safe_id(section_id),
                "duration_s": interval[1] - interval[0], "fps": 4.0,
                "use_audio_in_video": True, "max_new_tokens": 4096,
                "stop_after_json_object": True},
        } for section_id, interval in specs]
        answers = runner.watch_many(watch_requests)
        for (section_id, interval), answer in zip(specs, answers):
            rows.append(_validate_section_answer(
                _answer_text(answer), answer, section_id, interval,
                requests[section_id]["evidence"], ledger, output_dir,
                raw_name=f"section_{_safe_id(section_id)}.txt",
                prompt=requests[section_id]["prompt"]))
    else:
        for section_id, interval in specs:
            rows.append(_observe_section(Path(reference), section_id, interval,
                                         ledger, output_dir, runner=runner))
    # 密集切点补判：跨 Section 汇总后一次批量下发（可用池时）。
    pending = {str(row.get("section_id")): list(row.get("pending_cut_assessments") or [])
               for row in rows if row.get("pending_cut_assessments")}
    if pending:
        boundary_pts = _boundary_map(ledger)
        duration = float(ledger["reference"]["duration_s"])
        batch_requests = []
        batch_meta = []
        for row in rows:
            section_id = str(row.get("section_id"))
            for batch_index in range(0, len(pending.get(section_id) or []), 3):
                batch = pending[section_id][batch_index:batch_index + 3]
                images = []
                for boundary_id in batch:
                    pair_dir = (output_dir / "dense_cut_frames" /
                                _safe_id(section_id) / _safe_id(boundary_id))
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    for tag, offset in (("before", -0.25), ("after", 0.25)):
                        timestamp = min(max(
                            boundary_pts[boundary_id] + offset, 0.02), duration - 0.02)
                        jpg = pair_dir / f"{tag}.jpg"
                        if not jpg.is_file():
                            common.run_ffmpeg(ffmpeg_bin, [
                                "-y", "-ss", f"{timestamp:.3f}", "-i",
                                str(reference), "-frames:v", "1", "-q:v", "3",
                                str(jpg)], timeout_s=60)
                        images.append(jpg)
                payload = [{"boundary_id": bid} for bid in batch]
                batch_requests.append({
                    "image_paths": images,
                    "prompt": DENSE_CUT_PROMPT + json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")),
                    "kwargs": {"stop_after_json_object": True,
                               "max_new_tokens": 1024}})
                batch_meta.append((row, section_id, batch, batch_index))
        if hasattr(runner, "inspect_media_many") and len(batch_requests) > 1:
            answers = runner.inspect_media_many(batch_requests)
        else:
            answers = [runner.inspect_media(prompt=item["prompt"],
                                            image_paths=item["image_paths"],
                                            **item["kwargs"])
                       for item in batch_requests]
        for (row, section_id, batch, batch_index), answer in zip(
                batch_meta, answers):
            _merge_dense_cut_answer(reference, row, section_id, batch,
                                    batch_index, _answer_text(answer), answer,
                                    output_dir)
        for row in rows:
            row.pop("pending_cut_assessments", None)
            _split_shots_at_assessed_real_cuts(row, ledger)
    result = {"schema_version": SECTION_OBSERVATIONS_VERSION,
              "reference_sha256": source_hash, "input_sha256": input_hash,
              "sections": rows}
    _write_json(output_path, result)
    return result


def _merge_dense_cut_answer(reference: Path, row: dict[str, Any],
                            section_id: str, batch: list[str],
                            batch_index: int, raw: str, answer: Any,
                            output_dir: Path) -> None:
    raw_path = (output_dir / "raw_responses" /
                f"dense_cuts_{_safe_id(section_id)}_{batch_index}.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")
    value = _parse_one_object(raw, stage="section_watch")
    covered = set()
    for assessment in value.get("assessments") or []:
        boundary_id = str(assessment.get("boundary_id") or "")
        if (boundary_id not in batch or
                assessment.get("status") not in {
                    "real_cut", "not_cut", "uncertain"} or
                not str(assessment.get("reason") or "").strip()):
            continue
        row.setdefault("cut_assessments", []).append(
            {"boundary_id": boundary_id,
             "status": assessment["status"],
             "reason": f"[dense-frame-check] {assessment['reason']}"})
        covered.add(boundary_id)
    if set(batch) - covered:
        raise V9Blocked("section_watch", "cut_assessment_missing",
                        f"{section_id}:{sorted(set(batch) - covered)}")


def normalize_section_shots(section_observations: dict[str, Any],
                            ledger: dict[str, Any]) -> dict[str, Any]:
    """P0.2 Shot Normalizer + Transition Segment Resolver (deterministic).

    被 cut_assessments 确认为 real_cut 的边界必须成为内容镜头边界；短于
    TRANSITION_MAX_S 且两侧均有内容段的边界段归类为转场段，不再承担信息增量。
    """
    boundary_pts = _boundary_map(ledger)
    sections_out = []
    for row in section_observations.get("sections") or []:
        section_id = str(row.get("section_id") or "")
        interval = [float(value) for value in row.get("source_interval") or [0.0, 0.0]]
        real_cuts = sorted({
            boundary_pts[str(assessment.get("boundary_id"))]
            for assessment in row.get("cut_assessments") or []
            if assessment.get("status") == "real_cut" and
            str(assessment.get("boundary_id")) in boundary_pts and
            interval[0] + 1e-6 < boundary_pts[str(assessment.get("boundary_id"))] <
            interval[1] - 1e-6
        })
        edges = sorted({interval[0], *real_cuts, interval[1]})
        model_shots = row.get("shots") or []

        def _carry_model_shot(seg_interval: list[float]) -> dict[str, Any]:
            best, best_overlap = {}, 0.0
            for shot in model_shots:
                shot_interval = [float(value) for value in shot.get("interval") or []]
                if len(shot_interval) != 2:
                    continue
                overlap = (min(seg_interval[1], shot_interval[1]) -
                           max(seg_interval[0], shot_interval[0]))
                if overlap > best_overlap:
                    best, best_overlap = shot, overlap
            return best

        raw_segments = [
            {"interval": [round(left, 6), round(right, 6)],
             "duration_s": round(right - left, 6)}
            for left, right in zip(edges, edges[1:])]
        segments = []
        for position, segment in enumerate(raw_segments):
            flanked = 0 < position < len(raw_segments) - 1
            is_transition = (segment["duration_s"] < TRANSITION_MAX_S and flanked)
            segments.append({**segment, "kind": "transition" if is_transition else "content"})
        content_index = transition_index = 0
        content_shots, transition_segments = [], []
        for segment in segments:
            carrier = _carry_model_shot(segment["interval"])
            if segment["kind"] == "content":
                content_index += 1
                content_shots.append({
                    "shot_id": f"{section_id}.shot_C{content_index:02d}",
                    "interval": segment["interval"],
                    "duration_s": segment["duration_s"],
                    "information_added": str(carrier.get("information_added") or ""),
                    "edit_function": str(carrier.get("edit_function") or ""),
                    "event_relation": carrier.get("event_relation"),
                    "carried_model_shot_id": carrier.get("shot_id"),
                    "interior_real_cuts": [
                        cut for cut in real_cuts if
                        segment["interval"][0] + 1e-6 < cut < segment["interval"][1] - 1e-6],
                })
            else:
                transition_index += 1
                transition_segments.append({
                    "transition_id": f"{section_id}.transition_T{transition_index:02d}",
                    "interval": segment["interval"],
                    "duration_s": segment["duration_s"],
                    "transition_type": "unclassified",
                    "classification_basis": [
                        f"duration_lt_{TRANSITION_MAX_S}", "flanked_by_content"],
                    "carried_model_shot_id": carrier.get("shot_id"),
                })
        sections_out.append({
            "section_id": section_id, "interval": interval,
            "real_cut_pts": [round(value, 6) for value in real_cuts],
            "content_shots": content_shots, "transition_segments": transition_segments,
        })
    return {"schema_version": NORMALIZATION_VERSION,
            "reference_sha256": section_observations.get("reference_sha256"),
            "sections": sections_out}


def apply_montage_reclassification(normalization: dict[str, Any],
                                   montage: dict[str, Any] | None) -> dict[str, Any]:
    """P0.3：转场段永远保持 transition 身份；逐镜头观察只补充类型与叠加文字。

    转场期间画面上挂着字幕不代表它是内容镜头——文字记为 overlay_text 属性。
    """
    if not montage:
        return normalization
    by_segment = {
        str(row.get("segment_id")): row
        for section in montage.get("sections") or []
        for row in section.get("segments") or []}
    for section in normalization.get("sections") or []:
        for transition in section.get("transition_segments") or []:
            observed = by_segment.get(str(transition.get("transition_id")))
            if not observed:
                continue
            if str(observed.get("transition_type") or "") in TRANSITION_TYPES:
                transition["transition_type"] = str(observed["transition_type"])
                transition["classification_basis"] = ["montage_observation"]
            overlay = str(observed.get("on_screen_text") or "").strip()
            if overlay:
                transition["overlay_text"] = overlay
                transition["overlay_basis"] = "montage_observation"
    return normalization


BOUNDARY_FRAME_CHECK_PROMPT = narrative_boundary.NARRATIVE_BOUNDARY_PROMPT


def _boundary_frames(ffmpeg_bin: str, reference: Path, pts: float,
                      out_dir: Path, span: tuple[float, float]) -> list[Path]:
    """边界前后各三帧（before/after 各 -2/-1/-0.4 与 +0.4/+1/+2 秒）。"""
    offsets = (-2.0, -1.0, -0.4, 0.4, 1.0, 2.0)
    names = ("before_1", "before_2", "before_3", "after_1", "after_2", "after_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, offset in zip(names, offsets):
        timestamp = min(max(pts + offset, span[0] + 0.02), span[1] - 0.02)
        jpg = out_dir / f"{name}.jpg"
        if not jpg.is_file():
            common.run_ffmpeg(ffmpeg_bin, [
                "-y", "-ss", f"{timestamp:.3f}", "-i", str(reference),
                "-frames:v", "1", "-q:v", "3", str(jpg)], timeout_s=60)
        frames.append(jpg)
    return frames


def _frame_check_boundary(reference: Path, pts: float, ledger: dict[str, Any],
                          output_dir: Path, *, runner, ffmpeg_bin: str,
                          attempt: int, slug: str,
                          global_outline: dict[str, Any] | None = None
                          ) -> dict[str, Any]:
    """P0.4 叙事功能边界判定：LLM 只选功能枚举，决策在代码里。"""
    duration = float(ledger["reference"]["duration_s"])
    images = _boundary_frames(ffmpeg_bin, Path(reference), pts,
                              output_dir / "boundary_frames" /
                              f"{slug}_a{attempt}", (0.0, duration))
    prompt = narrative_boundary.NARRATIVE_BOUNDARY_PROMPT
    if global_outline:
        prompt += json.dumps({"global_narrative_outline": global_outline},
                             ensure_ascii=False, separators=(",", ":"))
    answer = runner.inspect_media(
        images, prompt, max_new_tokens=768, stop_after_json_object=True)
    raw = _answer_text(answer)
    raw_path = (output_dir / "raw_responses" /
                f"boundary_{slug}_a{attempt}.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")
    verdict = _parse_one_object(raw, stage="boundary_reconciliation")
    narrative_boundary.validate_narrative_verdict(verdict)
    return verdict


def reconcile_section_boundaries(
        reference: Path, ledger: dict[str, Any],
        section_observations: dict[str, Any], output_dir: Path, *, runner,
        ffmpeg_bin: str = "ffmpeg", force: bool = False,
        draft: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """P0.4 迭代式 Section Boundary Reconciliation（叙事功能边界）。

    每个内部边界：下一 Section 首 shot 报 same_event 时记为确定性信号（不单独
    放行），抽边界前后各三帧，让模型只从**有限功能表**选 before/after 功能
    枚举，keep/move 由 `narrative_boundary.decide_narrative_boundary` 做
    字符串不等比较（纯确定性）。功能相同 → 移动到下一候选并复查新边界，
    直到功能真正切换或候选耗尽（BLOCKED）。全局叙事纲要（来自粗读草稿，
    不含边界信息）作为 local-to-global 上下文随帧一起下发。
    边界精度要求是"功能不混"，不是精确秒点（GEBD 粒度宽容原则）。
    """
    output_dir = Path(output_dir)
    rows = [dict(row) for row in section_observations.get("sections") or []]
    source_hash = ledger["reference"]["sha256"]
    boundary_pts = _boundary_map(ledger)
    global_outline = None
    if isinstance(draft, dict):
        global_outline = {
            "core_expression": draft.get("core_expression_draft") or {},
            "unit_purposes": [
                {"unit_id": unit.get("unit_id"), "purpose": unit.get("purpose")}
                for unit in draft.get("meaningful_units") or []],
        }

    def _nearest_boundary_id(pts_value: float) -> str:
        return min(boundary_pts, key=lambda bid: abs(
            boundary_pts[bid] - pts_value))

    records: list[dict[str, Any]] = []
    rewatched: list[str] = []
    for index in range(len(rows) - 1):
        previous_row, next_row = rows[index], rows[index + 1]
        section_span = (float(previous_row["source_interval"][0]),
                        float(next_row["source_interval"][1]))
        original_id = _nearest_boundary_id(float(next_row["source_interval"][0]))
        candidates = sorted(
            (bid for bid, pts_value in boundary_pts.items()
             if section_span[0] + 1e-6 < pts_value < section_span[1] - 1e-6
             and bid not in {"video_start", "video_end"}),
            key=lambda bid: boundary_pts[bid])
        current_id = original_id
        attempts: list[dict[str, Any]] = []
        accepted_id = None
        for _attempt in range(RECONCILE_MAX_ITERATIONS):
            current_pts = boundary_pts[current_id]
            head_shots = next_row.get("shots") or []
            head_same_event = bool(
                head_shots and
                head_shots[0].get("event_relation") == "same_event" and
                abs(float(head_shots[0]["interval"][0]) - current_pts) < 0.5)
            verdict = _frame_check_boundary(
                reference, current_pts, ledger, output_dir, runner=runner,
                ffmpeg_bin=ffmpeg_bin, attempt=_attempt,
                slug=_safe_id(f"{previous_row.get('section_id')}_"
                              f"{next_row.get('section_id')}"),
                global_outline=global_outline)
            # P0.4：LLM 只从有限功能表选枚举；keep/move = 枚举不等比较 +
            # same_event 闭环否决（同一完整事件的收尾庆祝是 phase 不是 Section）。
            boundary_needed = narrative_boundary.decide_narrative_boundary(
                verdict.get("before_function"), verdict.get("after_function"),
                verdict.get("same_event"))
            attempts.append({
                "boundary_id": current_id,
                "method": ("frame_check+same_event_signal" if head_same_event
                           else "frame_check"),
                "same_event_signal": head_same_event,
                "decision": "keep" if boundary_needed else "move",
                "before_function": verdict.get("before_function"),
                "after_function": verdict.get("after_function"),
                "same_event": verdict.get("same_event"),
                "boundary_needed": boundary_needed,
                "semantic_change": boundary_needed,
                "detail": {"reason": verdict.get("reason")}})
            if boundary_needed:
                accepted_id = current_id
                break
            forward = [bid for bid in candidates
                       if boundary_pts[bid] > current_pts + 0.05]
            if not forward:
                break
            current_id = forward[0]
        record = {
            "boundary_id": original_id,
            "between": [str(previous_row.get("section_id")),
                        str(next_row.get("section_id"))],
            "attempts": attempts, "action": "kept", "moved_to": None,
            "semantic_change": accepted_id is not None,
        }
        if accepted_id is None:
            raise V9Blocked(
                "boundary_reconciliation", "unjustified_section_boundary",
                f"{original_id}: no semantic change point within "
                f"{section_span}")
        if accepted_id != original_id:
            record.update({"action": "moved", "moved_to": accepted_id})
        records.append(record)
        final_pts = boundary_pts[accepted_id]
        if (abs(float(previous_row["source_interval"][1]) - final_pts) > 1e-6 or
                abs(float(next_row["source_interval"][0]) - final_pts) > 1e-6):
            pairs = (
                (index, str(previous_row.get("section_id")),
                 float(previous_row["source_interval"][0]), final_pts),
                (index + 1, str(next_row.get("section_id")), final_pts,
                 float(next_row["source_interval"][1])),
            )
            for row_index, section_id, start, end in pairs:
                rows[row_index] = _complete_dense_cut_assessments(
                    Path(reference), ledger,
                    _observe_section(
                        Path(reference), section_id, [start, end], ledger,
                        output_dir, runner=runner,
                        raw_name=(f"section_{_safe_id(section_id)}_reconciled.txt")),
                    output_dir, runner=runner, ffmpeg_bin=ffmpeg_bin)
                rewatched.append(section_id)
            previous_row, next_row = rows[index], rows[index + 1]

    reconciled = {"schema_version": SECTION_OBSERVATIONS_VERSION,
                  "reference_sha256": source_hash,
                  "input_sha256": json_hash({"sections": rows,
                                             "prompt": SECTION_WATCH_PROMPT,
                                             "reconciled": True}),
                  "reconciled": True, "sections": rows}
    _write_json(output_dir / "section_observations.json", reconciled)
    result = {"schema_version": RECONCILIATION_VERSION,
              "input_sha256": json_hash({"observations": section_observations}),
              "boundaries": records,
              "sections_after": [{"section_id": str(row.get("section_id")),
                                  "interval": row.get("source_interval")}
                                 for row in rows],
              "rewatched": rewatched}
    _write_json(output_dir / "boundary_reconciliation.json", result)
    return result, reconciled


PER_SHOT_PROMPT = """只看这几帧（同一画面单元的起/中/尾帧）。只输出一个 JSON 对象，
描述该画面单元可见的内容与屏幕文字；不确定就写 null，不要推断叙事意义：
{"segment_id":"...","visible_content":"可见主体与动作","on_screen_text":null,
 "transition_type":null}
transition_type 仅当输入 segment_kind 为 transition 时填写，取值
zoom_blur|whip_pan|flash|crossfade|match_cut|other；内容单元恒为 null。
输入："""


def _extract_segment_frames(ffmpeg_bin: str, reference: Path,
                            interval: list[float], out_dir: Path,
                            count: int = 3) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = float(interval[0]), float(interval[1])
    span = max(end - start, 0.04)
    frames = []
    for index in range(count):
        offset = min(span * (index + 0.5) / count, max(span - 0.02, 0.0))
        jpg = out_dir / f"f{index}.jpg"
        if not jpg.is_file():
            common.run_ffmpeg(ffmpeg_bin, [
                "-y", "-ss", f"{start + offset:.3f}", "-i", str(reference),
                "-frames:v", "1", "-q:v", "3", str(jpg)], timeout_s=60)
        frames.append(jpg)
    return frames


def watch_fast_montage_shots(
        reference: Path, ledger: dict[str, Any], normalization: dict[str, Any],
        output_dir: Path, *, runner, ffmpeg_bin: str = "ffmpeg",
        force: bool = False) -> dict[str, Any]:
    """P0.2 快速蒙太奇逐镜头观察：内容单元与转场段各自抽起/中/尾三帧单独看。

    局部负责“看清是什么”，整段 Rewatch 负责“为什么这样连”。带实义屏幕文字的
    转场段会被 apply_montage_reclassification 升回内容镜头。
    """
    output_dir = Path(output_dir)
    output_path = output_dir / "montage_shot_observations.json"
    input_hash = json_hash({"normalization": normalization,
                            "prompt": PER_SHOT_PROMPT,
                            "reference": ledger["reference"]["sha256"]})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    sections_out, requests = [], []
    for section in normalization.get("sections") or []:
        content = section.get("content_shots") or []
        transitions = section.get("transition_segments") or []
        durations = sorted(float(shot["duration_s"]) for shot in content)
        median = durations[len(durations) // 2] if durations else None
        trigger = (
            len(content) >= MONTAGE_SHOT_TRIGGER_COUNT or
            (median is not None and median <= MONTAGE_MEDIAN_SHOT_MAX_S) or
            len(transitions) >= MONTAGE_TRANSITION_TRIGGER_COUNT)
        if not trigger:
            continue
        section_entry = {
            "section_id": section["section_id"], "trigger_reason": {
                "content_shot_count": len(content),
                "median_duration_s": median,
                "transition_count": len(transitions)},
            "segments": []}
        for segment in content:
            requests.append({
                "segment_id": segment["shot_id"], "segment_kind": "content",
                "interval": segment["interval"],
                "images": _extract_segment_frames(
                    ffmpeg_bin, Path(reference), segment["interval"],
                    output_dir / "montage_frames" /
                    _safe_id(section["section_id"]) / _safe_id(segment["shot_id"]))})
        for segment in transitions:
            requests.append({
                "segment_id": segment["transition_id"], "segment_kind": "transition",
                "interval": segment["interval"],
                "images": _extract_segment_frames(
                    ffmpeg_bin, Path(reference), segment["interval"],
                    output_dir / "montage_frames" /
                    _safe_id(section["section_id"]) / _safe_id(segment["transition_id"]))})
        sections_out.append(section_entry)
    if not requests:
        result = {"schema_version": MONTAGE_WATCH_VERSION,
                  "input_sha256": input_hash, "sections": []}
        _write_json(output_path, result)
        return result
    prompts = [{
        "image_paths": request["images"],
        "prompt": PER_SHOT_PROMPT + json.dumps({
            "segment_id": request["segment_id"],
            "segment_kind": request["segment_kind"],
            "interval": request["interval"]}, ensure_ascii=False,
            separators=(",", ":")),
        "kwargs": {"stop_after_json_object": True, "max_new_tokens": 768},
    } for request in requests]
    if hasattr(runner, "inspect_media_many"):
        answers = runner.inspect_media_many(prompts)
    else:
        answers = [
            runner.inspect_media(prompt["image_paths"], prompt["prompt"],
                                 **prompt["kwargs"])
            for prompt in prompts]
    for request, answer in zip(requests, answers):
        raw = _answer_text(answer)
        raw_path = (output_dir / "raw_responses" / "montage" /
                    f"{_safe_id(request['segment_id'])}.txt")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        value = _parse_one_object(raw, stage="montage_watch")
        if not str(value.get("visible_content") or "").strip():
            raise V9Blocked("montage_watch", "shot_visible_content_missing",
                            request["segment_id"])
        entry = {
            "segment_id": request["segment_id"],
            "segment_kind": request["segment_kind"],
            "interval": request["interval"],
            "visible_content": str(value.get("visible_content")),
            "on_screen_text": (str(value["on_screen_text"]).strip()
                               if str(value.get("on_screen_text") or "").strip()
                               else None),
            "raw_response": str(raw_path),
            "raw_response_sha256": sha256_file(raw_path),
        }
        if request["segment_kind"] == "transition":
            transition_type = str(value.get("transition_type") or "")
            if transition_type not in TRANSITION_TYPES:
                raise V9Blocked("montage_watch", "transition_type_invalid",
                                f"{request['segment_id']}:{transition_type}")
            entry["transition_type"] = transition_type
        for section_entry in sections_out:
            if section_entry["section_id"] == request["segment_id"].split(".")[0]:
                section_entry["segments"].append(entry)
                break
    result = {"schema_version": MONTAGE_WATCH_VERSION,
              "input_sha256": input_hash, "sections": sections_out}
    _write_json(output_path, result)
    return result


CONFLICT_GATE_PROMPT = """你是事实对账器。输入是同一参考视频在不同观察阶段产生的
陈述（全局看片/逐段复看/探针）。只找**可观察事实层面的实质矛盾**（事件性质、主体
数量、动作、结果、地点）；措辞差异或详略差异不算矛盾。只输出一个 JSON 对象：
{"contradictions":[{"topic":"事件性质","source_a":"...","quote_a":"逐字引文",
  "source_b":"...","quote_b":"逐字引文","material":true}],
 "consistent":true}
输入："""


NEUTRAL_EVENT_RECHECK_PROMPT = """只观察这段视频画面本身，不判断赛事名称或专业分类。
只输出一个 JSON 对象：
{"observable_actions":["按时间顺序的可观察动作"],
 "subject_count":null,"mutual_attacks_visible":null,"falls_visible":null,
 "result_visible":null,"event_nature_observable":"用可观察结构描述，不使用分类名称"}
布尔字段看不清填 null，不得猜。输入为视频本身。"""


CONFLICT_RESOLUTION_PROMPT = """根据中立复核观察，判断该矛盾是否被解决。只输出一个
JSON 对象：{"resolution":"resolved|unresolved","resolved_description":"以可观察结构
表述的一致结论","must_not_assert":["矛盾双方使用且中立复核无法支持的具体名称或分类词"]}
unresolved 时必须给出 must_not_assert。输入："""


def semantic_conflict_gate(
        reference: Path, draft: dict[str, Any], ledger: dict[str, Any],
        section_observations: dict[str, Any], resolved: dict[str, Any],
        output_dir: Path, *, runner, force: bool = False) -> dict[str, Any]:
    """P0.2 Semantic Conflict Gate：跨阶段关键事实对账，矛盾必须中立复核或标 unknown。"""
    output_dir = Path(output_dir)
    output_path = output_dir / "semantic_conflicts.json"
    sections = section_observations.get("sections") or []
    sources = [{
        "source": "global_watch",
        "statement": " ".join(filter(None, [
            str((draft.get("core_expression_draft") or {}).get("reference_specific") or ""),
            *[str(row.get("description") or "") for row in
              (draft.get("observations") or [])[:3]]]))[:900],
    }]
    section_intervals = {}
    for row in sections:
        section_id = str(row.get("section_id") or "")
        section_intervals[section_id] = [
            float(value) for value in row.get("source_interval") or [0.0, 0.0]]
        sources.append({
            "source": f"section:{section_id}",
            "statement": " ".join(str(shot.get("information_added") or "")
                                  for shot in row.get("shots") or [])[:900]})
    for probe in resolved.get("probe_history") or []:
        raw_path = probe.get("raw_response")
        if not raw_path or probe.get("result_status") != "resolved":
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        sources.append({
            "source": f"probe:{probe.get('question_id')}:{probe.get('probe')}",
            "statement": path.read_text(encoding="utf-8")[:1200]})
    input_hash = json_hash({"sources": sources, "prompt": CONFLICT_GATE_PROMPT})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    value, audit = _ask_object(
        runner, CONFLICT_GATE_PROMPT + json.dumps(
            sources, ensure_ascii=False, separators=(",", ":")),
        output_dir / "raw_responses" / "conflict_gate.txt",
        stage="conflict_gate", max_new_tokens=2048)
    contradictions = [
        row for row in value.get("contradictions") or []
        if row.get("material") is True and str(row.get("topic") or "").strip()]
    rechecks, unresolved, must_not_assert = [], [], []
    for row in contradictions[:3]:
        interval = None
        for source_name in (row.get("source_a"), row.get("source_b")):
            section_id = str(source_name or "").removeprefix("section:")
            if section_id in section_intervals:
                interval = section_intervals[section_id]
                break
        if interval is None:
            interval = [0.0, float(ledger["reference"]["duration_s"])]
        slug = _safe_id(str(row["topic"]))[:40]
        answer = runner.watch(
            Path(reference), NEUTRAL_EVENT_RECHECK_PROMPT,
            start_s=interval[0], end_s=interval[1],
            clip_dir=output_dir / "neutral_rechecks" / slug,
            duration_s=interval[1] - interval[0], fps=6.0,
            use_audio_in_video=True, max_new_tokens=1024,
            stop_after_json_object=True)
        raw = _answer_text(answer)
        raw_path = output_dir / "raw_responses" / f"neutral_recheck_{slug}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        neutral = _parse_one_object(raw, stage="conflict_gate")
        resolution, res_audit = _ask_object(
            runner, CONFLICT_RESOLUTION_PROMPT + json.dumps({
                "topic": row.get("topic"), "contradiction": {
                    "source_a": row.get("source_a"), "quote_a": row.get("quote_a"),
                    "source_b": row.get("source_b"), "quote_b": row.get("quote_b")},
                "neutral_recheck": neutral}, ensure_ascii=False,
                separators=(",", ":")),
            output_dir / "raw_responses" / f"conflict_resolution_{slug}.txt",
            stage="conflict_gate", max_new_tokens=1024)
        entry = {
            "topic": str(row.get("topic")), "quotes": {
                "source_a": row.get("source_a"), "quote_a": row.get("quote_a"),
                "source_b": row.get("source_b"), "quote_b": row.get("quote_b")},
            "neutral_recheck": neutral,
            "resolution": str(resolution.get("resolution") or "unresolved"),
            "resolved_description": str(resolution.get("resolved_description") or ""),
            "raw_neutral_sha256": sha256_file(raw_path),
        }
        terms = [str(term).strip() for term in resolution.get("must_not_assert") or []
                 if str(term).strip()]
        if entry["resolution"] == "resolved":
            entry["must_not_assert"] = []
        else:
            entry["must_not_assert"] = terms or [str(row.get("topic"))]
            unresolved.append(entry["topic"])
            must_not_assert.extend(entry["must_not_assert"])
        rechecks.append(entry)
    result = {"schema_version": CONFLICT_GATE_VERSION,
              "input_sha256": input_hash,
              "contradictions": contradictions, "neutral_rechecks": rechecks,
              "unresolved_topics": unresolved,
              "must_not_assert": sorted(set(must_not_assert)),
              "provenance": audit}
    _write_json(output_path, result)
    return result


def collect_reference_gaps(draft: dict[str, Any],
                           section_observations: dict[str, Any],
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
                previous_importance = known.get("importance")
                if question.get("importance") == "required":
                    known["importance"] = "required"
                    if previous_importance != "required":
                        known["selected_probe"] = question.get("selected_probe")
                known["interval"] = [min(float(prior[0]), float(interval[0])),
                                     max(float(prior[1]), float(interval[1]))]
                known["trigger_ids"] = sorted(set(
                    (known.get("trigger_ids") or []) +
                    (question.get("trigger_ids") or [])))
                known["trigger_sources"] = sorted({
                    str(value) for value in (
                        known.get("trigger_sources") or []) +
                    [known.get("gap_source"), question.get("gap_source")]
                    if value})
                if question.get("question") and question["question"] not in str(
                        known.get("question") or ""):
                    known["question"] = " / ".join(filter(None, (
                        str(known.get("question") or ""), str(question["question"]))))
                if question.get("status") == "open":
                    known["status"] = "open"
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

NATIVE_PROBE_MAX_FRAMES = 60


def _validate_probe_selection(question: dict[str, Any]) -> None:
    gap_type = str(question.get("gap_type") or "")
    selected = str(question.get("selected_probe") or "")
    allowed = _GAP_PROBES.get(gap_type)
    if selected not in PROBE_TYPES or allowed is None or selected not in allowed:
        raise V9Blocked("probe", "probe_does_not_match_gap",
                        f"gap={gap_type!r} probe={selected!r}")


def _native_probe_frame_count(ledger: dict[str, Any], interval: list[float]) -> int:
    start, end = map(float, interval)
    return sum(start - 1e-6 <= float(row["pts_s"]) <= end + 1e-6
               for row in ledger.get("native_frames") or [])


def _native_probe_images(reference: Path, interval: list[float], output: Path, *,
                         ffmpeg_bin: str,
                         max_frames: int = NATIVE_PROBE_MAX_FRAMES) -> list[Path]:
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
        frame_count = _native_probe_frame_count(ledger, interval)
        if frame_count > NATIVE_PROBE_MAX_FRAMES:
            raise V9Blocked(
                "probe", "native_frame_probe_too_wide",
                f"{frame_count} frames exceeds {NATIVE_PROBE_MAX_FRAMES}")
        images = _native_probe_images(
            reference, interval, output / "native_frames",
            ffmpeg_bin=ffmpeg_bin)
        if not images or not hasattr(runner, "inspect_media"):
            raise V9Blocked("probe", "native_frame_probe_unavailable")
        answer = runner.inspect_media(images, prompt, max_new_tokens=3072,
                                      stop_after_json_object=True)
    elif selected == "ocr_context":
        if not hasattr(runner, "inspect_media"):
            raise V9Blocked("probe", "ocr_frame_probe_unavailable")
        clip = cut_clip(ffmpeg_bin, reference, output / "clip", start_s=interval[0],
                        end_s=interval[1])
        answer = runner.inspect_media(
            ocr_images, prompt, video_path=clip, fps=4.0,
            source_origin_s=interval[0], max_new_tokens=3072,
            stop_after_json_object=True)
    else:
        fps = 12.0 if selected == "dense_video" else 4.0
        answer = runner.watch(
            reference, prompt, start_s=interval[0], end_s=interval[1],
            clip_dir=output / "clip", duration_s=interval[1] - interval[0],
            fps=fps, use_audio_in_video=True, max_new_tokens=3072,
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
    observed_types = {
        "observed_visual", "observed_textual_claim", "observed_audio_claim"}
    if result["status"] == "resolved" and not any(
            row.get("evidence_type") in observed_types and
            str(row.get("description") or "").strip()
            for row in result.get("evidence") or []):
        result["status"] = "open"
        result["resolution_gate"] = "supported_evidence_missing"
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
        pending = [row for row in questions if row.get("status") == "open" and
                   not row.get("probe_disposition")]
        if not pending:
            break
        for question in pending[:max_probes_per_round]:
            if (question.get("selected_probe") == "native_frames" and
                    question.get("importance") != "required"):
                frame_count = _native_probe_frame_count(
                    ledger, [float(value) for value in question["interval"]])
                if frame_count > NATIVE_PROBE_MAX_FRAMES:
                    question["probe_disposition"] = "skipped_scope_too_wide"
                    question["probe_skip_reason"] = (
                        "optional_native_frame_scope_exceeds_limit")
                    probe_log.append({
                        "round": round_index, "probe": "native_frames",
                        "question_id": question.get("id"),
                        "interval": question.get("interval"),
                        "execution_status": "skipped",
                        "skip_reason": question["probe_skip_reason"],
                        "candidate_frame_count": frame_count,
                        "max_frames": NATIVE_PROBE_MAX_FRAMES,
                        "evidence_fields": [], "result_status": "open",
                    })
                    continue
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
    "primary_focus": {"type": "subject|event|theme|place|contrast|mixed",
                      "detail": "一句话说明为什么这是主焦点"},
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
    "continuity_basis": {
      "subject":{"reason":"画面依据","evidence_ids":[]},
      "opponent":{"reason":"画面依据","evidence_ids":[]},
      "scene":{"reason":"画面依据","evidence_ids":[]},
      "event":{"reason":"画面依据","evidence_ids":[]},
      "actor_role":{"reason":"画面依据","evidence_ids":[]},
      "spatial_orientation":{"reason":"画面依据","evidence_ids":[]}
    },
    "final_text_quote":"",
    "hook_text_quote":""
  }]
}
primary_focus 必须是对象，type 取 subject|event|theme|place|contrast|mixed 之一。
continuity 六个维度 subject/opponent/scene/event/actor_role/spatial_orientation 都要填写；
unknown 表示尚不能判断，不得写成 not_required。
continuity_basis 必须与六个维度一一对应，每个维度都写 reason（依据可见画面）；
标 required 或 preferred 的维度必须给出 evidence_ids，其余维度 evidence_ids 可为空数组。
不要自动补固定故事槽；Section 必须来自实际画面信息变化。
sections 的数量、顺序与区间必须与输入 section_observations 的分段完全一致，
不得使用对账前的旧边界，也不得自创边界。
同一连续事件（如一场完整的对抗：准备→对抗→决定性结果→反应）不得被切成两个 Section；
以输入的 boundary_reconciliation 与逐段观察为准，每个 Section 的事实描述必须覆盖
该区间内全部可辨画面（包括区间开头的事件结果/反应镜头），不得无声跳过。
输入 conflict_constraints.must_not_assert 中的词在整个输出中禁止出现；有 unresolved_topics
时相关描述必须改用可观察结构（主体数量、动作顺序、可见结果），不使用争议分类名。
最后一个 Section 若输入 OCR/逐镜头观察含屏幕文字，final_text_quote 必须逐字引用其中
一条收尾文字，并据此判断其修辞功能（幽默/反差/抒情/宣告等），理由要引用原文。
final_text_quote 非空时，audience_takeaway 与 cognition_change.after 必须直接解释这句
收尾文字对观众的效果（例如自嘲/反差/轻松收尾），不得只写抽象抒情。
第一个含屏幕文字的 Section，若该文字提出断言/立场/命题（如社会偏见），hook_text_quote
必须逐字引用，并在 audience_takeaway 中把它呈现为全片待检验/待反驳的命题——
不得把 Hook 段降级为人物或场景介绍。
其余 Section 的 final_text_quote 与 hook_text_quote 留空字符串。
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
    "shot_refs":["section_01.shot_C01"],
    "transition_refs":[],
    "composition_mode":"continuous_clip|micro_montage|event_compression_montage|evidence_montage|dialogue_compression|reaction_result_pair|multi_angle_action|contrast_montage|text_led_montage",
    "duration_budget_s":0.0, "source_semantics":"one_long_event|multiple_events|dialogue|continuous_moment|paired_moments",
    "semantic_phases":[], "snippet_count_range":[1,1],
    "source_continuity":"continuous_required|non_contiguous_allowed",
    "semantic_continuity":"required", "ordering_constraint":"preserve_event_progression|claim_consistency|source_order|verified_relation",
    "individual_duration_policy":"minimum_sufficient_duration",
    "audience_requirement":"", "evidence_ids":[]
  }]
}
shot_refs 只能引用输入 normalized_shots 的内容镜头 id（shot_C..）；转场段引用
transition_refs（transition_T..），不得把转场段当内容镜头。两个数组的 id 只能来自
输入中实际存在的 id，没有对应元素就留空数组，禁止按命名规律编造 id。
operation_type 与 composition_mode 是两套枚举：蒙太奇属于 composition_mode，
operation_type 里不得出现任何 *_montage 值，蒙太奇段的拼接操作就写 hard_cut。
composition_mode 必须与镜头结构一致：内容镜头多于 1 个或有内部真切点的 Section 不得
用 continuous_clip；单一内容镜头且无转场的 Section 不得声称蒙太奇类模式。
同一连续事件多角度快速切换用 multi_angle_action；以屏幕文字/照片卡为主体的快速罗列
用 text_led_montage；对照式并列用 contrast_montage。
不要输出 measured_style；镜头、Section 和信息间隔的时长统计由程序根据真实 PTS 计算。
不同事件不能被编造成单一事件因果；对白压缩不得改变原意。
输入："""


def _ask_object(runner: Any, prompt: str, output: Path, *, stage: str,
                max_new_tokens: int = 4096) -> tuple[dict[str, Any], dict[str, Any]]:
    answer = runner.ask(
        prompt, max_new_tokens=max_new_tokens, stop_after_json_object=True)
    raw = _answer_text(answer)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw, encoding="utf-8")
    return _parse_one_object(raw, stage=stage), {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": str(output), "raw_response_sha256": sha256_file(output),
        "model_audit": _answer_audit(answer),
    }


REPAIR_HEADER = """上一版输出违反契约校验，未通过。错误清单如下（逐条修复）：
"""


def _sanitize_repair_hint(errors: list[str]) -> list[str]:
    """修复提示去毒：泄漏类错误原文含参考专有词，回喂即教模型照抄。

    p04d 实测：hint 带 ['双手','废人','跆拳道'] → 重写的 requirements 原样
    出现这三个词。凡泄漏类错误，词表替换为计数描述。
    """
    sanitized: list[str] = []
    for error in errors:
        if "leaked into requirements" in error:
            sanitized.append(re.sub(
                r"\[.*\]", "(若干参考专有事实词，此处隐去，不得在重写中出现任何参考专有事实)",
                error))
        else:
            sanitized.append(error)
    return sanitized


def build_reference_content_program(
        draft: dict[str, Any], resolved: dict[str, Any], ledger: dict[str, Any],
        output_dir: Path, *, runner,
        section_observations: dict[str, Any] | None = None,
        normalization: dict[str, Any] | None = None,
        reconciliation: dict[str, Any] | None = None,
        conflicts: dict[str, Any] | None = None,
        montage: dict[str, Any] | None = None,
        repair_hint: list[str] | None = None,
        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_path = output_dir / "reference_content_program.json"
    suffix = "_repair" if repair_hint else ""
    input_hash = json_hash({"draft": draft, "resolved": resolved,
                            "ledger": json_hash(ledger),
                            "section_observations": section_observations,
                            "normalization": normalization,
                            "reconciliation": reconciliation,
                            "conflicts": conflicts, "montage": montage,
                            "repair_hint": repair_hint,
                            "prompt": CONTENT_PROGRAM_PROMPT})
    if output_path.is_file() and not force and not repair_hint:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    if resolved.get("required_unresolved_ids"):
        raise V9Blocked("content_program", "required_questions_unresolved",
                        ",".join(resolved["required_unresolved_ids"]))
    conflict_constraints = {
        "must_not_assert": (conflicts or {}).get("must_not_assert") or [],
        "unresolved_topics": (conflicts or {}).get("unresolved_topics") or []}
    draft_payload = dict(draft or {})
    if section_observations:
        # 对账已取代初稿分段：剥离旧 section_drafts，防止模型混用对账前边界。
        draft_payload.pop("section_drafts", None)
    payload = {
        "deterministic_evidence": _compact_evidence(ledger),
        "draft": draft_payload, "question_resolutions": resolved,
        "section_observations": section_observations or {},
        "normalized_shots": normalization or {},
        "boundary_reconciliation": reconciliation or {},
        "conflict_constraints": conflict_constraints,
        "montage_shot_observations": montage or {},
    }
    prefix = CONTENT_PROGRAM_PROMPT
    if repair_hint:
        prefix = (CONTENT_PROGRAM_PROMPT + REPAIR_HEADER +
                  json.dumps(repair_hint, ensure_ascii=False) + "\n")
    value, audit = _ask_object(
        runner, prefix + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")),
        output_dir / "raw_responses" / f"content_program{suffix}.txt",
        stage="content_program")
    _attach_intervals(value, ledger, keys=("meaningful_units", "sections"),
                      stage="content_program")
    # P0.4：continuity_basis 引用维度漏填 evidence_ids 时回填 Section 自身
    # 引用的观察证据（同源弱链接，记录审计），不再赌修复轮能补上。
    known_ids = {str(row.get("evidence_id")) for row in
                 draft.get("observations") or [] if row.get("evidence_id")}
    evidence_backfills: list[dict[str, Any]] = []
    for section in value.get("sections") or []:
        section_ids = [str(item) for item in section.get("evidence_ids") or []
                       if str(item) in known_ids]
        for dimension, support in (section.get("continuity_basis") or {}).items():
            level = (section.get("continuity") or {}).get(dimension)
            if (level in {"required", "preferred"} and
                    not (support.get("evidence_ids") or []) and section_ids):
                support["evidence_ids"] = list(section_ids)
                evidence_backfills.append({
                    "section_id": section.get("section_id"),
                    "dimension": dimension,
                    "basis": "section_level_evidence"})
    if evidence_backfills:
        value["programmatic_evidence_backfill"] = evidence_backfills
    # P0.4：Section 边界以对账结果为权威（p04d 实测：模型重写时抄回旧草稿
    # 边界）。section_id 匹配时程序侧直接覆盖 interval/边界引用，记录对齐。
    observed_by_id = {
        str(row.get("section_id")): row
        for row in (section_observations or {}).get("sections") or []}
    section_alignments: list[dict[str, Any]] = []
    for section in value.get("sections") or []:
        row = observed_by_id.get(str(section.get("section_id") or ""))
        if not row or not isinstance(row.get("source_interval"), list):
            continue
        target = [float(row["source_interval"][0]),
                  float(row["source_interval"][1])]
        current = section.get("interval")
        try:
            same = (len(current) == 2 and
                    round(float(current[0]), 3) == round(target[0], 3) and
                    round(float(current[1]), 3) == round(target[1], 3))
        except (TypeError, ValueError):
            same = False
        if same:
            continue
        section["interval"] = target
        shots = row.get("shots") or []
        if shots:
            section["start_boundary_id"] = shots[0].get("start_boundary_id")
            section["end_boundary_id"] = shots[-1].get("end_boundary_id")
        section_alignments.append({
            "section_id": section.get("section_id"),
            "from": current, "to": target,
            "basis": "reconciliation_authority"})
    if section_alignments:
        value["programmatic_section_alignment"] = section_alignments
    value.update({
        "schema_version": CONTENT_VERSION,
        "input_sha256": input_hash,
        "reference_sha256": ledger["reference"]["sha256"],
        "evidence_ledger_sha256": json_hash(ledger),
        "reference_observations": draft.get("observations") or [],
        "unresolved_questions": resolved.get("questions") or [],
        "probe_history": resolved.get("probe_history") or [],
        "contract_repair": bool(repair_hint),
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
        normalization: dict[str, Any] | None = None,
        conflicts: dict[str, Any] | None = None,
        montage: dict[str, Any] | None = None,
        repair_hint: list[str] | None = None,
        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_path = output_dir / "reference_edit_program.json"
    if not section_observations or not section_observations.get("sections"):
        raise V9Blocked("edit_program", "direct_section_watch_required")
    suffix = "_repair" if repair_hint else ""
    input_hash = json_hash({"content": content, "ledger": json_hash(ledger),
                            "section_observations": section_observations,
                            "normalization": normalization,
                            "conflicts": conflicts, "montage": montage,
                            "repair_hint": repair_hint,
                            "prompt": EDIT_PROGRAM_PROMPT})
    if output_path.is_file() and not force and not repair_hint:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    payload = {"content_program": content,
               "deterministic_evidence": _compact_evidence(ledger),
               "direct_section_watches": section_observations,
               "normalized_shots": normalization or {},
               "conflict_constraints": {
                   "must_not_assert": (conflicts or {}).get("must_not_assert") or []},
               "montage_shot_observations": montage or {}}
    prefix = EDIT_PROGRAM_PROMPT
    if repair_hint:
        prefix = (EDIT_PROGRAM_PROMPT + REPAIR_HEADER +
                  json.dumps(repair_hint, ensure_ascii=False) + "\n")
    value, audit = _ask_object(
        runner, prefix + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")),
        output_dir / "raw_responses" / f"edit_program{suffix}.txt",
        stage="edit_program")
    boundaries = _boundary_map(ledger)
    for operation in value.get("operations") or []:
        start_id = operation.get("start_boundary_id")
        end_id = operation.get("end_boundary_id")
        if start_id is not None or end_id is not None:
            start, end = _require_boundary_pair(operation, boundaries,
                                                stage="edit_program")
            operation["interval"] = [round(start, 6), round(end, 6)]
            operation["time_source"] = "deterministic_boundary_registry"
    # P0.2 修复 C：模式配套常量与矛盾模式的确定性矫正（记录依据，不静默）。
    normalized_by_section = {
        str(row.get("section_id")): row
        for row in (normalization or {}).get("sections") or []}
    programmatic_alignments: list[dict[str, Any]] = []
    # P0.4：operation_type 误填 composition_mode 枚举（如 text_led_montage）
    # 是两套枚举的类别错误；蒙太奇在 operation 层就是 hard_cut 拼接。
    for index, operation in enumerate(value.get("operations") or []):
        op_type = str(operation.get("operation_type") or "")
        if op_type in COMPOSITION_MODES:
            operation["operation_type"] = "hard_cut"
            programmatic_alignments.append({
                "section_id": str(operation.get("section_id") or ""),
                "field": f"operations[{index}].operation_type",
                "from": op_type, "to": "hard_cut",
                "basis": "composition_mode_misused_as_operation_type"})
    for pattern in value.get("editorial_patterns") or []:
        section_id = str(pattern.get("section_id") or "")
        mode = str(pattern.get("composition_mode") or "")
        row = normalized_by_section.get(section_id) or {}
        if mode == "event_compression_montage":
            if pattern.get("source_continuity") != "non_contiguous_allowed":
                pattern["source_continuity"] = "non_contiguous_allowed"
                programmatic_alignments.append(
                    {"section_id": section_id, "field": "source_continuity",
                     "basis": "mode_mandated_constant"})
            if pattern.get("ordering_constraint") != "preserve_event_progression":
                pattern["ordering_constraint"] = "preserve_event_progression"
                programmatic_alignments.append(
                    {"section_id": section_id, "field": "ordering_constraint",
                     "basis": "mode_mandated_constant"})
            phases = [str(item).strip() for item in
                      pattern.get("semantic_phases") or [] if str(item).strip()]
            if len(phases) < 2:
                shot_count = max(len(row.get("content_shots") or []), 2)
                pattern["semantic_phases"] = [
                    f"observed_moment_{index}" for index in range(1, shot_count + 1)]
                programmatic_alignments.append(
                    {"section_id": section_id, "field": "semantic_phases",
                     "basis": "derived_from_observed_content_shots"})
        if mode == "dialogue_compression":
            pattern["semantic_continuity"] = "required"
        content_shot_count = len(row.get("content_shots") or [])
        has_real_cuts = bool(row.get("real_cut_pts"))
        if mode == "continuous_clip" and (content_shot_count > 1 or has_real_cuts):
            same_event_majority = sum(
                1 for shot in row.get("content_shots") or []
                if shot.get("event_relation") == "same_event") > content_shot_count / 2
            # P0.4（用户规则）：同一事件多镜头关键瞬间 = event_compression_montage；
            # 并列不同事件 = evidence_montage。模式由镜头结构派生，不信 LLM 直判。
            replacement = ("event_compression_montage" if same_event_majority
                           else "evidence_montage")
            pattern["composition_mode"] = replacement
            mode = replacement
            programmatic_alignments.append({
                "section_id": section_id, "field": "composition_mode",
                "from": "continuous_clip", "to": replacement,
                "basis": "derived_from_shot_sequence"})
        if mode in MONTAGE_LIKE_MODES:
            shot_count = len(row.get("content_shots") or [])
            low = max(2, math.ceil(0.6 * shot_count)) if shot_count else 2
            high = max(low, shot_count)
            current = pattern.get("snippet_count_range")
            if not (isinstance(current, list) and len(current) == 2 and
                    isinstance(current[0], int) and current[0] >= 2 and
                    isinstance(current[1], int) and current[1] >= current[0]):
                pattern["snippet_count_range"] = [low, high]
                programmatic_alignments.append({
                    "section_id": section_id, "field": "snippet_count_range",
                    "from": current, "to": [low, high],
                    "basis": "aligned_to_observed_content_shot_count"})
    value["programmatic_mode_alignment"] = programmatic_alignments
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
        "contract_repair": bool(repair_hint),
        "direct_section_watch_sha256": json_hash(section_observations),
        "section_rewatch_refs": [
            {"section_id": row["section_id"],
             "raw_response_sha256": row["raw_response_sha256"],
             "shot_ids": [shot["shot_id"] for shot in row["shots"]]}
            for row in section_observations["sections"]],
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
                                  output_dir: Path,
                                  normalization: dict[str, Any] | None = None
                                  ) -> dict[str, Any]:
    """Compile source-agnostic requirements; never copy reference-specific facts."""
    patterns = {str(row.get("section_id")): row
                for row in edit.get("editorial_patterns") or []}
    normalized_by_section = {
        str(row.get("section_id")): row
        for row in (normalization or {}).get("sections") or []}
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
                "reference_content_shot_count": (
                    len((normalized_by_section.get(section_id) or {})
                        .get("content_shots") or [])),
                "reference_transition_count": (
                    len((normalized_by_section.get(section_id) or {})
                        .get("transition_segments") or [])),
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
                   ledger: dict[str, Any], section_observations: dict[str, Any] | None,
                   errors: list[str], *,
                   recomputed: dict[str, Any] | None = None,
                   warnings: list[str] | None = None) -> None:
    if edit.get("schema_version") != EDIT_VERSION:
        errors.append("edit schema_version invalid")
    section_ids = {str(row.get("section_id")) for row in content.get("sections") or []}
    direct_sections = {str(row.get("section_id")): row for row in
                       (section_observations or {}).get("sections") or []}
    direct_refs = {str(row.get("section_id")): row for row in
                   edit.get("section_rewatch_refs") or []}
    if (not section_observations or
            section_observations.get("reference_sha256") !=
            ledger["reference"]["sha256"] or
            edit.get("direct_section_watch_sha256") != json_hash(section_observations)):
        errors.append("edit direct Section watch missing or hash mismatch")
    if set(direct_sections) != section_ids or set(direct_refs) != section_ids:
        errors.append("edit direct Section coverage incomplete")
    for section_id in section_ids & set(direct_sections) & set(direct_refs):
        observed = direct_sections[section_id]
        recorded = direct_refs[section_id]
        shot_ids = {str(shot.get("shot_id")) for shot in observed.get("shots") or []}
        if (not observed.get("raw_response_sha256") or
                recorded.get("raw_response_sha256") !=
                observed.get("raw_response_sha256") or
                set(recorded.get("shot_ids") or []) != shot_ids):
            errors.append(f"edit Section {section_id} rewatch reference invalid")
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
        normalized_section = {
            str(row.get("section_id")): row
            for row in (recomputed or {}).get("sections") or {}}
        normalized_row = normalized_section.get(section_id) or {}
        content_ids = {str(shot.get("shot_id")) for shot in
                       normalized_row.get("content_shots") or []}
        transition_ids = {str(row.get("transition_id")) for row in
                          normalized_row.get("transition_segments") or []}
        if recomputed is not None:
            if (not pattern.get("shot_refs") or
                    not set(pattern["shot_refs"]).issubset(content_ids)):
                errors.append(
                    f"editorial_patterns[{index}] normalized content shot refs missing")
            bad_transitions = (set(pattern.get("transition_refs") or []) -
                               transition_ids)
            if bad_transitions:
                errors.append(
                    f"editorial_patterns[{index}] unknown transition refs: "
                    f"{sorted(bad_transitions)}")
            content_count = len(set(pattern.get("shot_refs") or []) & content_ids)
            has_structure = (len(normalized_row.get("content_shots") or []) > 1 or
                             bool(normalized_row.get("real_cut_pts")))
            mode = pattern.get("composition_mode")
            if has_structure and mode == "continuous_clip":
                errors.append(
                    f"editorial_patterns[{index}] continuous_clip contradicts "
                    "observed real cuts/shots")
            if (not has_structure and content_count <= 1 and
                    mode != "continuous_clip" and warnings is not None):
                warnings.append(
                    f"editorial_patterns[{index}] montage mode with single "
                    "content shot (keyframe extraction)")
        else:
            observed_shots = {str(shot.get("shot_id")) for shot in
                              direct_sections.get(section_id, {}).get("shots") or []}
            if (not pattern.get("shot_refs") or
                    not set(pattern["shot_refs"]).issubset(observed_shots)):
                errors.append(f"editorial_patterns[{index}] direct shot refs missing")
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
                                ledger: dict[str, Any],
                                section_observations: dict[str, Any] | None = None,
                                reconciliation: dict[str, Any] | None = None,
                                conflicts: dict[str, Any] | None = None,
                                montage: dict[str, Any] | None = None,
                                normalization: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    recomputed = (apply_montage_reclassification(
        normalize_section_shots(section_observations or {}, ledger), montage)
        if section_observations else None)
    _validate_content(content, ledger, errors, warnings)
    _validate_edit(edit, content, ledger, section_observations, errors,
                   recomputed=recomputed, warnings=warnings)
    _validate_requirements(requirements, content, edit, errors)
    _validate_p02_structure(content, edit, requirements, ledger,
                            section_observations, recomputed, reconciliation,
                            conflicts, normalization, errors, warnings)
    usability = _validate_narrative_usability(content, edit, requirements,
                                              reconciliation, conflicts,
                                              normalization, errors)
    for key, value in usability.items():
        if value is False:
            errors.append(f"narrative_usability:{key}")
    return {
        "schema_version": "reference_program_validation_v9",
        "passed": not errors, "errors": errors, "warnings": warnings,
        "narrative_usability": usability,
    }


def _validate_narrative_usability(
        content: dict[str, Any], edit_program: dict[str, Any],
        requirements: dict[str, Any],
        reconciliation: dict[str, Any] | None,
        conflicts: dict[str, Any] | None,
        normalization: dict[str, Any] | None,
        errors: list[str]) -> dict[str, bool]:
    """P0.4 Narrative Usability Validation（用户放行标准）。

    边界秒点精度不设门（GEBD 粒度宽容）；只回答"这份理解能不能指导生成"：
    每条对应 P0 人工放行的五条硬线之一。
    """
    functions = [str(section.get("content_function") or "").strip()
                 for section in content.get("sections") or []]
    boundaries_justified = True
    for record in (reconciliation or {}).get("boundaries") or []:
        if record.get("action") in {"kept", "moved"} and not (
                record.get("semantic_change") is True or any(
                    attempt.get("boundary_needed")
                    for attempt in record.get("attempts") or [])):
            boundaries_justified = False
    segments_typed = all(
        isinstance(pattern.get("transition_refs"), list)
        for pattern in edit_program.get("editorial_patterns") or [])
    if normalization:
        segments_typed = segments_typed and all(
            isinstance(section.get("transition_segments"), list) and
            isinstance(section.get("content_shots"), list)
            for section in normalization.get("sections") or [])
    return {
        "sections_have_narrative_purpose": (
            bool(functions) and all(functions) and boundaries_justified),
        "montage_not_misread_as_continuous": not any(
            "continuous_clip contradicts" in error for error in errors),
        "segments_typed_content_transition_overlay": segments_typed,
        "facts_reconciled_no_open_contradiction": not bool(
            (conflicts or {}).get("unresolved_topics")),
        "material_requirements_searchable": bool(
            requirements.get("requirements")) and not any(
            "leaked" in error for error in errors),
    }


def _normalized_text(value: str) -> str:
    return re.sub(r"[\s，。！？、,.!?：:；;\"'“”‘’()（）\-—]+", "", str(value or ""))


def _validate_p02_structure(
        content: dict[str, Any], edit_program: dict[str, Any],
        requirements: dict[str, Any],
        ledger: dict[str, Any],
        section_observations: dict[str, Any] | None,
        recomputed: dict[str, Any] | None,
        reconciliation: dict[str, Any] | None, conflicts: dict[str, Any] | None,
        normalization: dict[str, Any] | None, errors: list[str],
        warnings: list[str]) -> None:
    """P0.2 结构规则：边界有据、shot 归一一致、冲突禁词、收尾文字引用。"""
    sections = content.get("sections") or []
    if recomputed is not None:
        observed = [
            (str(row.get("section_id")),
             [round(float(value), 6) for value in row.get("source_interval") or []])
            for row in (section_observations or {}).get("sections") or []]
        declared = [
            (str(row.get("section_id")),
             [round(float(value), 6) for value in row.get("interval") or []])
            for row in sections]
        if declared != observed:
            errors.append("content sections diverge from reconciled observed "
                          f"sections: {declared} != {observed}")
    if recomputed is not None and normalization is not None:
        for expected, stored in zip(recomputed.get("sections") or [],
                                    normalization.get("sections") or []):
            if (expected.get("content_shots") != stored.get("content_shots") or
                    expected.get("transition_segments") !=
                    stored.get("transition_segments")):
                errors.append(
                    f"shot normalization drift: {expected.get('section_id')}")
                break
    for index in range(len(sections) - 1):
        boundary_start = float(sections[index + 1]["interval"][0])
        record = None
        for row in (reconciliation or {}).get("boundaries") or []:
            pts = _boundary_map(ledger).get(str(row.get("boundary_id")))
            moved_pts = (_boundary_map(ledger).get(str(row.get("moved_to")))
                         if row.get("action") == "moved" else None)
            if ((pts is not None and abs(pts - boundary_start) < 1e-6) or
                    (moved_pts is not None and
                     abs(moved_pts - boundary_start) < 1e-6)):
                record = row
                break
        if record is None:
            errors.append(
                f"section boundary {sections[index + 1]['section_id']} lacks "
                "reconciliation justification")
        elif record.get("semantic_change") is not True and record.get(
                "action") != "moved":
            errors.append(
                f"section boundary {sections[index + 1]['section_id']} "
                "unjustified: no semantic change at boundary")
    terms = [str(term) for term in (conflicts or {}).get("must_not_assert") or []
             if str(term).strip()]
    if terms:
        for name, program in (("content", content), ("edit", None),
                              ("requirements", requirements)):
            if program is None:
                continue
            text = json.dumps(program, ensure_ascii=False)
            for term in terms:
                if term in text:
                    errors.append(f"{name} asserts contested term: {term}")
    by_section = {str(row.get("section_id")): row
                  for row in (recomputed or {}).get("sections") or []}
    last = sections[-1] if sections else None
    if last is not None:
        ocr_events = _section_evidence(ledger, last["interval"])["ocr_text_events"]
        ocr_texts = [str(row.get("text") or "") for row in ocr_events
                     if str(row.get("text") or "").strip()]
        if ocr_texts:
            quote = _normalized_text(last.get("final_text_quote"))
            if not quote:
                errors.append(
                    f"{last['section_id']} final_text_quote missing despite "
                    "on-screen text")
            elif not any(quote in _normalized_text(text) or
                         _normalized_text(text) in quote for text in ocr_texts):
                errors.append(
                    f"{last['section_id']} final_text_quote not verbatim "
                    "on-screen text")
    # P0.3：第一个含屏幕文字的 Section 必须保留 Hook 命题（逐字引用）。
    for first in sections:
        first_ocr = [str(row.get("text") or "") for row in
                     _section_evidence(ledger, first["interval"])[
                         "ocr_text_events"]
                     if str(row.get("text") or "").strip()]
        if first_ocr:
            quote = _normalized_text(first.get("hook_text_quote"))
            if not quote:
                errors.append(
                    f"{first['section_id']} hook_text_quote missing despite "
                    "on-screen assertion text")
            elif not any(quote in _normalized_text(text) or
                         _normalized_text(text) in quote
                         for text in first_ocr):
                errors.append(
                    f"{first['section_id']} hook_text_quote not verbatim "
                    "on-screen text")
            else:
                takeaway = _normalized_text(first.get("audience_takeaway"))
                refers_proposition = (
                    any(takeaway.find(quote[index:index + 4]) >= 0
                        for index in range(max(len(quote) - 3, 1))) or
                    any(word in takeaway for word in (
                        "偏见", "反驳", "质疑", "推翻", "命题", "刻板", "认为",
                        "觉得", "成见")))
                if not refers_proposition:
                    errors.append(
                        f"{first['section_id']} hook takeaway does not present "
                        "the quoted proposition (reads as introduction)")
            break
    # P0.3：蒙太奇类模式 snippet 下限必须 >=2，与参考镜头结构一致。
    patterns_by_section = {
        str(row.get("section_id")): row
        for row in (edit_program or {}).get("editorial_patterns") or []}
    for section in sections:
        pattern = patterns_by_section.get(str(section["section_id"]))
        if pattern is None:
            continue
        mode = str(pattern.get("composition_mode") or "")
        count = pattern.get("snippet_count_range") or []
        if (mode in MONTAGE_LIKE_MODES and
                (not isinstance(count, list) or len(count) != 2 or
                 not isinstance(count[0], int) or count[0] < 2)):
            errors.append(
                f"{section['section_id']} montage snippet_count_range min "
                "must be >= 2")
    for section in sections:
        row = by_section.get(str(section["section_id"])) or {}
        content_shots = row.get("content_shots") or []
        real_cuts = row.get("real_cut_pts") or []
        if len(content_shots) > 1 or real_cuts:
            warnings.append(
                f"{section['section_id']} has {len(content_shots)} content shots "
                f"and {len(real_cuts)} interior real cuts; montage expected")


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
                       assets: list[dict[str, Any]], *,
                       normalization: dict[str, Any] | None = None,
                       reconciliation: dict[str, Any] | None = None,
                       conflicts: dict[str, Any] | None = None,
                       montage: dict[str, Any] | None = None) -> None:
    patterns = {str(row.get("section_id")): row
                for row in edit.get("editorial_patterns") or []}
    reqs = {str(row.get("section_id")): row
            for row in requirements.get("requirements") or []}
    assets_by_id = {str(row.get("section_id")): row for row in assets}
    normalized_by_id = {str(row.get("section_id")): row
                        for row in (normalization or {}).get("sections") or []}
    montage_by_id = {str(row.get("section_id")): row
                     for row in (montage or {}).get("sections") or []}
    lines = [
        "# V9 Reference Program 人工审核表", "",
        "自动验证通过不等于交付通过。请逐段观看片段与关键帧，再填写 human_review.template.json。",
        "",
        "## P0.4 审核辅助（放行标准=生成可用，非边界秒点精度）", "",
        "- 边界叙事功能对账：" + json.dumps(
            [{"boundary": row.get("boundary_id"),
              "action": row.get("action"), "moved_to": row.get("moved_to"),
              "functions": [
                  {"candidate": attempt.get("boundary_id"),
                   "before": attempt.get("before_function"),
                   "after": attempt.get("after_function"),
                   "boundary_needed": attempt.get("boundary_needed")}
                  for attempt in row.get("attempts") or []]}
             for row in (reconciliation or {}).get("boundaries") or []],
            ensure_ascii=False),
        "- 语义冲突门：" + json.dumps(
            {"material_contradictions": len(
                (conflicts or {}).get("contradictions") or []),
             "unresolved_topics": (conflicts or {}).get("unresolved_topics") or [],
             "must_not_assert": (conflicts or {}).get("must_not_assert") or []},
            ensure_ascii=False),
        "",
    ]
    review_rows = []
    for section in content.get("sections") or []:
        section_id = str(section["section_id"])
        pattern = patterns.get(section_id) or {}
        asset = assets_by_id.get(section_id) or {}
        normalized = normalized_by_id.get(section_id) or {}
        montage_texts = [
            row.get("on_screen_text") for row in
            (montage_by_id.get(section_id) or {}).get("segments") or []
            if row.get("on_screen_text")]
        content_count = len(normalized.get("content_shots") or [])
        transition_count = len(normalized.get("transition_segments") or [])
        real_cuts = json.dumps(normalized.get("real_cut_pts") or [],
                               ensure_ascii=False)
        transition_types = json.dumps(
            [row.get("transition_type") for row in
             normalized.get("transition_segments") or []], ensure_ascii=False)
        lines.extend([
            f"## {section_id}", "",
            f"- 区间：{section.get('interval')}",
            f"- 观众所得：{section.get('audience_takeaway', '')}",
            f"- 认知变化：{json.dumps(section.get('cognition_change') or {}, ensure_ascii=False)}",
            f"- 连续性：{json.dumps(section.get('continuity') or {}, ensure_ascii=False)}",
            f"- 连续性依据：{json.dumps(section.get('continuity_basis') or {}, ensure_ascii=False)}",
            f"- Composition：{pattern.get('composition_mode', 'missing')}",
            f"- 归一化镜头：{content_count} 个内容镜头 / {transition_count} 个转场段"
            f"（内部真切点 {real_cuts}）",
            f"- 转场类型：{transition_types}",
            f"- 逐镜头屏幕文字：{json.dumps(montage_texts, ensure_ascii=False)}",
            f"- 收尾文字引用：{section.get('final_text_quote', '')}",
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
        "schema_version": HUMAN_REVIEW_VERSION,
        "review_binding": _review_binding(output_dir),
        "reviewer": "", "sections": review_rows,
        "continuity_resolutions": [],
        "requirement_transfer_test": {"passed": None, "notes": ""},
        "overall_notes": "",
    })


def _review_binding(output_dir: Path) -> dict[str, Any]:
    """Bind a human signature to the exact source and Program snapshot."""
    output_dir = Path(output_dir)
    ledger = json.loads((output_dir / "reference_evidence.json").read_text(
        encoding="utf-8"))
    return {
        "reference_sha256": ledger["reference"]["sha256"],
        "evidence_ledger_sha256": sha256_file(output_dir / "reference_evidence.json"),
        "content_program_sha256": sha256_file(output_dir / "reference_content_program.json"),
        "edit_program_sha256": sha256_file(output_dir / "reference_edit_program.json"),
        "material_requirements_sha256": sha256_file(output_dir / "material_requirements.json"),
    }


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


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return (json.loads(path.read_text(encoding="utf-8"))
            if path.is_file() else None)


def accept_reference_programs(output_dir: Path, human_review: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    automatic = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    if not automatic.get("automated_passed"):
        raise V9Blocked("human_acceptance", "automatic_validation_not_passed")
    review = json.loads(Path(human_review).read_text(encoding="utf-8"))
    current_binding = _review_binding(output_dir)
    if (review.get("schema_version") != HUMAN_REVIEW_VERSION or
            review.get("review_binding") != current_binding):
        return _write_acceptance(
            output_dir, automated_passed=True, decision="BLOCKED",
            failure_stage="human", reason_code="stale_human_review",
            detail="Review schema or signed Program snapshot does not match current files",
            failure_class="verification", human_passed=False)
    content = json.loads((output_dir / "reference_content_program.json").read_text(
        encoding="utf-8"))
    edit = json.loads((output_dir / "reference_edit_program.json").read_text(
        encoding="utf-8"))
    requirements = json.loads((output_dir / "material_requirements.json").read_text(
        encoding="utf-8"))
    ledger = json.loads((output_dir / "reference_evidence.json").read_text(
        encoding="utf-8"))
    sections = json.loads((output_dir / "section_observations.json").read_text(
        encoding="utf-8"))
    p02 = {
        "reconciliation": _load_optional_json(
            output_dir / "boundary_reconciliation.json"),
        "conflicts": _load_optional_json(
            output_dir / "semantic_conflicts.json"),
        "montage": _load_optional_json(
            output_dir / "montage_shot_observations.json"),
        "normalization": _load_optional_json(
            output_dir / "shot_normalization.json")}
    current_validation = validate_reference_programs(
        content, edit, requirements, ledger, sections, **p02)
    if not current_validation["passed"]:
        return _write_acceptance(
            output_dir, automated_passed=False, decision="BLOCKED",
            failure_stage="validation", reason_code="current_program_validation_failed",
            detail="; ".join(current_validation["errors"]),
            validation=current_validation, human_passed=False,
            failure_class="verification")
    amendments = review.get("continuity_resolutions") or []
    amendment_errors = []
    amended = False
    updated_validation = current_validation
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
            edit["content_program_sha256"] = json_hash(content)
            proposed = compile_material_requirements(
                content, edit, output_dir / "human_review_draft",
                normalization=p02.get("normalization"))
            checked = validate_reference_programs(content, edit, proposed, ledger,
                                                  sections, **p02)
            if checked["passed"]:
                _write_json(output_dir / "reference_content_program.json", content)
                _write_json(output_dir / "reference_edit_program.json", edit)
                _write_json(output_dir / "material_requirements.json", proposed)
                _write_json(output_dir / "validation.json", checked)
                asset_manifest = output_dir / "review_assets" / "manifest.json"
                assets = (json.loads(asset_manifest.read_text(
                    encoding="utf-8")).get("sections") or []
                    if asset_manifest.is_file() else [])
                _write_review_pack(output_dir, content, edit, proposed, assets,
                                   normalization=p02["normalization"],
                                   reconciliation=p02["reconciliation"],
                                   conflicts=p02["conflicts"],
                                   montage=p02["montage"])
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
            reference, draft, ledger, output_dir, runner=runner,
            ffmpeg_bin=ffmpeg_bin, force=force)
        manifest["stages"]["section_watch"] = {
            "status": "complete", "sha256": json_hash(section_observations)}
        _snapshot_stage(output_dir, "p0_b", section_observations["input_sha256"],
                        [output_dir / "section_observations.json", *sorted(
                            (output_dir / "raw_responses").glob("section_*.txt"))])
        reconciliation, section_observations = reconcile_section_boundaries(
            reference, ledger, section_observations, output_dir,
            runner=runner, force=force, draft=draft)
        manifest["stages"]["boundary_reconciliation"] = {
            "status": "complete",
            "moved": [row["boundary_id"] for row in reconciliation["boundaries"]
                      if row.get("action") == "moved"],
            "rewatched": reconciliation.get("rewatched") or []}
        _snapshot_stage(output_dir, "p0_b1", reconciliation["input_sha256"],
                        [output_dir / "boundary_reconciliation.json",
                         output_dir / "section_observations.json"])
        normalization = normalize_section_shots(section_observations, ledger)
        montage = watch_fast_montage_shots(
            reference, ledger, normalization, output_dir, runner=runner,
            ffmpeg_bin=ffmpeg_bin, force=force)
        normalization = apply_montage_reclassification(normalization, montage)
        _write_json(output_dir / "shot_normalization.json", normalization)
        manifest["stages"]["shot_normalization"] = {
            "status": "complete",
            "sections": {
                str(row["section_id"]): {
                    "content_shots": len(row.get("content_shots") or []),
                    "transitions": len(row.get("transition_segments") or [])}
                for row in normalization.get("sections") or []}}
        manifest["stages"]["montage_watch"] = {
            "status": "complete",
            "sections": ([str(row["section_id"])
                          for row in montage.get("sections") or []]
                         if montage.get("sections") else [])}
        _snapshot_stage(output_dir, "p0_b2", json_hash(
            {"normalization": normalization, "montage": montage}),
            [output_dir / "shot_normalization.json",
             output_dir / "montage_shot_observations.json"])
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
        conflicts = semantic_conflict_gate(
            reference, draft, ledger, section_observations, resolved,
            output_dir, runner=runner, force=force)
        manifest["stages"]["conflict_gate"] = {
            "status": "complete",
            "material_contradictions": len(conflicts.get("contradictions") or []),
            "unresolved_topics": conflicts.get("unresolved_topics") or [],
            "must_not_assert": conflicts.get("must_not_assert") or []}
        _snapshot_stage(output_dir, "p0_b3", conflicts["input_sha256"],
                        [output_dir / "semantic_conflicts.json"])
        content = build_reference_content_program(
            draft, resolved, ledger, output_dir, runner=runner,
            section_observations=section_observations,
            normalization=normalization, reconciliation=reconciliation,
            conflicts=conflicts, montage=montage, force=force)
        edit = build_reference_edit_program(
            content, ledger, output_dir, runner=runner,
            section_observations=section_observations,
            normalization=normalization, conflicts=conflicts, montage=montage,
            force=force)
        requirements = compile_material_requirements(
            content, edit, output_dir, normalization=normalization)
        validation = validate_reference_programs(
            content, edit, requirements, ledger, section_observations,
            reconciliation=reconciliation, conflicts=conflicts, montage=montage,
            normalization=normalization)
        if not validation["passed"]:
            # P0.2：一轮有界契约修复——把错误清单回喂模型各重问一次（extract_template 同款）。
            # 修复后两版候选择优：取契约错误更少的一版（防“修好A弄坏B”，全程留档）。
            # p04d 实测：泄漏类错误的原文含参考专有词，直接回喂等于教模型照抄——
            # 先把禁词泛化成计数再进 hint。
            repair_hint = _sanitize_repair_hint(list(validation["errors"]))
            first_candidate = (content, edit, requirements, validation)
            content = build_reference_content_program(
                draft, resolved, ledger, output_dir, runner=runner,
                section_observations=section_observations,
                normalization=normalization, reconciliation=reconciliation,
                conflicts=conflicts, montage=montage,
                repair_hint=repair_hint, force=True)
            edit = build_reference_edit_program(
                content, ledger, output_dir, runner=runner,
                section_observations=section_observations,
                normalization=normalization, conflicts=conflicts,
                montage=montage, repair_hint=repair_hint, force=True)
            requirements = compile_material_requirements(
            content, edit, output_dir, normalization=normalization)
            validation = validate_reference_programs(
                content, edit, requirements, ledger, section_observations,
                reconciliation=reconciliation, conflicts=conflicts,
                montage=montage, normalization=normalization)
            selection = "repair"
            if (not validation["passed"] and
                    len(validation["errors"]) > len(first_candidate[3]["errors"])):
                content, edit, requirements, validation = first_candidate
                selection = "first_pass_fewer_errors"
                compile_material_requirements(
                    content, edit, output_dir, normalization=normalization)
                _write_json(output_dir / "reference_content_program.json", content)
                _write_json(output_dir / "reference_edit_program.json", edit)
            manifest["stages"]["program_repair"] = {
                "status": "complete", "rounds": 1,
                "hint_errors": repair_hint,
                "passed_after_repair": validation["passed"],
                "candidate_selection": selection,
                "first_pass_errors": len(first_candidate[3]["errors"]),
                "repaired_errors": len(validation["errors"]) if selection == "repair"
                else None}
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
             output_dir / "boundary_reconciliation.json",
             output_dir / "shot_normalization.json",
             output_dir / "montage_shot_observations.json",
             output_dir / "semantic_conflicts.json",
             output_dir / "raw_responses" / "content_program.txt",
             output_dir / "raw_responses" / "edit_program.txt"])
        if not validation["passed"]:
            raise V9Blocked("validation", "reference_program_contract_failed",
                            "; ".join(validation["errors"]))
        assets = _build_review_assets(reference, content, output_dir,
                                      ffmpeg_bin=ffmpeg_bin)
        _write_review_pack(output_dir, content, edit, requirements, assets,
                           normalization=normalization,
                           reconciliation=reconciliation,
                           conflicts=conflicts, montage=montage)
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
