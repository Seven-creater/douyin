"""V7 reference-driven, target-centered perception and micro-edit experiment.

This module deliberately contains no detector, tracker or ReID dependency.
Models propose observations; deterministic code owns media bounds, hashes,
identity consistency contracts, escalation and native-frame PTS mapping.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.agentic_video.evidence_units import EvidenceUnitV3
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.renderer import render_micro_montage
from src.config import AppConfig, repo_root
from src.perception import common
from src.perception.detect_shots import detect_shots
from src.perception.omni_runner import cut_clip


ALLOWED_BROWSE_RELATIONS = {
    "target_direct", "related_interaction", "possible_outcome", "uncertain",
    "not_observed",
}
REQUIRED_GOALS = ("adversity", "agency", "outcome")
ARM_CONTRACT = {
    "A": {"fps": 2.0, "retention_ratio": 0.10, "backend": "flashvid"},
    "B": {"fps": 4.0, "retention_ratio": 0.10, "backend": "flashvid"},
    "C": {"fps": 4.0, "retention_ratio": 0.25, "backend": "flashvid"},
    "D": {"fps": 4.0, "retention_ratio": 1.00, "backend": "native_bypass"},
}
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SHOWINFO = re.compile(r"pts_time:([+-]?\d+(?:\.\d+)?)")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class V7Blocked(RuntimeError):
    def __init__(self, failure_stage: str, reason_code: str, message: str = ""):
        super().__init__(message or reason_code)
        self.failure_stage = failure_stage
        self.reason_code = reason_code


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_json(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise V7Blocked("model", "invalid_json_response")
        try:
            value = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError as exc:
            raise V7Blocked("model", "invalid_json_response", str(exc)) from exc
    if not isinstance(value, dict):
        raise V7Blocked("model", "json_response_not_object")
    return value


def _parse_candidate_response(text: str) -> list[dict[str, Any]]:
    """Accept the requested object and the common bare-list model variant."""
    candidate = str(text or "").strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
        for opening, closing in (("{", "}"), ("[", "]")):
            start, end = candidate.find(opening), candidate.rfind(closing)
            if start < 0 or end <= start:
                continue
            try:
                parsed = json.loads(candidate[start:end + 1])
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            raise V7Blocked("model", "invalid_json_response")
        value = parsed
    rows = value.get("candidates") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise V7Blocked("model", "json_candidates_not_list")
    return rows


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root() / value


def _safe_id(value: Any, fallback: str) -> str:
    result = _SAFE_ID.sub("_", str(value or fallback)).strip("._")
    return result[:80] or fallback


def read_v7_spec(path: Path) -> tuple[dict[str, Any], str]:
    spec = _read_json(path)
    if spec.get("spec_version") != "target_microcut_v7":
        raise ValueError("V7 spec_version must be target_microcut_v7")
    scope = spec.get("source_scope") or {}
    if float(scope.get("end_s", 0)) <= float(scope.get("start_s", 0)):
        raise ValueError("V7 source_scope is invalid")
    if abs(float((spec.get("trusted_seed") or {}).get("source_time_s", -1)) - 5391.5) > 1e-6:
        raise ValueError("V7 trusted seed must be source time 5391.5s")
    arms = spec.get("browse_arms") or {}
    for arm, contract in ARM_CONTRACT.items():
        row = arms.get(arm) or {}
        for key, expected in contract.items():
            actual = row.get(key)
            if isinstance(expected, float):
                if abs(float(actual) - expected) > 1e-6:
                    raise ValueError(f"arm {arm} {key} violates V7 contract")
            elif actual != expected:
                raise ValueError(f"arm {arm} {key} violates V7 contract")
    raw = Path(path).read_bytes()
    return spec, hashlib.sha256(raw).hexdigest()


def collect_flashvid_runtime(runtime_path: Path) -> dict[str, Any]:
    """Read, but never mutate, the external custom-port worktree provenance."""
    runtime_path = Path(runtime_path)
    result: dict[str, Any] = {"path": str(runtime_path), "exists": runtime_path.is_dir()}
    if not runtime_path.is_dir():
        return result
    try:
        head = subprocess.run(
            ["git", "-C", str(runtime_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(runtime_path), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=30).stdout
        result.update({"git_head": head, "dirty": bool(status.strip()),
                       "dirty_entry_count": len(status.splitlines())})
    except (OSError, subprocess.SubprocessError) as exc:
        result["git_error"] = f"{type(exc).__name__}: {exc}"
    core = (
        "src/flashvid_vllm/model.py", "src/flashvid/compression.py",
        "src/flashvid_eval/client.py", "scripts/launch_flashvid_budget_bank.sh",
    )
    result["core_file_sha256"] = {
        relative: sha256_file(runtime_path / relative)
        for relative in core if (runtime_path / relative).is_file()
    }
    return result


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def _rate(value: Any) -> float | None:
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def prepare_reference_task(cfg: AppConfig, experiment_spec: dict[str, Any],
                           output_dir: Path, *, vanilla_client) -> dict[str, Any]:
    """Measure cuts deterministically and ask vanilla Qwen only for semantics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _resolve(experiment_spec["reference_video"])
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    ffprobe = cfg.perception.get("ffprobe_bin", "ffprobe")
    duration = common.video_duration_s(ffprobe, reference)
    try:
        probe = common.run_ffprobe_json(ffprobe, reference)
        video_stream = next(row for row in probe.get("streams") or []
                            if row.get("codec_type") == "video")
        reference_fps = _rate(video_stream.get("avg_frame_rate") or
                              video_stream.get("r_frame_rate"))
        reference_frames = (int(video_stream["nb_frames"])
                            if video_stream.get("nb_frames") is not None else None)
    except (common.FFmpegError, OSError, StopIteration, TypeError, ValueError):
        reference_fps, reference_frames = None, None
    shot_cfg = experiment_spec.get("reference_analysis") or {}
    detected = detect_shots(
        reference, ffmpeg_bin=ffmpeg,
        threshold=float(shot_cfg.get("scene_threshold", 0.3)),
        min_shot_len_s=float(shot_cfg.get("min_shot_len_s", 0.12)),
        duration_s=duration)
    prompt = (
        "Analyze this short reference video at full semantic fidelity. Do not infer cut "
        "times: deterministic shot rows are supplied below. Explain the actual editing "
        "sections and which evidence goals each section serves. Do not invent a fixed "
        "four-slot grammar. Return JSON only as {\"perception_center\":{\"kind\":"
        "\"subject\",\"target_id\":\"S0\"},\"edit_sections\":[{\"id\":\"...\","
        "\"shot_indices\":[0],\"purpose\":\"...\",\"goal_ids\":[\"adversity\"]}]} .\n"
        f"Deterministic shots: {json.dumps(detected['shots'], ensure_ascii=False)}"
    )
    answer = vanilla_client.watch(reference, prompt, duration_s=duration)
    raw_path = output_dir / "reference_semantics_raw.txt"
    raw_path.write_text(answer.text, encoding="utf-8")
    semantic = _parse_json(answer.text)
    sections = []
    shots = detected["shots"]
    for index, row in enumerate(semantic.get("edit_sections") or []):
        indices = sorted({int(value) for value in row.get("shot_indices") or []
                          if 0 <= int(value) < len(shots)})
        if not indices:
            continue
        goals = [str(value) for value in row.get("goal_ids") or []
                 if str(value) in REQUIRED_GOALS]
        sections.append({
            "id": str(row.get("id") or f"section_{index:02d}"),
            "purpose": str(row.get("purpose") or "").strip(),
            "shot_indices": indices,
            "source_interval": [shots[indices[0]]["start_s"],
                                shots[indices[-1]]["end_s"]],
            "goal_ids": goals,
        })
    if not sections:
        raise V7Blocked("reference", "reference_sections_not_observed")
    durations = [float(row["duration_s"]) for row in shots]
    result = {
        "schema_version": "reference_task_v2",
        "reference_video": str(reference),
        "reference_sha256": sha256_file(reference),
        "perception_center": {"kind": "subject", "target_id": "S0"},
        "evidence_goals": [{"id": goal, "required": True} for goal in REQUIRED_GOALS],
        "consistency_rules": {
            "adversity_and_agency_require_target_visible": True,
            "outcome_requires_verified_relation": True,
        },
        "edit_sections": sections,
        "measured_style": {
            "reference_duration_s": round(duration, 6),
            "reference_fps": round(reference_fps, 6) if reference_fps else None,
            "reference_frame_count": reference_frames,
            "shot_intervals": [[row["start_s"], row["end_s"]] for row in shots],
            "shot_duration_p50_s": round(statistics.median(durations), 6),
            "shot_duration_p90_s": round(_percentile(durations, 0.9), 6),
            "section_proportions": [round((row["source_interval"][1] -
                                                   row["source_interval"][0]) / duration, 6)
                                    for row in sections],
        },
        "provenance": {
            "timing_backend": "ffmpeg_scene_showinfo",
            "semantic_backend": "Qwen3.5-4B_native_bypass",
            "flashvid_loaded": False,
            "requested_fps": 4.0,
            "request_audit": getattr(answer, "request_audit", {}),
        },
    }
    _write_json(output_dir / "reference_shots.json", detected)
    _write_json(output_dir / "reference_task.json", result)
    return result


def export_frame(ffmpeg_bin: str, source_video: Path, timestamp_s: float,
                 destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-loglevel", "error", "-ss", f"{float(timestamp_s):.6f}",
        "-i", str(source_video), "-vf", "scale='min(1280,iw)':-2",
        "-frames:v", "1", "-q:v", "2", str(destination),
    ], timeout_s=180)
    return destination


def export_roi(ffmpeg_bin: str, full_frame: Path, roi: Iterable[float],
               destination: Path) -> Path:
    x0, y0, x1, y1 = map(float, roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI must be normalized [x0,y0,x1,y1]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop = f"crop=iw*{x1-x0:.8f}:ih*{y1-y0:.8f}:iw*{x0:.8f}:ih*{y0:.8f},scale=768:-2"
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-loglevel", "error", "-i", str(full_frame), "-vf", crop,
        "-frames:v", "1", str(destination),
    ], timeout_s=120)
    return destination


def _identity_result(text: str) -> str:
    result = str(_parse_json(text).get("result") or "").strip().lower()
    if result not in {"same", "different", "uncertain"}:
        raise V7Blocked("identity", "invalid_identity_result")
    return result


def _compare_pair(runner, left: Mapping[str, Any], right: Mapping[str, Any], *,
                  direction: str, output_dir: Path) -> dict[str, Any]:
    images = [Path(left["full_frame"])]
    if left.get("roi_crop"):
        images.append(Path(left["roi_crop"]))
    images.append(Path(right["full_frame"]))
    if right.get("roi_crop") and len(images) < 4:
        images.append(Path(right["roi_crop"]))
    prompt = (
        "Identity-only comparison. Images for LEFT come first, then RIGHT. ROI crops "
        "only magnify their accompanying full frame and are never standalone truth. "
        "Do not discuss actions, plot coverage or editorial functions. "
        "Return exactly one JSON object: {\"result\":\"same|different|uncertain\"}."
    )
    answer = runner.inspect_media(images, prompt, max_new_tokens=64)
    result = _identity_result(answer.text)
    row = {"direction": direction, "left": left["id"], "right": right["id"],
           "result": result, "images": [str(path) for path in images],
           "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens}
    _write_json(output_dir / f"{direction}.json", row)
    return row


def validate_album_consistency(positives: list[dict[str, Any]],
                               negatives: list[dict[str, Any]], *,
                               compare: Callable[[Mapping[str, Any], Mapping[str, Any], str], str]
                               ) -> list[dict[str, Any]]:
    """Enforce every directed positive/positive and positive/negative edge."""
    if len(positives) < 3 or not negatives:
        raise V7Blocked("identity", "target_album_insufficient_examples")
    checks = []
    for left in positives:
        for right in positives:
            if left["id"] == right["id"]:
                continue
            result = compare(left, right, f"{left['id']}__to__{right['id']}")
            checks.append({"left": left["id"], "right": right["id"],
                           "expected": "same", "result": result})
    for positive in positives:
        for negative in negatives:
            for left, right in ((positive, negative), (negative, positive)):
                result = compare(left, right, f"{left['id']}__to__{right['id']}")
                checks.append({"left": left["id"], "right": right["id"],
                               "expected": "different", "result": result})
    if any(row["result"] != row["expected"] for row in checks):
        raise V7Blocked("identity", "target_album_consistency_failed")
    return checks


def evaluate_target_album(target_album: dict[str, Any], oracle: dict[str, Any]) \
        -> dict[str, Any]:
    """Post-hoc identity audit; held-out labels never enter album construction."""
    labels = list(oracle.get("identity_gt") or [])

    def truth(timestamp: float) -> str | None:
        for row in labels:
            start, end = map(float, row["interval"])
            if start <= timestamp <= end:
                return str(row["label"])
        return None

    false_merges, false_rejections, checked = [], [], []
    for predicted, rows in (("same", target_album.get("positive") or []),
                            ("different", target_album.get("hard_negative") or [])):
        for row in rows:
            if row.get("id") == "seed":
                continue
            actual = truth(float(row["source_time_s"]))
            checked.append({"id": row["id"], "source_time_s": row["source_time_s"],
                            "predicted": predicted, "heldout_label": actual})
            if predicted == "same" and actual == "different":
                false_merges.append(row["id"])
            if predicted == "different" and actual == "same":
                false_rejections.append(row["id"])
    return {"gt_entered_album_prompts": False, "checked": checked,
            "unlabeled_count": sum(row["heldout_label"] is None for row in checked),
            "hard_negative_false_merges": len(false_merges),
            "hard_negative_false_merge_ids": false_merges,
            "positive_false_rejections": len(false_rejections),
            "positive_false_rejection_ids": false_rejections}


def build_target_album(cfg: AppConfig, experiment_spec: dict[str, Any],
                       output_dir: Path, *, trusted_seed: Path,
                       source_video: Path, vanilla_client, runner,
                       candidate_proposals: list[dict[str, Any]] | None = None) \
        -> dict[str, Any]:
    """Expand one trusted visual seed using identity-only proposals/checks."""
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    checks_dir = output_dir / "checks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    seed = {"id": "seed", "source_time_s": 5391.5,
            "full_frame": str(Path(trusted_seed)), "roi_crop": None,
            "seed_sha256": sha256_file(trusted_seed)}
    proposals = candidate_proposals
    proposal_audit: dict[str, Any] = {"source": "injected_control"}
    if proposals is None:
        scope = experiment_spec["source_scope"]
        prompt = (
            "Identity-only album expansion. The first image is the trusted S0 seed. "
            "Find appearances that may show the same visual subject and visually "
            "confusable but different subjects. Do not use plot, action or story coverage. "
            "Report at most three representative candidates total; do not enumerate "
            "sampling frames. If none are visible, return an empty candidates list. "
            "Return JSON only: {\"candidates\":[{\"id\":\"...\","
            "\"relative_time_s\":0.0,\"candidate_class\":\"possible_same|hard_negative|"
            "uncertain\",\"roi\":[0,0,1,1]}]}."
        )
        proposals, proposal_audits = [], []
        raw_dir = output_dir / "proposal_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        windows = transport_windows(float(scope["start_s"]), float(scope["end_s"]),
                                    duration_s=6.0, overlap_s=0.5)
        proposal_clips = {}
        for index, (start, end) in enumerate(windows):
            window_id = f"w{index:03d}"
            proposal_clips[window_id] = cut_clip(
                ffmpeg, source_video, output_dir / "proposal_clips" / window_id,
                start_s=start, end_s=end)

        def propose(index: int, start: float, end: float):
            window_id = f"w{index:03d}"
            answer = vanilla_client.watch(
                proposal_clips[window_id], prompt, duration_s=end - start,
                image_paths=[trusted_seed])
            (raw_dir / f"{window_id}.txt").write_text(answer.text, encoding="utf-8")
            rows = []
            for row in _parse_candidate_response(answer.text):
                normalized_row = dict(row)
                normalized_row["source_time_s"] = start + float(
                    normalized_row.get("relative_time_s", -1))
                normalized_row["id"] = f"{window_id}_{normalized_row.get('id') or 'candidate'}"
                rows.append(normalized_row)
            audit = {"window_id": window_id, "source_interval": [start, end],
                     **dict(getattr(answer, "request_audit", {}) or {})}
            return index, rows, audit

        with ThreadPoolExecutor(max_workers=min(8, len(windows))) as executor:
            futures = [executor.submit(propose, index, start, end)
                       for index, (start, end) in enumerate(windows)]
            ordered = sorted((future.result() for future in as_completed(futures)),
                             key=lambda row: row[0])
        for _index, rows, audit in ordered:
            proposals.extend(rows)
            proposal_audits.append(audit)
        proposal_audit = {"source": "vanilla_transport_windows",
                          "window_count": len(windows), "requests": proposal_audits}
    scope_start = float(experiment_spec["source_scope"]["start_s"])
    normalized = []
    for index, proposal in enumerate(proposals or []):
        timestamp = proposal.get("source_time_s")
        if timestamp is None:
            timestamp = scope_start + float(proposal.get("relative_time_s", -1))
        timestamp = float(timestamp)
        scope_end = float(experiment_spec["source_scope"]["end_s"])
        if not scope_start <= timestamp <= scope_end:
            continue
        candidate_id = _safe_id(proposal.get("id"), f"candidate_{index:02d}")
        full = export_frame(ffmpeg, source_video, timestamp,
                            frames_dir / f"{candidate_id}_full.jpg")
        crop = None
        if proposal.get("roi") is not None:
            crop = export_roi(ffmpeg, full, proposal["roi"],
                              frames_dir / f"{candidate_id}_roi.jpg")
        normalized.append({
            "id": candidate_id, "source_time_s": timestamp,
            "candidate_class": str(proposal.get("candidate_class") or "uncertain"),
            "full_frame": str(full), "roi_crop": str(crop) if crop else None,
        })
    positive_candidates = [row for row in normalized
                           if row["candidate_class"] == "possible_same"]
    negatives = [row for row in normalized
                 if row["candidate_class"] == "hard_negative"][:1]
    selected_positive = []
    for row in positive_candidates:
        if abs(float(row["source_time_s"]) - float(seed["source_time_s"])) < 1.0:
            continue
        if any(abs(float(row["source_time_s"]) - float(other["source_time_s"])) < 1.0
               for other in selected_positive):
            continue
        selected_positive.append(row)
        if len(selected_positive) == 2:
            break
    positives = [seed, *selected_positive]

    def compare(left, right, direction):
        return _compare_pair(runner, left, right, direction=direction,
                             output_dir=checks_dir)["result"]

    checks = validate_album_consistency(positives, negatives, compare=compare)
    result = {
        "schema_version": "target_album_v1",
        "target_id": "S0",
        "status": "trusted_seed_model_expanded",
        "selection_basis": "visual_identity_only",
        "trusted_seed": seed,
        "positive": positives,
        "hard_negative": negatives,
        "uncertain": [row for row in normalized
                      if row["candidate_class"] == "uncertain"],
        "consistency_checks": checks,
        "uses_model_confidence": False,
        "uses_margin_threshold": False,
        "proposal_audit": proposal_audit,
        "frozen": True,
    }
    _write_json(output_dir / "target_album.json", result)
    return result


def transport_windows(start_s: float, end_s: float, *, duration_s: float = 6.0,
                      overlap_s: float = 0.5) -> list[tuple[float, float]]:
    if end_s <= start_s or duration_s <= overlap_s:
        raise ValueError("invalid transport window contract")
    windows, cursor = [], float(start_s)
    while cursor < end_s - 1e-6:
        stop = min(float(end_s), cursor + duration_s)
        windows.append((round(cursor, 6), round(stop, 6)))
        if stop >= end_s - 1e-6:
            break
        cursor += duration_s - overlap_s
    return windows


def _normalize_candidates(payload: dict[str, Any], *, arm: str, window_id: str,
                          start_s: float, end_s: float) -> list[dict[str, Any]]:
    candidates = []
    for index, row in enumerate(payload.get("candidates") or []):
        if len(candidates) >= 3:
            break
        relation = str(row.get("relation") or "uncertain")
        if relation not in ALLOWED_BROWSE_RELATIONS:
            raise V7Blocked("browsing", "invalid_browse_relation")
        if relation == "not_observed":
            continue
        interval = row.get("relative_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        absolute = [start_s + float(interval[0]), start_s + float(interval[1])]
        if not (start_s <= absolute[0] < absolute[1] <= end_s + 1e-6):
            raise V7Blocked("browsing", "browse_candidate_out_of_window")
        goals = [str(goal) for goal in row.get("goal_hypotheses") or []
                 if str(goal) in REQUIRED_GOALS]
        candidates.append({
            "id": f"{arm}_{window_id}_{index:02d}", "arm": arm,
            "window_id": window_id, "relation": relation,
            "relative_interval": [float(interval[0]), float(interval[1])],
            "observation_interval": [round(absolute[0], 6), round(absolute[1], 6)],
            "observation": dict(row.get("observation") or {}),
            "goal_hypotheses": goals,
            "roi": row.get("roi"),
        })
    return candidates


def _browse_prompt(reference_task: dict[str, Any]) -> str:
    goals = [row["id"] for row in reference_task.get("evidence_goals") or []]
    return (
        "Browse this source window around the frozen visual subject S0 shown in the album "
        "images. Report at most three candidate regions, not final evidence. "
        f"Evidence goals: {goals}. The relation field must be one of target_direct, "
        "related_interaction, possible_outcome, uncertain, not_observed. not_observed "
        "means this sampling did not see it and never means absent. Times are relative to "
        "this transport clip. Return JSON only: {\"candidates\":[{\"relative_interval\":"
        "[0.0,1.0],\"relation\":\"target_direct\",\"goal_hypotheses\":[\"agency\"],"
        "\"observation\":{\"visible_fact\":\"...\"},\"roi\":[0,0,1,1]}],"
        "\"not_observed_goals\":[]} ."
    )


def run_browse_matrix(cfg: AppConfig, experiment_spec: dict[str, Any],
                      output_dir: Path, *, clients: Mapping[str, Any],
                      source_video: Path, reference_task: dict[str, Any],
                      target_album: dict[str, Any], oracle: dict[str, Any] | None = None) \
        -> dict[str, Any]:
    output_dir = Path(output_dir)
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    scope = experiment_spec["source_scope"]
    perception = experiment_spec.get("transport") or {}
    windows = transport_windows(
        float(scope["start_s"]), float(scope["end_s"]),
        duration_s=float(perception.get("window_s", 6.0)),
        overlap_s=float(perception.get("overlap_s", 0.5)))
    images = [Path(row["full_frame"]) for row in target_album["positive"][:3]]
    if target_album.get("hard_negative"):
        images.append(Path(target_album["hard_negative"][0]["full_frame"]))
    clips = {}
    for index, (start, end) in enumerate(windows):
        window_id = f"w{index:03d}"
        clips[window_id] = cut_clip(
            ffmpeg, source_video, output_dir / "transport" / window_id,
            start_s=start, end_s=end)
    arms = {
        arm: {"arm": arm, "contract": experiment_spec["browse_arms"][arm],
              "window_count": len(windows), "candidates": [],
              "request_audits": [], "failures": []}
        for arm in ("A", "B", "C", "D")
    }
    for arm in ("A", "B", "C", "D"):
        client = clients[arm]
        contract = experiment_spec["browse_arms"][arm]
        if (abs(float(client.endpoint.fps) - float(contract["fps"])) > 1e-6 or
                abs(float(client.endpoint.retention_ratio) -
                    float(contract["retention_ratio"])) > 1e-6 or
                client.endpoint.backend != contract["backend"]):
            raise V7Blocked("policy", "browse_arm_contract_conflict")
    def browse_one(arm: str, window_id: str, start: float, end: float):
        answer = clients[arm].watch(
            clips[window_id], _browse_prompt(reference_task), duration_s=end - start,
            image_paths=images)
        raw_dir = output_dir / "arms" / arm / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{window_id}.txt").write_text(answer.text, encoding="utf-8")
        parsed = _parse_json(answer.text)
        rows = _normalize_candidates(
            parsed, arm=arm, window_id=window_id, start_s=start, end_s=end)
        audit = {**answer.request_audit, "window_id": window_id,
                 "source_interval": [start, end]}
        return rows, audit

    jobs = []
    with ThreadPoolExecutor(max_workers=min(8, len(windows) * 4)) as executor:
        for arm in ("A", "B", "C", "D"):
            for index, (start, end) in enumerate(windows):
                window_id = f"w{index:03d}"
                future = executor.submit(browse_one, arm, window_id, start, end)
                jobs.append((future, arm, window_id))
        for future in as_completed([row[0] for row in jobs]):
            arm, window_id = next((row[1], row[2]) for row in jobs if row[0] is future)
            try:
                rows, audit = future.result()
                arms[arm]["candidates"].extend(rows)
                arms[arm]["request_audits"].append(audit)
            except Exception as exc:  # preserve matrix failures instead of hiding them
                arms[arm]["failures"].append({
                    "window_id": window_id,
                    "error": f"{type(exc).__name__}: {exc}"})
    for arm, arm_row in arms.items():
        arm_row["candidates"].sort(
            key=lambda row: (row["observation_interval"][0], row["id"]))
        arm_row["request_audits"].sort(key=lambda row: row["window_id"])
        arm_row["failures"].sort(key=lambda row: row["window_id"])
        _write_json(output_dir / "arms" / arm / "browse_results.json", arm_row)
    comparison = compare_browse_arms(arms, oracle=oracle)
    _write_json(output_dir / "arm_comparison.json", comparison)
    return {"arms": arms, "comparison": comparison}


def _interval_hit(predicted: Iterable[float], truth: Iterable[float],
                  threshold: float = 0.3) -> bool:
    p0, p1 = map(float, predicted)
    t0, t1 = map(float, truth)
    intersection = max(0.0, min(p1, t1) - max(p0, t0))
    union = max(p1, t1) - min(p0, t0)
    center = (p0 + p1) / 2
    return t0 <= center <= t1 or (union > 0 and intersection / union > threshold)


def compare_browse_arms(arms: Mapping[str, dict[str, Any]], *,
                        oracle: dict[str, Any] | None) -> dict[str, Any]:
    facts = list((oracle or {}).get("facts") or [])
    metrics = {}
    for arm, row in arms.items():
        candidates = row.get("candidates") or []
        hits = [fact for fact in facts if any(
            _interval_hit(candidate["observation_interval"], fact["core_interval"])
            for candidate in candidates)]
        goal_recall = {}
        legacy_goal = {"adversity": "struggle", "agency": "reversal",
                       "outcome": "payoff"}
        for goal in REQUIRED_GOALS:
            goal_facts = [fact for fact in facts
                          if (goal in (fact.get("expected_goals") or []) or
                              legacy_goal[goal] in (fact.get("expected_slots") or []))]
            hit_count = sum(any(
                goal in candidate.get("goal_hypotheses", []) and
                _interval_hit(candidate["observation_interval"], fact["core_interval"])
                for candidate in candidates) for fact in goal_facts)
            goal_recall[goal] = (round(hit_count / len(goal_facts), 6)
                                 if goal_facts else None)
        duplicate_pairs = 0
        for index, left in enumerate(candidates):
            duplicate_pairs += sum(_interval_hit(
                right["observation_interval"], left["observation_interval"], 0.5)
                for right in candidates[index + 1:])
        usage = [audit.get("usage") or {} for audit in row.get("request_audits") or []]
        visual_tokens = sum(int(value.get("prompt_tokens_details", {}).get(
            "video_tokens", value.get("visual_tokens", 0)) or 0) for value in usage)
        s0_facts = [fact for fact in facts if fact.get("subject_track_id")]
        s0_hits = sum(any(
            _interval_hit(candidate["observation_interval"], fact["core_interval"])
            for candidate in candidates) for fact in s0_facts)
        metrics[arm] = {
            "candidate_temporal_hits": len(hits),
            "candidate_temporal_recall": round(len(hits) / len(facts), 6) if facts else None,
            "s0_appearance_recall": (round(s0_hits / len(s0_facts), 6)
                                     if s0_facts else None),
            "goal_recall": goal_recall,
            "candidate_count": len(candidates),
            "duplicate_candidate_rate": round(
                duplicate_pairs / max(1, len(candidates)), 6),
            "visual_tokens": visual_tokens,
            "latency_s": round(sum(float(audit.get("latency_s", 0))
                                   for audit in row.get("request_audits") or []), 6),
            "failure_rate": round(len(row.get("failures") or []) /
                                  max(1, int(row.get("window_count") or 0)), 6),
        }
    return {"schema_version": "v7_browse_comparison_v1", "metrics": metrics,
            "gt_used_in_prompts": False, "diagnostic_attribution": diagnose_browse(metrics)}


def diagnose_browse(metrics: Mapping[str, dict[str, Any]]) -> list[str]:
    def score(arm: str) -> float:
        return float((metrics.get(arm) or {}).get("candidate_temporal_recall") or 0)
    findings = []
    if score("B") > score("A"):
        findings.append("temporal_sampling_insufficient_at_2fps")
    if score("C") > score("B"):
        findings.append("r010_compression_too_aggressive")
    if score("D") > score("C"):
        findings.append("custom_port_or_compression_semantic_loss")
    if all(abs(score(arm) - score("A")) < 1e-9 for arm in ("B", "C", "D")):
        findings.append("all_arms_equivalent_choose_A_on_cost")
    return findings


def unique_candidates(rows: Iterable[dict[str, Any]], seen: list[list[float]]) \
        -> list[dict[str, Any]]:
    result = []
    for row in rows:
        interval = row["observation_interval"]
        if any(_interval_hit(interval, prior, 0.5) for prior in seen):
            continue
        seen.append(list(interval))
        result.append(row)
    return result


def run_browse_escalation(arms: Mapping[str, dict[str, Any]], *,
                          verify: Callable[[dict[str, Any]], dict[str, Any] | None],
                          required_goals: Iterable[str] = REQUIRED_GOALS) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing = set(required_goals)
    verified, logs, seen = [], [], []
    for arm_index, arm in enumerate(("A", "B", "C", "D")):
        for candidate in unique_candidates(arms[arm].get("candidates") or [], seen):
            if not missing.intersection(candidate.get("goal_hypotheses") or []):
                continue
            evidence = verify(candidate)
            if evidence:
                verified.append(evidence)
                missing.difference_update(evidence.get("eligible_goals") or [])
            if not missing:
                return verified, logs
        if missing and arm_index < 3:
            logs.append({
                "missing_goal": sorted(missing), "previous_arm": arm,
                "next_arm": ("A", "B", "C", "D")[arm_index + 1],
                "reason": "all_unique_candidates_exhausted_without_verified_goal",
            })
    if missing:
        raise V7Blocked("browsing", "required_goal_not_observed_after_vanilla",
                        ",".join(sorted(missing)))
    return verified, logs


def extract_native_frames(ffmpeg_bin: str, source_video: Path,
                          source_interval: Iterable[float], output_dir: Path,
                          *, max_frames: int = 60) -> dict[str, Any]:
    """Decode every source frame in a narrow range and retain true showinfo PTS."""
    start, end = map(float, source_interval)
    if not 0 < end - start <= 1.0 + 1e-6:
        raise ValueError("native frame interval must be in (0, 1.0] seconds")
    output_dir = Path(output_dir)
    raw_dir, labeled_dir = output_dir / "raw", output_dir / "labeled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    pattern = raw_dir / "frame_%04d.png"
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "info", "-ss", f"{start:.6f}",
        "-copyts", "-i", str(source_video), "-an",
        "-vf", f"trim=duration={end-start:.6f},showinfo",
        "-fps_mode", "passthrough", str(pattern),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if process.returncode != 0:
        raise V7Blocked("boundary", "native_frame_decode_failed",
                        (process.stderr or "")[-500:])
    times = [float(value) for value in _SHOWINFO.findall(process.stderr or "")]
    files = sorted(raw_dir.glob("frame_*.png"))
    if len(files) != len(times) or not files or len(files) > max_frames:
        raise V7Blocked("boundary", "native_frame_manifest_mismatch",
                        f"files={len(files)} pts={len(times)} max={max_frames}")
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise V7Blocked("boundary", "pillow_required_for_frame_labels") from exc
    frames = []
    for index, (path, pts) in enumerate(zip(files, times)):
        frame_id = f"f{index:03d}"
        labeled = labeled_dir / f"{frame_id}.png"
        with Image.open(path) as image:
            canvas = image.convert("RGB")
            canvas.thumbnail((768, 768))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 150, 42), fill="black")
            draw.text((10, 8), frame_id, fill="white")
            canvas.save(labeled)
        frames.append({"frame_id": frame_id, "pts_s": round(pts, 6),
                       "path": str(path), "labeled_path": str(labeled),
                       "sha256": sha256_file(path)})
    manifest = {"source_interval": [start, end], "source_fps": round(
        len(frames) / (end - start), 6), "frames": frames, "selected": None,
        "command": command}
    _write_json(output_dir / "native_frames_manifest.json", manifest)
    return manifest


def apply_native_selection(manifest: dict[str, Any], selection: dict[str, Any]) \
        -> dict[str, Any]:
    by_id = {row["frame_id"]: row for row in manifest["frames"]}
    ids = [str(selection.get(key) or "")
           for key in ("start_frame", "peak_frame", "end_frame")]
    if any(frame_id not in by_id for frame_id in ids):
        raise V7Blocked("boundary", "native_frame_selection_unknown_id")
    indexes = [int(frame_id[1:]) for frame_id in ids]
    if not indexes[0] <= indexes[1] <= indexes[2]:
        raise V7Blocked("boundary", "native_frame_selection_not_ordered")
    result = dict(manifest)
    result["selected"] = {
        "start_frame": ids[0], "peak_frame": ids[1], "end_frame": ids[2],
        "core_interval": [by_id[ids[0]]["pts_s"], by_id[ids[2]]["pts_s"]],
    }
    return result


def _identity_candidate(runner, cfg: AppConfig, source_video: Path,
                        candidate: dict[str, Any], album: dict[str, Any],
                        output_dir: Path) -> dict[str, Any]:
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    start, end = candidate["observation_interval"]
    full = export_frame(ffmpeg, source_video, (start + end) / 2,
                        output_dir / "candidate_full.jpg")
    images = [full]
    if candidate.get("roi"):
        images.append(export_roi(ffmpeg, full, candidate["roi"],
                                 output_dir / "candidate_roi.jpg"))
    images.append(Path(album["positive"][0]["full_frame"]))
    if album.get("hard_negative"):
        images.append(Path(album["hard_negative"][0]["full_frame"]))
    prompt = (
        "Identity-only check. The first full frame (and optional crop immediately after it) "
        "is the candidate. The following images are a trusted S0 positive and a hard "
        "negative. A crop is only an aid paired with its full frame. Do not infer actions "
        "or editorial roles. Return JSON only: {\"result\":\"same|different|uncertain\"}."
    )
    answer = runner.inspect_media(images, prompt, max_new_tokens=64)
    result = {"result": _identity_result(answer.text),
              "images": [str(path) for path in images]}
    _write_json(output_dir / "identity.json", result)
    return result


def _verified_action(runner, clip: Path, output_dir: Path) -> dict[str, Any]:
    prompt = (
        "Describe only visible action in this continuous original clip. Do not assign an "
        "editing goal and do not infer offscreen consequences. Return JSON only with "
        "actor, patient, visible_action, state_before, state_after, can_prove (array), "
        "cannot_prove (array), source_form, and passed (boolean)."
    )
    answer = runner.watch(clip, prompt, fps=12.0, duration_s=common.video_duration_s(
        "ffprobe", clip), max_new_tokens=512)
    value = _parse_json(answer.text)
    value["sampling"] = getattr(answer, "sampling", None)
    if value.get("passed") is not True:
        raise V7Blocked("action", "action_check_failed")
    _write_json(output_dir / "action.json", value)
    return value


def _verified_relation(runner, clip: Path, output_dir: Path) -> dict[str, Any]:
    prompt = (
        "Independently inspect the continuous context for a causal relation between S0's "
        "visible action and the later opponent/environment change, including across a cut. "
        "Do not upgrade correlation to victory. Return JSON only with passed (boolean), "
        "relation (related_outcome|uncertain), can_prove (array), cannot_prove (array)."
    )
    answer = runner.watch(clip, prompt, fps=12.0, duration_s=common.video_duration_s(
        "ffprobe", clip), max_new_tokens=384)
    value = _parse_json(answer.text)
    if value.get("passed") is not True or value.get("relation") != "related_outcome":
        raise V7Blocked("relation", "outcome_relation_not_verified")
    _write_json(output_dir / "relation.json", value)
    return value


def _verified_boundary(runner, cfg: AppConfig, source_video: Path,
                       observation_interval: list[float], output_dir: Path) \
        -> dict[str, Any]:
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    start, end = observation_interval
    clip = cut_clip(ffmpeg, source_video, output_dir / "coarse_clip",
                    start_s=start, end_s=end)
    prompt = (
        "Coarsely localize the smallest 0.4-1.0 second interval containing the complete "
        "visible action. Times must be relative to this clip. This is not the final frame "
        "boundary. Return JSON only: {\"relative_interval\":[0.0,0.8]}."
    )
    answer = runner.watch(clip, prompt, fps=12.0, duration_s=end - start,
                          max_new_tokens=128)
    row = _parse_json(answer.text)
    relative = row.get("relative_interval") or []
    if len(relative) != 2:
        raise V7Blocked("boundary", "coarse_boundary_missing")
    coarse = [start + float(relative[0]), start + float(relative[1])]
    if not start <= coarse[0] < coarse[1] <= end + 1e-6:
        raise V7Blocked("boundary", "coarse_boundary_out_of_range")
    if coarse[1] - coarse[0] > 0.9:
        center = sum(coarse) / 2
        coarse = [center - 0.45, center + 0.45]
    manifest = extract_native_frames(ffmpeg, source_video, coarse,
                                     output_dir / "native", max_frames=60)
    labeled = [Path(frame["labeled_path"]) for frame in manifest["frames"]]
    frame_ids = [frame["frame_id"] for frame in manifest["frames"]]
    final_prompt = (
        "Select exact native source frames. Images are ordered exactly as these IDs: "
        f"{frame_ids}. Return JSON only with start_frame, peak_frame, end_frame. "
        "Use only listed IDs and preserve their order; do not calculate seconds."
    )
    final = runner.inspect_media(labeled, final_prompt, max_new_tokens=128)
    selected = apply_native_selection(manifest, _parse_json(final.text))
    _write_json(output_dir / "native" / "native_frames_manifest.json", selected)
    return selected


def _verify_target_evidence_batched(cfg: AppConfig, output_dir: Path, *, runner,
                                    source_video: Path,
                                    browse_arms: Mapping[str, dict[str, Any]],
                                    target_album: dict[str, Any]) -> dict[str, Any]:
    """Run each Omni stage as a batch so all configured model replicas are used."""
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    missing, seen = set(REQUIRED_GOALS), []
    verified, escalations, failures = [], [], []
    identity_prompt = (
        "Identity-only check. The first full frame and optional crop are the candidate; "
        "the final images are a trusted S0 positive and hard negative. A crop is only an "
        "aid paired with its full frame. Do not infer actions or editorial roles. Return "
        "JSON only: {\"result\":\"same|different|uncertain\"}."
    )
    action_prompt = (
        "Describe only visible action in this continuous original clip. Do not assign an "
        "editing goal or infer offscreen consequences. Return JSON only with actor, patient, "
        "visible_action, state_before, state_after, can_prove, cannot_prove, source_form, "
        "and passed (boolean)."
    )
    relation_prompt = (
        "Independently inspect the continuous context for a causal relation between S0's "
        "visible action and the later opponent/environment change, including across a cut. "
        "Do not upgrade correlation to victory. Return JSON only with passed (boolean), "
        "relation (related_outcome|uncertain), can_prove, cannot_prove."
    )
    coarse_prompt = (
        "Coarsely localize the smallest 0.4-1.0 second interval containing the complete "
        "visible action. Times are relative to this clip and are not the final boundary. "
        "Return JSON only: {\"relative_interval\":[0.0,0.8]}."
    )

    def record(candidate, stage, code, detail=""):
        failures.append({"candidate_id": candidate["id"], "arm": candidate["arm"],
                         "failure_stage": stage, "reason_code": code,
                         "detail": detail})

    for arm_index, arm in enumerate(("A", "B", "C", "D")):
        available = unique_candidates(browse_arms[arm].get("candidates") or [], seen)
        consumed: set[str] = set()
        while missing:
            wave = []
            for goal in sorted(missing):
                candidate = next((row for row in available
                                  if row["id"] not in consumed and
                                  goal in row.get("goal_hypotheses", [])), None)
                if candidate is not None and candidate not in wave:
                    wave.append(candidate)
                    consumed.add(candidate["id"])
            if not wave:
                break

            identity_requests = []
            prepared = []
            for candidate in wave:
                candidate_dir = output_dir / "candidates" / candidate["id"]
                start, end = candidate["observation_interval"]
                full = export_frame(ffmpeg, source_video, (start + end) / 2,
                                    candidate_dir / "identity" / "candidate_full.jpg")
                images = [full]
                if candidate.get("roi"):
                    images.append(export_roi(
                        ffmpeg, full, candidate["roi"],
                        candidate_dir / "identity" / "candidate_roi.jpg"))
                images.append(Path(target_album["positive"][0]["full_frame"]))
                images.append(Path(target_album["hard_negative"][0]["full_frame"]))
                identity_requests.append({"image_paths": images, "prompt": identity_prompt,
                                          "kwargs": {"max_new_tokens": 64}})
                prepared.append((candidate, candidate_dir, images))
            identity_answers = runner.inspect_media_many(identity_requests)
            identity_passed = []
            for (candidate, candidate_dir, images), answer in zip(prepared, identity_answers):
                try:
                    result = _identity_result(answer.text)
                except V7Blocked as exc:
                    record(candidate, exc.failure_stage, exc.reason_code, str(exc))
                    continue
                identity = {"result": result, "images": [str(path) for path in images]}
                _write_json(candidate_dir / "identity" / "identity.json", identity)
                if result != "same":
                    record(candidate, "identity", f"candidate_identity_{result}")
                    continue
                identity_passed.append((candidate, candidate_dir, identity))
            if not identity_passed:
                continue

            action_requests, action_prepared = [], []
            for candidate, candidate_dir, identity in identity_passed:
                start, end = candidate["observation_interval"]
                clip = cut_clip(ffmpeg, source_video, candidate_dir / "source_clip",
                                start_s=start, end_s=end)
                action_requests.append({"video_path": clip, "prompt": action_prompt,
                                        "kwargs": {"fps": 12.0, "duration_s": end - start,
                                                   "max_new_tokens": 512}})
                action_prepared.append((candidate, candidate_dir, identity, clip))
            action_answers = runner.watch_many(action_requests)
            action_passed = []
            for prepared_row, answer in zip(action_prepared, action_answers):
                candidate, candidate_dir, identity, clip = prepared_row
                try:
                    action = _parse_json(answer.text)
                    action["sampling"] = getattr(answer, "sampling", None)
                    if action.get("passed") is not True:
                        raise V7Blocked("action", "action_check_failed")
                    if action.get("source_form") not in {
                            "visual_instant", "dynamic_action", "reaction",
                            "dialogue_span", "establishing_visual"}:
                        raise V7Blocked("action", "invalid_action_source_form")
                except V7Blocked as exc:
                    record(candidate, exc.failure_stage, exc.reason_code, str(exc))
                    continue
                _write_json(candidate_dir / "action" / "action.json", action)
                action_passed.append((candidate, candidate_dir, identity, clip, action))

            relation_rows = [row for row in action_passed
                             if row[0]["relation"] == "possible_outcome"]
            relation_by_id: dict[str, dict[str, Any]] = {}
            if relation_rows:
                relation_answers = runner.watch_many([
                    {"video_path": row[3], "prompt": relation_prompt,
                     "kwargs": {"fps": 12.0,
                                "duration_s": row[0]["observation_interval"][1] -
                                              row[0]["observation_interval"][0],
                                "max_new_tokens": 384}}
                    for row in relation_rows])
                for row, answer in zip(relation_rows, relation_answers):
                    candidate, candidate_dir = row[0], row[1]
                    try:
                        relation = _parse_json(answer.text)
                        if (relation.get("passed") is not True or
                                relation.get("relation") != "related_outcome"):
                            raise V7Blocked("relation", "outcome_relation_not_verified")
                    except V7Blocked as exc:
                        record(candidate, exc.failure_stage, exc.reason_code, str(exc))
                        continue
                    relation_by_id[candidate["id"]] = relation
                    _write_json(candidate_dir / "relation" / "relation.json", relation)

            boundary_rows = [row for row in action_passed
                             if (row[0]["relation"] != "possible_outcome" or
                                 row[0]["id"] in relation_by_id)]
            if not boundary_rows:
                continue
            coarse_answers = runner.watch_many([
                {"video_path": row[3], "prompt": coarse_prompt,
                 "kwargs": {"fps": 12.0,
                            "duration_s": row[0]["observation_interval"][1] -
                                          row[0]["observation_interval"][0],
                            "max_new_tokens": 128}}
                for row in boundary_rows])
            native_rows = []
            for row, answer in zip(boundary_rows, coarse_answers):
                candidate, candidate_dir = row[0], row[1]
                start, end = candidate["observation_interval"]
                try:
                    relative = _parse_json(answer.text).get("relative_interval") or []
                    if len(relative) != 2:
                        raise V7Blocked("boundary", "coarse_boundary_missing")
                    coarse = [start + float(relative[0]), start + float(relative[1])]
                    if not start <= coarse[0] < coarse[1] <= end + 1e-6:
                        raise V7Blocked("boundary", "coarse_boundary_out_of_range")
                    if coarse[1] - coarse[0] > 0.9:
                        center = sum(coarse) / 2
                        coarse = [center - .45, center + .45]
                    manifest = extract_native_frames(
                        ffmpeg, source_video, coarse, candidate_dir / "boundary" / "native")
                except (V7Blocked, ValueError) as exc:
                    record(candidate, getattr(exc, "failure_stage", "boundary"),
                           getattr(exc, "reason_code", "native_frame_invalid"), str(exc))
                    continue
                native_rows.append((*row, manifest))
            if not native_rows:
                continue
            final_answers = runner.inspect_media_many([
                {"image_paths": [Path(frame["labeled_path"])
                                 for frame in row[5]["frames"]],
                 "prompt": ("Select exact native source frames. Images are ordered as: " +
                            str([frame["frame_id"] for frame in row[5]["frames"]]) +
                            ". Return JSON only with start_frame, peak_frame, end_frame. "
                            "Use listed IDs; do not calculate seconds."),
                 "kwargs": {"max_new_tokens": 128}}
                for row in native_rows])
            for row, answer in zip(native_rows, final_answers):
                candidate, candidate_dir, identity, _clip, action, manifest = row
                try:
                    boundary = apply_native_selection(manifest, _parse_json(answer.text))
                    _write_json(candidate_dir / "boundary" / "native" /
                                "native_frames_manifest.json", boundary)
                    core = boundary["selected"]["core_interval"]
                    start, end = candidate["observation_interval"]
                    renderable = [max(start, core[0] - .15), min(end, core[1] + .15)]
                    relation = relation_by_id.get(candidate["id"])
                    target_relation = "related_outcome" if relation else "self"
                    evidence = EvidenceUnitV3(
                        id=f"ev_{candidate['id']}", observation_interval=(start, end),
                        core_interval=(float(core[0]), float(core[1])),
                        renderable_interval=(float(renderable[0]), float(renderable[1])),
                        observation={"visible_action": action.get("visible_action"),
                                     "actor": action.get("actor"),
                                     "patient": action.get("patient"),
                                     "state_before": action.get("state_before"),
                                     "state_after": action.get("state_after")},
                        source_form=str(action["source_form"]),
                        target_relation=target_relation,
                        identity_verification=identity, action_verification=action,
                        relation_verification=relation,
                        claim_bounds={"can_prove": action.get("can_prove") or [],
                                      "cannot_prove": action.get("cannot_prove") or []},
                        native_frame_provenance=boundary,
                        source_video=str(source_video)).to_dict()
                except (V7Blocked, ValueError) as exc:
                    record(candidate, getattr(exc, "failure_stage", "boundary"),
                           getattr(exc, "reason_code", "invalid_verified_evidence"), str(exc))
                    continue
                evidence["eligible_goals"] = list(candidate.get("goal_hypotheses") or [])
                evidence["browse_candidate_id"] = candidate["id"]
                _write_json(candidate_dir / "evidence.json", evidence)
                verified.append(evidence)
                missing.difference_update(evidence["eligible_goals"])
        if not missing:
            break
        if arm_index < 3:
            escalations.append({
                "missing_goal": sorted(missing), "previous_arm": arm,
                "next_arm": ("A", "B", "C", "D")[arm_index + 1],
                "reason": "all_unique_candidates_exhausted_without_verified_goal"})
    if missing:
        _write_json(output_dir / "browse_escalations.json", {
            "events": escalations, "candidate_failures": failures})
        raise V7Blocked("browsing", "required_goal_not_observed_after_vanilla",
                        ",".join(sorted(missing)))
    _write_json(output_dir / "browse_escalations.json", {
        "events": escalations, "candidate_failures": failures})
    return {"schema_version": "evidence_bank_v3", "target_id": "S0",
            "evidence": verified, "required_goals": list(REQUIRED_GOALS),
            "omni_execution": "stage_batched"}


def verify_target_evidence(cfg: AppConfig, experiment_spec: dict[str, Any],
                           output_dir: Path, *, runner, source_video: Path,
                           browse_arms: Mapping[str, dict[str, Any]],
                           target_album: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    attempt_failures: list[dict[str, Any]] = []

    if hasattr(runner, "inspect_media_many") and hasattr(runner, "watch_many"):
        bank = _verify_target_evidence_batched(
            cfg, output_dir, runner=runner, source_video=source_video,
            browse_arms=browse_arms, target_album=target_album)
        _write_json(output_dir.parent / "evidence_bank.json", bank)
        return bank

    def verify_one(candidate: dict[str, Any]) -> dict[str, Any] | None:
        candidate_dir = output_dir / "candidates" / candidate["id"]
        identity = _identity_candidate(
            runner, cfg, source_video, candidate, target_album,
            candidate_dir / "identity")
        if identity["result"] != "same":
            raise V7Blocked("identity", f"candidate_identity_{identity['result']}")
        start, end = candidate["observation_interval"]
        clip = cut_clip(ffmpeg, source_video, candidate_dir / "source_clip",
                        start_s=start, end_s=end)
        action = _verified_action(runner, clip, candidate_dir / "action")
        relation = None
        target_relation = "self"
        if candidate["relation"] == "possible_outcome":
            relation = _verified_relation(runner, clip, candidate_dir / "relation")
            target_relation = "related_outcome"
        boundary = _verified_boundary(
            runner, cfg, source_video, candidate["observation_interval"],
            candidate_dir / "boundary")
        core = boundary["selected"]["core_interval"]
        renderable = [max(start, core[0] - 0.15), min(end, core[1] + 0.15)]
        source_form = str(action.get("source_form") or "dynamic_action")
        evidence = EvidenceUnitV3(
            id=f"ev_{candidate['id']}", observation_interval=(start, end),
            core_interval=(float(core[0]), float(core[1])),
            renderable_interval=(float(renderable[0]), float(renderable[1])),
            observation={"visible_action": action.get("visible_action"),
                         "actor": action.get("actor"), "patient": action.get("patient"),
                         "state_before": action.get("state_before"),
                         "state_after": action.get("state_after")},
            source_form=source_form, target_relation=target_relation,
            identity_verification=identity, action_verification=action,
            relation_verification=relation,
            claim_bounds={"can_prove": action.get("can_prove") or [],
                          "cannot_prove": action.get("cannot_prove") or []},
            native_frame_provenance=boundary, source_video=str(source_video)).to_dict()
        evidence["eligible_goals"] = list(candidate.get("goal_hypotheses") or [])
        evidence["browse_candidate_id"] = candidate["id"]
        _write_json(candidate_dir / "evidence.json", evidence)
        return evidence

    def verify(candidate: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return verify_one(candidate)
        except (V7Blocked, ValueError) as exc:
            attempt_failures.append({
                "candidate_id": candidate["id"], "arm": candidate["arm"],
                "failure_stage": getattr(exc, "failure_stage", "verification"),
                "reason_code": getattr(exc, "reason_code", "invalid_verified_evidence"),
                "detail": str(exc),
            })
            return None

    verified, escalations = run_browse_escalation(browse_arms, verify=verify)
    _write_json(output_dir / "browse_escalations.json", {
        "events": escalations, "candidate_failures": attempt_failures})
    bank = {"schema_version": "evidence_bank_v3", "target_id": "S0",
            "evidence": verified, "required_goals": list(REQUIRED_GOALS)}
    _write_json(output_dir.parent / "evidence_bank.json", bank)
    return bank


def oracle_evidence_bank(oracle: dict[str, Any], source_video: Path) -> dict[str, Any]:
    evidence = []
    role_map = {"struggle": "adversity", "reversal": "agency", "payoff": "outcome"}
    for fact in oracle.get("facts") or []:
        goals = [role_map.get(value, value) for value in
                 (fact.get("expected_goals") or fact.get("expected_slots") or [])]
        goals = [goal for goal in goals if goal in REQUIRED_GOALS]
        if not goals:
            continue
        core = list(map(float, fact["core_interval"]))
        container = list(map(float, fact.get("container_interval") or core))
        target_relation = "related_outcome" if "outcome" in goals else "self"
        evidence.append({
            "id": fact["id"], "observation_interval": container,
            "core_interval": core, "renderable_interval": container,
            "observation": fact.get("observation") or {},
            "source_form": fact.get("source_form") or "dynamic_action",
            "target_relation": target_relation,
            "identity_verification": {"result": "same", "source": "human_oracle"},
            "action_verification": {"passed": True, "source": "human_oracle"},
            "relation_verification": ({"passed": True, "source": "human_oracle"}
                                      if target_relation == "related_outcome" else None),
            "claim_bounds": {"can_prove": [fact.get("observation")],
                             "cannot_prove": []},
            "native_frame_provenance": {"source": "human_oracle"},
            "source_video": str(source_video), "eligible_goals": goals,
        })
    return {"schema_version": "evidence_bank_v3_oracle", "target_id": "S0",
            "evidence": evidence, "bypassed": ["browsing", "automatic_verification"]}


def build_reference_driven_edit_plan(reference_task: dict[str, Any],
                                     evidence_bank: dict[str, Any]) -> dict[str, Any]:
    rows = list(evidence_bank.get("evidence") or [])
    missing = [goal for goal in REQUIRED_GOALS if not any(
        goal in row.get("eligible_goals", []) for row in rows)]
    if missing:
        return {"schema_version": "reference_driven_edit_plan_v1", "passed": False,
                "failure_class": "content", "failure_stage": "planning",
                "reason_code": "required_verified_goal_missing", "missing_goals": missing,
                "segments": [], "duration_s": 0.0}
    used, segments = set(), []
    sections = reference_task.get("edit_sections") or [
        {"id": goal, "goal_ids": [goal]} for goal in REQUIRED_GOALS]
    p90 = float(reference_task["measured_style"]["shot_duration_p90_s"])
    for section in sections:
        for goal in section.get("goal_ids") or []:
            candidates = [row for row in rows if goal in row.get("eligible_goals", [])]
            if not candidates:
                continue
            candidate = next((row for row in candidates if row["id"] not in used),
                             candidates[0])
            if candidate["id"] in used:
                continue
            used.add(candidate["id"])
            interval = list(map(float, candidate["renderable_interval"]))
            duration = interval[1] - interval[0]
            segment = {
                "id": f"segment_{len(segments):02d}", "section_id": section["id"],
                "evidence_id": candidate["id"], "assigned_goal": goal,
                "render_mode": "micro_clip", "render_once": True,
                "render_interval": interval, "core_interval": candidate["core_interval"],
                "source_video": candidate.get("source_video"),
                "duration_s": round(duration, 6),
            }
            if duration > p90 + 1e-6:
                segment["semantic_completeness_reason"] = "verified_renderable_interval"
            segments.append(segment)
    assigned = {row["assigned_goal"] for row in segments}
    unmapped = [goal for goal in REQUIRED_GOALS if goal not in assigned]
    if unmapped:
        return {"schema_version": "reference_driven_edit_plan_v1", "passed": False,
                "failure_class": "verification", "failure_stage": "planning",
                "reason_code": "reference_goal_not_mapped_to_section",
                "missing_goals": unmapped, "segments": segments,
                "duration_s": round(sum(float(row["duration_s"])
                                        for row in segments), 6)}
    duration = sum(float(row["duration_s"]) for row in segments)
    reference_duration = float(reference_task["measured_style"]["reference_duration_s"])
    plan = {
        "schema_version": "reference_driven_edit_plan_v1", "passed": True,
        "target_id": "S0", "segments": segments, "duration_s": round(duration, 6),
        "style": {"preferred_duration_s": reference_duration,
                  "soft_range_s": [round(reference_duration * 0.7, 6),
                                   round(reference_duration * 1.2, 6)],
                  "segment_preferred_max_s": p90},
        "requires_human_style_review": duration > reference_duration * 1.2,
        "short_complete_content_allowed": duration < reference_duration * 0.7,
        "filler_added": False,
    }
    return plan


def _message_similarity(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        compact = "".join(character.lower() for character in value if character.isalnum())
        return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}
    a, b = grams(left), grams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def finalize_target_microcut(cfg: AppConfig, plan: dict[str, Any], output_dir: Path, *,
                             source_video: Path, bgm_path: Path, runner,
                             human_acceptance: dict[str, Any] | None = None,
                             force: bool = False) -> dict[str, Any]:
    """Render one master, blind-review both variants, and gate formal delivery."""
    from src.agentic_video.evidence_pipeline import blind_review_variants

    output_dir = Path(output_dir)
    render = render_micro_montage(
        cfg, plan, output_dir / "render", source_video=source_video,
        bgm_path=bgm_path, force=force)
    variants = {key: Path(value) for key, value in render["variants"].items()}
    reviews = blind_review_variants(variants, runner=runner)
    for name, review in reviews.items():
        required = (
            review.get("parsed") is True and review.get("hook_clear") is True and
            review.get("montage_coherent") is True and
            review.get("functionless_span_present") is False and
            bool(str(review.get("core_message") or "").strip()))
        if review.get("audible_dialogue_present") is True:
            required = required and review.get("speech_clear") is True
        if name == "bgm_mix":
            required = required and review.get("music_present") is True
        review["passed"] = bool(required)
    similarity = _message_similarity(
        str(reviews.get("source_only", {}).get("core_message") or ""),
        str(reviews.get("bgm_mix", {}).get("core_message") or ""))
    reviews["variant_message_similarity"] = round(similarity, 6)
    _write_json(output_dir / "blind_review.json", reviews)
    automatic = (
        set(variants) == {"source_only", "bgm_mix"} and
        all(bool(reviews[name].get("passed")) for name in ("source_only", "bgm_mix")) and
        similarity >= 0.5)
    human = bool((human_acceptance or {}).get("passed"))
    acceptance = {
        "schema_version": "v7_acceptance_v1", "automated_passed": automatic,
        "human_passed": human if human_acceptance is not None else None,
        "passed": automatic and human, "delivery": "passed" if automatic and human else "blocked",
        "failure_class": None if automatic else "verification",
        "failure_stage": None if automatic else "blind",
        "reason_code": None if automatic else "blind_review_failed",
    }
    if automatic and human:
        source = variants["bgm_mix"]
        shutil.copy2(source, output_dir / "rendered.mp4")
    else:
        (output_dir / "rendered.mp4").unlink(missing_ok=True)
    _write_json(output_dir / "acceptance.json", acceptance)
    return {"render": render, "blind_review": reviews, "acceptance": acceptance}


def accept_v7_output(output_dir: Path, human_acceptance: Path | dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    human = (_read_json(human_acceptance) if isinstance(human_acceptance, Path)
             else dict(human_acceptance))
    acceptance = _read_json(output_dir / "acceptance.json")
    if acceptance.get("automated_passed") is not True:
        raise V7Blocked("blind", "automated_acceptance_not_passed")
    if human.get("passed") is not True:
        raise V7Blocked("human", "human_acceptance_not_passed")
    source = output_dir / "render" / "variants" / "bgm_mix" / "rendered.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_dir / "rendered.mp4"
    shutil.copy2(source, destination)
    acceptance.update({"human_passed": True, "passed": True, "delivery": "passed",
                       "failure_class": None, "failure_stage": None,
                       "reason_code": None, "human_acceptance": human})
    _write_json(output_dir / "acceptance.json", acceptance)
    return destination
