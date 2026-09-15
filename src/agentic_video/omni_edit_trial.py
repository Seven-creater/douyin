"""Controlled Omni editorial trial with human-confirmed identity anchors.

This is deliberately a debug workflow rather than an automatic identity claim.
Human-confirmed frames define S0 and a hard negative; Omni observes the source
clip, selects evidence-backed intervals, and existing deterministic code renders
one content master plus two audio variants.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from src.agentic_video.evidence_pipeline import blind_review_variants
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.renderer import render_micro_montage
from src.agentic_video.target_v7 import (
    V7Blocked,
    _message_similarity,
    _parse_json,
    export_frame,
    review_planned_segments,
)
from src.config import AppConfig, repo_root
from src.perception.omni_runner import cut_clip


TRIAL_SCHEMA_VERSION = "omni_editorial_trial_v1"


class OmniEditTrialBlocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = ""):
        super().__init__(detail or reason_code)
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _json_hash(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def _answer_audit(answer: Any) -> dict[str, Any]:
    return {
        "raw_response": str(getattr(answer, "text", "")),
        "input_tokens": int(getattr(answer, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(answer, "output_tokens", 0) or 0),
        "elapsed_s": float(getattr(answer, "elapsed_s", 0.0) or 0.0),
        "input_build_s": float(getattr(answer, "input_build_s", 0.0) or 0.0),
        "gpu_pair": getattr(answer, "gpu_pair", None),
        "sampling": getattr(answer, "sampling", None),
    }


def read_omni_edit_trial_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    if not isinstance(spec, dict) or spec.get("spec_version") != TRIAL_SCHEMA_VERSION:
        raise ValueError(f"expected {TRIAL_SCHEMA_VERSION}: {path}")
    context = list(map(float, spec["context_interval"]))
    scope = list(map(float, spec["editing_scope"]))
    if len(context) != 2 or context[1] <= context[0]:
        raise ValueError("context_interval must be increasing")
    if len(scope) != 2 or scope[1] <= scope[0] or not (
            context[0] <= scope[0] < scope[1] <= context[1]):
        raise ValueError("editing_scope must be contained in context_interval")
    positives = spec.get("identity_anchors", {}).get("positive_source_times_s") or []
    negatives = spec.get("identity_anchors", {}).get("hard_negative_source_times_s") or []
    if len(positives) < 2 or not negatives:
        raise ValueError("at least two S0 positives and one hard negative are required")
    return spec, hashlib.sha256(raw).hexdigest()


def perception_prompt(context_duration_s: float) -> str:
    return f"""You are observing one continuous source clip for a controlled edit trial.

Identity input is fixed by human review, not inferred from story knowledge:
- Image 1 and Image 2 show S0, the SAME small white child.
- Image 3 shows N0, a DIFFERENT large dark beast. N0 is NOT S0.
- The video is the continuous source clip. Its time starts at 0.000 seconds and ends at
  {context_duration_s:.3f} seconds.

Do not use character names, plot knowledge, adversity/agency/outcome labels, or an expected
answer. Describe only visible facts. Distinguish initiator from affected subject. A later
state does not prove who caused it unless the continuous image sequence supports that
relation. Return exactly one JSON object and no trailing text:
{{
  "status": "observed|unreliable",
  "subjects": [
    {{"subject_id":"S0|N0|U1", "appearance":"specific visible description",
      "anchor_match":"S0|N0|unknown"}}
  ],
  "events": [
    {{"event_id":"e1", "interval":[0.0,0.0],
      "initiator":"subject_id or null", "affected":"subject_id or null",
      "visible_action":"specific visible action", "state_before":"specific or unknown",
      "state_after":"specific or unknown", "relation_supported":true,
      "target_involvement":"direct|related|none"}}
  ],
  "claim_bounds": {{"can_prove":[], "cannot_prove":[]}},
  "scene_summary":"one source-grounded sentence"
}}
Use relative seconds from this exact clip. If blur or cuts prevent a reliable direction,
say relation_supported=false instead of guessing. Do not invent an event to satisfy editing.
"""


def validate_perception(payload: Mapping[str, Any], *, duration_s: float) -> dict[str, Any]:
    if payload.get("status") != "observed":
        raise OmniEditTrialBlocked("perception", "source_observation_unreliable")
    subjects = payload.get("subjects")
    events = payload.get("events")
    if not isinstance(subjects, list) or not isinstance(events, list) or not events:
        raise OmniEditTrialBlocked("perception", "missing_subjects_or_events")
    subject_ids = {str(row.get("subject_id") or "") for row in subjects
                   if isinstance(row, Mapping)}
    if "S0" not in subject_ids or "N0" not in subject_ids:
        raise OmniEditTrialBlocked("perception", "identity_anchors_not_found_in_clip")
    for row in subjects:
        if not isinstance(row, Mapping):
            raise OmniEditTrialBlocked("perception", "invalid_subject_row")
        sid = str(row.get("subject_id") or "")
        anchor = str(row.get("anchor_match") or "")
        if sid == "N0" and anchor == "S0":
            raise OmniEditTrialBlocked("identity", "hard_negative_merged_into_target")
    target_events: set[str] = set()
    normalized_events = []
    for row in events:
        if not isinstance(row, Mapping):
            raise OmniEditTrialBlocked("perception", "invalid_event_row")
        event_id = str(row.get("event_id") or "").strip()
        interval = row.get("interval")
        action = str(row.get("visible_action") or "").strip()
        if not event_id or not action or action.lower() in {
                "visible action", "some action", "activity", "unknown"}:
            raise OmniEditTrialBlocked("perception", "non_specific_event")
        if not isinstance(interval, list) or len(interval) != 2:
            raise OmniEditTrialBlocked("perception", "invalid_event_interval")
        start, end = map(float, interval)
        if start < 0 or end <= start or end > duration_s + 0.05:
            raise OmniEditTrialBlocked("scope", "event_interval_outside_context")
        initiator = row.get("initiator")
        affected = row.get("affected")
        for reference in (initiator, affected):
            if reference is not None and str(reference) not in subject_ids:
                raise OmniEditTrialBlocked("perception", "event_references_unknown_subject")
        target_related = (
            str(initiator) == "S0" or str(affected) == "S0" or
            row.get("target_involvement") in {"direct", "related"})
        if target_related:
            target_events.add(event_id)
        normalized_events.append({**dict(row), "interval": [start, end]})
    if not target_events:
        raise OmniEditTrialBlocked("content", "no_observed_target_event")
    result = dict(payload)
    result["events"] = normalized_events
    result["target_event_ids"] = sorted(target_events)
    result["identity_source"] = "human_confirmed_anchors"
    return result


def planner_prompt(perception: Mapping[str, Any], *, allowed_relative: list[float],
                   max_segments: int, max_total_s: float) -> str:
    facts = json.dumps(perception, ensure_ascii=False, separators=(",", ":"))
    return f"""Act as an evidence-bound video editor. The identity anchors and observed facts
below are immutable. Select the shortest coherent source-ordered montage that foregrounds
S0, the small white child, and lets a blind viewer understand a visible action or change.
Do not turn N0, the large dark beast, into S0. Do not add a fact absent from the evidence.
An N0-only result shot is allowed only when an observed event explicitly relates it to S0.

Allowed relative source interval: {allowed_relative}
Maximum segments: {max_segments}; maximum total selected duration: {max_total_s:.3f}s.
No filler, slow motion, duplicate interval, invented transition, or minimum cut count.

Observed evidence:
{facts}

Return exactly one JSON object and no trailing text:
{{"passed":true, "summary":"source-grounded edit intent", "segments":[
  {{"id":"seg_01", "relative_interval":[0.0,0.0],
    "purpose":"what this interval visibly contributes",
    "evidence_event_ids":["e1"]}}
]}}
If the observations cannot support a coherent S0-centered edit, return
{{"passed":false,"reason":"specific source-grounded reason","segments":[]}}.
"""


def normalize_edit_plan(payload: Mapping[str, Any], *, perception: Mapping[str, Any],
                        context_start_s: float, editing_scope: list[float],
                        source_video: Path, max_segments: int,
                        max_total_s: float) -> dict[str, Any]:
    if payload.get("passed") is not True:
        raise OmniEditTrialBlocked("planning", "omni_declined_edit",
                                   str(payload.get("reason") or ""))
    rows = payload.get("segments")
    if not isinstance(rows, list) or not rows or len(rows) > max_segments:
        raise OmniEditTrialBlocked("planning", "invalid_segment_count")
    by_event = {str(row["event_id"]): row for row in perception["events"]}
    target_ids = set(perception["target_event_ids"])
    normalized = []
    total = 0.0
    prior_end = None
    selected_event_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise OmniEditTrialBlocked("planning", "invalid_segment_row")
        interval = row.get("relative_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise OmniEditTrialBlocked("planning", "invalid_segment_interval")
        relative_start, relative_end = map(float, interval)
        start, end = context_start_s + relative_start, context_start_s + relative_end
        if end <= start or start < editing_scope[0] - 0.01 or end > editing_scope[1] + 0.01:
            raise OmniEditTrialBlocked("scope", "selected_interval_outside_editing_scope")
        if prior_end is not None and start < prior_end - 0.01:
            raise OmniEditTrialBlocked("planning", "segments_not_source_ordered")
        refs = [str(value) for value in row.get("evidence_event_ids") or []]
        if not refs or any(value not in by_event for value in refs):
            raise OmniEditTrialBlocked("planning", "segment_has_unknown_evidence_reference")
        selected_event_ids.update(refs)
        duration = end - start
        total += duration
        normalized.append({
            "id": str(row.get("id") or f"seg_{index + 1:02d}"),
            "evidence_id": "+".join(refs),
            "evidence_event_ids": refs,
            "purpose": str(row.get("purpose") or "").strip(),
            "source_video": str(source_video),
            "render_mode": "micro_clip",
            "render_interval": [round(start, 6), round(end, 6)],
            "final_source_interval": [round(start, 6), round(end, 6)],
            "duration_s": round(duration, 6),
        })
        prior_end = end
    if total > max_total_s + 0.01:
        raise OmniEditTrialBlocked("budget", "selected_duration_exceeds_trial_budget")
    if not selected_event_ids.intersection(target_ids):
        raise OmniEditTrialBlocked("content", "edit_does_not_use_target_event")
    return {
        "schema_version": "omni_evidence_edit_plan_v1",
        "passed": True,
        "identity_source": "human_confirmed_anchors",
        "identity_automation_claimed": False,
        "summary": str(payload.get("summary") or "").strip(),
        "segments": normalized,
        "duration_s": round(total, 6),
        "filler_added": False,
        "source_order_preserved": True,
        "editorial_policy_version": TRIAL_SCHEMA_VERSION,
    }


def _blind_pass(review: Mapping[str, Any], *, require_music: bool) -> bool:
    passed = (
        review.get("parsed") is True and review.get("hook_clear") is True and
        review.get("montage_coherent") is True and
        review.get("functionless_span_present") is False and
        review.get("too_short_intervals") == [] and
        review.get("redundant_intervals") == [] and
        review.get("subject_relation_clear") is True and
        bool(str(review.get("supported_result") or "").strip()) and
        bool(str(review.get("core_message") or "").strip()))
    if review.get("audible_dialogue_present") is True:
        passed = passed and review.get("speech_clear") is True
    if require_music:
        passed = passed and review.get("music_present") is True
    return bool(passed)


def run_omni_edit_trial(cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path,
                        *, source_video: Path, bgm_path: Path, runner,
                        spec_sha256: str | None = None,
                        force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered.mp4").unlink(missing_ok=True)
    source_video, bgm_path = Path(source_video), Path(bgm_path)
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    if not bgm_path.is_file():
        raise OmniEditTrialBlocked("audio", "bgm_missing", str(bgm_path))

    context_start, context_end = map(float, spec["context_interval"])
    context_duration = context_end - context_start
    editing_scope = list(map(float, spec["editing_scope"]))
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    context_clip = cut_clip(
        ffmpeg, source_video, output_dir / "inputs" / "context_clip",
        start_s=context_start, end_s=context_end)
    anchors_dir = output_dir / "inputs" / "identity_anchors"
    anchor_times = (
        list(spec["identity_anchors"]["positive_source_times_s"][:2]) +
        [spec["identity_anchors"]["hard_negative_source_times_s"][0]])
    anchor_paths = []
    for index, timestamp in enumerate(anchor_times):
        label = "s0_positive" if index < 2 else "n0_hard_negative"
        anchor_paths.append(export_frame(
            ffmpeg, source_video, float(timestamp),
            anchors_dir / f"{index + 1:02d}_{label}_{float(timestamp):.3f}.jpg"))

    perception_answer = runner.inspect_media(
        anchor_paths, perception_prompt(context_duration),
        video_path=context_clip, fps=float(spec.get("perception_fps", 12.0)),
        source_origin_s=context_start, max_new_tokens=1024)
    perception_audit = _answer_audit(perception_answer)
    _write_json(output_dir / "omni" / "perception_raw.json", perception_audit)
    try:
        perception = validate_perception(
            _parse_json(perception_answer.text), duration_s=context_duration)
    except V7Blocked as exc:
        raise OmniEditTrialBlocked("perception", exc.reason_code, str(exc)) from exc
    _write_json(output_dir / "omni" / "perception.json", perception)

    allowed_relative = [round(editing_scope[0] - context_start, 6),
                        round(editing_scope[1] - context_start, 6)]
    max_segments = int(spec.get("max_segments", 3))
    max_total_s = float(spec.get("max_total_s", 12.0))
    plan_answer = runner.ask(planner_prompt(
        perception, allowed_relative=allowed_relative,
        max_segments=max_segments, max_total_s=max_total_s), max_new_tokens=768)
    plan_audit = _answer_audit(plan_answer)
    _write_json(output_dir / "omni" / "planner_raw.json", plan_audit)
    try:
        plan = normalize_edit_plan(
            _parse_json(plan_answer.text), perception=perception,
            context_start_s=context_start, editing_scope=editing_scope,
            source_video=source_video, max_segments=max_segments,
            max_total_s=max_total_s)
    except V7Blocked as exc:
        raise OmniEditTrialBlocked("planning", exc.reason_code, str(exc)) from exc
    _write_json(output_dir / "edit_plan.json", plan)

    render = render_micro_montage(
        cfg, plan, output_dir / "render", source_video=source_video,
        bgm_path=bgm_path, force=force)
    variants = {name: Path(path) for name, path in render["variants"].items()}
    if set(variants) != {"source_only", "bgm_mix"}:
        raise OmniEditTrialBlocked("audio", "audio_variants_incomplete")
    debug_preview = output_dir / "debug_preview.mp4"
    shutil.copy2(variants["bgm_mix"], debug_preview)

    segment_review_passed = False
    try:
        segment_review = review_planned_segments(
            cfg, plan, output_dir / "segment_review", runner=runner)
        segment_review_passed = bool(segment_review.get("passed"))
    except V7Blocked:
        review_path = output_dir / "segment_review" / "segment_readability_review.json"
        segment_review = (json.loads(review_path.read_text(encoding="utf-8"))
                          if review_path.is_file() else {"passed": False})

    blind = blind_review_variants(variants, runner=runner)
    for name, review in blind.items():
        review["passed"] = _blind_pass(review, require_music=name == "bgm_mix")
    similarity = _message_similarity(
        str(blind["source_only"].get("core_message") or ""),
        str(blind["bgm_mix"].get("core_message") or ""))
    blind["variant_message_similarity"] = round(similarity, 6)
    _write_json(output_dir / "blind_review.json", blind)

    automatic = (
        segment_review_passed and all(blind[name]["passed"] for name in variants) and
        similarity >= 0.5)
    acceptance = {
        "schema_version": "omni_editorial_trial_acceptance_v1",
        "passed": False,
        "automated_editorial_passed": bool(automatic),
        "human_identity_anchors_used": True,
        "automatic_identity_claimed": False,
        "human_final_review": "pending",
        "delivery": "debug_only",
        "failure_class": None if automatic else "verification",
        "failure_stage": None if automatic else (
            "segment_review" if not segment_review_passed else "blind"),
        "reason_code": "human_review_required" if automatic else
        "automated_editorial_review_failed",
        "debug_preview": str(debug_preview),
        "formal_rendered_mp4_created": False,
    }
    _write_json(output_dir / "acceptance.json", acceptance)
    manifest = {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "spec_sha256": spec_sha256 or _json_hash(spec),
        "source_video": str(source_video),
        "source_sha256": sha256_file(source_video),
        "bgm_path": str(bgm_path),
        "bgm_sha256": sha256_file(bgm_path),
        "context_interval": [context_start, context_end],
        "editing_scope": editing_scope,
        "identity_anchor_times_s": list(map(float, anchor_times)),
        "identity_anchor_sha256": [sha256_file(path) for path in anchor_paths],
        "identity_source": "human_confirmed",
        "omni_selected_intervals": [row["render_interval"] for row in plan["segments"]],
        "perception_sha256": _json_hash(perception),
        "edit_plan_sha256": _json_hash(plan),
        "render": _jsonable(render),
        "acceptance": acceptance,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {"output": str(output_dir), "debug_preview": str(debug_preview),
            "delivery": "debug_only", "automated_editorial_passed": bool(automatic)}


def write_blocked_acceptance(output_dir: Path, exc: OmniEditTrialBlocked) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered.mp4").unlink(missing_ok=True)
    acceptance = {
        "schema_version": "omni_editorial_trial_acceptance_v1",
        "passed": False,
        "automated_editorial_passed": False,
        "human_identity_anchors_used": True,
        "automatic_identity_claimed": False,
        "human_final_review": "not_started",
        "delivery": "blocked",
        "failure_class": "infrastructure" if exc.stage == "infrastructure" else
        "audio" if exc.stage == "audio" else "verification",
        "failure_stage": exc.stage,
        "reason_code": exc.reason_code,
        "detail": exc.detail or str(exc),
        "formal_rendered_mp4_created": False,
    }
    _write_json(output_dir / "acceptance.json", acceptance)
    return acceptance
