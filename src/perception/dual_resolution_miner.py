"""Two-resolution, pattern-independent evidence observation for V6.1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.evidence_units import EvidenceUnitV2, SOURCE_FORMS, interval_contained
from src.template.schema import extract_json_block


COARSE_PROMPT = """你是视频观察员。不要使用任何剪辑模板、故事槽位或角色名，只记录当前输入
片段中哪些区域值得更高帧率复看。输出严格 JSON：
{"timebase":"relative","candidate_regions":[
 {"interval":[0.0,3.0],"observation":{"subject_call_id":"person_A",
  "subject_description":"仅写可见外观","action":"可见动作或状态变化",
  "state_before":"可见前态","state_after":"可见后态"},
  "source_form":"visual_instant|dynamic_action|reaction|dialogue_span|establishing_visual",
  "attributes":["可验证事实标签"],"salience":0.0,"confidence":0.0}
]}
候选区域应为2–6秒，只回答哪里发生显著动作、状态变化、反应、对白或建立性画面。
不要输出hook/struggle/reversal/payoff等编辑角色，不要决定如何渲染。"""


DENSE_PROMPT = """你正在以高帧率复看一个已经定位的候选区域。只输出画面或声音中可验证的
最小事实，不使用任何剪辑模板、故事槽位或官方角色名。输出严格 JSON：
{"timebase":"relative","evidence":[
 {"core_interval":[0.0,0.8],"observation":{"subject_call_id":"person_A",
  "subject_description":"仅写可见外观","action":"可见动作或状态变化",
  "state_before":"可见前态","state_after":"可见后态"},
  "source_form":"visual_instant|dynamic_action|reaction|dialogue_span|establishing_visual",
  "attributes":["可验证事实标签"],"visual_strength":0.0,"confidence":0.0}
]}
core_interval 是事实真正发生的最小区间，0.15–2秒是优先目标但不是硬上限；不要添加
前后观看上下文，不要输出semantic_role、assigned_slot、render_mode或故事解释。"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bounded_score(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_payload(raw: str) -> dict[str, Any] | None:
    block = extract_json_block(str(raw or ""))
    if not block:
        return None
    try:
        payload = json.loads(block)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_interval(value: Any, *, timebase: str,
                        container: tuple[float, float],
                        scope: tuple[float, float]) -> tuple[float, float] | None:
    if isinstance(value, (int, float)):
        left = right = float(value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            left, right = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    else:
        return None
    if right == left:
        right += 0.15
    if timebase == "relative":
        left, right = container[0] + left, container[0] + right
    elif timebase != "absolute":
        return None
    result = (left, right)
    return result if right > left and interval_contained(result, container) \
        and interval_contained(result, scope) else None


def _fit_coarse_region(interval: tuple[float, float],
                       container: tuple[float, float]) -> tuple[float, float]:
    left, right = interval
    if right - left >= 2.0:
        return left, min(right, left + 6.0)
    center = (left + right) / 2.0
    left = max(container[0], center - 1.0)
    right = min(container[1], left + 2.0)
    left = max(container[0], right - 2.0)
    return left, right


def _fact_fields(row: dict[str, Any]) -> tuple[dict[str, Any], str, list[str]]:
    """Normalize a common Omni drift that nests sibling fact fields."""
    observation = dict(row.get("observation") or {})
    form = str(row.get("source_form") or
               observation.pop("source_form", "dynamic_action"))
    attributes = row.get("attributes")
    if attributes is None:
        attributes = observation.pop("attributes", [])
    return observation, form, [str(value) for value in attributes or []]


def validate_sampling(sampling: dict[str, Any] | None, *,
                      fps_min: float, fps_max: float) -> list[str]:
    if not isinstance(sampling, dict) or sampling.get("sampling_verified") is not True:
        return ["sampling_metadata_unverified"]
    timestamps = sampling.get("actual_frame_timestamps_absolute_s") or []
    count = sampling.get("actual_frame_count")
    try:
        effective = float(sampling.get("effective_fps"))
        count = int(count)
    except (TypeError, ValueError):
        return ["sampling_values_invalid"]
    reasons = []
    if count <= 0 or count != len(timestamps):
        reasons.append("sampling_frame_timestamp_count_mismatch")
    if not fps_min <= effective <= fps_max:
        reasons.append(f"effective_fps_outside_range:{effective:g}")
    if any(float(right) < float(left) for left, right in zip(timestamps, timestamps[1:])):
        reasons.append("sampling_timestamps_not_monotonic")
    return reasons


def _analysis_intervals(_shots: Any, scope: tuple[float, float], *,
                        max_watch_s: float, overlap_s: float) \
        -> list[tuple[str, float, float]]:
    # Shots are retained as an audit artifact but must not become the model or
    # editing unit.  Fixed transport windows make the requested sampling rate
    # comparable across A/B and prevent sub-second shots from being rounded to
    # an apparent 4–10 fps by Qwen's frame factor.
    start, end = scope
    if end <= start or max_watch_s <= 0 or overlap_s < 0 or overlap_s >= max_watch_s:
        return []
    if end - start <= max_watch_s:
        return [("transport_0000", start, end)]
    result = []
    cursor = start
    index = 0
    while cursor + max_watch_s < end - 1e-6:
        result.append((f"transport_{index:04d}", cursor, cursor + max_watch_s))
        cursor += max_watch_s - overlap_s
        index += 1
    final_start = end - max_watch_s
    if not result or final_start > result[-1][1] - 1e-6:
        result.append((f"transport_{index:04d}", final_start, end))
    elif abs(final_start - result[-1][0]) > 1e-6:
        result.append((f"transport_{index:04d}", final_start, end))
    return result


def parse_coarse_response(raw: str, *, watch_id: str,
                          container: tuple[float, float],
                          scope: tuple[float, float],
                          source_video: str) -> tuple[list[dict], list[dict]]:
    payload = _parse_payload(raw)
    if payload is None:
        return [], [{"reason": "coarse_json_invalid"}]
    timebase = str(payload.get("timebase") or "relative").lower()
    accepted, rejected = [], []
    for index, row in enumerate(payload.get("candidate_regions") or []):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "row_not_object"})
            continue
        normalized = _normalize_interval(
            row.get("interval"), timebase=timebase,
            container=container, scope=scope)
        observation, form, attributes = _fact_fields(row)
        if normalized is None or form not in SOURCE_FORMS:
            rejected.append({"index": index, "reason": "region_or_source_form_invalid"})
            continue
        normalized = _fit_coarse_region(normalized, container)
        accepted.append({
            "id": f"{watch_id}:region_{index:03d}",
            "interval": [round(normalized[0], 3), round(normalized[1], 3)],
            "container_interval": list(container), "observation": observation,
            "source_form": form,
            "attributes": attributes,
            "salience": _bounded_score(
                row.get("salience", (row.get("observation") or {}).get("salience"))),
            "confidence": _bounded_score(
                row.get("confidence", (row.get("observation") or {}).get("confidence"))),
            "source_video": source_video, "coarse_watch_id": watch_id,
        })
    return accepted, rejected


def parse_dense_response(raw: str, *, watch_id: str, coarse_watch_id: str,
                         container: tuple[float, float],
                         scope: tuple[float, float], source_video: str) \
        -> tuple[list[EvidenceUnitV2], list[dict]]:
    payload = _parse_payload(raw)
    if payload is None:
        return [], [{"reason": "dense_json_invalid"}]
    timebase = str(payload.get("timebase") or "relative").lower()
    accepted, rejected = [], []
    for index, row in enumerate(payload.get("evidence") or []):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "row_not_object"})
            continue
        normalized = _normalize_interval(
            row.get("core_interval", row.get("interval")), timebase=timebase,
            container=container, scope=scope)
        observation, form, attributes = _fact_fields(row)
        if normalized is None or form not in SOURCE_FORMS:
            rejected.append({"index": index, "reason": "core_or_source_form_invalid"})
            continue
        try:
            accepted.append(EvidenceUnitV2(
                id=f"{watch_id}:ev_{index:03d}", core_interval=normalized,
                container_interval=container,
                observation=observation, source_form=form,
                attributes=tuple(attributes),
                visual_strength=_bounded_score(row.get(
                    "visual_strength", (row.get("observation") or {}).get(
                        "visual_strength"))),
                confidence=_bounded_score(row.get(
                    "confidence", (row.get("observation") or {}).get("confidence"))),
                source_video=source_video,
                sampling_provenance={"coarse_watch_id": coarse_watch_id,
                                     "dense_rewatch_id": watch_id},
            ))
        except ValueError as exc:
            rejected.append({"index": index, "reason": str(exc)})
    return accepted, rejected


def _subject_key(unit: EvidenceUnitV2) -> str:
    return str(unit.subject_track_id or
               unit.observation.get("subject_description") or
               unit.observation.get("subject_call_id") or "unknown").strip().lower()


def _compatible(left: EvidenceUnitV2, right: EvidenceUnitV2) -> bool:
    if left.source_video != right.source_video or left.source_form != right.source_form:
        return False
    # subject_call_id is local to one watch and must never be used to connect
    # people across transport chunks.  Before subject linking, both track IDs
    # are intentionally unknown; the action/attribute check below is the
    # conservative deduplication signal required by the V6.1 contract.
    if (left.subject_track_id or right.subject_track_id) and \
            left.subject_track_id != right.subject_track_id:
        return False
    left_action = str(left.observation.get("action") or "").strip().lower()
    right_action = str(right.observation.get("action") or "").strip().lower()
    shared = set(left.attributes).intersection(right.attributes)
    return bool(left_action and left_action == right_action) or bool(shared)


def merge_coarse_regions(regions: Iterable[dict]) -> list[dict]:
    """Deduplicate broad hints emitted by overlapping 6-second transports."""
    merged: list[dict] = []
    for row in sorted(regions, key=lambda value: (
            float(value["interval"][0]), float(value["interval"][1]))):
        left, right = map(float, row["interval"])
        action = str((row.get("observation") or {}).get("action") or "").strip().lower()
        attributes = set(row.get("attributes") or [])
        match = None
        for old in merged:
            old_left, old_right = map(float, old["interval"])
            old_action = str((old.get("observation") or {}).get("action") or "").strip().lower()
            if row.get("source_video") != old.get("source_video") or \
                    row.get("source_form") != old.get("source_form"):
                continue
            if not (action and action == old_action) and not \
                    attributes.intersection(old.get("attributes") or []):
                continue
            if left <= old_right + 0.15 and old_left <= right + 0.15:
                match = old
                break
        if match is None:
            copy = dict(row)
            copy["merged_from_watch_ids"] = [str(row.get("coarse_watch_id"))]
            merged.append(copy)
            continue
        match["interval"] = [round(min(float(match["interval"][0]), left), 3),
                             round(max(float(match["interval"][1]), right), 3)]
        match["salience"] = max(float(match.get("salience") or 0),
                                float(row.get("salience") or 0))
        match["confidence"] = max(float(match.get("confidence") or 0),
                                  float(row.get("confidence") or 0))
        match["attributes"] = sorted(set(match.get("attributes") or []).union(attributes))
        match["merged_from_watch_ids"] = sorted(set(
            match.get("merged_from_watch_ids") or []).union(
                {str(row.get("coarse_watch_id"))}))
    return merged


def merge_observed_evidence(units: Iterable[EvidenceUnitV2]) -> list[EvidenceUnitV2]:
    """Merge the same observed fact emitted from overlapping transport chunks."""
    merged: list[EvidenceUnitV2] = []
    for unit in sorted(units, key=lambda value: (value.start_s, value.end_s, value.id)):
        match = next((old for old in merged if _compatible(old, unit) and
                      unit.start_s <= old.end_s + 0.15 and
                      old.start_s <= unit.end_s + 0.15), None)
        if match is None:
            merged.append(unit)
            continue
        watch_ids = set(match.metadata.get("merged_from_watch_ids") or [])
        watch_ids.update(value for value in (
            match.sampling_provenance.get("dense_rewatch_id"),
            unit.sampling_provenance.get("dense_rewatch_id")) if value)
        replacement = EvidenceUnitV2(
            id=match.id,
            core_interval=(min(match.start_s, unit.start_s),
                           max(match.end_s, unit.end_s)),
            container_interval=(min(match.container_interval[0], unit.container_interval[0]),
                                max(match.container_interval[1], unit.container_interval[1])),
            observation=(match.observation if match.confidence >= unit.confidence
                         else unit.observation),
            source_form=match.source_form,
            attributes=tuple(sorted(set(match.attributes).union(unit.attributes))),
            visual_strength=max(match.visual_strength, unit.visual_strength),
            confidence=max(match.confidence, unit.confidence),
            subject_track_id=match.subject_track_id or unit.subject_track_id,
            source_video=match.source_video,
            sampling_provenance={
                "coarse_watch_id": (match.sampling_provenance.get("coarse_watch_id")
                                    or unit.sampling_provenance.get("coarse_watch_id")),
                "dense_rewatch_id": match.sampling_provenance.get("dense_rewatch_id"),
            },
            metadata={"merged_from_watch_ids": sorted(watch_ids)},
        )
        merged[merged.index(match)] = replacement
    return merged


def coarse_regions_as_evidence(regions: Iterable[dict]) -> list[EvidenceUnitV2]:
    result = []
    for row in regions:
        interval = tuple(float(value) for value in row["interval"])
        result.append(EvidenceUnitV2(
            id=str(row["id"]), core_interval=interval,
            container_interval=tuple(float(value) for value in row["container_interval"]),
            observation=dict(row.get("observation") or {}),
            source_form=str(row["source_form"]),
            attributes=tuple(row.get("attributes") or []),
            visual_strength=float(row.get("salience") or 0.0),
            confidence=float(row.get("confidence") or 0.0),
            source_video=row.get("source_video"),
            sampling_provenance={"coarse_watch_id": row.get("coarse_watch_id"),
                                 "dense_rewatch_id": None},
        ))
    return result


def run_dual_resolution_mining(video: Path, shots: Any, *, runner,
                               scope_interval: Iterable[float], output_dir: Path,
                               coarse_fps: float = 2.0, dense_fps: float = 12.0,
                               max_watch_s: float = 6.0, overlap_s: float = 0.5,
                               dense_padding_s: float = 0.5,
                               max_dense_regions: int = 8) -> dict[str, Any]:
    """Run one shared coarse pass and a targeted dense pass."""
    video = Path(video).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scope = tuple(float(value) for value in scope_interval)
    intervals = _analysis_intervals(
        shots, scope, max_watch_s=max_watch_s, overlap_s=overlap_s)
    if not intervals:
        raise ValueError("no analysis intervals overlap diagnostic scope")
    coarse_requests, coarse_rows = [], []
    for watch_id, start, end in intervals:
        prompt = f"{COARSE_PROMPT}\n输入源片绝对范围：{start:.3f}–{end:.3f}s。"
        coarse_requests.append({
            "video_path": video, "prompt": prompt,
            "kwargs": {"start_s": start, "end_s": end,
                       "clip_dir": output_dir / "clips" / "coarse" / watch_id,
                       "duration_s": end - start, "fps": coarse_fps,
                       "max_new_tokens": 1024},
        })
        coarse_rows.append((watch_id, start, end, prompt))
    coarse_answers = runner.watch_many(coarse_requests)
    regions, calls, errors = [], [], []
    raw_dir = output_dir / "raw" / "coarse"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for (watch_id, start, end, prompt), answer in zip(coarse_rows, coarse_answers):
        raw = str(getattr(answer, "text", answer) or "")
        raw_path = raw_dir / f"{watch_id}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        sampling = getattr(answer, "sampling", None)
        sampling_errors = validate_sampling(sampling, fps_min=1.5, fps_max=2.5)
        parsed, rejected = parse_coarse_response(
            raw, watch_id=watch_id, container=(start, end), scope=scope,
            source_video=str(video))
        if sampling_errors:
            errors.append({"watch_id": watch_id, "failure_stage": "sampling",
                           "reasons": sampling_errors})
        elif any(row.get("reason") == "coarse_json_invalid" for row in rejected):
            errors.append({"watch_id": watch_id, "failure_stage": "mining",
                           "reasons": ["coarse_json_invalid"]})
        else:
            regions.extend(parsed)
        calls.append({
            "watch_id": watch_id, "stage": "coarse", "input_interval": [start, end],
            "requested_fps": coarse_fps, "sampling": sampling,
            "raw_response_file": str(raw_path), "prompt": prompt,
            "accepted": len(parsed) if not sampling_errors else 0,
            "rejected": rejected, "gpu_pair": getattr(answer, "gpu_pair", None),
        })
    regions = merge_coarse_regions(regions)
    regions.sort(key=lambda row: (-float(row["salience"]),
                                  -float(row["confidence"]), row["interval"][0]))
    dense_targets = regions[:max_dense_regions]
    dense_requests, dense_rows = [], []
    for index, region in enumerate(dense_targets):
        core_start, core_end = map(float, region["interval"])
        start = max(scope[0], core_start - dense_padding_s)
        end = min(scope[1], core_end + dense_padding_s)
        if end - start > max_watch_s:
            center = (core_start + core_end) / 2.0
            start = max(scope[0], center - max_watch_s / 2.0)
            end = min(scope[1], start + max_watch_s)
            start = max(scope[0], end - max_watch_s)
        watch_id = f"dense_{index:03d}"
        prompt = f"{DENSE_PROMPT}\n输入源片绝对范围：{start:.3f}–{end:.3f}s。"
        dense_requests.append({
            "video_path": video, "prompt": prompt,
            "kwargs": {"start_s": start, "end_s": end,
                       "clip_dir": output_dir / "clips" / "dense" / watch_id,
                       "duration_s": end - start, "fps": dense_fps,
                       "max_new_tokens": 1024},
        })
        dense_rows.append((watch_id, region, start, end, prompt))
    dense_answers = runner.watch_many(dense_requests) if dense_requests else []
    dense_units: list[EvidenceUnitV2] = []
    raw_dir = output_dir / "raw" / "dense"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for (watch_id, region, start, end, prompt), answer in zip(dense_rows, dense_answers):
        raw = str(getattr(answer, "text", answer) or "")
        raw_path = raw_dir / f"{watch_id}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        sampling = getattr(answer, "sampling", None)
        sampling_errors = validate_sampling(sampling, fps_min=10.0, fps_max=13.0)
        parsed, rejected = parse_dense_response(
            raw, watch_id=watch_id, coarse_watch_id=str(region["coarse_watch_id"]),
            container=(start, end), scope=scope, source_video=str(video))
        if sampling_errors:
            errors.append({"watch_id": watch_id, "failure_stage": "sampling",
                           "reasons": sampling_errors})
        elif any(row.get("reason") == "dense_json_invalid" for row in rejected):
            errors.append({"watch_id": watch_id, "failure_stage": "mining",
                           "reasons": ["dense_json_invalid"]})
        else:
            dense_units.extend(parsed)
        calls.append({
            "watch_id": watch_id, "stage": "dense", "input_interval": [start, end],
            "requested_fps": dense_fps, "sampling": sampling,
            "raw_response_file": str(raw_path), "prompt": prompt,
            "accepted": len(parsed) if not sampling_errors else 0,
            "rejected": rejected, "gpu_pair": getattr(answer, "gpu_pair", None),
            "coarse_region_id": region["id"],
        })
    coarse_units = coarse_regions_as_evidence(regions)
    dense_units = merge_observed_evidence(dense_units)
    coarse_bank = {
        "schema_version": "evidence_bank_v2", "arm": "coarse_only",
        "source_video": str(video), "scope_interval": list(scope),
        "units": [unit.to_dict() for unit in coarse_units],
    }
    dense_bank = {
        "schema_version": "evidence_bank_v2", "arm": "coarse_dense",
        "source_video": str(video), "scope_interval": list(scope),
        "units": [unit.to_dict() for unit in dense_units],
    }
    sampling_audit = {
        "schema_version": "sampling_audit_v1", "calls": calls,
        "passed": not errors, "errors": errors,
    }
    _write_json(output_dir / "coarse_regions.json", {"regions": regions})
    _write_json(output_dir / "coarse_only" / "evidence_bank.json", coarse_bank)
    _write_json(output_dir / "coarse_dense" / "evidence_bank.json", dense_bank)
    _write_json(output_dir / "sampling_audit.json", sampling_audit)
    return {
        "coarse_regions": regions, "coarse_bank": coarse_bank,
        "dense_bank": dense_bank, "sampling_audit": sampling_audit,
    }
