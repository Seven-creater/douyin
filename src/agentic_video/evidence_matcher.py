"""Subject linking, pattern matching, and deterministic V6.1 planning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.evidence_units import (EvidenceUnitV2, SOURCE_FORMS,
                                               evidence_v2_from_dict)
from src.template.schema import extract_json_block


RENDER_MODES = {"micro_clip", "keyframe_hold"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_json(raw: str) -> dict[str, Any] | None:
    block = extract_json_block(str(raw or ""))
    if not block:
        return None
    try:
        value = json.loads(block)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def validate_pattern_contract(pattern: dict[str, Any]) -> list[str]:
    reasons = []
    slots = pattern.get("slots") or []
    if not slots:
        return ["pattern_slots_missing"]
    seen = set()
    for row in slots:
        if not isinstance(row, dict) or not str(row.get("id") or "").strip():
            reasons.append("pattern_slot_invalid")
            continue
        slot_id = str(row["id"])
        if slot_id in seen:
            reasons.append(f"pattern_slot_duplicate:{slot_id}")
        seen.add(slot_id)
        forms = set(row.get("allowed_source_forms") or [])
        modes = set(row.get("render_modes") or [])
        if not forms or not forms.issubset(SOURCE_FORMS):
            reasons.append(f"pattern_source_form_unsupported:{slot_id}")
        if not modes or not modes.issubset(RENDER_MODES):
            reasons.append(f"pattern_render_mode_unsupported:{slot_id}")
    return reasons


def link_subject_tracks(bank: dict[str, Any], *, runner, output: Path) -> dict[str, Any]:
    """Link call-local visual subjects without assigning official identities."""
    rows = []
    for row in bank.get("units") or []:
        observation = row.get("observation") or {}
        rows.append({
            "evidence_id": row.get("id"),
            "source_interval": row.get("core_interval"),
            "subject_call_id": observation.get("subject_call_id"),
            "subject_description": observation.get("subject_description"),
            "action": observation.get("action"),
        })
    prompt = """你是局部人物轨迹连接器。只根据可见外观描述、相邻时间和动作，把同一个可见
主体连接为local_subject_01等局部ID；不知道就写null，绝不猜官方姓名。输出严格JSON：
{"links":[{"evidence_id":"...","subject_track_id":"local_subject_01|null",
"confidence":0.0}],"uncertain_evidence_ids":[]}
输入：""" + json.dumps(rows, ensure_ascii=False)
    answer = runner.ask(prompt, max_new_tokens=1024)
    raw = str(getattr(answer, "text", answer) or "")
    parsed = _parse_json(raw)
    links = {}
    if parsed:
        for row in parsed.get("links") or []:
            if not isinstance(row, dict) or row.get("evidence_id") is None:
                continue
            track = row.get("subject_track_id")
            if track and str(track).lower() != "null":
                links[str(row["evidence_id"])] = str(track)
    updated = []
    for row in bank.get("units") or []:
        copy = dict(row)
        copy["subject_track_id"] = links.get(str(row.get("id")))
        updated.append(copy)
    result = {
        "schema_version": "subject_tracks_v1", "parsed": parsed is not None,
        "links": [{"evidence_id": key, "subject_track_id": value}
                  for key, value in sorted(links.items())],
        "uncertain_evidence_ids": ([str(value) for value in
                                    (parsed or {}).get("uncertain_evidence_ids") or []]),
        "raw_response": raw,
    }
    bank = {**bank, "units": updated, "subject_linking": {
        "parsed": result["parsed"], "linked": len(links),
        "total": len(updated),
    }}
    _write_json(Path(output), result)
    return bank


def match_evidence_to_pattern(bank: dict[str, Any], pattern: dict[str, Any], *,
                              runner, output: Path) -> dict[str, Any]:
    """Score observations against slots without deleting any Evidence."""
    slots = [{
        "id": row["id"], "function": row.get("function"),
        "allowed_source_forms": row.get("allowed_source_forms"),
    } for row in pattern.get("slots") or []]
    evidence = [{
        "evidence_id": row.get("id"), "core_interval": row.get("core_interval"),
        "observation": row.get("observation"), "source_form": row.get("source_form"),
        "attributes": row.get("attributes"),
    } for row in bank.get("units") or []]
    prompt = """你是编辑模式匹配器。Evidence是观察事实，Slot是当前参考视频的编辑功能。
为每个Evidence对每个Slot给0到1的适配分；不得改Evidence、不得补画面外信息。
输出严格JSON：{"matches":[{"evidence_id":"...","slot_fit":{"hook":0.0}}]}。
Slots：""" + json.dumps(slots, ensure_ascii=False) + "\nEvidence：" + \
        json.dumps(evidence, ensure_ascii=False)
    answer = runner.ask(prompt, max_new_tokens=2048)
    raw = str(getattr(answer, "text", answer) or "")
    parsed = _parse_json(raw)
    valid_ids = {str(row.get("id")) for row in bank.get("units") or []}
    slot_ids = {str(row.get("id")) for row in pattern.get("slots") or []}
    matches = []
    if parsed:
        for row in parsed.get("matches") or []:
            evidence_id = str(row.get("evidence_id") or "")
            if evidence_id not in valid_ids:
                continue
            scores = {}
            for key, value in (row.get("slot_fit") or {}).items():
                if str(key) not in slot_ids:
                    continue
                try:
                    scores[str(key)] = min(1.0, max(0.0, float(value)))
                except (TypeError, ValueError):
                    scores[str(key)] = 0.0
            matches.append({"evidence_id": evidence_id, "slot_fit": scores})
    result = {
        "schema_version": "pattern_matches_v1", "parsed": parsed is not None,
        "matches": matches, "raw_response": raw,
    }
    _write_json(Path(output), result)
    return result


def oracle_pattern_matches(bank: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    expected = {str(row["id"]): set(str(value) for value in row.get("expected_slots") or [])
                for row in oracle.get("facts") or [] if row.get("id")}
    return {
        "schema_version": "pattern_matches_v1", "parsed": True, "oracle": True,
        "matches": [{
            "evidence_id": str(row.get("id")),
            "slot_fit": {slot: 1.0 for slot in expected.get(str(row.get("id")), set())},
        } for row in bank.get("units") or []],
    }


def _render_choice(unit: EvidenceUnitV2, slot: dict[str, Any], *,
                   max_segment_s: float, pre_roll_s: float,
                   post_roll_s: float) -> dict[str, Any] | None:
    core_start, core_end = unit.core_interval
    if core_end - core_start > max_segment_s + 1e-6:
        return None
    modes = list(slot.get("render_modes") or ["micro_clip"])
    mode = "keyframe_hold" if unit.source_form == "visual_instant" \
        and "keyframe_hold" in modes else "micro_clip"
    if mode not in modes:
        mode = modes[0]
    container_start, container_end = unit.container_interval
    left = max(container_start, core_start - pre_roll_s)
    right = min(container_end, core_end + post_roll_s)
    overflow = right - left - max_segment_s
    if overflow > 0:
        trim_left = min(core_start - left, overflow / 2.0)
        left += trim_left
        right -= overflow - trim_left
    if left > core_start + 1e-6 or right < core_end - 1e-6:
        return None
    duration = right - left
    if mode == "keyframe_hold":
        duration = min(max_segment_s, max(0.3, min(1.5, duration)))
    return {
        "render_mode": mode,
        "render_interval": [round(left, 3), round(right, 3)],
        "source_interval": [round(left, 3), round(right, 3)],
        "frame_time_s": round((core_start + core_end) / 2.0, 3),
        "duration_s": round(duration, 3),
        "context_padding": {
            "pre_roll_s": round(core_start - left, 3),
            "post_roll_s": round(right - core_end, 3),
            "reason": "anticipation_and_impact",
        },
    }


def build_v61_edit_plan(bank: dict[str, Any], pattern: dict[str, Any],
                        matches: dict[str, Any], *,
                        min_duration_s: float = 4.0,
                        max_duration_s: float = 20.0,
                        max_segment_s: float = 4.0,
                        pre_roll_s: float = 0.3,
                        post_roll_s: float = 0.2) -> dict[str, Any]:
    """Assign matched observations to slots and decide rendering explicitly."""
    units = {unit.id: unit for unit in (
        evidence_v2_from_dict(row) for row in bank.get("units") or [])}
    score_map = {str(row.get("evidence_id")): dict(row.get("slot_fit") or {})
                 for row in matches.get("matches") or []}
    diagnostics, segments, missing = [], [], []
    used: dict[str, int] = {}
    group_subjects: dict[str, str] = {}
    slot_order = [str(row.get("id")) for row in pattern.get("slots") or []]
    for slot_index, slot in enumerate(pattern.get("slots") or []):
        slot_id = str(slot["id"])
        threshold = float(slot.get("slot_fit_threshold", 0.7))
        semantic = [unit for unit in units.values()
                    if float(score_map.get(unit.id, {}).get(slot_id, 0.0)) >= threshold]
        allowed_forms = set(slot.get("allowed_source_forms") or [])
        eligible = [unit for unit in semantic if unit.source_form in allowed_forms]
        group = ("" if pattern.get("subject_switch_allowed") is True else
                 str(slot.get("same_subject_group") or ""))
        required_subject = group_subjects.get(group) if group else None
        if group and required_subject:
            eligible = [unit for unit in eligible
                        if unit.subject_track_id == required_subject]
        elif group:
            eligible = [unit for unit in eligible if unit.subject_track_id]
        eligible.sort(key=lambda unit: (
            -float(score_map.get(unit.id, {}).get(slot_id, 0.0)),
            -unit.visual_strength, -unit.confidence, unit.start_s))
        diagnostics.append({
            "slot_id": slot_id, "semantic_available": bool(semantic),
            "eligible_available": bool(eligible),
            "semantic_candidate_ids": [unit.id for unit in semantic],
            "eligible_candidate_ids": [unit.id for unit in eligible],
            "excluded": [{"evidence_id": unit.id,
                          "reason": "source_form_not_allowed",
                          "source_form": unit.source_form}
                         for unit in semantic if unit.source_form not in allowed_forms],
        })
        if not eligible:
            if slot.get("required", True):
                missing.append(slot_id)
            continue
        unused = [unit for unit in eligible if unit.id not in used]
        chosen = unused[0] if unused else eligible[0]
        if chosen.id in used:
            previous_index = used[chosen.id]
            previous = segments[previous_index]
            previous_slots = previous["assigned_slots"]
            previous_slot_index = slot_order.index(previous_slots[-1])
            if slot_index == previous_slot_index + 1 and slot.get("allow_slot_fusion", True):
                previous_slots.append(slot_id)
                previous["slot_fit"][slot_id] = score_map[chosen.id][slot_id]
                continue
            if slot.get("required", True):
                missing.append(slot_id)
            continue
        render = _render_choice(
            chosen, slot, max_segment_s=max_segment_s,
            pre_roll_s=pre_roll_s, post_roll_s=post_roll_s)
        if render is None:
            if slot.get("required", True):
                missing.append(slot_id)
            diagnostics[-1]["render_rejection"] = "core_exceeds_render_limit"
            continue
        if group and group not in group_subjects:
            group_subjects[group] = str(chosen.subject_track_id)
        segment = {
            "index": len(segments), "evidence_id": chosen.id,
            "unit_id": chosen.id, "assigned_slots": [slot_id],
            "editing_role": slot_id, "render_once": True,
            "render_mode": render["render_mode"],
            "unit_type": render["render_mode"],
            "core_interval": list(chosen.core_interval),
            "subject_track_id": chosen.subject_track_id,
            "source_form": chosen.source_form, "source_video": chosen.source_video,
            "slot_fit": {slot_id: score_map[chosen.id][slot_id]}, **render,
        }
        used[chosen.id] = len(segments)
        segments.append(segment)
    duration = sum(float(row["duration_s"]) for row in segments)
    reasons = []
    if missing:
        reasons.append("required_slots_missing:" + ",".join(missing))
    if duration < min_duration_s:
        reasons.append(f"montage_below_min_duration:{duration:.3f}")
    if duration > max_duration_s + 1e-6:
        reasons.append(f"montage_over_max_duration:{duration:.3f}")
    hook_diag = next((row for row in diagnostics if row["slot_id"] == "hook"), {})
    return {
        "schema_version": "evidence_edit_plan_v61", "passed": not reasons,
        "failure_class": None if not reasons else "content",
        "failure_stage": None if not reasons else "planning",
        "reason_code": None if not reasons else reasons[0].split(":", 1)[0],
        "reasons": reasons, "duration_s": round(duration, 3),
        "min_duration_s": min_duration_s, "max_duration_s": max_duration_s,
        "segments": segments, "slot_diagnostics": diagnostics,
        "missing_slots": missing,
        "semantic_hook_available": bool(hook_diag.get("semantic_available")),
        "eligible_visual_hook": any(
            units[evidence_id].source_form != "dialogue_span"
            for evidence_id in hook_diag.get("eligible_candidate_ids") or []),
        "subject_groups": group_subjects,
    }
