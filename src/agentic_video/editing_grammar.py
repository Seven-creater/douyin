"""Measured facts and evidence-linked editing grammar for P0-R2."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

from src.agentic_video.manifest import json_hash

MEASURED_EDITING_VERSION = "measured_editing_facts_v4"
STRUCTURAL_EDITING_VERSION = "measured_structural_editing_grammar_v1"
EDITING_PATTERN_VERSION = "semantic_editing_patterns_v1"
EDITING_FUNCTION_VERSION = "editing_functions_v2"
EDITING_GRAMMAR_VERSION = "editing_grammar_v3"

PATTERN_TYPES = {
    "rapid_montage", "reaction_cut", "contrast_cut", "delayed_reveal",
    "cut_on_action", "match_cut", "shot_reverse_shot", "parallel_editing",
    "insert_shot", "audio_bridge",
}
FUNCTION_TYPES = {
    "establish", "withhold", "escalate", "demonstrate", "accumulate",
    "reveal", "reframe", "contrast", "release", "humanize",
}
PATTERN_REQUIREMENTS = {
    value: {"min_shots": 2, "semantic_evidence": True}
    for value in PATTERN_TYPES
}
EDITING_CLAIM_FIELDS = (
    "claim_id", "subject", "predicate", "object", "interval", "modality",
    "polarity", "visibility",
)
EDITING_EVENT_FIELDS = (
    "event_id", "participants", "action_claim_ids", "object_ids", "ordering",
    "outcome_claim_ids", "context_claim_ids", "interval",
)

EDITING_PATTERN_PROMPT = """You are a semantic editing-pattern recognition
analyst. Deterministic pace, duration, cut, and transition structure has
already been measured and is immutable. Identify only content-dependent
editing relations that require understanding what adjacent shots contain. Do
not infer narrative function, motives, or audience beliefs. Do not return a
semantic pattern merely to restate a measured structural fact.

Allowed pattern types: rapid_montage, reaction_cut, contrast_cut,
delayed_reveal, cut_on_action, match_cut, shot_reverse_shot,
parallel_editing, insert_shot, audio_bridge.

Return exactly one JSON object:
{"schema_version":"semantic_editing_patterns_v1",
"patterns":[{"pattern_id":"EP1","type":"rapid_montage",
"scope":"local|global","shot_ids":["..."],"fact_ids":["..."],
"evidence_ids":["claim or event ID"],"confidence":0.0}],
"limitations":["..."]}

Every pattern requires at least two shots and local semantic evidence IDs.
fact_ids must correspond exactly to shot_ids. A local pattern must not cover
most of the full video. The patterns list may be empty. JSON only. Input:
"""

EDITING_FUNCTION_PROMPT = """You are an editing-function reasoning analyst.
The supplied editing patterns have already passed recognition validation. Read
each pattern with its local shots, evidence semantics, and the verified
narrative units and relations, then assign only the function supported in this
case. Do not rename, add, or remove patterns.

Allowed functions: establish, withhold, escalate, demonstrate, accumulate,
reveal, reframe, contrast, release, humanize.

Return exactly one JSON object:
{"schema_version":"editing_functions_v2",
"functions":[{"pattern_id":"EP1","function":"accumulate",
"relation_to_story":["verified relation ID"],"confidence":0.0}],
"limitations":["..."]}

Return one function for every supplied pattern. relation_to_story may be empty.
JSON only. Input:
"""


class EditingGrammarError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class EditingFact:
    fact_id: str
    shot_id: str
    section_id: str
    interval: list[float]
    duration_s: float
    transition_in: dict[str, Any] | None
    transition_out: dict[str, Any] | None
    local_shot_rate: float
    local_edit_boundary_rate: float
    local_hard_cut_rate: float
    local_transition_rate: float
    local_raw_segment_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interval(value: Any, *, reason_code: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise EditingGrammarError(reason_code)
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise EditingGrammarError(reason_code) from exc
    if start < 0 or end <= start:
        raise EditingGrammarError(reason_code)
    return start, end


def _rate(count: int, duration: float) -> float:
    return round(count / duration, 6)


def _pace_changes(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = []
    for left, right in zip(curve, curve[1:]):
        before = float(left["shot_rate"])
        after = float(right["shot_rate"])
        low = min(before, after)
        high = max(before, after)
        ratio = high / low if low > 0 else float("inf")
        if abs(after - before) < 0.25 or ratio < 1.5:
            continue
        changes.append({
            "from_section": str(left["section_id"]),
            "to_section": str(right["section_id"]),
            "direction": "accelerates" if after > before else "decelerates",
            "shot_rate_before": before,
            "shot_rate_after": after,
        })
    return changes


def _relative_pace_labels(curve: list[dict[str, Any]]) -> dict[str, str]:
    rates = sorted({float(row["shot_rate"]) for row in curve})
    if len(rates) == 1:
        labels = {rates[0]: "steady"}
    elif len(rates) == 2:
        labels = {rates[0]: "low", rates[1]: "high"}
    else:
        labels = {
            rate: ("low" if index == 0 else
                   "high" if index == len(rates) - 1 else "medium")
            for index, rate in enumerate(rates)
        }
    return {
        str(row["section_id"]): labels[float(row["shot_rate"])]
        for row in curve
    }


def _duration_outliers(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return duration outliers using Tukey's inclusive 1.5-IQR fences."""
    durations = [float(row["duration_s"]) for row in facts]
    if len(durations) < 4:
        return []
    first_quartile, _, third_quartile = quantiles(
        durations, n=4, method="inclusive")
    spread = third_quartile - first_quartile
    if spread == 0:
        return []
    lower = first_quartile - 1.5 * spread
    upper = third_quartile + 1.5 * spread
    return [{
        "fact_id": row["fact_id"],
        "shot_id": row["shot_id"],
        "section_id": row["section_id"],
        "duration_s": row["duration_s"],
        "direction": "long" if float(row["duration_s"]) > upper else "short",
        "first_quartile_s": round(first_quartile, 6),
        "third_quartile_s": round(third_quartile, 6),
        "interquartile_range_s": round(spread, 6),
    } for row in facts if (
        float(row["duration_s"]) < lower or float(row["duration_s"]) > upper)]


def _build_structural_grammar(
        pace_curve: list[dict[str, Any]],
        transition_types: dict[str, list[tuple[str, str | None]]],
        pace_changes: list[dict[str, Any]],
        facts: list[dict[str, Any]]) -> dict[str, Any]:
    relative = _relative_pace_labels(pace_curve)
    transition_profile = []
    transition_sequences = []
    for row in pace_curve:
        section_id = str(row["section_id"])
        transitions = transition_types[section_id]
        counts: dict[str, int] = {}
        for _, transition_type in transitions:
            name = transition_type or "unknown"
            counts[name] = counts.get(name, 0) + 1
        maximum = max(counts.values(), default=0)
        transition_profile.append({
            "section_id": section_id,
            "transition_count": row["transition_count"],
            "transition_rate": row["transition_rate"],
            "hard_cut_count": row["hard_cut_count"],
            "hard_cut_rate": row["hard_cut_rate"],
            "type_counts": counts,
            "dominant_types": sorted(
                key for key, count in counts.items() if count == maximum),
        })
        if transitions:
            transition_sequences.append({
                "section_id": section_id,
                "transition_ids": [value[0] for value in transitions],
                "transition_types": [value[1] or "unknown" for value in transitions],
            })
    result = {
        "schema_version": STRUCTURAL_EDITING_VERSION,
        "pace_profile": [{
            "section_id": row["section_id"],
            "shot_rate": row["shot_rate"],
            "relative_pace": relative[str(row["section_id"])],
        } for row in pace_curve],
        "pace_changes": copy.deepcopy(pace_changes),
        "duration_profile": [{
            "section_id": row["section_id"],
            "mean_shot_duration_s": row["mean_shot_duration_s"],
            "median_shot_duration_s": row["median_shot_duration_s"],
            "min_shot_duration_s": row["min_shot_duration_s"],
            "max_shot_duration_s": row["max_shot_duration_s"],
        } for row in pace_curve],
        "shot_duration_outliers": _duration_outliers(facts),
        "transition_profile": transition_profile,
        "transition_sequences": transition_sequences,
    }
    result["artifact_sha"] = json_hash(result)
    return result


def build_measured_editing_facts(agent_reference: dict[str, Any]) -> dict[str, Any]:
    """Measure shot, cut and transition pace without semantic labels."""
    timeline = agent_reference.get("deterministic_timeline") or {}
    sections = list(timeline.get("sections") or [])
    segments = list(timeline.get("segments") or [])
    cards = list(
        (agent_reference.get("shot_storyboard") or {}).get("shot_cards") or [])
    if not sections or not segments or not cards:
        raise EditingGrammarError("editing_measurement_input_missing")

    section_rows: dict[str, dict[str, Any]] = {}
    transition_types: dict[str, list[tuple[str, str | None]]] = {}
    pace_curve: list[dict[str, Any]] = []
    for section in sections:
        section_id = str(section.get("section_id") or "")
        start, end = _interval(
            section.get("interval"), reason_code="section_interval_invalid")
        if not section_id or section_id in section_rows:
            raise EditingGrammarError("section_id_invalid", section_id)
        member_segments = [
            row for row in segments if str(row.get("section_id") or "") == section_id
        ]
        content_segments = [
            row for row in member_segments if row.get("segment_kind") == "content"
        ]
        transition_segments = [
            row for row in member_segments if row.get("segment_kind") == "transition"
        ]
        duration = end - start
        shot_durations = [
            shot_end - shot_start for shot_start, shot_end in (
                _interval(row.get("interval"), reason_code="shot_interval_invalid")
                for row in content_segments)
        ]
        if not shot_durations:
            raise EditingGrammarError("section_content_shots_missing", section_id)
        shot_count = len(content_segments)
        edit_boundary_count = max(shot_count - 1, 0)
        transition_count = len(transition_segments)
        hard_cut_count = max(edit_boundary_count - transition_count, 0)
        row = {
            "section_id": section_id,
            "interval": [start, end],
            "duration_s": round(duration, 6),
            "content_shot_count": shot_count,
            "edit_boundary_count": edit_boundary_count,
            "hard_cut_count": hard_cut_count,
            "transition_count": transition_count,
            "shot_rate": _rate(shot_count, duration),
            "edit_boundary_rate": _rate(edit_boundary_count, duration),
            "hard_cut_rate": _rate(hard_cut_count, duration),
            "transition_rate": _rate(transition_count, duration),
            "mean_shot_duration_s": round(mean(shot_durations), 6),
            "median_shot_duration_s": round(median(shot_durations), 6),
            "min_shot_duration_s": round(min(shot_durations), 6),
            "max_shot_duration_s": round(max(shot_durations), 6),
            "raw_segment_count": len(member_segments),
            "raw_segment_rate": _rate(len(member_segments), duration),
            "shot_ids": [str(row.get("segment_id")) for row in content_segments],
            "transition_ids": [
                str(row.get("segment_id")) for row in transition_segments],
        }
        section_rows[section_id] = row
        transition_types[section_id] = [
            (str(segment.get("segment_id")),
             str(segment.get("transition_type"))
             if segment.get("transition_type") else None)
            for segment in transition_segments
        ]
        pace_curve.append(row)
    pace_curve.sort(key=lambda row: row["interval"])

    facts: list[dict[str, Any]] = []
    for index, card in enumerate(cards, start=1):
        section_id = str(card.get("section_id") or "")
        if section_id not in section_rows:
            raise EditingGrammarError("shot_section_missing", str(card.get("shot_id")))
        start, end = _interval(
            [card.get("start_s"), card.get("end_s")],
            reason_code="shot_card_interval_invalid")
        section = section_rows[section_id]
        facts.append(EditingFact(
            fact_id=f"EF{index}",
            shot_id=str(card.get("shot_id") or ""),
            section_id=section_id,
            interval=[start, end],
            duration_s=round(end - start, 6),
            transition_in=card.get("transition_in"),
            transition_out=card.get("transition_out"),
            local_shot_rate=float(section["shot_rate"]),
            local_edit_boundary_rate=float(section["edit_boundary_rate"]),
            local_hard_cut_rate=float(section["hard_cut_rate"]),
            local_transition_rate=float(section["transition_rate"]),
            local_raw_segment_rate=float(section["raw_segment_rate"]),
        ).to_dict())

    start_s = min(row["interval"][0] for row in pace_curve)
    end_s = max(row["interval"][1] for row in pace_curve)
    pace_changes = _pace_changes(pace_curve)
    result = {
        "schema_version": MEASURED_EDITING_VERSION,
        "source_sha": agent_reference.get("source_sha"),
        "agent_reference_sha": agent_reference.get("artifact_sha"),
        "duration_s": round(end_s - start_s, 6),
        "facts": facts,
        "pace_curve": pace_curve,
        "pace_changes": pace_changes,
        "structural_grammar": _build_structural_grammar(
            pace_curve, transition_types, pace_changes, facts),
    }
    result["artifact_sha"] = json_hash(result)
    return result


def build_editing_payload(agent_reference: dict[str, Any],
                          measured: dict[str, Any]) -> dict[str, Any]:
    """Publish compact, deduplicated semantics for pattern recognition."""
    claims = {
        str(row.get("claim_id")): row for row in agent_reference.get("claims") or []
    }
    events = {
        str(row.get("event_id")): row for row in agent_reference.get("events") or []
    }

    def require(ids: list[str], index: dict[str, dict[str, Any]],
                reason_code: str) -> list[str]:
        missing = [str(value) for value in ids if str(value) not in index]
        if missing:
            raise EditingGrammarError(reason_code, ",".join(missing))
        return [str(value) for value in ids]

    def project(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        return {key: row.get(key) for key in fields if key in row}

    shot_rows = []
    used_claim_ids: set[str] = set()
    used_event_ids: set[str] = set()
    for card in (agent_reference.get("shot_storyboard") or {}).get(
            "shot_cards") or []:
        visual_ids = require(
            list(card.get("visual_claim_ids") or []), claims,
            "editing_visual_claim_missing")
        text_ids = require(
            list(card.get("text_claim_ids") or []), claims,
            "editing_text_claim_missing")
        audio_ids = require(
            list(card.get("audio_claim_ids") or []), claims,
            "editing_audio_claim_missing")
        event_ids = require(
            list(card.get("event_ids") or []), events,
            "editing_event_missing")
        used_claim_ids.update(visual_ids + text_ids + audio_ids)
        used_event_ids.update(event_ids)
        shot_rows.append({
            "shot_id": card.get("shot_id"),
            "section_id": card.get("section_id"),
            "interval": [card.get("start_s"), card.get("end_s")],
            "transition_in": card.get("transition_in"),
            "transition_out": card.get("transition_out"),
            "visual_claim_ids": visual_ids,
            "text_claim_ids": text_ids,
            "audio_claim_ids": audio_ids,
            "event_ids": event_ids,
            "canonical_subject_ids": list(card.get("entity_ids") or []),
        })
    measured_payload = {
        key: value for key, value in measured.items()
        if key not in {"source_sha", "agent_reference_sha", "artifact_sha"}
    }
    return {
        "measured_editing": measured_payload,
        "claim_index": {
            claim_id: project(claims[claim_id], EDITING_CLAIM_FIELDS)
            for claim_id in sorted(used_claim_ids)
        },
        "event_index": {
            event_id: project(events[event_id], EDITING_EVENT_FIELDS)
            for event_id in sorted(used_event_ids)
        },
        "shots": shot_rows,
    }


def _shot_evidence_ids(payload: dict[str, Any]) -> dict[str, set[str]]:
    fields = ("visual_claim_ids", "text_claim_ids", "audio_claim_ids", "event_ids")
    return {
        str(row.get("shot_id")): {
            str(value) for field in fields for value in row.get(field) or []
        }
        for row in payload.get("shots") or []
    }


def build_editing_function_payload(
        patterns: dict[str, Any], interpretation: dict[str, Any],
        editing_payload: dict[str, Any]) -> dict[str, Any]:
    """Attach only each recognized pattern's local compact semantics."""
    shots = {
        str(row.get("shot_id")): row for row in editing_payload.get("shots") or []
    }
    claim_index = editing_payload.get("claim_index") or {}
    event_index = editing_payload.get("event_index") or {}
    contexts = []
    for pattern in patterns.get("patterns") or []:
        shot_ids = list(map(str, pattern.get("shot_ids") or []))
        selected = [shots[shot_id] for shot_id in shot_ids]
        claim_ids = {
            str(value) for row in selected
            for field in ("visual_claim_ids", "text_claim_ids", "audio_claim_ids")
            for value in row.get(field) or []
        }
        event_ids = {
            str(value) for row in selected for value in row.get("event_ids") or []
        }
        contexts.append({
            "pattern": pattern,
            "shots": selected,
            "claim_index": {
                claim_id: claim_index[claim_id] for claim_id in sorted(claim_ids)
            },
            "event_index": {
                event_id: event_index[event_id] for event_id in sorted(event_ids)
            },
        })
    return {
        "patterns": contexts,
        "verified_narrative_interpretation": interpretation,
    }


def _confidence(value: Any, reason_code: str, detail: str) -> None:
    if not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise EditingGrammarError(reason_code, detail)


def validate_editing_patterns(
        value: dict[str, Any], measured: dict[str, Any], *,
        shot_evidence_ids: dict[str, set[str]]) -> dict[str, Any]:
    if value.get("schema_version") != EDITING_PATTERN_VERSION:
        raise EditingGrammarError("editing_pattern_schema_invalid")
    patterns = value.get("patterns")
    if not isinstance(patterns, list) or not isinstance(value.get("limitations"), list):
        raise EditingGrammarError("editing_pattern_shape_invalid")
    facts = {str(row.get("fact_id")): row for row in measured.get("facts") or []}
    shots = {str(row.get("shot_id")): row for row in measured.get("facts") or []}
    facts_by_shot = {
        str(row.get("shot_id")): str(row.get("fact_id"))
        for row in measured.get("facts") or []
    }
    total = float(measured.get("duration_s") or 0.0)
    if total <= 0:
        raise EditingGrammarError("editing_duration_invalid")

    result = copy.deepcopy(value)
    pattern_ids: set[str] = set()
    for pattern in result["patterns"]:
        pattern_id = str(pattern.get("pattern_id") or "")
        if not pattern_id or pattern_id in pattern_ids:
            raise EditingGrammarError("pattern_id_invalid", pattern_id)
        pattern_ids.add(pattern_id)
        pattern_type = str(pattern.get("type") or "")
        if pattern_type not in PATTERN_TYPES:
            raise EditingGrammarError("pattern_type_invalid", pattern_id)
        requirements = PATTERN_REQUIREMENTS[pattern_type]
        if pattern.get("scope") not in {"local", "global"}:
            raise EditingGrammarError("pattern_scope_invalid", pattern_id)
        shot_ids = list(map(str, pattern.get("shot_ids") or []))
        if (len(shot_ids) < int(requirements["min_shots"]) or
                len(set(shot_ids)) != len(shot_ids)):
            raise EditingGrammarError("pattern_shot_cardinality_invalid", pattern_id)
        if not set(shot_ids).issubset(shots):
            raise EditingGrammarError("pattern_shot_invalid", pattern_id)
        fact_ids = list(map(str, pattern.get("fact_ids") or []))
        if not fact_ids or not set(fact_ids).issubset(facts):
            raise EditingGrammarError("pattern_fact_invalid", pattern_id)
        expected_fact_ids = {facts_by_shot[shot_id] for shot_id in shot_ids}
        if (len(set(fact_ids)) != len(fact_ids) or
                set(fact_ids) != expected_fact_ids):
            raise EditingGrammarError("pattern_fact_scope_invalid", pattern_id)
        if not set(shot_ids).issubset(shot_evidence_ids):
            raise EditingGrammarError("pattern_evidence_scope_invalid", pattern_id)
        local_evidence = set().union(
            *(shot_evidence_ids[shot_id] for shot_id in shot_ids))
        cited = list(map(str, pattern.get("evidence_ids") or []))
        if requirements["semantic_evidence"] and not cited:
            raise EditingGrammarError("pattern_evidence_required", pattern_id)
        if not set(cited).issubset(local_evidence):
            raise EditingGrammarError("pattern_evidence_scope_invalid", pattern_id)
        _confidence(pattern.get("confidence"), "pattern_confidence_invalid", pattern_id)
        start = min(float(shots[shot_id]["interval"][0]) for shot_id in shot_ids)
        end = max(float(shots[shot_id]["interval"][1]) for shot_id in shot_ids)
        pattern["span"] = [start, end]
        if pattern["scope"] != "global" and (end - start) / total > 0.60:
            raise EditingGrammarError("local_pattern_too_broad", pattern_id)
    result["measured_editing_sha"] = measured.get("artifact_sha")
    result["artifact_sha"] = json_hash(result)
    return result


def validate_editing_functions(
        value: dict[str, Any], patterns: dict[str, Any], *,
        relation_ids: set[str]) -> dict[str, Any]:
    if value.get("schema_version") != EDITING_FUNCTION_VERSION:
        raise EditingGrammarError("editing_function_schema_invalid")
    functions = value.get("functions")
    if not isinstance(functions, list) or not isinstance(value.get("limitations"), list):
        raise EditingGrammarError("editing_function_shape_invalid")
    pattern_ids = {
        str(row.get("pattern_id")) for row in patterns.get("patterns") or []
    }
    actual: set[str] = set()
    result = copy.deepcopy(value)
    for function in result["functions"]:
        pattern_id = str(function.get("pattern_id") or "")
        if pattern_id not in pattern_ids or pattern_id in actual:
            raise EditingGrammarError("function_pattern_invalid", pattern_id)
        actual.add(pattern_id)
        if function.get("function") not in FUNCTION_TYPES:
            raise EditingGrammarError("function_type_invalid", pattern_id)
        links = function.get("relation_to_story")
        if not isinstance(links, list) or not set(map(str, links)).issubset(relation_ids):
            raise EditingGrammarError("function_story_relation_invalid", pattern_id)
        _confidence(function.get("confidence"), "function_confidence_invalid", pattern_id)
    if actual != pattern_ids:
        raise EditingGrammarError("pattern_function_coverage_invalid")
    result["patterns_sha"] = patterns.get("artifact_sha")
    result["artifact_sha"] = json_hash(result)
    return result


def _parse_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EditingGrammarError("editing_response_json_invalid") from exc
    if not isinstance(value, dict):
        raise EditingGrammarError("editing_response_not_object")
    return value


def _write_trace(trace_dir: Path | None, name: str, content: str) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / name).write_text(content, encoding="utf-8")


def analyze_editing_grammar(
        pattern_runner: Any, function_runner: Any,
        agent_reference: dict[str, Any], interpretation: dict[str, Any], *,
        trace_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure once, recognize patterns once, then reason functions once."""
    measured = build_measured_editing_facts(agent_reference)
    pattern_payload = build_editing_payload(agent_reference, measured)
    pattern_request = EDITING_PATTERN_PROMPT + json.dumps(
        pattern_payload, ensure_ascii=False, separators=(",", ":"))
    _write_trace(trace_dir, "editing_pattern_request.txt", pattern_request)
    pattern_answer = pattern_runner.ask(
        pattern_request, max_new_tokens=4096, stop_after_json_object=True)
    pattern_raw = str(getattr(pattern_answer, "text", pattern_answer))
    _write_trace(trace_dir, "editing_pattern_response.txt", pattern_raw)
    patterns = validate_editing_patterns(
        _parse_object(pattern_raw), measured,
        shot_evidence_ids=_shot_evidence_ids(pattern_payload))

    function_payload = build_editing_function_payload(
        patterns, interpretation, pattern_payload)
    function_request = EDITING_FUNCTION_PROMPT + json.dumps(
        function_payload, ensure_ascii=False, separators=(",", ":"))
    _write_trace(trace_dir, "editing_function_request.txt", function_request)
    function_answer = function_runner.ask(
        function_request, max_new_tokens=3072, stop_after_json_object=True)
    function_raw = str(getattr(function_answer, "text", function_answer))
    _write_trace(trace_dir, "editing_function_response.txt", function_raw)
    relation_ids = {
        str(row.get("relation_id"))
        for row in interpretation.get("relations") or []
        if row.get("verification_status") == "SUPPORTED"
    }
    functions = validate_editing_functions(
        _parse_object(function_raw), patterns, relation_ids=relation_ids)
    result = {
        "schema_version": EDITING_GRAMMAR_VERSION,
        "structural_grammar": measured["structural_grammar"],
        "semantic_patterns": patterns["patterns"],
        "functions": functions["functions"],
        "pattern_limitations": patterns["limitations"],
        "function_limitations": functions["limitations"],
        "measured_editing_sha": measured.get("artifact_sha"),
        "semantic_patterns_sha": patterns.get("artifact_sha"),
        "functions_sha": functions.get("artifact_sha"),
    }
    result["artifact_sha"] = json_hash(result)
    return result, measured
