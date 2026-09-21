"""Measured facts and evidence-linked editing grammar for P0-R2."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash

MEASURED_EDITING_VERSION = "measured_editing_facts_v1"
EDITING_GRAMMAR_VERSION = "editing_grammar_v1"

PATTERN_TYPES = {
    "long_hold", "rapid_montage", "rhythmic_acceleration",
    "rhythmic_deceleration", "reaction_cut", "contrast_cut",
    "delayed_reveal", "cut_on_action", "match_cut",
    "shot_reverse_shot", "transition_chain", "parallel_editing",
    "insert_shot", "audio_bridge",
}
FUNCTION_TYPES = {
    "establish", "withhold", "escalate", "demonstrate", "accumulate",
    "reveal", "reframe", "contrast", "release", "humanize",
}
PACE_PATTERN_TYPES = {
    "long_hold", "rapid_montage", "rhythmic_acceleration",
    "rhythmic_deceleration",
}

EDITING_GRAMMAR_PROMPT = """You are an editing-pattern analyst. The supplied
durations, order, transitions, and density values are immutable measurements.
First identify local editing patterns; then assign a separate information
function to each pattern. Do not recalculate timing and do not infer character
motives or audience beliefs.

Allowed pattern types: long_hold, rapid_montage, rhythmic_acceleration,
rhythmic_deceleration, reaction_cut, contrast_cut, delayed_reveal,
cut_on_action, match_cut, shot_reverse_shot, transition_chain,
parallel_editing, insert_shot, audio_bridge.

Allowed functions: establish, withhold, escalate, demonstrate, accumulate,
reveal, reframe, contrast, release, humanize.

Return exactly one JSON object:
{"schema_version":"editing_grammar_v1",
"patterns":[{"pattern_id":"EP1","type":"rapid_montage",
"scope":"local|global","shot_ids":["..."],"fact_ids":["..."],
"confidence":0.0}],
"functions":[{"pattern_id":"EP1","function":"accumulate",
"relation_to_story":["relation ID"],"confidence":0.0}],
"limitations":["..."]}

Every pattern requires at least two shots. A local pattern must not cover most
of the full video. A significant density change requires a local pace pattern.
It is valid to leave relation_to_story empty. JSON only. Input:
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
    transition_in: str | None
    transition_out: str | None
    local_cut_density: float

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


def build_measured_editing_facts(agent_reference: dict[str, Any]) -> dict[str, Any]:
    """Measure timing and density without assigning editing functions."""
    timeline = agent_reference.get("deterministic_timeline") or {}
    sections = list(timeline.get("sections") or [])
    segments = list(timeline.get("segments") or [])
    cards = list(
        (agent_reference.get("shot_storyboard") or {}).get("shot_cards") or [])
    if not sections or not segments or not cards:
        raise EditingGrammarError("editing_measurement_input_missing")

    section_rows: dict[str, dict[str, Any]] = {}
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
        member_shots = [
            str(row.get("shot_id")) for row in cards
            if str(row.get("section_id") or "") == section_id
        ]
        duration = end - start
        density = round(len(member_segments) / duration, 6)
        row = {
            "section_id": section_id,
            "interval": [start, end],
            "duration_s": round(duration, 6),
            "segment_count": len(member_segments),
            "shot_ids": member_shots,
            "density": density,
        }
        section_rows[section_id] = row
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
        facts.append(EditingFact(
            fact_id=f"EF{index}",
            shot_id=str(card.get("shot_id") or ""),
            section_id=section_id,
            interval=[start, end],
            duration_s=round(end - start, 6),
            transition_in=card.get("transition_in"),
            transition_out=card.get("transition_out"),
            local_cut_density=float(section_rows[section_id]["density"]),
        ).to_dict())

    start_s = min(row["interval"][0] for row in pace_curve)
    end_s = max(row["interval"][1] for row in pace_curve)
    result = {
        "schema_version": MEASURED_EDITING_VERSION,
        "source_sha": agent_reference.get("source_sha"),
        "agent_reference_sha": agent_reference.get("artifact_sha"),
        "duration_s": round(end_s - start_s, 6),
        "facts": facts,
        "pace_curve": pace_curve,
    }
    result["artifact_sha"] = json_hash(result)
    return result


def _significant_pace_changes(measured: dict[str, Any]) -> list[dict[str, Any]]:
    curve = list(measured.get("pace_curve") or [])
    changes = []
    for left, right in zip(curve, curve[1:]):
        before = float(left.get("density") or 0.0)
        after = float(right.get("density") or 0.0)
        low = min(before, after)
        high = max(before, after)
        ratio = high / low if low > 0 else float("inf")
        if abs(after - before) < 0.25 or ratio < 1.5:
            continue
        changes.append({
            "from_section": str(left.get("section_id")),
            "to_section": str(right.get("section_id")),
            "direction": "accelerates" if after > before else "decelerates",
        })
    return changes


def validate_editing_grammar(
        value: dict[str, Any], measured: dict[str, Any], *,
        relation_ids: set[str]) -> dict[str, Any]:
    if value.get("schema_version") != EDITING_GRAMMAR_VERSION:
        raise EditingGrammarError("editing_grammar_schema_invalid")
    patterns = value.get("patterns")
    functions = value.get("functions")
    if not isinstance(patterns, list) or not isinstance(functions, list):
        raise EditingGrammarError("editing_grammar_shape_invalid")
    if not isinstance(value.get("limitations"), list):
        raise EditingGrammarError("editing_grammar_limitations_invalid")

    facts = {str(row.get("fact_id")): row for row in measured.get("facts") or []}
    shots = {str(row.get("shot_id")): row for row in measured.get("facts") or []}
    total = float(measured.get("duration_s") or 0.0)
    if total <= 0:
        raise EditingGrammarError("editing_duration_invalid")
    result = copy.deepcopy(value)
    pattern_ids: set[str] = set()
    pattern_sections: dict[str, set[str]] = {}
    pattern_types: dict[str, str] = {}
    for pattern in result["patterns"]:
        pattern_id = str(pattern.get("pattern_id") or "")
        if not pattern_id or pattern_id in pattern_ids:
            raise EditingGrammarError("pattern_id_invalid", pattern_id)
        pattern_ids.add(pattern_id)
        pattern_type = str(pattern.get("type") or "")
        if pattern_type not in PATTERN_TYPES:
            raise EditingGrammarError("pattern_type_invalid", pattern_id)
        pattern_types[pattern_id] = pattern_type
        if pattern.get("scope") not in {"local", "global"}:
            raise EditingGrammarError("pattern_scope_invalid", pattern_id)
        shot_ids = list(map(str, pattern.get("shot_ids") or []))
        if len(shot_ids) < 2 or len(set(shot_ids)) != len(shot_ids):
            raise EditingGrammarError("pattern_requires_multiple_shots", pattern_id)
        if not set(shot_ids).issubset(shots):
            raise EditingGrammarError("pattern_shot_invalid", pattern_id)
        fact_ids = list(map(str, pattern.get("fact_ids") or []))
        if not fact_ids or not set(fact_ids).issubset(facts):
            raise EditingGrammarError("pattern_fact_invalid", pattern_id)
        confidence = pattern.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 < confidence <= 1:
            raise EditingGrammarError("pattern_confidence_invalid", pattern_id)
        start = min(float(shots[shot_id]["interval"][0]) for shot_id in shot_ids)
        end = max(float(shots[shot_id]["interval"][1]) for shot_id in shot_ids)
        pattern["span"] = [start, end]
        if pattern["scope"] != "global" and (end - start) / total > 0.60:
            raise EditingGrammarError("local_pattern_too_broad", pattern_id)
        pattern_sections[pattern_id] = {
            str(shots[shot_id]["section_id"]) for shot_id in shot_ids
        }

    function_patterns: set[str] = set()
    for function in result["functions"]:
        pattern_id = str(function.get("pattern_id") or "")
        if pattern_id not in pattern_ids or pattern_id in function_patterns:
            raise EditingGrammarError("function_pattern_invalid", pattern_id)
        function_patterns.add(pattern_id)
        if function.get("function") not in FUNCTION_TYPES:
            raise EditingGrammarError("function_type_invalid", pattern_id)
        links = function.get("relation_to_story")
        if not isinstance(links, list) or not set(map(str, links)).issubset(relation_ids):
            raise EditingGrammarError("function_story_relation_invalid", pattern_id)
        confidence = function.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 < confidence <= 1:
            raise EditingGrammarError("function_confidence_invalid", pattern_id)
    if function_patterns != pattern_ids:
        raise EditingGrammarError("pattern_function_coverage_invalid")

    for change in _significant_pace_changes(measured):
        right_section = change["to_section"]
        direction = change["direction"]
        allowed = (
            {"rapid_montage", "rhythmic_acceleration"}
            if direction == "accelerates"
            else {"long_hold", "rhythmic_deceleration"}
        )
        covered = any(
            pattern_types[pattern_id] in allowed and
            right_section in pattern_sections[pattern_id]
            for pattern_id in pattern_ids
        )
        if not covered:
            raise EditingGrammarError(
                "pace_change_uncovered",
                f"{change['from_section']}->{right_section}")

    result["measured_editing_sha"] = measured.get("artifact_sha")
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


def analyze_editing_grammar(
        runner: Any, agent_reference: dict[str, Any],
        interpretation: dict[str, Any], *,
        trace_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure deterministically, then make one pattern/function request."""
    measured = build_measured_editing_facts(agent_reference)
    payload = {
        "measured_editing": measured,
        "shot_cards": list(
            (agent_reference.get("shot_storyboard") or {}).get("shot_cards") or []),
        "verified_interpretation": interpretation,
    }
    request = EDITING_GRAMMAR_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "editing_grammar_request.txt").write_text(
            request, encoding="utf-8")
    answer = runner.ask(
        request, max_new_tokens=4096, stop_after_json_object=True)
    raw = str(getattr(answer, "text", answer))
    if trace_dir is not None:
        (trace_dir / "editing_grammar_response.txt").write_text(
            raw, encoding="utf-8")
    value = _parse_object(raw)
    relation_ids = {
        str(row.get("relation_id"))
        for row in interpretation.get("relations") or []
        if row.get("verification_status") == "SUPPORTED"
    }
    return validate_editing_grammar(
        value, measured, relation_ids=relation_ids), measured
