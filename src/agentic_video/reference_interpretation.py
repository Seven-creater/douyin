"""One-pass hierarchical narrative inference for the P0-R2 reference track."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash

INTERPRETATION_VERSION = "reference_narrative_interpretation_v2"
RELATION_AUDIT_VERSION = "reference_narrative_audit_v1"
IDENTITY_PREDICATES = {"same_entity_as", "different_entity_from"}
RELATION_TYPES = {
    "supports", "contradicts", "qualifies", "reframes", "causes",
    "enables", "precedes", "contrasts", "accumulates", "reveals",
    "contextualizes",
}
INTERPRETATION_MAX_NEW_TOKENS = 4096
AUDIT_MAX_NEW_TOKENS = 3072
STOP_AFTER_JSON_OBJECT = True

INTERPRETATION_PROMPT = """You are a hierarchical narrative analyst.
The input already contains atomic evidence organized into temporal section
bundles. Do NOT restate every atomic claim.

Step 1: For each temporal section, compress related events, visual
observations, audio observations, and attributed on-screen statements into a
small number of narrative units. Each section may contribute zero to three
units. A unit may span multiple sections when its cited support does. A
narrative unit should express what a group of evidence contributes at the
section or event level.

Do not create a unit solely to restate identity alignment, one low-level
visual action, or the literal existence of on-screen text unless that fact is
independently important to a cross-unit relation. Identity has already been
resolved into anonymous canonical entities. On-screen text is attributed
content presented by the video: it may be interpreted semantically, but it
must not be upgraded to independently verified physical fact.

Step 2: Record only supported relations among the resulting narrative units.
Relations may be empty. Do not assume a reversal, challenge, qualification,
or any other preset structure. Do not assign required narrative roles. Do not
infer motives, audience reactions, unseen activity, anatomy, or facts absent
from the input.

Allowed relation types:
supports, contradicts, qualifies, reframes, causes, enables, precedes,
contrasts, accumulates, reveals, contextualizes.

Return exactly one JSON object:
{"schema_version":"reference_narrative_interpretation_v2",
"narrative_units":[{"unit_id":"N1","section_ids":["section ID"],
"summary":"...","supported_by":["claim or event ID"]}],
"relations":[{"relation_id":"R1","type":"supports","source":"N1",
"target":"N2","reason":"...","supported_by":["claim or event ID"]}],
"limitations":["..."]}

Every unit support ID must occur in one of its declared sections. Every
relation support ID must exist in the input. A relation may not introduce an
unstated claim in its reason. JSON only. Input:
"""

RELATION_AUDIT_PROMPT = """You are an independent evidence auditor. Audit
every narrative unit, then every relation. Do not rewrite the candidate and do
not judge salience or demand a particular story structure. For each unit,
check whether its cited evidence entails the summary, preserves its temporal
scope, and preserves attribution of on-screen statements. For each relation,
check whether its source and target units are entailed, whether the named
relation holds, and whether unit scope is preserved.

Return exactly one JSON object:
{"schema_version":"reference_narrative_audit_v1",
"unit_checks":[{"unit_id":"N1","entailed":true,
"scope_valid":true,"attribution_preserved":true,"reason_codes":[]}],
"relation_checks":[{"relation_id":"R1","source_entailed":true,
"target_entailed":true,"relation_valid":true,"scope_valid":true,
"reason_codes":[]}],"pass":true}

Include every unit ID and relation ID exactly once. Empty candidate lists are
valid and require the corresponding empty checks. Set pass=true only when
every boolean in every check is true. JSON only. Input:
"""


class ReferenceInterpretationError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _answer_text(answer: Any) -> str:
    return str(getattr(answer, "text", answer))


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
        raise ReferenceInterpretationError("response_json_invalid") from exc
    if not isinstance(value, dict):
        raise ReferenceInterpretationError("response_not_object")
    return value


def _source_ids(agent_reference: dict[str, Any]) -> set[str]:
    return {
        str(row.get("claim_id"))
        for row in agent_reference.get("claims") or []
        if (row.get("claim_id") and
            row.get("predicate") not in IDENTITY_PREDICATES)
    } | {
        str(row.get("event_id"))
        for row in agent_reference.get("events") or []
        if row.get("event_id")
    }


def interpretation_contract_sha() -> str:
    return json_hash({
        "interpretation_version": INTERPRETATION_VERSION,
        "audit_version": RELATION_AUDIT_VERSION,
        "relation_types": sorted(RELATION_TYPES),
        "interpretation_prompt": INTERPRETATION_PROMPT,
        "audit_prompt": RELATION_AUDIT_PROMPT,
    })


def runner_identity(runner: Any) -> dict[str, Any]:
    explicit = getattr(runner, "model_id", None)
    if explicit:
        return {"class": type(runner).__name__, "model_id": str(explicit)}

    cfg = getattr(runner, "cfg", None)
    if not isinstance(cfg, dict):
        cfg = getattr(runner, "omni_cfg", None)
    if isinstance(cfg, dict):
        return {
            "class": type(runner).__name__,
            "model_path": str(cfg.get("model_path") or "unknown"),
            "dtype": str(cfg.get("dtype") or "unknown"),
            "repetition_penalty": cfg.get("repetition_penalty"),
        }
    return {"class": type(runner).__name__, "identity": "unknown"}


def analysis_fingerprint(interpreter_runner: Any, audit_runner: Any) -> str:
    return json_hash({
        "contract_sha": interpretation_contract_sha(),
        "interpreter": runner_identity(interpreter_runner),
        "auditor": runner_identity(audit_runner),
        "generation_contract": {
            "interpretation_max_new_tokens": INTERPRETATION_MAX_NEW_TOKENS,
            "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
            "stop_after_json_object": STOP_AFTER_JSON_OBJECT,
        },
    })


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if start >= 0 and end > start else None


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _entity_root(value: Any) -> str | None:
    text = str(value or "")
    match = re.match(r"^(E(?:_[A-Za-z0-9_]+|[0-9]+))(?:\.|$)", text)
    return match.group(1) if match else None


def _canonical_entities(agent_reference: dict[str, Any]) -> dict[str, str]:
    """Resolve identity infrastructure into anonymous, stable entity aliases."""
    parent: dict[str, str] = {}
    ordered: list[str] = []

    def add(value: Any) -> str | None:
        root = _entity_root(value)
        if root and root not in parent:
            parent[root] = root
            ordered.append(root)
        return root

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for claim in agent_reference.get("claims") or []:
        add(claim.get("subject"))
        add(claim.get("object"))
    for event in agent_reference.get("events") or []:
        for value in list(event.get("participants") or []) + list(
                event.get("object_ids") or []):
            add(value)
    for claim in agent_reference.get("claims") or []:
        if claim.get("predicate") != "same_entity_as":
            continue
        left, right = add(claim.get("subject")), add(claim.get("object"))
        if left and right:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

    component_alias: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for entity in ordered:
        component = find(entity)
        if component not in component_alias:
            component_alias[component] = f"SUBJECT_{len(component_alias) + 1:02d}"
        aliases[entity] = component_alias[component]
    return aliases


def _canonicalize(value: Any, aliases: dict[str, str]) -> Any:
    root = _entity_root(value)
    if not root or root not in aliases:
        return value
    text = str(value)
    return aliases[root] + text[len(root):]


def _project_event(event: dict[str, Any], aliases: dict[str, str],
                   identity_ids: set[str]) -> dict[str, Any]:
    claim_fields = (
        "action_claim_ids", "outcome_claim_ids", "context_claim_ids",
    )
    result = {
        "event_id": event.get("event_id"),
        "participants": [
            _canonicalize(value, aliases)
            for value in event.get("participants") or []
        ],
        "object_ids": [
            _canonicalize(value, aliases)
            for value in event.get("object_ids") or []
        ],
        "ordering": event.get("ordering"),
        "interval": event.get("interval"),
    }
    for field in claim_fields:
        result[field] = [
            str(value) for value in event.get(field) or []
            if str(value) not in identity_ids
        ]
    return result


def build_interpretation_payload(agent_reference: dict[str, Any]) -> dict[str, Any]:
    """Build temporal evidence bundles without narrative or identity claims."""
    timeline = agent_reference.get("deterministic_timeline") or {}
    sections = list(timeline.get("sections") or [])
    segments = list(timeline.get("segments") or [])
    claims = list(agent_reference.get("claims") or [])
    events = list(agent_reference.get("events") or [])
    aliases = _canonical_entities(agent_reference)
    identity_ids = {
        str(row.get("claim_id")) for row in claims
        if row.get("predicate") in IDENTITY_PREDICATES
    }
    bundles: list[dict[str, Any]] = []
    for section in sections:
        section_id = str(section.get("section_id") or "")
        section_interval = _interval(section.get("interval"))
        if not section_id or section_interval is None:
            raise ReferenceInterpretationError(
                "interpretation_section_invalid", section_id)
        visual: list[dict[str, Any]] = []
        text: list[dict[str, Any]] = []
        audio: list[dict[str, Any]] = []
        canonical_entities: set[str] = set()
        for claim in claims:
            if claim.get("predicate") in IDENTITY_PREDICATES:
                continue
            claim_interval = _interval(claim.get("interval"))
            if claim_interval is None or not _overlaps(section_interval, claim_interval):
                continue
            modality = str(claim.get("modality") or "")
            subject = _canonicalize(claim.get("subject"), aliases)
            obj = _canonicalize(claim.get("object"), aliases)
            for value in (subject, obj):
                if str(value).startswith("SUBJECT_"):
                    canonical_entities.add(str(value).split(".", 1)[0])
            if modality == "T":
                text.append({
                    "claim_id": claim.get("claim_id"),
                    "source_type": "on_screen_text",
                    "observed_text": claim.get("object"),
                    "interval": claim.get("interval"),
                })
                continue
            projected = {
                key: value for key, value in {
                    "claim_id": claim.get("claim_id"),
                    "subject": subject,
                    "predicate": claim.get("predicate"),
                    "object": obj,
                    "interval": claim.get("interval"),
                    "polarity": claim.get("polarity"),
                    "visibility": claim.get("visibility"),
                }.items() if value is not None
            }
            if modality == "V":
                visual.append(projected)
            elif modality == "A":
                audio.append(projected)
            else:
                raise ReferenceInterpretationError(
                    "interpretation_modality_invalid",
                    str(claim.get("claim_id") or ""))

        section_events = []
        for event in events:
            event_interval = _interval(event.get("interval"))
            if event_interval is None or not _overlaps(section_interval, event_interval):
                continue
            projected = _project_event(event, aliases, identity_ids)
            section_events.append(projected)
            for value in projected["participants"] + projected["object_ids"]:
                if str(value).startswith("SUBJECT_"):
                    canonical_entities.add(str(value).split(".", 1)[0])

        bundles.append({
            "section_id": section_id,
            "interval": section.get("interval"),
            "shot_ids": [
                str(row.get("segment_id")) for row in segments
                if (str(row.get("section_id") or "") == section_id and
                    row.get("segment_kind") == "content")
            ],
            "canonical_entities": sorted(canonical_entities),
            "events": section_events,
            "visual_observations": visual,
            "text_statements": text,
            "audio_observations": audio,
        })
    return {
        "reference_sha": agent_reference.get("artifact_sha"),
        "section_bundles": bundles,
        "unresolved_limitations": list(
            agent_reference.get("unresolved_limitations") or []),
    }


def validate_interpretation(value: dict[str, Any],
                            agent_reference: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != INTERPRETATION_VERSION:
        raise ReferenceInterpretationError("interpretation_schema_invalid")
    narrative_units = value.get("narrative_units")
    relations = value.get("relations")
    if not isinstance(narrative_units, list):
        raise ReferenceInterpretationError("narrative_units_invalid")
    if not isinstance(relations, list):
        raise ReferenceInterpretationError("relations_invalid")
    if not isinstance(value.get("limitations"), list):
        raise ReferenceInterpretationError("limitations_invalid")

    allowed_sources = _source_ids(agent_reference)
    payload = build_interpretation_payload(agent_reference)
    section_sources: dict[str, set[str]] = {}
    for section in payload["section_bundles"]:
        source_ids = {
            str(row.get("claim_id"))
            for field in (
                "visual_observations", "text_statements", "audio_observations",
            )
            for row in section.get(field) or [] if row.get("claim_id")
        } | {
            str(row.get("event_id")) for row in section.get("events") or []
            if row.get("event_id")
        }
        section_sources[str(section["section_id"])] = source_ids

    unit_ids: set[str] = set()
    section_unit_counts = {section_id: 0 for section_id in section_sources}
    for unit in narrative_units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id or unit_id in unit_ids:
            raise ReferenceInterpretationError(
                "narrative_unit_id_invalid", unit_id)
        unit_ids.add(unit_id)
        if not str(unit.get("summary") or "").strip():
            raise ReferenceInterpretationError(
                "narrative_unit_summary_missing", unit_id)
        section_ids = unit.get("section_ids")
        if (not isinstance(section_ids, list) or not section_ids or
                len(set(map(str, section_ids))) != len(section_ids) or
                not set(map(str, section_ids)).issubset(section_sources)):
            raise ReferenceInterpretationError(
                "narrative_unit_section_invalid", unit_id)
        for section_id in map(str, section_ids):
            section_unit_counts[section_id] += 1
            if section_unit_counts[section_id] > 3:
                raise ReferenceInterpretationError(
                    "section_narrative_unit_limit_exceeded", section_id)
        support = unit.get("supported_by")
        if not isinstance(support, list) or not support:
            raise ReferenceInterpretationError(
                "narrative_unit_support_missing", unit_id)
        if not set(map(str, support)).issubset(allowed_sources):
            raise ReferenceInterpretationError(
                "narrative_unit_support_invalid", unit_id)
        local_sources = set().union(*(
            section_sources[str(section_id)] for section_id in section_ids))
        if not set(map(str, support)).issubset(local_sources):
            raise ReferenceInterpretationError(
                "narrative_unit_support_out_of_scope", unit_id)

    relation_ids: set[str] = set()
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        if not relation_id or relation_id in relation_ids:
            raise ReferenceInterpretationError("relation_id_invalid", relation_id)
        relation_ids.add(relation_id)
        if relation.get("type") not in RELATION_TYPES:
            raise ReferenceInterpretationError("relation_type_invalid", relation_id)
        if str(relation.get("source") or "") not in unit_ids:
            raise ReferenceInterpretationError("relation_source_invalid", relation_id)
        if str(relation.get("target") or "") not in unit_ids:
            raise ReferenceInterpretationError("relation_target_invalid", relation_id)
        if relation.get("source") == relation.get("target"):
            raise ReferenceInterpretationError("relation_self_link_invalid", relation_id)
        if not str(relation.get("reason") or "").strip():
            raise ReferenceInterpretationError("relation_reason_missing", relation_id)
        support = relation.get("supported_by")
        if not isinstance(support, list) or not support:
            raise ReferenceInterpretationError("relation_support_missing", relation_id)
        if not set(map(str, support)).issubset(allowed_sources):
            raise ReferenceInterpretationError("relation_support_invalid", relation_id)
    return value


def validate_relation_audit(audit: dict[str, Any],
                            interpretation: dict[str, Any]) -> dict[str, Any]:
    if audit.get("schema_version") != RELATION_AUDIT_VERSION:
        raise ReferenceInterpretationError("relation_audit_schema_invalid")
    unit_checks = audit.get("unit_checks")
    relation_checks = audit.get("relation_checks")
    if (not isinstance(unit_checks, list) or
            not isinstance(relation_checks, list) or
            not isinstance(audit.get("pass"), bool)):
        raise ReferenceInterpretationError("relation_audit_shape_invalid")
    expected_units = {
        str(row["unit_id"])
        for row in interpretation.get("narrative_units") or []
    }
    expected_relations = {
        str(row["relation_id"]) for row in interpretation.get("relations") or []
    }
    actual_units: set[str] = set()
    actual_relations: set[str] = set()
    all_valid = True
    for check in unit_checks:
        unit_id = str(check.get("unit_id") or "")
        if not unit_id or unit_id in actual_units:
            raise ReferenceInterpretationError(
                "narrative_unit_audit_id_invalid", unit_id)
        actual_units.add(unit_id)
        for key in ("entailed", "scope_valid", "attribution_preserved"):
            if not isinstance(check.get(key), bool):
                raise ReferenceInterpretationError(
                    "narrative_unit_audit_boolean_invalid",
                    f"{unit_id}:{key}")
            all_valid = all_valid and bool(check[key])
        if not isinstance(check.get("reason_codes"), list):
            raise ReferenceInterpretationError(
                "narrative_unit_audit_reason_codes_invalid", unit_id)
    for check in relation_checks:
        relation_id = str(check.get("relation_id") or "")
        if not relation_id or relation_id in actual_relations:
            raise ReferenceInterpretationError(
                "relation_audit_id_invalid", relation_id)
        actual_relations.add(relation_id)
        for key in (
            "source_entailed", "target_entailed", "relation_valid", "scope_valid",
        ):
            if not isinstance(check.get(key), bool):
                raise ReferenceInterpretationError(
                    "relation_audit_boolean_invalid", f"{relation_id}:{key}")
            all_valid = all_valid and bool(check[key])
        if not isinstance(check.get("reason_codes"), list):
            raise ReferenceInterpretationError(
                "relation_audit_reason_codes_invalid", relation_id)
    if actual_units != expected_units:
        raise ReferenceInterpretationError("narrative_unit_audit_coverage_invalid")
    if actual_relations != expected_relations:
        raise ReferenceInterpretationError("relation_audit_coverage_invalid")
    if bool(audit["pass"]) != all_valid:
        raise ReferenceInterpretationError("relation_audit_pass_invalid")
    return audit


def finalize_interpretation(interpretation: dict[str, Any],
                            audit: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(interpretation)
    unit_checks = {
        str(row["unit_id"]): row
        for row in audit.get("unit_checks") or []
    }
    relation_checks = {
        str(row["relation_id"]): row
        for row in audit.get("relation_checks") or []
    }
    for unit in result.get("narrative_units") or []:
        check = unit_checks[str(unit["unit_id"])]
        unit["verification_status"] = (
            "SUPPORTED" if all(bool(check[key]) for key in (
                "entailed", "scope_valid", "attribution_preserved",
            ))
            else "UNRESOLVED")
    for relation in result.get("relations") or []:
        check = relation_checks[str(relation["relation_id"])]
        relation["verification_status"] = (
            "SUPPORTED" if all(bool(check[key]) for key in (
                "source_entailed", "target_entailed", "relation_valid", "scope_valid",
            )) else "UNRESOLVED"
        )
    result["verification_status"] = (
        "SUPPORTED" if audit.get("pass") else "UNRESOLVED")
    result["relation_audit_sha"] = json_hash(audit)
    result["artifact_sha"] = json_hash(result)
    return result


def _write_trace(trace_dir: Path | None, name: str, content: str) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / name).write_text(content, encoding="utf-8")


def run_reference_interpretation(
        interpreter_runner: Any, audit_runner: Any,
        agent_reference: dict[str, Any], *,
    trace_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate once, audit once, and never retry unchanged evidence."""
    fingerprint = analysis_fingerprint(interpreter_runner, audit_runner)
    payload = build_interpretation_payload(agent_reference)
    request = INTERPRETATION_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    _write_trace(trace_dir, "interpretation_request.txt", request)
    answer = interpreter_runner.ask(
        request, max_new_tokens=INTERPRETATION_MAX_NEW_TOKENS,
        stop_after_json_object=STOP_AFTER_JSON_OBJECT)
    raw = _answer_text(answer)
    _write_trace(trace_dir, "interpretation_response.txt", raw)
    interpretation = _parse_object(raw)
    validate_interpretation(interpretation, agent_reference)

    audit_payload = {
        "evidence": payload,
        "candidate_interpretation": interpretation,
    }
    audit_request = RELATION_AUDIT_PROMPT + json.dumps(
        audit_payload, ensure_ascii=False, separators=(",", ":"))
    _write_trace(trace_dir, "relation_audit_request.txt", audit_request)
    audit_answer = audit_runner.ask(
        audit_request, max_new_tokens=AUDIT_MAX_NEW_TOKENS,
        stop_after_json_object=STOP_AFTER_JSON_OBJECT)
    audit_raw = _answer_text(audit_answer)
    _write_trace(trace_dir, "relation_audit_response.txt", audit_raw)
    audit = _parse_object(audit_raw)
    validate_relation_audit(audit, interpretation)
    result = finalize_interpretation(interpretation, audit)
    result["agent_reference_sha"] = agent_reference.get("artifact_sha")
    result["analysis_fingerprint"] = fingerprint
    result.pop("artifact_sha", None)
    result["artifact_sha"] = json_hash(result)
    audit["agent_reference_sha"] = agent_reference.get("artifact_sha")
    audit["analysis_fingerprint"] = fingerprint
    audit["interpretation_sha"] = result["artifact_sha"]
    audit["artifact_sha"] = json_hash(audit)
    return result, audit


def run_reference_interpretation_checkpointed(
        interpreter_runner: Any, audit_runner: Any,
        agent_reference: dict[str, Any], *, checkpoint_dir: Path,
        trace_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse output only while evidence and the analysis contract match."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    interpretation_path = checkpoint_dir / "reference_interpretation.json"
    audit_path = checkpoint_dir / "reference_relation_audit.json"
    existing = (interpretation_path.exists(), audit_path.exists())
    if any(existing) and not all(existing):
        raise ReferenceInterpretationError("interpretation_checkpoint_incomplete")
    if all(existing):
        try:
            interpretation = json.loads(
                interpretation_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceInterpretationError(
                "interpretation_checkpoint_invalid") from exc
        evidence_sha = agent_reference.get("artifact_sha")
        fingerprint = analysis_fingerprint(interpreter_runner, audit_runner)
        if (interpretation.get("agent_reference_sha") == evidence_sha and
                audit.get("agent_reference_sha") == evidence_sha and
                interpretation.get("analysis_fingerprint") == fingerprint and
                audit.get("analysis_fingerprint") == fingerprint):
            validate_interpretation(interpretation, agent_reference)
            validate_relation_audit(audit, interpretation)
            return interpretation, audit

    interpretation, audit = run_reference_interpretation(
        interpreter_runner, audit_runner, agent_reference, trace_dir=trace_dir)
    interpretation_path.write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return interpretation, audit
