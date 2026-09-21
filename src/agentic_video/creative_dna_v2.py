"""Independent interpretation/edit analysis and relation-first Creative DNA v2."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.manifest import json_hash

INTERPRETATION_VERSION = "reference_interpretation_v1"
EDITING_ANALYSIS_VERSION = "editing_analysis_v1"
DNA_AUDIT_VERSION = "creative_dna_audit_v2"
DNA_PUBLISH_VERSION = "creative_dna_v2"

INTERPRETATION_PROMPT = """You are an evidence-grounded interpretation analyst.
Read only the supplied validated claims and events. Return one JSON object:
{"relations":[{"relation_id":"IR1","type":"claim_to_inference|evidence_to_belief_revision|social_interpretation","source_ids":["..."],"target":"abstract proposition","scope":"specific|general","confidence":0.0}],"limitations":["..."]}
Every relation must cite claim or event IDs. Separate a domain-specific judgment
from an overgeneralized judgment. Do not invent motives, unseen training, or
facts absent from the input. JSON only. Input:
"""

EDITING_ANALYSIS_PROMPT = """You are an editing-function analyst. Deterministic
timing measurements have already been calculated and are immutable. Using only
the validated events and those measurements, return one JSON object:
{"functional_relations":[{"relation_id":"ER1","timeline_ids":["..."],"information_function":"...","ordering_effect":"...","confidence":0.0}],"limitations":["..."]}
Do not recalculate shot counts or durations. Do not assume a fixed number of
narrative functions and do not assign motives or audience beliefs without an
evidence reference. JSON only. Input:
"""

DIRECTOR_DNA_PROMPT = """You are a structure-abduction director. Infer only
cross-domain relational structure supported by the three supplied, versioned
artifacts. Return one JSON object:
{"variables":[{"variable_id":"V1","kind":"agent|observer|proposition|evidence|state","description":"abstract role"}],
"relations":[{"relation_id":"R1","type":"logical|causal|temporal","source":"V1","target":"V2","condition":"..."}],
"constraints":[{"constraint_id":"C1","rule":"...","required":true}],
"free_slots":[{"slot_id":"S1","kind":"domain|relationship|inference_source|counterevidence_form|resolution_tone","constraint_ids":["C1"]}],
"editing_relations":[{"relation_id":"E1","editing_operation":"abstract organization","information_effect":"...","conditions":["..."]}],
"supported_by":{"R1":["claim/event/interpretation/edit relation ids"]},
"concrete_bindings":[{"variable_id":"V1","source_ids":["..."]}],
"anti_invariants":["surface features that must not be transferred"],
"confidence":0.0}
The relation set must follow the evidence; do not force a stock reversal pattern.
A counterexample must negate the actual proposition it challenges. Evidence in
one domain cannot establish universal ability without additional support. Use
abstract roles, never source wording, media paths, named activities, or identity
descriptions. JSON only. Input:
"""


class DNAV2Error(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _answer_text(answer: Any) -> str:
    return str(getattr(answer, "text", answer))


def _parse_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DNAV2Error("response_json_invalid") from exc
    if not isinstance(value, dict):
        raise DNAV2Error("response_not_object")
    return value


def _source_ids(validated_reference: dict[str, Any]) -> set[str]:
    return ({str(row.get("claim_id"))
             for row in validated_reference.get("accepted_claims") or []} |
            {str(row.get("event_id"))
             for row in validated_reference.get("accepted_events") or []})


def validate_interpretation(value: dict[str, Any],
                            validated_reference: dict[str, Any]) -> None:
    valid = _source_ids(validated_reference)
    relations = value.get("relations")
    if not isinstance(relations, list) or not relations:
        raise DNAV2Error("interpretation_relations_missing")
    for row in relations:
        refs = set(map(str, row.get("source_ids") or []))
        if not refs or not refs.issubset(valid):
            raise DNAV2Error("interpretation_evidence_invalid")
        if row.get("scope") not in {"specific", "general"}:
            raise DNAV2Error("interpretation_scope_invalid")


def compute_timeline_measurements(validated_reference: dict[str, Any]) -> dict[str, Any]:
    timeline = validated_reference.get("deterministic_timeline") or {}
    segments = timeline.get("segments") or []
    measurements = []
    for row in segments:
        interval = row.get("interval") or []
        if len(interval) != 2:
            raise DNAV2Error("timeline_interval_invalid")
        measurements.append({
            "timeline_id": row.get("segment_id"),
            "interval": [float(interval[0]), float(interval[1])],
            "duration_s": round(float(interval[1]) - float(interval[0]), 6),
            "segment_kind": row.get("segment_kind"),
            "section_id": row.get("section_id"),
        })
    sections: dict[str, dict[str, Any]] = {}
    for row in measurements:
        group = sections.setdefault(str(row.get("section_id")),
                                    {"segment_count": 0, "duration_s": 0.0})
        group["segment_count"] += 1
        group["duration_s"] += row["duration_s"]
    for group in sections.values():
        group["duration_s"] = round(group["duration_s"], 6)
        group["density_per_s"] = round(
            group["segment_count"] / max(group["duration_s"], 0.000001), 6)
    return {"segments": measurements, "sections": sections,
            "measurement_owner": "deterministic_program"}


def validate_editing_analysis(value: dict[str, Any],
                              measurements: dict[str, Any]) -> None:
    valid_ids = {str(row.get("timeline_id"))
                 for row in measurements.get("segments") or []}
    relations = value.get("functional_relations")
    if not isinstance(relations, list) or not relations:
        raise DNAV2Error("editing_relations_missing")
    for row in relations:
        refs = set(map(str, row.get("timeline_ids") or []))
        if not refs or not refs.issubset(valid_ids):
            raise DNAV2Error("editing_timeline_reference_invalid")


def run_independent_analyses(validated_reference: dict[str, Any], *,
                             runner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use two clean text calls; neither draft appears in the other request."""
    base = json.dumps(validated_reference, ensure_ascii=False,
                      separators=(",", ":"))
    interpretation_answer = runner.ask(
        INTERPRETATION_PROMPT + base, max_new_tokens=2048,
        stop_after_json_object=True)
    interpretation = _parse_object(_answer_text(interpretation_answer))
    validate_interpretation(interpretation, validated_reference)
    interpretation = {
        "schema_version": INTERPRETATION_VERSION,
        "validated_reference_sha": validated_reference.get("artifact_sha"),
        **interpretation,
    }
    interpretation["artifact_sha"] = json_hash(interpretation)

    measurements = compute_timeline_measurements(validated_reference)
    edit_input = {"validated_reference": validated_reference,
                  "deterministic_measurements": measurements}
    edit_answer = runner.ask(
        EDITING_ANALYSIS_PROMPT + json.dumps(
            edit_input, ensure_ascii=False, separators=(",", ":")),
        max_new_tokens=2048, stop_after_json_object=True)
    proposed = _parse_object(_answer_text(edit_answer))
    validate_editing_analysis(proposed, measurements)
    editing = {
        "schema_version": EDITING_ANALYSIS_VERSION,
        "validated_reference_sha": validated_reference.get("artifact_sha"),
        "deterministic_measurements": measurements,
        **proposed,
    }
    editing["artifact_sha"] = json_hash(editing)
    return interpretation, editing


def _ids(rows: Iterable[dict[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if str(row.get(key) or "")}


def validate_dna_audit(value: dict[str, Any]) -> None:
    variables = value.get("variables") or []
    relations = value.get("relations") or []
    constraints = value.get("constraints") or []
    slots = value.get("free_slots") or []
    edits = value.get("editing_relations") or []
    if not all(isinstance(rows, list) and rows
               for rows in (variables, relations, constraints, slots, edits)):
        raise DNAV2Error("dna_required_collection_missing")
    variable_ids = _ids(variables, "variable_id")
    relation_ids = _ids(relations, "relation_id")
    constraint_ids = _ids(constraints, "constraint_id")
    if len(variable_ids) != len(variables) or len(relation_ids) != len(relations):
        raise DNAV2Error("dna_identifier_missing_or_duplicate")
    for relation in relations:
        if str(relation.get("source")) not in variable_ids or \
                str(relation.get("target")) not in variable_ids:
            raise DNAV2Error("dna_relation_variable_missing")
        if relation.get("type") not in {"logical", "causal", "temporal"}:
            raise DNAV2Error("dna_relation_type_invalid")
    for slot in slots:
        if not set(map(str, slot.get("constraint_ids") or [])).issubset(constraint_ids):
            raise DNAV2Error("dna_slot_constraint_missing")
    supported = value.get("supported_by") or {}
    if not relation_ids.issubset(set(map(str, supported))):
        raise DNAV2Error("dna_relation_support_missing")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise DNAV2Error("dna_confidence_invalid")


def _sensitive_strings(validated_reference: dict[str, Any]) -> set[str]:
    values: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"object", "path", "reference_specific_fact", "text"}:
                    text = str(value).strip()
                    if len(text) >= 6:
                        values.add(text.casefold())
                walk(value)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(validated_reference)
    return values


_PUBLISH_KEYS = ("variables", "relations", "constraints", "free_slots",
                 "editing_relations")


def publish_dna(audit: dict[str, Any], *,
                validated_reference: dict[str, Any]) -> dict[str, Any]:
    """Whitelist construction: audit bindings and support maps cannot escape."""
    validate_dna_audit(audit)
    publish = {"schema_version": DNA_PUBLISH_VERSION,
               **{key: audit[key] for key in _PUBLISH_KEYS}}
    serialized = json.dumps(publish, ensure_ascii=False).casefold()
    if re.search(r"(?:[a-z]:\\|/data/|/home/|media_path|source_sha)", serialized):
        raise DNAV2Error("dna_publish_path_or_provenance_leak")
    if any(value and value in serialized
           for value in _sensitive_strings(validated_reference)):
        raise DNAV2Error("dna_publish_reference_binding_leak")
    publish["artifact_sha"] = json_hash(publish)
    return publish


def run_director(validated_reference: dict[str, Any], interpretation: dict[str, Any],
                 editing_analysis: dict[str, Any], *, runner: Any
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    if interpretation.get("validated_reference_sha") != \
            validated_reference.get("artifact_sha"):
        raise DNAV2Error("interpretation_parent_mismatch")
    if editing_analysis.get("validated_reference_sha") != \
            validated_reference.get("artifact_sha"):
        raise DNAV2Error("editing_parent_mismatch")
    payload = {"validated_reference": validated_reference,
               "interpretation": interpretation,
               "editing_analysis": editing_analysis}
    answer = runner.ask(DIRECTOR_DNA_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")),
        max_new_tokens=3072, stop_after_json_object=True)
    proposed = _parse_object(_answer_text(answer))
    validate_dna_audit(proposed)
    audit = {
        "schema_version": DNA_AUDIT_VERSION,
        "parent_shas": [validated_reference.get("artifact_sha"),
                        interpretation.get("artifact_sha"),
                        editing_analysis.get("artifact_sha")],
        **proposed,
    }
    audit["artifact_sha"] = json_hash(audit)
    publish = publish_dna(audit, validated_reference=validated_reference)
    return audit, publish


def build_writer_payload(dna_publish: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the release interface; no annex or blocklist is exposed."""
    unexpected = set(dna_publish) - ({"schema_version", "artifact_sha"} |
                                     set(_PUBLISH_KEYS))
    if unexpected or dna_publish.get("schema_version") != DNA_PUBLISH_VERSION:
        raise DNAV2Error("writer_payload_not_published_dna")
    return {key: dna_publish[key] for key in ("schema_version", *_PUBLISH_KEYS)}


def write_dna_artifacts(output_dir: Path, audit: dict[str, Any],
                        publish: dict[str, Any], *, release_status: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "creative_dna_audit_v2.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    envelope = {"release_status": release_status, "dna": publish}
    (output_dir / "creative_dna_v2.json").write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

