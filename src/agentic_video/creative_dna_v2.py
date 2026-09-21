"""Independent interpretation/edit analysis and relation-first Creative DNA v2."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from src.agentic_video.manifest import json_hash

INTERPRETATION_VERSION = "reference_interpretation_v2"
EDITING_ANALYSIS_VERSION = "editing_analysis_v2"
DNA_AUDIT_VERSION = "creative_dna_audit_v2"
DNA_PUBLISH_VERSION = "creative_dna_v2"

INTERPRETATION_PROMPT = """You are an evidence-grounded interpretation analyst.
Read only the supplied accepted claims, accepted events, timeline, and coverage.
First separate what on-screen text ASSERTS from what visual evidence OBSERVES.
Then infer only evidence-linked changes in proposition scope. A text assertion is
not a visual fact. A counterexample can contradict only the proposition or
implication it actually bears on; one domain never proves universal ability.
Do not create one proposition per claim and do not merely restate that a text
string or action exists. Produce 3-10 aggregated propositions. Identity claims
are support for cross-segment alignment, not standalone story propositions.
Represent the strongest opening assertion, the evidence that bears on it, and
any closing scope limit when they are supported. If no proposition revision is
supported, use context and ordering roles rather than inventing one.
Interpret the semantic content of a complete text proposition; do not reduce it
to "text appears." When the accepted evidence contains a broad negative
assertion and later bounded competence evidence about an identity-aligned
subject, label the latter as counterevidence and use a contradicts relation.
When a final limitation narrows what the evidence establishes, label it as a
scope limit and use a qualifies relation. These are conditional definitions,
not permission to infer a pattern absent from the evidence.
An optional analysis_contract lists per-reference review questions as required
roles/relation types. Satisfy it only with cited evidence; never invent a
relation to make the contract pass.

Return exactly one JSON object:
{"propositions":[{"proposition_id":"P1","statement":"abstract proposition",
"source_ids":["claim/event id"],"epistemic_role":"initial_assertion|counterevidence|scope_limit|context",
"scope":"specific|domain_bounded|general"}],
"relations":[{"relation_id":"IR1","type":"supports|contradicts|qualifies|reframes|orders",
"source_proposition_ids":["P2"],"target_proposition_id":"P1",
"source_ids":["claim/event id"],"interpretation":"abstract relational claim",
"confidence":0.0}],"limitations":["..."]}

Every proposition and relation must cite supplied claim/event IDs. Use an
identity-alignment claim before treating anonymous entities from different
segments as one subject. Do not invent motives, audience reactions, unseen
training, or facts absent from the input. JSON only. Input:
"""

EDITING_ANALYSIS_PROMPT = """You are an editing-function analyst. Deterministic
timing measurements are immutable. Separate measured order/density/duration
from a proposed information function. Analyze changes in information state,
not the source topic or literal action.

Return exactly one JSON object:
{"functional_relations":[{"relation_id":"ER1","timeline_ids":["..."],
"event_ids":["..."],"editing_operation":"establish|demonstrate|accumulate|qualify|contrast|bridge",
"information_before":"abstract state","information_after":"abstract state",
"ordering_effect":"abstract effect","confidence":0.0}],"limitations":["..."]}

Do not recalculate measurements. Do not force one function per section or a
fixed number of functions. A relation with zero confidence must be omitted.
Every section containing an accepted event must appear in at least one
functional relation; one relation may span multiple sections. Do not assign
motives or audience beliefs. JSON only. Input:
"""

DIRECTOR_DNA_PROMPT = """You are a structure-abduction director. Infer only
cross-domain relational structure supported by the three supplied, versioned
artifacts. Structure means proposition, evidence, revision, scope, and
information-order relations. It does NOT mean a literal inventory of source
actions, body properties, activities, places, objects, identities, or captions.

Return exactly one JSON object:
{"variables":[{"variable_id":"V1","kind":"observer|subject|proposition|evidence|information_state|scope_boundary","description":"cross-domain relational role"}],
"relations":[{"relation_id":"R1","type":"logical|causal|temporal",
"mechanism":"holds_belief|supports_proposition|contradicts_proposition|triggers_revision|qualifies_scope|accumulates_evidence|orders_disclosure|reframes_context|precedes",
"source":"V1","target":"V2","condition":"abstract applicability condition"}],
"constraints":[{"constraint_id":"C1","rule_type":"exact_proposition_match|evidence_required|scope_bound|revision_required|ordering_required|qualification_preserved","rule":"cross-domain rule","required":true}],
"free_slots":[{"slot_id":"S1","kind":"domain|relationship|inference_source|counterevidence_form|resolution_tone","constraint_ids":["C1"]}],
"editing_relations":[{"relation_id":"E1","editing_operation":"establish|demonstrate|accumulate|qualify|contrast|bridge","information_effect":"abstract information-state change","conditions":["..."]}],
"supported_by":{"R1":["claim id, event id, interpretation proposition/relation id, or editing relation id"]},
"concrete_bindings":[{"variable_id":"V1","source_ids":["..."]}],
"anti_invariants":["surface features that must not be transferred"],
"validation_record":{"structure_only":true,"source_bindings_confined_to_audit":true,"provenance_rules_not_creative":true},
"confidence":0.0}
The relation set must follow the interpretation; do not force a stock reversal
pattern when the cited evidence does not support one. If the interpretation
does contain contradiction, revision, or qualification, preserve that mechanism
rather than replacing it with literal action causality. A counterexample must
negate the actual proposition it challenges. Evidence in one domain cannot
establish universal ability without additional support. Source-modality and
provenance rules stay in the audit system; they are not creative invariants for
a new story. Use abstract roles only. Never copy source wording, media paths,
named activities, physical action forms, body properties, or identity
descriptions into variables, relations, constraints, free slots, or editing
relations. Concrete details are allowed only in concrete_bindings and
anti_invariants, which are removed before publishing. JSON only. Input:
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


def _ask_validated_object(*, runner: Any, prompt: str, max_new_tokens: int,
                          validator: Callable[[dict[str, Any]], None],
                          attempts: int = 2, trace_dir: Path | None = None,
                          trace_name: str = "analysis") -> dict[str, Any]:
    """Retry only with a reason code; never reflect model/source text."""
    last_error: DNAV2Error | None = None
    trace_path = Path(trace_dir) if trace_dir is not None else None
    if trace_path is not None:
        trace_path.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        suffix = ""
        if last_error is not None:
            guidance = {
                "interpretation_required_role_missing": (
                    "Do not label every later fact as an initial assertion. "
                    "Semantically test later evidence against the earlier "
                    "proposition and assign counterevidence or scope_limit when "
                    "the cited evidence supports those roles."),
                "interpretation_required_relation_missing": (
                    "Use contradicts only for evidence bearing on the exact "
                    "target proposition, and qualifies only for a real scope "
                    "limit. Do not substitute supports for these mechanisms."),
                "interpretation_not_aggregated": (
                    "Aggregate claims into the smallest cross-segment "
                    "propositions allowed by the declared limit."),
            }.get(last_error.reason_code, "Follow the declared schema exactly.")
            suffix = ("\nCORRECTION: the previous object failed gate "
                      f"{last_error.reason_code}. {guidance} Return a complete fresh JSON "
                      "object following the original schema. Do not quote the "
                      "previous response.\n")
        answer = runner.ask(prompt + suffix, max_new_tokens=max_new_tokens,
                            stop_after_json_object=True)
        raw = _answer_text(answer)
        if trace_path is not None:
            (trace_path / f"{trace_name}_attempt_{attempt:03d}.txt").write_text(
                raw, encoding="utf-8")
        try:
            value = _parse_object(raw)
            validator(value)
            if trace_path is not None:
                (trace_path / f"{trace_name}_attempt_{attempt:03d}_gate.json").write_text(
                    json.dumps({"passed": True}, indent=2), encoding="utf-8")
            return value
        except DNAV2Error as exc:
            last_error = exc
            if trace_path is not None:
                (trace_path / f"{trace_name}_attempt_{attempt:03d}_gate.json").write_text(
                    json.dumps({"passed": False,
                                "reason_code": exc.reason_code}, indent=2),
                    encoding="utf-8")
    assert last_error is not None
    raise last_error


def _source_ids(validated_reference: dict[str, Any]) -> set[str]:
    return ({str(row.get("claim_id"))
             for row in validated_reference.get("accepted_claims") or []} |
            {str(row.get("event_id"))
             for row in validated_reference.get("accepted_events") or []})


def _source_modalities(validated_reference: dict[str, Any]) -> dict[str, list[str]]:
    """Derive modality provenance; never trust a model to relabel it."""
    claim_modalities = {
        str(row.get("claim_id")): str(row.get("modality"))
        for row in validated_reference.get("accepted_claims") or []}
    result = {key: [value] for key, value in claim_modalities.items()}
    for event in validated_reference.get("accepted_events") or []:
        refs = (list(event.get("action_claim_ids") or []) +
                list(event.get("outcome_claim_ids") or []) +
                list(event.get("context_claim_ids") or []))
        result[str(event.get("event_id"))] = sorted({
            claim_modalities[str(ref)] for ref in refs
            if str(ref) in claim_modalities
        })
    return result


def validate_interpretation(value: dict[str, Any],
                            validated_reference: dict[str, Any],
                            analysis_contract: dict[str, Any] | None = None) -> None:
    contract = analysis_contract or {}
    allowed_roles = {"initial_assertion", "counterevidence", "scope_limit",
                     "context"}
    allowed_relation_types = {"supports", "contradicts", "qualifies",
                              "reframes", "orders"}
    if not set(map(str, contract.get("required_roles") or [])).issubset(
            allowed_roles) or not set(map(str, contract.get(
                "required_relation_types") or [])).issubset(
                    allowed_relation_types):
        raise DNAV2Error("interpretation_contract_invalid")
    valid = _source_ids(validated_reference)
    propositions = value.get("propositions")
    relations = value.get("relations")
    if not isinstance(propositions, list) or not propositions:
        raise DNAV2Error("interpretation_propositions_missing")
    maximum = int(contract.get("max_propositions", 10))
    if not 3 <= len(propositions) <= maximum:
        raise DNAV2Error("interpretation_not_aggregated")
    if not isinstance(relations, list) or not relations:
        raise DNAV2Error("interpretation_relations_missing")
    proposition_ids = _ids(propositions, "proposition_id")
    if len(proposition_ids) != len(propositions):
        raise DNAV2Error("interpretation_proposition_id_invalid")
    for row in propositions:
        refs = set(map(str, row.get("source_ids") or []))
        if not refs or not refs.issubset(valid):
            raise DNAV2Error("interpretation_proposition_evidence_invalid")
        if row.get("epistemic_role") not in allowed_roles:
            raise DNAV2Error("interpretation_role_invalid")
        if row.get("scope") not in {"specific", "domain_bounded", "general"}:
            raise DNAV2Error("interpretation_scope_invalid")
    roles = {str(row.get("epistemic_role")) for row in propositions}
    if len(roles) < 2:
        raise DNAV2Error("interpretation_roles_collapsed")
    if not set(map(str, contract.get("required_roles") or [])).issubset(roles):
        raise DNAV2Error("interpretation_required_role_missing")
    for row in relations:
        refs = set(map(str, row.get("source_ids") or []))
        if not refs or not refs.issubset(valid):
            raise DNAV2Error("interpretation_evidence_invalid")
        if row.get("type") not in allowed_relation_types:
            raise DNAV2Error("interpretation_relation_type_invalid")
        source_props = set(map(str, row.get("source_proposition_ids") or []))
        if not source_props or not source_props.issubset(proposition_ids):
            raise DNAV2Error("interpretation_relation_source_invalid")
        if str(row.get("target_proposition_id")) not in proposition_ids:
            raise DNAV2Error("interpretation_relation_target_invalid")
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 < float(confidence) <= 1:
            raise DNAV2Error("interpretation_confidence_invalid")
    relation_types = {str(row.get("type")) for row in relations}
    if not set(map(str, contract.get(
            "required_relation_types") or [])).issubset(relation_types):
        raise DNAV2Error("interpretation_required_relation_missing")


def _attach_interpretation_provenance(
        value: dict[str, Any], validated_reference: dict[str, Any]) -> None:
    modalities = _source_modalities(validated_reference)
    for row in value.get("propositions") or []:
        row["source_modalities"] = sorted({
            modality for ref in row.get("source_ids") or []
            for modality in modalities.get(str(ref), [])
        })
    for row in value.get("relations") or []:
        row["source_modalities"] = sorted({
            modality for ref in row.get("source_ids") or []
            for modality in modalities.get(str(ref), [])
        })


def _analysis_view(validated_reference: dict[str, Any]) -> dict[str, Any]:
    """Keep accepted evidence intact while excluding failed-draft diagnostics."""
    limitation_counts: dict[str, int] = {}
    for row in validated_reference.get("unresolved_limitations") or []:
        status = str(row.get("status") or "UNKNOWN")
        limitation_counts[status] = limitation_counts.get(status, 0) + 1
    return {
        "schema_version": validated_reference.get("schema_version"),
        "artifact_sha": validated_reference.get("artifact_sha"),
        "accepted_claims": validated_reference.get("accepted_claims") or [],
        "accepted_events": validated_reference.get("accepted_events") or [],
        "deterministic_timeline": validated_reference.get(
            "deterministic_timeline") or {},
        "coverage": validated_reference.get("coverage") or {},
        "unresolved_limitation_counts": limitation_counts,
    }


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
                              measurements: dict[str, Any],
                              validated_reference: dict[str, Any]) -> None:
    valid_ids = {str(row.get("timeline_id"))
                 for row in measurements.get("segments") or []}
    valid_ids.update(map(str, (measurements.get("sections") or {}).keys()))
    valid_event_ids = {str(row.get("event_id"))
                       for row in validated_reference.get("accepted_events") or []}
    relations = value.get("functional_relations")
    if not isinstance(relations, list) or not relations:
        raise DNAV2Error("editing_relations_missing")
    for row in relations:
        refs = set(map(str, row.get("timeline_ids") or []))
        if not refs or not refs.issubset(valid_ids):
            raise DNAV2Error("editing_timeline_reference_invalid")
        event_refs = set(map(str, row.get("event_ids") or []))
        if not event_refs.issubset(valid_event_ids):
            raise DNAV2Error("editing_event_reference_invalid")
        if row.get("editing_operation") not in {
                "establish", "demonstrate", "accumulate", "qualify",
                "contrast", "bridge"}:
            raise DNAV2Error("editing_operation_invalid")
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 < float(confidence) <= 1:
            raise DNAV2Error("editing_confidence_invalid")
    event_intervals = [event.get("interval") or []
                       for event in validated_reference.get("accepted_events") or []]
    required_sections = set()
    for segment in measurements.get("segments") or []:
        segment_interval = segment.get("interval") or []
        if any(len(interval) == 2 and
               float(interval[0]) < float(segment_interval[1]) and
               float(interval[1]) > float(segment_interval[0])
               for interval in event_intervals):
            required_sections.add(str(segment.get("section_id")))
    covered_ids = {str(ref) for row in relations
                   for ref in row.get("timeline_ids") or []}
    covered_sections = {
        str(row.get("section_id")) for row in measurements.get("segments") or []
        if str(row.get("timeline_id")) in covered_ids
    }
    covered_sections.update(required_sections.intersection(covered_ids))
    if not required_sections.issubset(covered_sections):
        raise DNAV2Error("editing_event_section_uncovered")


def run_independent_analyses(validated_reference: dict[str, Any], *,
                             runner: Any, trace_dir: Path | None = None,
                             analysis_contract: dict[str, Any] | None = None,
                             cache_dir: Path | None = None
                             ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use two clean text calls; neither draft appears in the other request."""
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
    if trace_dir is not None:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)

    def load_cache(name: str) -> dict[str, Any] | None:
        if cache_path is None or not (cache_path / name).is_file():
            return None
        value = json.loads((cache_path / name).read_text(encoding="utf-8"))
        unhashed = dict(value)
        expected = unhashed.pop("artifact_sha", None)
        if expected != json_hash(unhashed):
            return None
        return value

    def save_cache(name: str, value: dict[str, Any]) -> None:
        if cache_path is not None:
            (cache_path / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    view = _analysis_view(validated_reference)
    interpretation_input = {
        "validated_reference": view,
        "analysis_contract": analysis_contract or {},
    }
    base = json.dumps(interpretation_input, ensure_ascii=False,
                      separators=(",", ":"))
    interpretation = load_cache("interpretation.json")
    if interpretation is not None:
        if interpretation.get("schema_version") != INTERPRETATION_VERSION or \
                interpretation.get("validated_reference_sha") != \
                validated_reference.get("artifact_sha"):
            interpretation = None
        else:
            try:
                validate_interpretation(
                    interpretation, validated_reference, analysis_contract)
            except DNAV2Error:
                interpretation = None
    if interpretation is None:
        proposed_interpretation = _ask_validated_object(
            runner=runner, prompt=INTERPRETATION_PROMPT + base,
            max_new_tokens=3072,
            validator=lambda value: validate_interpretation(
                value, validated_reference, analysis_contract), trace_dir=trace_dir,
            trace_name="interpretation", attempts=3)
        _attach_interpretation_provenance(
            proposed_interpretation, validated_reference)
        interpretation = {
            "schema_version": INTERPRETATION_VERSION,
            "validated_reference_sha": validated_reference.get("artifact_sha"),
            **proposed_interpretation,
        }
        interpretation["artifact_sha"] = json_hash(interpretation)
        save_cache("interpretation.json", interpretation)
    elif trace_dir is not None:
        (Path(trace_dir) / "interpretation_reuse_gate.json").write_text(
            json.dumps({"passed": True, "reused": True}, indent=2),
            encoding="utf-8")

    measurements = compute_timeline_measurements(validated_reference)
    edit_input = {"validated_reference": view,
                  "deterministic_measurements": measurements}
    editing = load_cache("editing_analysis.json")
    if editing is not None:
        if editing.get("schema_version") != EDITING_ANALYSIS_VERSION or \
                editing.get("validated_reference_sha") != \
                validated_reference.get("artifact_sha"):
            editing = None
        else:
            try:
                validate_editing_analysis(editing, measurements,
                                          validated_reference)
            except DNAV2Error:
                editing = None
    if editing is None:
        proposed = _ask_validated_object(
            runner=runner,
            prompt=EDITING_ANALYSIS_PROMPT + json.dumps(
                edit_input, ensure_ascii=False, separators=(",", ":")),
            max_new_tokens=3072,
            validator=lambda value: validate_editing_analysis(
                value, measurements, validated_reference), trace_dir=trace_dir,
            trace_name="editing")
        editing = {
            "schema_version": EDITING_ANALYSIS_VERSION,
            "validated_reference_sha": validated_reference.get("artifact_sha"),
            "deterministic_measurements": measurements,
            **proposed,
        }
        editing["artifact_sha"] = json_hash(editing)
        save_cache("editing_analysis.json", editing)
    elif trace_dir is not None:
        (Path(trace_dir) / "editing_reuse_gate.json").write_text(
            json.dumps({"passed": True, "reused": True}, indent=2),
            encoding="utf-8")
    return interpretation, editing


def _ids(rows: Iterable[dict[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if str(row.get(key) or "")}


_VARIABLE_KINDS = {"observer", "subject", "proposition", "evidence",
                   "information_state", "scope_boundary"}
_RELATION_MECHANISMS = {
    "holds_belief", "supports_proposition", "contradicts_proposition",
    "triggers_revision", "qualifies_scope", "accumulates_evidence",
    "orders_disclosure", "reframes_context", "precedes",
}
_CONSTRAINT_RULE_TYPES = {
    "exact_proposition_match", "evidence_required", "scope_bound",
    "revision_required", "ordering_required", "qualification_preserved",
}
_EDITING_OPERATIONS = {"establish", "demonstrate", "accumulate", "qualify",
                       "contrast", "bridge"}


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
    if not {"proposition", "evidence"}.issubset(
            {str(row.get("kind")) for row in variables}):
        raise DNAV2Error("dna_core_variable_kind_missing")
    for variable in variables:
        if variable.get("kind") not in _VARIABLE_KINDS:
            raise DNAV2Error("dna_variable_kind_invalid")
        if not str(variable.get("description") or "").strip():
            raise DNAV2Error("dna_variable_description_missing")
    for relation in relations:
        if str(relation.get("source")) not in variable_ids or \
                str(relation.get("target")) not in variable_ids:
            raise DNAV2Error("dna_relation_variable_missing")
        if relation.get("type") not in {"logical", "causal", "temporal"}:
            raise DNAV2Error("dna_relation_type_invalid")
        if relation.get("mechanism") not in _RELATION_MECHANISMS:
            raise DNAV2Error("dna_relation_mechanism_invalid")
    for constraint in constraints:
        if constraint.get("rule_type") not in _CONSTRAINT_RULE_TYPES:
            raise DNAV2Error("dna_constraint_rule_type_invalid")
        if constraint.get("required") is not True:
            raise DNAV2Error("dna_constraint_not_required")
    for slot in slots:
        if not set(map(str, slot.get("constraint_ids") or [])).issubset(constraint_ids):
            raise DNAV2Error("dna_slot_constraint_missing")
    supported = value.get("supported_by") or {}
    if not relation_ids.issubset(set(map(str, supported))):
        raise DNAV2Error("dna_relation_support_missing")
    for relation_id in relation_ids:
        if not isinstance(supported.get(relation_id), list) or not supported[relation_id]:
            raise DNAV2Error("dna_relation_support_empty")
    for row in edits:
        if row.get("editing_operation") not in _EDITING_OPERATIONS:
            raise DNAV2Error("dna_editing_operation_invalid")
    validation = value.get("validation_record") or {}
    if validation != {
            "structure_only": True,
            "source_bindings_confined_to_audit": True,
            "provenance_rules_not_creative": True}:
        raise DNAV2Error("dna_validation_record_invalid")
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


_SURFACE_TOKEN_ALLOWLIST = {
    "about", "abstract", "action", "after", "agent", "before", "bounded",
    "cause", "claim", "context", "domain", "effect", "entity", "evidence",
    "general", "information", "later", "observer", "ordering", "person",
    "proposition", "relation", "result", "scope", "sequence", "state",
    "subject", "support", "supported", "visible",
}


def _surface_stem(token: str) -> str:
    token = token.casefold()
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) >= 7 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def _reference_surface_tokens(validated_reference: dict[str, Any]) -> set[str]:
    """Build an internal leak gate without publishing its source vocabulary."""
    tokens: set[str] = set()
    for row in validated_reference.get("accepted_claims") or []:
        text = str(row.get("object") or "")
        for token in re.findall(r"[A-Za-z]{5,}|[\u4e00-\u9fff]{2,}", text):
            stem = _surface_stem(token)
            if stem not in _SURFACE_TOKEN_ALLOWLIST:
                tokens.add(stem)
    return tokens


def _validate_abstract_surface(publish: dict[str, Any],
                               validated_reference: dict[str, Any]) -> None:
    publish_tokens: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)
        elif isinstance(item, str):
            publish_tokens.update(_surface_stem(token) for token in re.findall(
                r"[A-Za-z]{5,}|[\u4e00-\u9fff]{2,}", item))

    walk(publish)
    if publish_tokens.intersection(_reference_surface_tokens(validated_reference)):
        # Never echo the matching source token in the error path.
        raise DNAV2Error("dna_publish_surface_binding_leak")


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
    _validate_abstract_surface(publish, validated_reference)
    publish["artifact_sha"] = json_hash(publish)
    return publish


def _validate_dna_supports(audit: dict[str, Any], *,
                           validated_reference: dict[str, Any],
                           interpretation: dict[str, Any],
                           editing_analysis: dict[str, Any]) -> None:
    valid_support = (_source_ids(validated_reference) |
                     _ids(interpretation.get("propositions") or [],
                          "proposition_id") |
                     _ids(interpretation.get("relations") or [], "relation_id") |
                     _ids(editing_analysis.get("functional_relations") or [],
                          "relation_id"))
    for refs in (audit.get("supported_by") or {}).values():
        if not set(map(str, refs or [])).issubset(valid_support):
            raise DNAV2Error("dna_support_id_invalid")
    interpretation_types = {
        str(row.get("type")) for row in interpretation.get("relations") or []}
    mechanisms = {str(row.get("mechanism"))
                  for row in audit.get("relations") or []}
    if "contradicts" in interpretation_types and \
            "contradicts_proposition" not in mechanisms:
        raise DNAV2Error("dna_contradiction_mechanism_missing")
    if "qualifies" in interpretation_types and \
            "qualifies_scope" not in mechanisms:
        raise DNAV2Error("dna_qualification_mechanism_missing")


def run_director(validated_reference: dict[str, Any], interpretation: dict[str, Any],
                 editing_analysis: dict[str, Any], *, runner: Any,
                 trace_dir: Path | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    if interpretation.get("validated_reference_sha") != \
            validated_reference.get("artifact_sha"):
        raise DNAV2Error("interpretation_parent_mismatch")
    if editing_analysis.get("validated_reference_sha") != \
            validated_reference.get("artifact_sha"):
        raise DNAV2Error("editing_parent_mismatch")
    payload = {"validated_reference": _analysis_view(validated_reference),
               "interpretation": interpretation,
               "editing_analysis": editing_analysis}
    base_prompt = DIRECTOR_DNA_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    proposed_publish: tuple[dict[str, Any], dict[str, Any]] | None = None
    last_error: DNAV2Error | None = None
    trace_path = Path(trace_dir) if trace_dir is not None else None
    if trace_path is not None:
        trace_path.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        suffix = ""
        if last_error is not None:
            guidance = {
                "dna_publish_surface_binding_leak": (
                    "Use only belief, proposition, evidence, revision, scope, "
                    "and information-state roles. Remove every physical, body, "
                    "activity, object, place, identity, and caption description."),
                "dna_support_id_invalid": (
                    "supported_by may contain only supplied claim IDs, event IDs, "
                    "interpretation proposition/relation IDs, or editing relation "
                    "IDs; never timeline segment IDs or variable IDs."),
            }.get(last_error.reason_code, "Follow the declared schema exactly.")
            suffix = ("\nCORRECTION: the previous object failed gate "
                      f"{last_error.reason_code}. {guidance} Re-run structure abduction and "
                      "return a complete fresh JSON object. Do not quote the "
                      "previous response or any source phrase.\n")
        answer = runner.ask(base_prompt + suffix, max_new_tokens=4096,
                            stop_after_json_object=True)
        raw = _answer_text(answer)
        if trace_path is not None:
            (trace_path / f"director_attempt_{attempt:03d}.txt").write_text(
                raw, encoding="utf-8")
        try:
            candidate = _parse_object(raw)
            validate_dna_audit(candidate)
            _validate_dna_supports(
                candidate, validated_reference=validated_reference,
                interpretation=interpretation,
                editing_analysis=editing_analysis)
            candidate_publish = publish_dna(
                candidate, validated_reference=validated_reference)
            proposed_publish = (candidate, candidate_publish)
            if trace_path is not None:
                (trace_path / f"director_attempt_{attempt:03d}_gate.json").write_text(
                    json.dumps({"passed": True}, indent=2), encoding="utf-8")
            break
        except DNAV2Error as exc:
            last_error = exc
            if trace_path is not None:
                (trace_path / f"director_attempt_{attempt:03d}_gate.json").write_text(
                    json.dumps({"passed": False,
                                "reason_code": exc.reason_code}, indent=2),
                    encoding="utf-8")
    if proposed_publish is None:
        assert last_error is not None
        raise last_error
    proposed, publish = proposed_publish
    audit = {
        "schema_version": DNA_AUDIT_VERSION,
        "parent_shas": [validated_reference.get("artifact_sha"),
                        interpretation.get("artifact_sha"),
                        editing_analysis.get("artifact_sha")],
        **proposed,
    }
    audit["artifact_sha"] = json_hash(audit)
    # Rebuild from the versioned audit so the release interface is explicitly
    # derived from the artifact that retains bindings and validation records.
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
