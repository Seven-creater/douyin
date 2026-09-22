"""One extraction, one independent audit, and deterministic IR publication."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.agentic_video.creative_structure_v1.publisher import publish_structure_spec
from src.agentic_video.creative_structure_v1.schema import (
    ANTI_INVARIANT_CATEGORIES,
    AUDIT_SCHEMA_VERSION,
    INDEPENDENT_AUDIT_SCHEMA_VERSION,
    VALIDATION_RECORD_KEYS,
)
from src.agentic_video.creative_structure_v1.transfer_test import (
    audit_transfer_profiles,
    build_transfer_test_plan,
)
from src.agentic_video.creative_structure_v1.validator import (
    StructureValidationError,
    validate_abstraction_boundary,
    validate_structure_audit,
)
from src.agentic_video.manifest import json_hash

EXTRACTION_MAX_NEW_TOKENS = 6144
AUDIT_MAX_NEW_TOKENS = 4096

_DRAFT_KEYS = {
    "plan_status",
    "abstraction_confidence",
    "dimension_status",
    "structural_schema",
    "information_state_arc",
    "generation_constraints",
    "editing_schema",
    "binding_slots",
    "audit_annex",
}

STRUCTURE_EXTRACTION_PROMPT = """You are a Creative Structure Plan extractor.

Convert the supplied verified narrative and editing structures into one
domain-independent intermediate representation for downstream creative skills.
Do not describe what happened in the reference. Describe only the necessary
relations and information-state changes that another work may realize with
different people, domains, settings, and events.

The Structural Schema is a necessary relation representation of the reference
effect; it is not expected to cover a complete narrative. A downstream
screenwriting skill, not this task, binds it to concrete characters, domains,
events, acts, scenes, or dialogue.

This is one extraction call. Return one JSON object and no commentary. Do not
emit a Narrative Function layer, mechanism_graph, experience_arc, event
constraints, story beats, plot summary, theme, premise, or screenplay. Do not
assume reframe, reversal, belief revision, counterevidence, or any required
relation. Counts may be zero. Prefer partial or blocked over invented content.

Return exactly these top-level fields:
plan_status, abstraction_confidence, dimension_status, structural_schema,
information_state_arc, generation_constraints, editing_schema, binding_slots,
audit_annex.

abstraction_confidence and every relation/constraint confidence are JSON
numbers from 0.0 through 1.0, never labels.

dimension_status has exactly these keys in this order:
structural_schema, information_state_arc, generation_constraints,
editing_structure, editing_function, binding_slots. Each value is supported,
insufficient, or not_applicable. Content exists exactly for supported
dimensions. complete requires no insufficient dimension and requires supported
structural_schema plus generation_constraints. partial requires at least one
supported and one insufficient dimension. blocked contains no supported
content.

structural_schema is null or exactly {elements, relations}.

Each element has exactly:
element_id, kind, abstract_role, support_refs.
Allowed kinds are information_state, proposition, evidence_role,
scope_boundary, resolution_state. element_id is newly minted and cannot reuse a
source evidence ID. abstract_role must remain true after replacing reference
people, domain, setting, physical form, actions, wording, and visual motifs.

Each relation has exactly:
relation_id, relation_kind, predicate, arguments, preconditions, effects,
support_refs, confidence.
Allowed relation_kind values are causal, evidential, temporal, scope,
disclosure. predicate and argument roles are open, domain-independent relation
language; there is no required predicate. arguments is a non-empty list of
exactly {role, element_id}. preconditions and effects are lists of
domain-independent strings and may be empty. A relation is not a rewritten
reference event.

information_state_arc items have exactly:
state_id, order, available_element_ids, transition_from_state_id,
trigger_relation_ids, support_refs.
They describe which abstract elements are informationally available, not story
beats or guaranteed audience psychology. The first state has null
transition_from_state_id and empty trigger_relation_ids. Every later state
references an earlier state and at least one structural relation. The list may
be empty; there are no required state roles or counts.

generation_constraints items have exactly:
constraint_id, obligation, constraint_type, requires_relation_ids, rule,
support_refs, confidence.
obligation is required, preferred, or prohibited. constraint_type is
relational, ordering, scope, or exclusion. requires_relation_ids is non-empty.
rule constrains preservation of abstract relations; it must not require a
reference competition, achievement, performance, person type, occupation,
body, activity, wording, setting, or visual motif.

editing_schema is exactly {constraints}. Each constraint has exactly:
constraint_id, obligation, dimension, operator, applies_to_state_ids,
applies_to_relation_ids, source_strength, transfer_policy, support_refs,
confidence.
obligation is required, preferred, or advisory. dimension is pace,
shot_duration, transition_density, information_density, information_function,
or pattern_type. operator is increases, decreases, holds, contrasts,
accumulates, reveals, qualifies, bridges, or free. source_strength is measured,
audited_semantic, or unaudited_semantic. transfer_policy is preserve_relation,
preserve_function, or free_realization. Measured items may be required or
preferred. Any unaudited_semantic item must be advisory with free_realization.
Anchor editing only to information state or structural relation IDs. If those
dimensions are unsupported, anchor lists may be empty.

binding_slots items have exactly:
slot_id, slot_type, bound_by, constraints.
slot_type is domain, entity_role, setting, surface_realization, evidence_form,
tone, or visual_motif. bound_by is always downstream_skill. constraints must be
cross-domain and cannot preserve source surface content.

audit_annex is exactly {source_bindings, anti_invariants}. Every element,
relation, information state, generation constraint, editing constraint, and
binding slot has exactly one source_bindings row with abstract_id, source_refs,
reference_specific_summary. Concrete reference detail is permitted only in
reference_specific_summary. Each anti_invariants row has binding_id, category,
source_refs; category is entity, domain, setting, physical_form, literal_event,
wording, or visual_motif. Evidence IDs occur only in support_refs/source_refs.

On-screen statements remain attributed content, not independently verified
physical facts. Identity alignment, raw claims, paths, prompts, traces, section
summaries, and source IDs are not creative structure. JSON only.

INPUT:
"""

INDEPENDENT_AUDIT_PROMPT = """You are an independent Creative Structure Plan auditor.

Audit the candidate against the verified bundle and the four transfer profiles.
Do not rewrite, repair, complete, or improve it. Do not reward a familiar story
template. Check each item for: evidence grounding, relation entailment,
domain-independent abstraction, replaceability across the three positive
profiles, rejection of the surface-lure negative, and constraint validity.

Return exactly:
{
  "schema_version": "creative_structure_independent_audit_v1",
  "item_checks": [
    {
      "item_id": "...",
      "item_type": "element|relation|information_state|generation_constraint|editing_constraint|binding_slot",
      "grounding_status": "supported|unsupported|not_applicable",
      "abstraction_valid": true,
      "transfer_valid": true,
      "constraint_valid": true,
      "reason": "..."
    }
  ],
  "leakage_findings": [
    {
      "finding_id": "L1",
      "item_id": "...",
      "field": "...",
      "leaked_surface": "...",
      "category": "entity|domain|setting|physical_form|literal_event|wording|visual_motif",
      "reason": "..."
    }
  ],
  "dimension_status_confirmed": {
    "structural_schema": "supported|insufficient|not_applicable",
    "information_state_arc": "supported|insufficient|not_applicable",
    "generation_constraints": "supported|insufficient|not_applicable",
    "editing_structure": "supported|insufficient|not_applicable",
    "editing_function": "supported|insufficient|not_applicable",
    "binding_slots": "supported|insufficient|not_applicable"
  },
  "overall": {
    "grounding_passed": true,
    "relation_entailment_passed": true,
    "abstraction_boundary_passed": true,
    "transferability_passed": true,
    "pass": true
  },
  "limitations": []
}

Cover every candidate item exactly once. grounding_status may be not_applicable
only for a genuinely free binding_slot. Any unsupported item, reference-bound
abstraction, failed transfer, invalid constraint, leakage finding, or dishonest
dimension status makes overall.pass false. Reference-specific detail inside
audit_annex is audit-only and must not be reported as publish leakage. JSON only.

INPUT:
"""


class StructureExtractionError(RuntimeError):
    pass


class IndependentStructureAuditError(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


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
        raise StructureExtractionError("response_json_invalid") from exc
    if not isinstance(value, dict):
        raise StructureExtractionError("response_not_object")
    return value


def _artifact_hash_valid(value: dict[str, Any]) -> bool:
    claimed = value.get("artifact_sha")
    payload = {key: item for key, item in value.items() if key != "artifact_sha"}
    return isinstance(claimed, str) and claimed == json_hash(payload)


def _input_artifacts(narrative: dict[str, Any], editing: dict[str, Any]
                     ) -> list[dict[str, str]]:
    return [
        {
            "artifact_id": "reference_narrative",
            "artifact_type": "narrative_interpretation",
            "sha": str(narrative["artifact_sha"]),
            "schema_version": str(narrative["schema_version"]),
        },
        {
            "artifact_id": "editing_grammar",
            "artifact_type": "editing_grammar",
            "sha": str(editing["artifact_sha"]),
            "schema_version": str(editing["schema_version"]),
        },
    ]


def _editing_evidence(structural: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = (
        ("PACE", "pace_profile"),
        ("PACE_CHANGE", "pace_changes"),
        ("DURATION", "duration_profile"),
        ("DURATION_OUTLIER", "shot_duration_outliers"),
        ("TRANSITION", "transition_profile"),
        ("TRANSITION_SEQUENCE", "transition_sequences"),
    )
    for prefix, key in groups:
        for index, fact in enumerate(structural.get(key) or [], start=1):
            evidence_id = str(fact.get("fact_id") or f"{prefix}_{index:02d}")
            rows.append({
                "evidence_id": evidence_id,
                "kind": key,
                "source_strength": "measured",
                "fact": copy.deepcopy(fact),
            })
    return rows


def build_verified_bundle(narrative: dict[str, Any], editing: dict[str, Any]
                          ) -> dict[str, Any]:
    """Build the only extraction view; raw claims and identity data are absent."""
    if not _artifact_hash_valid(narrative):
        raise StructureExtractionError("narrative_artifact_sha_invalid")
    if not _artifact_hash_valid(editing):
        raise StructureExtractionError("editing_artifact_sha_invalid")
    if narrative.get("verification_status") != "SUPPORTED":
        raise StructureExtractionError("narrative_not_verified")
    units = narrative.get("narrative_units") or []
    relations = narrative.get("relations") or []
    if any(row.get("verification_status") != "SUPPORTED" for row in units):
        raise StructureExtractionError("narrative_unit_not_verified")
    if any(row.get("verification_status") != "SUPPORTED" for row in relations):
        raise StructureExtractionError("narrative_relation_not_verified")

    measured = _editing_evidence(editing.get("structural_grammar") or {})
    semantic_patterns = [
        {**copy.deepcopy(row), "source_strength": "unaudited_semantic"}
        for row in editing.get("semantic_patterns") or []
    ]
    functions = [
        {
            **copy.deepcopy(row),
            "evidence_id": f"FUNCTION_{row.get('pattern_id')}",
            "source_strength": "unaudited_semantic",
        }
        for row in editing.get("functions") or []
    ]
    bundle: dict[str, Any] = {
        "schema_version": "r2_verified_bundle_for_structure_v1",
        "input_artifacts": _input_artifacts(narrative, editing),
        "narrative_interpretation": {
            "narrative_units": copy.deepcopy(units),
            "relations": copy.deepcopy(relations),
            "limitations": copy.deepcopy(narrative.get("limitations") or []),
        },
        "editing_grammar": {
            "measured_evidence": measured,
            "semantic_patterns": semantic_patterns,
            "functions": functions,
            "limitations": {
                "patterns": copy.deepcopy(editing.get("pattern_limitations") or []),
                "functions": copy.deepcopy(editing.get("function_limitations") or []),
            },
        },
    }
    catalog: list[dict[str, str]] = []
    for row in units:
        catalog.append({
            "evidence_id": row["unit_id"],
            "kind": "narrative_unit",
            "source_strength": "audited_semantic",
        })
    for row in relations:
        catalog.append({
            "evidence_id": row["relation_id"],
            "kind": "narrative_relation",
            "source_strength": "audited_semantic",
        })
    catalog.extend({key: row[key] for key in (
        "evidence_id", "kind", "source_strength")}
                   for row in measured)
    catalog.extend({
        "evidence_id": row["pattern_id"],
        "kind": "semantic_editing_pattern",
        "source_strength": "unaudited_semantic",
    } for row in semantic_patterns)
    catalog.extend({
        "evidence_id": row["evidence_id"],
        "kind": "editing_function",
        "source_strength": "unaudited_semantic",
    } for row in functions)
    bundle["evidence_catalog"] = catalog
    bundle["artifact_sha"] = json_hash(bundle)
    return bundle


def _model_metadata(answer: Any) -> dict[str, Any]:
    return {
        "input_tokens": getattr(answer, "input_tokens", None),
        "output_tokens": getattr(answer, "output_tokens", None),
        "elapsed_s": getattr(answer, "elapsed_s", None),
    }


def extract_structure_plan(runner: Any, narrative: dict[str, Any],
                           editing: dict[str, Any], *, trace_dir: Path
                           ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Call the structure model exactly once and validate without repair."""
    stage = Path(trace_dir) / "01_extraction"
    stage.mkdir(parents=True, exist_ok=True)
    bundle = build_verified_bundle(narrative, editing)
    payload = {"verified_r2_bundle": bundle}
    prompt = STRUCTURE_EXTRACTION_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    _write_json(stage / "request.json", {
        "stage": "creative_structure_extraction",
        "attempt": 1,
        "max_new_tokens": EXTRACTION_MAX_NEW_TOKENS,
        "stop_after_json_object": True,
        "prompt": prompt,
        "payload": payload,
    })
    (stage / "prompt.txt").write_text(prompt, encoding="utf-8")
    answer = runner.ask(prompt, max_new_tokens=EXTRACTION_MAX_NEW_TOKENS,
                        stop_after_json_object=True)
    raw = str(getattr(answer, "text", answer))
    (stage / "raw_response.txt").write_text(raw, encoding="utf-8")
    try:
        draft = _parse_object(raw)
        if set(draft) != _DRAFT_KEYS:
            raise StructureExtractionError("extraction_response_keys_invalid")
        candidate = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            **copy.deepcopy(draft),
            "input_artifacts": _input_artifacts(narrative, editing),
            "validation_record": {
                key: False for key in VALIDATION_RECORD_KEYS
            },
        }
        candidate["artifact_sha"] = json_hash(candidate)
        validate_structure_audit(candidate)
        validate_abstraction_boundary(candidate, bundle)
        known = {row["evidence_id"] for row in bundle["evidence_catalog"]}
        used = {
            ref
            for binding in candidate["audit_annex"]["source_bindings"]
            for ref in binding["source_refs"]
        }
        used.update(
            ref
            for row in candidate["audit_annex"]["anti_invariants"]
            for ref in row["source_refs"]
        )
        unknown = sorted(used - known)
        if unknown:
            raise StructureValidationError(
                "candidate_source_ref_not_in_bundle", unknown)
    except BaseException as exc:
        _write_json(stage / "validation.json", {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_call": _model_metadata(answer),
        })
        raise
    metadata = _model_metadata(answer)
    _write_json(stage / "candidate.json", candidate)
    _write_json(stage / "validation.json", {
        "status": "PASS",
        "candidate_sha": candidate["artifact_sha"],
        "bundle_sha": bundle["artifact_sha"],
        "model_call": metadata,
    })
    return candidate, bundle, metadata


def _candidate_items(candidate: dict[str, Any]) -> dict[str, str]:
    structural = candidate.get("structural_schema") or {
        "elements": [], "relations": []}
    rows = [
        *((row["element_id"], "element") for row in structural["elements"]),
        *((row["relation_id"], "relation") for row in structural["relations"]),
        *((row["state_id"], "information_state")
          for row in candidate["information_state_arc"]),
        *((row["constraint_id"], "generation_constraint")
          for row in candidate["generation_constraints"]),
        *((row["constraint_id"], "editing_constraint")
          for row in candidate["editing_schema"]["constraints"]),
        *((row["slot_id"], "binding_slot") for row in candidate["binding_slots"]),
    ]
    result: dict[str, str] = {}
    for item_id, item_type in rows:
        if item_id in result:
            raise IndependentStructureAuditError(
                f"candidate_item_id_duplicate:{item_id}")
        result[item_id] = item_type
    return result


def validate_independent_audit(value: dict[str, Any],
                               candidate: dict[str, Any]) -> bool:
    expected_keys = {
        "schema_version", "item_checks", "leakage_findings",
        "dimension_status_confirmed", "overall", "limitations",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise IndependentStructureAuditError("independent_audit_shape_invalid")
    if value["schema_version"] != INDEPENDENT_AUDIT_SCHEMA_VERSION:
        raise IndependentStructureAuditError("independent_audit_schema_invalid")
    expected = _candidate_items(candidate)
    actual: dict[str, dict[str, Any]] = {}
    item_types = {
        "element", "relation", "information_state", "generation_constraint",
        "editing_constraint", "binding_slot",
    }
    for check in value["item_checks"]:
        if not isinstance(check, dict) or set(check) != {
            "item_id", "item_type", "grounding_status", "abstraction_valid",
            "transfer_valid", "constraint_valid", "reason",
        }:
            raise IndependentStructureAuditError("independent_item_check_invalid")
        item_id = str(check["item_id"])
        item_type = str(check["item_type"])
        if item_type not in item_types or expected.get(item_id) != item_type:
            raise IndependentStructureAuditError(
                f"independent_item_unknown:{item_id}")
        if item_id in actual:
            raise IndependentStructureAuditError(
                f"independent_item_duplicate:{item_id}")
        if check["grounding_status"] not in {
            "supported", "unsupported", "not_applicable"
        }:
            raise IndependentStructureAuditError(
                f"grounding_status_invalid:{item_id}")
        if check["grounding_status"] == "not_applicable" \
                and item_type != "binding_slot":
            raise IndependentStructureAuditError(
                f"grounding_not_applicable_invalid:{item_id}")
        if any(not isinstance(check[key], bool) for key in (
                "abstraction_valid", "transfer_valid", "constraint_valid")) \
                or not str(check["reason"]).strip():
            raise IndependentStructureAuditError(
                f"independent_item_fields_invalid:{item_id}")
        actual[item_id] = check
    if set(actual) != set(expected):
        raise IndependentStructureAuditError("independent_item_coverage_invalid")

    findings = value["leakage_findings"]
    if not isinstance(findings, list):
        raise IndependentStructureAuditError("leakage_findings_invalid")
    finding_ids: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {
            "finding_id", "item_id", "field", "leaked_surface", "category",
            "reason",
        }:
            raise IndependentStructureAuditError("leakage_finding_invalid")
        finding_id = str(finding["finding_id"])
        if finding_id in finding_ids or str(finding["item_id"]) not in expected:
            raise IndependentStructureAuditError(
                f"leakage_finding_reference_invalid:{finding_id}")
        finding_ids.add(finding_id)
        if finding["category"] not in ANTI_INVARIANT_CATEGORIES:
            raise IndependentStructureAuditError(
                f"leakage_category_invalid:{finding_id}")
        if not all(str(finding[key]).strip() for key in (
                "field", "leaked_surface", "reason")):
            raise IndependentStructureAuditError(
                f"leakage_finding_fields_invalid:{finding_id}")

    if value["dimension_status_confirmed"] != candidate["dimension_status"]:
        raise IndependentStructureAuditError("dimension_status_audit_mismatch")
    overall = value["overall"]
    overall_keys = {
        "grounding_passed", "relation_entailment_passed",
        "abstraction_boundary_passed", "transferability_passed", "pass",
    }
    if not isinstance(overall, dict) or set(overall) != overall_keys \
            or any(not isinstance(overall[key], bool) for key in overall_keys):
        raise IndependentStructureAuditError("independent_audit_overall_invalid")
    if not isinstance(value["limitations"], list) or any(
            not isinstance(item, str) for item in value["limitations"]):
        raise IndependentStructureAuditError("independent_audit_limitations_invalid")

    items_pass = all(
        check["grounding_status"] in {"supported", "not_applicable"}
        and check["abstraction_valid"]
        and check["transfer_valid"]
        and check["constraint_valid"]
        for check in actual.values()
    )
    computed_pass = (
        candidate["plan_status"] != "blocked"
        and items_pass
        and not findings
        and all(overall[key] for key in overall_keys if key != "pass")
    )
    if overall["pass"] != computed_pass:
        raise IndependentStructureAuditError("independent_audit_pass_inconsistent")
    return computed_pass


def audit_structure_plan(runner: Any, candidate: dict[str, Any],
                         bundle: dict[str, Any], *, trace_dir: Path
                         ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Call the independent auditor exactly once and never repair the plan."""
    stage = Path(trace_dir) / "02_independent_audit"
    stage.mkdir(parents=True, exist_ok=True)
    payload = {
        "verified_r2_bundle": bundle,
        "candidate_structure_plan": candidate,
        "transfer_profiles": audit_transfer_profiles(),
    }
    prompt = INDEPENDENT_AUDIT_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    _write_json(stage / "request.json", {
        "stage": "independent_structure_audit",
        "attempt": 1,
        "max_new_tokens": AUDIT_MAX_NEW_TOKENS,
        "stop_after_json_object": True,
        "prompt": prompt,
        "payload": payload,
    })
    (stage / "prompt.txt").write_text(prompt, encoding="utf-8")
    answer = runner.ask(prompt, max_new_tokens=AUDIT_MAX_NEW_TOKENS,
                        stop_after_json_object=True)
    raw = str(getattr(answer, "text", answer))
    (stage / "raw_response.txt").write_text(raw, encoding="utf-8")
    try:
        result = _parse_object(raw)
        passed = validate_independent_audit(result, candidate)
    except BaseException as exc:
        _write_json(stage / "validation.json", {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_call": _model_metadata(answer),
        })
        raise
    result["candidate_sha"] = candidate["artifact_sha"]
    result["bundle_sha"] = bundle["artifact_sha"]
    result["artifact_sha"] = json_hash(result)
    metadata = _model_metadata(answer)
    _write_json(stage / "audit_result.json", result)
    _write_json(stage / "validation.json", {
        "status": "PASS",
        "audit_pass": passed,
        "audit_result_sha": result["artifact_sha"],
        "model_call": metadata,
    })
    return result, passed, metadata


def _accepted_audit(candidate: dict[str, Any],
                    audit_result: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(candidate)
    overall = audit_result["overall"]
    result["validation_record"] = {
        "grounding_passed": overall["grounding_passed"],
        "relation_entailment_passed": overall["relation_entailment_passed"],
        "abstraction_boundary_passed": overall["abstraction_boundary_passed"],
        "transferability_passed": overall["transferability_passed"],
        "editing_promotion_policy_passed": True,
        "publish_whitelist_passed": True,
    }
    result.pop("artifact_sha", None)
    result["artifact_sha"] = json_hash(result)
    validate_structure_audit(result)
    return result


def run_structure_acceptance(extraction_runner: Any, audit_runner: Any,
                             narrative: dict[str, Any], editing: dict[str, Any], *,
                             trace_dir: Path, spec_id: str) -> dict[str, Any]:
    """Run one extraction, one audit, then deterministic publish and transfer plan."""
    root = Path(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        candidate, bundle, extraction_meta = extract_structure_plan(
            extraction_runner, narrative, editing, trace_dir=root)
        audit_result, passed, audit_meta = audit_structure_plan(
            audit_runner, candidate, bundle, trace_dir=root)
        if not passed:
            summary = {
                "schema_version": "creative_structure_run_summary_v1",
                "status": "BLOCKED",
                "reason": "independent_audit_failed",
                "model_calls": 2,
                "retry_count": 0,
                "candidate_sha": candidate["artifact_sha"],
                "audit_result_sha": audit_result["artifact_sha"],
                "creative_structure_spec_sha": None,
                "extraction_call": extraction_meta,
                "audit_call": audit_meta,
            }
            _write_json(root / "run_summary.json", summary)
            return {
                **summary,
                "candidate": candidate,
                "audit_result": audit_result,
                "creative_structure_spec": None,
            }

        accepted = _accepted_audit(candidate, audit_result)
        spec = publish_structure_spec(accepted, spec_id=spec_id)
        publish_dir = root / "03_publish"
        _write_json(publish_dir / "creative_structure_audit_v1.json", accepted)
        _write_json(publish_dir / "creative_structure_spec_v1.json", spec)
        _write_json(publish_dir / "publish_validation.json", {
            "status": "PASS",
            "publisher": "deterministic_whitelist",
            "audit_sha": accepted["artifact_sha"],
            "spec_sha": spec["artifact_sha"],
        })
        transfer = build_transfer_test_plan(spec)
        _write_json(root / "04_transferability/transfer_test_plan.json", transfer)
        summary = {
            "schema_version": "creative_structure_run_summary_v1",
            "status": "PUBLISHED",
            "reason": None,
            "model_calls": 2,
            "retry_count": 0,
            "candidate_sha": candidate["artifact_sha"],
            "accepted_audit_sha": accepted["artifact_sha"],
            "audit_result_sha": audit_result["artifact_sha"],
            "creative_structure_spec_sha": spec["artifact_sha"],
            "extraction_call": extraction_meta,
            "audit_call": audit_meta,
            "transfer_review_status": transfer["status"],
        }
        _write_json(root / "run_summary.json", summary)
        return {
            **summary,
            "candidate": accepted,
            "audit_result": audit_result,
            "creative_structure_spec": spec,
        }
    except BaseException as exc:
        model_calls = int((root / "01_extraction/raw_response.txt").is_file())
        model_calls += int((root / "02_independent_audit/raw_response.txt").is_file())
        summary = {
            "schema_version": "creative_structure_run_summary_v1",
            "status": "BLOCKED",
            "reason": "pipeline_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_calls": model_calls,
            "retry_count": 0,
        }
        _write_json(root / "run_summary.json", summary)
        raise StructureExtractionError(str(exc)) from exc
