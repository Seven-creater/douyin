"""Independent one-shot audit and deterministic R2-C.2 publish dry-run."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.agentic_video.creative_dna_v3.abstraction_prompt import (
    AUDIT_MAX_NEW_TOKENS,
    INDEPENDENT_AUDIT_PROMPT,
)
from src.agentic_video.creative_dna_v3.extractor import (
    DNAExtractionError,
    _parse_object,
    _write_json,
    extract_creative_dna,
)
from src.agentic_video.creative_dna_v3.publisher import publish_creative_spec
from src.agentic_video.creative_dna_v3.schema import (
    ANTI_INVARIANT_CATEGORIES,
)
from src.agentic_video.creative_dna_v3.validators import validate_dna_audit
from src.agentic_video.manifest import json_hash

AUDIT_SCHEMA_VERSION = "creative_dna_independent_audit_v1"
_ITEM_TYPES = {
    "node", "edge", "experience_state", "event_constraint",
    "editing_constraint", "free_slot",
}


class DNAAuditError(RuntimeError):
    pass


def _candidate_items(candidate: dict[str, Any]) -> dict[str, str]:
    graph = candidate.get("mechanism_graph") or {"nodes": [], "edges": []}
    rows = [
        *((row["node_id"], "node") for row in graph["nodes"]),
        *((row["edge_id"], "edge") for row in graph["edges"]),
        *((row["state_id"], "experience_state")
          for row in candidate["experience_arc"]),
        *((row["constraint_id"], "event_constraint")
          for row in candidate["event_constraints"]),
        *((row["constraint_id"], "editing_constraint")
          for row in candidate["editing_constraints"]),
        *((row["slot_id"], "free_slot") for row in candidate["free_slots"]),
    ]
    result: dict[str, str] = {}
    for item_id, item_type in rows:
        if item_id in result:
            raise DNAAuditError(f"candidate_item_id_duplicate:{item_id}")
        result[item_id] = item_type
    return result


def validate_independent_audit(value: dict[str, Any],
                               candidate: dict[str, Any]) -> bool:
    expected_keys = {
        "schema_version", "item_checks", "leakage_findings",
        "unsupported_dimensions_confirmed", "overall", "limitations",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise DNAAuditError("independent_audit_shape_invalid")
    if value["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise DNAAuditError("independent_audit_schema_invalid")
    expected = _candidate_items(candidate)
    actual: dict[str, dict[str, Any]] = {}
    for check in value["item_checks"]:
        if not isinstance(check, dict) or set(check) != {
            "item_id", "item_type", "grounding_status", "abstraction_valid",
            "constraint_valid", "reason",
        }:
            raise DNAAuditError("independent_item_check_invalid")
        item_id = str(check["item_id"])
        item_type = str(check["item_type"])
        if item_type not in _ITEM_TYPES or expected.get(item_id) != item_type:
            raise DNAAuditError(f"independent_item_unknown:{item_id}")
        if item_id in actual:
            raise DNAAuditError(f"independent_item_duplicate:{item_id}")
        if check["grounding_status"] not in {
            "supported", "unsupported", "not_applicable"
        }:
            raise DNAAuditError(f"grounding_status_invalid:{item_id}")
        if check["grounding_status"] == "not_applicable" and item_type != "free_slot":
            raise DNAAuditError(f"grounding_not_applicable_invalid:{item_id}")
        if not isinstance(check["abstraction_valid"], bool) \
                or not isinstance(check["constraint_valid"], bool) \
                or not str(check["reason"]).strip():
            raise DNAAuditError(f"independent_item_fields_invalid:{item_id}")
        actual[item_id] = check
    if set(actual) != set(expected):
        raise DNAAuditError("independent_item_coverage_invalid")

    findings = value["leakage_findings"]
    if not isinstance(findings, list):
        raise DNAAuditError("leakage_findings_invalid")
    finding_ids: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {
            "finding_id", "item_id", "field", "leaked_surface", "category",
            "reason",
        }:
            raise DNAAuditError("leakage_finding_invalid")
        finding_id = str(finding["finding_id"])
        if finding_id in finding_ids or str(finding["item_id"]) not in expected:
            raise DNAAuditError(f"leakage_finding_reference_invalid:{finding_id}")
        finding_ids.add(finding_id)
        if finding["category"] not in ANTI_INVARIANT_CATEGORIES:
            raise DNAAuditError(f"leakage_category_invalid:{finding_id}")
        if not all(str(finding[key]).strip() for key in (
                "field", "leaked_surface", "reason")):
            raise DNAAuditError(f"leakage_finding_fields_invalid:{finding_id}")

    unsupported = value["unsupported_dimensions_confirmed"]
    if unsupported != candidate["unsupported_dimensions"]:
        raise DNAAuditError("unsupported_dimensions_audit_mismatch")
    overall = value["overall"]
    overall_keys = {
        "grounding_passed", "relation_entailment_passed",
        "surface_binding_confined_to_audit", "abstraction_useful", "pass",
    }
    if not isinstance(overall, dict) or set(overall) != overall_keys \
            or any(not isinstance(overall[key], bool) for key in overall_keys):
        raise DNAAuditError("independent_audit_overall_invalid")
    if not isinstance(value["limitations"], list) or any(
            not isinstance(item, str) for item in value["limitations"]):
        raise DNAAuditError("independent_audit_limitations_invalid")

    items_pass = all(
        check["grounding_status"] in {"supported", "not_applicable"}
        and check["abstraction_valid"] and check["constraint_valid"]
        for check in actual.values()
    )
    computed_pass = (
        candidate["dna_status"] != "blocked"
        and items_pass
        and not findings
        and all(overall[key] for key in overall_keys if key != "pass")
    )
    if overall["pass"] != computed_pass:
        raise DNAAuditError("independent_audit_pass_inconsistent")
    return computed_pass


def _model_metadata(answer: Any) -> dict[str, Any]:
    return {
        "input_tokens": getattr(answer, "input_tokens", None),
        "output_tokens": getattr(answer, "output_tokens", None),
        "elapsed_s": getattr(answer, "elapsed_s", None),
    }


def audit_creative_dna(runner: Any, candidate: dict[str, Any],
                       bundle: dict[str, Any], *, trace_dir: Path
                       ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Call the independent auditor exactly once; never repair the candidate."""
    stage = Path(trace_dir) / "02_independent_audit"
    stage.mkdir(parents=True, exist_ok=True)
    payload = {"r2_bundle": bundle, "candidate_creative_dna": candidate}
    prompt = INDEPENDENT_AUDIT_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    _write_json(stage / "request.json", {
        "stage": "independent_dna_audit", "attempt": 1,
        "max_new_tokens": AUDIT_MAX_NEW_TOKENS,
        "stop_after_json_object": True, "prompt": prompt, "payload": payload,
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
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc), "model_call": _model_metadata(answer),
        })
        raise
    result["candidate_sha"] = candidate["artifact_sha"]
    result["r2_bundle_sha"] = bundle["artifact_sha"]
    result["artifact_sha"] = json_hash(result)
    _write_json(stage / "audit_result.json", result)
    metadata = _model_metadata(answer)
    _write_json(stage / "validation.json", {
        "status": "PASS", "audit_pass": passed,
        "audit_result_sha": result["artifact_sha"], "model_call": metadata,
    })
    return result, passed, metadata


def _accepted_audit(candidate: dict[str, Any], audit_result: dict[str, Any]
                    ) -> dict[str, Any]:
    result = copy.deepcopy(candidate)
    overall = audit_result["overall"]
    result["validation_record"] = {
        "grounding_passed": overall["grounding_passed"],
        "relation_entailment_passed": overall["relation_entailment_passed"],
        "surface_binding_confined_to_audit": (
            overall["surface_binding_confined_to_audit"]),
        "editing_promotion_policy_passed": True,
        "publish_whitelist_passed": True,
    }
    result.pop("artifact_sha", None)
    result["artifact_sha"] = json_hash(result)
    validate_dna_audit(result)
    return result


def _manual_transfer_cases(spec: dict[str, Any]) -> dict[str, Any]:
    surfaces = (
        ("TRANSFER_A", "software engineering", "engineer"),
        ("TRANSFER_B", "professional cooking", "chef"),
        ("TRANSFER_C", "education", "teacher"),
    )
    questions = [
        "Can every required constraint be instantiated without reference details?",
        "Does the binding preserve relations while changing people, domain, and events?",
        "Does the result avoid adding a mechanism absent from the Creative Spec?",
    ]
    return {
        "schema_version": "creative_spec_manual_transfer_review_v1",
        "spec_sha": spec["artifact_sha"],
        "status": "PENDING_HUMAN_REVIEW",
        "cases": [
            {
                "case_id": case_id,
                "surface_profile": {"domain": domain, "primary_role": role},
                "review_questions": questions,
                "decision": None,
            }
            for case_id, domain, role in surfaces
        ],
    }


def run_r2c2(extraction_runner: Any, audit_runner: Any,
              narrative: dict[str, Any], editing: dict[str, Any], *,
              trace_dir: Path, spec_id: str) -> dict[str, Any]:
    """Run one extraction, one audit, then a deterministic publish dry-run."""
    root = Path(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        candidate, bundle, extraction_meta = extract_creative_dna(
            extraction_runner, narrative, editing, trace_dir=root)
        audit_result, passed, audit_meta = audit_creative_dna(
            audit_runner, candidate, bundle, trace_dir=root)
        if not passed:
            summary = {
                "schema_version": "r2c2_run_summary_v1", "status": "BLOCKED",
                "reason": "independent_audit_failed", "model_calls": 2,
                "retry_count": 0, "candidate_sha": candidate["artifact_sha"],
                "audit_result_sha": audit_result["artifact_sha"],
                "creative_spec_sha": None, "extraction_call": extraction_meta,
                "audit_call": audit_meta,
            }
            _write_json(root / "run_summary.json", summary)
            return {**summary, "candidate": candidate,
                    "audit_result": audit_result, "creative_spec": None}

        accepted = _accepted_audit(candidate, audit_result)
        spec = publish_creative_spec(accepted, spec_id=spec_id)
        publish_dir = root / "03_publish"
        _write_json(publish_dir / "creative_dna_audit_v3.json", accepted)
        _write_json(publish_dir / "creative_spec_v1.json", spec)
        _write_json(publish_dir / "publish_validation.json", {
            "status": "PASS", "publisher": "deterministic_whitelist",
            "audit_sha": accepted["artifact_sha"],
            "creative_spec_sha": spec["artifact_sha"],
        })
        transfer = _manual_transfer_cases(spec)
        _write_json(root / "04_transferability/manual_review_cases.json", transfer)
        summary = {
            "schema_version": "r2c2_run_summary_v1", "status": "PUBLISHED",
            "reason": None, "model_calls": 2, "retry_count": 0,
            "candidate_sha": candidate["artifact_sha"],
            "accepted_audit_sha": accepted["artifact_sha"],
            "audit_result_sha": audit_result["artifact_sha"],
            "creative_spec_sha": spec["artifact_sha"],
            "extraction_call": extraction_meta, "audit_call": audit_meta,
            "transfer_review_status": transfer["status"],
        }
        _write_json(root / "run_summary.json", summary)
        return {**summary, "candidate": accepted,
                "audit_result": audit_result, "creative_spec": spec}
    except BaseException as exc:
        summary = {
            "schema_version": "r2c2_run_summary_v1", "status": "BLOCKED",
            "reason": "pipeline_error", "error_type": type(exc).__name__,
            "error": str(exc), "retry_count": 0,
        }
        _write_json(root / "run_summary.json", summary)
        raise DNAExtractionError(str(exc)) from exc
