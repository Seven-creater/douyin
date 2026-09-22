"""Deterministic Wave 5 candidate checks; visual judgment is out of scope."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.generation.adapter import (
    request_parent_refs, validate_generation_request,
)
from src.agentic_video.creative_pipeline.generation.image import (
    validate_image_candidate_schema,
)
from src.agentic_video.creative_pipeline.generation.video import (
    validate_video_candidate_schema,
)
from src.agentic_video.creative_structure_v1.freeze import CORE_COMPONENT_IDS
from src.agentic_video.manifest import json_hash


MEDIA_EVALUATION_FIELDS = {
    "schema_version", "evaluation_id", "media_kind", "candidate_ref",
    "shot_contract_ref", "asset_graph_ref", "status", "checks",
    "visual_quality", "creative_quality",
}
CHECK_NAMES = (
    "candidate_schema", "shot_contract", "asset_sha", "parent_sha",
    "structure_trace_coverage",
)


def _check(name: str, assertion: Callable[[], bool],
           reason: str) -> dict[str, Any]:
    try:
        passed = bool(assertion())
    except (ContractError, KeyError, TypeError, ValueError):
        passed = False
    return {
        "check": name,
        "passed": passed,
        "reason_code": None if passed else reason,
    }


def evaluate_media_candidate(*, candidate: dict[str, Any],
                             candidate_ref: dict[str, str],
                             request: dict[str, Any]) -> dict[str, Any]:
    """Evaluate contract fidelity without claiming any visual assessment."""
    validate_generation_request(request)
    kind = request["media_kind"]
    schema_validator = (validate_image_candidate_schema
                        if kind == "image" else validate_video_candidate_schema)
    contract = request["shot_contract"]

    def schema_ok() -> bool:
        schema_validator(candidate)
        return True

    rows = [
        _check("candidate_schema", schema_ok,
               f"{kind}_candidate_schema_invalid"),
        _check("shot_contract", lambda: (
            candidate["media_kind"] == kind
            and candidate["request_sha"] == json_hash(request)
            and candidate["shot_id"] == contract["shot_id"]
            and candidate["shot_contract_sha"]
            == request["shot_contract_ref"]["sha"]),
            f"{kind}_candidate_shot_contract_mismatch"),
        _check("asset_sha", lambda: (
            candidate["asset_graph_sha"] == request["asset_graph_ref"]["sha"]
            and candidate["asset_refs"] == contract["asset_refs"]),
            f"{kind}_candidate_asset_sha_mismatch"),
        _check("parent_sha", lambda: (
            candidate["parent_refs"] == request_parent_refs(request)
            and (kind != "video" or candidate["source_image_ref"]
                 == request["source_image_ref"])),
            f"{kind}_candidate_parent_sha_mismatch"),
        _check("structure_trace_coverage", lambda: (
            candidate["structure_trace"] == contract["structure_trace"]),
            f"{kind}_candidate_structure_trace_mismatch"),
    ]
    passed = all(row["passed"] for row in rows)
    result = {
        "schema_version": "media_candidate_evaluation_v1",
        "evaluation_id": f"EVAL_{candidate.get('candidate_id', 'UNKNOWN')}",
        "media_kind": kind,
        "candidate_ref": dict(candidate_ref),
        "shot_contract_ref": dict(request["shot_contract_ref"]),
        "asset_graph_ref": dict(request["asset_graph_ref"]),
        "status": "PASS" if passed else "FAIL",
        "checks": rows,
        "visual_quality": "not_run",
        "creative_quality": "not_run",
    }
    validate_media_evaluation(result)
    return result


def validate_media_evaluation(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != MEDIA_EVALUATION_FIELDS:
        raise ContractError("media_evaluation_keys_invalid")
    if value["schema_version"] != "media_candidate_evaluation_v1":
        raise ContractError("media_evaluation_schema_invalid")
    if value["media_kind"] not in {"image", "video"}:
        raise ContractError("media_evaluation_kind_invalid")
    for key in ("candidate_ref", "shot_contract_ref", "asset_graph_ref"):
        ref = value[key]
        if (not isinstance(ref, dict)
                or set(ref) != {"artifact_id", "sha"}
                or not isinstance(ref["artifact_id"], str)
                or not isinstance(ref["sha"], str)
                or len(ref["sha"]) != 64):
            raise ContractError("media_evaluation_ref_invalid")
    rows = value["checks"]
    if (not isinstance(rows, list)
            or [row.get("check") for row in rows] != list(CHECK_NAMES)):
        raise ContractError("media_evaluation_checks_invalid")
    for row in rows:
        if (set(row) != {"check", "passed", "reason_code"}
                or not isinstance(row["passed"], bool)
                or (row["reason_code"] is not None
                    and not isinstance(row["reason_code"], str))):
            raise ContractError("media_evaluation_check_invalid")
    passed = all(row["passed"] for row in rows)
    if value["status"] != ("PASS" if passed else "FAIL"):
        raise ContractError("media_evaluation_status_mismatch")
    if (value["visual_quality"] != "not_run"
            or value["creative_quality"] != "not_run"):
        raise ContractError("media_evaluation_scope_invalid")


def build_structure_coverage_report(
        *, selected_candidates: list[dict[str, Any]],
        shot_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    expected_roles = {
        role for contract in shot_contracts
        for role in contract["structure_trace"]["structure_roles"]}
    expected_relations = {
        relation for contract in shot_contracts
        for relation in contract["structure_trace"]["relation_ids"]}
    actual_roles = {
        role for candidate in selected_candidates
        for role in candidate["structure_trace"]["structure_roles"]}
    actual_relations = {
        relation for candidate in selected_candidates
        for relation in candidate["structure_trace"]["relation_ids"]}
    expected_core_roles = set(CORE_COMPONENT_IDS[:3])
    passed = (len(selected_candidates) == len(shot_contracts)
              and actual_roles == expected_roles == expected_core_roles
              and actual_relations == expected_relations
              == {"R1_INFORMATION_UPDATE"})
    result = {
        "schema_version": "selected_structure_coverage_v1",
        "selected_candidate_ids": [row["candidate_id"]
                                   for row in selected_candidates],
        "expected_roles": sorted(expected_roles),
        "actual_roles": sorted(actual_roles),
        "expected_relations": sorted(expected_relations),
        "actual_relations": sorted(actual_relations),
        "status": "PASS" if passed else "FAIL",
        "reason_codes": [] if passed else [
            "selected_structure_trace_coverage_incomplete"],
    }
    return deepcopy(result)


def validate_structure_coverage_report(value: dict[str, Any]) -> None:
    expected = {
        "schema_version", "selected_candidate_ids", "expected_roles",
        "actual_roles", "expected_relations", "actual_relations", "status",
        "reason_codes",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("selected_structure_coverage_keys_invalid")
    if value["schema_version"] != "selected_structure_coverage_v1":
        raise ContractError("selected_structure_coverage_schema_invalid")
    if value["status"] not in {"PASS", "FAIL"}:
        raise ContractError("selected_structure_coverage_status_invalid")
