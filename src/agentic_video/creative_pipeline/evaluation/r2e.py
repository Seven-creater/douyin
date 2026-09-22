"""R2-E evaluation contracts and a metadata-only fake evaluator."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from src.agentic_video.creative_pipeline.contracts import (
    ContractError, artifact_ref,
)
from src.agentic_video.creative_pipeline.generation.image import (
    validate_image_candidate_schema,
)
from src.agentic_video.creative_pipeline.generation.video import (
    validate_video_candidate_schema,
)


STRUCTURE_FIELDS = {
    "schema_version", "evaluation_id", "candidate_ref",
    "shot_contract_ref", "expected_trace", "declared_trace", "checks",
    "deterministic_status", "semantic_realization", "status",
}
VISUAL_FIELDS = {
    "schema_version", "evaluation_id", "candidate_ref",
    "shot_contract_ref", "asset_graph_ref", "expected_asset_refs",
    "declared_asset_refs", "checks", "dimensions", "status",
}
SCORECARD_FIELDS = {
    "schema_version", "scorecard_id", "candidate_ref", "experiment_ref",
    "structure_evaluation_ref", "visual_consistency_ref",
    "shot_contract_validation", "quality_dimensions",
    "deterministic_status", "overall_status", "selection_eligible",
}
STRUCTURE_CHECKS = (
    "trace_binding", "role_binding", "relation_binding",
)
VISUAL_CHECKS = ("asset_graph_binding", "asset_reference_binding")
SHOT_CHECKS = (
    "candidate_schema", "shot_id_binding", "shot_contract_sha_binding",
    "asset_graph_sha_binding", "parent_sha_binding",
)
VISUAL_DIMENSIONS = ("identity", "environment", "object", "temporal")
QUALITY_DIMENSIONS = (
    "motion", "artifact", "camera", "realism", "coherence",
    "emotional_effect", "originality",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def _validate_ref(value: object, reason: str) -> None:
    try:
        if not isinstance(value, dict):
            raise ContractError(reason)
        artifact_ref(value.get("artifact_id"), value.get("sha"))
    except ContractError:
        raise ContractError(reason) from None


def _check(name: str, assertion: Callable[[], bool],
           reason_code: str) -> dict[str, Any]:
    try:
        passed = bool(assertion())
    except (ContractError, KeyError, TypeError, ValueError):
        passed = False
    return {
        "check": name,
        "passed": passed,
        "reason_code": None if passed else reason_code,
    }


def _validate_checks(value: object, names: tuple[str, ...], reason: str
                     ) -> None:
    _require(isinstance(value, list)
             and [row.get("check") for row in value] == list(names), reason)
    for row in value:
        _require(set(row) == {"check", "passed", "reason_code"}
                 and isinstance(row["passed"], bool)
                 and (row["reason_code"] is None
                      or isinstance(row["reason_code"], str)), reason)


def _status(checks: list[dict[str, Any]]) -> str:
    return "PASS" if all(row["passed"] for row in checks) else "FAIL"


class FakeGenerationEvaluator:
    """Evaluate lineage declarations while leaving media judgments not run."""

    def evaluate_structure(
            self, *, candidate: dict[str, Any],
            candidate_ref: dict[str, str], shot_contract: dict[str, Any],
            shot_contract_ref: dict[str, str]) -> dict[str, Any]:
        expected = shot_contract["structure_trace"]
        declared = candidate["structure_trace"]
        checks = [
            _check("trace_binding", lambda: declared == expected,
                   "structure_trace_binding_mismatch"),
            _check("role_binding", lambda: declared["structure_roles"]
                   == expected["structure_roles"],
                   "structure_role_binding_mismatch"),
            _check("relation_binding", lambda: declared["relation_ids"]
                   == expected["relation_ids"],
                   "structure_relation_binding_mismatch"),
        ]
        deterministic = _status(checks)
        value = {
            "schema_version": "structure_evaluation_v1",
            "evaluation_id": f"STRUCTURE_{candidate['candidate_id']}",
            "candidate_ref": dict(candidate_ref),
            "shot_contract_ref": dict(shot_contract_ref),
            "expected_trace": deepcopy(expected),
            "declared_trace": deepcopy(declared),
            "checks": checks,
            "deterministic_status": deterministic,
            "semantic_realization": "not_run",
            "status": "PARTIAL" if deterministic == "PASS" else "FAIL",
        }
        validate_structure_evaluation(value)
        return value

    def evaluate_visual_consistency(
            self, *, candidate: dict[str, Any],
            candidate_ref: dict[str, str], shot_contract: dict[str, Any],
            shot_contract_ref: dict[str, str],
            asset_graph_ref: dict[str, str]) -> dict[str, Any]:
        checks = [
            _check("asset_graph_binding", lambda: (
                candidate["asset_graph_sha"] == asset_graph_ref["sha"]
                == shot_contract["asset_graph_sha"]),
                "visual_asset_graph_binding_mismatch"),
            _check("asset_reference_binding", lambda: (
                candidate["asset_refs"] == shot_contract["asset_refs"]),
                "visual_asset_reference_binding_mismatch"),
        ]
        deterministic = _status(checks)
        value = {
            "schema_version": "visual_consistency_evaluation_v1",
            "evaluation_id": f"VISUAL_{candidate['candidate_id']}",
            "candidate_ref": dict(candidate_ref),
            "shot_contract_ref": dict(shot_contract_ref),
            "asset_graph_ref": dict(asset_graph_ref),
            "expected_asset_refs": deepcopy(shot_contract["asset_refs"]),
            "declared_asset_refs": deepcopy(candidate["asset_refs"]),
            "checks": checks,
            "dimensions": {name: "not_run" for name in VISUAL_DIMENSIONS},
            "status": "PARTIAL" if deterministic == "PASS" else "FAIL",
        }
        validate_visual_consistency_evaluation(value)
        return value

    def build_quality_scorecard(
            self, *, candidate: dict[str, Any],
            candidate_ref: dict[str, str], shot_contract: dict[str, Any],
            shot_contract_ref: dict[str, str],
            asset_graph_ref: dict[str, str], experiment_ref: dict[str, str],
            structure_evaluation: dict[str, Any],
            structure_evaluation_ref: dict[str, str],
            visual_consistency: dict[str, Any],
            visual_consistency_ref: dict[str, str]) -> dict[str, Any]:
        schema_validator = (validate_image_candidate_schema
                            if candidate.get("media_kind") == "image"
                            else validate_video_candidate_schema)

        def schema_ok() -> bool:
            schema_validator(candidate)
            return True

        parent_refs = candidate.get("parent_refs") or []
        checks = [
            _check("candidate_schema", schema_ok,
                   "generation_candidate_schema_invalid"),
            _check("shot_id_binding", lambda: (
                candidate["shot_id"] == shot_contract["shot_id"]),
                "generation_shot_id_binding_mismatch"),
            _check("shot_contract_sha_binding", lambda: (
                candidate["shot_contract_sha"] == shot_contract_ref["sha"]),
                "generation_shot_contract_sha_mismatch"),
            _check("asset_graph_sha_binding", lambda: (
                candidate["asset_graph_sha"] == asset_graph_ref["sha"]),
                "generation_asset_graph_sha_mismatch"),
            _check("parent_sha_binding", lambda: (
                shot_contract_ref in parent_refs
                and asset_graph_ref in parent_refs),
                "generation_parent_sha_mismatch"),
        ]
        deterministic = _status(checks)
        upstream_ok = (structure_evaluation["status"] == "PARTIAL"
                       and visual_consistency["status"] == "PARTIAL")
        value = {
            "schema_version": "generation_quality_scorecard_v1",
            "scorecard_id": f"SCORE_{candidate['candidate_id']}",
            "candidate_ref": dict(candidate_ref),
            "experiment_ref": dict(experiment_ref),
            "structure_evaluation_ref": dict(structure_evaluation_ref),
            "visual_consistency_ref": dict(visual_consistency_ref),
            "shot_contract_validation": {
                "status": deterministic, "checks": checks},
            "quality_dimensions": {
                name: "not_run" for name in QUALITY_DIMENSIONS},
            "deterministic_status": (
                "PASS" if deterministic == "PASS" and upstream_ok
                else "FAIL"),
            "overall_status": (
                "INCOMPLETE" if deterministic == "PASS" and upstream_ok
                else "FAIL"),
            "selection_eligible": False,
        }
        validate_generation_quality_scorecard(value)
        return value


def validate_structure_evaluation(value: dict[str, Any]) -> None:
    _require(isinstance(value, dict) and set(value) == STRUCTURE_FIELDS,
             "structure_evaluation_keys_invalid")
    _require(value["schema_version"] == "structure_evaluation_v1",
             "structure_evaluation_schema_invalid")
    _validate_ref(value["candidate_ref"],
                  "structure_evaluation_candidate_ref_invalid")
    _validate_ref(value["shot_contract_ref"],
                  "structure_evaluation_contract_ref_invalid")
    _validate_checks(value["checks"], STRUCTURE_CHECKS,
                     "structure_evaluation_checks_invalid")
    deterministic = _status(value["checks"])
    _require(value["deterministic_status"] == deterministic,
             "structure_evaluation_status_mismatch")
    _require(value["semantic_realization"] == "not_run"
             and value["status"]
             == ("PARTIAL" if deterministic == "PASS" else "FAIL"),
             "structure_evaluation_scope_invalid")


def validate_visual_consistency_evaluation(value: dict[str, Any]) -> None:
    _require(isinstance(value, dict) and set(value) == VISUAL_FIELDS,
             "visual_consistency_keys_invalid")
    _require(value["schema_version"]
             == "visual_consistency_evaluation_v1",
             "visual_consistency_schema_invalid")
    for key in ("candidate_ref", "shot_contract_ref", "asset_graph_ref"):
        _validate_ref(value[key], "visual_consistency_ref_invalid")
    _validate_checks(value["checks"], VISUAL_CHECKS,
                     "visual_consistency_checks_invalid")
    _require(isinstance(value["dimensions"], dict)
             and set(value["dimensions"]) == set(VISUAL_DIMENSIONS)
             and set(value["dimensions"].values()) == {"not_run"},
             "visual_consistency_dimensions_invalid")
    deterministic = _status(value["checks"])
    _require(value["status"]
             == ("PARTIAL" if deterministic == "PASS" else "FAIL"),
             "visual_consistency_status_mismatch")


def validate_generation_quality_scorecard(value: dict[str, Any]) -> None:
    _require(isinstance(value, dict) and set(value) == SCORECARD_FIELDS,
             "generation_scorecard_keys_invalid")
    _require(value["schema_version"] == "generation_quality_scorecard_v1",
             "generation_scorecard_schema_invalid")
    for key in ("candidate_ref", "experiment_ref",
                "structure_evaluation_ref", "visual_consistency_ref"):
        _validate_ref(value[key], "generation_scorecard_ref_invalid")
    validation = value["shot_contract_validation"]
    _require(isinstance(validation, dict)
             and set(validation) == {"status", "checks"},
             "generation_scorecard_contract_validation_invalid")
    _validate_checks(validation["checks"], SHOT_CHECKS,
                     "generation_scorecard_checks_invalid")
    _require(validation["status"] == _status(validation["checks"]),
             "generation_scorecard_check_status_mismatch")
    _require(isinstance(value["quality_dimensions"], dict)
             and set(value["quality_dimensions"]) == set(QUALITY_DIMENSIONS)
             and set(value["quality_dimensions"].values()) == {"not_run"},
             "generation_scorecard_dimensions_invalid")
    _require(value["deterministic_status"] in {"PASS", "FAIL"}
             and value["overall_status"] in {"INCOMPLETE", "FAIL"}
             and value["selection_eligible"] is False,
             "generation_scorecard_scope_invalid")
    if value["deterministic_status"] == "PASS":
        _require(value["overall_status"] == "INCOMPLETE",
                 "generation_scorecard_overall_status_mismatch")
    else:
        _require(value["overall_status"] == "FAIL",
                 "generation_scorecard_overall_status_mismatch")
