"""Choose the earliest failed artifact and commit a replacement version."""
from __future__ import annotations

from typing import Any, Callable

from src.agentic_video.creative_pipeline.contracts import (
    ArtifactEnvelope, ContractError, validate_artifact_envelope,
)
from src.agentic_video.creative_pipeline.evaluation.media import (
    validate_media_evaluation,
)
from src.agentic_video.manifest import json_hash
from src.agentic_video.workspace import Workspace


REPAIR_STAGE_ORDER = {
    "screenplay": 0,
    "asset_graph": 1,
    "storyboard": 2,
    "shot_contract": 3,
    "image_candidate": 4,
    "image_selection": 5,
    "video_candidate": 6,
    "video_selection": 7,
    "final_assembly": 8,
}
REPAIR_FAILURE_CLASSES = frozenset({
    "structure_failure", "asset_failure", "shot_failure",
    "generation_failure",
})
FAILURE_CLASS_TARGETS = {
    "structure_failure": frozenset({"screenplay", "storyboard"}),
    "asset_failure": frozenset({"asset_graph"}),
    "shot_failure": frozenset({"shot_contract"}),
    "generation_failure": frozenset({
        "image_candidate", "video_candidate"}),
}
FAILURE_FIELDS = {
    "schema_version", "failed_artifact_ref", "stage", "reason_codes",
}
PLAN_FIELDS = {
    "schema_version", "repair_plan_id", "target_ref", "target_stage",
    "reason_codes", "action", "downstream_policy",
}
CLASSIFICATION_FIELDS = {
    "schema_version", "classification_id", "failure_class",
    "source_evaluation_ref", "failed_artifact_ref", "repair_target_ref",
    "repair_target_stage", "reason_codes", "classifier_origin",
}


def _stage_for_artifact(artifact_id: str) -> str:
    if artifact_id.startswith("creative:screenplay:"):
        return "screenplay"
    if artifact_id == "creative:asset_graph":
        return "asset_graph"
    if artifact_id == "creative:storyboard":
        return "storyboard"
    if artifact_id.startswith("creative:shot_contract:"):
        return "shot_contract"
    if artifact_id.startswith("creative:image_candidate:"):
        return "image_candidate"
    if artifact_id.startswith("creative:video_candidate:"):
        return "video_candidate"
    raise ContractError("repair_classification_target_unknown")


def build_repair_classification(
        *, failure_class: str, source_evaluation_ref: dict[str, str],
        failed_artifact_ref: dict[str, str],
        repair_target_ref: dict[str, str],
        reason_codes: list[str]) -> dict[str, Any]:
    """Record an explicit diagnosis; do not infer semantics from score alone."""
    target_stage = _stage_for_artifact(repair_target_ref["artifact_id"])
    value = {
        "schema_version": "r2e_repair_classification_v1",
        "classification_id": f"CLASSIFY_{json_hash({
            'failure_class': failure_class,
            'source_evaluation_ref': source_evaluation_ref,
            'failed_artifact_ref': failed_artifact_ref,
            'repair_target_ref': repair_target_ref,
            'reason_codes': reason_codes,
        })[:16]}",
        "failure_class": failure_class,
        "source_evaluation_ref": dict(source_evaluation_ref),
        "failed_artifact_ref": dict(failed_artifact_ref),
        "repair_target_ref": dict(repair_target_ref),
        "repair_target_stage": target_stage,
        "reason_codes": list(reason_codes),
        "classifier_origin": "deterministic_explicit_v1",
    }
    validate_repair_classification(value)
    return value


def validate_repair_classification(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != CLASSIFICATION_FIELDS:
        raise ContractError("repair_classification_keys_invalid")
    if value["schema_version"] != "r2e_repair_classification_v1":
        raise ContractError("repair_classification_schema_invalid")
    failure_class = value["failure_class"]
    if failure_class not in REPAIR_FAILURE_CLASSES:
        raise ContractError("repair_classification_class_invalid")
    for key in ("source_evaluation_ref", "failed_artifact_ref",
                "repair_target_ref"):
        ref = value[key]
        if (not isinstance(ref, dict) or set(ref) != {"artifact_id", "sha"}
                or not isinstance(ref["artifact_id"], str)
                or not isinstance(ref["sha"], str) or len(ref["sha"]) != 64):
            raise ContractError("repair_classification_ref_invalid")
    stage = _stage_for_artifact(value["repair_target_ref"]["artifact_id"])
    if (value["repair_target_stage"] != stage
            or stage not in FAILURE_CLASS_TARGETS[failure_class]):
        raise ContractError("repair_classification_target_invalid")
    if (not isinstance(value["reason_codes"], list)
            or not value["reason_codes"]
            or not all(isinstance(item, str) and item
                       for item in value["reason_codes"])
            or value["classifier_origin"] != "deterministic_explicit_v1"):
        raise ContractError("repair_classification_content_invalid")


def failure_from_repair_classification(
        value: dict[str, Any]) -> dict[str, Any]:
    validate_repair_classification(value)
    failure = {
        "schema_version": "repair_failure_v1",
        "failed_artifact_ref": dict(value["repair_target_ref"]),
        "stage": value["repair_target_stage"],
        "reason_codes": list(value["reason_codes"]),
    }
    validate_repair_failure(failure)
    return failure


def failure_from_media_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    validate_media_evaluation(value)
    if value["status"] != "FAIL":
        raise ContractError("repair_failure_requires_failed_evaluation")
    return {
        "schema_version": "repair_failure_v1",
        "failed_artifact_ref": dict(value["candidate_ref"]),
        "stage": f"{value['media_kind']}_candidate",
        "reason_codes": [row["reason_code"] for row in value["checks"]
                         if not row["passed"]],
    }


def validate_repair_failure(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != FAILURE_FIELDS:
        raise ContractError("repair_failure_keys_invalid")
    if value["schema_version"] != "repair_failure_v1":
        raise ContractError("repair_failure_schema_invalid")
    ref = value["failed_artifact_ref"]
    if (not isinstance(ref, dict) or set(ref) != {"artifact_id", "sha"}
            or not isinstance(ref["artifact_id"], str)
            or not isinstance(ref["sha"], str) or len(ref["sha"]) != 64):
        raise ContractError("repair_failure_ref_invalid")
    if value["stage"] not in REPAIR_STAGE_ORDER:
        raise ContractError("repair_failure_stage_invalid")
    if (not isinstance(value["reason_codes"], list)
            or not value["reason_codes"]
            or not all(isinstance(item, str) and item
                       for item in value["reason_codes"])):
        raise ContractError("repair_failure_reasons_invalid")


def route_repair(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the earliest failed artifact; do not diagnose or rewrite it."""
    if not failures:
        raise ContractError("repair_failures_empty")
    for failure in failures:
        validate_repair_failure(failure)
    earliest = min(
        enumerate(failures),
        key=lambda item: (REPAIR_STAGE_ORDER[item[1]["stage"]], item[0]))[1]
    target = earliest["failed_artifact_ref"]
    result = {
        "schema_version": "repair_plan_v1",
        "repair_plan_id": f"REPAIR_{json_hash(earliest)[:16]}",
        "target_ref": dict(target),
        "target_stage": earliest["stage"],
        "reason_codes": list(earliest["reason_codes"]),
        "action": "create_new_version",
        "downstream_policy": "mark_stale",
    }
    validate_repair_plan(result)
    return result


def validate_repair_plan(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != PLAN_FIELDS:
        raise ContractError("repair_plan_keys_invalid")
    if (value["schema_version"] != "repair_plan_v1"
            or value["target_stage"] not in REPAIR_STAGE_ORDER
            or value["action"] != "create_new_version"
            or value["downstream_policy"] != "mark_stale"):
        raise ContractError("repair_plan_contract_invalid")
    validate_repair_failure({
        "schema_version": "repair_failure_v1",
        "failed_artifact_ref": value["target_ref"],
        "stage": value["target_stage"],
        "reason_codes": value["reason_codes"],
    })


def _descendants(workspace: Workspace, target: str) -> set[str]:
    reverse: dict[str, list[str]] = {}
    for child, parents in workspace.state["dependencies"].items():
        for parent in parents:
            reverse.setdefault(parent, []).append(child)
    found: set[str] = set()
    queue = [target]
    while queue:
        parent = queue.pop(0)
        for child in reverse.get(parent, []):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def commit_repair_version(
        workspace: Workspace, plan: dict[str, Any],
        replacement_payload: dict[str, Any], *,
        validator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Commit one replacement while Workspace performs recursive invalidation."""
    validate_repair_plan(plan)
    name = plan["target_ref"]["artifact_id"]
    if (workspace.effective_status(name) != "committed"
            or workspace.get_sha(name) != plan["target_ref"]["sha"]):
        raise ContractError("repair_target_not_current")
    current = workspace.read_artifact(name)
    if not isinstance(current, dict):
        raise ContractError("repair_target_missing")
    validate_artifact_envelope(current)
    if json_hash(replacement_payload) == json_hash(current["payload"]):
        raise ContractError("repair_payload_unchanged")
    validator(replacement_payload)

    existing = workspace.state["artifacts"][name]
    version = f"v{len(existing['versions']) + 1}"
    producer = dict(current["producer"])
    parents = [dict(row) for row in current["derived_from"]]
    envelope = ArtifactEnvelope(
        artifact_id=name,
        artifact_type=current["artifact_type"],
        schema_version=current["payload_schema_version"],
        version=version,
        payload=replacement_payload,
        producer=producer,
        derived_from=tuple(parents),
        policy_versions=dict(current["policy_versions"]),
        status="committed",
    ).to_dict()
    descendants = _descendants(workspace, name)
    written = workspace.write_draft(name, envelope, metadata={
        "derived_from": parents,
        "created_by": producer["skill_id"],
        "schema_version": "artifact_provenance_v1",
        "skill_id": producer["skill_id"],
        "skill_version": producer["skill_version"],
        "package_sha": producer["package_sha"],
        "prompt_or_instruction_sha": producer["prompt_or_instruction_sha"],
    })
    workspace.mark_tested(name)
    workspace.commit(name)
    stale = sorted(child for child in descendants
                   if workspace.effective_status(child) == "stale")
    return {
        "schema_version": "repair_commit_v1",
        "artifact_id": name,
        "previous_sha": plan["target_ref"]["sha"],
        "version": written,
        "sha": workspace.get_sha(name),
        "downstream_stale": stale,
    }
