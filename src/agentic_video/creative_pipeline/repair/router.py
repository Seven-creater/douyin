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
    "asset_graph": 0,
    "shot_contract": 1,
    "image_candidate": 2,
    "image_selection": 3,
    "video_candidate": 4,
    "video_selection": 5,
    "final_assembly": 6,
}
FAILURE_FIELDS = {
    "schema_version", "failed_artifact_ref", "stage", "reason_codes",
}
PLAN_FIELDS = {
    "schema_version", "repair_plan_id", "target_ref", "target_stage",
    "reason_codes", "action", "downstream_policy",
}


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
