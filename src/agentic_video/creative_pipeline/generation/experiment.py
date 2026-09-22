"""Immutable experiment records for fake or real generation adapters."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_pipeline.contracts import (
    ContractError, artifact_ref,
)
from src.agentic_video.manifest import json_hash


EXPERIMENT_FIELDS = {
    "schema_version", "experiment_id", "media_kind",
    "output_candidate_ref", "input_artifact_refs", "model_id",
    "model_version", "model_sha", "config_sha", "prompt_sha", "seed",
    "generation_time_ms", "execution_mode",
}
EXECUTION_MODES = frozenset({"fake_metadata_only", "real_model"})
REAL_EXECUTION_FIELDS = {
    "model_id", "model_version", "model_sha", "config_sha", "prompt_sha",
    "seed", "generation_time_ms",
}


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


def _model_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    backend = candidate["backend"]
    return {
        "adapter_id": backend["adapter_id"],
        "adapter_version": backend["adapter_version"],
        "model_id": backend["model_id"],
    }


def _generation_config(candidate: dict[str, Any]) -> dict[str, Any]:
    backend = candidate["backend"]
    value = {
        "media_kind": candidate["media_kind"],
        "adapter_id": backend["adapter_id"],
        "adapter_version": backend["adapter_version"],
    }
    if candidate["media_kind"] == "video":
        value["duration_s"] = candidate["duration_s"]
    return value


def build_generation_experiment_record(
        *, candidate: dict[str, Any], candidate_ref: dict[str, str],
        generation_time_ms: int = 0) -> dict[str, Any]:
    """Record the adapter inputs without claiming a real model was called."""
    backend = candidate["backend"]
    value = {
        "schema_version": "generation_experiment_record_v1",
        "experiment_id": f"EXPERIMENT_{candidate['candidate_id']}",
        "media_kind": candidate["media_kind"],
        "output_candidate_ref": dict(candidate_ref),
        "input_artifact_refs": [dict(row)
                                for row in candidate["parent_refs"]],
        "model_id": backend["model_id"],
        "model_version": backend["adapter_version"],
        "model_sha": json_hash(_model_identity(candidate)),
        "config_sha": json_hash(_generation_config(candidate)),
        # Fake adapters have no free-form prompt. Their canonical request SHA
        # occupies the prompt provenance slot until a real adapter is used.
        "prompt_sha": candidate["request_sha"],
        "seed": backend["seed"],
        "generation_time_ms": generation_time_ms,
        "execution_mode": "fake_metadata_only",
    }
    validate_generation_experiment_record(value, candidate=candidate)
    return value


def build_real_generation_experiment_record(
        *, candidate: dict[str, Any], candidate_ref: dict[str, str],
        execution: dict[str, Any]) -> dict[str, Any]:
    """Bind a real output to its exact model, prompt, config, and inputs."""
    _require(isinstance(execution, dict)
             and set(execution) == REAL_EXECUTION_FIELDS,
             "real_generation_execution_keys_invalid")
    value = {
        "schema_version": "generation_experiment_record_v1",
        "experiment_id": f"EXPERIMENT_{candidate['candidate_id']}",
        "media_kind": candidate["media_kind"],
        "output_candidate_ref": dict(candidate_ref),
        "input_artifact_refs": [dict(row)
                                for row in candidate["parent_refs"]],
        **execution,
        "execution_mode": "real_model",
    }
    validate_generation_experiment_record(
        value, candidate=candidate, real_execution=execution)
    return value


def validate_generation_experiment_record(
        value: dict[str, Any], *,
        candidate: dict[str, Any] | None = None,
        real_execution: dict[str, Any] | None = None) -> None:
    _require(isinstance(value, dict) and set(value) == EXPERIMENT_FIELDS,
             "generation_experiment_keys_invalid")
    _require(value["schema_version"] == "generation_experiment_record_v1",
             "generation_experiment_schema_invalid")
    _require(value["media_kind"] in {"image", "video"},
             "generation_experiment_media_kind_invalid")
    for key in ("experiment_id", "model_id", "model_version"):
        _require(isinstance(value[key], str) and bool(value[key].strip()),
                 f"generation_experiment_{key}_invalid")
    for key in ("model_sha", "config_sha", "prompt_sha"):
        try:
            artifact_ref("sha_check", value[key])
        except ContractError:
            raise ContractError(
                f"generation_experiment_{key}_invalid") from None
    _validate_ref(value["output_candidate_ref"],
                  "generation_experiment_output_ref_invalid")
    refs = value["input_artifact_refs"]
    _require(isinstance(refs, list) and bool(refs),
             "generation_experiment_input_refs_invalid")
    for ref in refs:
        _validate_ref(ref, "generation_experiment_input_ref_invalid")
    _require(len({(row["artifact_id"], row["sha"]) for row in refs})
             == len(refs), "generation_experiment_input_ref_duplicate")
    _require(isinstance(value["seed"], int),
             "generation_experiment_seed_invalid")
    _require(isinstance(value["generation_time_ms"], int)
             and value["generation_time_ms"] >= 0,
             "generation_experiment_time_invalid")
    _require(value["execution_mode"] in EXECUTION_MODES,
             "generation_experiment_mode_invalid")
    if candidate is not None:
        backend = candidate["backend"]
        _require(value["media_kind"] == candidate["media_kind"]
                 and value["output_candidate_ref"]["artifact_id"]
                 == (f"creative:{candidate['media_kind']}_candidate:"
                     f"{candidate['candidate_id']}")
                 and value["input_artifact_refs"]
                 == candidate["parent_refs"]
                 and value["model_id"] == backend["model_id"]
                 and value["seed"] == backend["seed"],
                 "generation_experiment_candidate_mismatch")
        if value["execution_mode"] == "fake_metadata_only":
            _require(value["prompt_sha"] == candidate["request_sha"]
                     and value["model_version"] == backend["adapter_version"]
                     and value["model_sha"]
                     == json_hash(_model_identity(candidate))
                     and value["config_sha"]
                     == json_hash(_generation_config(candidate))
                     and backend["model_id"] == "none"
                     and candidate["placeholder"]["generated"] is False
                     and value["generation_time_ms"] == 0,
                     "generation_experiment_fake_mode_invalid")
        else:
            _require(backend["adapter_id"] in {
                "real_image_adapter", "real_video_adapter"}
                and candidate["placeholder"]["generated"] is True,
                "generation_experiment_real_mode_invalid")
            _require(real_execution is not None
                     and set(real_execution) == REAL_EXECUTION_FIELDS
                     and all(value[key] == real_execution[key]
                             for key in REAL_EXECUTION_FIELDS),
                     "generation_experiment_real_execution_mismatch")
