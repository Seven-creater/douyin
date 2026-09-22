"""Video candidate schema and deterministic fake generator."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.generation.adapter import (
    request_parent_refs, validate_generation_request,
)
from src.agentic_video.creative_pipeline.generation.image import (
    BACKEND_FIELDS, PLACEHOLDER_FIELDS, placeholder_content_sha,
)
from src.agentic_video.creative_pipeline.production.asset_validator import (
    validate_production_boundary,
)
from src.agentic_video.manifest import json_hash


VIDEO_CANDIDATE_FIELDS = {
    "schema_version", "candidate_id", "media_kind", "shot_id",
    "request_sha", "shot_contract_sha", "asset_graph_sha", "parent_refs",
    "asset_refs", "source_image_ref", "duration_s", "backend",
    "placeholder", "structure_trace",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def validate_video_candidate_schema(value: dict[str, Any]) -> None:
    validate_production_boundary(value)
    _require(isinstance(value, dict) and set(value) == VIDEO_CANDIDATE_FIELDS,
             "video_candidate_keys_invalid")
    _require(value["schema_version"] == "video_candidate_v1"
             and value["media_kind"] == "video",
             "video_candidate_schema_invalid")
    for key in ("candidate_id", "shot_id", "request_sha",
                "shot_contract_sha", "asset_graph_sha"):
        _require(isinstance(value[key], str) and bool(value[key].strip()),
                 f"video_candidate_{key}_invalid")
    _require(isinstance(value["parent_refs"], list)
             and all(isinstance(row, dict)
                     and set(row) == {"artifact_id", "sha"}
                     for row in value["parent_refs"]),
             "video_candidate_parent_refs_invalid")
    _require(isinstance(value["asset_refs"], list)
             and isinstance(value["source_image_ref"], dict)
             and set(value["source_image_ref"]) == {"artifact_id", "sha"},
             "video_candidate_refs_invalid")
    _require(isinstance(value["duration_s"], (int, float))
             and value["duration_s"] > 0,
             "video_candidate_duration_invalid")
    backend = value["backend"]
    _require(isinstance(backend, dict) and set(backend) == BACKEND_FIELDS
             and backend["adapter_id"] == "fake_video_generator"
             and backend["model_id"] == "none"
             and isinstance(backend["seed"], int),
             "video_candidate_backend_invalid")
    placeholder = value["placeholder"]
    _require(isinstance(placeholder, dict)
             and set(placeholder) == PLACEHOLDER_FIELDS
             and placeholder["media_type"] == "video"
             and placeholder["materialization"] == "metadata_only"
             and placeholder["generated"] is False
             and placeholder["content_sha"] == placeholder_content_sha(
                 media_kind="video", request_sha=value["request_sha"],
                 candidate_id=value["candidate_id"], seed=backend["seed"]),
             "video_candidate_placeholder_invalid")
    _require(isinstance(value["structure_trace"], dict),
             "video_candidate_trace_invalid")


def validate_video_candidate(value: dict[str, Any],
                             request: dict[str, Any]) -> None:
    validate_video_candidate_schema(value)
    validate_generation_request(request)
    contract = request["shot_contract"]
    _require(request["media_kind"] == "video",
             "video_candidate_request_kind_invalid")
    _require(value["request_sha"] == json_hash(request)
             and value["shot_id"] == contract["shot_id"]
             and value["shot_contract_sha"]
             == request["shot_contract_ref"]["sha"],
             "video_candidate_shot_contract_mismatch")
    _require(value["asset_graph_sha"] == request["asset_graph_ref"]["sha"]
             and value["asset_refs"] == contract["asset_refs"],
             "video_candidate_asset_sha_mismatch")
    _require(value["parent_refs"] == request_parent_refs(request),
             "video_candidate_parent_sha_mismatch")
    _require(value["source_image_ref"] == request["source_image_ref"],
             "video_candidate_source_image_mismatch")
    _require(value["structure_trace"] == contract["structure_trace"],
             "video_candidate_structure_trace_mismatch")
    _require(contract["duration_budget_s"][0] <= value["duration_s"]
             <= contract["duration_budget_s"][1],
             "video_candidate_duration_out_of_contract")


class FakeVideoGenerator:
    """Return metadata-only candidates; no image/video service is invoked."""

    def generate(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        validate_generation_request(request)
        if request["media_kind"] != "video":
            raise ContractError("fake_video_request_kind_invalid")
        contract = request["shot_contract"]
        request_sha = json_hash(request)
        duration = sum(contract["duration_budget_s"]) / 2
        rows = []
        for index in range(1, request["candidate_count"] + 1):
            candidate_id = f"VID_{contract['shot_id']}_{index:02d}"
            seed = index
            row = {
                "schema_version": "video_candidate_v1",
                "candidate_id": candidate_id,
                "media_kind": "video",
                "shot_id": contract["shot_id"],
                "request_sha": request_sha,
                "shot_contract_sha": request["shot_contract_ref"]["sha"],
                "asset_graph_sha": request["asset_graph_ref"]["sha"],
                "parent_refs": request_parent_refs(request),
                "asset_refs": deepcopy(contract["asset_refs"]),
                "source_image_ref": dict(request["source_image_ref"]),
                "duration_s": duration,
                "backend": {
                    "adapter_id": "fake_video_generator",
                    "adapter_version": "1.0.0", "model_id": "none",
                    "seed": seed,
                },
                "placeholder": {
                    "media_type": "video",
                    "materialization": "metadata_only",
                    "generated": False,
                    "content_sha": placeholder_content_sha(
                        media_kind="video", request_sha=request_sha,
                        candidate_id=candidate_id, seed=seed),
                },
                "structure_trace": deepcopy(contract["structure_trace"]),
            }
            validate_video_candidate(row, request)
            rows.append(row)
        return rows
