"""Image candidate schema and deterministic fake generator."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.generation.adapter import (
    request_parent_refs, validate_generation_request,
)
from src.agentic_video.creative_pipeline.production.asset_validator import (
    validate_production_boundary,
)
from src.agentic_video.manifest import json_hash


IMAGE_CANDIDATE_FIELDS = {
    "schema_version", "candidate_id", "media_kind", "shot_id",
    "request_sha", "shot_contract_sha", "asset_graph_sha", "parent_refs",
    "asset_refs", "backend", "placeholder", "structure_trace",
}
BACKEND_FIELDS = {"adapter_id", "adapter_version", "model_id", "seed"}
PLACEHOLDER_FIELDS = {
    "media_type", "materialization", "generated", "content_sha",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def placeholder_content_sha(*, media_kind: str, request_sha: str,
                            candidate_id: str, seed: int) -> str:
    return json_hash({
        "backend": "fake", "media_kind": media_kind,
        "request_sha": request_sha, "candidate_id": candidate_id,
        "seed": seed,
    })


def validate_image_candidate_schema(value: dict[str, Any]) -> None:
    validate_production_boundary(value)
    _require(isinstance(value, dict) and set(value) == IMAGE_CANDIDATE_FIELDS,
             "image_candidate_keys_invalid")
    _require(value["schema_version"] == "image_candidate_v1"
             and value["media_kind"] == "image",
             "image_candidate_schema_invalid")
    for key in ("candidate_id", "shot_id", "request_sha",
                "shot_contract_sha", "asset_graph_sha"):
        _require(isinstance(value[key], str) and bool(value[key].strip()),
                 f"image_candidate_{key}_invalid")
    _require(isinstance(value["parent_refs"], list)
             and all(isinstance(row, dict)
                     and set(row) == {"artifact_id", "sha"}
                     for row in value["parent_refs"]),
             "image_candidate_parent_refs_invalid")
    _require(isinstance(value["asset_refs"], list),
             "image_candidate_asset_refs_invalid")
    backend = value["backend"]
    is_fake = (isinstance(backend, dict) and set(backend) == BACKEND_FIELDS
               and backend["adapter_id"] == "fake_image_generator"
               and backend["model_id"] == "none")
    is_real = (isinstance(backend, dict) and set(backend) == BACKEND_FIELDS
               and backend["adapter_id"] == "real_image_adapter"
               and isinstance(backend["model_id"], str)
               and backend["model_id"] not in {"", "none"})
    _require((is_fake or is_real) and isinstance(backend["seed"], int),
             "image_candidate_backend_invalid")
    placeholder = value["placeholder"]
    fake_placeholder = (isinstance(placeholder, dict)
                        and set(placeholder) == PLACEHOLDER_FIELDS
                        and placeholder["media_type"] == "image"
                        and placeholder["materialization"] == "metadata_only"
                        and placeholder["generated"] is False
                        and placeholder["content_sha"]
                        == placeholder_content_sha(
                            media_kind="image",
                            request_sha=value["request_sha"],
                            candidate_id=value["candidate_id"],
                            seed=backend["seed"]))
    real_placeholder = (isinstance(placeholder, dict)
                        and set(placeholder) == PLACEHOLDER_FIELDS
                        and placeholder["media_type"] == "image"
                        and placeholder["materialization"]
                        == ("artifact:creative:generated_media:"
                            f"{value['candidate_id']}")
                        and placeholder["generated"] is True
                        and _sha(placeholder["content_sha"]))
    _require((is_fake and fake_placeholder) or (is_real and real_placeholder),
             "image_candidate_placeholder_invalid")
    _require(isinstance(value["structure_trace"], dict),
             "image_candidate_trace_invalid")


def validate_image_candidate(value: dict[str, Any],
                             request: dict[str, Any]) -> None:
    validate_image_candidate_schema(value)
    validate_generation_request(request)
    contract = request["shot_contract"]
    _require(request["media_kind"] == "image",
             "image_candidate_request_kind_invalid")
    _require(value["request_sha"] == json_hash(request)
             and value["shot_id"] == contract["shot_id"]
             and value["shot_contract_sha"]
             == request["shot_contract_ref"]["sha"],
             "image_candidate_shot_contract_mismatch")
    _require(value["asset_graph_sha"] == request["asset_graph_ref"]["sha"]
             and value["asset_refs"] == contract["asset_refs"],
             "image_candidate_asset_sha_mismatch")
    _require(value["parent_refs"] == request_parent_refs(request),
             "image_candidate_parent_sha_mismatch")
    _require(value["structure_trace"] == contract["structure_trace"],
             "image_candidate_structure_trace_mismatch")


class FakeImageGenerator:
    """Return metadata-only candidates; it never creates or opens media."""

    def generate(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        validate_generation_request(request)
        if request["media_kind"] != "image":
            raise ContractError("fake_image_request_kind_invalid")
        contract = request["shot_contract"]
        request_sha = json_hash(request)
        rows = []
        for index in range(1, request["candidate_count"] + 1):
            candidate_id = f"IMG_{contract['shot_id']}_{index:02d}"
            seed = index
            row = {
                "schema_version": "image_candidate_v1",
                "candidate_id": candidate_id,
                "media_kind": "image",
                "shot_id": contract["shot_id"],
                "request_sha": request_sha,
                "shot_contract_sha": request["shot_contract_ref"]["sha"],
                "asset_graph_sha": request["asset_graph_ref"]["sha"],
                "parent_refs": request_parent_refs(request),
                "asset_refs": deepcopy(contract["asset_refs"]),
                "backend": {
                    "adapter_id": "fake_image_generator",
                    "adapter_version": "1.0.0", "model_id": "none",
                    "seed": seed,
                },
                "placeholder": {
                    "media_type": "image",
                    "materialization": "metadata_only",
                    "generated": False,
                    "content_sha": placeholder_content_sha(
                        media_kind="image", request_sha=request_sha,
                        candidate_id=candidate_id, seed=seed),
                },
                "structure_trace": deepcopy(contract["structure_trace"]),
            }
            validate_image_candidate(row, request)
            rows.append(row)
        return rows
