"""Generation request contracts compiled from Wave 4 production artifacts."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.production.asset_validator import (
    validate_production_boundary,
)
from src.agentic_video.manifest import json_hash


SHOT_CONTRACT_FIELDS = {
    "schema_version", "contract_id", "shot_id", "source_shot_sha",
    "source_board_sha", "shot_plan_sha", "storyboard_sha",
    "asset_graph_sha", "duration_budget_s", "characters", "environment",
    "camera", "actions", "frame_description", "lighting", "composition",
    "asset_refs", "structure_trace",
}
ASSET_REF_FIELDS = {"asset_id", "asset_sha"}
GENERATION_REQUEST_BASE_FIELDS = {
    "schema_version", "request_id", "media_kind", "candidate_count",
    "shot_contract_ref", "asset_graph_ref", "shot_contract",
}
VIDEO_REQUEST_EXTRA_FIELDS = {
    "source_image_ref", "source_image_selection_ref", "source_image",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def _artifact_ref(value: object) -> bool:
    return (isinstance(value, dict) and set(value) == {"artifact_id", "sha"}
            and isinstance(value["artifact_id"], str)
            and bool(value["artifact_id"].strip())
            and isinstance(value["sha"], str) and len(value["sha"]) == 64)


def _asset_refs(asset_graph: dict[str, Any], asset_ids: list[str]
                ) -> list[dict[str, str]]:
    assets = {row["asset_id"]: row for row in asset_graph["assets"]}
    return [{"asset_id": asset_id, "asset_sha": json_hash(assets[asset_id])}
            for asset_id in asset_ids]


def build_shot_contract(*, shot: dict[str, Any], board: dict[str, Any],
                        asset_graph: dict[str, Any], shot_plan_sha: str,
                        storyboard_sha: str,
                        asset_graph_sha: str) -> dict[str, Any]:
    """Compile only already-approved shot, board, and asset information."""
    contract = {
        "schema_version": "generation_shot_contract_v1",
        "contract_id": f"GEN_CONTRACT_{shot['shot_id']}",
        "shot_id": shot["shot_id"],
        "source_shot_sha": json_hash(shot),
        "source_board_sha": json_hash(board),
        "shot_plan_sha": shot_plan_sha,
        "storyboard_sha": storyboard_sha,
        "asset_graph_sha": asset_graph_sha,
        "duration_budget_s": deepcopy(shot["duration_budget_s"]),
        "characters": list(shot["characters"]),
        "environment": shot["location"],
        "camera": deepcopy(shot["camera"]),
        "actions": list(shot["visual_events"]),
        "frame_description": board["frame_description"],
        "lighting": board["lighting"],
        "composition": board["composition"],
        "asset_refs": _asset_refs(
            asset_graph, list(board["reference_asset_ids"])),
        "structure_trace": deepcopy(shot["structure_trace"]),
    }
    validate_shot_contract(
        contract, shot=shot, board=board, asset_graph=asset_graph,
        shot_plan_sha=shot_plan_sha, storyboard_sha=storyboard_sha,
        asset_graph_sha=asset_graph_sha)
    return contract


def validate_shot_contract_schema(value: dict[str, Any]) -> None:
    validate_production_boundary(value)
    _require(isinstance(value, dict) and set(value) == SHOT_CONTRACT_FIELDS,
             "generation_shot_contract_keys_invalid")
    _require(value["schema_version"] == "generation_shot_contract_v1",
             "generation_shot_contract_schema_invalid")
    for key in ("contract_id", "shot_id", "source_shot_sha",
                "source_board_sha", "shot_plan_sha", "storyboard_sha",
                "asset_graph_sha", "environment", "frame_description",
                "lighting", "composition"):
        _require(isinstance(value[key], str) and bool(value[key].strip()),
                 f"generation_shot_contract_{key}_invalid")
    _require(isinstance(value["duration_budget_s"], list)
             and len(value["duration_budget_s"]) == 2
             and all(isinstance(item, (int, float)) and item > 0
                     for item in value["duration_budget_s"]),
             "generation_shot_contract_duration_invalid")
    _require(isinstance(value["characters"], list)
             and all(isinstance(item, str) and item
                     for item in value["characters"]),
             "generation_shot_contract_characters_invalid")
    _require(isinstance(value["camera"], dict)
             and isinstance(value["actions"], list)
             and all(isinstance(item, str) and item
                     for item in value["actions"]),
             "generation_shot_contract_content_invalid")
    refs = value["asset_refs"]
    _require(isinstance(refs, list) and bool(refs),
             "generation_shot_contract_asset_refs_invalid")
    _require(all(isinstance(row, dict) and set(row) == ASSET_REF_FIELDS
                 and isinstance(row["asset_id"], str) and row["asset_id"]
                 and isinstance(row["asset_sha"], str)
                 and len(row["asset_sha"]) == 64 for row in refs),
             "generation_shot_contract_asset_refs_invalid")
    _require(len({row["asset_id"] for row in refs}) == len(refs),
             "generation_shot_contract_asset_refs_duplicate")
    trace = value["structure_trace"]
    _require(isinstance(trace, dict)
             and set(trace) == {
                 "source_beat_id", "structure_roles", "relation_ids"}
             and isinstance(trace["source_beat_id"], str)
             and isinstance(trace["structure_roles"], list)
             and isinstance(trace["relation_ids"], list),
             "generation_shot_contract_trace_invalid")


def validate_shot_contract(value: dict[str, Any], *,
                           shot: dict[str, Any], board: dict[str, Any],
                           asset_graph: dict[str, Any], shot_plan_sha: str,
                           storyboard_sha: str,
                           asset_graph_sha: str) -> None:
    validate_shot_contract_schema(value)
    _require(value["shot_id"] == shot["shot_id"] == board["shot_id"],
             "generation_shot_contract_shot_id_mismatch")
    _require(value["source_shot_sha"] == json_hash(shot)
             and value["source_board_sha"] == json_hash(board),
             "generation_shot_contract_source_sha_mismatch")
    _require(value["shot_plan_sha"] == shot_plan_sha
             and value["storyboard_sha"] == storyboard_sha
             and value["asset_graph_sha"] == asset_graph_sha,
             "generation_shot_contract_parent_sha_mismatch")
    expected_ids = list(board["reference_asset_ids"])
    _require(value["asset_refs"] == _asset_refs(asset_graph, expected_ids),
             "generation_shot_contract_asset_sha_mismatch")
    _require(value["structure_trace"] == shot["structure_trace"],
             "generation_shot_contract_trace_mismatch")


def build_image_generation_request(
        *, shot_contract: dict[str, Any], shot_contract_ref: dict[str, str],
        asset_graph_ref: dict[str, str], candidate_count: int = 2
        ) -> dict[str, Any]:
    value = {
        "schema_version": "image_generation_request_v1",
        "request_id": f"IMG_REQ_{shot_contract['shot_id']}",
        "media_kind": "image",
        "candidate_count": candidate_count,
        "shot_contract_ref": dict(shot_contract_ref),
        "asset_graph_ref": dict(asset_graph_ref),
        "shot_contract": deepcopy(shot_contract),
    }
    validate_generation_request(value)
    return value


def build_video_generation_request(
        *, shot_contract: dict[str, Any], shot_contract_ref: dict[str, str],
        asset_graph_ref: dict[str, str], source_image: dict[str, Any],
        source_image_ref: dict[str, str],
        source_image_selection_ref: dict[str, str], candidate_count: int = 2
        ) -> dict[str, Any]:
    value = {
        "schema_version": "video_generation_request_v1",
        "request_id": f"VID_REQ_{shot_contract['shot_id']}",
        "media_kind": "video",
        "candidate_count": candidate_count,
        "shot_contract_ref": dict(shot_contract_ref),
        "asset_graph_ref": dict(asset_graph_ref),
        "shot_contract": deepcopy(shot_contract),
        "source_image_ref": dict(source_image_ref),
        "source_image_selection_ref": dict(source_image_selection_ref),
        "source_image": deepcopy(source_image),
    }
    validate_generation_request(value)
    return value


def validate_generation_request(value: dict[str, Any]) -> None:
    validate_production_boundary(value)
    _require(isinstance(value, dict), "generation_request_invalid")
    kind = value.get("media_kind")
    expected = (GENERATION_REQUEST_BASE_FIELDS
                if kind == "image"
                else GENERATION_REQUEST_BASE_FIELDS | VIDEO_REQUEST_EXTRA_FIELDS)
    _require(kind in {"image", "video"} and set(value) == expected,
             "generation_request_keys_invalid")
    _require(value["schema_version"] == f"{kind}_generation_request_v1",
             "generation_request_schema_invalid")
    _require(isinstance(value["request_id"], str)
             and bool(value["request_id"].strip()),
             "generation_request_id_invalid")
    _require(isinstance(value["candidate_count"], int)
             and 1 <= value["candidate_count"] <= 8,
             "generation_candidate_count_invalid")
    _require(_artifact_ref(value["shot_contract_ref"])
             and _artifact_ref(value["asset_graph_ref"]),
             "generation_request_parent_ref_invalid")
    validate_shot_contract_schema(value["shot_contract"])
    _require(value["asset_graph_ref"]["sha"]
             == value["shot_contract"]["asset_graph_sha"],
             "generation_request_parent_sha_mismatch")
    if kind == "video":
        _require(_artifact_ref(value["source_image_ref"])
                 and _artifact_ref(value["source_image_selection_ref"]),
                 "generation_request_image_ref_invalid")
        from src.agentic_video.creative_pipeline.generation.image import (
            validate_image_candidate_schema,
        )
        validate_image_candidate_schema(value["source_image"])
        image = value["source_image"]
        _require(value["source_image_ref"]["artifact_id"]
                 == f"creative:image_candidate:{image['candidate_id']}"
                 and value["source_image_selection_ref"]["artifact_id"]
                 == f"creative:image_selection:{image['shot_id']}",
                 "generation_request_image_artifact_invalid")
        _require(image["shot_id"] == value["shot_contract"]["shot_id"]
                 and image["shot_contract_sha"]
                 == value["shot_contract_ref"]["sha"]
                 and image["asset_graph_sha"]
                 == value["asset_graph_ref"]["sha"]
                 and image["structure_trace"]
                 == value["shot_contract"]["structure_trace"],
                 "generation_request_image_binding_mismatch")


def request_parent_refs(value: dict[str, Any]) -> list[dict[str, str]]:
    validate_generation_request(value)
    refs = [dict(value["shot_contract_ref"]), dict(value["asset_graph_ref"])]
    if value["media_kind"] == "video":
        refs.extend([dict(value["source_image_ref"]),
                     dict(value["source_image_selection_ref"])])
    return refs


class ImageGeneratorAdapter(Protocol):
    def generate(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        ...


class VideoGeneratorAdapter(Protocol):
    def generate(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        ...
