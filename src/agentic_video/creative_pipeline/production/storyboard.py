"""Storyboard v1 is a text-only visualization contract in Wave 4."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.production.asset_validator import (
    validate_production_boundary,
)
from src.agentic_video.creative_pipeline.production.shot_validator import (
    validate_production_shot_plan,
)
from src.agentic_video.manifest import json_hash


STORYBOARD_FIELDS = {
    "schema_version", "storyboard_id", "shot_plan_sha", "asset_graph_sha",
    "boards",
}
BOARD_FIELDS = {
    "shot_id", "source_shot_sha", "frame_description", "camera",
    "lighting", "composition", "reference_asset_ids", "structure_trace",
}


class StoryboardAdapter(Protocol):
    def run(self, shot_plan: dict[str, Any], *, shot_plan_sha: str,
            asset_graph_sha: str) -> dict:
        ...


def shot_asset_ids(shot: dict[str, Any]) -> list[str]:
    values = [*shot["characters"], *shot["props"], *shot["wardrobe"],
              *shot["motion_refs"]]
    if shot["location"]:
        values.append(shot["location"])
    return list(dict.fromkeys(values))


class FakeStoryboardAdapter:
    """Create textual frame contracts without an image-generation backend."""

    def run(self, shot_plan: dict[str, Any], *, shot_plan_sha: str,
            asset_graph_sha: str) -> dict:
        return deepcopy({
            "schema_version": "storyboard_v1",
            "storyboard_id": f"STORYBOARD_{shot_plan['shot_plan_id']}",
            "shot_plan_sha": shot_plan_sha,
            "asset_graph_sha": asset_graph_sha,
            "boards": [{
                "shot_id": shot["shot_id"],
                "source_shot_sha": json_hash(shot),
                "frame_description": (
                    f"{shot['start_state']} -> {shot['end_state']}; "
                    f"events={shot['visual_events']}"),
                "camera": deepcopy(shot["camera"]),
                "lighting": "neutral planning placeholder",
                "composition": "show only the contracted shot content",
                "reference_asset_ids": shot_asset_ids(shot),
                "structure_trace": deepcopy(shot["structure_trace"]),
            } for shot in shot_plan["shots"]],
        })


def validate_storyboard(storyboard: dict[str, Any], *,
                        screenplay: dict[str, Any], screenplay_sha: str,
                        asset_graph: dict[str, Any], asset_graph_sha: str,
                        shot_plan: dict[str, Any], shot_plan_sha: str) -> None:
    validate_production_boundary(storyboard)
    validate_production_shot_plan(
        shot_plan, screenplay=screenplay, screenplay_sha=screenplay_sha,
        asset_graph=asset_graph, asset_graph_sha=asset_graph_sha)
    if not isinstance(storyboard, dict) or set(storyboard) != STORYBOARD_FIELDS:
        raise ContractError("storyboard_keys_invalid")
    if storyboard["schema_version"] != "storyboard_v1":
        raise ContractError("storyboard_schema_invalid")
    if storyboard["shot_plan_sha"] != shot_plan_sha:
        raise ContractError("storyboard_shot_plan_sha_mismatch")
    if storyboard["asset_graph_sha"] != asset_graph_sha:
        raise ContractError("storyboard_asset_graph_sha_mismatch")
    boards = storyboard["boards"]
    shots = {row["shot_id"]: row for row in shot_plan["shots"]}
    if (not isinstance(boards, list) or len(boards) != len(shots)
            or {row.get("shot_id") for row in boards} != set(shots)):
        raise ContractError("storyboard_shot_coverage")
    asset_ids = {row["asset_id"] for row in asset_graph["assets"]}
    for board in boards:
        if not isinstance(board, dict) or set(board) != BOARD_FIELDS:
            raise ContractError("storyboard_board_keys_invalid")
        shot = shots[board["shot_id"]]
        if board["source_shot_sha"] != json_hash(shot):
            raise ContractError("storyboard_source_shot_sha_mismatch")
        if (board["camera"] != shot["camera"]
                or board["structure_trace"] != shot["structure_trace"]):
            raise ContractError("storyboard_shot_contract_mismatch")
        if (board["reference_asset_ids"] != shot_asset_ids(shot)
                or not set(board["reference_asset_ids"]) <= asset_ids):
            raise ContractError("storyboard_asset_refs_mismatch")
        for key in ("frame_description", "lighting", "composition"):
            if not isinstance(board[key], str) or not board[key].strip():
                raise ContractError("storyboard_text_field_invalid")
