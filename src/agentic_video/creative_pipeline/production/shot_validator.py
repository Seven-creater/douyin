"""Wave 4 structure inheritance layered over the existing shot validator."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.production.asset import (
    legacy_screenplay_view,
)
from src.agentic_video.creative_pipeline.production.asset_validator import (
    validate_production_asset_graph, validate_production_boundary,
)
from src.agentic_video.creative_pipeline.production.shot import (
    SHOT_FIELDS, SHOT_PLAN_FIELDS, SHOT_TRACE_FIELDS, expected_shot_trace,
)
from src.agentic_video.creative_structure_v1.freeze import CORE_COMPONENT_IDS
from src.agentic_video.storyboard.shot_schema import validate_shot_plan


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def validate_production_shot_plan(plan: dict[str, Any], *,
                                  screenplay: dict[str, Any],
                                  screenplay_sha: str,
                                  asset_graph: dict[str, Any],
                                  asset_graph_sha: str) -> None:
    validate_production_boundary(plan)
    validate_production_asset_graph(
        asset_graph, screenplay=screenplay, screenplay_sha=screenplay_sha)
    _require(isinstance(plan, dict) and set(plan) == SHOT_PLAN_FIELDS,
             "production_shot_plan_keys_invalid")
    _require(plan["schema_version"] == "shot_plan_v1",
             "production_shot_plan_schema_invalid")
    _require(isinstance(plan["shot_plan_id"], str)
             and bool(plan["shot_plan_id"].strip()),
             "production_shot_plan_id_invalid")
    _require(plan["screenplay_sha"] == screenplay_sha,
             "production_shot_screenplay_sha_mismatch")
    _require(plan["asset_graph_sha"] == asset_graph_sha,
             "production_shot_asset_sha_mismatch")
    shots = plan["shots"]
    _require(isinstance(shots, list) and bool(shots),
             "production_shots_empty")
    roles = set()
    relations = set()
    for shot in shots:
        _require(isinstance(shot, dict) and set(shot) == SHOT_FIELDS,
                 "production_shot_keys_invalid")
        _require(isinstance(shot["camera_goal"], str)
                 and bool(shot["camera_goal"].strip()),
                 "production_shot_camera_goal_invalid")
        trace = shot["structure_trace"]
        _require(isinstance(trace, dict) and set(trace) == SHOT_TRACE_FIELDS,
                 "production_shot_trace_keys_invalid")
        _require(trace == expected_shot_trace(screenplay, shot["beat_id"]),
                 "production_shot_trace_mismatch")
        roles.update(trace["structure_roles"])
        relations.update(trace["relation_ids"])
    _require(roles == set(CORE_COMPONENT_IDS[:3])
             and relations == {"R1_INFORMATION_UPDATE"},
             "production_structure_trace_coverage")
    failures = validate_shot_plan(
        plan, legacy_screenplay_view(screenplay, asset_graph), asset_graph)
    if failures:
        raise ContractError(f"production_shot_plan_{failures[0]['check']}")
