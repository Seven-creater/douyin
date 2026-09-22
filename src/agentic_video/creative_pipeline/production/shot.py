"""Adapt screenplay_v1 beats to the existing shot_plan_v1 contract."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from src.agentic_video.creative_pipeline.production.asset import (
    location_asset_map,
)
from src.agentic_video.creative_structure_v1.freeze import CORE_COMPONENT_IDS


SHOT_PLAN_FIELDS = {
    "schema_version", "shot_plan_id", "screenplay_sha", "asset_graph_sha",
    "shots",
}
SHOT_FIELDS = {
    "shot_id", "beat_id", "narrative_role", "duration_budget_s",
    "characters", "location", "props", "wardrobe", "motion_refs",
    "camera", "camera_goal", "start_state", "end_state", "visual_events",
    "structure_trace",
}
SHOT_TRACE_FIELDS = {
    "source_beat_id", "structure_roles", "relation_ids",
}


class ShotPlanAdapter(Protocol):
    def run(self, screenplay: dict[str, Any], asset_graph: dict[str, Any], *,
            screenplay_sha: str, asset_graph_sha: str) -> dict:
        ...


def expected_shot_trace(screenplay: dict[str, Any],
                        beat_id: str) -> dict[str, Any]:
    trace = screenplay["structure_trace"]
    roles = [role for role in CORE_COMPONENT_IDS[:3]
             if trace[role] == beat_id]
    relation = trace["R1_INFORMATION_UPDATE"]
    relation_beats = {
        relation["prior_beat_id"], relation["evidence_beat_id"],
        relation["updated_beat_id"],
    }
    return {
        "source_beat_id": beat_id,
        "structure_roles": roles,
        "relation_ids": (["R1_INFORMATION_UPDATE"]
                         if beat_id in relation_beats else []),
    }


class FakeShotPlanAdapter:
    """Create one structurally traceable placeholder shot per screenplay beat."""

    def run(self, screenplay: dict[str, Any], asset_graph: dict[str, Any], *,
            screenplay_sha: str, asset_graph_sha: str) -> dict:
        locations = location_asset_map(asset_graph)
        scene_by_beat = {
            beat_id: scene for scene in screenplay["scenes"]
            for beat_id in scene["beat_ids"]
        }
        shots = []
        for index, beat in enumerate(screenplay["beats"], 1):
            scene = scene_by_beat[beat["beat_id"]]
            location = locations[scene["setting"]]
            state_asset = (beat["character_ids"] or [location])[0]
            trace = expected_shot_trace(screenplay, beat["beat_id"])
            shots.append({
                "shot_id": f"PROD_SHOT_{index:03d}",
                "beat_id": beat["beat_id"],
                "narrative_role": (
                    trace["structure_roles"][0]
                    if trace["structure_roles"] else "UNASSIGNED"),
                "duration_budget_s": [
                    beat["duration_s"], beat["duration_s"]],
                "characters": list(beat["character_ids"]),
                "location": location,
                "props": list(beat["prop_ids"]),
                "wardrobe": [], "motion_refs": [],
                "camera": {
                    "shot_size": "medium", "angle": "eye_level",
                    "movement": "static",
                },
                "camera_goal": "Present the source beat without changing it.",
                "start_state": {
                    f"{state_asset}.state": f"before {beat['beat_id']}"},
                "end_state": {
                    f"{state_asset}.state": f"after {beat['beat_id']}"},
                "visual_events": [beat["action"]],
                "structure_trace": trace,
            })
        return deepcopy({
            "schema_version": "shot_plan_v1",
            "shot_plan_id": f"SHOT_PLAN_{screenplay['screenplay_id']}",
            "screenplay_sha": screenplay_sha,
            "asset_graph_sha": asset_graph_sha,
            "shots": shots,
        })
