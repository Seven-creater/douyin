"""Adapt screenplay_v1 production requirements to the existing asset_graph_v1."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol


ASSET_GRAPH_FIELDS = {
    "schema_version", "asset_graph_id", "screenplay_sha", "assets",
}
ASSET_FIELDS = {
    "asset_id", "type", "tier", "description", "immutable", "mutable",
    "usage", "status", "source_requirement", "identity_constraints",
    "visual_constraints",
}


class AssetGraphAdapter(Protocol):
    def run(self, screenplay: dict[str, Any], *, screenplay_sha: str) -> dict:
        ...


def location_asset_map(graph: dict[str, Any]) -> dict[str, str]:
    return {
        str(asset["source_requirement"]): str(asset["asset_id"])
        for asset in graph.get("assets") or []
        if asset.get("type") == "location"
    }


def legacy_screenplay_view(screenplay: dict[str, Any],
                           graph: dict[str, Any]) -> dict[str, Any]:
    """Map the new screenplay layout to the legacy validators' read view."""
    beats = {row["beat_id"]: row for row in screenplay["beats"]}
    locations = location_asset_map(graph)
    return {
        "target_duration": screenplay["target_duration_s"],
        "characters": [{"id": row["character_id"]}
                       for row in screenplay["characters"]],
        "scenes": [{
            "scene_id": scene["scene_id"],
            "location": locations[scene["setting"]],
            "beats": [{
                "beat_id": beat_id,
                "purpose": beats[beat_id]["action"],
                "production_requirements": {
                    "characters": list(beats[beat_id]["character_ids"]),
                    "location": locations[scene["setting"]],
                    "props": list(beats[beat_id]["prop_ids"]),
                    "motion_refs": [],
                },
            } for beat_id in scene["beat_ids"]],
        } for scene in screenplay["scenes"]],
    }


class FakeAssetGraphAdapter:
    """Build identity/location/prop records without generating any media."""

    def run(self, screenplay: dict[str, Any], *, screenplay_sha: str) -> dict:
        beat_usage: dict[str, list[str]] = {}
        for beat in screenplay["beats"]:
            for asset_id in [*beat["character_ids"], *beat["prop_ids"]]:
                beat_usage.setdefault(asset_id, []).append(beat["beat_id"])
        assets = []
        for character in screenplay["characters"]:
            asset_id = character["character_id"]
            assets.append({
                "asset_id": asset_id, "type": "character", "tier": "A",
                "description": character["role"],
                "immutable": {"identity": f"canonical {asset_id}"},
                "mutable": {}, "usage": beat_usage.get(asset_id, []),
                "status": "active", "source_requirement": asset_id,
                "identity_constraints": ["preserve canonical identity"],
                "visual_constraints": [],
            })
        for index, location in enumerate(
                screenplay["production_requirements"]["locations"], 1):
            usage = [scene_beat for scene in screenplay["scenes"]
                     if scene["setting"] == location
                     for scene_beat in scene["beat_ids"]]
            assets.append({
                "asset_id": f"ENV_{index:02d}", "type": "location",
                "tier": "A", "description": location,
                "immutable": {}, "mutable": {}, "usage": usage,
                "status": "active", "source_requirement": location,
                "identity_constraints": [],
                "visual_constraints": ["preserve spatial layout"],
            })
        for prop_id in screenplay["production_requirements"]["prop_ids"]:
            assets.append({
                "asset_id": prop_id, "type": "prop", "tier": "B",
                "description": prop_id, "immutable": {}, "mutable": {},
                "usage": beat_usage.get(prop_id, []), "status": "active",
                "source_requirement": prop_id, "identity_constraints": [],
                "visual_constraints": ["preserve recognisable form"],
            })
        return deepcopy({
            "schema_version": "asset_graph_v1",
            "asset_graph_id": f"ASSET_GRAPH_{screenplay['screenplay_id']}",
            "screenplay_sha": screenplay_sha,
            "assets": assets,
        })
