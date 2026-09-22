"""Wave 4 constraints layered over the existing asset_graph_v1 validator."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.creative_pipeline.production.asset import (
    ASSET_FIELDS, ASSET_GRAPH_FIELDS, legacy_screenplay_view,
)
from src.agentic_video.skills.asset_schema import validate_asset_graph


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def _strings(value: object) -> bool:
    return (isinstance(value, list)
            and all(isinstance(item, str) and item.strip() for item in value)
            and len(value) == len(set(value)))


def validate_production_boundary(value: object) -> None:
    try:
        validate_creative_boundary(value)
    except ContractError as exc:
        raise ContractError(exc.reason_code) from None


def validate_production_asset_graph(graph: dict[str, Any], *,
                                    screenplay: dict[str, Any],
                                    screenplay_sha: str) -> None:
    validate_production_boundary(graph)
    _require(isinstance(graph, dict) and set(graph) == ASSET_GRAPH_FIELDS,
             "production_asset_graph_keys_invalid")
    _require(graph["schema_version"] == "asset_graph_v1",
             "production_asset_graph_schema_invalid")
    _require(isinstance(graph["asset_graph_id"], str)
             and bool(graph["asset_graph_id"].strip()),
             "production_asset_graph_id_invalid")
    _require(graph["screenplay_sha"] == screenplay_sha,
             "production_asset_graph_parent_sha_mismatch")
    assets = graph["assets"]
    _require(isinstance(assets, list) and bool(assets),
             "production_assets_empty")
    ids = set()
    for asset in assets:
        _require(isinstance(asset, dict) and set(asset) == ASSET_FIELDS,
                 "production_asset_keys_invalid")
        asset_id = asset["asset_id"]
        _require(isinstance(asset_id, str) and bool(asset_id.strip())
                 and asset_id not in ids, "production_asset_id_invalid")
        ids.add(asset_id)
        _require(isinstance(asset["description"], str)
                 and bool(asset["description"].strip()),
                 "production_asset_description_invalid")
        _require(isinstance(asset["immutable"], dict)
                 and isinstance(asset["mutable"], dict),
                 "production_asset_state_invalid")
        _require(_strings(asset["usage"]), "production_asset_usage_invalid")
        _require(isinstance(asset["source_requirement"], str)
                 and bool(asset["source_requirement"].strip()),
                 "production_asset_source_invalid")
        _require(_strings(asset["identity_constraints"])
                 and _strings(asset["visual_constraints"]),
                 "production_asset_constraints_invalid")
        if asset["type"] == "character":
            _require(bool(asset["immutable"])
                     and bool(asset["identity_constraints"]),
                     "production_character_identity_invalid")

    required = screenplay["production_requirements"]
    by_type = {
        kind: [row for row in assets if row["type"] == kind]
        for kind in ("character", "location", "prop")
    }
    _require({row["asset_id"] for row in by_type["character"]}
             == set(required["character_ids"]),
             "production_character_completeness")
    _require({row["source_requirement"] for row in by_type["location"]}
             == set(required["locations"]),
             "production_location_completeness")
    _require({row["asset_id"] for row in by_type["prop"]}
             == set(required["prop_ids"]),
             "production_prop_completeness")

    failures = validate_asset_graph(
        graph, legacy_screenplay_view(screenplay, graph))
    if failures:
        raise ContractError(
            f"production_asset_graph_{failures[0]['check']}")
