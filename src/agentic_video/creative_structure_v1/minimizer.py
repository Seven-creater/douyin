"""Deterministic graph reduction for a minimal Creative Structure Plan."""
from __future__ import annotations

import copy
from typing import Any, Iterable

from src.agentic_video.manifest import json_hash


def _structural(value: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return value.get("structural_schema") or {"elements": [], "relations": []}


def _relation_roots(value: dict[str, Any]) -> set[str]:
    roots = {
        relation_id
        for row in value.get("generation_constraints") or []
        for relation_id in row.get("requires_relation_ids") or []
    }
    roots.update(
        relation_id
        for row in value.get("information_state_arc") or []
        for relation_id in row.get("trigger_relation_ids") or []
    )
    roots.update(
        relation_id
        for row in (value.get("editing_schema") or {}).get("constraints") or []
        for relation_id in row.get("applies_to_relation_ids") or []
    )
    return roots


def removable_item_ids(value: dict[str, Any]) -> set[str]:
    """Find graph items that carry no relation or constrained binding."""
    structural = _structural(value)
    relations = structural["relations"]
    relation_ids = {row["relation_id"] for row in relations}
    roots = _relation_roots(value).intersection(relation_ids)
    removable_relations = relation_ids - roots if roots else set()
    retained_relations = [
        row for row in relations
        if row["relation_id"] not in removable_relations
    ]
    used_elements = {
        argument["element_id"]
        for row in retained_relations
        for argument in row.get("arguments") or []
    }
    removable_elements = (
        {row["element_id"] for row in structural["elements"]} - used_elements
        if retained_relations else set()
    )
    removable_slots = {
        row["slot_id"] for row in value.get("binding_slots") or []
        if not row.get("constraints")
    }
    return removable_relations | removable_elements | removable_slots


def _item_ids(value: dict[str, Any]) -> set[str]:
    structural = _structural(value)
    return {
        *(row["element_id"] for row in structural["elements"]),
        *(row["relation_id"] for row in structural["relations"]),
        *(row["state_id"] for row in value.get("information_state_arc") or []),
        *(row["constraint_id"] for row in value.get("generation_constraints") or []),
        *(row["constraint_id"] for row in (
            value.get("editing_schema") or {}).get("constraints") or []),
        *(row["slot_id"] for row in value.get("binding_slots") or []),
    }


def _counts(value: dict[str, Any]) -> dict[str, int]:
    structural = _structural(value)
    return {
        "elements": len(structural["elements"]),
        "relations": len(structural["relations"]),
        "information_states": len(value.get("information_state_arc") or []),
        "generation_constraints": len(value.get("generation_constraints") or []),
        "editing_constraints": len(
            (value.get("editing_schema") or {}).get("constraints") or []),
        "binding_slots": len(value.get("binding_slots") or []),
    }


def minimize_structure(
        value: dict[str, Any], *, transfer_profiles: Iterable[dict[str, Any]]
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove graph-redundant items without interpreting surface vocabulary."""
    result = copy.deepcopy(value)
    removed = removable_item_ids(result)
    before = _counts(result)
    structural = result.get("structural_schema")
    if structural is not None:
        structural["elements"] = [
            row for row in structural["elements"]
            if row["element_id"] not in removed
        ]
        structural["relations"] = [
            row for row in structural["relations"]
            if row["relation_id"] not in removed
        ]
        retained_elements = {row["element_id"] for row in structural["elements"]}
        retained_relations = {row["relation_id"] for row in structural["relations"]}
        for state in result.get("information_state_arc") or []:
            state["available_element_ids"] = [
                item for item in state["available_element_ids"]
                if item in retained_elements
            ]
            state["trigger_relation_ids"] = [
                item for item in state["trigger_relation_ids"]
                if item in retained_relations
            ]
    result["binding_slots"] = [
        row for row in result.get("binding_slots") or []
        if row["slot_id"] not in removed
    ]
    if not result["binding_slots"] \
            and result.get("dimension_status", {}).get("binding_slots") == "supported":
        result["dimension_status"]["binding_slots"] = "not_applicable"

    retained_ids = _item_ids(result)
    annex = result.get("audit_annex") or {}
    annex["source_bindings"] = [
        row for row in annex.get("source_bindings") or []
        if row["abstract_id"] in retained_ids
    ]
    annex["anti_invariants"] = [
        row for row in annex.get("anti_invariants") or []
        if row["binding_id"] not in removed
    ]
    if "artifact_sha" in result:
        result.pop("artifact_sha")
        result["artifact_sha"] = json_hash(result)

    profiles = list(transfer_profiles)
    report = {
        "schema_version": "creative_structure_minimization_trace_v1",
        "method": "relation_reachability_and_constrained_binding_reduction",
        "new_model_calls": 0,
        "before_counts": before,
        "after_counts": _counts(result),
        "removed_ids": sorted(removed),
        "positive_transfer_profile_ids": [
            row["case_id"] for row in profiles
            if row.get("case_type") == "cross_domain_positive"
        ],
        "negative_transfer_profile_ids": [
            row["case_id"] for row in profiles
            if row.get("case_type") == "surface_lure_negative"
        ],
    }
    report["artifact_sha"] = json_hash(report)
    return result, report
