"""CPU-only P1 draft: separate generation processes from rendered snippets.

This module consumes fixed, reviewable fixtures while P0 has no human PASS.
Its drafts are not executable V9-G contracts and cannot authorize H3 calls.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


P1_DRAFT_VERSION = "v9g_p1_draft"
CONTINUITY_LEVELS = {"required", "preferred", "not_required", "unknown"}
CONTINUITY_DIMENSIONS = (
    "subject", "opponent", "scene", "event", "actor_role", "spatial_orientation",
)


class P1Blocked(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)
        self.reason_code = reason_code


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def plan_generation_units(requirement: dict[str, Any],
                          action_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate an explicit causal-action plan; do not infer units from cuts."""
    section_id = str(requirement.get("section_id") or "")
    semantic = requirement.get("semantic_requirement") or {}
    presentation = requirement.get("presentation_requirement") or {}
    continuity = (requirement.get("continuity_requirement") or {}).get("levels") or {}
    phases = semantic.get("required_semantic_phases") or []
    if (not section_id or not phases or not all(isinstance(value, str) and value
                                               for value in phases)):
        raise P1Blocked("section_semantics_missing", section_id)
    if set(continuity) != set(CONTINUITY_DIMENSIONS) or any(
            value not in CONTINUITY_LEVELS for value in continuity.values()):
        raise P1Blocked("continuity_schema_invalid", section_id)
    if "unknown" in continuity.values():
        raise P1Blocked("continuity_unknown", section_id)
    snippet_range = presentation.get("snippet_count_range")
    if (not isinstance(snippet_range, list) or len(snippet_range) != 2 or
            not all(isinstance(value, int) and value >= 1 for value in snippet_range) or
            snippet_range[0] > snippet_range[1]):
        raise P1Blocked("snippet_range_invalid", section_id)
    if not isinstance(action_plan, list) or not action_plan:
        raise P1Blocked("causal_action_plan_missing", section_id)
    units = []
    observed_phases = []
    required_identity = {
        key: set() for key in ("subject", "opponent", "scene", "event", "actor_role")
        if continuity[key] == "required"
    }
    for index, action in enumerate(action_plan, 1):
        if not isinstance(action, dict):
            raise P1Blocked("generation_unit_incomplete", f"{section_id}/G{index}")
        goal = str(action.get("primary_causal_goal") or "").strip()
        unit_phases = action.get("semantic_phases") or []
        try:
            duration = float(action.get("generated_duration_s") or 0)
        except (TypeError, ValueError) as exc:
            raise P1Blocked("generation_unit_incomplete", f"{section_id}/G{index}") from exc
        if not goal or not unit_phases or not 4 <= duration <= 15:
            raise P1Blocked("generation_unit_incomplete", f"{section_id}/G{index}")
        if not isinstance(unit_phases, list) or not all(
                isinstance(phase, str) and phase for phase in unit_phases):
            raise P1Blocked("generation_phase_invalid", f"{section_id}/G{index}")
        if len(set(unit_phases)) != len(unit_phases):
            raise P1Blocked("generation_phase_repeated", f"{section_id}/G{index}")
        observed_phases.extend(unit_phases)
        identifiers = action.get("continuity_ids") or {}
        condition_keys = action.get("condition_keys") or []
        if (not isinstance(identifiers, dict) or not isinstance(condition_keys, list) or
                any(not isinstance(key, str) or not key for key in condition_keys) or
                len(set(condition_keys)) != len(condition_keys)):
            raise P1Blocked("generation_condition_invalid", f"{section_id}/G{index}")
        for key, values in required_identity.items():
            value = str(identifiers.get(key) or "").strip()
            if not value:
                raise P1Blocked("required_continuity_id_missing", f"{section_id}/{key}")
            values.add(value)
        units.append({
            "unit_id": f"{section_id}.G{index}", "section_id": section_id,
            "primary_causal_goal": goal,
            "semantic_phases": list(unit_phases),
            "generated_duration_s": duration,
            "continuity_ids": dict(identifiers),
            "condition_keys": list(condition_keys),
            "generation_complexity_note": str(action.get("generation_complexity_note") or ""),
        })
    if observed_phases != phases:
        raise P1Blocked("required_phase_partition_invalid", section_id)
    if any(len(values) != 1 for values in required_identity.values()):
        raise P1Blocked("required_continuity_broken", section_id)
    return units


def compile_p1_contract_draft(requirements: dict[str, Any],
                              target_setting: dict[str, Any],
                              action_plans: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Produce a non-executable draft for P1 human review, never a frozen contract."""
    if requirements.get("schema_version") != "material_requirements_v9_p0":
        raise P1Blocked("p0_requirements_schema_required")
    rows = requirements.get("requirements") or []
    if not rows or not isinstance(target_setting.get("assets"), dict):
        raise P1Blocked("p1_inputs_incomplete")
    sections = []
    for row in rows:
        section_id = str(row.get("section_id") or "")
        semantic = row.get("semantic_requirement") or {}
        presentation = row.get("presentation_requirement") or {}
        evidence = row.get("evidence_requirement") or {}
        if (not semantic.get("meaning_to_prove") or
                not presentation.get("composition_mode") or
                not evidence.get("minimum_sufficient_evidence_set")):
            raise P1Blocked("requirement_layers_incomplete", section_id)
        units = plan_generation_units(row, action_plans.get(section_id) or [])
        sections.append({
            "section_id": section_id,
            "semantic_goal": semantic["meaning_to_prove"],
            "composition_mode": presentation["composition_mode"],
            "render_duration_budget_s": presentation.get("target_duration_s"),
            "snippet_count_range": presentation["snippet_count_range"],
            "continuity": row["continuity_requirement"]["levels"],
            "required_phases": semantic["required_semantic_phases"],
            "generation_units": units,
            "rendered_snippets": [],
        })
    if set(action_plans) != {section["section_id"] for section in sections}:
        raise P1Blocked("action_plan_section_mismatch")
    draft = {
        "contract_version": P1_DRAFT_VERSION,
        "execution_authorized": False,
        "source_requirements_sha256": _hash(requirements),
        "target_setting_sha256": _hash(target_setting),
        "sections": sections,
    }
    draft["draft_sha256"] = _hash(draft)
    return draft
