"""Deterministic R2-D Wave 1 skills used only to verify runtime contracts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.skills.registry import SkillRegistry, SkillSpec


def _package_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _relation(statement: str) -> dict[str, str]:
    return {
        "prior": "I0_PRIOR_INTERPRETATION",
        "evidence": "E1_NEW_INFORMATION",
        "updated": "I1_UPDATED_INTERPRETATION",
        "statement": statement,
    }


def _theme_candidate(structure_sha: str, *, theme_id: str, domain: str,
                     entity: str, evidence: str, setting: str) -> dict[str, Any]:
    return {
        "schema_version": "theme_candidate_v1",
        "theme_id": theme_id,
        "parent_structure_sha": structure_sha,
        "domain": domain,
        "premise": "New information changes an earlier interpretation.",
        "audience_promise": "The meaning of an earlier judgment becomes clearer.",
        "tone": "observational",
        "binding_slots": {
            "B_DOMAIN": domain,
            "B_ENTITY_ROLE": entity,
            "B_EVIDENCE_FORM": evidence,
            "B_SETTING": setting,
        },
        "structure_bindings": {
            "I0_PRIOR_INTERPRETATION": "an initial interpretation is held",
            "E1_NEW_INFORMATION": "new evidence becomes available",
            "I1_UPDATED_INTERPRETATION": "the interpretation is revised",
            "R1_INFORMATION_UPDATE": _relation(
                "the new evidence updates the initial interpretation"),
        },
    }


def _fake_theme(workspace, *, creative_structure_spec: dict[str, Any],
                user_brief: dict[str, Any] | None = None) -> dict[str, Any]:
    del workspace, user_brief
    structure_sha = str(creative_structure_spec["artifact_sha"])
    return {"candidates": [
        _theme_candidate(
            structure_sha,
            theme_id="FAKE_THEME_A",
            domain="system diagnosis",
            entity="reviewer",
            evidence="diagnostic record",
            setting="controlled test environment",
        ),
        _theme_candidate(
            structure_sha,
            theme_id="FAKE_THEME_B",
            domain="learning assessment",
            entity="assessor",
            evidence="later work sample",
            setting="review session",
        ),
    ]}


def _blueprint_candidate(structure_sha: str, theme: dict[str, Any], *,
                         blueprint_id: str) -> dict[str, Any]:
    suffix = blueprint_id.rsplit("_", 1)[-1]
    prior = f"EVENT_PRIOR_{suffix}"
    evidence = f"EVENT_EVIDENCE_{suffix}"
    updated = f"EVENT_UPDATED_{suffix}"
    relation_id = f"REL_UPDATE_{suffix}"
    return {
        "schema_version": "story_blueprint_v1",
        "blueprint_id": blueprint_id,
        "theme_id": theme["theme_id"],
        "parent_structure_sha": structure_sha,
        "logline": "A deterministic fixture that instantiates an information update.",
        "characters": [{"character_id": "ENTITY_01", "role": "observer"}],
        "setting": theme["binding_slots"]["B_SETTING"],
        "goal": "resolve the meaning of the available evidence",
        "stakes": "the initial interpretation may remain inaccurate",
        "events": [
            {"event_id": prior, "role": "prior_interpretation"},
            {"event_id": evidence, "role": "new_information"},
            {"event_id": updated, "role": "updated_interpretation"},
        ],
        "event_relations": [{
            "relation_id": relation_id,
            "type": "information_update",
            "prior_event_id": prior,
            "evidence_event_id": evidence,
            "updated_event_id": updated,
        }],
        "structure_bindings": {
            "I0_PRIOR_INTERPRETATION": prior,
            "E1_NEW_INFORMATION": evidence,
            "I1_UPDATED_INTERPRETATION": updated,
            "R1_INFORMATION_UPDATE": _relation(relation_id),
        },
        "production_assumptions": ["contract fixture; no media production"],
    }


def _fake_story(workspace, *, creative_structure_spec: dict[str, Any],
                selected_theme: dict[str, Any]) -> dict[str, Any]:
    del workspace
    structure_sha = str(creative_structure_spec["artifact_sha"])
    return {"candidates": [
        _blueprint_candidate(
            structure_sha, selected_theme, blueprint_id="FAKE_BLUEPRINT_A"),
        _blueprint_candidate(
            structure_sha, selected_theme, blueprint_id="FAKE_BLUEPRINT_B"),
    ]}


def build_fake_registry() -> SkillRegistry:
    registry = SkillRegistry()
    package_sha = _package_sha()
    registry.register(SkillSpec(
        name="fake_theme",
        description="Create deterministic theme contract fixtures",
        inputs=["creative_structure_spec"],
        outputs=["theme_candidate_batch"],
        preconditions=["creative:structure_spec:committed"],
        validators=["validate_theme_candidate"],
        cost_class="cheap_text",
        idempotent=True,
        execute_fn=_fake_theme,
        skill_version="1.0.0",
        input_schema_versions=["creative_structure_spec_v1"],
        output_schema_version="theme_candidate_batch_v1",
        permission_profile="payload_only",
        max_calls=1,
        max_repair_attempts=0,
        package_sha=package_sha,
        prompt_or_instruction_sha=json_hash({"fixture": "fake_theme_v1"}),
    ))
    registry.register(SkillSpec(
        name="fake_story",
        description="Create deterministic story-blueprint contract fixtures",
        inputs=["creative_structure_spec", "theme_candidate"],
        outputs=["story_blueprint_batch"],
        preconditions=["creative:theme_selection:committed"],
        validators=["validate_story_blueprint"],
        cost_class="cheap_text",
        idempotent=True,
        execute_fn=_fake_story,
        skill_version="1.0.0",
        input_schema_versions=[
            "creative_structure_spec_v1", "theme_candidate_v1"],
        output_schema_version="story_blueprint_batch_v1",
        permission_profile="payload_only",
        max_calls=1,
        max_repair_attempts=0,
        package_sha=package_sha,
        prompt_or_instruction_sha=json_hash({"fixture": "fake_story_v1"}),
    ))
    return registry
