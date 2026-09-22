"""Deterministic R2-D skills used only to verify runtime contracts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.creative_pipeline.planning.story import (
    build_fake_story_blueprints,
)
from src.agentic_video.creative_pipeline.planning.theme import (
    build_fake_theme_candidates,
)
from src.agentic_video.skills.registry import SkillRegistry, SkillSpec


def _package_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _fake_theme(workspace, *, creative_structure_spec: dict[str, Any],
                user_brief: dict[str, Any] | None = None) -> dict[str, Any]:
    del workspace, user_brief
    structure_sha = str(creative_structure_spec["artifact_sha"])
    return {"candidates": build_fake_theme_candidates(structure_sha)}


def _fake_story(workspace, *, creative_structure_spec: dict[str, Any],
                selected_theme: dict[str, Any]) -> dict[str, Any]:
    del workspace
    structure_sha = str(creative_structure_spec["artifact_sha"])
    return {"candidates": build_fake_story_blueprints(
        structure_sha, selected_theme)}


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
        max_calls=5,
        max_repair_attempts=0,
        package_sha=package_sha,
        prompt_or_instruction_sha=json_hash({"fixture": "fake_story_v1"}),
    ))
    return registry
