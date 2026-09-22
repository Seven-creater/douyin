"""Payload-only writer interface and deterministic contract fixture."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from src.agentic_video.creative_pipeline.contracts import (
    ContractError, validate_artifact_envelope,
)
from src.agentic_video.creative_pipeline.writing.screenplay import (
    SCREENPLAY_MAX_CANDIDATES, Screenplay,
)
from src.agentic_video.creative_pipeline.writing.validator import (
    require, validate_format_constraints, validate_screenplay,
    validate_writer_blueprint,
)
from src.agentic_video.creative_structure_v1.freeze import (
    CORE_COMPONENT_IDS, validate_frozen_spec,
)
from src.agentic_video.manifest import json_hash
from src.agentic_video.skills.registry import SkillSpec


class ScreenplayAdapter(Protocol):
    def run(self, request: dict[str, Any]) -> Screenplay:
        """Return one candidate; no workspace, model, or tool access supplied."""
        ...


def build_screenplay_request(*, creative_structure_spec: dict,
                             blueprint_envelope: dict, blueprint_sha: str,
                             format_constraints: dict) -> dict[str, Any]:
    validate_frozen_spec(creative_structure_spec)
    validate_format_constraints(format_constraints)
    validate_artifact_envelope(blueprint_envelope)
    require(blueprint_envelope["artifact_type"] == "story_blueprint"
            and blueprint_envelope["payload_schema_version"] == "story_blueprint_v1"
            and blueprint_envelope["status"] == "committed",
            "writer_blueprint_envelope_invalid")
    require(json_hash(blueprint_envelope) == blueprint_sha,
            "writer_blueprint_sha_mismatch")
    blueprint = blueprint_envelope["payload"]
    try:
        validate_writer_blueprint(blueprint, creative_structure_spec["artifact_sha"])
    except ContractError as exc:
        raise ContractError(exc.reason_code) from None
    # Exact public/nested schemas above reject extras. Envelope producer, paths,
    # selection decisions and history are never part of the writer payload.
    return deepcopy({
        "schema_version": "screenplay_request_v1",
        "creative_structure_spec": creative_structure_spec,
        "story_blueprint": blueprint,
        "story_blueprint_sha": blueprint_sha,
        "format_constraints": format_constraints,
    })


class FakeScreenplaySkill:
    def run(self, request: dict[str, Any]) -> Screenplay:
        """One placeholder beat per event, all in one scene; not a real script."""
        blueprint = request["story_blueprint"]
        constraints = request["format_constraints"]
        events = blueprint["events"]
        duration = constraints["target_duration_s"]
        per_beat = duration / len(events)
        event_beats = {event["event_id"]: f"BEAT_{i:02d}"
                       for i, event in enumerate(events, 1)}
        characters = deepcopy(blueprint["characters"])
        character_ids = [row["character_id"] for row in characters]
        trace = {key: event_beats[blueprint["structure_bindings"][key]]
                 for key in CORE_COMPONENT_IDS[:3]}
        trace["R1_INFORMATION_UPDATE"] = {
            "blueprint_relation_id": blueprint["event_relations"][0]["relation_id"],
            "prior_beat_id": trace[CORE_COMPONENT_IDS[0]],
            "evidence_beat_id": trace[CORE_COMPONENT_IDS[1]],
            "updated_beat_id": trace[CORE_COMPONENT_IDS[2]],
        }
        return {
            "schema_version": "screenplay_v1",
            "screenplay_id": f"FAKE_SCREENPLAY_{blueprint['blueprint_id']}",
            "blueprint_id": blueprint["blueprint_id"],
            "story_blueprint_sha": request["story_blueprint_sha"],
            "creative_structure_sha": request["creative_structure_spec"]["artifact_sha"],
            "format_constraints_sha": json_hash(constraints),
            "target_duration_s": duration,
            "characters": characters,
            "scenes": [{"scene_id": "SCENE_01", "setting": blueprint["setting"],
                        "beat_ids": list(event_beats.values())}],
            "beats": [{
                "beat_id": event_beats[event["event_id"]],
                "scene_id": "SCENE_01",
                "blueprint_event_ids": [event["event_id"]],
                "action": f"Contract placeholder for event {i + 1}.",
                "character_ids": list(character_ids), "prop_ids": [],
                "duration_s": (duration - per_beat * i
                               if i == len(events) - 1 else per_beat),
            } for i, event in enumerate(events)],
            "dialogue_or_text_cues": [],
            "structure_trace": trace,
            "production_requirements": {
                "character_ids": character_ids, "locations": [blueprint["setting"]],
                "prop_ids": [], "notes": ["Deterministic contract fixture only."],
            },
        }


def build_blind_evaluation_request(candidate: dict, *, evaluation_id: str,
                                  writer_request: dict) -> dict:
    """Build a future judge view without source, producer, selection or trace."""
    validate_screenplay(
        candidate, blueprint=writer_request["story_blueprint"],
        blueprint_sha=writer_request["story_blueprint_sha"],
        structure_sha=writer_request["creative_structure_spec"]["artifact_sha"],
        format_constraints=writer_request["format_constraints"],
    )
    return deepcopy({
        "schema_version": "screenplay_blind_evaluation_request_v1",
        "evaluation_id": evaluation_id,
        "rubric": ["coherence", "originality", "emotional_effect"],
        "screenplay": {
            "target_duration_s": candidate["target_duration_s"],
            "characters": candidate["characters"], "scenes": candidate["scenes"],
            "beats": [{key: beat[key] for key in (
                "beat_id", "scene_id", "action", "character_ids", "prop_ids",
                "duration_s")} for beat in candidate["beats"]],
            "dialogue_or_text_cues": candidate["dialogue_or_text_cues"],
            "production_requirements": candidate["production_requirements"],
        },
    })


def fake_screenplay_spec() -> SkillSpec:
    def execute(workspace, *, request: dict) -> dict:
        del workspace
        adapter: ScreenplayAdapter = FakeScreenplaySkill()
        return {"candidates": [adapter.run(request)]}

    root = Path(__file__).parent
    package_sha = json_hash({name: (root / name).read_text(encoding="utf-8")
                             for name in ("adapter.py", "screenplay.py", "validator.py")})
    return SkillSpec(
        name="fake_screenplay", description="Compile a screenplay contract fixture",
        inputs=["creative_structure_spec", "story_blueprint", "format_constraints"],
        outputs=["screenplay_batch"],
        preconditions=["creative:story_blueprint_selection:committed"],
        validators=["validate_screenplay"], permission_profile="payload_only",
        max_calls=SCREENPLAY_MAX_CANDIDATES, max_repair_attempts=0,
        skill_version="1.0.0", execute_fn=execute,
        input_schema_versions=["screenplay_request_v1"],
        output_schema_version="screenplay_batch_v1", package_sha=package_sha,
        prompt_or_instruction_sha=json_hash({"fixture": "fake_screenplay_v1"}),
    )
