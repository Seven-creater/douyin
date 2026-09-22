"""Deterministic coverage/trace/feasibility checks, not semantic entailment."""
from __future__ import annotations

import math
import re
from typing import Any

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
    validate_story_structure,
)
from src.agentic_video.creative_pipeline.writing.screenplay import (
    Beat, Character, FormatConstraints, ProductionRequirements, RelationTrace,
    Scene, Screenplay, StructureTrace, TextCue,
)
from src.agentic_video.creative_structure_v1.freeze import CORE_COMPONENT_IDS
from src.agentic_video.manifest import json_hash


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def exact_keys(value: object, fields: object, reason: str) -> None:
    require(isinstance(value, dict) and set(value) == set(fields), reason)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]{0,159}", value) is not None


def _strings(value: object, *, ids: bool = False) -> bool:
    check = _id if ids else _text
    return (isinstance(value, list) and all(check(item) for item in value)
            and len(value) == len(set(value)))


def _rows(value: object, fields: object, id_key: str,
          reason: str, *, empty: bool = False) -> dict[str, dict]:
    require(isinstance(value, list) and (empty or bool(value)), reason)
    rows = {}
    for row in value:
        exact_keys(row, fields, reason)
        require(_id(row[id_key]) and row[id_key] not in rows, reason)
        rows[row[id_key]] = row
    return rows


def validate_boundary(value: object) -> None:
    try:
        validate_creative_boundary(value)
    except ContractError as exc:
        # Writer feedback never echoes reference strings from the old guard.
        raise ContractError(exc.reason_code) from None


def validate_format_constraints(value: dict[str, Any]) -> None:
    exact_keys(value, FormatConstraints.__annotations__, "format_keys_invalid")
    require(value["schema_version"] == "screenplay_format_constraints_v1",
            "format_schema_invalid")
    require(all(_number(value[key]) for key in (
        "target_duration_s", "min_duration_s", "max_duration_s")),
        "format_duration_invalid")
    require(value["min_duration_s"] <= value["target_duration_s"]
            <= value["max_duration_s"], "format_duration_range_invalid")
    for key, minimum in (("max_scenes", 1), ("max_characters", 1),
                         ("max_props", 0)):
        require(type(value[key]) is int and value[key] >= minimum,
                "format_production_limit_invalid")


def validate_writer_blueprint(blueprint: dict, structure_sha: str) -> None:
    """Close nested input fields that the Wave 2 schema leaves open."""
    validate_boundary(blueprint)
    validate_story_structure(blueprint, structure_sha=structure_sha,
                             theme_id=blueprint.get("theme_id"))
    require(_id(blueprint["blueprint_id"]), "writer_blueprint_id_invalid")
    characters = _rows(blueprint["characters"], Character.__annotations__,
                       "character_id", "writer_characters_invalid")
    require(all(_text(row["role"]) for row in characters.values()),
            "writer_character_role_invalid")
    for event in blueprint["events"]:
        require(isinstance(event, dict)
                and {"event_id", "role"} <= set(event)
                <= {"event_id", "role", "description"},
                "writer_event_keys_invalid")
        require(_id(event["event_id"]) and _text(event["role"]),
                "writer_event_invalid")
        if "description" in event:
            require(_text(event["description"]), "writer_event_invalid")
    require(_strings(blueprint["production_assumptions"]),
            "writer_production_assumptions_invalid")


def validate_screenplay(value: dict, *, blueprint: dict,
                        blueprint_sha: str, structure_sha: str,
                        format_constraints: dict) -> None:
    validate_format_constraints(format_constraints)
    validate_writer_blueprint(blueprint, structure_sha)
    validate_boundary(value)
    exact_keys(value, Screenplay.__annotations__, "screenplay_keys_invalid")
    require(value["schema_version"] == "screenplay_v1",
            "screenplay_schema_invalid")
    require(_id(value["screenplay_id"]), "screenplay_id_invalid")
    require(value["blueprint_id"] == blueprint["blueprint_id"],
            "screenplay_blueprint_mismatch")
    require(value["story_blueprint_sha"] == blueprint_sha,
            "screenplay_blueprint_sha_mismatch")
    require(value["creative_structure_sha"] == structure_sha,
            "screenplay_structure_sha_mismatch")
    require(value["format_constraints_sha"] == json_hash(format_constraints),
            "screenplay_format_sha_mismatch")
    require(_number(value["target_duration_s"])
            and value["target_duration_s"]
            == format_constraints["target_duration_s"],
            "screenplay_target_duration_mismatch")

    characters = _rows(value["characters"], Character.__annotations__,
                       "character_id", "screenplay_characters_invalid")
    require(characters == {row["character_id"]: row
                           for row in blueprint["characters"]},
            "screenplay_blueprint_characters_mismatch")
    scenes = _rows(value["scenes"], Scene.__annotations__, "scene_id",
                  "screenplay_scenes_invalid")
    beats = _rows(value["beats"], Beat.__annotations__, "beat_id",
                 "screenplay_beats_invalid")
    cues = _rows(value["dialogue_or_text_cues"], TextCue.__annotations__,
                "cue_id", "screenplay_cues_invalid", empty=True)
    events = {row["event_id"] for row in blueprint["events"]}
    covered = set()
    used_characters = set()
    used_props = set()
    for beat in beats.values():
        require(_id(beat["scene_id"]) and beat["scene_id"] in scenes,
                "screenplay_beat_scene_unresolved")
        require(_text(beat["action"]), "screenplay_action_invalid")
        require(_strings(beat["blueprint_event_ids"], ids=True)
                and bool(beat["blueprint_event_ids"])
                and set(beat["blueprint_event_ids"]) <= events,
                "screenplay_event_unresolved")
        require(_strings(beat["character_ids"], ids=True)
                and set(beat["character_ids"]) <= set(characters),
                "screenplay_character_unresolved")
        require(_strings(beat["prop_ids"], ids=True), "screenplay_props_invalid")
        require(_number(beat["duration_s"]), "screenplay_beat_duration_invalid")
        covered.update(beat["blueprint_event_ids"])
        used_characters.update(beat["character_ids"])
        used_props.update(beat["prop_ids"])
    require(covered == events, "screenplay_blueprint_coverage_missing")
    require(used_characters == set(characters),
            "screenplay_character_coverage_missing")
    ordered_beats = []
    for scene in scenes.values():
        require(_text(scene["setting"]) and _strings(scene["beat_ids"], ids=True)
                and bool(scene["beat_ids"]), "screenplay_scene_invalid")
        require(all(item in beats and beats[item]["scene_id"]
                    == scene["scene_id"] for item in scene["beat_ids"]),
                "screenplay_scene_beat_mismatch")
        ordered_beats.extend(scene["beat_ids"])
    require(ordered_beats == list(beats), "screenplay_beat_order_or_coverage")
    total = sum(beat["duration_s"] for beat in beats.values())
    require(math.isclose(total, value["target_duration_s"], rel_tol=0,
                         abs_tol=1e-6), "screenplay_duration_sum_mismatch")
    require(format_constraints["min_duration_s"] <= total
            <= format_constraints["max_duration_s"], "screenplay_duration_limit")

    trace = value["structure_trace"]
    exact_keys(trace, StructureTrace.__annotations__, "screenplay_trace_keys")
    bound_beats = [trace[key] for key in CORE_COMPONENT_IDS[:3]]
    require(all(_id(item) and item in beats for item in bound_beats)
            and len(set(bound_beats)) == 3, "screenplay_trace_roles_invalid")
    for key in CORE_COMPONENT_IDS[:3]:
        require(blueprint["structure_bindings"][key]
                in beats[trace[key]]["blueprint_event_ids"],
                "screenplay_trace_event_mismatch")
    relation = trace["R1_INFORMATION_UPDATE"]
    exact_keys(relation, RelationTrace.__annotations__, "screenplay_relation_keys")
    require(relation == {
        "blueprint_relation_id": blueprint["event_relations"][0]["relation_id"],
        "prior_beat_id": bound_beats[0], "evidence_beat_id": bound_beats[1],
        "updated_beat_id": bound_beats[2],
    }, "screenplay_relation_mismatch")

    cue_durations = dict.fromkeys(beats, 0.0)
    for cue in cues.values():
        require(_id(cue["beat_id"]) and cue["beat_id"] in beats,
                "screenplay_cue_beat_unresolved")
        require(cue["kind"] in ("dialogue", "on_screen_text")
                and _text(cue["text"]) and _number(cue["duration_s"]),
                "screenplay_cue_invalid")
        require((cue["kind"] == "on_screen_text" and cue["speaker_id"] is None)
                or (cue["kind"] == "dialogue" and _id(cue["speaker_id"])
                    and cue["speaker_id"]
                    in beats[cue["beat_id"]]["character_ids"]),
                "screenplay_cue_speaker_invalid")
        cue_durations[cue["beat_id"]] += cue["duration_s"]
    require(all(duration <= beats[key]["duration_s"]
                for key, duration in cue_durations.items()),
            "screenplay_cue_duration_limit")

    production = value["production_requirements"]
    exact_keys(production, ProductionRequirements.__annotations__,
               "screenplay_production_keys")
    for key in ("character_ids", "prop_ids", "locations", "notes"):
        require(_strings(production[key], ids=key.endswith("_ids")),
                "screenplay_production_field_invalid")
    require(set(production["character_ids"]) == used_characters
            and set(production["prop_ids"]) == used_props
            and set(production["locations"])
            == {row["setting"] for row in scenes.values()},
            "screenplay_production_coverage_mismatch")
    require(len(scenes) <= format_constraints["max_scenes"]
            and len(characters) <= format_constraints["max_characters"]
            and len(used_props) <= format_constraints["max_props"],
            "screenplay_production_limit")
