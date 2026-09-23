"""One-chain, text-only creative baseline using the frozen public structure spec.

This experiment does not publish into the R2-D production workspace. It keeps
model responses and deterministic compilation separate for later comparison.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from src.agentic_video.creative_pipeline.contracts import (
    ArtifactEnvelope, ContractError, artifact_ref,
)
from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary, validate_story_structure,
    validate_theme_structure,
)
from src.agentic_video.creative_pipeline.writing.adapter import (
    build_screenplay_request,
)
from src.agentic_video.creative_pipeline.writing.validator import (
    validate_format_constraints, validate_screenplay,
)
from src.agentic_video.creative_structure_v1.freeze import validate_frozen_spec
from src.agentic_video.manifest import json_hash


THEME_PROMPT = """Create ONE original short-video theme from the public mechanism.
Use only the supplied creative_structure_spec and user_brief. Do not infer or
reconstruct any reference video. Make the premise concrete, visual and distinct.
Return one JSON object with exactly: domain, premise, audience_promise, tone,
binding_slots, structure_bindings. binding_slots has B_DOMAIN, B_ENTITY_ROLE,
B_EVIDENCE_FORM, B_SETTING. structure_bindings has I0_PRIOR_INTERPRETATION,
E1_NEW_INFORMATION, I1_UPDATED_INTERPRETATION, R1_INFORMATION_UPDATE.
The R1 value is an object with prior, evidence, updated, statement; its first
three values are the corresponding component IDs. Each information-state value
must describe a different state. Do not output markdown or commentary.
"""

STORY_PROMPT = """Create ONE original, filmable story blueprint for this theme.
Use only the supplied public structure and selected theme. Preserve the
information-update mechanism without copying any reference video.
Return one JSON object with exactly: logline, characters, setting, goal,
stakes, events, event_relations, structure_bindings, production_assumptions.
characters: [{character_id, role}]. events: at least three objects with
{event_id, role, description}; IDs use letters, digits and underscores.
event_relations: exactly one {relation_id, type, prior_event_id,
evidence_event_id, updated_event_id}, type=information_update, and its three
event IDs must be distinct. structure_bindings maps I0_PRIOR_INTERPRETATION,
E1_NEW_INFORMATION, I1_UPDATED_INTERPRETATION to those event IDs and maps
R1_INFORMATION_UPDATE to {prior, evidence, updated, statement}; the first
three are component IDs and statement is the relation_id.
production_assumptions is a list of strings. Do not output markdown.
"""

SCREENPLAY_PROMPT = """Write ONE concrete, filmable short screenplay from the
selected story blueprint. Use only the supplied writer_request. Show specific
visible and audible actions; do not write contract placeholders or generic
summaries. Return one JSON object with exactly: scenes, beats,
dialogue_or_text_cues. scenes: [{scene_id, setting, beat_ids}]. beats:
[{beat_id, scene_id, blueprint_event_ids, action, character_ids, prop_ids,
duration_s}]. Each blueprint event must appear in at least one beat; the prior,
new-information and updated-interpretation events must map to distinct beats.
List beats in scene order. Durations must sum exactly to target_duration_s.
dialogue_or_text_cues: [{cue_id, beat_id, kind, speaker_id, text, duration_s}],
where kind is dialogue or on_screen_text; speaker_id is null for on_screen_text.
Respect max_scenes, max_characters and max_props. Do not output markdown.
"""

PROMPTS = (THEME_PROMPT, STORY_PROMPT, SCREENPLAY_PROMPT)
STAGES = ("01_theme", "02_story", "03_screenplay")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _parse_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("model_response_not_object")
    return value


def _fields(value: dict[str, Any], expected: set[str], stage: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{stage}_fields_invalid")


def _ask(runner: Any, root: Path, stage: str, prompt: str,
         payload: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    validate_creative_boundary(payload)
    directory = root / stage
    full_prompt = prompt + "\nINPUT_JSON:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    _write_json(directory / "request.json", {
        "prompt": full_prompt, "prompt_sha": json_hash(full_prompt),
        "payload_sha": json_hash(payload), "max_new_tokens": max_new_tokens,
        "stop_after_json_object": True,
    })
    answer = runner.ask(full_prompt, max_new_tokens=max_new_tokens,
                        stop_after_json_object=True)
    raw = str(getattr(answer, "text", answer))
    (directory / "raw_response.txt").write_text(raw, encoding="utf-8")
    _write_json(directory / "model_call.json", {
        "input_tokens": getattr(answer, "input_tokens", None),
        "output_tokens": getattr(answer, "output_tokens", None),
        "elapsed_s": getattr(answer, "elapsed_s", None),
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    })
    return _parse_object(raw)


def _compile_screenplay(draft: dict[str, Any], request: dict[str, Any]
                        ) -> dict[str, Any]:
    _fields(draft, {"scenes", "beats", "dialogue_or_text_cues"}, "screenplay")
    blueprint = request["story_blueprint"]
    beats = draft["beats"]
    if any("contract placeholder" in beat["action"].lower()
           for beat in beats):
        raise ContractError("screenplay_fixture_placeholder_reused")
    event_to_beats = {
        event["event_id"]: [beat["beat_id"] for beat in beats
                            if event["event_id"] in beat["blueprint_event_ids"]]
        for event in blueprint["events"]
    }
    bindings = blueprint["structure_bindings"]
    roles = {}
    for key in ("I0_PRIOR_INTERPRETATION", "E1_NEW_INFORMATION",
                "I1_UPDATED_INTERPRETATION"):
        matches = event_to_beats[bindings[key]]
        if len(matches) != 1:
            raise ContractError("screenplay_role_beat_ambiguous", key)
        roles[key] = matches[0]
    relation = blueprint["event_relations"][0]
    roles["R1_INFORMATION_UPDATE"] = {
        "blueprint_relation_id": relation["relation_id"],
        "prior_beat_id": roles["I0_PRIOR_INTERPRETATION"],
        "evidence_beat_id": roles["E1_NEW_INFORMATION"],
        "updated_beat_id": roles["I1_UPDATED_INTERPRETATION"],
    }
    return {
        "schema_version": "screenplay_v1",
        "screenplay_id": "REAL_BASELINE_SCREENPLAY_01",
        "blueprint_id": blueprint["blueprint_id"],
        "story_blueprint_sha": request["story_blueprint_sha"],
        "creative_structure_sha": request["creative_structure_spec"]["artifact_sha"],
        "format_constraints_sha": json_hash(request["format_constraints"]),
        "target_duration_s": request["format_constraints"]["target_duration_s"],
        "characters": blueprint["characters"],
        "scenes": draft["scenes"],
        "beats": beats,
        "dialogue_or_text_cues": draft["dialogue_or_text_cues"],
        "structure_trace": roles,
        "production_requirements": {
            "character_ids": list(dict.fromkeys(
                item for beat in beats for item in beat["character_ids"])),
            "locations": list(dict.fromkeys(
                scene["setting"] for scene in draft["scenes"])),
            "prop_ids": list(dict.fromkeys(
                item for beat in beats for item in beat["prop_ids"])),
            "notes": [],
        },
    }


def run_real_text_baseline(runner: Any, structure: dict[str, Any],
                           output_dir: Path, *, format_constraints: dict[str, Any],
                           user_brief: dict[str, Any] | None = None,
                           model_id: str = "omni",
                           model_config_sha: str | None = None) -> dict[str, Any]:
    """Run exactly three text model calls, stop at the first invalid stage."""
    validate_frozen_spec(structure)
    validate_format_constraints(format_constraints)
    validate_creative_boundary(user_brief or {})
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"baseline output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "input.json", {
        "structure_spec_sha": structure["artifact_sha"],
        "format_constraints": format_constraints,
        "user_brief": user_brief or {}, "model_id": model_id,
        "model_config_sha": model_config_sha,
        "intent_used": False, "reference_context_used": False,
    })
    stage = STAGES[0]
    calls = 0
    try:
        theme_draft = _ask(runner, root, stage, THEME_PROMPT, {
            "creative_structure_spec": structure, "user_brief": user_brief or {},
        }, 2048)
        calls += 1
        _fields(theme_draft, {"domain", "premise", "audience_promise", "tone",
                              "binding_slots", "structure_bindings"}, "theme")
        theme = {"schema_version": "theme_candidate_v1",
                 "theme_id": "REAL_BASELINE_THEME_01",
                 "parent_structure_sha": structure["artifact_sha"],
                 **theme_draft}
        validate_theme_structure(theme, structure_sha=structure["artifact_sha"])
        _write_json(root / stage / "candidate.json", theme)
        theme_sha = json_hash(theme)
        _write_json(root / stage / "validation.json", {
            "status": "PASS", "candidate_sha": theme_sha})

        stage = STAGES[1]
        story_draft = _ask(runner, root, stage, STORY_PROMPT, {
            "creative_structure_spec": structure, "selected_theme": theme,
        }, 3072)
        calls += 1
        _fields(story_draft, {"logline", "characters", "setting", "goal",
                              "stakes", "events", "event_relations",
                              "structure_bindings", "production_assumptions"},
                "story")
        story = {"schema_version": "story_blueprint_v1",
                 "blueprint_id": "REAL_BASELINE_BLUEPRINT_01",
                 "theme_id": theme["theme_id"],
                 "parent_structure_sha": structure["artifact_sha"],
                 **story_draft}
        validate_story_structure(story, structure_sha=structure["artifact_sha"],
                                 theme_id=theme["theme_id"])
        _write_json(root / stage / "candidate.json", story)
        story_sha = json_hash(story)
        _write_json(root / stage / "validation.json", {
            "status": "PASS", "candidate_sha": story_sha})

        stage = STAGES[2]
        producer = {"skill_id": "real_text_baseline.story",
                    "skill_version": "1.0.0",
                    "package_sha": hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest(),
                    "prompt_or_instruction_sha": json_hash(STORY_PROMPT)}
        envelope = ArtifactEnvelope(
            artifact_id="creative:story_blueprint:REAL_BASELINE_BLUEPRINT_01",
            artifact_type="story_blueprint", schema_version="story_blueprint_v1",
            version="v1", payload=story, producer=producer,
            derived_from=(artifact_ref(structure["spec_id"],
                                       structure["artifact_sha"]),
                          artifact_ref(theme["theme_id"], theme_sha)),
            status="committed",
        ).to_dict()
        _write_json(root / "02_story" / "envelope.json", envelope)
        writer_request = build_screenplay_request(
            creative_structure_spec=structure, blueprint_envelope=envelope,
            blueprint_sha=json_hash(envelope),
            format_constraints=format_constraints)
        screenplay_draft = _ask(runner, root, stage, SCREENPLAY_PROMPT,
                                {"writer_request": writer_request}, 4096)
        calls += 1
        screenplay = _compile_screenplay(screenplay_draft, writer_request)
        validate_screenplay(
            screenplay, blueprint=story, blueprint_sha=json_hash(envelope),
            structure_sha=structure["artifact_sha"],
            format_constraints=format_constraints)
        _write_json(root / stage / "candidate.json", screenplay)
        _write_json(root / stage / "validation.json", {
            "status": "PASS", "candidate_sha": json_hash(screenplay)})
        _write_json(root / "lineage.json", {
            "structure_sha": structure["artifact_sha"],
            "theme": {"sha": theme_sha,
                      "derived_from": [structure["artifact_sha"]]},
            "story": {"sha": story_sha,
                      "envelope_sha": json_hash(envelope),
                      "derived_from": [structure["artifact_sha"], theme_sha]},
            "screenplay": {"sha": json_hash(screenplay),
                           "derived_from": [structure["artifact_sha"],
                                            json_hash(envelope),
                                            json_hash(format_constraints)]},
        })
        result = {
            "status": "PASS", "model_calls": calls,
            "structure_sha": structure["artifact_sha"],
            "theme_sha": theme_sha, "story_sha": story_sha,
            "screenplay_sha": json_hash(screenplay),
            "quality_evaluation": "not_run", "selection": "not_run",
            "media_generation": "not_run", "intent_used": False,
            "production_committed": False,
        }
    except BaseException as exc:
        reason = getattr(exc, "reason_code", str(exc))
        observed_calls = sum(int((root / item / "raw_response.txt").is_file())
                             for item in STAGES)
        outcome_unknown = ((root / stage / "request.json").is_file()
                           and not (root / stage / "raw_response.txt").is_file())
        _write_json(root / stage / "validation.json", {
            "status": "FAIL", "error_type": type(exc).__name__,
            "reason": reason})
        result = {"status": "BLOCKED", "failed_stage": stage,
                  "model_calls_with_response": observed_calls,
                  "in_flight_call_outcome_unknown": outcome_unknown,
                  "error_type": type(exc).__name__,
                  "reason": reason,
                  "quality_evaluation": "not_run", "selection": "not_run",
                  "media_generation": "not_run", "intent_used": False,
                  "production_committed": False}
        _write_json(root / "result.json", result)
        raise
    _write_json(root / "result.json", result)
    return result
