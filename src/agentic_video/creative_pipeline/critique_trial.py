"""Isolated two-arm story revision trial; never publishes production artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary, validate_story_structure,
    validate_theme_structure,
)
from src.agentic_video.creative_pipeline.real_text_baseline import (
    _ask, _compile_story, _write_json,
)
from src.agentic_video.creative_structure_v1.freeze import validate_frozen_spec
from src.agentic_video.manifest import json_hash


REVISION_PROMPT = """Revise the supplied short-video story blueprint directly.
Use only the public creative structure, selected theme, current story, and
experiment constraints. Preserve the theme's concrete premise while improving
causal coherence, a visible information update, originality, and filmability.
The prior interpretation, new information, and updated interpretation must
be distinct events connected by an actual information update. Do not copy or
infer any reference video. Return exactly one JSON object with: logline,
characters, setting, goal, stakes, events, event_relations,
production_assumptions. characters: [{character_id, role}]. events:
[{event_id, role, description}]. event_relations: a JSON array containing
exactly one object with
{relation_id, type, prior_event_id, evidence_event_id, updated_event_id};
type=information_update and the three event IDs are distinct. Do not include
structure_bindings; the program derives them. No markdown or commentary.
"""

CRITIC_PROMPT = """Critique the supplied short-video story blueprint.
Use only the public creative structure, selected theme, current story, and
experiment constraints. Assess whether the mapped events actually enact the
prior interpretation, new information, and updated interpretation; also
assess causal coherence, filmability, originality, and production feasibility.
Identify concrete problems only when present. Do not write a revised story,
infer a reference video, or prescribe a preset narrative role beyond the
public structure. Return exactly one JSON object with strengths (list of
strings), issues (list of objects with criterion, observation, suggestion),
and limitations (list of strings). No markdown or commentary.
"""

GUIDED_REVISION_PROMPT = REVISION_PROMPT + """Consider the supplied critic
feedback where justified by the public structure and story. Do not treat
critic suggestions as facts or add details merely to satisfy them.
"""

JUDGE_PROMPT = """Compare two anonymous short-video story blueprints.
Use only the public creative structure, selected theme, both stories, and
experiment constraints. Judge whether each mapped event actually enacts the
prior interpretation, new information, and updated interpretation, and compare
causal coherence, filmability, originality, and production feasibility.
Do not infer a reference video, arm identity, or the process used to create
either story. A structurally valid ID mapping is not by itself a meaningful
information update. Return exactly one JSON object with preferred
(left, right, or tie), rationale (string), and criterion_notes (object with
mechanism, coherence, filmability, originality, feasibility string fields).
No markdown or commentary.
"""

CONSTRAINTS = {"target_duration_s": 30, "max_scenes": 3,
               "text_only": True, "media_generation": False}
STAGES = ("01_direct_revision_1", "02_direct_revision_2", "03_critic",
          "04_guided_revision", "05_judge_lr", "06_judge_rl")


def _compile_and_save(raw: dict[str, Any], structure: dict[str, Any],
                      theme: dict[str, Any], root: Path,
                      stage: str) -> dict[str, Any]:
    story = _compile_story(raw, structure["artifact_sha"], theme["theme_id"])
    validate_story_structure(story, structure_sha=structure["artifact_sha"],
                             theme_id=theme["theme_id"])
    _write_json(root / stage / "candidate.json", story)
    _write_json(root / stage / "validation.json", {
        "status": "PASS", "candidate_sha": json_hash(story)})
    return story


def _check_critic(value: dict[str, Any]) -> None:
    if set(value) != {"strengths", "issues", "limitations"}:
        raise ValueError("critic_fields_invalid")
    if not isinstance(value["strengths"], list) or not isinstance(
            value["limitations"], list) or not isinstance(value["issues"], list):
        raise ValueError("critic_types_invalid")
    if any(not isinstance(item, str) for item in
           value["strengths"] + value["limitations"]):
        raise ValueError("critic_types_invalid")
    for issue in value["issues"]:
        if not isinstance(issue, dict) or set(issue) != {
                "criterion", "observation", "suggestion"} or any(
                not isinstance(item, str) for item in issue.values()):
            raise ValueError("critic_issue_invalid")
    validate_creative_boundary(value)


def _check_judge(value: dict[str, Any]) -> None:
    if set(value) != {"preferred", "rationale", "criterion_notes"} or value[
            "preferred"] not in {"left", "right", "tie"} or not isinstance(
            value["rationale"], str):
        raise ValueError("judge_fields_invalid")
    notes = value["criterion_notes"]
    if not isinstance(notes, dict) or set(notes) != {
            "mechanism", "coherence", "filmability", "originality",
            "feasibility"} or any(not isinstance(item, str)
                                  for item in notes.values()):
        raise ValueError("judge_notes_invalid")


def run_critique_trial(runner: Any, structure: dict[str, Any],
                       theme: dict[str, Any], baseline_story: dict[str, Any],
                       output_dir: Path, *, model_id: str = "omni",
                       model_config_sha: str | None = None) -> dict[str, Any]:
    """Six one-shot text calls: two revisions per arm and two swapped judges."""
    validate_frozen_spec(structure)
    validate_theme_structure(theme, structure_sha=structure["artifact_sha"])
    validate_story_structure(baseline_story,
                             structure_sha=structure["artifact_sha"],
                             theme_id=theme["theme_id"])
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"trial output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    input_shas = {"structure": json_hash(structure),
                  "theme": json_hash(theme),
                  "baseline_story": json_hash(baseline_story)}
    _write_json(root / "input.json", {
        "input_shas": input_shas, "structure": structure,
        "theme": theme, "baseline_story": baseline_story,
        "constraints": CONSTRAINTS, "model_id": model_id,
        "model_config_sha": model_config_sha,
        "production_committed": False, "media_generation": "not_run"})
    shared = {"creative_structure_spec": structure, "selected_theme": theme,
              "experiment_constraints": CONSTRAINTS}
    stage = STAGES[0]
    try:
        direct_1 = _compile_and_save(_ask(runner, root, stage, REVISION_PROMPT,
                                          {**shared, "current_story": baseline_story},
                                          3072), structure, theme, root, stage)
        stage = STAGES[1]
        direct_2 = _compile_and_save(_ask(runner, root, stage, REVISION_PROMPT,
                                          {**shared, "current_story": direct_1},
                                          3072), structure, theme, root, stage)
        stage = STAGES[2]
        critique = _ask(runner, root, stage, CRITIC_PROMPT,
                        {**shared, "current_story": baseline_story}, 3072)
        _check_critic(critique)
        _write_json(root / stage / "critique.json", critique)
        _write_json(root / stage / "validation.json", {"status": "PASS"})
        stage = STAGES[3]
        guided = _compile_and_save(_ask(
            runner, root, stage, GUIDED_REVISION_PROMPT,
            {**shared, "current_story": baseline_story,
             "critic_feedback": critique}, 3072), structure, theme, root, stage)
        judges: list[dict[str, Any]] = []
        for stage, left, right in (
                (STAGES[4], direct_2, guided),
                (STAGES[5], guided, direct_2)):
            judgment = _ask(runner, root, stage, JUDGE_PROMPT,
                            {**shared, "left_story": left,
                             "right_story": right}, 1024)
            _check_judge(judgment)
            _write_json(root / stage / "judgment.json", judgment)
            _write_json(root / stage / "validation.json", {"status": "PASS"})
            judges.append(judgment)
        preferences = (judges[0]["preferred"], judges[1]["preferred"])
        result = {"status": "PASS", "model_calls_with_response": 6,
                  "input_shas": input_shas,
                  "direct_story_sha": json_hash(direct_2),
                  "guided_story_sha": json_hash(guided),
                  "judge_preferences_left_right": list(preferences),
                  "order_consistent": preferences in (("left", "right"),
                                                       ("right", "left"),
                                                       ("tie", "tie")),
                  "quality_conclusion": "exploratory_only",
                  "production_committed": False,
                  "media_generation": "not_run"}
    except BaseException as exc:
        reason = getattr(exc, "reason_code", str(exc))
        _write_json(root / stage / "validation.json", {
            "status": "FAIL", "error_type": type(exc).__name__,
            "reason": reason})
        result = {"status": "BLOCKED", "failed_stage": stage,
                  "model_calls_with_response": sum(int(
                      (root / name / "raw_response.txt").is_file())
                      for name in STAGES),
                  "in_flight_call_outcome_unknown": (
                      (root / stage / "request.json").is_file() and not
                      (root / stage / "raw_response.txt").is_file()),
                  "error_type": type(exc).__name__, "reason": reason,
                  "production_committed": False,
                  "media_generation": "not_run"}
        _write_json(root / "result.json", result)
        raise
    _write_json(root / "result.json", result)
    return result
