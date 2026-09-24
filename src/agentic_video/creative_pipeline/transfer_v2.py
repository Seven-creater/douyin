"""Parallel, candidate-only theme and timed-story trial for transfer spec v2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.creative_pipeline.real_text_baseline import _ask, _write_json
from src.agentic_video.manifest import json_hash


THEME_PROMPT = """Create exactly three ORIGINAL short-video theme candidates
from the supplied public transfer specification. Preserve its communicative
stance, change in audience understanding, and ending relation; do not merely
use the generic fact that some explanation changes. Explore distinct domains
and causal situations. Do not reconstruct a reference video or make cosmetic
replacements of source characters and actions. Return JSON only:
{"schema_version":"transfer_theme_candidates_v2","themes":[
{"theme_id":"T1","domain":"...","premise":"...","stance":"...",
"audience_prior":"...","evidence_mechanism":"...",
"audience_update":"...","ending_relation":"...","tone":"..."}]}
Input: """

THEME_CRITIC_PROMPT = """Evaluate each theme against the supplied public
transfer specification, not against an imagined reference video. A surprise
or information update by itself is insufficient when the communicative stance
differs. Check that each premise offers its own filmable causal events and is
not just a label swap. Do not invent source details. Return JSON only:
{"schema_version":"transfer_theme_critique_v2","checks":[
{"theme_id":"T1","stance_preserved":false,"mechanism_preserved":false,
"independent_story":false,"reason":"..."}]}
Input: """

STORY_PROMPT = """Create one original, filmable, time-constrained story for
the selected theme. Fill EVERY content edit slot with a distinct concrete
event or information-bearing image. Preserve the exact start/end times and
transition slots supplied by the public transfer spec. Preserve supported
adjacent edit-edge information relations with new story events. The middle sequence
must have observable progression rather than repeated generic action labels.
The original BGM timing is retained later, but narration, dialogue and
on-screen wording must be newly authored for this story. Do not force a new
action into an impossible duration; if the theme cannot
fit, return feasible=false with a reason and empty slots. On-screen text in
the new story is authored content, not physical proof by itself. No reference
people, activity, objects, or quoted text. Return JSON only:
{"schema_version":"timed_story_candidate_v2","feasible":true,
"reason":"...","logline":"...","theme_id":"T1",
"slots":[{"slot_id":"slot_01","start_s":0.0,"end_s":0.0,
"event":"...","new_information":"...","visual_action":"...",
"cues":[{"kind":"voiceover|dialogue|on_screen_text","text":"...",
"start_s":0.0,"end_s":0.0}]}],
"ending":"...","transition_slots":[]}
Input: """

STORY_CRITIC_PROMPT = """Independently check the timed story against the
public transfer specification and selected theme. Check the theme stance,
causal information progression, ending relation, filmability of every exact
time slot, and whether adjacent slots satisfy the specified information
relations rather than only adding non-duplicate content. Do not
compare it with an unavailable original reference. Return JSON only:
{"schema_version":"timed_story_critique_v2","stance_preserved":false,
"causal_progression":false,"all_slots_filmable":false,
"adjacent_information_distinct":false,"ending_preserved":false,
"issues":[]}
Input: """


def _surface_leaks(value: object, markers: list[str]) -> bool:
    serialized = json.dumps(value, ensure_ascii=False).casefold()
    return any(marker and marker.casefold() in serialized for marker in markers)


def validate_theme_batch(value: dict[str, Any]) -> list[str]:
    if value.get("schema_version") != "transfer_theme_candidates_v2":
        return ["theme_schema_invalid"]
    themes = value.get("themes") or []
    if len(themes) != 3 or [row.get("theme_id") for row in themes] != [
            "T1", "T2", "T3"]:
        return ["theme_count_or_ids_invalid"]
    required = {"theme_id", "domain", "premise", "stance", "audience_prior",
                "evidence_mechanism", "audience_update", "ending_relation", "tone"}
    return ["theme_fields_invalid"] if any(set(row) != required or any(
        not isinstance(row[key], str) or not row[key].strip()
        for key in required) for row in themes) else []


def validate_timed_story(value: dict[str, Any], spec: dict[str, Any],
                         theme_id: str) -> list[str]:
    if value.get("schema_version") != "timed_story_candidate_v2" or value.get(
            "theme_id") != theme_id:
        return ["story_schema_or_parent_invalid"]
    if value.get("feasible") is False:
        return ["story_declared_infeasible"]
    expected = [(row["slot_id"], row["start_s"], row["end_s"])
                for row in spec["edit_slots"]]
    actual = [(row.get("slot_id"), row.get("start_s"), row.get("end_s"))
              for row in value.get("slots") or []]
    errors = []
    if actual != expected:
        errors.append("story_slots_not_exact")
    if value.get("transition_slots") != spec["transition_slots"]:
        errors.append("story_transitions_not_exact")
    if any(not row.get("event") or not row.get("new_information") or not row.get(
            "visual_action") for row in value.get("slots") or []):
        errors.append("story_slot_content_missing")
    for slot in value.get("slots") or []:
        for cue in slot.get("cues") or []:
            if (cue.get("kind") not in {"voiceover", "dialogue", "on_screen_text"}
                    or not isinstance(cue.get("text"), str) or not cue["text"].strip()
                    or not isinstance(cue.get("start_s"), (int, float))
                    or not isinstance(cue.get("end_s"), (int, float))
                    or not slot["start_s"] <= cue["start_s"] < cue["end_s"] <= slot["end_s"]):
                errors.append("story_cue_outside_slot_or_invalid")
    return errors


def run_transfer_creative_trial(runner: Any, spec: dict[str, Any],
                                output_dir: Path, *, forbidden_markers: list[str]
                                ) -> dict[str, Any]:
    """Four text calls at most; never publishes to the old production DAG."""
    if spec.get("schema_version") != "creative_transfer_spec_v2" or spec.get(
            "status") != "model_checked_candidate":
        raise ValueError("transfer_spec_not_candidate")
    validate_creative_boundary(spec)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("creative_trial_output_not_empty")
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "input.json", {"spec_sha": spec["artifact_sha"],
               "production_committed": False, "media_generation": "not_run"})
    stage = "01_themes"
    try:
        themes = _ask(runner, root, stage, THEME_PROMPT,
                      {"creative_transfer_spec": spec}, 3072)
        problems = validate_theme_batch(themes)
        validate_creative_boundary(themes)
        if _surface_leaks(themes, forbidden_markers):
            problems.append("reference_surface_leak")
        _write_json(root / stage / "candidates.json", themes)
        if problems:
            raise ValueError(",".join(problems))

        stage = "02_theme_critique"
        critique = _ask(runner, root, stage, THEME_CRITIC_PROMPT,
                        {"creative_transfer_spec": spec, "themes": themes}, 2048)
        _write_json(root / stage / "critique.json", critique)
        if critique.get("schema_version") != "transfer_theme_critique_v2" or [
                row.get("theme_id") for row in critique.get("checks") or []] != [
                    "T1", "T2", "T3"]:
            raise ValueError("theme_critique_coverage_invalid")
        passed = [row["theme_id"] for row in critique["checks"] if all(
            row.get(key) is True for key in (
                "stance_preserved", "mechanism_preserved", "independent_story"))]
        if not passed:
            raise ValueError("no_theme_passed_critique")
        chosen = next(row for row in themes["themes"] if row["theme_id"] == passed[0])

        stage = "03_timed_story"
        story = _ask(runner, root, stage, STORY_PROMPT,
                     {"creative_transfer_spec": spec, "selected_theme": chosen}, 6144)
        _write_json(root / stage / "candidate.json", story)
        problems = validate_timed_story(story, spec, chosen["theme_id"])
        validate_creative_boundary(story)
        if _surface_leaks(story, forbidden_markers):
            problems.append("reference_surface_leak")
        if problems:
            raise ValueError(",".join(problems))

        stage = "04_story_critique"
        assessment = _ask(runner, root, stage, STORY_CRITIC_PROMPT,
                          {"creative_transfer_spec": spec,
                           "selected_theme": chosen, "timed_story": story}, 2048)
        _write_json(root / stage / "critique.json", assessment)
        criteria = ("stance_preserved", "causal_progression", "all_slots_filmable",
                    "adjacent_information_distinct", "ending_preserved")
        if assessment.get("schema_version") != "timed_story_critique_v2" or not all(
                assessment.get(key) is True for key in criteria):
            raise ValueError("story_critique_not_passed")
        result = {"schema_version": "transfer_creative_trial_v2",
                  "status": "model_checked_candidate",
                  "spec_sha": spec["artifact_sha"], "selected_theme_id": chosen["theme_id"],
                  "selected_theme_sha": json_hash(chosen), "story_sha": json_hash(story),
                  "model_calls": 4, "screenplay_generation": "not_run",
                  "media_generation": "not_run", "production_committed": False}
    except Exception as exc:
        result = {"schema_version": "transfer_creative_trial_v2",
                  "status": "blocked", "stage": stage,
                  "reason": str(exc), "spec_sha": spec["artifact_sha"],
                  "screenplay_generation": "not_run", "media_generation": "not_run",
                  "production_committed": False}
        _write_json(root / "result.json", result)
        raise
    _write_json(root / "result.json", result)
    return result
