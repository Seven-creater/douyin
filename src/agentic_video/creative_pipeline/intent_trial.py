"""Small theme-to-story trial driven only by a public intent brief."""
from __future__ import annotations

import json
from typing import Any

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.reference_intent_v1 import BRIEF_VERSION


THEME_PROMPT = """From the public creative story brief, propose exactly
three original, filmable short-video themes in different domains. Preserve
the communicative goal and the role of concrete evidence in changing an
audience's understanding. A surprise alone is not sufficient. Do not infer
or recreate a reference video. The listed beats and tone are preferences,
not mandatory plot roles. Return JSON only:
{"schema_version":"intent_theme_batch_v1","themes":[
{"theme_id":"T1","domain":"...","premise":"...",
"communicative_goal":"...","audience_prior":"...",
"evidence_mechanism":"...","audience_update":"...","tone":"..."}]}
Use IDs T1, T2, T3 exactly. Input: """

THEME_CRITIC_PROMPT = """Independently review each proposed theme against
the supplied public brief. Check the actual communicative position,
credible evidence mechanism, filmability, and independence from a possible
reference source. A generic reversal is not enough. Only after those hard
checks, score similarity of optional information-order preferences from
0 to 3; this soft score must not reject an otherwise good theme. Return
JSON only: {"schema_version":"intent_theme_critique_v1","checks":[
{"theme_id":"T1","goal_match":false,"evidence_logic":false,
"original":false,"filmable":false,"structure_preference":0,
"reason_codes":[]}]}. One check each for T1, T2, T3. Input: """

STORY_PROMPT = """Write one concise, original and filmable story synopsis
for the selected theme and public brief. Show concrete actions and how their
information changes the audience's understanding. The brief's beat order
and tone are preferences, not fixed acts; do not copy a source video or
invent a shot-by-shot timing plan. A different ending is allowed if it
preserves the communicative goal. Return JSON only:
{"schema_version":"intent_story_synopsis_v1","theme_id":"T1",
"logline":"...","events":[{"order":1,"action":"...",
"new_information":"..."}],"ending":"...","tone":"..."}.
Input: """

STORY_CRITIC_PROMPT = """Review each story synopsis against the supplied
public brief and selected theme, never against an unavailable reference
video. Hard checks: same communicative goal, credible information-changing
evidence, an independent story, and filmable actions. Rate optional
functional-beat similarity 0 to 3 only for ranking. Do not reject solely
for different shot count, seconds, section count, or ending style. Return
JSON only: {"schema_version":"intent_story_critique_v1","checks":[
{"theme_id":"T1","goal_match":false,"evidence_logic":false,
"original":false,"filmable":false,"structure_preference":0,
"reason_codes":[]}]}. One check per supplied story. Input: """

HARD_KEYS = ("goal_match", "evidence_logic", "original", "filmable")


def validate_brief_input(brief: dict[str, Any]) -> None:
    if (brief.get("schema_version") != BRIEF_VERSION or brief.get("status") !=
            "model_checked_candidate" or brief.get(
                "production_release_allowed") is not False):
        raise ValueError("public_brief_not_candidate")
    validate_creative_boundary(brief)


def validate_themes(batch: dict[str, Any]) -> list[str]:
    themes = batch.get("themes") or []
    if batch.get("schema_version") != "intent_theme_batch_v1" or [
            row.get("theme_id") for row in themes] != ["T1", "T2", "T3"]:
        return ["theme_count_or_schema_invalid"]
    required = {"theme_id", "domain", "premise", "communicative_goal",
                "audience_prior", "evidence_mechanism", "audience_update",
                "tone"}
    return ["theme_fields_invalid"] if any(set(row) != required or any(
        not isinstance(row[key], str) or not row[key].strip()
        for key in required) for row in themes) else []


def select_themes(critique: dict[str, Any]) -> list[str]:
    checks = critique.get("checks") or []
    if (critique.get("schema_version") != "intent_theme_critique_v1" or
            [row.get("theme_id") for row in checks] != ["T1", "T2", "T3"] or
            any(not isinstance(row.get("structure_preference"), int) or
                not 0 <= row["structure_preference"] <= 3 for row in checks)):
        raise ValueError("theme_critique_invalid")
    passed = [row for row in checks if all(row.get(key) is True
                                           for key in HARD_KEYS)]
    return [row["theme_id"] for row in sorted(
        passed, key=lambda row: (-row["structure_preference"],
                                 row["theme_id"]))[:2]]


def validate_story(story: dict[str, Any], theme_id: str) -> list[str]:
    if (story.get("schema_version") != "intent_story_synopsis_v1" or
            story.get("theme_id") != theme_id):
        return ["story_schema_or_parent_invalid"]
    events = story.get("events") or []
    if (not events or [row.get("order") for row in events] != list(
            range(1, len(events)+1)) or any(
                not row.get("action") or not row.get("new_information")
                for row in events)):
        return ["story_events_invalid"]
    if not all(isinstance(story.get(key), str) and story[key].strip()
               for key in ("logline", "ending", "tone")):
        return ["story_fields_missing"]
    validate_creative_boundary(story)
    return []


def review_stories(critique: dict[str, Any], theme_ids: list[str]) -> dict:
    checks = critique.get("checks") or []
    if (critique.get("schema_version") != "intent_story_critique_v1" or
            [row.get("theme_id") for row in checks] != theme_ids or
            any(not isinstance(row.get("structure_preference"), int) or
                not 0 <= row["structure_preference"] <= 3 for row in checks)):
        raise ValueError("story_critique_invalid")
    passed = [row for row in checks if all(row.get(key) is True
                                           for key in HARD_KEYS)]
    ranked = sorted(passed, key=lambda row: (-row["structure_preference"],
                                            row["theme_id"]))
    return {"status": "model_checked_candidate" if ranked else "blocked",
            "selected_theme_id": ranked[0]["theme_id"] if ranked else None,
            "hard_pass_ids": [row["theme_id"] for row in ranked],
            "all_checks": checks, "screenplay_generation": "not_run",
            "media_generation": "not_run", "production_committed": False}


def has_reference_surface(value: object, markers: list[str]) -> bool:
    serialized = json.dumps(value, ensure_ascii=False).casefold()
    return any(marker and marker.casefold() in serialized for marker in markers)
