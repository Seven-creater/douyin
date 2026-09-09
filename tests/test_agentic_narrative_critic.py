from __future__ import annotations

from src.agentic_video.narrative_critic import (apply_story_patches,
                                                 preflight_story_faults,
                                                 score_comprehension_answers,
                                                 validate_story_patch)
from src.agentic_video.story_planner import build_story_plan
from tests.test_agentic_narrative import valid_program
from tests.test_agentic_story_planner import _candidate


def _plan() -> dict:
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "resolution", "person", start=9, event_id="e3")]
    return build_story_plan(valid_program(), rows, theme="守护与牺牲",
                            library="guimie", target_duration_s=45.0)


def test_story_patch_allowlist():
    plan = _plan()
    assert validate_story_patch(plan, {"op": "replace", "path": "/slots/0/source",
                                       "value": plan["slots"][0]["source"]}) == []
    assert validate_story_patch(plan, {"op": "replace", "path": "/theme",
                                       "value": "篡改"})


def test_invalid_story_patch_is_rejected():
    plan = _plan()
    patched, audit = apply_story_patches(plan, [
        {"op": "replace", "path": "/slots/0/source/video", "value": ""},
        {"op": "remove", "path": "/slots/0"},
    ])
    assert patched == plan
    assert all(item["status"] == "rejected" for item in audit)


def test_comprehension_requires_four_of_five():
    result = score_comprehension_answers([
        {"question_id": i, "correct": i < 4} for i in range(5)])
    assert result["correct"] == 4
    assert result["passes"] is True


def test_preflight_story_faults_covers_causal_and_execution_failures():
    plan = _plan()
    plan["slots"][1]["source"]["entity_ids"] = ["other"]
    plan["slots"][1]["source"]["dialogue"] = [{"start_s": 4.0, "end_s": 40.0}]
    plan["slots"][1]["target_interval"] = [15.0, 20.0]
    plan["slots"][2]["role"] = "conflict"
    plan["slots"][2]["status"] = "unsupported"
    for slot in plan["slots"]:
        if slot["role"] == "resolution":
            slot["status"] = "unsupported"
    faults = preflight_story_faults(plan)
    assert {fault["type"] for fault in faults} >= {
        "entity_switch", "missing_resolution", "missing_cause",
        "subtitle_timing", "dialogue_cut"}
