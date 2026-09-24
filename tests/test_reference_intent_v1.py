from __future__ import annotations

import copy

import pytest

from src.agentic_video.creative_pipeline.intent_trial import (
    select_themes, validate_brief_input, validate_story, validate_themes,
)
from src.agentic_video.reference_intent_v1 import (
    ABSTRACT_PROMPT, AUDIT_PROMPT, INTENT_PROMPT, PROBE_PROMPT,
    build_intent, intent_payload, publish_story_brief, validate_intent,
    validate_intent_audit,
)


def fixture():
    static = {"source_sha": "media_sha", "artifact_sha": "static_sha",
              "duration_s": 10.0, "audio_transcript_candidate": None,
              "section_bundles": [
                  {"section_id": "s1", "interval": [0.0, 4.0],
                   "events": [{"event_id": "EV1"}],
                   "visual_observations": [{"claim_id": "V1",
                                            "object": "a person waits"}],
                   "text_statements": [{"claim_id": "T1",
                                        "observed_text": "some words",
                                        "interval": [0.0, 4.0]}],
                   "audio_observations": []},
                  {"section_id": "s2", "interval": [4.0, 10.0],
                   "events": [{"event_id": "EV2"}],
                   "visual_observations": [{"claim_id": "V2",
                                            "object": "a person acts"}],
                   "text_statements": [], "audio_observations": []}]}
    analysis = {"schema_version": "reference_intent_analysis_v1",
                "communicative_goal": {"statement": "Observe a richer picture",
                                       "support_ids": ["T1", "V2"]},
                "audience_prior": {"interpretation": "One narrow view",
                                   "support_ids": ["T1"]},
                "evidence_mechanism": {"description": "An action adds evidence",
                                       "support_ids": ["EV2", "V2"]},
                "audience_update": {"interpretation": "A wider view",
                                    "support_ids": ["V2"]},
                "ending": {"effect": "unknown", "support_ids": []},
                "tone": {"description": "unknown"},
                "beat_preferences": [{"function": "Show new evidence",
                                      "support_ids": ["V2"]}],
                "limitations": ["ending effect uncertain"]}
    audit = {"schema_version": "reference_intent_audit_v1",
             "checks": [{"id": key, "verdict": "supported", "reason": "ok"}
                        for key in ("communicative_goal", "audience_path",
                                    "evidence_mechanism", "attribution")],
             "critical_conflict": {"present": False, "reason": ""},
             "probe_request": None}
    return static, analysis, audit


def test_payload_keeps_attributed_text_and_no_prior_story():
    static, _, _ = fixture()
    payload = intent_payload(static)
    assert payload["sections"][0]["text_statements"][0][
        "source_type"] == "attributed_on_screen_text"
    assert "analysis" not in payload
    assert payload["audio_transcript_verified"] is False


def test_optional_ending_does_not_block_core_intent():
    static, analysis, audit = fixture()
    intent = build_intent(static, analysis, audit,
                          source_run="run", model_calls=[])
    assert intent["story_candidate_ready"] is True
    assert intent["editing_candidate_ready"] is False
    assert intent["production_release_allowed"] is False


def test_missing_core_or_bad_ref_blocks_even_if_audit_positive():
    static, analysis, audit = fixture()
    analysis["communicative_goal"]["statement"] = "unknown"
    assert "communicative_goal_missing" in validate_intent(analysis, static)
    analysis["communicative_goal"]["statement"] = "Known"
    analysis["communicative_goal"]["support_ids"] = ["DOES_NOT_EXIST"]
    assert "communicative_goal_support_invalid" in validate_intent(
        analysis, static)
    assert not build_intent(static, analysis, audit, source_run="run",
                            model_calls=[])["story_candidate_ready"]


def test_audit_insufficient_or_conflict_blocks():
    static, analysis, audit = fixture()
    audit["checks"][0]["verdict"] = "insufficient"
    assert not build_intent(static, analysis, audit, source_run="run",
                            model_calls=[])["story_candidate_ready"]
    audit["checks"][0]["verdict"] = "supported"
    audit["critical_conflict"]["present"] = True
    assert not build_intent(static, analysis, audit, source_run="run",
                            model_calls=[])["story_candidate_ready"]
    audit["checks"] = audit["checks"][:-1]
    assert "audit_coverage_or_verdict_invalid" in validate_intent_audit(audit)


def test_brief_is_abstract_and_no_timed_slots():
    static, analysis, audit = fixture()
    intent = build_intent(static, analysis, audit,
                          source_run="run", model_calls=[])
    brief = {"schema_version": "creative_story_brief_v1",
             "communicative_goal": "A person merits a fuller assessment",
             "audience_prior": "A narrow inference",
             "evidence_mechanism": "Several concrete actions broaden evidence",
             "audience_update": "A revised understanding",
             "tone": None, "beat_preferences": ["Initial view", "New evidence"],
             "free_slots": ["domain", "relationship"], "limitations": []}
    brief_audit = {"schema_version": "creative_story_brief_audit_v1",
                   "goal_preserved": True, "evidence_logic_preserved": True,
                   "source_surface_absent": True, "structure_optional": True,
                   "reason_codes": []}
    published = publish_story_brief(brief, intent, brief_audit, ["source words"])
    validate_brief_input(published)
    assert "edit_slots" not in published
    assert published["production_release_allowed"] is False
    leaked = copy.deepcopy(brief)
    leaked["communicative_goal"] = "source words"
    with pytest.raises(ValueError, match="reference_surface_leak"):
        publish_story_brief(leaked, intent, brief_audit, ["source words"])
    copied = copy.deepcopy(brief)
    copied["communicative_goal"] = analysis["communicative_goal"]["statement"]
    with pytest.raises(ValueError, match="private_sentence_reused"):
        publish_story_brief(copied, intent, brief_audit, [])
    no_slots = copy.deepcopy(brief)
    no_slots["free_slots"] = []
    with pytest.raises(ValueError, match="brief_free_slots_missing"):
        publish_story_brief(no_slots, intent, brief_audit, [])


def test_theme_hard_checks_before_soft_structure_rank():
    critique = {"schema_version": "intent_theme_critique_v1",
                "checks": [
                    {"theme_id": "T1", "goal_match": False,
                     "evidence_logic": True, "original": True,
                     "filmable": True, "structure_preference": 3},
                    {"theme_id": "T2", "goal_match": True,
                     "evidence_logic": True, "original": True,
                     "filmable": True, "structure_preference": 1},
                    {"theme_id": "T3", "goal_match": True,
                     "evidence_logic": True, "original": True,
                     "filmable": True, "structure_preference": 2}]}
    assert select_themes(critique) == ["T3", "T2"]


def test_story_needs_progression_but_no_exact_timing():
    story = {"schema_version": "intent_story_synopsis_v1", "theme_id": "T1",
             "logline": "New story", "events": [
                 {"order": 1, "action": "First action",
                  "new_information": "Initial view"},
                 {"order": 2, "action": "Second action",
                  "new_information": "New evidence"}],
             "ending": "Open ending", "tone": "warm"}
    assert validate_story(story, "T1") == []
    story["events"][1]["order"] = 4
    assert "story_events_invalid" in validate_story(story, "T1")


def test_no_user_gold_or_fixed_bias_template_in_perception_prompts():
    prompts = INTENT_PROMPT + AUDIT_PROMPT + PROBE_PROMPT + ABSTRACT_PROMPT
    for marker in ("打破偏见", "老人厨师", "不会剪脚指甲", "人们常常"):
        assert marker not in prompts
