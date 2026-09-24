"""No-model contracts for the parallel reference-transfer v2 path."""
from __future__ import annotations

import copy
import json

import pytest

from src.agentic_video.creative_pipeline.transfer_v2 import (
    STORY_CRITIC_PROMPT, STORY_PROMPT, THEME_CRITIC_PROMPT, THEME_PROMPT,
    run_transfer_creative_trial, validate_theme_batch, validate_timed_story,
)
from src.agentic_video.reference_transfer_v2 import (
    _analysis_errors, abstraction_payload, build_analysis_payload,
    build_reference_blueprint, inspect_bgm_asset, plan_gap_probes,
    publish_candidate_spec,
    validate_transfer_spec,
)
from src.agentic_video.reference_transfer_v2_shards import (
    global_audit_ids, global_story_payload, section_audit_ids, section_payload,
)


def fixture():
    shots = [
        {"shot_id": "s1", "section_id": "a", "start_s": 0.0, "end_s": 1.0,
         "visual_claim_ids": ["V1"], "text_claim_ids": ["T1"],
         "audio_claim_ids": []},
        {"shot_id": "s2", "section_id": "b", "start_s": 1.0, "end_s": 1.9,
         "visual_claim_ids": ["V2"], "text_claim_ids": [],
         "audio_claim_ids": []},
        {"shot_id": "s3", "section_id": "b", "start_s": 2.0, "end_s": 3.0,
         "visual_claim_ids": ["V3"], "text_claim_ids": [],
         "audio_claim_ids": []},
    ]
    static = {"source_sha": "a" * 64, "artifact_sha": "b" * 64,
              "duration_s": 3.0, "shots": shots,
              "transitions": [{"segment_id": "x1", "transition_type": "whip_pan",
                               "interval": [1.9, 2.0]}],
              "text_timeline": [{"claim_id": "T1", "observed_text": "source text",
                                 "interval": [0.0, 1.0],
                                 "timing_status": "accepted_interval_not_frame_verified"}],
              "audio": {"source_sha": "a" * 64,
                        "candidate_onsets_s": [1.1],
                        "music_beat_status": "unverified",
                        "beat_synced_status": "unverified"},
              "measured_editing": {"pace_curve": []}}
    local = {"observations": [{"probe_id": "p1", "shot_ids": ["s1", "s2"],
                               "sampling": {"sampling_verified": True,
                                            "actual_frame_timestamps_absolute_s": [0.5, 1.5]},
                               "validation": {"change_findings": []},
                               "response": {"shots": [], "connections": []}}]}
    analysis = {"schema_version": "reference_blueprint_analysis_v2",
                "theme_stance": {"position": "Challenge a narrow judgment",
                                 "shot_ids": ["s1", "s2"], "statement_ids": ["T1"],
                                 "alternative": "unknown", "status": "provisional"},
                "viewer_change": {"prior": "narrow judgment", "later": "revised",
                                  "shot_ids": ["s1", "s2"], "statement_ids": ["T1"],
                                  "status": "provisional"},
                "ending": {"relation": "adds ordinary limit", "tone": "light",
                           "shot_ids": ["s3"], "statement_ids": [],
                           "status": "provisional"},
                "narrative_units": [{"unit_id": "u1", "shot_ids": ["s1", "s2"],
                                     "contribution": "view changes", "status": "provisional"}],
                "shots": [{"shot_id": sid, "information_added": f"information {sid}",
                           "action_phase": "unknown", "support_probe_ids": [],
                           "statement_ids": ["T1"] if sid == "s1" else [],
                           "shot_size": None, "camera_angle": None,
                           "camera_motion": None, "motion_speed": None,
                           "status": "provisional", "technique_status": "insufficient"}
                          for sid in ("s1", "s2", "s3")],
                "edges": [{"from_shot_id": left, "to_shot_id": right,
                           "information_added": "new information",
                           "function_hypothesis": "unknown",
                           "support_probe_ids": [], "status": "provisional"}
                          for left, right in (("s1", "s2"), ("s2", "s3"))],
                "unknowns": []}
    audit_ids = (["theme_stance", "viewer_change", "ending"] +
                 [f"shot:{sid}" for sid in ("s1", "s2", "s3")] +
                 [f"technique:{sid}" for sid in ("s1", "s2", "s3")] +
                 ["edge:s1->s2", "edge:s2->s3"])
    audit = {"schema_version": "reference_blueprint_audit_v2",
             "checks": [{"id": item, "verdict": "supported", "reason": "test"}
                        for item in audit_ids],
             "media_truth_guaranteed": False}
    return static, local, analysis, audit


def test_gap_planner_only_requests_uncovered_neighbor_and_keeps_cap():
    static, local, _, _ = fixture()
    probes = plan_gap_probes(static, local)
    assert len(probes) == 1
    assert probes[0]["shot_ids"] == ["s2", "s3"]
    assert probes[0]["channel"] == "AV"
    assert plan_gap_probes(static, local, limit=0) == []


def test_payload_keeps_text_attributed_and_beat_unverified():
    static, local, _, _ = fixture()
    payload = build_analysis_payload(static, local)
    assert payload["text_statements"][0]["source_type"] == (
        "attributed_on_screen_text")
    assert payload["audio"]["beat_synced_status"] == "unverified"
    assert payload["transitions"][0]["interval"] == [1.9, 2.0]
    assert "prior_reading" not in payload


def test_duplicate_edge_cannot_be_promoted_or_hidden():
    static, local, analysis, audit = fixture()
    analysis["edges"].append(copy.deepcopy(analysis["edges"][-1]))
    assert "edge_coverage_invalid" in _analysis_errors(analysis, static, {"p1"})
    blueprint = build_reference_blueprint(static, local, analysis, audit)
    assert blueprint["story_candidate_ready"]
    assert not blueprint["editing_candidate_ready"]
    assert not blueprint["production_release_allowed"]


def test_model_self_check_separates_story_and_editing_readiness():
    static, local, analysis, audit = fixture()
    audit["checks"][-1]["verdict"] = "insufficient"
    blueprint = build_reference_blueprint(static, local, analysis, audit)
    assert blueprint["story_candidate_ready"]
    assert not blueprint["editing_candidate_ready"]
    assert len(blueprint["shots"]) == 3
    assert len(blueprint["transitions"]) == 1
    assert blueprint["audio"]["beat_synced_status"] == "unverified"
    assert len(abstraction_payload(blueprint)["slots"]) == 3


def test_missing_narrative_audit_still_blocks_story():
    static, local, analysis, audit = fixture()
    audit["checks"] = [row for row in audit["checks"]
                       if row["id"] != "theme_stance"]
    blueprint = build_reference_blueprint(static, local, analysis, audit)
    assert "audit_coverage_invalid" in blueprint["validation_issues"]
    assert not blueprint["story_candidate_ready"]


def test_transfer_spec_preserves_exact_slots_but_rejects_surface_and_beat():
    static, local, analysis, audit = fixture()
    blueprint = build_reference_blueprint(static, local, analysis, audit)
    spec = {"schema_version": "creative_transfer_spec_v2",
            "theme_contract": {"stance": "Do not reduce a person to one label",
                               "audience_prior": "narrow inference",
                               "audience_update": "more specific assessment",
                               "ending_relation": "ordinary limitation",
                               "tone": "light"},
            "story_beats": [{"beat_id": "B1", "slot_ids": ["slot_01", "slot_02", "slot_03"],
                             "information_task": "present a judgment"}],
            "edit_slots": [{"slot_id": f"slot_{i:02d}",
                            "start_s": row["start_s"], "end_s": row["end_s"],
                            "information_task": "new story task",
                            "verified_techniques": [], "unresolved": []}
                           for i, row in enumerate(static["shots"], 1)],
            "edit_edges": [{"from_slot_id": "slot_01", "to_slot_id": "slot_02",
                            "information_relation": "adds evidence",
                            "editing_function": "progression", "status": "supported"},
                           {"from_slot_id": "slot_02", "to_slot_id": "slot_03",
                            "information_relation": "adds context",
                            "editing_function": "expansion", "status": "supported"}],
            "transition_slots": [{"transition_id": "transition_01",
                                  "start_s": 1.9, "end_s": 2.0,
                                  "type": "whip_pan"}],
            "free_slots": ["domain", "characters"],
            "audio_policy": "reuse_reference_bgm_replace_voice",
            "limitations": []}
    assert validate_transfer_spec(spec, blueprint, ["source text"]) == []
    candidate = publish_candidate_spec(spec, blueprint, ["source text"])
    assert candidate["status"] == "model_checked_candidate"
    assert not candidate["production_release_allowed"]
    wrong = copy.deepcopy(spec)
    wrong["edit_slots"][1]["end_s"] = 1.8
    assert "shot_slots_not_exact" in validate_transfer_spec(
        wrong, blueprint, [])
    wrong = copy.deepcopy(spec)
    wrong["theme_contract"]["stance"] = "source text"
    assert "reference_surface_leak" in validate_transfer_spec(
        wrong, blueprint, ["source text"])
    wrong = copy.deepcopy(spec)
    wrong["edit_slots"][0]["verified_techniques"] = ["beat_sync"]
    assert "unverified_beat_claim" in validate_transfer_spec(
        wrong, blueprint, [])
    wrong = copy.deepcopy(spec)
    wrong["edit_edges"][0]["from_slot_id"] = "slot_03"
    assert "edit_edges_not_exact" in validate_transfer_spec(
        wrong, blueprint, [])


def test_timed_story_requires_every_exact_slot_and_transition():
    static, _, _, _ = fixture()
    spec = {"edit_slots": [{"slot_id": f"slot_{i:02d}",
                             "start_s": row["start_s"], "end_s": row["end_s"]}
                            for i, row in enumerate(static["shots"], 1)],
            "transition_slots": [{"transition_id": "transition_01",
                                  "start_s": 1.9, "end_s": 2.0,
                                  "type": "whip_pan"}]}
    story = {"schema_version": "timed_story_candidate_v2", "theme_id": "T1",
             "feasible": True,
             "slots": [{**row, "event": f"event {i}",
                        "new_information": f"fact {i}", "visual_action": "act"}
                       for i, row in enumerate(spec["edit_slots"])],
             "transition_slots": spec["transition_slots"]}
    assert validate_timed_story(story, spec, "T1") == []
    story["slots"][0]["cues"] = [{"kind": "voiceover", "text": "new line",
                                  "start_s": 0.0, "end_s": 1.2}]
    assert "story_cue_outside_slot_or_invalid" in validate_timed_story(
        story, spec, "T1")
    story["slots"][0]["cues"] = []
    story["slots"].pop()
    assert "story_slots_not_exact" in validate_timed_story(story, spec, "T1")


def test_theme_batch_is_small_and_versioned():
    value = {"schema_version": "transfer_theme_candidates_v2",
             "themes": [{"theme_id": f"T{i}", "domain": f"d{i}",
                         "premise": "p", "stance": "s", "audience_prior": "a",
                         "evidence_mechanism": "e", "audience_update": "u",
                         "ending_relation": "r", "tone": "t"}
                        for i in range(1, 4)]}
    assert validate_theme_batch(value) == []
    value["themes"].pop()
    assert "theme_count_or_ids_invalid" in validate_theme_batch(value)


def test_missing_bgm_never_becomes_verified_stem(tmp_path):
    assert inspect_bgm_asset(tmp_path / "source.mp4", None)["status"] == "missing"
    assert inspect_bgm_asset(tmp_path / "source.mp4", tmp_path / "bgm.m4a")[
        "status"] == "missing"


def test_creative_trial_is_parallel_candidate_only(tmp_path):
    spec = {"schema_version": "creative_transfer_spec_v2",
            "status": "model_checked_candidate", "artifact_sha": "a" * 64,
            "edit_slots": [{"slot_id": "slot_01", "start_s": 0.0, "end_s": 1.0},
                           {"slot_id": "slot_02", "start_s": 1.0, "end_s": 2.0}],
            "transition_slots": [],
            "theme_contract": {"stance": "narrow judgments need wider evidence"}}
    themes = {"schema_version": "transfer_theme_candidates_v2", "themes": [
        {"theme_id": f"T{i}", "domain": f"domain {i}", "premise": "p",
         "stance": "s", "audience_prior": "a", "evidence_mechanism": "e",
         "audience_update": "u", "ending_relation": "r", "tone": "t"}
        for i in range(1, 4)]}
    critique = {"schema_version": "transfer_theme_critique_v2", "checks": [
        {"theme_id": f"T{i}", "stance_preserved": i == 2,
         "mechanism_preserved": i == 2, "independent_story": i == 2,
         "reason": "test"} for i in range(1, 4)]}
    story = {"schema_version": "timed_story_candidate_v2", "feasible": True,
             "theme_id": "T2", "transition_slots": [], "reason": "fits",
             "logline": "A new story", "ending": "light",
             "slots": [{**row, "event": f"event {i}",
                        "new_information": f"fact {i}",
                        "visual_action": "filmable action", "cues": []}
                       for i, row in enumerate(spec["edit_slots"])]}
    assessment = {"schema_version": "timed_story_critique_v2",
                  "stance_preserved": True, "causal_progression": True,
                  "all_slots_filmable": True,
                  "adjacent_information_distinct": True,
                  "ending_preserved": True, "issues": []}

    class Runner:
        def ask(self, prompt, **_):
            for prefix, value in ((THEME_PROMPT, themes),
                                  (THEME_CRITIC_PROMPT, critique),
                                  (STORY_PROMPT, story),
                                  (STORY_CRITIC_PROMPT, assessment)):
                if prompt.startswith(prefix):
                    return json.dumps(value)
            raise AssertionError("unknown prompt")

    result = run_transfer_creative_trial(Runner(), spec, tmp_path / "trial",
                                         forbidden_markers=["source text"])
    assert result["status"] == "model_checked_candidate"
    assert result["selected_theme_id"] == "T2"
    assert result["media_generation"] == "not_run"
    assert not result["production_committed"]
    critique["checks"][1]["stance_preserved"] = False
    with pytest.raises(ValueError, match="no_theme_passed_critique"):
        run_transfer_creative_trial(Runner(), spec, tmp_path / "blocked",
                                    forbidden_markers=["source text"])
    assert (tmp_path / "blocked" / "result.json").is_file()


def test_shards_partition_shots_and_cross_section_edges():
    static, local, _, _ = fixture()
    first = section_payload(static, local, "a")
    second = section_payload(static, local, "b")
    assert [row["shot_id"] for row in first["shots"]] == ["s1"]
    assert [row["shot_id"] for row in second["shots"]] == ["s2", "s3"]
    assert section_audit_ids(static, "b") == [
        "shot:s2", "shot:s3", "technique:s2", "technique:s3", "edge:s2->s3"]
    assert global_audit_ids(static) == ["theme_stance", "viewer_change",
                                         "ending", "edge:s1->s2"]
    payload = global_story_payload(static, [])
    assert payload["cross_edges"] == [{"from_shot_id": "s1", "to_shot_id": "s2"}]
