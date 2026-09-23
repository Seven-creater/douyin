from collections import Counter

import pytest

from src.agentic_video.reference_understanding_v2 import (
    VISUAL_PROBE_PROMPT, build_editing_graph, build_review_issues,
    validate_local_observation, validate_probe_selection,
    validate_story_audit, validate_story_graph,
)


def _fixture():
    shots = []
    for index, (section, start, end, claims) in enumerate((
        ("A", 0.0, 1.0, ["V1"]),
        ("A", 1.0, 2.0, ["V1"]),
        ("A", 2.0, 3.0, ["V1"]),
        ("B", 3.0, 4.0, ["V2"]),
        ("B", 4.0, 5.0, ["V2"]),
        ("C", 5.0, 6.0, ["V3"]),
        ("C", 6.0, 7.0, ["V3"]),
    ), start=1):
        shots.append({"shot_id": f"S{index}", "section_id": section,
                      "start_s": start, "end_s": end,
                      "duration_s": end - start,
                      "visual_claim_ids": claims,
                      "transition_in": None})
    static = {
        "source_sha": "a" * 64, "artifact_sha": "b" * 64,
        "duration_s": 7.0, "shots": shots,
        "section_bundles": [
            {"section_id": "A", "interval": [0.0, 3.0]},
            {"section_id": "B", "interval": [3.0, 5.0]},
            {"section_id": "C", "interval": [5.0, 7.0]}],
        "text_timeline": [{"claim_id": "T1", "interval": [6.0, 7.0]}],
        "measured_editing": {"structural_grammar": {}},
        "audio": {"music_beat_status": "unverified",
                  "beat_synced_status": "unverified"},
    }
    reference = {
        "claims": [{"claim_id": "V1", "predicate": "performs_action"},
                   {"claim_id": "V2", "predicate": "appears_in"},
                   {"claim_id": "V3", "predicate": "appears_in"},
                   {"claim_id": "T1", "predicate": "contains_text"}],
        "events": [{"event_id": "E1"}],
        "deterministic_timeline": {"sections": [
            {"section_id": value} for value in "ABC"]},
        "shot_storyboard": {"shot_cards": shots},
    }
    return static, reference


def test_generic_gaps_are_about_evidence_not_an_expected_theme():
    static, reference = _fixture()
    issues = build_review_issues(static, reference)
    by_id = {row["issue_id"]: row for row in issues}
    assert by_id["coarse_action_V1"]["shot_ids"] == ["S1", "S2", "S3"]
    assert by_id["terminal_statement_relation"]["shot_ids"] == ["S6", "S7"]
    assert "kick" not in str(issues).lower()
    assert "reframe" not in str(issues).lower()


def test_planner_selection_enforces_scope_and_per_issue_budget():
    static, reference = _fixture()
    issues = build_review_issues(static, reference)
    selected = {"schema_version": "reference_probe_selection_v2",
                "probes": [{"issue_id": "coarse_action_V1",
                            "shot_ids": ["S1", "S2", "S3"],
                            "reason": "action detail missing"}]}
    assert validate_probe_selection(selected, issues, static)[0]["interval"] == [
        0.0, 3.0]
    with pytest.raises(ValueError, match="probe_issue_budget_exceeded"):
        validate_probe_selection(selected, issues, static,
                                 Counter({"coarse_action_V1": 2}))
    selected["probes"][0]["shot_ids"] = ["S1", "S3"]
    with pytest.raises(ValueError, match="probe_shots_not_consecutive"):
        validate_probe_selection(selected, issues, static)


def test_visual_observation_requires_every_shot_and_adjacent_pair():
    probe = {"channel": "V", "shot_ids": ["S1", "S2", "S3"]}
    value = {"schema_version": "reference_local_observation_v2",
             "observations": [{"shot_id": sid} for sid in probe["shot_ids"]],
             "connections": [
                 {"from_shot_id": "S1", "to_shot_id": "S2"},
                 {"from_shot_id": "S2", "to_shot_id": "S3"}]}
    validate_local_observation(value, probe)
    value["observations"][0]["audible_event"] = "speech"
    with pytest.raises(ValueError, match="crosses_modality"):
        validate_local_observation(value, probe)
    assert "kick" not in VISUAL_PROBE_PROMPT.lower()


def test_editing_graph_keeps_measured_time_and_unknown_semantics():
    static, _reference = _fixture()
    static["shots"][1]["transition_in"] = {
        "transition_id": "TR1", "start_s": 0.8, "end_s": 1.0}
    record = {"probe_id": "probe_01", "media_sha": "c" * 64,
              "response": {"observations": [
                  {"shot_id": "S1", "visible_action": "action A"},
                  {"shot_id": "S2", "visible_action": "action B"}],
                  "connections": [{"from_shot_id": "S1",
                                   "to_shot_id": "S2",
                                   "observed_continuity": "uncertain"}]}}
    graph = build_editing_graph(static, [record])
    assert len(graph["shots"]) == 7
    assert len(graph["adjacent_edges"]) == 6
    assert graph["adjacent_edges"][0]["edit_start_s"] == 0.8
    assert graph["adjacent_edges"][0]["next_content_start_s"] == 1.0
    assert graph["adjacent_edges"][1]["semantic_status"] == "unobserved"
    assert graph["sections"][0]["relative_duration"] == round(3 / 7, 6)
    assert graph["audio_measurement_status"]["beat_synced_status"] == "unverified"


def test_story_graph_and_audit_never_turn_unsupported_claims_into_facts():
    static, reference = _fixture()
    issues = build_review_issues(static, reference)
    observations = [{"probe_id": "probe_01", "shot_ids": ["S1", "S2"]}]
    story = {
        "schema_version": "reference_story_graph_v2",
        "narrative_units": [
            {"unit_id": "N1", "section_ids": ["A"],
             "support_ids": ["V1"],
             "local_observation_refs": ["probe_01:S1"]}],
        "relations": [],
        "theme_hypotheses": [
            {"hypothesis_id": "H1", "support_ids": ["V1"]}],
        "unresolved_issue_ids": [],
    }
    validate_story_graph(story, reference, issues, observations)
    story["theme_hypotheses"][0]["support_ids"] = ["unseen"]
    with pytest.raises(ValueError, match="story_theme_support_invalid"):
        validate_story_graph(story, reference, issues, observations)
    story["theme_hypotheses"][0]["support_ids"] = ["V1"]
    audit = {"schema_version": "reference_story_audit_v2", "checks": [
        {"kind": kind, "id": ident, "grounded": True,
         "attribution_preserved": True, "temporal_scope_valid": True}
        for kind, ident in (("unit", "N1"), ("theme", "H1"))]}
    assert validate_story_audit(audit, story) is True
    audit["checks"][1]["grounded"] = False
    assert validate_story_audit(audit, story) is False
