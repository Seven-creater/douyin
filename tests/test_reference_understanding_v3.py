from __future__ import annotations

import copy

import pytest

from src.agentic_video.reference_understanding_v3 import (
    MAX_LOCAL_MEDIA_CALLS, V_PROMPT, _counterfactual_sections,
    _observation_view, _probe_payload, build_editing_graph, plan_probes,
    sampling_from_runner, select_neutral_verification,
    validate_local_observation, validate_probe_plan,
)


def fixture():
    specs = [
        ("A", 0., 1., ["V0"], ["E0"]),
        ("B", 1., 2., ["V1"], ["E1"]),
        ("B", 2., 3., ["V1"], ["E1"]),
        ("B", 3., 4., ["V1"], ["E1"]),
        ("B", 4., 5., ["V1"], ["E1"]),
        ("B", 5., 6., ["V1"], ["E1"]),
        ("B", 6., 7., ["V2"], ["E1"]),
        ("B", 7., 8., ["V3"], ["E1"]),
        ("C", 8., 8.1, ["V4"], ["E2"]),
        ("C", 8.1, 9., ["V5"], ["E3"]),
        ("C", 9., 10., ["V6"], ["E4"]),
        ("C", 10., 11., ["V7"], ["E5"]),
    ]
    shots = [{"shot_id": f"s{i}", "section_id": sec, "start_s": start,
              "end_s": end, "duration_s": end-start,
              "visual_claim_ids": vc, "text_claim_ids": ["T1"] if i == 11 else [],
              "event_ids": ev, "transition_in": None}
             for i, (sec, start, end, vc, ev) in enumerate(specs)]
    static = {"source_sha": "a"*64, "artifact_sha": "b"*64,
              "shots": shots, "transitions": [], "duration_s": 11.,
              "section_bundles": [
                  {"section_id": "A", "interval": [0., 1.],
                   "visual_observations": [], "text_statements": []},
                  {"section_id": "B", "interval": [1., 8.],
                   "visual_observations": [], "text_statements": []},
                  {"section_id": "C", "interval": [8., 11.],
                   "visual_observations": [], "text_statements": [
                       {"claim_id": "T1", "observed_text": "last line"}]}],
              "text_timeline": [{"claim_id": "T1", "observed_text": "last line",
                                 "interval": [10., 11.]}],
              "measured_editing": {},
              "audio": {"music_beat_status": "unverified",
                        "beat_synced_status": "unverified"}}
    reference = {"claims": [{"claim_id": "V1", "predicate": "performs_action"}],
                 "events": []}
    return static, reference


def test_plan_covers_full_coarse_action_and_later_result_without_duplicate():
    static, reference = fixture()
    probes, _ = plan_probes(static, reference)
    validate_probe_plan(probes, static)
    assert len(probes) <= MAX_LOCAL_MEDIA_CALLS - 1
    visual = [p for p in probes if p["channel"] == "V"]
    covered = set().union(*(set(p["shot_ids"]) for p in visual))
    assert set(f"s{i}" for i in range(1, 8)) <= covered
    assert len({(p["issue_id"], p["channel"], tuple(p["shot_ids"]), p["fps"])
                for p in probes}) == len(probes)
    duplicate = copy.deepcopy(probes[0])
    with pytest.raises(ValueError, match="duplicate_probe"):
        validate_probe_plan(probes + [duplicate], static)


def test_v_payload_excludes_text_and_model_prompt_has_no_case_answer():
    static, reference = fixture()
    probes, _ = plan_probes(static, reference)
    visual = next(p for p in probes if p["channel"] == "V")
    assert "text_statements" not in _probe_payload(visual, static)
    assert "kick" not in V_PROMPT.lower()
    assert "reframe" not in V_PROMPT.lower()
    av = next(p for p in probes if p["issue_id"] == "terminal_statement_relation")
    assert _probe_payload(av, static)["text_statements"] == [
        {"shot_id": "s11", "claim_id": "T1", "observed_text": "last line",
         "source_type": "attributed_on_screen_text"}]


def test_actual_sampling_times_and_short_shot_block_motion():
    static, reference = fixture()
    probe = {"probe_id": "p1", "issue_id": "x", "channel": "V",
             "shot_ids": ["s8", "s9"], "interval": [8., 9.], "fps": 8.}
    payload = _probe_payload(probe, static)
    class Runner:
        _last_sampling = {"sampling_verified": True,
                          "actual_frame_timestamps_relative_s": [0.0, 0.125, 0.25],
                          "actual_frame_indices": [0, 3, 6]}
    sampling = sampling_from_runner(Runner(), probe)
    assert sampling["actual_frame_timestamps_absolute_s"] == [8., 8.125, 8.25]
    value = {"schema_version": "reference_local_observation_v3",
             "shots": [{"shot_id": "s8", "changes": [{"description": "movement",
                         "kind": "posture", "first_visible_s": 0.0,
                         "epistemic_status": "supported"}]},
                       {"shot_id": "s9", "changes": []}],
             "connections": [{"from_shot_id": "s8", "to_shot_id": "s9",
                              "continuity": "unknown"}]}
    checked = validate_local_observation(value, probe, payload, sampling)
    assert checked["change_findings"][0]["effective_status"] == "insufficient"
    value["shots"][0]["changes"][0]["first_visible_s"] = 0.5
    invalid_time = validate_local_observation(value, probe, payload, sampling)
    assert invalid_time["change_findings"][0]["effective_status"] == "insufficient"
    assert invalid_time["change_findings"][0]["issue_code"] == (
        "first_visible_time_outside_shot")


def test_generic_action_change_is_valid_but_still_needs_sample_support():
    static, _ = fixture()
    probe = {"probe_id": "p1", "issue_id": "x", "channel": "V",
             "shot_ids": ["s1", "s2"], "interval": [1., 3.], "fps": 8.}
    payload = _probe_payload(probe, static)
    sampling = {"sampling_verified": True,
                "actual_frame_timestamps_relative_s": [0., .125, 1., 1.125]}
    value = {"schema_version": "reference_local_observation_v3",
             "shots": [{"shot_id": "s1", "changes": [{"description": "an action",
                         "kind": "action", "first_visible_s": 0.,
                         "epistemic_status": "supported"}]},
                       {"shot_id": "s2", "changes": []}],
             "connections": [{"from_shot_id": "s1", "to_shot_id": "s2",
                              "continuity": "unknown"}]}
    assert validate_local_observation(value, probe, payload, sampling)[
        "change_findings"][0]["effective_status"] == "supported"


def test_high_impact_verification_is_new_sampling_and_neutral():
    static, _ = fixture()
    record = {"probe_id": "probe_01", "issue_id": "outcome", "channel": "V",
              "purpose": "action_to_later_state", "shot_ids": ["s5", "s6", "s7"],
              "validation": {"change_findings": [{"shot_id": "s6",
                  "kind": "posture", "declared_status": "supported"}]}}
    chosen = select_neutral_verification([record], static)
    assert chosen["fps"] == 16.0
    assert chosen["purpose"] == "neutral_verify"
    assert chosen["shot_ids"] == ["s5", "s6", "s7"]
    assert "description" not in chosen
    assert chosen["issue_id"].startswith("verify_s6_posture")
    assert select_neutral_verification([record]*MAX_LOCAL_MEDIA_CALLS,
                                       static) is None


def test_edit_graph_never_upgrades_unknown_or_onset_to_function():
    static, _ = fixture()
    record = {"probe_id": "p1", "sampling": {"sampling_verified": True},
              "response": {"connections": [
                  {"from_shot_id": "s0", "to_shot_id": "s1",
                   "continuity": "unknown"}]}}
    graph = build_editing_graph(static, [record])
    assert len(graph["adjacent_edges"]) == len(static["shots"])-1
    assert graph["adjacent_edges"][0]["semantic_status"] == "insufficient"
    assert graph["audio_measurement_status"]["beat_synced_status"] == "unverified"


def test_invalid_time_cannot_flow_as_supported_story_or_edit_observation():
    static, _ = fixture()
    record = {"probe_id": "p1", "channel": "V", "shot_ids": ["s1", "s2"],
              "interval": [1., 3.], "sampling": {"sampling_verified": True},
              "validation": {"change_findings": [{
                  "shot_id": "s2", "description": "an unsupported event",
                  "kind": "action", "first_visible_s": 0.,
                  "effective_status": "insufficient",
                  "sample_count_in_shot": 4,
                  "issue_code": "first_visible_time_outside_shot"}]},
              "response": {"shots": [
                  {"shot_id": "s1", "entry_state": "first state", "changes": []},
                  {"shot_id": "s2", "entry_state": "invented state",
                   "visible_action": "invented action", "exit_state": "invented exit",
                   "changes": []}],
                  "connections": [{"from_shot_id": "s1", "to_shot_id": "s2",
                                   "continuity": "same_action",
                                   "information_added": "invented link"}]}}
    view = _observation_view([record])[0]
    assert view["shots"][1]["entry_state"] is None
    assert view["shots"][1]["visible_action"] is None
    assert view["connections"][0]["continuity"] == "unknown"
    graph = build_editing_graph(static, [record])
    assert graph["adjacent_edges"][1]["semantic_status"] == "insufficient"


def test_counterfactual_inputs_remove_terminal_statement_and_reorder():
    static, _ = fixture()
    masked = _counterfactual_sections(static, mask_terminal_text=True)
    assert "last line" not in str(masked)
    assert "T1" not in str(masked)
    reversed_rows = _counterfactual_sections(static, reverse_order=True)
    assert reversed_rows[0]["attributed_statements"][0]["observed_text"] == "last line"
    assert reversed_rows[0]["ordinal"] == 1
