"""Deterministic v4 contracts; these tests never load a model."""
from __future__ import annotations

import copy

import pytest

from src.agentic_video.reference_understanding_v4 import (
    AUDIT_PROMPT, GLOBAL_PROMPT, LOCAL_PROMPT, MAX_LOCAL_CALLS,
    global_payload, plan_initial_probes, plan_verification, validate_audit,
    validate_editing, validate_probe_plan, validate_reading,
)


def fixture():
    rows = []
    section_sizes = (1, 8, 8)
    at = 0.0
    for section_index, size in enumerate(section_sizes):
        for index in range(size):
            sid = f"s{len(rows)}"
            rows.append({"shot_id": sid, "section_id": f"section_{section_index}",
                         "start_s": at, "end_s": at + 1.0, "transition_in": None,
                         "visual_claim_ids": (["V1"] if section_index == 1 and
                                              1 <= index <= 5 else []),
                         "text_claim_ids": (["T1"] if len(rows) == 16 else []),
                         "event_ids": []})
            at += 1.0
    static = {"shots": rows, "duration_s": at,
              "section_bundles": [{"section_id": f"section_{i}",
                                   "interval": [sum(section_sizes[:i]),
                                                sum(section_sizes[:i+1])],
                                   "text_statements": []}
                                  for i in range(3)],
              "text_timeline": [{"claim_id": "T1", "observed_text": "a statement",
                                 "interval": [16.0, 17.0],
                                 "timing_status": "accepted_claim_interval_not_frame_verified"}],
              "audio_transcript_candidate": {"status": "candidate_not_human_verified"}}
    reference = {"claims": [{"claim_id": "V1", "predicate": "performs_action"}],
                 "events": []}
    return static, reference


def test_global_payload_preserves_attribution_and_only_local_delta():
    static, _ = fixture()
    before = global_payload(static, [])
    after = global_payload(static, [{"probe_id": "p1"}])
    assert before["text_statements"][0]["source_type"] == "attributed_on_screen_text"
    assert before["audio_transcript"]["status"] == "candidate_not_human_verified"
    assert before["local_observations"] == []
    same = copy.deepcopy(after)
    same["local_observations"] = []
    assert same == before
    assert "prior_reading" not in after


def test_initial_plan_is_av_and_covers_action_result_and_final_sequence():
    static, reference = fixture()
    probes, _ = plan_initial_probes(static, reference)
    assert len(probes) <= 6
    assert all(probe["channel"] == "AV" for probe in probes)
    covered = {sid for probe in probes for sid in probe["shot_ids"]}
    assert {f"s{i}" for i in range(2, 9)} <= covered
    assert "s16" in covered
    assert any(probe["purpose"] == "adjacent_sequence" and
               probe["shot_ids"][0] == "s9" for probe in probes)
    assert all(probe["fps"] == 12.0 for probe in probes)
    validate_probe_plan(probes, static)


def test_verification_changes_sampling_and_never_sends_first_reading():
    static, reference = fixture()
    probes, _ = plan_initial_probes(static, reference)
    records = [{**probe, "validation": {"change_findings": [{
        "effective_status": "insufficient"}]}, "response": {"unresolved": []}}
        for probe in probes]
    later = plan_verification(static, probes, records,
                              {"unresolved": [{"shot_ids": ["s16"]}]})
    assert len(later) <= 2
    assert len(probes) + len(later) <= MAX_LOCAL_CALLS
    assert all(row["fps"] == 16.0 for row in later)
    assert all("description" not in row and "reading" not in row for row in later)
    assert later[0]["shot_ids"][-1] == "s16"
    validate_probe_plan(probes + later, static)


def test_reading_allows_empty_relations_but_rejects_unknown_refs():
    static, _ = fixture()
    reading = {"schema_version": "reference_reading_v4",
               "narrative_units": [{"unit_id": "N1", "time_range_s": [0, 1],
                                    "shot_ids": ["s0"], "statement_ids": [],
                                    "local_refs": []}],
               "relations": [], "message_hypotheses": []}
    assert validate_reading(reading, static, set()) == []
    reading["narrative_units"][0]["statement_ids"] = ["fake"]
    assert "unit_statement_unknown" in validate_reading(reading, static, set())


def test_editing_and_audit_cannot_claim_unobserved_edge_or_visual_truth():
    graph = {"adjacent_edges": [{"from_shot_id": "s0", "to_shot_id": "s1",
                                 "observations": []}]}
    edit = {"schema_version": "reference_editing_functions_v4", "edges": [
        {"from_shot_id": "s0", "to_shot_id": "s1", "status": "provisional",
         "support_probe_ids": []}]}
    assert "unobserved_edge_not_insufficient" in validate_editing(edit, graph)
    edit["edges"][0]["status"] = "insufficient"
    assert validate_editing(edit, graph) == []
    reading = {"narrative_units": [], "relations": [], "message_hypotheses": []}
    audit = {"schema_version": "reference_grounding_audit_v4",
             "visual_truth_verified": True,
             "checks": [{"kind": "edit", "id": "s0->s1"}]}
    assert "audit_overclaims_visual_truth" in validate_audit(audit, reading, edit)


def test_prompts_are_generic_and_unforced():
    for prompt in (GLOBAL_PROMPT, LOCAL_PROMPT, AUDIT_PROMPT):
        assert "跆拳道" not in prompt and "剪脚指甲" not in prompt
        assert "kick" not in prompt.lower()
    assert "empty relations" in GLOBAL_PROMPT
    assert "not a test of whether the most" in AUDIT_PROMPT


def test_duplicate_or_out_of_budget_probe_rejected():
    static, reference = fixture()
    probes, _ = plan_initial_probes(static, reference)
    with pytest.raises(ValueError, match="duplicate_probe"):
        validate_probe_plan(probes + [probes[0]], static)
    with pytest.raises(ValueError, match="media_budget_exceeded"):
        validate_probe_plan(probes + [probes[0]] * 3, static)
