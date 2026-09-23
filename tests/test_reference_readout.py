from pathlib import Path

import pytest

from src.agentic_video.modality_isolation import IsolationError, audit_request
from src.agentic_video.reference_readout import (
    _citation_issues, _transfer_leaks, select_local_probes,
    validate_local_patterns)


def test_av_channel_requires_both_streams(monkeypatch) -> None:
    import src.agentic_video.modality_isolation as isolation

    monkeypatch.setattr(isolation, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(isolation, "_stream_types", lambda *_args, **_kw: ["video"])
    payload = {"request_id": "local_1", "channel": "AV",
               "interval": [0.0, 1.0], "claim_ids": [], "event_ids": [],
               "observation_dimensions": ["cut"], "sampling": {"fps": 8}}
    with pytest.raises(IsolationError, match="audiovisual_request_missing_stream"):
        audit_request(channel="AV", prompt="Generic editing observation",
                      payload=payload, media_paths=[Path("clip.mp4")])
    monkeypatch.setattr(isolation, "_stream_types",
                        lambda *_args, **_kw: ["video", "audio"])
    result = audit_request(channel="AV", prompt="Generic editing observation",
                           payload=payload, media_paths=[Path("clip.mp4")])
    assert result["contract_passed"] is True


def test_probe_planner_uses_unresolved_motion_and_measured_edits() -> None:
    static = {
        "duration_s": 8.0,
        "text_timeline": [],
        "section_bundles": [
            {"section_id": "A", "interval": [0.0, 5.0]},
            {"section_id": "B", "interval": [5.0, 7.0]},
            {"section_id": "C", "interval": [7.0, 8.0]}],
        "shots": [{"section_id": "A", "unresolved": ["camera_motion_unobserved"]},
                  {"section_id": "B", "unresolved": []},
                  {"section_id": "C", "unresolved": []}],
        "measured_editing": {"structural_grammar": {
            "transition_profile": [
                {"section_id": "A", "hard_cut_count": 0,
                 "transition_count": 0},
                {"section_id": "B", "hard_cut_count": 4,
                 "transition_count": 0},
                {"section_id": "C", "hard_cut_count": 0,
                 "transition_count": 3}]}}}
    probes = select_local_probes(static)
    assert [row["probe_id"] for row in probes] == [
        "visual_A", "edit_boundary_1", "edit_boundary_2",
        "edit_sequence_C", "edit_sequence_B"]
    assert len(probes) <= 8


def test_unknown_citation_does_not_become_valid_evidence() -> None:
    reference = {"claims": [{"claim_id": "V1"}], "events": []}
    reading = {"narrative_units": [{"unit_id": "N1", "support_ids": ["V9"]}],
               "relations": [], "message_hypotheses": []}
    assert _citation_issues(reading, [], {"functions": []}, reference) == [
        "unit:unknown_support:V9"]


def test_editing_pattern_rejects_absolute_time_and_joined_types() -> None:
    context = {"clip_duration_s": 1.7,
               "shots": [{"shot_id": "A", "relative_interval": [0.0, 0.85]},
                         {"shot_id": "B", "relative_interval": [0.85, 1.7]}],
               "claim_ids": ["V1"], "event_ids": []}
    response = {"schema_version": "reference_local_editing_v1",
                "patterns": [
                    {"pattern_id": "P1", "type": "cut_on_action",
                     "shot_ids": ["A", "B"], "media_interval": [4.25, 5.95],
                     "support_ids": ["V1"]},
                    {"pattern_id": "P2", "type": "rapid_montage|insert_shot",
                     "shot_ids": ["A", "B"], "media_interval": [0.1, 1.6],
                     "support_ids": ["V1"]},
                    {"pattern_id": "P3", "type": "contrast_cut",
                     "shot_ids": ["A", "B"], "media_interval": [0.1, 1.6],
                     "support_ids": ["V1"]}]}
    accepted, rejected = validate_local_patterns(response, context)
    assert [row["pattern_id"] for row in accepted] == ["P3"]
    assert rejected[0]["reason_codes"] == ["clip_local_interval_invalid"]
    assert rejected[1]["reason_codes"] == ["pattern_type_invalid"]


def test_transfer_draft_blocks_reference_text_and_ids() -> None:
    reference = {"source_sha": "a" * 64,
                 "claims": [{"claim_id": "T1", "object": "Visible source caption",
                             "modality": "T"}],
                 "events": [],
                 "shot_storyboard": {"shot_cards": [{"shot_id": "S1"}]}}
    draft = {"communicative_goal": "Visible source caption",
             "free_slots": ["T1"]}
    assert _transfer_leaks(draft, reference, Path("reference.mp4"))
    clean = {"communicative_goal": "Reconsider a broad judgment using new evidence",
             "free_slots": ["domain", "relationship"]}
    assert _transfer_leaks(clean, reference, Path("reference.mp4")) == []
