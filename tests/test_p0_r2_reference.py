from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agentic_video.editing_grammar import (
    EditingGrammarError,
    analyze_editing_grammar,
    build_editing_payload,
    build_measured_editing_facts,
    validate_editing_patterns,
)
from src.agentic_video.reference_interpretation import (
    finalize_interpretation,
    run_reference_interpretation,
    run_reference_interpretation_checkpointed,
    validate_interpretation,
)
from src.agentic_video.reference_storyboard import build_agent_reference


def _validated_reference() -> dict:
    segments = [
        {"segment_id": "S1", "section_id": "A", "segment_kind": "content",
         "interval": [0.0, 5.0]},
        {"segment_id": "S2", "section_id": "B", "segment_kind": "content",
         "interval": [5.0, 6.0]},
        {"segment_id": "T1", "section_id": "B", "segment_kind": "transition",
         "transition_type": "zoom_blur", "interval": [6.0, 6.1]},
        {"segment_id": "S3", "section_id": "B", "segment_kind": "content",
         "interval": [6.1, 7.0]},
        {"segment_id": "S4", "section_id": "C", "segment_kind": "content",
         "interval": [7.0, 7.5]},
        {"segment_id": "T2", "section_id": "C", "segment_kind": "transition",
         "transition_type": "whip_pan", "interval": [7.5, 7.6]},
        {"segment_id": "S5", "section_id": "C", "segment_kind": "content",
         "interval": [7.6, 8.0]},
    ]
    return {
        "schema_version": "validated_reference_v1",
        "source_sha": "a" * 64,
        "artifact_sha": "b" * 64,
        "accepted_claims": [
            {"claim_id": "V1", "subject": "E1", "predicate": "appears_in",
             "object": "scene", "interval": [0.0, 5.0], "modality": "V"},
            {"claim_id": "ID1", "subject": "E1", "predicate": "same_entity_as",
             "object": "E2", "interval": [0.0, 8.0], "modality": "V"},
            {"claim_id": "T01", "subject": "TEXT", "predicate": "contains_text",
             "object": "statement", "interval": [5.0, 6.0], "modality": "T"},
            {"claim_id": "A1", "subject": "AUDIO", "predicate": "audio_event",
             "object": "sound", "interval": [7.0, 7.5], "modality": "A"},
        ],
        "accepted_events": [
            {"event_id": "EV1", "participants": ["E1"],
             "action_claim_ids": ["V1"], "object_ids": [],
             "ordering": "observed", "outcome_claim_ids": [],
             "context_claim_ids": [], "interval": [0.0, 5.0]},
        ],
        "deterministic_timeline": {
            "schema_version": "timeline_v1",
            "sections": [
                {"section_id": "A", "interval": [0.0, 5.0]},
                {"section_id": "B", "interval": [5.0, 7.0]},
                {"section_id": "C", "interval": [7.0, 8.0]},
            ],
            "segments": segments,
        },
        "coverage": {},
        "unresolved_limitations": [],
    }


def _interpretation() -> dict:
    return {
        "schema_version": "reference_interpretation_v1",
        "propositions": [{
            "proposition_id": "P1", "statement": "A bounded observation",
            "supported_by": ["V1"],
        }],
        "relations": [],
        "limitations": [],
    }


def _passing_audit() -> dict:
    return {
        "schema_version": "reference_relation_audit_v2",
        "proposition_checks": [{
            "proposition_id": "P1", "entailed": True,
            "scope_valid": True, "reason_codes": [],
        }],
        "relation_checks": [],
        "pass": True,
    }


class _Runner:
    def __init__(self, value: dict, model_id: str) -> None:
        self.value = value
        self.model_id = model_id
        self.calls = 0
        self.prompts: list[str] = []

    def ask(self, prompt: str, **_kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return SimpleNamespace(text=json.dumps(self.value))


def test_agent_reference_separates_identity_and_structures_transitions() -> None:
    result = build_agent_reference(_validated_reference())
    cards = result["shot_storyboard"]["shot_cards"]

    assert [row["shot_id"] for row in cards] == ["S1", "S2", "S3", "S4", "S5"]
    assert cards[0]["visual_claim_ids"] == ["V1"]
    assert cards[0]["identity_claim_ids"] == ["ID1"]
    assert cards[0]["event_ids"] == ["EV1"]
    assert cards[1]["transition_out"] == {
        "transition_id": "T1", "transition_type": "zoom_blur",
        "start_s": 6.0, "end_s": 6.1, "duration_s": 0.1,
        "from_shot_id": "S2", "to_shot_id": "S3",
    }
    assert cards[2]["transition_in"] == cards[1]["transition_out"]
    serialized = json.dumps(result)
    assert "analysis_contract" not in serialized
    assert "required_roles" not in serialized


def test_no_relation_still_requires_proposition_audit() -> None:
    interpreter = _Runner(_interpretation(), "interpreter-v1")
    auditor = _Runner(_passing_audit(), "auditor-v1")
    result, audit = run_reference_interpretation(
        interpreter, auditor, build_agent_reference(_validated_reference()))

    assert interpreter.calls == 1
    assert auditor.calls == 1
    assert result["relations"] == []
    assert result["propositions"][0]["verification_status"] == "SUPPORTED"
    assert result["verification_status"] == "SUPPORTED"
    assert audit["pass"] is True


def test_failed_proposition_audit_marks_no_relation_result_unresolved() -> None:
    audit = _passing_audit()
    audit["proposition_checks"][0]["entailed"] = False
    audit["proposition_checks"][0]["reason_codes"] = ["NOT_ENTAILED"]
    audit["pass"] = False

    result = finalize_interpretation(_interpretation(), audit)

    assert result["propositions"][0]["verification_status"] == "UNRESOLVED"
    assert result["verification_status"] == "UNRESOLVED"


def test_failed_relation_audit_marks_relation_unresolved() -> None:
    interpretation = _interpretation()
    interpretation["propositions"].append({
        "proposition_id": "P2", "statement": "Second proposition",
        "supported_by": ["T01"],
    })
    interpretation["relations"].append({
        "relation_id": "R1", "type": "reframes", "source": "P2",
        "target": "P1", "reason": "Candidate relation",
        "supported_by": ["V1", "T01"],
    })
    validate_interpretation(
        interpretation, build_agent_reference(_validated_reference()))
    audit = {
        "schema_version": "reference_relation_audit_v2",
        "proposition_checks": [
            {"proposition_id": "P1", "entailed": True,
             "scope_valid": True, "reason_codes": []},
            {"proposition_id": "P2", "entailed": True,
             "scope_valid": True, "reason_codes": []},
        ],
        "relation_checks": [{
            "relation_id": "R1", "source_entailed": True,
            "target_entailed": True, "relation_valid": False,
            "scope_valid": True, "reason_codes": ["RELATION_INVALID"],
        }],
        "pass": False,
    }

    result = finalize_interpretation(interpretation, audit)
    assert result["relations"][0]["verification_status"] == "UNRESOLVED"
    assert result["verification_status"] == "UNRESOLVED"


def test_checkpoint_requires_evidence_and_analysis_fingerprint(tmp_path) -> None:
    agent_reference = build_agent_reference(_validated_reference())
    interpreter = _Runner(_interpretation(), "interpreter-v1")
    auditor = _Runner(_passing_audit(), "auditor-v1")
    first, _ = run_reference_interpretation_checkpointed(
        interpreter, auditor, agent_reference, checkpoint_dir=tmp_path)
    second, _ = run_reference_interpretation_checkpointed(
        interpreter, auditor, agent_reference, checkpoint_dir=tmp_path)

    assert first == second
    assert interpreter.calls == 1
    assert auditor.calls == 1
    assert first["analysis_fingerprint"]

    changed_interpreter = _Runner(_interpretation(), "interpreter-v2")
    changed_auditor = _Runner(_passing_audit(), "auditor-v1")
    third, _ = run_reference_interpretation_checkpointed(
        changed_interpreter, changed_auditor, agent_reference,
        checkpoint_dir=tmp_path)
    assert changed_interpreter.calls == 1
    assert changed_auditor.calls == 1
    assert third["analysis_fingerprint"] != first["analysis_fingerprint"]


def test_measured_editing_metrics_split_shot_cut_transition_rates() -> None:
    measured = build_measured_editing_facts(
        build_agent_reference(_validated_reference()))
    rows = measured["pace_curve"]

    assert [row["content_shot_count"] for row in rows] == [1, 2, 2]
    assert [row["cut_count"] for row in rows] == [0, 1, 1]
    assert [row["transition_count"] for row in rows] == [0, 1, 1]
    assert [row["shot_rate"] for row in rows] == [0.2, 1.0, 2.0]
    assert [row["cut_rate"] for row in rows] == [0.0, 0.5, 1.0]
    assert [row["transition_rate"] for row in rows] == [0.0, 0.5, 1.0]
    assert [row["raw_segment_rate"] for row in rows] == [0.2, 1.5, 3.0]
    assert rows[1]["mean_shot_duration_s"] == 0.95
    assert rows[1]["median_shot_duration_s"] == 0.95
    assert measured["facts"][1]["transition_out"]["transition_id"] == "T1"
    assert measured["pace_changes"][0]["shot_rate_before"] == 0.2


def test_editing_payload_dereferences_claims_events_and_identity() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)
    payload = build_editing_payload(agent_reference, measured)
    first = payload["shots"][0]

    assert first["visual_claims"][0]["claim_id"] == "V1"
    assert first["identity_claims"][0]["claim_id"] == "ID1"
    assert first["events"][0]["event_id"] == "EV1"
    assert first["transition_in"] is None
    assert payload["measured_editing"]["pace_curve"][1]["shot_rate"] == 1.0


def _pattern_value(pattern_type: str = "contrast_cut") -> dict:
    return {
        "schema_version": "editing_patterns_v1",
        "patterns": [{
            "pattern_id": "EP1", "type": pattern_type, "scope": "local",
            "shot_ids": ["S4", "S5"], "fact_ids": ["EF4", "EF5"],
            "evidence_ids": ["A1"], "confidence": 0.8,
        }],
        "limitations": [],
    }


def test_local_editing_pattern_cannot_span_most_of_video() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)
    value = _pattern_value("rapid_montage")
    value["patterns"][0]["shot_ids"] = ["S1", "S2", "S3", "S4", "S5"]
    value["patterns"][0]["fact_ids"] = ["EF1", "EF2"]
    value["patterns"][0]["evidence_ids"] = ["V1"]

    with pytest.raises(EditingGrammarError) as excinfo:
        validate_editing_patterns(
            value, measured, evidence_ids={"V1", "T01", "A1", "EV1"})
    assert excinfo.value.reason_code == "local_pattern_too_broad"


def test_unexplained_pace_change_is_warning_not_failure() -> None:
    measured = build_measured_editing_facts(
        build_agent_reference(_validated_reference()))
    result = validate_editing_patterns(
        _pattern_value(), measured,
        evidence_ids={"V1", "T01", "A1", "EV1"})

    assert result["patterns"][0]["type"] == "contrast_cut"
    assert result["coverage_warnings"] == [
        "pace_change_unexplained:A->B",
        "pace_change_unexplained:B->C",
    ]


def test_editing_recognition_and_function_reasoning_are_two_single_calls() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    pattern_runner = _Runner(_pattern_value("rapid_montage"), "pattern-v1")
    function_runner = _Runner({
        "schema_version": "editing_functions_v1",
        "functions": [{
            "pattern_id": "EP1", "function": "accumulate",
            "relation_to_story": [], "confidence": 0.8,
        }],
        "limitations": [],
    }, "function-v1")

    result, measured = analyze_editing_grammar(
        pattern_runner, function_runner, agent_reference, _interpretation())

    assert pattern_runner.calls == 1
    assert function_runner.calls == 1
    assert "verified_interpretation" not in pattern_runner.prompts[0]
    assert '\"predicate\":\"audio_event\"' in pattern_runner.prompts[0]
    assert "verified_interpretation" in function_runner.prompts[0]
    assert result["patterns"][0]["evidence_ids"] == ["A1"]
    assert result["functions"][0]["function"] == "accumulate"
    assert measured["schema_version"] == "measured_editing_facts_v2"
