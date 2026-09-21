from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agentic_video.editing_grammar import (
    EditingGrammarError,
    build_measured_editing_facts,
    validate_editing_grammar,
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
         "interval": [6.0, 6.1]},
        {"segment_id": "S3", "section_id": "B", "segment_kind": "content",
         "interval": [6.1, 7.0]},
        {"segment_id": "S4", "section_id": "C", "segment_kind": "content",
         "interval": [7.0, 7.5]},
        {"segment_id": "T2", "section_id": "C", "segment_kind": "transition",
         "interval": [7.5, 7.6]},
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


def test_agent_reference_maps_evidence_without_semantic_contract() -> None:
    result = build_agent_reference(_validated_reference())
    cards = result["shot_storyboard"]["shot_cards"]

    assert [row["shot_id"] for row in cards] == ["S1", "S2", "S3", "S4", "S5"]
    assert cards[0]["visual_claim_ids"] == ["V1"]
    assert cards[0]["event_ids"] == ["EV1"]
    assert cards[1]["transition_out"] == "T1"
    assert cards[2]["transition_in"] == "T1"
    serialized = json.dumps(result)
    assert "analysis_contract" not in serialized
    assert "required_roles" not in serialized


def test_interpretation_allows_no_relation_and_calls_each_model_once() -> None:
    interpretation_value = {
        "schema_version": "reference_interpretation_v1",
        "propositions": [{
            "proposition_id": "P1",
            "statement": "A bounded observation is present.",
            "supported_by": ["V1"],
        }],
        "relations": [],
        "limitations": ["No reliable relation was found."],
    }
    audit_value = {
        "schema_version": "reference_relation_audit_v1",
        "checks": [],
        "pass": True,
    }

    class Runner:
        def __init__(self, value: dict) -> None:
            self.value = value
            self.calls = 0

        def ask(self, prompt: str, **_kwargs):
            self.calls += 1
            assert "analysis_contract" not in prompt
            return SimpleNamespace(text=json.dumps(self.value))

    interpreter = Runner(interpretation_value)
    auditor = Runner(audit_value)
    result, audit = run_reference_interpretation(
        interpreter, auditor, build_agent_reference(_validated_reference()))

    assert interpreter.calls == 1
    assert auditor.calls == 1
    assert result["relations"] == []
    assert result["verification_status"] == "SUPPORTED"
    assert audit["pass"] is True


def test_unchanged_evidence_reuses_interpretation_checkpoint(tmp_path) -> None:
    interpretation_value = {
        "schema_version": "reference_interpretation_v1",
        "propositions": [{
            "proposition_id": "P1", "statement": "Bounded observation",
            "supported_by": ["V1"],
        }],
        "relations": [],
        "limitations": [],
    }
    audit_value = {
        "schema_version": "reference_relation_audit_v1",
        "checks": [], "pass": True,
    }

    class Runner:
        def __init__(self, value: dict) -> None:
            self.value = value
            self.calls = 0

        def ask(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(text=json.dumps(self.value))

    interpreter = Runner(interpretation_value)
    auditor = Runner(audit_value)
    agent_reference = build_agent_reference(_validated_reference())
    first, _ = run_reference_interpretation_checkpointed(
        interpreter, auditor, agent_reference, checkpoint_dir=tmp_path)
    second, _ = run_reference_interpretation_checkpointed(
        interpreter, auditor, agent_reference, checkpoint_dir=tmp_path)

    assert first == second
    assert interpreter.calls == 1
    assert auditor.calls == 1


def test_failed_relation_audit_marks_relation_unresolved() -> None:
    interpretation = {
        "schema_version": "reference_interpretation_v1",
        "propositions": [
            {"proposition_id": "P1", "statement": "First proposition",
             "supported_by": ["V1"]},
            {"proposition_id": "P2", "statement": "Second proposition",
             "supported_by": ["T01"]},
        ],
        "relations": [{
            "relation_id": "R1", "type": "reframes", "source": "P2",
            "target": "P1", "reason": "Candidate relation",
            "supported_by": ["V1", "T01"],
        }],
        "limitations": [],
    }
    validate_interpretation(
        interpretation, build_agent_reference(_validated_reference()))
    audit = {
        "schema_version": "reference_relation_audit_v1",
        "checks": [{
            "relation_id": "R1", "source_entailed": True,
            "target_entailed": True, "relation_valid": False,
            "scope_valid": True, "reason_codes": ["RELATION_INVALID"],
        }],
        "pass": False,
    }

    result = finalize_interpretation(interpretation, audit)
    assert result["relations"][0]["verification_status"] == "UNRESOLVED"
    assert result["verification_status"] == "UNRESOLVED"


def test_measured_editing_facts_include_transition_density() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)

    assert [row["segment_count"] for row in measured["pace_curve"]] == [1, 3, 3]
    assert [row["density"] for row in measured["pace_curve"]] == [0.2, 1.5, 3.0]
    assert measured["facts"][1]["transition_out"] == "T1"


def test_local_editing_pattern_cannot_span_most_of_video() -> None:
    measured = build_measured_editing_facts(
        build_agent_reference(_validated_reference()))
    grammar = {
        "schema_version": "editing_grammar_v1",
        "patterns": [{
            "pattern_id": "EP1", "type": "rapid_montage", "scope": "local",
            "shot_ids": ["S1", "S2", "S3", "S4", "S5"],
            "fact_ids": ["EF1", "EF2"], "confidence": 0.9,
        }],
        "functions": [{
            "pattern_id": "EP1", "function": "accumulate",
            "relation_to_story": [], "confidence": 0.8,
        }],
        "limitations": [],
    }

    with pytest.raises(EditingGrammarError) as excinfo:
        validate_editing_grammar(grammar, measured, relation_ids=set())
    assert excinfo.value.reason_code == "local_pattern_too_broad"


def test_pace_change_requires_a_local_pace_pattern() -> None:
    measured = build_measured_editing_facts(
        build_agent_reference(_validated_reference()))
    grammar = {
        "schema_version": "editing_grammar_v1",
        "patterns": [{
            "pattern_id": "EP1", "type": "contrast_cut", "scope": "local",
            "shot_ids": ["S4", "S5"], "fact_ids": ["EF4", "EF5"],
            "confidence": 0.8,
        }],
        "functions": [{
            "pattern_id": "EP1", "function": "contrast",
            "relation_to_story": [], "confidence": 0.8,
        }],
        "limitations": [],
    }

    with pytest.raises(EditingGrammarError) as excinfo:
        validate_editing_grammar(grammar, measured, relation_ids=set())
    assert excinfo.value.reason_code == "pace_change_uncovered"


def test_valid_local_pace_patterns_pass() -> None:
    measured = build_measured_editing_facts(
        build_agent_reference(_validated_reference()))
    grammar = {
        "schema_version": "editing_grammar_v1",
        "patterns": [
            {"pattern_id": "EP1", "type": "rhythmic_acceleration",
             "scope": "local", "shot_ids": ["S2", "S3"],
             "fact_ids": ["EF2", "EF3"], "confidence": 0.8},
            {"pattern_id": "EP2", "type": "rapid_montage", "scope": "local",
             "shot_ids": ["S4", "S5"], "fact_ids": ["EF4", "EF5"],
             "confidence": 0.9},
        ],
        "functions": [
            {"pattern_id": "EP1", "function": "escalate",
             "relation_to_story": [], "confidence": 0.7},
            {"pattern_id": "EP2", "function": "accumulate",
             "relation_to_story": [], "confidence": 0.8},
        ],
        "limitations": [],
    }

    result = validate_editing_grammar(grammar, measured, relation_ids=set())
    assert result["patterns"][0]["span"] == [5.0, 7.0]
