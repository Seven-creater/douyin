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
    ReferenceInterpretationError,
    analysis_fingerprint,
    build_interpretation_payload,
    finalize_interpretation,
    runner_identity,
    run_reference_interpretation,
    run_reference_interpretation_checkpointed,
    validate_interpretation,
    validate_relation_audit,
)
from src.agentic_video.reference_storyboard import build_agent_reference
from src.perception.omni_pool import OmniProcessPool
from src.perception.omni_runner import OmniRunner


def _validated_reference() -> dict:
    segments = [
        {"segment_id": "S1", "section_id": "A", "segment_kind": "content",
         "interval": [0.0, 5.0]},
        {"segment_id": "S2", "section_id": "B", "segment_kind": "content",
         "interval": [5.0, 6.0]},
        {"segment_id": "S3", "section_id": "B", "segment_kind": "content",
         "interval": [6.0, 7.0]},
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
             "object": "scene", "interval": [0.0, 5.0], "modality": "V",
             "producer": "human", "source_sha": "secret", "reviewer": "R1",
             "support_refs": [{"path": "C:/private/reference.mp4"}]},
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
             "context_claim_ids": [], "interval": [0.0, 5.0],
             "producer": "human", "source_sha": "secret"},
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
        "schema_version": "reference_narrative_interpretation_v2",
        "narrative_units": [{
            "unit_id": "N1", "section_ids": ["A"],
            "summary": "A bounded event occurs in the opening section.",
            "supported_by": ["V1", "EV1"],
        }],
        "relations": [],
        "limitations": [],
    }


def _passing_audit() -> dict:
    return {
        "schema_version": "reference_narrative_audit_v1",
        "unit_checks": [{
            "unit_id": "N1", "entailed": True,
            "scope_valid": True, "attribution_preserved": True,
            "reason_codes": [],
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
    assert cards[3]["transition_out"] == {
        "transition_id": "T2", "transition_type": "whip_pan",
        "start_s": 7.5, "end_s": 7.6, "duration_s": 0.1,
        "from_shot_id": "S4", "to_shot_id": "S5",
    }
    assert cards[4]["transition_in"] == cards[3]["transition_out"]
    serialized = json.dumps(result)
    assert "analysis_contract" not in serialized
    assert "required_roles" not in serialized


def test_interpretation_payload_builds_section_bundles_without_identity_claims() -> None:
    payload = build_interpretation_payload(
        build_agent_reference(_validated_reference()))
    sections = {row["section_id"]: row for row in payload["section_bundles"]}

    assert set(sections) == {"A", "B", "C"}
    assert "claims" not in payload
    assert "shot_cards" not in payload
    assert sections["A"]["visual_observations"][0]["subject"] == "SUBJECT_01"
    assert sections["A"]["events"][0]["participants"] == ["SUBJECT_01"]
    assert sections["B"]["text_statements"] == [{
        "claim_id": "T01", "source_type": "on_screen_text",
        "observed_text": "statement", "interval": [5.0, 6.0],
    }]
    serialized = json.dumps(payload)
    assert "ID1" not in serialized
    assert "same_entity_as" not in serialized
    assert "contains_text" not in serialized
    assert "E1" not in serialized
    assert "E2" not in serialized


def test_no_relation_still_requires_narrative_unit_audit() -> None:
    interpreter = _Runner(_interpretation(), "interpreter-v1")
    auditor = _Runner(_passing_audit(), "auditor-v1")
    result, audit = run_reference_interpretation(
        interpreter, auditor, build_agent_reference(_validated_reference()))

    assert interpreter.calls == 1
    assert auditor.calls == 1
    assert result["relations"] == []
    assert result["narrative_units"][0]["verification_status"] == "SUPPORTED"
    assert result["verification_status"] == "SUPPORTED"
    assert audit["pass"] is True
    assert "Do NOT restate every atomic claim" in interpreter.prompts[0]
    assert "required_roles" not in interpreter.prompts[0]


def test_failed_unit_audit_marks_no_relation_result_unresolved() -> None:
    audit = _passing_audit()
    audit["unit_checks"][0]["entailed"] = False
    audit["unit_checks"][0]["reason_codes"] = ["NOT_ENTAILED"]
    audit["pass"] = False

    result = finalize_interpretation(_interpretation(), audit)

    assert result["narrative_units"][0]["verification_status"] == "UNRESOLVED"
    assert result["verification_status"] == "UNRESOLVED"


def test_empty_narrative_units_and_relations_are_valid() -> None:
    interpretation = {
        "schema_version": "reference_narrative_interpretation_v2",
        "narrative_units": [], "relations": [], "limitations": [],
    }
    audit = {
        "schema_version": "reference_narrative_audit_v1",
        "unit_checks": [], "relation_checks": [], "pass": True,
    }
    agent_reference = build_agent_reference(_validated_reference())

    validate_interpretation(interpretation, agent_reference)
    validate_relation_audit(audit, interpretation)
    result = finalize_interpretation(interpretation, audit)

    assert result["verification_status"] == "SUPPORTED"


def test_failed_relation_audit_marks_relation_unresolved() -> None:
    interpretation = _interpretation()
    interpretation["narrative_units"].append({
        "unit_id": "N2", "section_ids": ["B"],
        "summary": "The video presents an attributed text statement.",
        "supported_by": ["T01"],
    })
    interpretation["relations"].append({
        "relation_id": "R1", "type": "reframes", "source": "N2",
        "target": "N1", "reason": "Candidate relation",
        "supported_by": ["V1", "T01"],
    })
    validate_interpretation(
        interpretation, build_agent_reference(_validated_reference()))
    audit = {
        "schema_version": "reference_narrative_audit_v1",
        "unit_checks": [
            {"unit_id": "N1", "entailed": True, "scope_valid": True,
             "attribution_preserved": True, "reason_codes": []},
            {"unit_id": "N2", "entailed": True, "scope_valid": True,
             "attribution_preserved": True, "reason_codes": []},
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


def test_narrative_units_must_use_local_non_identity_evidence() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    out_of_scope = _interpretation()
    out_of_scope["narrative_units"][0]["supported_by"] = ["T01"]
    with pytest.raises(ReferenceInterpretationError) as scope_error:
        validate_interpretation(out_of_scope, agent_reference)
    assert getattr(scope_error.value, "reason_code", None) == (
        "narrative_unit_support_out_of_scope")

    identity_support = _interpretation()
    identity_support["narrative_units"][0]["supported_by"] = ["ID1"]
    with pytest.raises(ReferenceInterpretationError) as identity_error:
        validate_interpretation(identity_support, agent_reference)
    assert getattr(identity_error.value, "reason_code", None) == (
        "narrative_unit_support_invalid")

    too_many = _interpretation()
    too_many["narrative_units"] = [
        {"unit_id": f"N{i}", "section_ids": ["A"], "summary": "Summary",
         "supported_by": ["V1"]}
        for i in range(1, 5)
    ]
    with pytest.raises(ReferenceInterpretationError) as count_error:
        validate_interpretation(too_many, agent_reference)
    assert getattr(count_error.value, "reason_code", None) == (
        "section_narrative_unit_limit_exceeded")


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


def test_production_runner_config_is_part_of_analysis_fingerprint() -> None:
    interpreter_v1 = OmniRunner({
        "model_path": "models/omni-v1", "dtype": "bfloat16",
        "repetition_penalty": 1.05,
    })
    interpreter_v2 = OmniRunner({
        "model_path": "models/omni-v2", "dtype": "bfloat16",
        "repetition_penalty": 1.05,
    })
    auditor = OmniRunner({
        "model_path": "models/auditor", "dtype": "bfloat16",
        "repetition_penalty": 1.0,
    })

    identity = runner_identity(interpreter_v1)
    assert identity == {
        "class": "OmniRunner", "model_path": "models/omni-v1",
        "dtype": "bfloat16", "repetition_penalty": 1.05,
    }
    assert analysis_fingerprint(
        interpreter_v1, auditor) != analysis_fingerprint(interpreter_v2, auditor)

    pool = object.__new__(OmniProcessPool)
    pool.omni_cfg = {
        "model_path": "models/pooled-omni", "dtype": "float16",
        "repetition_penalty": 1.1,
    }
    assert runner_identity(pool) == {
        "class": "OmniProcessPool", "model_path": "models/pooled-omni",
        "dtype": "float16", "repetition_penalty": 1.1,
    }


def test_measured_editing_metrics_split_shot_cut_transition_rates() -> None:
    measured = build_measured_editing_facts(
        build_agent_reference(_validated_reference()))
    rows = measured["pace_curve"]

    assert [row["content_shot_count"] for row in rows] == [1, 2, 2]
    assert [row["edit_boundary_count"] for row in rows] == [0, 1, 1]
    assert [row["hard_cut_count"] for row in rows] == [0, 1, 0]
    assert [row["transition_count"] for row in rows] == [0, 0, 1]
    assert [row["shot_rate"] for row in rows] == [0.2, 1.0, 2.0]
    assert [row["edit_boundary_rate"] for row in rows] == [0.0, 0.5, 1.0]
    assert [row["hard_cut_rate"] for row in rows] == [0.0, 0.5, 0.0]
    assert [row["transition_rate"] for row in rows] == [0.0, 0.0, 1.0]
    assert [row["raw_segment_rate"] for row in rows] == [0.2, 1.0, 3.0]
    assert rows[1]["mean_shot_duration_s"] == 1.0
    assert rows[1]["median_shot_duration_s"] == 1.0
    assert measured["facts"][3]["transition_out"]["transition_id"] == "T2"
    assert measured["pace_changes"][0]["shot_rate_before"] == 0.2
    structural = measured["structural_grammar"]
    assert structural["schema_version"] == "measured_structural_editing_grammar_v1"
    assert [row["relative_pace"] for row in structural["pace_profile"]] == [
        "low", "medium", "high",
    ]
    assert [row["direction"] for row in structural["pace_changes"]] == [
        "accelerates", "accelerates",
    ]
    assert structural["duration_profile"][2]["min_shot_duration_s"] == 0.4
    assert structural["shot_duration_outliers"][0]["shot_id"] == "S1"
    assert structural["transition_profile"][2]["type_counts"] == {"whip_pan": 1}
    assert structural["transition_sequences"] == [{
        "section_id": "C", "transition_ids": ["T2"],
        "transition_types": ["whip_pan"],
    }]


def test_editing_payload_is_compact_indexed_and_excludes_identity_provenance() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)
    payload = build_editing_payload(agent_reference, measured)
    first = payload["shots"][0]

    assert first["visual_claim_ids"] == ["V1"]
    assert first["canonical_subject_ids"] == ["E1", "E2"]
    assert first["event_ids"] == ["EV1"]
    assert first["transition_in"] is None
    assert "identity_claim_ids" not in first
    assert set(payload["claim_index"]) == {"V1", "T01", "A1"}
    assert payload["claim_index"]["V1"] == {
        "claim_id": "V1", "subject": "E1", "predicate": "appears_in",
        "object": "scene", "interval": [0.0, 5.0], "modality": "V",
    }
    assert set(payload["event_index"]["EV1"]) == {
        "event_id", "participants", "action_claim_ids", "object_ids",
        "ordering", "outcome_claim_ids", "context_claim_ids", "interval",
    }
    serialized = json.dumps(payload)
    assert "source_sha" not in serialized
    assert "support_refs" not in serialized
    assert "reviewer" not in serialized
    assert "reference.mp4" not in serialized
    assert payload["measured_editing"]["pace_curve"][1]["shot_rate"] == 1.0


def _shot_evidence_ids(payload: dict) -> dict[str, set[str]]:
    fields = ("visual_claim_ids", "text_claim_ids", "audio_claim_ids", "event_ids")
    return {
        row["shot_id"]: {
            value for field in fields for value in row.get(field, [])
        }
        for row in payload["shots"]
    }


def _pattern_value(pattern_type: str = "contrast_cut") -> dict:
    return {
        "schema_version": "semantic_editing_patterns_v1",
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
    value["patterns"][0]["fact_ids"] = ["EF1", "EF2", "EF3", "EF4", "EF5"]
    value["patterns"][0]["evidence_ids"] = ["V1"]
    payload = build_editing_payload(agent_reference, measured)

    with pytest.raises(EditingGrammarError) as excinfo:
        validate_editing_patterns(value, measured,
                                  shot_evidence_ids=_shot_evidence_ids(payload))
    assert excinfo.value.reason_code == "local_pattern_too_broad"


@pytest.mark.parametrize("pattern_type", [
    "long_hold", "rhythmic_acceleration", "rhythmic_deceleration",
    "transition_chain",
])
def test_deterministic_editing_types_are_not_semantic_patterns(
        pattern_type: str) -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)
    evidence = _shot_evidence_ids(build_editing_payload(agent_reference, measured))
    value = _pattern_value(pattern_type)
    with pytest.raises(EditingGrammarError) as excinfo:
        validate_editing_patterns(value, measured, shot_evidence_ids=evidence)
    assert excinfo.value.reason_code == "pattern_type_invalid"


def test_semantic_pattern_requires_local_evidence_and_exact_fact_scope() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)
    evidence = _shot_evidence_ids(build_editing_payload(agent_reference, measured))

    wrong_facts = _pattern_value()
    wrong_facts["patterns"][0]["fact_ids"] = ["EF1", "EF2"]
    with pytest.raises(EditingGrammarError) as fact_error:
        validate_editing_patterns(
            wrong_facts, measured, shot_evidence_ids=evidence)
    assert fact_error.value.reason_code == "pattern_fact_scope_invalid"

    wrong_evidence = _pattern_value()
    wrong_evidence["patterns"][0]["evidence_ids"] = ["V1"]
    with pytest.raises(EditingGrammarError) as evidence_error:
        validate_editing_patterns(
            wrong_evidence, measured, shot_evidence_ids=evidence)
    assert evidence_error.value.reason_code == "pattern_evidence_scope_invalid"

    missing_evidence = _pattern_value()
    missing_evidence["patterns"][0]["evidence_ids"] = []
    with pytest.raises(EditingGrammarError) as missing_error:
        validate_editing_patterns(
            missing_evidence, measured, shot_evidence_ids=evidence)
    assert missing_error.value.reason_code == "pattern_evidence_required"


def test_deterministic_structure_does_not_require_semantic_coverage() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    measured = build_measured_editing_facts(agent_reference)
    evidence = _shot_evidence_ids(build_editing_payload(agent_reference, measured))
    result = validate_editing_patterns(
        _pattern_value(), measured, shot_evidence_ids=evidence)

    assert result["patterns"][0]["type"] == "contrast_cut"
    assert "coverage_warnings" not in result
    assert measured["structural_grammar"]["pace_changes"][0][
        "direction"] == "accelerates"


def test_editing_recognition_and_function_reasoning_are_two_single_calls() -> None:
    agent_reference = build_agent_reference(_validated_reference())
    pattern_runner = _Runner(_pattern_value("rapid_montage"), "pattern-v1")
    function_runner = _Runner({
        "schema_version": "editing_functions_v2",
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
    assert "verified_narrative_interpretation" not in pattern_runner.prompts[0]
    for deterministic_type in (
        "long_hold", "rhythmic_acceleration", "rhythmic_deceleration",
        "transition_chain",
    ):
        assert deterministic_type not in pattern_runner.prompts[0]
    assert '\"predicate\":\"audio_event\"' in pattern_runner.prompts[0]
    assert "verified_narrative_interpretation" in function_runner.prompts[0]
    assert '\"object\":\"sound\"' in function_runner.prompts[0]
    assert '\"object\":\"scene\"' not in function_runner.prompts[0]
    assert result["semantic_patterns"][0]["evidence_ids"] == ["A1"]
    assert result["structural_grammar"] == measured["structural_grammar"]
    assert result["functions"][0]["function"] == "accumulate"
    assert measured["schema_version"] == "measured_editing_facts_v4"
