from __future__ import annotations

from copy import deepcopy

import pytest

from src.agentic_video.manifest import json_hash


def _finalize(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("artifact_sha", None)
    result["artifact_sha"] = json_hash(result)
    return result


def _complete_audit(*, validated: bool = True) -> dict:
    return _finalize({
        "schema_version": "creative_structure_audit_v1",
        "plan_status": "complete",
        "abstraction_confidence": 0.87,
        "dimension_status": {
            "structural_schema": "supported",
            "information_state_arc": "supported",
            "generation_constraints": "supported",
            "editing_structure": "supported",
            "editing_function": "not_applicable",
            "binding_slots": "supported",
        },
        "input_artifacts": [
            {
                "artifact_id": "reference_narrative",
                "artifact_type": "narrative_interpretation",
                "sha": "a" * 64,
                "schema_version": "reference_narrative_interpretation_v2",
            },
            {
                "artifact_id": "editing_grammar",
                "artifact_type": "editing_grammar",
                "sha": "b" * 64,
                "schema_version": "editing_grammar_v4",
            },
        ],
        "structural_schema": {
            "elements": [
                {
                    "element_id": "V1",
                    "kind": "information_state",
                    "abstract_role": "an available but incomplete interpretation",
                    "support_refs": ["N1"],
                },
                {
                    "element_id": "V2",
                    "kind": "evidence_role",
                    "abstract_role": "observable information bearing on an interpretation",
                    "support_refs": ["N2"],
                },
            ],
            "relations": [
                {
                    "relation_id": "REL1",
                    "relation_kind": "evidential",
                    "predicate": "observable information bears on an available interpretation",
                    "arguments": [
                        {"role": "bearing_evidence", "element_id": "V2"},
                        {"role": "prior_interpretation", "element_id": "V1"},
                    ],
                    "preconditions": ["an interpretation is already available"],
                    "effects": ["the interpretation becomes eligible for revision"],
                    "support_refs": ["R1"],
                    "confidence": 0.84,
                }
            ],
        },
        "information_state_arc": [
            {
                "state_id": "IS1",
                "order": 1,
                "available_element_ids": ["V1"],
                "transition_from_state_id": None,
                "trigger_relation_ids": [],
                "support_refs": ["N1"],
            },
            {
                "state_id": "IS2",
                "order": 2,
                "available_element_ids": ["V1", "V2"],
                "transition_from_state_id": "IS1",
                "trigger_relation_ids": ["REL1"],
                "support_refs": ["N2", "R1"],
            },
        ],
        "generation_constraints": [
            {
                "constraint_id": "GC1",
                "obligation": "required",
                "constraint_type": "relational",
                "requires_relation_ids": ["REL1"],
                "rule": "provide a binding in which evidence bears on an available interpretation",
                "support_refs": ["R1"],
                "confidence": 0.82,
            }
        ],
        "editing_schema": {
            "constraints": [
                {
                    "constraint_id": "ED1",
                    "obligation": "preferred",
                    "dimension": "pace",
                    "operator": "increases",
                    "applies_to_state_ids": ["IS1", "IS2"],
                    "applies_to_relation_ids": ["REL1"],
                    "source_strength": "measured",
                    "transfer_policy": "preserve_relation",
                    "support_refs": ["PACE_CHANGE_01"],
                    "confidence": 0.91,
                }
            ]
        },
        "binding_slots": [
            {
                "slot_id": "B1",
                "slot_type": "domain",
                "bound_by": "downstream_skill",
                "constraints": ["bind a new domain without copying reference events"],
            }
        ],
        "audit_annex": {
            "source_bindings": [
                {
                    "abstract_id": "V1",
                    "source_refs": ["N1"],
                    "reference_specific_summary": "a source caption states a broad judgment",
                },
                {
                    "abstract_id": "V2",
                    "source_refs": ["N2"],
                    "reference_specific_summary": "a source contest supplies later evidence",
                },
                {
                    "abstract_id": "REL1",
                    "source_refs": ["R1"],
                    "reference_specific_summary": "later source evidence bears on the judgment",
                },
                {
                    "abstract_id": "IS1",
                    "source_refs": ["N1"],
                    "reference_specific_summary": "the source judgment is initially available",
                },
                {
                    "abstract_id": "IS2",
                    "source_refs": ["N2", "R1"],
                    "reference_specific_summary": "later source evidence becomes available",
                },
                {
                    "abstract_id": "GC1",
                    "source_refs": ["R1"],
                    "reference_specific_summary": "the source relation links evidence and judgment",
                },
                {
                    "abstract_id": "ED1",
                    "source_refs": ["PACE_CHANGE_01"],
                    "reference_specific_summary": "the source pace increases between sections",
                },
                {
                    "abstract_id": "B1",
                    "source_refs": ["N1", "N2"],
                    "reference_specific_summary": "the source domain and events are replaceable",
                },
            ],
            "anti_invariants": [
                {"binding_id": "AI1", "category": "domain", "source_refs": ["N2"]},
                {"binding_id": "AI2", "category": "literal_event", "source_refs": ["N2"]},
            ],
        },
        "validation_record": {
            "grounding_passed": validated,
            "relation_entailment_passed": validated,
            "abstraction_boundary_passed": validated,
            "transferability_passed": validated,
            "editing_promotion_policy_passed": validated,
            "publish_whitelist_passed": validated,
        },
    })


def _editing_only_partial() -> dict:
    audit = _complete_audit()
    audit["plan_status"] = "partial"
    audit["dimension_status"] = {
        "structural_schema": "insufficient",
        "information_state_arc": "insufficient",
        "generation_constraints": "insufficient",
        "editing_structure": "supported",
        "editing_function": "not_applicable",
        "binding_slots": "not_applicable",
    }
    audit["structural_schema"] = None
    audit["information_state_arc"] = []
    audit["generation_constraints"] = []
    audit["editing_schema"]["constraints"][0]["applies_to_state_ids"] = []
    audit["editing_schema"]["constraints"][0]["applies_to_relation_ids"] = []
    audit["binding_slots"] = []
    audit["audit_annex"]["source_bindings"] = [
        row for row in audit["audit_annex"]["source_bindings"]
        if row["abstract_id"] == "ED1"
    ]
    audit["audit_annex"]["anti_invariants"] = []
    return _finalize(audit)


def test_complete_plan_publishes_only_whitelisted_ir() -> None:
    from src.agentic_video.creative_structure_v1.publisher import publish_structure_spec

    spec = publish_structure_spec(_complete_audit(), spec_id="CS-TEST")

    assert spec["schema_version"] == "creative_structure_spec_v1"
    assert spec["structural_schema"]["relations"][0]["predicate"].startswith(
        "observable information")
    assert spec["downstream_contract"]["authorized_stages"] == [
        "theme", "story", "screenplay"
    ]
    serialized = repr(spec)
    for forbidden in (
        "support_refs", "source_refs", "audit_annex", "input_artifacts",
        "confidence", "N1", "R1", "PACE_CHANGE_01", "contest", "caption",
    ):
        assert forbidden not in serialized


def test_editing_only_partial_publishes_but_cannot_authorize_story() -> None:
    from src.agentic_video.creative_structure_v1.publisher import publish_structure_spec

    spec = publish_structure_spec(_editing_only_partial(), spec_id="CS-EDITING")

    assert spec["plan_status"] == "partial"
    assert spec["structural_schema"] is None
    assert spec["downstream_contract"]["authorized_stages"] == []
    assert spec["available_dimensions"] == ["editing_structure"]


def test_blocked_plan_cannot_publish() -> None:
    from src.agentic_video.creative_structure_v1.publisher import publish_structure_spec
    from src.agentic_video.creative_structure_v1.validator import StructureValidationError

    audit = _editing_only_partial()
    audit["plan_status"] = "blocked"
    audit["dimension_status"]["editing_structure"] = "insufficient"
    audit["editing_schema"]["constraints"] = []
    audit["audit_annex"]["source_bindings"] = []
    audit = _finalize(audit)

    with pytest.raises(StructureValidationError, match="plan_blocked"):
        publish_structure_spec(audit, spec_id="CS-BLOCKED")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.update(abstraction_confidence="high"),
         "abstraction_confidence_invalid"),
        (lambda value: value["dimension_status"].update(
            structural_schema="not_applicable"), "dimension_content_mismatch"),
        (lambda value: value.update(plan_status="complete") or value[
            "dimension_status"].update(editing_function="insufficient"),
         "complete_has_insufficient_dimension"),
    ],
)
def test_status_confidence_and_dimension_mismatches_are_rejected(
        mutate, reason: str) -> None:
    from src.agentic_video.creative_structure_v1.validator import (
        StructureValidationError,
        validate_structure_audit,
    )

    audit = _complete_audit()
    mutate(audit)
    audit = _finalize(audit)
    with pytest.raises(StructureValidationError, match=reason):
        validate_structure_audit(audit)


def test_relation_argument_and_information_transition_refs_are_strict() -> None:
    from src.agentic_video.creative_structure_v1.validator import (
        StructureValidationError,
        validate_structure_audit,
    )

    audit = _complete_audit()
    audit["structural_schema"]["relations"][0]["arguments"][0][
        "element_id"] = "UNKNOWN"
    audit = _finalize(audit)
    with pytest.raises(StructureValidationError, match="relation_element_unresolved"):
        validate_structure_audit(audit)

    audit = _complete_audit()
    audit["information_state_arc"][1]["trigger_relation_ids"] = []
    audit = _finalize(audit)
    with pytest.raises(StructureValidationError,
                       match="information_transition_trigger_missing"):
        validate_structure_audit(audit)


def test_generation_constraints_must_reference_supported_relations() -> None:
    from src.agentic_video.creative_structure_v1.validator import (
        StructureValidationError,
        validate_structure_audit,
    )

    audit = _complete_audit()
    audit["generation_constraints"][0]["requires_relation_ids"] = []
    audit = _finalize(audit)
    with pytest.raises(StructureValidationError,
                       match="generation_relation_refs_invalid"):
        validate_structure_audit(audit)


def test_unaudited_semantic_editing_cannot_be_promoted() -> None:
    from src.agentic_video.creative_structure_v1.validator import (
        StructureValidationError,
        validate_structure_audit,
    )

    audit = _complete_audit()
    row = audit["editing_schema"]["constraints"][0]
    row["source_strength"] = "unaudited_semantic"
    row["obligation"] = "preferred"
    row["transfer_policy"] = "preserve_function"
    audit["dimension_status"]["editing_structure"] = "not_applicable"
    audit["dimension_status"]["editing_function"] = "supported"
    audit = _finalize(audit)
    with pytest.raises(StructureValidationError, match="unaudited_semantic_promoted"):
        validate_structure_audit(audit)


def test_publisher_rejects_reference_surface_in_public_fields() -> None:
    from src.agentic_video.creative_structure_v1.publisher import publish_structure_spec
    from src.agentic_video.creative_structure_v1.validator import StructureValidationError

    audit = _complete_audit()
    audit["structural_schema"]["elements"][1]["abstract_role"] = (
        "evidence from a contest")
    audit = _finalize(audit)

    with pytest.raises(StructureValidationError, match="reference_surface_leak"):
        publish_structure_spec(audit, spec_id="CS-LEAK")


def _source_artifact(value: dict) -> dict:
    return _finalize(value)


def _r2_sources() -> tuple[dict, dict]:
    narrative = _source_artifact({
        "schema_version": "reference_narrative_interpretation_v2",
        "narrative_units": [
            {
                "unit_id": "N1",
                "section_ids": ["section_01"],
                "summary": "An attributed statement presents a broad judgment.",
                "supported_by": ["S1_T01"],
                "verification_status": "SUPPORTED",
            },
            {
                "unit_id": "N2",
                "section_ids": ["section_02"],
                "summary": "A fencing contest supplies observable evidence.",
                "supported_by": ["S2_V01"],
                "verification_status": "SUPPORTED",
            },
        ],
        "relations": [
            {
                "relation_id": "R1",
                "type": "reframes",
                "source": "N2",
                "target": "N1",
                "reason": "The later evidence bears on the earlier judgment.",
                "supported_by": ["S1_T01", "S2_V01"],
                "verification_status": "SUPPORTED",
            }
        ],
        "limitations": [],
        "verification_status": "SUPPORTED",
        "relation_audit_sha": "f" * 64,
        "agent_reference_sha": "e" * 64,
        "analysis_fingerprint": "d" * 64,
    })
    editing = _source_artifact({
        "schema_version": "editing_grammar_v4",
        "structural_grammar": {
            "schema_version": "measured_structural_editing_grammar_v1",
            "pace_profile": [],
            "pace_changes": [
                {
                    "from_section": "section_01",
                    "to_section": "section_02",
                    "direction": "accelerates",
                    "shot_rate_before": 0.2,
                    "shot_rate_after": 0.8,
                }
            ],
            "duration_profile": [],
            "shot_duration_outliers": [],
            "transition_profile": [],
            "transition_sequences": [],
            "artifact_sha": "c" * 64,
        },
        "semantic_patterns": [],
        "functions": [],
        "pattern_limitations": [],
        "function_limitations": [],
        "measured_editing_sha": "b" * 64,
        "semantic_patterns_sha": "a" * 64,
        "functions_sha": "9" * 64,
    })
    return narrative, editing


class _Answer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.input_tokens = 100
        self.output_tokens = 50
        self.elapsed_s = 1.25


class _Runner:
    def __init__(self, responses: list[dict]) -> None:
        import json

        self.responses = [json.dumps(row) for row in responses]
        self.calls: list[tuple[str, dict]] = []
        self.cfg = {"model_path": "fake", "repetition_penalty": 1.05}

    def ask(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        return _Answer(self.responses.pop(0))


def _extraction_response() -> dict:
    audit = _complete_audit(validated=False)
    for key in ("schema_version", "input_artifacts", "validation_record",
                "artifact_sha"):
        audit.pop(key)
    return audit


def _independent_audit_response(*, passed: bool = True) -> dict:
    checks = []
    for item_id, item_type in (
        ("V1", "element"), ("V2", "element"), ("REL1", "relation"),
        ("IS1", "information_state"), ("IS2", "information_state"),
        ("GC1", "generation_constraint"), ("ED1", "editing_constraint"),
        ("B1", "binding_slot"),
    ):
        checks.append({
            "item_id": item_id,
            "item_type": item_type,
            "grounding_status": (
                "not_applicable" if item_type == "binding_slot" else "supported"),
            "abstraction_valid": passed,
            "transfer_valid": passed,
            "constraint_valid": passed,
            "reason": "checked against the verified bundle and transfer profiles",
        })
    return {
        "schema_version": "creative_structure_independent_audit_v1",
        "item_checks": checks,
        "leakage_findings": [],
        "dimension_status_confirmed": _complete_audit()["dimension_status"],
        "overall": {
            "grounding_passed": passed,
            "relation_entailment_passed": passed,
            "abstraction_boundary_passed": passed,
            "transferability_passed": passed,
            "pass": passed,
        },
        "limitations": [],
    }


def test_r2_bundle_excludes_raw_claims_identity_and_old_dna_scaffolding() -> None:
    from src.agentic_video.creative_structure_v1.extractor import build_verified_bundle

    narrative, editing = _r2_sources()
    bundle = build_verified_bundle(narrative, editing)
    serialized = repr(bundle).lower()
    assert "accepted_claims" not in serialized
    assert "identity" not in serialized
    assert "narrative_functions" not in serialized
    assert "mechanism_graph" not in serialized
    assert bundle["evidence_catalog"][-1]["source_strength"] == "measured"


def test_dynamic_abstraction_boundary_rejects_source_surface_copy() -> None:
    from src.agentic_video.creative_structure_v1.extractor import build_verified_bundle
    from src.agentic_video.creative_structure_v1.validator import (
        StructureValidationError,
        validate_abstraction_boundary,
    )

    narrative, editing = _r2_sources()
    audit = _complete_audit()
    audit["structural_schema"]["elements"][1]["abstract_role"] = (
        "a fencing contest supplies evidence")
    audit = _finalize(audit)

    with pytest.raises(StructureValidationError,
                       match="reference_surface_in_public_field"):
        validate_abstraction_boundary(audit, build_verified_bundle(narrative, editing))


def test_single_extraction_single_audit_publish_and_transfer_trace(tmp_path) -> None:
    from src.agentic_video.creative_structure_v1.extractor import (
        run_structure_acceptance,
    )

    narrative, editing = _r2_sources()
    extraction_runner = _Runner([_extraction_response()])
    audit_runner = _Runner([_independent_audit_response()])

    result = run_structure_acceptance(
        extraction_runner,
        audit_runner,
        narrative,
        editing,
        trace_dir=tmp_path,
        spec_id="CS-STRUCTURE-TEST",
    )

    assert result["status"] == "PUBLISHED"
    assert len(extraction_runner.calls) == len(audit_runner.calls) == 1
    assert result["retry_count"] == 0
    assert result["creative_structure_spec"]["downstream_contract"][
        "reference_context_allowed"] is False
    for relative in (
        "01_extraction/request.json",
        "01_extraction/raw_response.txt",
        "01_extraction/candidate.json",
        "02_independent_audit/request.json",
        "02_independent_audit/raw_response.txt",
        "02_independent_audit/audit_result.json",
        "03_publish/creative_structure_audit_v1.json",
        "03_publish/creative_structure_spec_v1.json",
        "04_transferability/transfer_test_plan.json",
        "run_summary.json",
    ):
        assert (tmp_path / relative).is_file(), relative


def test_failed_independent_audit_blocks_without_retry_or_publish(tmp_path) -> None:
    from src.agentic_video.creative_structure_v1.extractor import (
        run_structure_acceptance,
    )

    narrative, editing = _r2_sources()
    extraction_runner = _Runner([_extraction_response()])
    audit_runner = _Runner([_independent_audit_response(passed=False)])

    result = run_structure_acceptance(
        extraction_runner,
        audit_runner,
        narrative,
        editing,
        trace_dir=tmp_path,
        spec_id="CS-STRUCTURE-BLOCKED",
    )

    assert result["status"] == "BLOCKED"
    assert result["creative_structure_spec"] is None
    assert len(extraction_runner.calls) == len(audit_runner.calls) == 1
    assert result["retry_count"] == 0
    assert not (tmp_path / "03_publish/creative_structure_spec_v1.json").exists()


def test_transfer_plan_has_three_cross_domain_bindings_and_surface_lure() -> None:
    from src.agentic_video.creative_structure_v1.publisher import publish_structure_spec
    from src.agentic_video.creative_structure_v1.transfer_test import (
        build_transfer_test_plan,
    )

    spec = publish_structure_spec(_complete_audit(), spec_id="CS-TRANSFER")
    plan = build_transfer_test_plan(spec)

    positives = [case for case in plan["cases"] if case["expected"] == "preserve"]
    negatives = [case for case in plan["cases"] if case["expected"] == "reject"]
    assert len(positives) == 3
    assert len({case["binding_profile"]["domain"] for case in positives}) == 3
    assert len(negatives) == 1
    assert negatives[0]["case_type"] == "surface_lure_negative"
    assert plan["status"] == "PENDING_INDEPENDENT_REVIEW"
