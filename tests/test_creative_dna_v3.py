from __future__ import annotations

from copy import deepcopy

import pytest

from src.agentic_video.manifest import json_hash


def _finalize(value: dict) -> dict:
    value = deepcopy(value)
    value.pop("artifact_sha", None)
    value["artifact_sha"] = json_hash(value)
    return value


def _complete_audit() -> dict:
    return _finalize({
        "schema_version": "creative_dna_audit_v3",
        "dna_status": "complete",
        "abstraction_confidence": 0.86,
        "unsupported_dimensions": [],
        "input_artifacts": [
            {
                "artifact_id": "reference_narrative",
                "artifact_type": "narrative_interpretation",
                "sha": "a" * 64,
                "schema_version": "narrative_interpretation_v2",
            },
            {
                "artifact_id": "editing_analysis",
                "artifact_type": "editing_analysis",
                "sha": "b" * 64,
                "schema_version": "editing_analysis_v2",
            },
        ],
        "mechanism_graph": {
            "nodes": [
                {
                    "node_id": "M1",
                    "kind": "information_state",
                    "abstract_role": "an initially available interpretation",
                    "support_refs": ["N1"],
                },
                {
                    "node_id": "M2",
                    "kind": "evidence_role",
                    "abstract_role": "later observable information",
                    "support_refs": ["N2"],
                },
            ],
            "edges": [
                {
                    "edge_id": "ME1",
                    "mechanism": "reframes",
                    "source": "M2",
                    "target": "M1",
                    "condition": "new observable information bears on an earlier interpretation",
                    "support_refs": ["R1"],
                    "confidence": 0.84,
                }
            ],
        },
        "experience_arc": [
            {
                "state_id": "X1",
                "order": 1,
                "state_role": "establish",
                "information_state": "one incomplete interpretation is available",
                "caused_by_edge_ids": [],
                "support_refs": ["N1"],
            }
        ],
        "event_constraints": [
            {
                "constraint_id": "EC1",
                "obligation": "required",
                "abstract_role": "supply observable information bearing on an earlier proposition",
                "satisfies_node_ids": ["M2"],
                "satisfies_edge_ids": ["ME1"],
                "must_precede_constraint_ids": [],
                "support_refs": ["N2", "R1"],
                "confidence": 0.82,
            }
        ],
        "editing_constraints": [
            {
                "constraint_id": "ED1",
                "level": "structural",
                "obligation": "preferred",
                "dimension": "pace",
                "operator": "increases",
                "phase_refs": ["X1"],
                "source_strength": "measured",
                "transfer_policy": "preserve_relation",
                "support_refs": ["EF1"],
                "confidence": 0.91,
            },
            {
                "constraint_id": "ED2",
                "level": "semantic_function",
                "obligation": "advisory",
                "dimension": "information_function",
                "operator": "accumulates",
                "phase_refs": ["X1"],
                "source_strength": "unaudited_semantic",
                "transfer_policy": "free_realization",
                "support_refs": ["EP1"],
                "confidence": 0.62,
            },
        ],
        "free_slots": [
            {
                "slot_id": "FS1",
                "kind": "domain",
                "constraints": ["must be newly bound without reference surface content"],
            }
        ],
        "source_bindings": [
            {
                "abstract_id": "M1",
                "source_refs": ["N1", "N2", "R1"],
                "reference_specific_summary": "a fencing contest provides the source example",
            },
            {
                "abstract_id": "ED1",
                "source_refs": ["EF1", "EP1"],
                "reference_specific_summary": "the source ending uses a rapid activity sequence",
            },
        ],
        "anti_invariants": [
            {
                "binding_id": "AI1",
                "category": "domain",
                "source_refs": ["N1"],
            }
        ],
        "validation_record": {
            "grounding_passed": True,
            "relation_entailment_passed": True,
            "surface_binding_confined_to_audit": True,
            "editing_promotion_policy_passed": True,
            "publish_whitelist_passed": True,
        },
    })


def _editing_only_partial() -> dict:
    value = _complete_audit()
    value["dna_status"] = "partial"
    value["unsupported_dimensions"] = [
        "narrative_mechanism",
        "experience_arc",
        "event_constraints",
        "editing_function",
        "free_slots",
    ]
    value["mechanism_graph"] = None
    value["experience_arc"] = []
    value["event_constraints"] = []
    value["editing_constraints"] = [value["editing_constraints"][0]]
    value["editing_constraints"][0]["phase_refs"] = []
    value["free_slots"] = []
    value["source_bindings"] = [value["source_bindings"][1]]
    value["anti_invariants"] = []
    return _finalize(value)


def test_complete_audit_publishes_only_whitelisted_control_data() -> None:
    from src.agentic_video.creative_dna_v3.publisher import publish_creative_spec
    from src.agentic_video.creative_dna_v3.validators import validate_creative_spec

    spec = publish_creative_spec(_complete_audit(), spec_id="CS-TEST")

    assert spec["dna_status"] == "complete"
    assert spec["available_dimensions"] == [
        "narrative_mechanism",
        "experience_arc",
        "event_constraints",
        "editing_structure",
        "editing_function",
        "free_slots",
    ]
    assert spec["narrative_control"]["mechanism_graph"]["nodes"][0] == {
        "node_id": "M1",
        "kind": "information_state",
        "abstract_role": "an initially available interpretation",
    }
    assert spec["editing_control"]["constraints"][1]["obligation"] == "advisory"
    assert spec["downstream_contract"] == {
        "must_preserve_constraint_ids": ["EC1"],
        "may_vary_slot_ids": ["FS1"],
        "reference_context_allowed": False,
    }
    serialized = repr(spec)
    for forbidden in (
        "support_refs", "source_bindings", "anti_invariants", "input_artifacts",
        "confidence", "N1", "R1", "EF1", "EP1", "fencing",
    ):
        assert forbidden not in serialized
    assert spec["artifact_sha"] == json_hash({
        key: value for key, value in spec.items() if key != "artifact_sha"
    })
    validate_creative_spec(spec)


def test_partial_editing_only_audit_can_publish_without_forced_narrative() -> None:
    from src.agentic_video.creative_dna_v3.publisher import publish_creative_spec

    spec = publish_creative_spec(_editing_only_partial(), spec_id="CS-PARTIAL")

    assert spec["dna_status"] == "partial"
    assert spec["available_dimensions"] == ["editing_structure"]
    assert spec["narrative_control"] is None
    assert len(spec["editing_control"]["constraints"]) == 1
    assert spec["free_slots"] == []


def test_blocked_audit_never_publishes() -> None:
    from src.agentic_video.creative_dna_v3.publisher import publish_creative_spec
    from src.agentic_video.creative_dna_v3.validators import DNAV3Error

    audit = _editing_only_partial()
    audit["dna_status"] = "blocked"
    audit["unsupported_dimensions"] = [
        "narrative_mechanism", "experience_arc", "event_constraints",
        "editing_structure", "editing_function", "free_slots",
    ]
    audit["editing_constraints"] = []
    audit = _finalize(audit)

    with pytest.raises(DNAV3Error, match="dna_blocked"):
        publish_creative_spec(audit, spec_id="CS-BLOCKED")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.update(unsupported_dimensions=["free_slots"]),
         "complete_has_unsupported_dimensions"),
        (lambda value: value.update(dna_status="partial", unsupported_dimensions=[]),
         "partial_missing_unsupported_dimensions"),
        (lambda value: value.update(abstraction_confidence=1.1),
         "abstraction_confidence_invalid"),
    ],
)
def test_status_and_confidence_mismatches_are_rejected(mutate, reason: str) -> None:
    from src.agentic_video.creative_dna_v3.validators import (
        DNAV3Error,
        validate_dna_audit,
    )

    audit = _complete_audit()
    mutate(audit)
    audit = _finalize(audit)
    with pytest.raises(DNAV3Error, match=reason):
        validate_dna_audit(audit)


@pytest.mark.parametrize(
    ("field", "value"),
    [("obligation", "required"), ("transfer_policy", "preserve_function")],
)
def test_unaudited_semantic_constraints_cannot_be_promoted(field: str, value: str) -> None:
    from src.agentic_video.creative_dna_v3.validators import (
        DNAV3Error,
        validate_dna_audit,
    )

    audit = _complete_audit()
    audit["editing_constraints"][1][field] = value
    audit = _finalize(audit)
    with pytest.raises(DNAV3Error, match="unaudited_semantic_promoted"):
        validate_dna_audit(audit)


def test_unknown_support_reference_is_rejected() -> None:
    from src.agentic_video.creative_dna_v3.validators import (
        DNAV3Error,
        validate_dna_audit,
    )

    audit = _complete_audit()
    audit["event_constraints"][0]["support_refs"].append("UNKNOWN99")
    audit = _finalize(audit)
    with pytest.raises(DNAV3Error, match="support_ref_unresolved"):
        validate_dna_audit(audit)


def test_reference_specific_language_cannot_cross_publish_boundary() -> None:
    from src.agentic_video.creative_dna_v3.publisher import publish_creative_spec
    from src.agentic_video.creative_dna_v3.validators import DNAV3Error

    audit = _complete_audit()
    audit["mechanism_graph"]["nodes"][0]["abstract_role"] = (
        "a fencing contest establishes the interpretation"
    )
    audit = _finalize(audit)
    with pytest.raises(DNAV3Error, match="reference_surface_leak"):
        publish_creative_spec(audit, spec_id="CS-LEAK")


def test_lineage_metadata_uses_exact_parent_shas() -> None:
    from src.agentic_video.creative_dna_v3.lineage import (
        build_audit_provenance,
        build_spec_provenance,
    )

    audit = _complete_audit()
    audit_metadata = build_audit_provenance(
        audit, producer_run="run-r2c", created_by="r2c-extractor"
    )
    assert audit_metadata["derived_from"] == [
        {"artifact_id": "reference_narrative", "sha": "a" * 64},
        {"artifact_id": "editing_analysis", "sha": "b" * 64},
    ]
    assert "version" not in audit_metadata

    spec_metadata = build_spec_provenance(
        spec_sha="c" * 64,
        audit_artifact_id="creative_dna_audit_v3:v1",
        audit_sha=audit["artifact_sha"],
        producer_run="run-r2c",
        created_by="r2c-publisher",
    )
    assert spec_metadata["derived_from"] == [{
        "artifact_id": "creative_dna_audit_v3:v1",
        "sha": audit["artifact_sha"],
    }]


@pytest.mark.parametrize("bad_parents", [
    [{"artifact_id": "a", "sha": "d" * 64},
     {"artifact_id": "a", "sha": "d" * 64}],
    [{"artifact_id": "", "sha": "d" * 64}],
    [{"artifact_id": "a", "sha": "not-a-sha"}],
])
def test_lineage_rejects_duplicate_or_inexact_parents(bad_parents: list[dict]) -> None:
    from src.agentic_video.creative_dna_v3.lineage import validate_parent_rows
    from src.agentic_video.creative_dna_v3.validators import DNAV3Error

    with pytest.raises(DNAV3Error):
        validate_parent_rows(bad_parents)


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


def _source_artifact(value: dict) -> dict:
    return _finalize(value)


def _r2_sources() -> tuple[dict, dict]:
    narrative = _source_artifact({
        "schema_version": "reference_narrative_interpretation_v2",
        "narrative_units": [
            {
                "unit_id": "N1",
                "section_ids": ["section_01"],
                "summary": "An initial attributed statement presents a broad judgment.",
                "supported_by": ["S1_T01"],
                "verification_status": "SUPPORTED",
            },
            {
                "unit_id": "N2",
                "section_ids": ["section_02"],
                "summary": "Later observable information bears on that judgment.",
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
                "reason": "The later information changes how the earlier judgment is read.",
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
            "pace_profile": [
                {"section_id": "section_01", "shot_rate": 0.2,
                 "relative_pace": "low"},
                {"section_id": "section_02", "shot_rate": 0.8,
                 "relative_pace": "high"},
            ],
            "pace_changes": [
                {"from_section": "section_01", "to_section": "section_02",
                 "direction": "accelerates", "shot_rate_before": 0.2,
                 "shot_rate_after": 0.8},
            ],
            "duration_profile": [],
            "shot_duration_outliers": [
                {"fact_id": "EF1", "shot_id": "shot_01",
                 "section_id": "section_01", "duration_s": 5.0,
                 "direction": "long", "first_quartile_s": 0.5,
                 "third_quartile_s": 1.0, "interquartile_range_s": 0.5},
            ],
            "transition_profile": [],
            "transition_sequences": [],
            "artifact_sha": "c" * 64,
        },
        "semantic_patterns": [
            {"pattern_id": "EP1", "type": "match_cut", "scope": "local",
             "shot_ids": ["shot_01", "shot_02"], "confidence": 0.8},
        ],
        "functions": [
            {"pattern_id": "EP1", "function": "accumulate",
             "relation_to_story": ["R1"], "confidence": 0.8},
        ],
        "pattern_limitations": [],
        "function_limitations": [],
        "measured_editing_sha": "b" * 64,
        "semantic_patterns_sha": "a" * 64,
        "functions_sha": "9" * 64,
    })
    return narrative, editing


def _extraction_response() -> dict:
    audit = _complete_audit()
    for key in ("schema_version", "input_artifacts", "validation_record",
                "artifact_sha"):
        audit.pop(key)
    return audit


def _independent_audit_response(*, passed: bool = True) -> dict:
    checks = []
    for item_id, item_type in (
        ("M1", "node"), ("M2", "node"), ("ME1", "edge"),
        ("X1", "experience_state"), ("EC1", "event_constraint"),
        ("ED1", "editing_constraint"), ("ED2", "editing_constraint"),
        ("FS1", "free_slot"),
    ):
        checks.append({
            "item_id": item_id,
            "item_type": item_type,
            "grounding_status": "not_applicable" if item_type == "free_slot" else "supported",
            "abstraction_valid": passed,
            "constraint_valid": passed,
            "reason": "independently checked against the supplied R2 bundle",
        })
    return {
        "schema_version": "creative_dna_independent_audit_v1",
        "item_checks": checks,
        "leakage_findings": [],
        "unsupported_dimensions_confirmed": [],
        "overall": {
            "grounding_passed": passed,
            "relation_entailment_passed": passed,
            "surface_binding_confined_to_audit": passed,
            "abstraction_useful": passed,
            "pass": passed,
        },
        "limitations": [],
    }


def test_r2c2_calls_each_model_once_and_writes_publish_trace(tmp_path) -> None:
    from src.agentic_video.creative_dna_v3.audit import run_r2c2

    narrative, editing = _r2_sources()
    extraction_runner = _Runner([_extraction_response()])
    audit_runner = _Runner([_independent_audit_response()])

    result = run_r2c2(
        extraction_runner, audit_runner, narrative, editing,
        trace_dir=tmp_path, spec_id="CS-R2C2-TEST",
    )

    assert result["status"] == "PUBLISHED"
    assert len(extraction_runner.calls) == 1
    assert len(audit_runner.calls) == 1
    assert result["creative_spec"]["schema_version"] == "creative_spec_v1"
    assert result["creative_spec"]["downstream_contract"][
        "reference_context_allowed"] is False
    for relative in (
        "01_extraction/request.json",
        "01_extraction/raw_response.txt",
        "01_extraction/candidate.json",
        "02_independent_audit/request.json",
        "02_independent_audit/raw_response.txt",
        "02_independent_audit/audit_result.json",
        "03_publish/creative_dna_audit_v3.json",
        "03_publish/creative_spec_v1.json",
        "04_transferability/manual_review_cases.json",
        "run_summary.json",
    ):
        assert (tmp_path / relative).is_file(), relative


def test_r2c2_failed_independent_audit_blocks_without_retry_or_publish(tmp_path) -> None:
    from src.agentic_video.creative_dna_v3.audit import run_r2c2

    narrative, editing = _r2_sources()
    extraction_runner = _Runner([_extraction_response()])
    audit_runner = _Runner([_independent_audit_response(passed=False)])

    result = run_r2c2(
        extraction_runner, audit_runner, narrative, editing,
        trace_dir=tmp_path, spec_id="CS-R2C2-BLOCKED",
    )

    assert result["status"] == "BLOCKED"
    assert result["creative_spec"] is None
    assert len(extraction_runner.calls) == len(audit_runner.calls) == 1
    assert not (tmp_path / "03_publish/creative_spec_v1.json").exists()


def test_r2_bundle_contains_no_raw_claim_or_identity_scaffolding() -> None:
    from src.agentic_video.creative_dna_v3.extractor import build_r2_bundle

    narrative, editing = _r2_sources()
    bundle = build_r2_bundle(narrative, editing)
    serialized = repr(bundle)
    assert "accepted_claims" not in serialized
    assert "identity" not in serialized.lower()
    assert "source_bindings" not in serialized
    assert bundle["editing_grammar"]["semantic_patterns"][0][
        "source_strength"] == "unaudited_semantic"


def test_abstraction_boundary_rejects_reused_source_ids() -> None:
    from src.agentic_video.creative_dna_v3.abstraction_validator import (
        AbstractionBoundaryError,
        validate_abstraction_boundary,
    )
    from src.agentic_video.creative_dna_v3.extractor import build_r2_bundle

    narrative, editing = _r2_sources()
    candidate = _complete_audit()
    candidate["mechanism_graph"]["nodes"][0]["node_id"] = "N1"
    candidate["mechanism_graph"]["edges"][0]["target"] = "N1"
    candidate = _finalize(candidate)

    with pytest.raises(AbstractionBoundaryError, match="abstract_id_reuses_source_id"):
        validate_abstraction_boundary(candidate, build_r2_bundle(narrative, editing))


def test_abstraction_boundary_rejects_section_role_labels() -> None:
    from src.agentic_video.creative_dna_v3.abstraction_validator import (
        AbstractionBoundaryError,
        validate_abstraction_boundary,
    )
    from src.agentic_video.creative_dna_v3.extractor import build_r2_bundle

    narrative, editing = _r2_sources()
    candidate = _complete_audit()
    candidate["mechanism_graph"]["nodes"][0]["abstract_role"] = "establish"
    candidate = _finalize(candidate)

    with pytest.raises(AbstractionBoundaryError, match="abstract_role_not_mechanism"):
        validate_abstraction_boundary(candidate, build_r2_bundle(narrative, editing))


def test_abstraction_boundary_rejects_dynamic_reference_surface_copy() -> None:
    from src.agentic_video.creative_dna_v3.abstraction_validator import (
        AbstractionBoundaryError,
        validate_abstraction_boundary,
    )
    from src.agentic_video.creative_dna_v3.extractor import build_r2_bundle

    narrative, editing = _r2_sources()
    narrative["narrative_units"][1]["summary"] = (
        "A fencing contest supplies later observable information."
    )
    narrative = _finalize(narrative)
    candidate = _complete_audit()
    candidate["mechanism_graph"]["nodes"][1]["abstract_role"] = (
        "a fencing contest supplies later observable information"
    )
    candidate = _finalize(candidate)

    with pytest.raises(AbstractionBoundaryError, match="reference_surface_in_public_field"):
        validate_abstraction_boundary(candidate, build_r2_bundle(narrative, editing))


def test_abstraction_boundary_accepts_cross_domain_mechanism_roles() -> None:
    from src.agentic_video.creative_dna_v3.abstraction_validator import (
        validate_abstraction_boundary,
    )
    from src.agentic_video.creative_dna_v3.extractor import build_r2_bundle

    narrative, editing = _r2_sources()
    validate_abstraction_boundary(
        _complete_audit(), build_r2_bundle(narrative, editing))
