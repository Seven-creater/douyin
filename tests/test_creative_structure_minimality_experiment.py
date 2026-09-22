from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/creative_structure_minimality"


def _read(relative: str) -> dict:
    return json.loads((EXPERIMENT / relative).read_text(encoding="utf-8"))


def test_frozen_experiment_assets_cover_screening_ablation_and_controls() -> None:
    from experiments.creative_structure_minimality.analysis import (
        validate_cases,
        validate_full_structure,
        validate_screening_candidates,
    )

    candidates = _read("candidates/screening_candidates.json")
    full = _read("candidates/full_structure.json")
    cases = _read("bindings/cases.json")

    validate_screening_candidates(candidates)
    validate_full_structure(full)
    validate_cases(cases)
    assert [row["candidate_id"] for row in candidates["candidates"]] == [
        "A", "B", "C"]
    assert len([row for row in cases["cases"] if row["case_type"] == "positive"]) == 3
    negatives = [
        row for row in cases["cases"] if row["case_type"] == "negative"]
    assert len(negatives) == 2
    assert any(row["control"] == "same_surface_missing_information_update"
               for row in negatives)


def test_single_deletion_variants_include_dependency_closure_and_compression() -> None:
    from experiments.creative_structure_minimality.analysis import (
        build_single_deletion_variants,
        validate_full_structure,
    )

    full = _read("candidates/full_structure.json")
    validate_full_structure(full)
    variants = build_single_deletion_variants(full)

    assert len(variants) == len(full["components"])
    assert _read("deletion_tests/ablation_plan.json")["variants"] == variants
    by_root = {row["removed_root_id"]: row for row in variants}
    prior = by_root["I0_PRIOR_INTERPRETATION"]
    assert "R1_INFORMATION_UPDATE" in prior["removed_component_ids"]
    assert prior["compression_ratio"] == round(
        len(prior["removed_component_ids"]) / len(full["components"]), 6)
    for row in variants:
        assert set(row["removed_component_ids"]).isdisjoint(
            row["retained_component_ids"])
        assert len(row["removed_component_ids"]) + len(
            row["retained_component_ids"]) == len(full["components"])


def test_blank_evaluation_never_produces_a_selected_structure() -> None:
    from experiments.creative_structure_minimality.analysis import analyze_results

    report = analyze_results(
        _read("candidates/full_structure.json"),
        _read("deletion_tests/ablation_plan.json")["variants"],
        _read("bindings/cases.json"),
        _read("evaluation_forms/minimality_evaluation.json"),
    )

    assert report["status"] == "NOT_READY"
    assert report["selected_variant_id"] is None
    assert report["missing_result_ids"]


def test_analysis_selects_most_compressed_completed_passing_variant() -> None:
    from experiments.creative_structure_minimality.analysis import (
        analyze_results,
        build_single_deletion_variants,
    )

    full = {
        "schema_version": "creative_structure_full_candidate_v1",
        "candidate_id": "FULL_TEST",
        "components": [
            {"component_id": "A", "kind": "state", "statement": "a", "depends_on": []},
            {"component_id": "B", "kind": "relation", "statement": "b", "depends_on": ["A"]},
        ],
    }
    variants = build_single_deletion_variants(full)
    cases = {
        "schema_version": "creative_structure_transfer_cases_v1",
        "cases": [
            {
                "case_id": "P1", "case_type": "positive",
                "domain": "one", "description": "positive",
                "control": "cross_domain", "available_binding_material": [],
            },
            {
                "case_id": "N1", "case_type": "negative",
                "domain": "two", "description": "negative",
                "control": "surface_lure", "available_binding_material": [],
            },
        ],
    }

    def result(variant_id: str, *, source: bool, positive: bool,
               reject: bool, additions: int = 0) -> dict:
        return {
            "variant_id": variant_id,
            "source_sufficient": source,
            "case_decisions": {"P1": positive, "N1": reject},
            "binding_additions": additions,
            "unresolved_binding_decisions": 0,
            "reference_bound_commitments": 0,
            "reviewer_disagreements": 0,
        }

    evaluation = {
        "schema_version": "creative_structure_minimality_evaluation_v1",
        "results": [
            result("FULL_TEST", source=True, positive=True, reject=True,
                   additions=1),
            result("DELETE_A", source=False, positive=False, reject=False),
            result("DELETE_B", source=True, positive=True, reject=True),
        ],
    }

    report = analyze_results(full, variants, cases, evaluation)

    assert report["status"] == "SELECTED"
    assert report["selected_variant_id"] == "DELETE_B"
    assert report["selected_compression_ratio"] == 0.5
    assert report["component_necessity"] == {"A": True, "B": False}


def test_experiment_analysis_is_isolated_from_production_runtime() -> None:
    source = (EXPERIMENT / "analysis.py").read_text(encoding="utf-8")
    assert "src.agentic_video" not in source
    assert "OmniRunner" not in source
