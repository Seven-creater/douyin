from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/creative_structure_minimality"
R2_C5 = EXPERIMENT / "r2_c5"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_spec_is_exactly_the_selected_four_component_contract() -> None:
    from src.agentic_video.creative_structure_v1.freeze import (
        CORE_COMPONENT_IDS,
        build_frozen_spec,
        validate_frozen_spec,
    )

    spec = build_frozen_spec()
    validate_frozen_spec(spec)
    element_ids = [
        row["element_id"] for row in spec["structural_schema"]["elements"]]
    relation_ids = [
        row["relation_id"] for row in spec["structural_schema"]["relations"]]
    assert tuple([*element_ids, *relation_ids]) == CORE_COMPONENT_IDS
    assert spec["editing_schema"] == {"constraints": []}
    assert spec["downstream_contract"]["reference_context_allowed"] is False


def test_frozen_spec_rejects_extra_or_changed_structure() -> None:
    from src.agentic_video.creative_structure_v1.freeze import (
        build_frozen_spec,
        validate_frozen_spec,
    )
    from src.agentic_video.creative_structure_v1.validator import (
        StructureValidationError,
    )
    from src.agentic_video.manifest import json_hash

    spec = build_frozen_spec()
    spec["structural_schema"]["elements"][0]["abstract_role"] = (
        "a prior interpretation about an athlete is available")
    spec["artifact_sha"] = json_hash({
        key: value for key, value in spec.items() if key != "artifact_sha"})
    with pytest.raises(StructureValidationError,
                       match="frozen_spec_contract_mismatch"):
        validate_frozen_spec(spec)


def test_transfer_validation_passes_three_domains_and_rejects_controls() -> None:
    from src.agentic_video.creative_structure_v1.freeze import (
        build_frozen_spec,
        run_transfer_validation,
    )

    report = run_transfer_validation(
        build_frozen_spec(),
        _read(EXPERIMENT / "bindings/cases.json"),
        _read(R2_C5 / "transfer_bindings.json"),
    )
    assert report["status"] == "PASS"
    assert [row["passed"] for row in report["case_results"]] == [True] * 5
    assert report["checks"] == {
        "structural_schema_valid": True,
        "reference_context_excluded": True,
        "positive_transfer_passed": True,
        "surface_lure_rejected": True,
    }


def test_internal_review_is_not_misreported_as_human_evidence() -> None:
    review = _read(R2_C5 / "internal_review.json")
    original_evaluation = _read(
        EXPERIMENT / "evaluation_forms/minimality_evaluation.json")

    assert review["review_mode"] == "internal_simulation"
    assert review["human_evaluation_claim"] is False
    assert original_evaluation["results"] == []


def test_checked_in_freeze_artifacts_match_deterministic_generation() -> None:
    from src.agentic_video.creative_structure_v1.freeze import (
        write_freeze_artifacts,
    )

    result = write_freeze_artifacts(EXPERIMENT)
    assert result["freeze_record"]["status"] == "FROZEN"
    assert result["freeze_record"]["decision_origin"] == "internal_simulation"
    assert _read(R2_C5 / "creative_structure_spec_v1.json") == result["spec"]
    assert _read(R2_C5 / "transfer_report.json") == result["transfer_report"]
