from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from scripts import run_transfer_kernel_v2 as trial
from scripts import run_intent_story_trial as creative_trial
from src.agentic_video.manifest import json_hash
from src.agentic_video.transfer_eval_v2 import (
    decisions, feedback_codes, make_acceptance, public_fields, score,
    validate_acceptance, validate_mapping, validate_stance,
)
from src.agentic_video.transfer_kernel_v1 import (
    public_brief_draft, publish_brief,
)
from tests.test_transfer_kernel_v1 import audit_fixture, kernel_fixture


def _brief():
    return public_brief_draft(kernel_fixture())


def _case():
    brief = _brief()
    return {"case_id": "X1", "fields": public_fields(brief),
            "story": "A narrow view is reassessed after new action evidence. "
                     "People are assessed through fuller evidence."}


def _mapping(case):
    return {"schema_version": "transfer_mapping_eval_v2", "checks": [
        {"case_id": case["case_id"], "mappings": [
            {"path": field["path"], "brief_quote": field["text"],
             "candidate_quote": "new action evidence", "verdict": "mapped",
             "reason": "Concrete event carries the relation"}
            for field in case["fields"]]}]}


def _stance(case):
    return {"schema_version": "transfer_stance_eval_v2", "checks": [
        {"case_id": case["case_id"], "stance_match": True,
         "reason": "Audience conclusion is preserved"}]}


def test_mapping_requires_full_field_text_and_exact_candidate_quote():
    case = _case()
    mapping = _mapping(case)
    assert validate_mapping(mapping, [case]) == []
    mapping["checks"][0]["mappings"][0]["brief_quote"] = "Assess people"
    assert validate_mapping(mapping, [case]) == [
        "mapping_quote_or_verdict_invalid"]
    mapping = _mapping(case)
    mapping["checks"][0]["mappings"][0]["candidate_quote"] = "fabricated"
    assert validate_mapping(mapping, [case]) == [
        "mapping_quote_or_verdict_invalid"]
    mapping = _mapping(case)
    mapping["checks"][0]["mappings"].pop()
    assert validate_mapping(mapping, [case]) == ["mapping_coverage_invalid"]


def test_negative_mapping_quotes_are_optional_but_must_be_exact_if_present():
    case = _case()
    mapping = _mapping(case)
    row = mapping["checks"][0]["mappings"][0]
    row.update({"verdict": "conflict", "candidate_quote": None})
    assert validate_mapping(mapping, [case]) == []
    row.update({"verdict": "unmapped", "candidate_quote": "new action evidence"})
    assert validate_mapping(mapping, [case]) == []
    row["candidate_quote"] = "fabricated"
    assert validate_mapping(mapping, [case]) == [
        "mapping_quote_or_verdict_invalid"]
    row.update({"verdict": "mapped", "candidate_quote": None})
    assert validate_mapping(mapping, [case]) == [
        "mapping_quote_or_verdict_invalid"]


def test_any_unmapped_hard_field_rejects_even_when_stance_matches():
    case = _case()
    mapping, stance = _mapping(case), _stance(case)
    row = mapping["checks"][0]["mappings"][1]
    row.update({"candidate_quote": None, "verdict": "unmapped"})
    assert validate_stance(stance, [case]) == []
    assert decisions(mapping, stance, [case]) == {"X1": False}
    assert not score({"X1": False}, {"X1": True})["passed"]
    assert feedback_codes(mapping, [case], {"X1": True}) == [
        {"code": "invariant_not_transferable",
         "field_path": row["path"]}]


def test_eight_calibration_cases_are_separate_from_labels():
    root = Path(__file__).resolve().parents[1] / "config"
    cases, labels = trial._calibration_inputs(
        root / "transfer_judge_calibration_v2_cases.json",
        root / "transfer_judge_calibration_v2_labels.json")
    assert len(cases) == 8
    assert sum(labels.values()) == 2
    assert "expected_accept" not in json.dumps(cases)
    assert all(any(field["path"] == "stance" for field in row["fields"])
               for row in cases)


def test_acceptance_binds_real_eval_artifacts_and_sha():
    kernel, audit = kernel_fixture(), audit_fixture()
    mapping, stance = _mapping(_case()), _stance(_case())
    calibration, regression = {"passed": True}, {"passed": True}
    frozen_badcase = {"passed": True}
    brief = publish_brief(_brief(), kernel_sha="kernel-sha", audit=audit,
                          evaluation={"mapping_sha": json_hash(mapping),
                                      "stance_sha": json_hash(stance)},
                          contrast_score=regression)
    acceptance = make_acceptance(
        brief=brief, kernel_sha="kernel-sha", audit=audit,
        calibration=calibration, mapping=mapping, stance=stance,
        regression=regression, frozen_badcase=frozen_badcase)
    validate_acceptance(brief, acceptance, kernel_sha="kernel-sha",
                        audit=audit, calibration=calibration, mapping=mapping,
                        stance=stance, regression=regression,
                        frozen_badcase=frozen_badcase)
    altered = copy.deepcopy(mapping)
    altered["checks"][0]["mappings"][0]["reason"] = "changed"
    with pytest.raises(ValueError, match="trial_acceptance_invalid"):
        validate_acceptance(brief, acceptance, kernel_sha="kernel-sha",
                            audit=audit, calibration=calibration,
                            mapping=altered, stance=stance,
                            regression=regression,
                            frozen_badcase=frozen_badcase)


def test_current_false_positive_is_rejected_when_qualifier_conflicts():
    brief = _brief()
    brief["stance"] = "A person lacking a physical capability is undervalued."
    case = {"case_id": "C02", "fields": public_fields(brief),
            "story": "A student is dismissed for a poor exam score, then builds a robot."}
    mapping = _mapping(case)
    for row in mapping["checks"][0]["mappings"]:
        row["candidate_quote"] = "builds a robot"
    mapping["checks"][0]["mappings"][0].update({
        "candidate_quote": "poor exam score", "verdict": "conflict",
        "reason": "An exam score is not a physical capability"})
    stance = _stance(case)
    assert decisions(mapping, stance, [case]) == {"C02": False}


def test_old_candidate_without_new_attestation_cannot_start_trial():
    audit = audit_fixture()
    brief = publish_brief(_brief(), kernel_sha="old-kernel", audit=audit,
                          evaluation={"checks": []},
                          contrast_score={"passed": True})
    with pytest.raises(ValueError, match="trial_acceptance_required_for_v2"):
        creative_trial._trial_acceptance(
            argparse.Namespace(trial_acceptance=None), brief,
            {"artifact_sha": "old-kernel"})
