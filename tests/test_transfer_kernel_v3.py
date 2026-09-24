from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from scripts import run_transfer_kernel_v3 as trial
from src.agentic_video.manifest import json_hash
from src.agentic_video.transfer_kernel_v3 import (
    GROUNDING_PROMPT, KERNEL_PROMPT, PORTABILITY_PROMPT, audit_paths,
    make_acceptance, public_brief_draft, public_fields, publish_brief,
    validate_acceptance, validate_grounding, validate_kernel,
    validate_portability, validate_public_brief, validate_themes,
)


def kernel_fixture():
    return {"schema_version": "transfer_kernel_v3",
            "source_claim": {"statement": "A source-specific statement",
                             "support_ids": ["CL_A"]},
            "transfer_goal": {"statement": "Judge from fuller evidence",
                              "mapping_rationale": "The initial judgement is revised",
                              "support_ids": ["CL_A", "EV_B"]},
            "units": [
                {"unit_id": "U1", "source_binding": "private binding A",
                 "near_concept": "near A", "transfer_function":
                 "An incomplete initial assessment",
                 "audience_effect": "Audience forms a tentative view",
                 "support_ids": ["CL_A"]},
                {"unit_id": "U2", "source_binding": "private binding B",
                 "near_concept": "near B", "transfer_function":
                 "New evidence changes that assessment",
                 "audience_effect": "Audience revises its view",
                 "support_ids": ["EV_B"]}],
            "relations": [{"relation_id": "R1", "from_unit": "U1",
                           "to_unit": "U2", "relation": "is revised by",
                           "support_ids": ["CL_A", "EV_B"]}],
            "optional_order": ["U1", "U2"], "uncertainties": []}


def grounding_fixture(kernel):
    ids = (["source_claim", "transfer_goal"] +
           [row["unit_id"] for row in kernel["units"]] +
           [row["relation_id"] for row in kernel["relations"]])
    return {"schema_version": "transfer_grounding_audit_v3",
            "checks": [{"id": item, "verdict": "supported",
                        "reason": "The evidence supports this level"}
                       for item in ids]}


def portability_fixture(brief):
    return {"schema_version": "transfer_portability_audit_v3",
            "checks": [{"path": path, "verdict": "portable",
                        "reason": "Works after far-domain rebinding",
                        "tested_bindings": ["museum policy", "garden schedule"]}
                       for path in audit_paths(brief)]}


def test_far_brief_whitelists_private_bindings_and_allows_empty_relations():
    kernel = kernel_fixture()
    assert validate_kernel(kernel, {"CL_A", "EV_B"}) == []
    brief = public_brief_draft(kernel)
    assert "private binding" not in json.dumps(brief)
    assert "near A" not in json.dumps(brief)
    assert "support_ids" not in json.dumps(brief)
    assert brief["free_role_ids"] == ["U1", "U2"]
    assert [row["path"] for row in public_fields(brief)] == audit_paths(brief)
    kernel["relations"] = []
    kernel["optional_order"] = []
    assert validate_kernel(kernel, {"CL_A", "EV_B"}) == []
    validate_public_brief(public_brief_draft(kernel), published=False)


def test_evidence_refs_and_two_independent_audit_gates():
    kernel = kernel_fixture()
    kernel["units"][0]["support_ids"] = ["MISSING"]
    assert "unit_invalid" in validate_kernel(kernel, {"CL_A", "EV_B"})
    kernel = kernel_fixture()
    brief = public_brief_draft(kernel)
    grounding = grounding_fixture(kernel)
    portability = portability_fixture(brief)
    assert validate_grounding(grounding, kernel)
    assert validate_portability(portability, brief)
    grounding["checks"][0]["verdict"] = "insufficient"
    assert not validate_grounding(grounding, kernel)
    portability["checks"][1]["verdict"] = "source_domain_required"
    assert not validate_portability(portability, brief)
    portability["checks"][1]["tested_bindings"] = ["same", "same"]
    with pytest.raises(ValueError, match="portability_audit_invalid"):
        validate_portability(portability, brief)
    portability = portability_fixture(brief)
    portability["checks"].pop(2)
    with pytest.raises(ValueError, match="portability_audit_invalid"):
        validate_portability(portability, brief)


def test_v3_attestation_binds_parent_and_old_v2_cannot_pass():
    kernel = kernel_fixture()
    brief = public_brief_draft(kernel)
    grounding = grounding_fixture(kernel)
    portability = portability_fixture(brief)
    mapping, stance = {"checks": []}, {"checks": []}
    calibration, regression = {"passed": True}, {"passed": True}
    record = {"schema_version": "transfer_kernel_record_v3",
              "source_intent_sha": "intent-sha", "source_media_sha": "media-sha",
              "kernel": kernel, "grounding_audit_sha": json_hash(grounding),
              "status": "private_grounded"}
    record["artifact_sha"] = json_hash(record)
    published = publish_brief(
        brief, kernel_sha=record["artifact_sha"], grounding=grounding,
        portability=portability, mapping=mapping, stance=stance,
        regression=regression)
    artifacts = {"record": record, "grounding": grounding,
                 "portability": portability, "calibration": calibration,
                 "mapping": mapping, "stance": stance,
                 "regression": regression}
    acceptance = make_acceptance(published, **artifacts)
    validate_acceptance(acceptance, published, **artifacts)
    old = dict(acceptance, schema_version="trial_acceptance_v2")
    with pytest.raises(ValueError, match="trial_acceptance_invalid"):
        validate_acceptance(old, published, **artifacts)
    changed = copy.deepcopy(mapping)
    changed["checks"].append({"new": True})
    with pytest.raises(ValueError, match="trial_acceptance_invalid"):
        validate_acceptance(acceptance, published, **(artifacts | {
            "mapping": changed}))


def test_prompts_have_no_reference_or_dev_answers():
    prompts = KERNEL_PROMPT + GROUNDING_PROMPT + PORTABILITY_PROMPT
    for marker in ("taekwondo", "chef", "dishwasher", "cursed", "C02",
                   "失去双手", "打破偏见"):
        assert marker not in prompts


def test_unverified_preview_cannot_issue_attestation(tmp_path, monkeypatch):
    kernel = kernel_fixture()
    draft = public_brief_draft(kernel)
    args = argparse.Namespace(output=tmp_path, reference=tmp_path / "ref.json",
                              video=tmp_path / "video.mp4")
    monkeypatch.setattr(trial, "_markers", lambda args: [])
    monkeypatch.setattr(trial, "_ask", lambda *args, **kwargs: (
        {"schema_version": "transfer_theme_batch_v3", "themes": [
            {"theme_id": f"T{i}", "domain": domain, "premise": "New story",
             "communicative_goal": "Goal", "audience_prior": "Prior",
             "evidence_mechanism": "Evidence", "audience_update": "Update",
             "tone": "Warm", "role_bindings": [{
                 "role_id": role_id, "new_binding": f"New {domain} role"}
                 for role_id in draft["free_role_ids"]]}
            for i, domain in enumerate(("museum", "garden", "music"), 1)]}
        if args[2] == "themes" else
        {"schema_version": "intent_theme_critique_v1", "checks": [
            {"theme_id": f"T{i}", "goal_match": True,
             "evidence_logic": True, "original": True, "filmable": True,
             "structure_preference": 2, "reason_codes": []}
            for i in (1, 2, 3)]}))
    monkeypatch.setattr(trial, "_model_call", lambda *args, **kwargs: {
        "schema_version": "intent_source_copy_audit_v1", "checks": [
            {"candidate_id": f"T{i}", "surface_copy": False,
             "reason": "Different surface"} for i in (1, 2, 3)]})
    report = trial._themes(args, object(), draft, kernel, verified=False,
                           failure_reasons=["contrast_regression_failed"])
    assert report["status"] == "unverified_preview"
    assert len(report["checks"]) == 3
    assert all(row["status"] == "unverified" for row in report["checks"])
    assert not (tmp_path / "trial_acceptance_v3.json").exists()
    assert not (tmp_path / "creative_story_brief_v3.json").exists()


def test_theme_requires_explicit_binding_for_each_public_role():
    brief = public_brief_draft(kernel_fixture())
    batch = {"schema_version": "transfer_theme_batch_v3", "themes": [
        {"theme_id": f"T{i}", "domain": domain, "premise": "New story",
         "communicative_goal": "Goal", "audience_prior": "Prior",
         "evidence_mechanism": "Evidence", "audience_update": "Update",
         "tone": "Warm", "role_bindings": [{"role_id": role_id,
             "new_binding": f"{domain} role"}
             for role_id in brief["free_role_ids"]]}
        for i, domain in enumerate(("museum", "garden", "music"), 1)]}
    assert validate_themes(batch, brief) == []
    batch["themes"][0]["role_bindings"].pop()
    assert validate_themes(batch, brief) == [
        "theme_fields_or_role_bindings_invalid"]


def test_labels_not_in_contrast_model_payload():
    root = Path(__file__).resolve().parents[1] / "config"
    cases, labels = trial._contrast_inputs(
        root / "transfer_contrast_v1_cases.json",
        root / "transfer_contrast_v1_labels.json")
    payload = [{"case_id": row["case_id"], "story": row["story"],
                "fields": public_fields(public_brief_draft(kernel_fixture()))}
               for row in cases]
    assert "expected_accept" not in json.dumps(payload)
    assert len(labels) == 4


@pytest.mark.parametrize("accept_c02", [True, False])
def test_one_bounded_run_publishes_only_after_regression(
        tmp_path, monkeypatch, accept_c02):
    root = Path(__file__).resolve().parents[1] / "config"
    args = argparse.Namespace(
        intent=tmp_path / "intent.json", reference=tmp_path / "reference.json",
        video=tmp_path / "video.mp4", output=tmp_path / "run",
        reuse_calibration=None, baseline_result=None, gpu_pair="0,1",
        config=None,
        calibration_cases=root / "transfer_judge_calibration_v2_cases.json",
        calibration_labels=root / "transfer_judge_calibration_v2_labels.json",
        cases=root / "transfer_contrast_v1_cases.json",
        labels=root / "transfer_contrast_v1_labels.json")
    args.intent.write_text(json.dumps({
        "story_candidate_ready": True, "artifact_sha": "intent-sha",
        "source_sha": "media-sha",
        "analysis": {"tone": "Warm", "limitations": []}}), encoding="utf-8")
    args.reference.write_text("{}", encoding="utf-8")
    args.video.write_bytes(b"media")
    reference = {"source_sha": "media-sha",
                 "claims": [{"claim_id": "CL_A"}],
                 "events": [{"event_id": "EV_B"}],
                 "shot_storyboard": {"shot_cards": []}}
    monkeypatch.setattr(trial, "load_reference", lambda path: reference)
    monkeypatch.setattr(trial, "sha256_file", lambda path: (
        "media-sha" if path == args.video else str(path)))
    monkeypatch.setattr(trial, "_runner", lambda args: object())
    monkeypatch.setattr(trial, "_reusable_calibration", lambda *args: (
        {"passed": True}, "fixture"))
    monkeypatch.setattr(trial, "_transfer_leaks", lambda *args: [])
    theme_args = []
    monkeypatch.setattr(trial, "_themes", lambda *args, **kwargs: (
        theme_args.append(kwargs) or {"status": (
            "model_checked_candidate" if kwargs["verified"] else
            "unverified_preview")}))

    def fake_model(runner, *, name, payload, **kwargs):
        if name == "kernel":
            return kernel_fixture()
        if name == "grounding_audit":
            return grounding_fixture(payload["kernel"])
        if name == "portability_audit":
            assert [row["path"] for row in payload["fields"]] == audit_paths(
                payload["public_brief"])
            return portability_fixture(payload["public_brief"])
        if name == "contrast_mapping":
            assert kwargs["max_new_tokens"] == 8192
            return {"schema_version": "transfer_mapping_eval_v2",
                    "checks": [{"case_id": case["case_id"],
                                "mappings": [{
                                    "path": field["path"],
                                    "brief_quote": field["text"],
                                    "candidate_quote": case["story"][:10]
                                      if case["case_id"] in (
                                          "C01", "C02") else None,
                                    "verdict": "mapped" if case["case_id"] in
                                      (("C01", "C02") if accept_c02 else
                                       ("C01",)) else "unmapped",
                                    "reason": "A complete field mapping"}
                                    for field in case["fields"]]}
                               for case in payload["cases"]]}
        assert name == "contrast_stance"
        return {"schema_version": "transfer_stance_eval_v2",
                "checks": [{"case_id": case["case_id"],
                            "stance_match": case["case_id"] in (
                                "C01", "C02"), "reason": "Stance checked"}
                           for case in payload["cases"]]}

    monkeypatch.setattr(trial, "_model_call", fake_model)
    result = trial.run(args)
    assert theme_args[0]["verified"] is accept_c02
    assert (args.output / "creative_story_brief_v3.json").exists() is accept_c02
    assert (args.output / "trial_acceptance_v3.json").exists() is accept_c02
    assert result["status"] == ("model_checked_candidate" if accept_c02 else
                                 "unverified_preview")
