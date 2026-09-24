from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from scripts import run_transfer_kernel_v1 as trial
from scripts import run_intent_story_trial as creative_trial
from src.agentic_video.creative_pipeline.intent_trial import (
    prompts_for_brief, validate_source_copy_audit,
)
from src.agentic_video.manifest import json_hash
from src.agentic_video.transfer_kernel_v1 import (
    CONTRAST_PROMPT, KERNEL_PROMPT, audit_feedback_codes,
    kernel_audit_passes, normalize_private_intent, public_brief_draft,
    publish_brief, score_contrast, validate_contrast_eval, validate_kernel,
    validate_kernel_audit, validate_public_brief,
)


def kernel_fixture():
    return {"schema_version": "transfer_kernel_v1",
            "stance": {"statement": "Assess people through fuller evidence",
                       "support_ids": ["CL_A"]},
            "roles": [
                {"role_id": "V1", "invariant_function": "Initial narrow view",
                 "source_binding": "private source detail A",
                 "rebindable_dimension": "source of first impression",
                 "support_ids": ["CL_A"]},
                {"role_id": "V2", "invariant_function": "New action evidence",
                 "source_binding": "private source detail B",
                 "rebindable_dimension": "demonstrated activity",
                 "support_ids": ["EV_B"]}],
            "relations": [{"relation_id": "R1", "from_role": "V1",
                           "to_role": "V2", "relation": "reassessed_after",
                           "support_ids": ["CL_A", "EV_B"]}],
            "optional_order": ["V1", "V2"], "uncertainties": []}


def audit_fixture(kernel=None):
    kernel = kernel or kernel_fixture()
    ids = ["stance"] + [row["role_id"] for row in kernel["roles"]] + [
        row["relation_id"] for row in kernel["relations"]]
    return {"schema_version": "transfer_kernel_audit_v1",
            "checks": [{"id": item, "verdict": "supported",
                        "reason": "Cited meaning is supported",
                        "issue_code": None, "field_path": None}
                       for item in ids], "binding_leaks": []}


def intent_fixture():
    return {"story_candidate_ready": True, "artifact_sha": "intent-sha",
            "source_sha": "media-sha",
            "analysis": {"tone": "warm", "limitations": "Some context is absent"}}


def test_normalizes_optional_fields_without_editing_source():
    parent = intent_fixture()
    before = copy.deepcopy(parent)
    normalized = normalize_private_intent(parent)
    assert parent == before
    assert normalized["analysis"]["tone"] == {"description": "warm"}
    assert normalized["analysis"]["limitations"] == [
        "Some context is absent"]
    assert normalized["normalization_originals"] == {
        "tone": "warm", "limitations": "Some context is absent"}


def test_kernel_schema_refs_and_non_reversal_reference():
    kernel = kernel_fixture()
    assert validate_kernel(kernel, {"CL_A", "EV_B"}) == []
    kernel["relations"] = []
    kernel["optional_order"] = []
    assert validate_kernel(kernel, {"CL_A", "EV_B"}) == []
    assert validate_kernel_audit(audit_fixture(kernel), kernel) == []
    kernel["roles"][0]["support_ids"] = ["NOT_ACCEPTED"]
    assert "role_invalid" in validate_kernel(kernel, {"CL_A", "EV_B"})


def test_audit_requires_coverage_reasons_and_generic_feedback():
    kernel = kernel_fixture()
    audit = audit_fixture()
    assert validate_kernel_audit(audit, kernel) == []
    assert kernel_audit_passes(audit)
    audit["checks"][-1].update({"verdict": "insufficient",
                                "reason": "Private phrase that must not echo",
                                "issue_code": "relation_unsupported",
                                "field_path": "relations.R1"})
    audit["binding_leaks"] = [{"field_path": "roles.V1.invariant_function",
                                "reason": "Another private phrase"}]
    assert not kernel_audit_passes(audit)
    assert "Private phrase" not in str(audit_feedback_codes(audit))
    audit["checks"][-1]["issue_code"] = None
    assert "kernel_audit_check_invalid" in validate_kernel_audit(audit, kernel)


def test_public_brief_is_whitelisted_and_strict():
    kernel, audit = kernel_fixture(), audit_fixture()
    draft = public_brief_draft(kernel)
    assert "private source detail" not in json.dumps(draft)
    assert "support_ids" not in json.dumps(draft)
    assert "source_binding" not in json.dumps(draft)
    published = publish_brief(draft, kernel_sha="kernel-sha", audit=audit,
                              evaluation={"checks": []},
                              contrast_score={"passed": True})
    validate_public_brief(published, published=True)
    prompts_for_brief(published)
    published["roles"][0]["source_binding"] = "leak"
    with pytest.raises(ValueError, match="public_roles_invalid"):
        validate_public_brief(published, published=True)


def test_contrast_labels_stay_out_of_model_cases():
    root = Path(__file__).resolve().parents[1] / "config"
    cases, labels = trial._contrast_inputs(
        root / "transfer_contrast_v1_cases.json",
        root / "transfer_contrast_v1_labels.json")
    assert set(labels.values()) == {True, False}
    assert "expected_accept" not in json.dumps(cases)
    checks = [{"case_id": row["case_id"], "stance_match": labels[
        row["case_id"]], "evidence_logic": labels[row["case_id"]],
               "surface_copy": False, "soft_order_similarity": 1,
               "reason": "Compared relation and position"} for row in cases]
    evaluation = {"schema_version": "transfer_contrast_eval_v1",
                  "checks": checks}
    assert validate_contrast_eval(evaluation, list(labels)) == []
    assert score_contrast(evaluation, labels)["passed"]
    checks[0]["stance_match"] = not checks[0]["stance_match"]
    assert not score_contrast(evaluation, labels)["passed"]


def test_source_copy_audit_coverage_and_nonempty_reason():
    audit = {"schema_version": "intent_source_copy_audit_v1",
             "checks": [{"candidate_id": "T1", "surface_copy": False,
                         "reason": "Only relational similarity"}]}
    assert validate_source_copy_audit(audit, ["T1"]) == []
    audit["checks"][0]["reason"] = ""
    assert validate_source_copy_audit(audit, ["T1"]) == [
        "source_copy_audit_invalid"]


def test_kernel_run_publishes_only_after_four_contrast_matches(
        tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "config"
    args = argparse.Namespace(intent=tmp_path / "intent.json",
                              reference=tmp_path / "reference.json",
                              video=tmp_path / "video.mp4",
                              cases=root / "transfer_contrast_v1_cases.json",
                              labels=root / "transfer_contrast_v1_labels.json",
                              output=tmp_path / "run", gpu_pair="0,1",
                              config=None)
    args.intent.write_text(json.dumps(intent_fixture()), encoding="utf-8")
    args.video.write_bytes(b"test media")
    args.reference.write_text("{}", encoding="utf-8")
    reference = {"source_sha": "media-sha",
                 "claims": [{"claim_id": "CL_A"}],
                 "events": [{"event_id": "EV_B"}]}
    monkeypatch.setattr(trial, "load_reference", lambda path: reference)
    monkeypatch.setattr(trial, "sha256_file", lambda path: (
        "media-sha" if path == args.video else "other-sha"))
    monkeypatch.setattr(trial, "_runner", lambda args: object())
    monkeypatch.setattr(trial, "_transfer_leaks", lambda *args: [])
    calls = []

    def fake_call(runner, *, name, prompt, payload, output, **kwargs):
        calls.append((name, payload))
        if name == "attempt_01_kernel":
            return kernel_fixture()
        if name == "attempt_01_audit":
            return audit_fixture()
        assert name == "contrast_evaluation"
        assert "expected_accept" not in json.dumps(payload)
        labels = {"C01": True, "C02": True, "C03": False, "C04": False}
        return {"schema_version": "transfer_contrast_eval_v1",
                "checks": [{"case_id": row["case_id"],
                            "stance_match": labels[row["case_id"]],
                            "evidence_logic": labels[row["case_id"]],
                            "surface_copy": False,
                            "soft_order_similarity": 1,
                            "reason": "Relation checked"}
                           for row in payload["cases"]]}

    monkeypatch.setattr(trial, "_model_call", fake_call)
    result = trial.run(args)
    assert result["status"] == "model_checked_candidate"
    assert len(calls) == 3
    assert (args.output / "creative_story_brief_v2.json").exists()
    assert "source_binding" not in (args.output /
        "creative_story_brief_v2.json").read_text(encoding="utf-8")
    assert "expected_accept" not in json.dumps(calls[0][1])


def test_no_case_answers_in_extraction_prompts():
    combined = KERNEL_PROMPT + CONTRAST_PROMPT
    for marker in ("elderly", "dishwasher", "taekwondo", "cursed",
                   "C01", "打破偏见"):
        assert marker not in combined


def test_v2_theme_story_trial_keeps_private_bindings_out_of_writer(
        tmp_path, monkeypatch):
    kernel, audit = kernel_fixture(), audit_fixture()
    record = {"schema_version": "transfer_kernel_record_v1",
              "source_media_sha": "media-sha", "source_intent_sha": "intent-sha",
              "kernel": kernel, "audit": audit, "status": "private_model_checked"}
    record["artifact_sha"] = json_hash(record)
    brief = publish_brief(public_brief_draft(kernel),
                          kernel_sha=record["artifact_sha"], audit=audit,
                          evaluation={"checks": []},
                          contrast_score={"passed": True})
    brief_path, kernel_path = tmp_path / "brief.json", tmp_path / "kernel.json"
    brief_path.write_text(json.dumps(brief), encoding="utf-8")
    kernel_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(creative_trial, "load_reference", lambda path: {
        "source_sha": "media-sha"})
    monkeypatch.setattr(creative_trial, "_markers", lambda args: [])
    monkeypatch.setattr(creative_trial, "_runner", lambda args: object())
    monkeypatch.setattr(creative_trial, "_trial_acceptance",
                        lambda args, brief, record: {
                            "artifact_sha": "accepted-trial-sha"})
    writer_payloads = []

    def fake_ask(runner, root, stage, prompt, payload, max_tokens):
        writer_payloads.append(payload)
        assert "source_binding" not in json.dumps(payload)
        if stage == "01_themes":
            return {"schema_version": "intent_theme_batch_v1", "themes": [
                {"theme_id": f"T{i}", "domain": domain,
                 "premise": "A concrete new story", "communicative_goal": "Goal",
                 "audience_prior": "Prior", "evidence_mechanism": "Evidence",
                 "audience_update": "Update", "tone": "Warm"}
                for i, domain in enumerate(("food", "school", "work"), 1)]}
        if stage == "02_theme_critique":
            return {"schema_version": "intent_theme_critique_v1",
                    "checks": [{"theme_id": f"T{i}", "goal_match": True,
                                "evidence_logic": True, "original": True,
                                "filmable": True,
                                "structure_preference": i,
                                "reason_codes": []} for i in (1, 2, 3)]}
        if stage == "story":
            theme_id = payload["selected_theme"]["theme_id"]
            return {"schema_version": "intent_story_synopsis_v1",
                    "theme_id": theme_id, "logline": "New story",
                    "events": [{"order": 1, "action": "Act",
                                "new_information": "New evidence"}],
                    "ending": "Open", "tone": "Warm"}
        assert stage == "story_critique"
        return {"schema_version": "intent_story_critique_v1",
                "checks": [{"theme_id": row["theme_id"],
                            "goal_match": True, "evidence_logic": True,
                            "original": True, "filmable": True,
                            "structure_preference": 1, "reason_codes": []}
                           for row in payload["stories"]]}

    def fake_private_call(runner, *, payload, **kwargs):
        assert "private_source_bindings" in payload
        return {"schema_version": "intent_source_copy_audit_v1",
                "checks": [{"candidate_id": row["candidate_id"],
                            "surface_copy": row["candidate_id"] == "T3",
                            "reason": "Private source comparison"}
                           for row in payload["candidates"]]}

    monkeypatch.setattr(creative_trial, "_ask", fake_ask)
    monkeypatch.setattr(creative_trial, "_model_call", fake_private_call)
    args = argparse.Namespace(brief=brief_path, private_kernel=kernel_path,
                              reference=tmp_path / "reference.json",
                              video=tmp_path / "video.mp4", gpu_pair="0,1",
                              config=None, output=tmp_path / "themes")
    selection = creative_trial.themes(args)
    assert selection["selected_theme_ids"] == ["T2", "T1"]
    selection_path = args.output / "selection.json"
    story_dirs = []
    for theme_id in selection["selected_theme_ids"]:
        args.selection = selection_path
        args.theme_id = theme_id
        args.output = tmp_path / f"story_{theme_id}"
        assert creative_trial.story(args)["status"] == "candidate"
        story_dirs.append(args.output)
    args.story_dirs = story_dirs
    args.output = tmp_path / "review"
    result = creative_trial.review(args)
    assert result["status"] == "model_checked_candidate"
    assert result["screenplay_generation"] == "not_run"
    assert len(writer_payloads) == 5
