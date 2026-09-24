"""The v5 trial separates source bindings from transferable relations."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts import run_reference_message_v5 as trial
from src.agentic_video.reference_message_v5 import (
    ABSTRACTION_PROMPT, private_input, public_brief, public_fields,
    validate_audit, validate_kernel, validate_theme_review, validate_themes,
)


IDS = {"S1_T02", "S2_V02", "S3_T06"}


def _kernel() -> dict:
    return {"schema_version": "message_kernel_v5",
            "source_claim": {"statement": "The work states a specific claim",
                             "support_ids": ["S1_T02"]},
            "roles": [
                {"role_id": "R1", "function": "an initial inference",
                 "source_binding": "private original setting",
                 "support_ids": ["S1_T02"]},
                {"role_id": "R2", "function": "observable information",
                 "source_binding": "private original action",
                 "support_ids": ["S2_V02"]}],
            "relations": [{"relation_id": "E1", "from_role": "R1",
                           "to_role": "R2", "relation": "is reconsidered after",
                           "scope": "one person's observed actions"}],
            "audience_goal": "Judge a person on more than one cue",
            "evidence_scope": "The observed case is not a universal proof",
            "free_slots": ["basis of the initial inference"],
            "optional_tone": None, "limitations": []}


def test_prompt_has_no_reference_answer_or_development_examples():
    for marker in ("C01", "C02", "older dishwasher", "chef", "taekwondo",
                   "lost both hands", "break prejudice", "disability"):
        assert marker not in ABSTRACTION_PROMPT.casefold()


def test_private_input_does_not_repeat_source_claim_as_takeaway():
    old = {"story_candidate_ready": True, "artifact_sha": "sha",
           "analysis": {"communicative_goal": {
               "statement": "specific statement", "support_ids": ["S1_T02"]},
               "audience_prior": {"interpretation": "prior",
                                  "support_ids": ["S1_T02"]},
               "evidence_mechanism": {"description": "evidence",
                                      "support_ids": ["S2_V02"]},
               "audience_update": {"interpretation": "update",
                                   "support_ids": ["S2_V02"]},
               "ending": {"effect": "ending", "support_ids": ["S3_T06"]},
               "tone": "warm", "limitations": "no formal result"}}
    normalized = private_input(old, old=True)
    assert normalized["schema_version"] == "private_message_input_v5"
    assert normalized["source_statement"]["text"] == "specific statement"
    assert "takeaway_candidate" not in normalized
    assert "communicative_goal" not in normalized


def test_public_brief_excludes_private_bindings_and_support_ids():
    kernel = _kernel()
    assert validate_kernel(kernel, IDS) == []
    brief = public_brief(kernel)
    assert brief["schema_version"] == "creative_message_brief_v5"
    assert "private original" not in str(brief)
    assert "support_ids" not in str(brief)
    assert "specific claim" not in str(brief)
    fields = public_fields(brief, include_optional=False)
    assert fields[0]["path"] == "audience_goal"
    assert fields[3]["text"].startswith("an initial inference")
    assert "private original" not in str(fields)


def test_kernel_requires_resolvable_relation_and_source_support():
    kernel = _kernel()
    kernel["relations"][0]["to_role"] = "unknown"
    assert "relations_invalid" in validate_kernel(kernel, IDS)
    kernel = _kernel()
    kernel["roles"][0]["support_ids"] = ["invented"]
    assert "roles_invalid" in validate_kernel(kernel, IDS)
    kernel = _kernel()
    kernel["relations"] = []
    assert "relations_invalid" in validate_kernel(kernel, IDS)


def test_audit_requires_exact_paths_and_all_good_verdicts():
    fields = [{"path": "audience_goal", "text": "goal"},
              {"path": "roles.R1.function", "text": "role"}]
    audit = {"schema_version": "message_portability_audit_v5",
             "checks": [{"path": row["path"], "verdict": "portable",
                         "reason": "not source-bound"} for row in fields]}
    assert validate_audit(audit, fields, kind="portability") == []
    audit["checks"][1]["verdict"] = "source_domain_required"
    assert validate_audit(audit, fields, kind="portability") == [
        "roles.R1.function"]
    audit["checks"] = audit["checks"][:1]
    assert validate_audit(audit, fields, kind="portability") == [
        "portability_audit_invalid"]


def test_frozen_self_assessment_is_not_a_deterministic_v5_gate(
        tmp_path: Path, monkeypatch):
    reference = {"source_sha": "media", "claims": [{"claim_id": "S1_T02"}],
                 "events": []}
    frozen = tmp_path / "frozen"
    kernels = ({"source_claim": {"statement": "private source",
                                 "support_ids": ["S1_T02"]},
                "private_substitutions": [{"literal_fit": False}]},) * 2
    monkeypatch.setattr(trial, "_inputs", lambda _: (reference, {}, {}))
    monkeypatch.setattr(trial, "_new_record", lambda *args: {})
    monkeypatch.setattr(trial, "_frozen_inputs", lambda *args: kernels)
    monkeypatch.setattr(trial, "_lineage", lambda *args: {})
    monkeypatch.setattr(trial, "_runner", lambda _: object())
    monkeypatch.setattr(trial, "_model_config_check", lambda *args: "sha")
    monkeypatch.setattr(trial, "v4_public_brief", lambda _: {
        "audience_takeaway": "source-bound"})
    monkeypatch.setattr(trial, "v4_public_fields", lambda *args, **kwargs: [
        {"path": "audience_takeaway", "text": "source-bound"}])
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(kwargs)
        return {"schema_version": "message_portability_audit_v4",
                "checks": [{"path": "audience_takeaway",
                            "verdict": "source_domain_required",
                            "reason": "the source setting is required"}]}

    monkeypatch.setattr(trial, "_model_call", fake_call)
    args = Namespace(output=tmp_path / "diagnosis", frozen_compare=frozen)
    report = trial.diagnose(args)
    assert report["status"] == "diagnosis_passed"
    assert len(calls) == 2
    assert all("media" not in call for call in calls)


def test_theme_schemas_require_six_ordered_checks():
    themes = [{"theme_id": f"T{i}", "domain": "work",
               "judgment_basis": f"basis {i}", "premise": "filmable",
               "initial_cue": "one cue", "mistaken_inference": "too broad",
               "observable_evidence": "action", "audience_update": "update"}
              for i in range(1, 7)]
    batch = {"schema_version": "message_theme_batch_v5", "themes": themes}
    assert validate_themes(batch) == []
    checks = [{"theme_id": f"T{i}", "message_match": True,
               "evidence_logic": True, "filmable": True,
               "surface_copy": False, "source_domain_overlap": False,
               "basis_category": f"category {i}", "reason": "sound"}
              for i in range(1, 7)]
    review = {"schema_version": "message_theme_review_v5", "checks": checks}
    assert validate_theme_review(review) == []
    checks[0]["theme_id"] = "T2"
    assert validate_theme_review(review) == ["theme_review_invalid"]


def test_trial_never_runs_themes_if_primary_branch_fails(tmp_path: Path,
                                                        monkeypatch):
    reference = {"source_sha": "media", "claims": [], "events": []}
    monkeypatch.setattr(trial, "_inputs", lambda _: (reference, {}, {}))
    monkeypatch.setattr(trial, "_new_record", lambda *args: {
        "reading": {}})
    monkeypatch.setattr(trial, "_frozen_inputs", lambda *args: ({}, {}))
    monkeypatch.setattr(trial, "_lineage", lambda *args: {})
    monkeypatch.setattr(trial, "_contrast_inputs", lambda *args: ([], {}))
    monkeypatch.setattr(trial, "_calibration_inputs", lambda *args: ([], {}))
    monkeypatch.setattr(trial, "_runner", lambda _: object())
    monkeypatch.setattr(trial, "_model_config_check", lambda *args: "sha")
    monkeypatch.setattr(trial, "_reusable_calibration", lambda *args: (
        {"passed": True}, "reused"))
    monkeypatch.setattr(trial, "private_input", lambda *args, **kwargs: {})
    monkeypatch.setattr(trial, "_assess", lambda *args, **kwargs: {
        "status": "blocked", "reason": "portability_failed"})
    monkeypatch.setattr(trial, "_themes", lambda *args: (_ for _ in ()
        ).throw(AssertionError("themes must not run")))
    diagnosis = tmp_path / "diagnosis"
    trial._write_json(diagnosis / "result.json", {
        "status": "diagnosis_passed"})
    trial._write_json(diagnosis / "input_lineage.json", {})
    (tmp_path / "cases").write_text("{}", encoding="utf-8")
    (tmp_path / "labels").write_text("{}", encoding="utf-8")
    args = Namespace(output=tmp_path / "trial", diagnosis=diagnosis,
                     cases=tmp_path / "cases", labels=tmp_path / "labels",
                     calibration_cases=tmp_path / "cases",
                     calibration_labels=tmp_path / "labels")
    report = trial.trial(args)
    assert report["status"] == "blocked"
    assert report["theme_generation"] == "not_run"
