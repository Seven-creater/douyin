"""The v4 trial keeps source facts private and stops before bad themes."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts import run_reference_message_v4 as trial
from src.agentic_video.manifest import json_hash
from src.agentic_video.reference_message_v4 import (
    ABSTRACTION_PROMPT, AUDIENCE_PROMPT, artifact_with_sha,
    old_message_input, public_brief, public_fields, validate_audit,
    validate_brief, validate_kernel, validate_message, validate_themes,
)


IDS = {"S1_T02", "S2_V02", "S3_T06"}


def _row(text: str = "An attributed observation") -> dict:
    return {"text": text, "support_ids": ["S1_T02"]}


def _reading() -> dict:
    return {"schema_version": "reference_message_v4",
            "source_statement": _row(), "audience_before": _row(),
            "evidence_path": _row(), "audience_after": _row(),
            "takeaway_candidate": _row(), "ending": _row(),
            "limitations": []}


def _kernel() -> dict:
    statement = {"statement": "A person is not defined by one cue",
                 "support_ids": ["S1_T02"]}
    return {"schema_version": "message_kernel_v4",
            **{key: dict(statement) for key in (
                "source_claim", "audience_takeaway",
                "misjudgment_mechanism", "corrective_evidence_role",
                "revised_judgment")},
            "optional_sequence": ["Information arrives later"],
            "free_dimensions": ["Basis of an initial judgment"],
            "private_substitutions": [
                {"context": f"Context {index}",
                 "initial_cue": f"Cue {index}",
                 "observable_evidence": f"Action {index}",
                 "literal_fit": True, "reason": "The full sentence applies"}
                for index in range(3)], "limitations": []}


def test_prompts_do_not_contain_known_answers_or_cases():
    for prompt in (AUDIENCE_PROMPT, ABSTRACTION_PROMPT):
        for marker in ("C01", "C02", "chef", "taekwondo", "lost both hands",
                       "break prejudice", "older dishwasher", "disability"):
            assert marker not in prompt.casefold()


def test_old_reading_is_normalized_to_same_input_shape():
    analysis = {"communicative_goal": {"statement": "Source claim",
                                      "support_ids": ["S1_T02"]},
                "audience_prior": {"interpretation": "Prior",
                                   "support_ids": ["S1_T02"]},
                "evidence_mechanism": {"description": "Evidence",
                                       "support_ids": ["S2_V02"]},
                "audience_update": {"interpretation": "Update",
                                    "support_ids": ["S2_V02"]},
                "ending": {"effect": "Ending",
                           "support_ids": ["S3_T06"]},
                "tone": "warm", "limitations": "unknown result"}
    old = old_message_input({"story_candidate_ready": True,
                             "artifact_sha": "sha", "analysis": analysis})
    assert old["schema_version"] == "private_message_input_v4"
    assert old["source_statement"]["text"] == "Source claim"
    assert old["limitations"] == ["unknown result"]
    assert analysis["tone"] == "warm"


def test_private_substitutions_and_source_claim_never_enter_public_brief():
    kernel = _kernel()
    kernel["source_claim"]["statement"] = "private source detail"
    kernel["private_substitutions"][0]["context"] = "private test context"
    assert validate_kernel(kernel, IDS) == []
    brief = public_brief(kernel)
    assert validate_brief(brief) == []
    assert "private source detail" not in str(brief)
    assert "private test context" not in str(brief)
    assert "source_claim" not in brief
    assert "parent_kernel_sha" not in brief
    assert [row["path"] for row in public_fields(
        brief, include_optional=False)] == [
            "audience_takeaway", "misjudgment_mechanism",
            "corrective_evidence_role", "revised_judgment"]


def test_failed_private_substitution_blocks_kernel():
    kernel = _kernel()
    kernel["private_substitutions"][1]["literal_fit"] = False
    assert "private_substitution_failed" in validate_kernel(kernel, IDS)


def test_audit_requires_full_field_coverage_and_all_supported():
    fields = [{"path": "audience_takeaway", "text": "x"},
              {"path": "optional_sequence.0", "text": "y"}]
    audit = {"schema_version": "message_portability_audit_v4",
             "checks": [{"path": row["path"], "verdict": "portable",
                         "reason": "Applies elsewhere"} for row in fields]}
    assert validate_audit(audit, fields, kind="portability")
    audit["checks"][1]["verdict"] = "uncertain"
    assert not validate_audit(audit, fields, kind="portability")
    audit["checks"] = audit["checks"][:1]
    assert not validate_audit(audit, fields, kind="portability")


def test_themes_require_six_items_and_distinct_judgment_cues():
    themes = [{"theme_id": f"T{i}", "domain": f"Domain {i}",
               "premise": "A filmable event", "initial_cue": f"Cue {i}",
               "mistaken_inference": "An overly broad judgment",
               "observable_evidence": "A visible action",
               "audience_update": "A revised judgment"}
              for i in range(1, 7)]
    assert validate_themes({"schema_version": "message_theme_batch_v4",
                            "themes": themes}) == []
    themes[2]["initial_cue"] = "Cue 1"
    for row in themes[3:]:
        row["initial_cue"] = "Cue 1"
    assert "theme_cues_not_distinct" in validate_themes({
        "schema_version": "message_theme_batch_v4", "themes": themes})
    for index, row in enumerate(themes):
        row["initial_cue"] = f"Cue {index}"
        row["domain"] = "Same domain"
    assert "theme_domains_not_distinct" in validate_themes({
        "schema_version": "message_theme_batch_v4", "themes": themes})


def test_extract_makes_exactly_one_av_call_without_old_intent_in_request(
        tmp_path: Path, monkeypatch):
    reference = {"source_sha": "media", "artifact_sha": "ref",
                 "claims": [{"claim_id": "S1_T02"}], "events": []}
    static = {"duration_s": 10, "section_bundles": [],
              "audio_transcript_candidate": None}
    monkeypatch.setattr(trial, "_inputs", lambda _: (reference, static, {}))
    monkeypatch.setattr(trial, "_lineage", lambda *args: {})
    monkeypatch.setattr(trial, "_runner", lambda _: object())
    calls = []

    def fake_call(_runner, **kwargs):
        calls.append(kwargs)
        return _reading()

    monkeypatch.setattr(trial, "_model_call", fake_call)
    static_path = tmp_path / "static.json"
    static_path.write_text("{}", encoding="utf-8")
    old_intent = tmp_path / "old" / "reference_intent_v1.json"
    baseline_meta = (old_intent.parent / "calls" / "intent_initial" /
                     "model_call.json")
    trial._write_json(baseline_meta, {"model_config_sha": json_hash({})})
    args = Namespace(output=tmp_path / "run", video=tmp_path / "video.mp4",
                     static=static_path, old_intent=old_intent)
    result = trial.extract(args)
    assert result["status"] == "reading_candidate"
    assert len(calls) == 1
    assert calls[0]["channel"] == "AV" and calls[0]["fps"] == 1.5
    assert "private_intent" not in calls[0]["payload"]


def test_compare_stops_before_themes_when_both_briefs_fail(
        tmp_path: Path, monkeypatch):
    static_path = tmp_path / "static.json"
    static_path.write_text("{}", encoding="utf-8")
    reading = _reading()
    record = artifact_with_sha({
        "schema_version": "reference_message_record_v4",
        "source_sha": "media", "static_sha": trial.sha256_file(static_path),
        "model_config_sha": json_hash({}),
        "reading": reading, "status": "private_candidate"})
    record_path = tmp_path / "new.json"
    trial._write_json(record_path, record)
    reference = {"source_sha": "media", "claims": [{"claim_id": "S1_T02"}],
                 "events": []}
    monkeypatch.setattr(trial, "_inputs", lambda _: (reference, {}, {}))
    monkeypatch.setattr(trial, "_lineage", lambda *args: {})
    monkeypatch.setattr(trial, "_runner", lambda _: object())
    monkeypatch.setattr(trial, "_baseline_config_sha", lambda _: json_hash({}))
    monkeypatch.setattr(trial, "_calibration_inputs", lambda *args: ([], {}))
    monkeypatch.setattr(trial, "_contrast_inputs", lambda *args: ([], {}))
    monkeypatch.setattr(trial, "_reusable_calibration", lambda *args: (
        {"passed": True}, "reused"))
    monkeypatch.setattr(trial, "old_message_input", lambda _: {})
    monkeypatch.setattr(trial, "_assess_branch", lambda *args: {
        "status": "blocked", "reason": "portability_failed"})
    monkeypatch.setattr(trial, "_theme_trial", lambda *args: (_ for _ in ()
        ).throw(AssertionError("theme call should not run")))
    args = Namespace(output=tmp_path / "compare", static=static_path,
                     new_message=record_path, cases=static_path,
                     labels=static_path, calibration_cases=static_path,
                     calibration_labels=static_path)
    result = trial.compare(args)
    assert result["status"] == "blocked"
    assert result["theme_generation"] == "not_run"
    assert not (args.output / "diagnostic_themes.json").exists()


def test_unknown_message_can_be_recorded_but_not_promoted():
    reading = _reading()
    reading["takeaway_candidate"]["text"] = "unknown"
    assert validate_message(reading, IDS) == []
    assert reading["takeaway_candidate"]["text"].casefold() == "unknown"


def test_extract_preserves_unknown_reading_without_promoting_it(
        tmp_path: Path, monkeypatch):
    static = {"duration_s": 10, "section_bundles": [],
              "audio_transcript_candidate": None}
    reference = {"source_sha": "media", "claims": [{"claim_id": "S1_T02"}],
                 "events": []}
    old = tmp_path / "old" / "reference_intent_v1.json"
    trial._write_json(old.parent / "calls" / "intent_initial" /
                      "model_call.json", {"model_config_sha": json_hash({})})
    static_path = tmp_path / "static.json"
    static_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(trial, "_inputs", lambda _: (reference, static, {}))
    monkeypatch.setattr(trial, "_lineage", lambda *args: {})
    monkeypatch.setattr(trial, "_runner", lambda _: object())
    reading = _reading()
    reading["takeaway_candidate"]["text"] = "unknown"
    monkeypatch.setattr(trial, "_model_call", lambda *args, **kwargs: reading)
    args = Namespace(output=tmp_path / "run", video=tmp_path / "video.mp4",
                     static=static_path, old_intent=old)
    result = trial.extract(args)
    assert result["status"] == "blocked"
    assert result["issues"] == ["message_unknown"]
    assert trial._read(args.output / "reference_message_v4.json")[
        "status"] == "unknown"


def test_theme_request_contains_only_public_brief(tmp_path: Path,
                                                  monkeypatch):
    brief = public_brief(_kernel())
    brief["parent_kernel_sha"] = "private_sha"
    brief["artifact_sha"] = "artifact_sha"
    captured = []

    def fake_call(_runner, **kwargs):
        captured.append(kwargs)
        return {"schema_version": "message_theme_batch_v4", "themes": []}

    monkeypatch.setattr(trial, "_model_call", fake_call)
    monkeypatch.setattr(trial, "_markers", lambda _: [])
    args = Namespace(output=tmp_path)
    report = trial._theme_trial(args, object(), {}, brief)
    assert report["status"] == "themes_blocked"
    assert len(captured) == 1
    request = captured[0]["payload"]["creative_message_brief"]
    assert "parent_kernel_sha" not in request
    assert "artifact_sha" not in request
    assert "private_substitutions" not in request
