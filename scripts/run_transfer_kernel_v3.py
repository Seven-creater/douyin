"""One bounded, text-only far-domain abstraction and theme trial."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_intent_story_trial import _markers
from scripts.run_transfer_kernel_v1 import (
    _contrast_inputs, _evidence_view, _fresh, _read, _runner,
)
from scripts.run_transfer_kernel_v2 import _calibrate, _calibration_inputs
from src.agentic_video.creative_pipeline.intent_trial import (
    SOURCE_COPY_PROMPT, has_reference_surface, select_themes,
    validate_source_copy_audit,
)
from src.agentic_video.creative_pipeline.real_text_baseline import _ask
from src.agentic_video.manifest import json_hash, sha256_file
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)
from src.agentic_video.transfer_eval_v2 import (
    MAPPING_PROMPT, STANCE_PROMPT, decisions, score, validate_mapping,
    validate_stance,
)
from src.agentic_video.transfer_kernel_v1 import normalize_private_intent
from src.agentic_video.transfer_kernel_v3 import (
    GROUNDING_PROMPT, KERNEL_PROMPT, PORTABILITY_PROMPT,
    THEME_CRITIC_PROMPT, THEME_PROMPT, make_acceptance,
    public_brief_draft, public_fields, publish_brief, validate_grounding,
    validate_kernel, validate_portability, validate_themes,
)


def _reusable_calibration(args, runner, cases: list[dict], labels: dict
                          ) -> tuple[dict, str]:
    previous = args.reuse_calibration
    if previous is not None and (previous / "calibration_report.json").exists():
        lineage = _read(previous / "input_lineage.json")
        report = _read(previous / "calibration_report.json")
        expected = (("calibration_cases_sha", args.calibration_cases),
                    ("calibration_labels_sha", args.calibration_labels))
        same_inputs = all(lineage.get(key) == sha256_file(path)
                          for key, path in expected)
        same_calls = all(
            _read(previous / "calls" / f"calibration_{turn:02d}_{kind}" /
                  "model_call.json").get("prompt_sha") == json_hash(prompt)
            and _read(previous / "calls" / f"calibration_{turn:02d}_{kind}" /
                      "model_call.json").get("model_config_sha") ==
                json_hash(getattr(runner, "cfg", {}) or {})
            for turn in (1, 2)
            for kind, prompt in (("mapping", MAPPING_PROMPT),
                                 ("stance", STANCE_PROMPT)))
        observed = [
            {row["case_id"]: row["observed"]
             for row in round_["case_results"]}
            for round_ in report.get("rounds", [])]
        if (same_inputs and same_calls and report.get("passed") is True and
                len(observed) == 2 and all(row == labels for row in observed)):
            return report, str(previous)
    return _calibrate(runner, args.output, cases, labels), "new_model_calls"


def _judge(runner, output: Path, brief: dict, cases: list[dict]
           ) -> tuple[dict, dict]:
    mapped_cases = [{"case_id": row["case_id"],
                     "fields": public_fields(brief), "story": row["story"]}
                    for row in cases]
    mapping = _model_call(runner, name="contrast_mapping",
                          prompt=MAPPING_PROMPT,
                          payload={"cases": mapped_cases}, output=output,
                          max_new_tokens=4096)
    _write_json(output / "contrast_mapping.json", mapping)
    issues = validate_mapping(mapping, mapped_cases)
    if issues:
        raise ValueError(",".join(issues))
    stance_cases = [{"case_id": row["case_id"], "stance": brief["goal"],
                     "story": row["story"]} for row in cases]
    stance = _model_call(runner, name="contrast_stance",
                         prompt=STANCE_PROMPT,
                         payload={"cases": stance_cases}, output=output,
                         max_new_tokens=1536)
    _write_json(output / "contrast_stance.json", stance)
    issues = validate_stance(stance, mapped_cases)
    if issues:
        raise ValueError(",".join(issues))
    return mapping, stance


def _themes(args, runner, brief: dict, kernel: dict, *, verified: bool,
            failure_reasons: list[str]) -> dict:
    """Preview has no credential and never becomes a production artifact."""
    payload = {"creative_story_brief": brief}
    batch = _ask(runner, args.output, "themes", THEME_PROMPT,
                 payload, 3072)
    _write_json(args.output / "theme_batch.json", batch)
    issues = validate_themes(batch, brief)
    if has_reference_surface(batch, _markers(args)):
        issues.append("theme_source_surface_leak")
    if issues:
        report = {"status": "themes_blocked", "reasons": issues,
                  "themes_generated": len(batch.get("themes") or []),
                  "production_release_allowed": False}
        _write_json(args.output / "theme_trial_report.json", report)
        return report
    critique = _ask(runner, args.output, "theme_critique",
                    THEME_CRITIC_PROMPT,
                    {"creative_story_brief": brief,
                     "theme_batch": batch}, 1536)
    _write_json(args.output / "theme_critique.json", critique)
    select_themes(critique)
    private_bindings = [{"role_id": row["unit_id"],
                         "source_binding": row["source_binding"]}
                        for row in kernel["units"]]
    copy_audit = _model_call(runner, name="source_copy_audit",
                             prompt=SOURCE_COPY_PROMPT,
                             payload={"private_source_bindings":
                                      private_bindings,
                                      "candidates": [{"candidate_id": row[
                                          "theme_id"], "candidate": row}
                                          for row in batch["themes"]]},
                             output=args.output, max_new_tokens=1024)
    _write_json(args.output / "source_copy_audit.json", copy_audit)
    issues = validate_source_copy_audit(copy_audit, ["T1", "T2", "T3"])
    if issues:
        raise ValueError(",".join(issues))
    copied = {row["candidate_id"] for row in copy_audit["checks"]
              if row["surface_copy"]}
    checks = []
    for check in critique["checks"]:
        hard_pass = all(check.get(key) is True for key in (
            "goal_match", "evidence_logic", "original", "filmable"))
        copied_from_source = check["theme_id"] in copied
        checks.append({"theme_id": check["theme_id"],
                       "status": "unverified" if not verified else (
                           "model_checked_candidate" if hard_pass and
                           not copied_from_source else "rejected"),
                       "source_copy": copied_from_source,
                       "critic_hard_pass": hard_pass,
                       "failure_reasons": failure_reasons if not verified
                                          else []})
    any_pass = any(row["status"] == "model_checked_candidate"
                   for row in checks)
    report = {"schema_version": "transfer_theme_trial_v3",
              "brief_sha": json_hash(brief),
              "status": ("model_checked_candidate" if any_pass else
                         "themes_blocked") if verified else
                        "unverified_preview",
              "checks": checks, "production_release_allowed": False,
              "story_generation": "not_run", "media_generation": "not_run"}
    _write_json(args.output / "theme_trial_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict:
    intent, reference = _read(args.intent), load_reference(args.reference)
    if (intent.get("source_sha") != reference["source_sha"] or
            reference["source_sha"] != sha256_file(args.video)):
        raise ValueError("source_media_sha_mismatch")
    normalized = normalize_private_intent(intent)
    calibration_cases, calibration_labels = _calibration_inputs(
        args.calibration_cases, args.calibration_labels)
    contrast_cases, labels = _contrast_inputs(args.cases, args.labels)
    known_ids = {str(row[key]) for collection, key in (
        ("claims", "claim_id"), ("events", "event_id"))
        for row in reference[collection]}
    _fresh(args.output)
    _write_json(args.output / "input_lineage.json", {
        "source_media_sha": reference["source_sha"],
        "source_intent_sha": intent["artifact_sha"],
        "reference_sha": sha256_file(args.reference),
        "calibration_cases_sha": sha256_file(args.calibration_cases),
        "calibration_labels_sha": sha256_file(args.calibration_labels),
        "contrast_cases_sha": sha256_file(args.cases),
        "contrast_labels_sha": sha256_file(args.labels),
        "kernel_prompt_sha": json_hash(KERNEL_PROMPT),
        "grounding_prompt_sha": json_hash(GROUNDING_PROMPT),
        "portability_prompt_sha": json_hash(PORTABILITY_PROMPT)})
    _write_json(args.output / "normalized_private_input.json", normalized)
    runner = _runner(args)
    calibration, reused_from = _reusable_calibration(
        args, runner, calibration_cases, calibration_labels)
    _write_json(args.output / "calibration_report.json", calibration)
    _write_json(args.output / "calibration_provenance.json", {
        "reused_from": reused_from,
        "model_config_sha": json_hash(getattr(runner, "cfg", {}) or {})})
    if not calibration["passed"]:
        result = {"status": "blocked", "reason": "calibration_failed"}
        _write_json(args.output / "result.json", result)
        return result
    kernel = _model_call(runner, name="kernel", prompt=KERNEL_PROMPT,
                         payload={"private_intent": normalized["analysis"]},
                         output=args.output, max_new_tokens=3072)
    _write_json(args.output / "transfer_kernel_v3.json", kernel)
    issues = validate_kernel(kernel, known_ids)
    _write_json(args.output / "kernel_validation.json", {"issues": issues})
    if issues:
        result = {"status": "blocked", "reason": "kernel_invalid",
                  "issues": issues}
        _write_json(args.output / "result.json", result)
        return result
    grounding = _model_call(runner, name="grounding_audit",
                            prompt=GROUNDING_PROMPT,
                            payload={"private_intent": normalized["analysis"],
                                     "accepted_evidence": _evidence_view(
                                         reference), "kernel": kernel},
                            output=args.output, max_new_tokens=2048)
    _write_json(args.output / "grounding_audit.json", grounding)
    if not validate_grounding(grounding, kernel):
        result = {"status": "blocked", "reason": "grounding_failed",
                  "theme_generation": "not_run"}
        _write_json(args.output / "result.json", result)
        return result
    record = {"schema_version": "transfer_kernel_record_v3",
              "source_intent_sha": intent["artifact_sha"],
              "source_media_sha": reference["source_sha"],
              "kernel": kernel, "grounding_audit_sha": json_hash(grounding),
              "status": "private_grounded"}
    record["artifact_sha"] = json_hash(record)
    _write_json(args.output / "private_kernel_record.json", record)
    draft = public_brief_draft(kernel)
    _write_json(args.output / "public_brief_draft.json", draft)
    leaks = _transfer_leaks(draft, reference, args.video)
    if leaks:
        result = {"status": "blocked", "reason": "literal_source_leak",
                  "leak_count": len(leaks), "theme_generation": "not_run"}
        _write_json(args.output / "result.json", result)
        return result
    portability = _model_call(runner, name="portability_audit",
                              prompt=PORTABILITY_PROMPT,
                              payload={"public_brief": draft,
                                       "source_bindings": [{"unit_id": row[
                                           "unit_id"], "source_binding": row[
                                           "source_binding"]}
                                           for row in kernel["units"]]},
                              output=args.output, max_new_tokens=2048)
    _write_json(args.output / "portability_audit.json", portability)
    portable = validate_portability(portability, draft)
    mapping, stance = _judge(runner, args.output, draft, contrast_cases)
    regression = score(decisions(mapping, stance, [{"case_id": row["case_id"],
        "fields": public_fields(draft), "story": row["story"]}
        for row in contrast_cases]), labels)
    _write_json(args.output / "contrast_regression.json", regression)
    verified = portable and regression["passed"]
    reasons = ([] if verified else
               (["portability_failed"] if not portable else []) +
               (["contrast_regression_failed"] if not regression["passed"]
                else []))
    brief = draft
    if verified:
        brief = publish_brief(draft, kernel_sha=record["artifact_sha"],
                              grounding=grounding, portability=portability,
                              mapping=mapping, stance=stance,
                              regression=regression)
        _write_json(args.output / "creative_story_brief_v3.json", brief)
        acceptance = make_acceptance(
            brief, record=record, grounding=grounding,
            portability=portability, calibration=calibration, mapping=mapping,
            stance=stance, regression=regression)
        _write_json(args.output / "trial_acceptance_v3.json", acceptance)
    theme_report = _themes(args, runner, brief, kernel, verified=verified,
                           failure_reasons=reasons)
    result = {"status": theme_report["status"], "reasons": reasons,
              "brief_published": verified,
              "trial_acceptance_sha": acceptance["artifact_sha"] if verified
                                      else None,
              "theme_report_sha": json_hash(theme_report),
              "production_release_allowed": False,
              "story_generation": "not_run"}
    if args.baseline_result is not None:
        baseline = _read(args.baseline_result)
        baseline_cases = {row["case_id"]: row["observed"]
                          for row in baseline.get("details", {}).get(
                              "case_results", [])}
        new_cases = {row["case_id"]: row["observed"]
                     for row in regression["case_results"]}
        _write_json(args.output / "baseline_comparison.json", {
            "baseline_result_sha": sha256_file(args.baseline_result),
            "baseline_status": baseline.get("status"),
            "baseline_C02_accepted": baseline_cases.get("C02"),
            "v3_status": result["status"],
            "v3_C02_accepted": new_cases.get("C02"),
            "v3_role_count": len(kernel["units"]),
            "v3_theme_count": len(_read(args.output / "theme_batch.json")[
                "themes"]),
            "v3_brief_published": verified})
    _write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-calibration", type=Path)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--calibration-cases", type=Path, default=Path(
        "config/transfer_judge_calibration_v2_cases.json"))
    parser.add_argument("--calibration-labels", type=Path, default=Path(
        "config/transfer_judge_calibration_v2_labels.json"))
    parser.add_argument("--cases", type=Path, default=Path(
        "config/transfer_contrast_v1_cases.json"))
    parser.add_argument("--labels", type=Path, default=Path(
        "config/transfer_contrast_v1_labels.json"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("trial_output_not_empty")
    try:
        result = run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "run_failure.json", {
            "error_type": type(exc).__name__, "reason": str(exc),
            "new_output_required_for_retry": True})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
