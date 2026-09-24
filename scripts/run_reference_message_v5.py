"""Diagnose the frozen v4 gate, then test one text-only v5 abstraction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_intent_story_trial import _markers
from scripts.run_reference_message_v4 import (
    _baseline_config_sha, _inputs, _known_ids,
)
from scripts.run_transfer_kernel_v1 import (
    _contrast_inputs, _evidence_view, _fresh, _read, _runner,
)
from scripts.run_transfer_kernel_v2 import _calibration_inputs
from scripts.run_transfer_kernel_v3 import _reusable_calibration
from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.creative_pipeline.intent_trial import has_reference_surface
from src.agentic_video.manifest import json_hash, sha256_file
from src.agentic_video.reference_message_v4 import (
    public_brief as v4_public_brief,
    public_fields as v4_public_fields,
    validate_kernel as v4_validate_kernel,
    validate_message as v4_validate_message,
)
from src.agentic_video.reference_message_v5 import (
    ABSTRACTION_PROMPT, GROUNDING_PROMPT, PORTABILITY_PROMPT,
    THEME_PROMPT, THEME_REVIEW_PROMPT, private_input, public_brief,
    public_fields, validate_audit, validate_kernel, validate_theme_review,
    validate_themes,
)
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json,
)
from src.agentic_video.transfer_eval_v2 import (
    MAPPING_PROMPT, STANCE_PROMPT, decisions, score, validate_mapping,
    validate_stance,
)


def _lineage(args, reference: dict) -> dict:
    return {"source_sha": reference["source_sha"],
            "reference_sha": sha256_file(args.reference),
            "static_sha": sha256_file(args.static),
            "old_intent_sha": sha256_file(args.old_intent),
            "new_message_sha": sha256_file(args.new_message),
            "frozen_compare_result_sha": sha256_file(
                args.frozen_compare / "result.json"),
            "abstraction_prompt_sha": json_hash(ABSTRACTION_PROMPT),
            "grounding_prompt_sha": json_hash(GROUNDING_PROMPT),
            "portability_prompt_sha": json_hash(PORTABILITY_PROMPT),
            "mapping_prompt_sha": json_hash(MAPPING_PROMPT),
            "stance_prompt_sha": json_hash(STANCE_PROMPT)}


def _frozen_inputs(args, reference: dict) -> tuple[dict, dict]:
    old_run = args.frozen_compare
    old_lineage = _read(old_run / "input_lineage.json")
    expected = _lineage(args, reference)
    for key in ("reference_sha", "static_sha", "old_intent_sha",
                "new_message_sha"):
        if old_lineage.get(key) != expected[key]:
            raise ValueError(f"frozen_{key}_mismatch")
    frozen_result = _read(old_run / "result.json")
    if (frozen_result.get("status") != "blocked" or
            any(frozen_result.get("branches", {}).get(name, {}).get(
                "reason") != "kernel_invalid" for name in ("old", "new"))):
        raise ValueError("frozen_result_unexpected")
    kernels = {}
    for name in ("old", "new"):
        stage = old_run / "calls" / f"{name}_abstraction"
        meta = _read(stage / "model_call.json")
        if (meta.get("response_sha") != sha256_file(
                stage / "raw_response.txt") or
                meta.get("model_config_sha") != _baseline_config_sha(args)):
            raise ValueError(f"frozen_{name}_call_mismatch")
        kernel = _read(old_run / f"{name}_kernel.json")
        if [issue for issue in v4_validate_kernel(kernel,
                _known_ids(reference)) if issue !=
                "private_substitution_failed"]:
            raise ValueError(f"frozen_{name}_kernel_malformed")
        kernels[name] = kernel
    return kernels["old"], kernels["new"]


def _model_config_check(args, runner, new_record: dict) -> str:
    config_sha = json_hash(getattr(runner, "cfg", {}) or {})
    if (config_sha != _baseline_config_sha(args) or
            new_record.get("model_config_sha") != config_sha):
        raise ValueError("baseline_model_config_mismatch")
    return config_sha


def _new_record(args, reference: dict) -> dict:
    record = _read(args.new_message)
    if (record.get("source_sha") != reference["source_sha"] or
            record.get("static_sha") != sha256_file(args.static) or
            record.get("status") != "private_candidate" or
            record.get("artifact_sha") != json_hash({
                key: value for key, value in record.items()
                if key != "artifact_sha"}) or
            v4_validate_message(record.get("reading") or {},
                                _known_ids(reference))):
        raise ValueError("new_message_lineage_invalid")
    return record


def diagnose(args) -> dict:
    reference, _, _ = _inputs(args)
    new_record = _new_record(args, reference)
    kernels = _frozen_inputs(args, reference)
    _fresh(args.output)
    _write_json(args.output / "input_lineage.json", _lineage(args, reference))
    runner = _runner(args)
    _model_config_check(args, runner, new_record)
    calibration_cases, calibration_labels = _calibration_inputs(
        args.calibration_cases, args.calibration_labels)
    calibration, reused = _reusable_calibration(
        args, runner, calibration_cases, calibration_labels)
    _write_json(args.output / "calibration_report.json", calibration)
    _write_json(args.output / "calibration_provenance.json", {
        "reused_from": reused})
    if not calibration["passed"]:
        result = {"status": "judge_unreliable",
                  "reason": "mapping_judge_calibration_failed",
                  "theme_generation": "not_run"}
        _write_json(args.output / "result.json", result)
        return result
    cases, _ = _contrast_inputs(args.cases, args.labels)
    positives = [row for row in cases if row["case_id"] in {"C01", "C02"}]
    if len(positives) != 2:
        raise ValueError("diagnosis_case_coverage_invalid")
    mapping_cases = []
    for name, kernel in zip(("old", "new"), kernels):
        draft = v4_public_brief(kernel)
        for row in positives:
            mapping_cases.append({"case_id": f"{name}_{row['case_id']}",
                                  "fields": v4_public_fields(
                                      draft, include_optional=False),
                                  "story": row["story"]})
    mapping = _model_call(runner, name="frozen_mapping_diagnosis",
                          prompt=MAPPING_PROMPT,
                          payload={"cases": mapping_cases},
                          output=args.output, max_new_tokens=8192)
    _write_json(args.output / "frozen_mapping_diagnosis.json", mapping)
    issues = validate_mapping(mapping, mapping_cases)
    if issues:
        result = {"status": "judge_unreliable", "reason": issues,
                  "theme_generation": "not_run"}
        _write_json(args.output / "result.json", result)
        return result
    results = {}
    for name in ("old", "new"):
        checks = [row for row in mapping["checks"] if row["case_id"].startswith(
            f"{name}_")]
        results[name] = {"case_results": [{
            "case_id": row["case_id"],
            "source_bound_paths": [item["path"] for item in row["mappings"]
                                   if item["verdict"] != "mapped"],
            "audience_takeaway_rejected": any(
                item["path"] == "audience_takeaway" and
                item["verdict"] != "mapped" for item in row["mappings"])}
            for row in checks]}
    passed = all(len(row["case_results"]) == 2 and all(
        item["audience_takeaway_rejected"] for item in row["case_results"])
        for row in results.values())
    result = {"status": "diagnosis_passed" if passed else "judge_unreliable",
              "branches": results, "video_calls": 0,
              "theme_generation": "not_run",
              "production_release_allowed": False}
    _write_json(args.output / "result.json", result)
    return result


def _ground_fields(kernel: dict) -> list[dict]:
    fields = [{"path": "source_claim", "text": kernel["source_claim"][
        "statement"]}]
    fields += [{"path": f"roles.{row['role_id']}",
                "text": f"{row['source_binding']} => {row['function']}"}
               for row in kernel["roles"]]
    fields += [{"path": f"relations.{row['relation_id']}",
                "text": f"{row['from_role']} {row['relation']} "
                        f"{row['to_role']}; scope: {row['scope']}"}
               for row in kernel["relations"]]
    fields += [{"path": "audience_goal", "text": kernel["audience_goal"]},
               {"path": "evidence_scope", "text": kernel["evidence_scope"]}]
    if kernel["optional_tone"] is not None:
        fields.append({"path": "optional_tone",
                       "text": kernel["optional_tone"]})
    return fields


def _assess(args, runner, reference: dict, reading: dict, name: str,
            cases: list[dict], labels: dict) -> dict:
    output = args.output
    _write_json(output / f"{name}_private_input.json", reading)
    kernel = _model_call(runner, name=f"{name}_abstraction",
                         prompt=ABSTRACTION_PROMPT,
                         payload={"private_reading": reading},
                         output=output, max_new_tokens=4096)
    _write_json(output / f"{name}_kernel.json", kernel)
    issues = validate_kernel(kernel, _known_ids(reference))
    if issues:
        return {"status": "blocked", "reason": "kernel_invalid",
                "issues": issues}
    ground_fields = _ground_fields(kernel)
    grounding = _model_call(runner, name=f"{name}_grounding",
                            prompt=GROUNDING_PROMPT,
                            payload={"private_reading": reading,
                                     "accepted_evidence": _evidence_view(
                                         reference), "kernel": kernel,
                                     "fields": ground_fields},
                            output=output, max_new_tokens=3072)
    _write_json(output / f"{name}_grounding.json", grounding)
    issues = validate_audit(grounding, ground_fields, kind="grounding")
    if issues:
        return {"status": "blocked", "reason": "grounding_failed",
                "issues": issues}
    draft = public_brief(kernel)
    _write_json(output / f"{name}_public_draft.json", draft)
    validate_creative_boundary(draft)
    leaks = _transfer_leaks(draft, reference, args.video)
    if leaks:
        return {"status": "blocked", "reason": "literal_source_leak",
                "issues": leaks}
    fields = public_fields(draft, include_optional=True)
    portability = _model_call(runner, name=f"{name}_portability",
                              prompt=PORTABILITY_PROMPT,
                              payload={"public_brief": draft,
                                       "public_fields": fields,
                                       "private_source_claim": kernel[
                                           "source_claim"]},
                              output=output, max_new_tokens=3072)
    _write_json(output / f"{name}_portability.json", portability)
    portability_issues = validate_audit(portability, fields,
                                        kind="portability")
    mapped_cases = [{"case_id": row["case_id"],
                     "fields": public_fields(draft, include_optional=False),
                     "story": row["story"]} for row in cases]
    mapping = _model_call(runner, name=f"{name}_mapping",
                          prompt=MAPPING_PROMPT,
                          payload={"cases": mapped_cases},
                          output=output, max_new_tokens=8192)
    _write_json(output / f"{name}_mapping.json", mapping)
    issues = validate_mapping(mapping, mapped_cases)
    if issues:
        return {"status": "blocked", "reason": "mapping_invalid",
                "issues": issues}
    stance = _model_call(runner, name=f"{name}_stance",
                         prompt=STANCE_PROMPT,
                         payload={"cases": [{"case_id": row["case_id"],
                           "stance": draft["audience_goal"],
                           "story": row["story"]} for row in cases]},
                         output=output, max_new_tokens=1536)
    _write_json(output / f"{name}_stance.json", stance)
    issues = validate_stance(stance, mapped_cases)
    if issues:
        return {"status": "blocked", "reason": "stance_invalid",
                "issues": issues}
    regression = score(decisions(mapping, stance, mapped_cases), labels)
    _write_json(output / f"{name}_regression.json", regression)
    if portability_issues or not regression["passed"]:
        return {"status": "blocked", "reason": (
            ["portability_failed"] if portability_issues else []) + (
            ["contrast_regression_failed"] if not regression["passed"]
            else []), "portability_issues": portability_issues,
            "regression": regression}
    brief = {**draft, "parent_kernel_sha": json_hash(kernel),
             "grounding_sha": json_hash(grounding),
             "portability_sha": json_hash(portability),
             "model_config_sha": json_hash(getattr(runner, "cfg", {}) or {}),
             "status": "model_checked_candidate",
             "production_release_allowed": False}
    brief["artifact_sha"] = json_hash(brief)
    _write_json(output / f"{name}_creative_message_brief_v5.json", brief)
    return {"status": "model_checked_candidate",
            "brief_sha": brief["artifact_sha"], "regression": regression}


def _themes(args, runner, reference: dict, brief: dict) -> dict:
    public = {key: brief[key] for key in (
        "schema_version", "audience_goal", "roles", "relations",
        "evidence_scope", "free_slots", "optional_tone")}
    validate_creative_boundary(public)
    batch = _model_call(runner, name="diagnostic_themes",
                        prompt=THEME_PROMPT,
                        payload={"creative_message_brief": public},
                        output=args.output, max_new_tokens=4096)
    _write_json(args.output / "diagnostic_themes.json", batch)
    issues = validate_themes(batch)
    if has_reference_surface(batch, _markers(args)):
        issues.append("literal_source_surface_leak")
    if issues:
        report = {"status": "blocked", "issues": issues,
                  "production_release_allowed": False}
        _write_json(args.output / "theme_report.json", report)
        return report
    review = _model_call(runner, name="theme_review",
                         prompt=THEME_REVIEW_PROMPT,
                         payload={"creative_message_brief": public,
                                  "private_source_bindings": [
                                      {"subject": row.get("subject"),
                                       "predicate": row.get("predicate"),
                                       "object": row.get("object")}
                                      for row in reference["claims"]],
                                  "themes": batch}, output=args.output,
                         max_new_tokens=3072)
    _write_json(args.output / "theme_review.json", review)
    issues = validate_theme_review(review)
    if issues:
        report = {"status": "blocked", "issues": issues,
                  "production_release_allowed": False}
    else:
        accepted = [row for row in review["checks"] if
                    row["message_match"] and row["evidence_logic"] and
                    row["filmable"] and not row["surface_copy"]]
        categories = {row["basis_category"].strip().casefold()
                      for row in accepted}
        cross_domain = sum(not row["source_domain_overlap"]
                           for row in accepted)
        passed = (len(accepted) >= 3 and len(categories) >= 3 and
                  cross_domain >= 2)
        report = {"status": "diagnostic_pass" if passed else "blocked",
                  "accepted_theme_ids": [row["theme_id"] for row in accepted],
                  "distinct_basis_categories": len(categories),
                  "cross_domain_accepted": cross_domain,
                  "production_release_allowed": False,
                  "story_generation": "not_run",
                  "media_generation": "not_run"}
    _write_json(args.output / "theme_report.json", report)
    return report


def trial(args) -> dict:
    reference, _, old = _inputs(args)
    new_record = _new_record(args, reference)
    _frozen_inputs(args, reference)
    diagnosis = _read(args.diagnosis / "result.json")
    if diagnosis.get("status") != "diagnosis_passed":
        raise ValueError("independent_judge_diagnosis_not_passed")
    if _read(args.diagnosis / "input_lineage.json") != _lineage(args,
                                                                  reference):
        raise ValueError("diagnosis_lineage_mismatch")
    _fresh(args.output)
    lineage = _lineage(args, reference)
    lineage["diagnosis_result_sha"] = sha256_file(args.diagnosis /
                                                  "result.json")
    lineage["cases_sha"] = sha256_file(args.cases)
    lineage["labels_sha"] = sha256_file(args.labels)
    _write_json(args.output / "input_lineage.json", lineage)
    cases, labels = _contrast_inputs(args.cases, args.labels)
    calibration_cases, calibration_labels = _calibration_inputs(
        args.calibration_cases, args.calibration_labels)
    runner = _runner(args)
    config_sha = _model_config_check(args, runner, new_record)
    calibration, reused = _reusable_calibration(
        args, runner, calibration_cases, calibration_labels)
    _write_json(args.output / "calibration_report.json", calibration)
    _write_json(args.output / "calibration_provenance.json", {
        "reused_from": reused, "model_config_sha": config_sha})
    if not calibration["passed"]:
        result = {"status": "blocked", "reason": "judge_calibration_failed",
                  "theme_generation": "not_run"}
        _write_json(args.output / "result.json", result)
        return result
    branches = {}
    for name, reading in (("old", private_input(old, old=True)),
                          ("new", private_input(new_record["reading"],
                                                old=False))):
        branches[name] = _assess(args, runner, reference, reading, name,
                                 cases, labels)
    selected = "old" if branches["old"]["status"] == \
        "model_checked_candidate" else None
    result = {"status": "blocked" if selected is None else
              "model_checked_candidate", "branches": branches,
              "selected_branch": selected, "video_calls": 0,
              "production_release_allowed": False}
    if selected is None:
        result["theme_generation"] = "not_run"
    else:
        brief = _read(args.output /
                      "old_creative_message_brief_v5.json")
        result["theme_trial"] = _themes(args, runner, reference, brief)
        if result["theme_trial"]["status"] != "diagnostic_pass":
            result["status"] = "theme_diagnostic_blocked"
    _write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnose", "trial"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--old-intent", type=Path, required=True)
    parser.add_argument("--new-message", type=Path, required=True)
    parser.add_argument("--frozen-compare", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reuse-calibration", type=Path)
    parser.add_argument("--calibration-cases", type=Path, default=Path(
        "config/transfer_judge_calibration_v2_cases.json"))
    parser.add_argument("--calibration-labels", type=Path, default=Path(
        "config/transfer_judge_calibration_v2_labels.json"))
    parser.add_argument("--cases", type=Path, default=Path(
        "config/transfer_contrast_v1_cases.json"))
    parser.add_argument("--labels", type=Path, default=Path(
        "config/transfer_contrast_v1_labels.json"))
    args = parser.parse_args()
    if args.stage == "trial" and args.diagnosis is None:
        parser.error("--diagnosis required for trial")
    try:
        result = diagnose(args) if args.stage == "diagnose" else trial(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "run_failure.json", {
            "stage": args.stage, "error_type": type(exc).__name__,
            "reason": str(exc), "new_output_required_for_retry": True})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
