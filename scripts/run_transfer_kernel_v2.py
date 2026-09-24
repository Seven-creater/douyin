"""Calibrate literal mapping judges, then run one bounded kernel trial."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_transfer_kernel_v1 import (
    _blocked, _contrast_inputs, _evidence_view, _fresh, _read, _runner,
)
from src.agentic_video.manifest import json_hash, sha256_file
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)
from src.agentic_video.transfer_eval_v2 import (
    MAPPING_PROMPT, STANCE_PROMPT, decisions, feedback_codes,
    make_acceptance, public_fields, score, validate_mapping,
    validate_stance,
)
from src.agentic_video.transfer_kernel_v1 import (
    KERNEL_AUDIT_PROMPT, KERNEL_PROMPT, audit_feedback_codes,
    kernel_audit_passes, normalize_private_intent, public_brief_draft,
    publish_brief, validate_kernel, validate_kernel_audit,
    validate_public_brief,
)

FROZEN_BADCASE_SHA = (
    "70d4d9db815cd5ad96152fa1c5e5addf196773643574b024a6ad85cc7908164b")


def _calibration_inputs(cases_path: Path, labels_path: Path
                        ) -> tuple[list[dict], dict[str, bool]]:
    data, label_data = _read(cases_path), _read(labels_path)
    cases, labels = data.get("cases"), label_data.get("expected_accept")
    if (data.get("schema_version") != "transfer_judge_calibration_v2" or
            label_data.get("schema_version") !=
            "transfer_judge_calibration_labels_v2" or
            not isinstance(cases, list) or len(cases) != 8 or
            not isinstance(labels, dict) or len(labels) != 8 or
            any(not isinstance(row, dict) or set(row) != {
                "case_id", "fields", "story"} or not isinstance(
                    row["story"], str) or not row["story"].strip() or
                not isinstance(row["fields"], list) or not row["fields"] or
                any(not isinstance(field, dict) or set(field) != {
                    "path", "text"} or not all(isinstance(item, str) and
                    item.strip() for item in field.values()) for field in
                    row["fields"]) for row in cases)):
        raise ValueError("calibration_fixtures_invalid")
    ids = [row["case_id"] for row in cases]
    if (len(set(ids)) != 8 or set(ids) != set(labels) or any(
            type(item) is not bool for item in labels.values()) or
            sum(labels.values()) != 2):
        raise ValueError("calibration_labels_invalid")
    return cases, labels


def _judge(runner, output: Path, prefix: str,
           cases: list[dict]) -> tuple[dict, dict]:
    payload = {"cases": cases}
    mapping = _model_call(runner, name=f"{prefix}_mapping",
                          prompt=MAPPING_PROMPT, payload=payload,
                          output=output, max_new_tokens=4096)
    _write_json(output / f"{prefix}_mapping.json", mapping)
    problems = validate_mapping(mapping, cases)
    if problems:
        raise ValueError(",".join(problems))
    stance_cases = [{"case_id": row["case_id"],
                     "stance": next((field["text"] for field in row["fields"]
                                     if field["path"] == "stance"), None),
                     "story": row["story"]} for row in cases]
    stance = _model_call(runner, name=f"{prefix}_stance",
                         prompt=STANCE_PROMPT,
                         payload={"cases": stance_cases}, output=output,
                         max_new_tokens=1536)
    _write_json(output / f"{prefix}_stance.json", stance)
    problems = validate_stance(stance, cases)
    if problems:
        raise ValueError(",".join(problems))
    return mapping, stance


def _calibrate(runner, output: Path, cases: list[dict], labels: dict[str, bool]
               ) -> dict:
    rounds = []
    for index, seed in enumerate((20260924, 20260925), start=1):
        ordered = list(cases)
        random.Random(seed).shuffle(ordered)
        mapping, stance = _judge(runner, output, f"calibration_{index:02d}",
                                 ordered)
        result = score(decisions(mapping, stance, ordered), labels)
        result["order"] = [row["case_id"] for row in ordered]
        rounds.append(result)
    report = {"schema_version": "transfer_judge_calibration_report_v2",
              "passed": all(row["passed"] for row in rounds) and all(
                  {item["case_id"]: item["observed"] for item in
                   rounds[0]["case_results"]}[case_id] ==
                  {item["case_id"]: item["observed"] for item in
                   rounds[1]["case_results"]}[case_id]
                  for case_id in labels), "rounds": rounds}
    _write_json(output / "calibration_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict:
    intent, reference = _read(args.intent), load_reference(args.reference)
    if (intent.get("source_sha") != reference["source_sha"] or
            intent["source_sha"] != sha256_file(args.video)):
        raise ValueError("source_media_sha_mismatch")
    normalized = normalize_private_intent(intent)
    calibration_cases, calibration_labels = _calibration_inputs(
        args.calibration_cases, args.calibration_labels)
    contrast_cases, ordered_labels = _contrast_inputs(args.cases, args.labels)
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
        "frozen_badcase_brief_file_sha": sha256_file(
            args.frozen_badcase_brief),
        "contrast_cases_sha": sha256_file(args.cases),
        "contrast_labels_sha": sha256_file(args.labels)})
    _write_json(args.output / "normalized_private_input.json", normalized)
    runner = _runner(args)
    calibration = _calibrate(runner, args.output, calibration_cases,
                             calibration_labels)
    if not calibration["passed"]:
        return _blocked(args.output, "judge_calibration_failed", attempt=0,
                        details=calibration)
    frozen_brief = _read(args.frozen_badcase_brief)
    validate_public_brief(frozen_brief, published=True)
    if frozen_brief["artifact_sha"] != FROZEN_BADCASE_SHA:
        raise ValueError("frozen_badcase_brief_sha_mismatch")
    frozen_story = next(row["story"] for row in contrast_cases
                        if row["case_id"] == "C02")
    frozen_case = [{"case_id": "FROZEN_BADCASE",
                    "fields": public_fields(frozen_brief),
                    "story": frozen_story}]
    frozen_mapping, frozen_stance = _judge(
        runner, args.output, "frozen_badcase", frozen_case)
    conflict_paths = [row["path"] for row in frozen_mapping[
        "checks"][0]["mappings"] if row["verdict"] == "conflict" and
        row["path"] in {"stance", "roles.V1.function", "roles.V2.function"}]
    frozen_badcase = {"schema_version": "frozen_badcase_regression_v2",
                      "passed": (not decisions(frozen_mapping, frozen_stance,
                                               frozen_case)["FROZEN_BADCASE"]
                                 and bool(conflict_paths)),
                      "conflict_paths": conflict_paths,
                      "brief_sha": frozen_brief["artifact_sha"]}
    _write_json(args.output / "frozen_badcase_regression.json",
                frozen_badcase)
    if not frozen_badcase["passed"]:
        return _blocked(args.output, "frozen_badcase_false_positive",
                        attempt=0, details=frozen_badcase)
    feedback: list[dict[str, str]] = []
    for attempt in (1, 2):
        prefix = f"attempt_{attempt:02d}"
        kernel = _model_call(runner, name=f"{prefix}_kernel",
                             prompt=KERNEL_PROMPT,
                             payload={"private_intent": normalized["analysis"],
                                      "feedback_codes": feedback},
                             output=args.output, max_new_tokens=2048)
        _write_json(args.output / f"{prefix}_kernel.json", kernel)
        problems = validate_kernel(kernel, known_ids)
        _write_json(args.output / f"{prefix}_validation.json",
                    {"issues": problems})
        if problems:
            feedback = [{"code": row, "field_path": "kernel"}
                        for row in problems]
            if attempt == 2:
                return _blocked(args.output, "kernel_invalid", attempt=2,
                                details=feedback)
            continue
        audit = _model_call(runner, name=f"{prefix}_audit",
                            prompt=KERNEL_AUDIT_PROMPT,
                            payload={"private_intent": normalized["analysis"],
                                     "accepted_evidence": _evidence_view(
                                         reference), "kernel": kernel},
                            output=args.output, max_new_tokens=2048)
        _write_json(args.output / f"{prefix}_audit.json", audit)
        problems = validate_kernel_audit(audit, kernel)
        if problems or not kernel_audit_passes(audit):
            feedback = ([{"code": row, "field_path": "audit"}
                         for row in problems] if problems else
                        audit_feedback_codes(audit))
            if attempt == 2:
                return _blocked(args.output, "kernel_audit_blocked",
                                attempt=2, details=feedback)
            continue
        draft = public_brief_draft(kernel)
        _write_json(args.output / f"{prefix}_public_draft.json", draft)
        leaks = _transfer_leaks(draft, reference, args.video)
        if leaks:
            feedback = [{"code": "literal_source_leak",
                         "field_path": "public_brief"}]
            if attempt == 2:
                return _blocked(args.output, "literal_source_leak",
                                attempt=2, details={"count": len(leaks)})
            continue
        cases = [{"case_id": row["case_id"],
                  "fields": public_fields(draft), "story": row["story"]}
                 for row in contrast_cases]
        mapping, stance = _judge(runner, args.output, prefix, cases)
        regression = score(decisions(mapping, stance, cases), ordered_labels)
        _write_json(args.output / f"{prefix}_regression.json", regression)
        if not regression["passed"]:
            feedback = feedback_codes(mapping, cases, ordered_labels)
            if not feedback:
                feedback = [{"code": "relation_or_stance_mismatch",
                             "field_path": "public_brief"}]
            if attempt == 2:
                return _blocked(args.output, "contrast_regression_failed",
                                attempt=2, details=regression)
            continue
        record = {"schema_version": "transfer_kernel_record_v1",
                  "source_intent_sha": intent["artifact_sha"],
                  "source_media_sha": reference["source_sha"],
                  "kernel": kernel, "audit": audit,
                  "normalization_originals": normalized[
                      "normalization_originals"],
                  "status": "private_model_checked"}
        record["artifact_sha"] = json_hash(record)
        brief = publish_brief(draft, kernel_sha=record["artifact_sha"],
                              audit=audit, evaluation={
                                  "mapping_sha": json_hash(mapping),
                                  "stance_sha": json_hash(stance)},
                              contrast_score=regression)
        acceptance = make_acceptance(
            brief=brief, kernel_sha=record["artifact_sha"], audit=audit,
            calibration=calibration, mapping=mapping, stance=stance,
            regression=regression, frozen_badcase=frozen_badcase)
        _write_json(args.output / "private_kernel_record.json", record)
        _write_json(args.output / "creative_story_brief_v2.json", brief)
        _write_json(args.output / "trial_acceptance_v2.json", acceptance)
        result = {"status": "model_checked_candidate",
                  "auto_trial_allowed": True,
                  "production_release_allowed": False,
                  "brief_sha": brief["artifact_sha"],
                  "acceptance_sha": acceptance["artifact_sha"],
                  "attempt": attempt, "theme_generation": "not_run"}
        _write_json(args.output / "result.json", result)
        return result
    raise RuntimeError("unreachable_kernel_attempts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frozen-badcase-brief", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
