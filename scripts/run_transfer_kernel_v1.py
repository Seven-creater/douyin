"""Isolated private-kernel -> public-brief contrast trial; no video calls."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic_video.manifest import json_hash, sha256_file
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)
from src.agentic_video.transfer_kernel_v1 import (
    CONTRAST_PROMPT, KERNEL_AUDIT_PROMPT, KERNEL_PROMPT,
    audit_feedback_codes, kernel_audit_passes, normalize_private_intent,
    public_brief_draft, publish_brief, score_contrast,
    validate_contrast_eval, validate_kernel, validate_kernel_audit,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(args: argparse.Namespace):
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus
    set_visible_gpus(args.gpu_pair)
    return OmniRunner(load_config(args.config).perception["omni"])


def _fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("trial_output_not_empty")
    path.mkdir(parents=True, exist_ok=True)


def _contrast_inputs(cases_path: Path, labels_path: Path
                     ) -> tuple[list[dict], dict[str, bool]]:
    cases_file, labels_file = _read(cases_path), _read(labels_path)
    cases = cases_file.get("cases")
    labels = labels_file.get("expected_accept")
    if (cases_file.get("schema_version") != "transfer_contrast_cases_v1" or
            labels_file.get("schema_version") != "transfer_contrast_labels_v1"
            or not isinstance(cases, list) or len(cases) != 4 or
            not isinstance(labels, dict) or any(
                not isinstance(row, dict) or set(row) != {"case_id", "story"}
                or not isinstance(row["story"], str) or not row["story"].strip()
                for row in cases)):
        raise ValueError("contrast_fixtures_invalid")
    ids = [row["case_id"] for row in cases]
    if (len(set(ids)) != len(ids) or set(ids) != set(labels) or
            any(type(value) is not bool for value in labels.values()) or
            sum(labels.values()) != 2):
        raise ValueError("contrast_labels_invalid")
    shuffled = list(cases)
    random.Random(20260924).shuffle(shuffled)
    return shuffled, {row["case_id"]: labels[row["case_id"]]
                      for row in shuffled}


def _blocked(output: Path, reason: str, *, attempt: int,
             details: object = None) -> dict:
    result = {"status": "blocked", "reason": reason, "attempt": attempt,
              "details": details, "brief_published": False,
              "theme_generation": "not_run", "production_release_allowed": False}
    _write_json(output / "result.json", result)
    return result


def _evidence_view(reference: dict) -> dict:
    claim_keys = ("claim_id", "modality", "subject", "predicate", "object",
                  "interval", "epistemic_status")
    event_keys = ("event_id", "participants", "ordering", "interval",
                  "action_claim_ids", "outcome_claim_ids",
                  "context_claim_ids")
    return {"claims": [{key: row[key] for key in claim_keys if key in row}
                       for row in reference["claims"]],
            "events": [{key: row[key] for key in event_keys if key in row}
                       for row in reference["events"]]}


def run(args: argparse.Namespace) -> dict:
    intent, reference = _read(args.intent), load_reference(args.reference)
    if (intent.get("source_sha") != reference["source_sha"] or
            intent["source_sha"] != sha256_file(args.video)):
        raise ValueError("source_media_sha_mismatch")
    normalized = normalize_private_intent(intent)
    cases, ordered_labels = _contrast_inputs(args.cases, args.labels)
    known_ids = {str(row[key]) for collection, key in (
        ("claims", "claim_id"), ("events", "event_id"))
        for row in reference[collection]}
    _fresh(args.output)
    _write_json(args.output / "input_lineage.json", {
        "source_media_sha": reference["source_sha"],
        "source_intent_sha": intent["artifact_sha"],
        "reference_sha": sha256_file(args.reference),
        "contrast_cases_sha": sha256_file(args.cases),
        "contrast_labels_sha": sha256_file(args.labels),
        "contrast_order_seed": 20260924})
    _write_json(args.output / "normalized_private_input.json", normalized)
    runner = _runner(args)
    feedback: list[dict[str, str]] = []
    for attempt in (1, 2):
        prefix = f"attempt_{attempt:02d}"
        kernel = _model_call(runner, name=f"{prefix}_kernel",
                             prompt=KERNEL_PROMPT,
                             payload={"private_intent": normalized["analysis"],
                                      "feedback_codes": feedback},
                             output=args.output, max_new_tokens=2048)
        _write_json(args.output / f"{prefix}_kernel.json", kernel)
        issues = validate_kernel(kernel, known_ids)
        _write_json(args.output / f"{prefix}_validation.json",
                    {"issues": issues})
        if issues:
            if attempt == 2:
                return _blocked(args.output, "kernel_shape_or_refs", attempt=2,
                                details=issues)
            feedback = [{"code": issue, "field_path": "kernel"}
                        for issue in issues]
            continue
        audit = _model_call(runner, name=f"{prefix}_audit",
                            prompt=KERNEL_AUDIT_PROMPT,
                            payload={"private_intent": normalized["analysis"],
                                     "accepted_evidence": _evidence_view(
                                         reference), "kernel": kernel},
                            output=args.output,
                            max_new_tokens=2048)
        _write_json(args.output / f"{prefix}_audit.json", audit)
        issues = validate_kernel_audit(audit, kernel)
        if issues:
            if attempt == 2:
                return _blocked(args.output, "kernel_audit_invalid",
                                attempt=2, details=issues)
            feedback = [{"code": issue, "field_path": "audit"}
                        for issue in issues]
            continue
        if not kernel_audit_passes(audit):
            feedback = audit_feedback_codes(audit)
            if attempt == 2:
                return _blocked(args.output, "kernel_audit_blocked",
                                attempt=2, details=feedback)
            continue
        draft = public_brief_draft(kernel)
        _write_json(args.output / f"{prefix}_public_draft.json", draft)
        leaks = _transfer_leaks(draft, reference, args.video)
        if leaks:
            if attempt == 2:
                return _blocked(args.output, "literal_source_leak",
                                attempt=2, details={"count": len(leaks)})
            feedback = [{"code": "literal_source_leak",
                         "field_path": "public_brief"}]
            continue
        record = {"schema_version": "transfer_kernel_record_v1",
                  "source_intent_sha": intent["artifact_sha"],
                  "source_media_sha": reference["source_sha"],
                  "kernel": kernel, "audit": audit,
                  "normalization_originals": normalized[
                      "normalization_originals"],
                  "status": "private_model_checked"}
        record["artifact_sha"] = json_hash(record)
        _write_json(args.output / "private_kernel_record.json", record)
        evaluation = _model_call(runner, name="contrast_evaluation",
                                 prompt=CONTRAST_PROMPT,
                                 payload={"public_brief": draft,
                                          "cases": cases}, output=args.output,
                                 max_new_tokens=1536)
        _write_json(args.output / "contrast_evaluation.json", evaluation)
        eval_issues = validate_contrast_eval(evaluation, list(ordered_labels))
        if eval_issues:
            return _blocked(args.output, "contrast_eval_invalid",
                            attempt=attempt, details=eval_issues)
        score = score_contrast(evaluation, ordered_labels)
        _write_json(args.output / "contrast_score_private.json", score)
        if not score["passed"]:
            return _blocked(args.output, "contrast_eval_blocked",
                            attempt=attempt, details=score)
        brief = publish_brief(draft, kernel_sha=record["artifact_sha"],
                              audit=audit, evaluation=evaluation,
                              contrast_score=score)
        _write_json(args.output / "creative_story_brief_v2.json", brief)
        result = {"status": "model_checked_candidate",
                  "brief_sha": brief["artifact_sha"],
                  "kernel_sha": record["artifact_sha"],
                  "contrast": "4_of_4_matched", "attempt": attempt,
                  "theme_generation": "not_run",
                  "production_release_allowed": False}
        _write_json(args.output / "result.json", result)
        return result
    raise RuntimeError("unreachable_kernel_attempts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(
        "config/transfer_contrast_v1_cases.json"))
    parser.add_argument("--labels", type=Path, default=Path(
        "config/transfer_contrast_v1_labels.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
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
