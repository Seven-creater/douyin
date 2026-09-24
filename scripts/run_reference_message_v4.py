"""One AV reading, then a bounded old-vs-new message transfer comparison."""
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
from scripts.run_transfer_kernel_v2 import _calibration_inputs
from scripts.run_transfer_kernel_v3 import _reusable_calibration
from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.creative_pipeline.intent_trial import (
    SOURCE_COPY_PROMPT, has_reference_surface,
    validate_source_copy_audit,
)
from src.agentic_video.manifest import json_hash, sha256_file
from src.agentic_video.reference_intent_v1 import intent_payload
from src.agentic_video.reference_message_v4 import (
    ABSTRACTION_PROMPT, AUDIENCE_PROMPT, GROUNDING_PROMPT,
    PORTABILITY_PROMPT, THEME_CRITIC_PROMPT, THEME_PROMPT,
    artifact_with_sha, new_message_input, old_message_input,
    public_brief, public_fields, validate_audit, validate_brief,
    validate_kernel, validate_message, validate_theme_critique,
    validate_themes,
)
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)
from src.agentic_video.transfer_eval_v2 import (
    MAPPING_PROMPT, STANCE_PROMPT, decisions, score, validate_mapping,
    validate_stance,
)


def _known_ids(reference: dict) -> set[str]:
    return {str(row[key]) for collection, key in (
        ("claims", "claim_id"), ("events", "event_id"))
        for row in reference[collection]}


def _lineage(args: argparse.Namespace, reference: dict) -> dict:
    return {"source_video_sha": reference["source_sha"],
            "reference_sha": sha256_file(args.reference),
            "static_sha": sha256_file(args.static),
            "old_intent_sha": sha256_file(args.old_intent),
            "audience_prompt_sha": json_hash(AUDIENCE_PROMPT),
            "abstraction_prompt_sha": json_hash(ABSTRACTION_PROMPT),
            "grounding_prompt_sha": json_hash(GROUNDING_PROMPT),
            "portability_prompt_sha": json_hash(PORTABILITY_PROMPT)}


def _inputs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    reference, static, old = (load_reference(args.reference),
                              _read(args.static), _read(args.old_intent))
    media_sha = sha256_file(args.video)
    if (reference["source_sha"] != media_sha or
            static.get("source_sha") != media_sha or
            old.get("source_sha") != media_sha or
            static.get("agent_reference_sha") != reference.get(
                "artifact_sha") or
            old.get("static_review_sha") != static.get("artifact_sha")):
        raise ValueError("reference_input_lineage_mismatch")
    return reference, static, old


def _baseline_config_sha(args: argparse.Namespace) -> str:
    baseline = _read(args.old_intent.parent / "calls" / "intent_initial" /
                     "model_call.json")
    return baseline["model_config_sha"]


def extract(args: argparse.Namespace) -> dict:
    reference, static, _ = _inputs(args)
    _fresh(args.output)
    _write_json(args.output / "input_lineage.json", _lineage(args, reference))
    runner = _runner(args)
    model_config_sha = json_hash(getattr(runner, "cfg", {}) or {})
    if model_config_sha != _baseline_config_sha(args):
        raise ValueError("baseline_model_config_mismatch")
    reading = _model_call(runner, name="audience_message",
                          prompt=AUDIENCE_PROMPT,
                          payload=intent_payload(static), output=args.output,
                          media=args.video, channel="AV", fps=1.5,
                          max_new_tokens=2048)
    _write_json(args.output / "reading_raw.json", reading)
    issues = validate_message(reading, _known_ids(reference))
    if not issues and reading["takeaway_candidate"]["text"].strip(
            ).casefold() in {"unknown", "unclear", "undetermined"}:
        issues.append("message_unknown")
    result = {"status": "blocked" if issues else "reading_candidate",
              "issues": issues, "full_video_calls": 1,
              "production_release_allowed": False}
    if not issues:
        record = artifact_with_sha({
            "schema_version": "reference_message_record_v4",
            "source_sha": reference["source_sha"],
            "static_sha": sha256_file(args.static),
            "model_config_sha": model_config_sha,
            "reading": reading, "status": "private_candidate"})
        _write_json(args.output / "reference_message_v4.json", record)
    _write_json(args.output / "result.json", result)
    return result


def _assess_branch(args: argparse.Namespace, runner, reference: dict,
                   reading: dict, name: str, cases: list[dict], labels: dict
                   ) -> dict:
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
    evidence = _evidence_view(reference)
    ground_fields = [{"path": key, "text": kernel[key]["statement"]}
                     for key in ("source_claim", "audience_takeaway",
                                 "misjudgment_mechanism",
                                 "corrective_evidence_role",
                                 "revised_judgment")]
    grounding = _model_call(
        runner, name=f"{name}_grounding", prompt=GROUNDING_PROMPT,
        payload={"private_reading": reading,
                 "accepted_evidence": evidence, "kernel": kernel,
                 "fields": ground_fields}, output=output,
        max_new_tokens=2048)
    _write_json(output / f"{name}_grounding.json", grounding)
    if not validate_audit(grounding, ground_fields, kind="grounding"):
        return {"status": "blocked", "reason": "grounding_failed"}
    draft = public_brief(kernel)
    _write_json(output / f"{name}_public_draft.json", draft)
    if validate_brief(draft):
        return {"status": "blocked", "reason": "brief_schema_invalid"}
    validate_creative_boundary(draft)
    if _transfer_leaks(draft, reference, args.video):
        return {"status": "blocked", "reason": "literal_source_leak"}
    portability_fields = public_fields(draft, include_optional=True)
    portability = _model_call(
        runner, name=f"{name}_portability", prompt=PORTABILITY_PROMPT,
        payload={"public_fields": portability_fields,
                 "source_claim": kernel["source_claim"],
                 "private_substitutions": kernel["private_substitutions"]},
        output=output, max_new_tokens=3072)
    _write_json(output / f"{name}_portability.json", portability)
    portable = validate_audit(portability, portability_fields,
                              kind="portability")
    mapped_cases = [{"case_id": row["case_id"],
                     "fields": public_fields(draft, include_optional=False),
                     "story": row["story"]} for row in cases]
    mapping = _model_call(
        runner, name=f"{name}_contrast_mapping", prompt=MAPPING_PROMPT,
        payload={"cases": mapped_cases}, output=output,
        max_new_tokens=4096)
    _write_json(output / f"{name}_contrast_mapping.json", mapping)
    if validate_mapping(mapping, mapped_cases):
        return {"status": "blocked", "reason": "mapping_format_invalid"}
    stance = _model_call(
        runner, name=f"{name}_contrast_stance", prompt=STANCE_PROMPT,
        payload={"cases": [{"case_id": row["case_id"],
                            "stance": draft["audience_takeaway"],
                            "story": row["story"]} for row in cases]},
        output=output, max_new_tokens=1536)
    _write_json(output / f"{name}_contrast_stance.json", stance)
    if validate_stance(stance, mapped_cases):
        return {"status": "blocked", "reason": "stance_format_invalid"}
    regression = score(decisions(mapping, stance, mapped_cases), labels)
    _write_json(output / f"{name}_contrast_regression.json", regression)
    if not portable or not regression["passed"]:
        return {"status": "blocked",
                "reason": (["portability_failed"] if not portable else []) +
                (["contrast_regression_failed"] if not regression["passed"]
                 else []), "regression": regression}
    brief = artifact_with_sha({**draft,
        "parent_kernel_sha": json_hash(kernel),
        "grounding_sha": json_hash(grounding),
        "portability_sha": json_hash(portability),
        "status": "model_checked_candidate",
        "production_release_allowed": False})
    _write_json(output / f"{name}_creative_message_brief_v4.json", brief)
    return {"status": "model_checked_candidate",
            "brief_sha": brief["artifact_sha"], "regression": regression}


def _theme_trial(args: argparse.Namespace, runner, reference: dict,
                 brief: dict) -> dict:
    public = {key: brief[key] for key in (
        "schema_version", "audience_takeaway", "misjudgment_mechanism",
        "corrective_evidence_role", "revised_judgment",
        "optional_sequence", "free_dimensions")}
    validate_creative_boundary(public)
    batch = _model_call(runner, name="diagnostic_themes",
                        prompt=THEME_PROMPT,
                        payload={"creative_message_brief": public},
                        output=args.output, max_new_tokens=3072)
    _write_json(args.output / "diagnostic_themes.json", batch)
    issues = validate_themes(batch)
    if has_reference_surface(batch, _markers(args)):
        issues.append("literal_source_surface_leak")
    if issues:
        return {"status": "themes_blocked", "issues": issues,
                "production_release_allowed": False}
    critique = _model_call(
        runner, name="theme_critique", prompt=THEME_CRITIC_PROMPT,
        payload={"creative_message_brief": public, "themes": batch},
        output=args.output, max_new_tokens=1536)
    _write_json(args.output / "theme_critique.json", critique)
    issues = validate_theme_critique(critique)
    if issues:
        return {"status": "themes_blocked", "issues": issues,
                "production_release_allowed": False}
    source_copy = _model_call(
        runner, name="theme_source_copy", prompt=SOURCE_COPY_PROMPT,
        payload={"private_source_bindings": [
            {"subject": row.get("subject"),
             "predicate": row.get("predicate"),
             "object": row.get("object")}
            for row in reference["claims"]],
                 "candidates": [{"candidate_id": row["theme_id"],
                                 "candidate": row}
                                for row in batch["themes"]]},
        output=args.output, max_new_tokens=2048)
    _write_json(args.output / "theme_source_copy.json", source_copy)
    issues = validate_source_copy_audit(source_copy,
                                        [f"T{i}" for i in range(1, 7)])
    if issues:
        return {"status": "themes_blocked", "issues": issues,
                "production_release_allowed": False}
    checks = []
    for row, copy in zip(critique["checks"], source_copy["checks"]):
        checks.append({"theme_id": row["theme_id"],
                       "message_match": row["message_match"],
                       "evidence_logic": row["evidence_logic"],
                       "filmable": row["filmable"],
                       "source_copy": copy["surface_copy"],
                       "critic_reason": row["reason"],
                       "source_copy_reason": copy["reason"]})
    report = {"status": "diagnostic_only", "checks": checks,
              "production_release_allowed": False,
              "story_generation": "not_run", "media_generation": "not_run"}
    _write_json(args.output / "theme_report.json", report)
    return report


def compare(args: argparse.Namespace) -> dict:
    reference, static, old = _inputs(args)
    new_record = _read(args.new_message)
    if (new_record.get("source_sha") != reference["source_sha"] or
            new_record.get("static_sha") != sha256_file(args.static) or
            new_record.get("artifact_sha") != json_hash({
                key: value for key, value in new_record.items()
                if key != "artifact_sha"}) or
            validate_message(new_record.get("reading") or {},
                             _known_ids(reference))):
        raise ValueError("new_message_lineage_invalid")
    _fresh(args.output)
    lineage = _lineage(args, reference)
    lineage["new_message_sha"] = sha256_file(args.new_message)
    lineage["contrast_cases_sha"] = sha256_file(args.cases)
    lineage["contrast_labels_sha"] = sha256_file(args.labels)
    _write_json(args.output / "input_lineage.json", lineage)
    cases, labels = _contrast_inputs(args.cases, args.labels)
    calibration_cases, calibration_labels = _calibration_inputs(
        args.calibration_cases, args.calibration_labels)
    runner = _runner(args)
    if (new_record.get("model_config_sha") != json_hash(
            getattr(runner, "cfg", {}) or {}) or
            new_record["model_config_sha"] != _baseline_config_sha(args)):
        raise ValueError("comparison_model_config_mismatch")
    calibration, reused = _reusable_calibration(
        args, runner, calibration_cases, calibration_labels)
    _write_json(args.output / "calibration_report.json", calibration)
    _write_json(args.output / "calibration_provenance.json", {
        "reused_from": reused,
        "model_config_sha": json_hash(getattr(runner, "cfg", {}) or {})})
    if not calibration["passed"]:
        result = {"status": "blocked", "reason": "judge_calibration_failed"}
        _write_json(args.output / "result.json", result)
        return result
    branches = {}
    for name, reading in (("old", old_message_input(old)),
                          ("new", new_message_input(
                              new_record["reading"]))):
        branches[name] = _assess_branch(
            args, runner, reference, reading, name, cases, labels)
    eligible = [name for name in ("new", "old") if branches[name][
        "status"] == "model_checked_candidate"]
    selected = eligible[0] if eligible else None
    result = {"status": "blocked" if selected is None else
              "model_checked_candidate", "branches": branches,
              "selected_branch": selected,
              "full_video_calls_this_stage": 0,
              "production_release_allowed": False}
    if selected is not None:
        brief = _read(args.output /
                      f"{selected}_creative_message_brief_v4.json")
        result["theme_trial"] = _theme_trial(args, runner, reference, brief)
    else:
        result["theme_generation"] = "not_run"
    _write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "compare"),
                        required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--old-intent", type=Path, required=True)
    parser.add_argument("--new-message", type=Path)
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
    if args.stage == "compare" and args.new_message is None:
        parser.error("--new-message is required for compare")
    try:
        result = extract(args) if args.stage == "extract" else compare(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "run_failure.json", {
            "stage": args.stage, "error_type": type(exc).__name__,
            "reason": str(exc), "new_output_required_for_retry": True})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
