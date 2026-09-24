"""Run isolated reference-to-creative v2 stages; never publish to R2-D v1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic_video.creative_pipeline.transfer_v2 import (
    run_transfer_creative_trial,
)
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)
from src.agentic_video.reference_transfer_v2 import (
    ABSTRACT_PROMPT, ANALYZE_PROMPT, AUDIT_PROMPT, TRANSFER_AUDIT_PROMPT,
    abstraction_payload,
    build_analysis_payload, build_reference_blueprint, expected_audit_ids,
    inspect_bgm_asset,
    plan_gap_probes,
    publish_candidate_spec, validate_transfer_audit,
)
from src.agentic_video.reference_understanding_v3 import (
    preserve_sampled_source_frames, sampling_from_runner,
)
from src.agentic_video.reference_understanding_v4 import _call_local
from src.agentic_video.recipe_v2 import sha256_file


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markers(reference: dict, video: Path) -> list[str]:
    value = [str(video), str(video.resolve()), reference["source_sha"]]
    value += [str(row["claim_id"]) for row in reference["claims"]]
    value += [str(row["event_id"]) for row in reference["events"]]
    value += [str(row["shot_id"]) for row in
              reference["shot_storyboard"]["shot_cards"]]
    value += [str(row["object"]) for row in reference["claims"]
              if row.get("modality") == "T" and len(str(row.get("object") or "")) >= 4]
    return value


def _runner(args: argparse.Namespace):
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus
    set_visible_gpus(args.gpu_pair)
    return OmniRunner(load_config(args.config).perception["omni"])


def _empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("output_directory_not_empty")
    path.mkdir(parents=True, exist_ok=True)


def analyze(args: argparse.Namespace) -> dict:
    reference = load_reference(args.reference)
    static_path = args.source_run / "static_review.json"
    local_path = args.source_run / "local_observations.json"
    static, local = _read(static_path), _read(local_path)
    if (static["source_sha"] != reference["source_sha"] or
            static["source_sha"] != sha256_file(args.video)):
        raise ValueError("source_media_sha_mismatch")
    _empty_output(args.output)
    _write_json(args.output / "input_lineage.json", {
        "source_run": str(args.source_run.resolve()),
        "static_review_sha256": sha256_file(static_path),
        "local_observations_sha256": sha256_file(local_path),
        "reference_sha": reference["artifact_sha"],
        "video_sha256": static["source_sha"],
        "source_run_unchanged": True,
    })
    runner = _runner(args)
    probes = plan_gap_probes(static, local, limit=args.max_new_probes)
    _write_json(args.output / "gap_probe_plan.json", {
        "schema_version": "reference_gap_probe_plan_v2", "probes": probes,
        "prior_interpretation_not_in_probe_request": True,
        "max_local_probes": 8, "max_probes_per_issue": 2})
    combined = {"schema_version": "reference_local_observations_v4",
                "observations": list(local.get("observations") or [])}
    for probe in probes:
        combined["observations"].append(_call_local(
            runner, probe, static, args.video, args.output))
        _write_json(args.output / "local_observations_combined.json", combined)
    payload = build_analysis_payload(static, combined)
    analysis = _model_call(
        runner, name="blueprint_analysis", prompt=ANALYZE_PROMPT,
        payload=payload, output=args.output, media=args.video,
        channel="AV", fps=2.0, max_new_tokens=8192)
    sampling = sampling_from_runner(runner, {"interval": payload["interval"]})
    _write_json(args.output / "calls" / "blueprint_analysis" /
                "sampling_audit.json", sampling)
    preserve_sampled_source_frames(args.video, sampling,
                                   args.output / "calls" / "blueprint_analysis" /
                                   "sampled_frames")
    audit = _model_call(
        runner, name="blueprint_audit", prompt=AUDIT_PROMPT,
        payload={**payload, "analysis": analysis,
                 "audit_ids": expected_audit_ids(static)}, output=args.output,
        media=args.video, channel="AV", fps=2.0, max_new_tokens=8192)
    _write_json(args.output / "calls" / "blueprint_audit" /
                "sampling_audit.json", sampling_from_runner(
                    runner, {"interval": payload["interval"]}))
    blueprint = build_reference_blueprint(static, combined, analysis, audit)
    bgm_check = inspect_bgm_asset(args.video, args.bgm)
    _write_json(args.output / "bgm_asset_check.json", bgm_check)
    blueprint["audio"]["bgm_asset"] = bgm_check
    blueprint["source_run_sha256"] = sha256_file(
        args.source_run / "reference_reading_v4.json")
    # Recompute after adding the immutable source-run link.
    from src.agentic_video.manifest import json_hash
    blueprint["artifact_sha"] = json_hash({key: value for key, value in
                                           blueprint.items() if key != "artifact_sha"})
    _write_json(args.output / "reference_blueprint_v2.json", blueprint)
    return {"status": "model_checked_candidate" if blueprint[
        "story_candidate_ready"] else "blocked",
        "story_candidate_ready": blueprint["story_candidate_ready"],
        "editing_candidate_ready": blueprint["editing_candidate_ready"],
        "validation_issues": blueprint["validation_issues"],
        "artifact_sha": blueprint["artifact_sha"]}


def abstract(args: argparse.Namespace) -> dict:
    blueprint = _read(args.blueprint)
    reference = load_reference(args.reference)
    if (blueprint["source_sha"] != reference["source_sha"] or
            blueprint["source_sha"] != sha256_file(args.video)):
        raise ValueError("source_media_sha_mismatch")
    _empty_output(args.output)
    runner = _runner(args)
    draft = _model_call(runner, name="transfer_abstraction",
                        prompt=ABSTRACT_PROMPT,
                        payload=abstraction_payload(blueprint),
                        output=args.output, max_new_tokens=6144)
    _write_json(args.output / "transfer_draft.json", draft)
    leaks = _transfer_leaks(draft, reference, args.video)
    if leaks:
        raise ValueError("reference_surface_leak")
    spec = publish_candidate_spec(draft, blueprint,
                                  _markers(reference, args.video))
    transfer_audit = _model_call(
        runner, name="transfer_audit", prompt=TRANSFER_AUDIT_PROMPT,
        payload={"private_blueprint": blueprint, "public_candidate": spec},
        output=args.output, max_new_tokens=1024)
    audit_issues = validate_transfer_audit(transfer_audit)
    _write_json(args.output / "transfer_audit_validation.json",
                {"issues": audit_issues})
    if audit_issues:
        raise ValueError("transfer_audit_blocked:" + ",".join(audit_issues))
    _write_json(args.output / "creative_transfer_spec_v2.json", spec)
    return {"status": spec["status"], "artifact_sha": spec["artifact_sha"],
            "editing_candidate_ready": spec["editing_candidate_ready"]}


def creative(args: argparse.Namespace) -> dict:
    spec = _read(args.spec)
    reference = load_reference(args.reference)
    if reference["source_sha"] != sha256_file(args.video):
        raise ValueError("source_media_sha_mismatch")
    return run_transfer_creative_trial(_runner(args), spec, args.output,
                                       forbidden_markers=_markers(
                                           reference, args.video))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("analyze", "abstract", "creative"),
                        required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--blueprint", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--max-new-probes", type=int, default=8)
    parser.add_argument("--bgm", type=Path)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.max_new_probes not in range(9):
        parser.error("--max-new-probes must be 0..8")
    required = {"analyze": "source_run", "abstract": "blueprint",
                "creative": "spec"}[args.stage]
    if getattr(args, required) is None:
        parser.error(f"--{required.replace('_', '-')} is required")
    try:
        result = {"analyze": analyze, "abstract": abstract,
                  "creative": creative}[args.stage](args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "run_failure.json", {
            "stage": args.stage, "error_type": type(exc).__name__,
            "reason": str(exc), "retry_requires_new_output": True})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
