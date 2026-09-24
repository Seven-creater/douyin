"""Isolated intent-led reference understanding and creative-brief experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.reference_intent_v1 import (
    ABSTRACT_PROMPT, AUDIT_PROMPT, BRIEF_AUDIT_PROMPT, CORE_AUDIT_IDS,
    INTENT_PROMPT, PROBE_PROMPT, REVISE_PROMPT, build_intent, intent_payload,
    publish_story_brief, validate_intent, validate_intent_audit,
)
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)
from src.agentic_video.reference_understanding_v3 import (
    preserve_sampled_source_frames, sampling_from_runner,
)
from src.perception.omni_runner import cut_clip


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(args: argparse.Namespace):
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus
    set_visible_gpus(args.gpu_pair)
    return OmniRunner(load_config(args.config).perception["omni"])


def _empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("output_directory_not_empty")
    path.mkdir(parents=True, exist_ok=True)


def _audit(runner, static: dict, analysis: dict, output: Path,
           video: Path, index: int) -> dict:
    payload = {**intent_payload(static), "analysis": analysis,
               "audit_ids": list(CORE_AUDIT_IDS),
               "observation_dimensions": ["story_intent", "attribution",
                                          "contradiction"]}
    return _model_call(runner, name=f"intent_audit_{index}",
                       prompt=AUDIT_PROMPT, payload=payload, output=output,
                       media=video, channel="AV", fps=1.0,
                       max_new_tokens=1536)


def _probe_request(audit: dict, duration: float) -> tuple[float, float] | None:
    request = audit.get("probe_request")
    if not isinstance(request, dict):
        return None
    interval = request.get("interval")
    if (not isinstance(interval, list) or len(interval) != 2 or
            not all(isinstance(value, (int, float)) for value in interval)):
        return None
    start, end = map(float, interval)
    if not 0 <= start < end <= duration or end - start > 5.0:
        return None
    return start, end


def extract(args: argparse.Namespace) -> dict:
    static = _read(args.static)
    reference = load_reference(args.reference)
    if static["source_sha"] != reference["source_sha"] or static[
            "source_sha"] != sha256_file(args.video):
        raise ValueError("source_media_sha_mismatch")
    _empty(args.output)
    _write_json(args.output / "input_lineage.json", {
        "reference_sha": reference["artifact_sha"],
        "static_sha": sha256_file(args.static),
        "source_video_sha": static["source_sha"],
        "old_runs_unchanged": True,
        "user_interpretation_in_request": False})
    runner = _runner(args)
    payload = intent_payload(static)
    analysis = _model_call(runner, name="intent_initial",
                           prompt=INTENT_PROMPT, payload=payload,
                           output=args.output, media=args.video,
                           channel="AV", fps=1.5, max_new_tokens=2048)
    audit = _audit(runner, static, analysis, args.output, args.video, 0)
    used: set[tuple[float, float, float]] = set()
    probes: list[dict] = []
    for index in range(1, 3):
        if (not validate_intent(analysis, static) and
                not validate_intent_audit(audit) and
                all(row["verdict"] == "supported" for row in audit["checks"]) and
                audit["critical_conflict"]["present"] is False):
            break
        interval = _probe_request(audit, static["duration_s"])
        if interval is None:
            break
        fps = 8.0 if index == 1 else 12.0
        key = (*interval, fps)
        if key in used:
            break
        used.add(key)
        clip = cut_clip("ffmpeg", args.video, args.output / "clips" /
                        f"intent_probe_{index}", start_s=interval[0],
                        end_s=interval[1], include_audio=True)
        statements = [row for row in static["text_timeline"] if
                      row["interval"][1] > interval[0] and
                      row["interval"][0] < interval[1]]
        probe_payload = {"interval": [0.0, interval[1] - interval[0]],
                         "source_interval": list(interval),
                         "text_statements": [{"source_type":
                             "attributed_on_screen_text", **row}
                             for row in statements],
                         "observation_dimensions": ["visible_change",
                                                    "sound", "on_screen_text"]}
        # The auditor's question and previous story are not sent to the probe.
        observation = _model_call(runner, name=f"intent_probe_{index}",
                                  prompt=PROBE_PROMPT, payload=probe_payload,
                                  output=args.output, media=clip,
                                  channel="AV", fps=fps, max_new_tokens=1024)
        sampling = sampling_from_runner(runner, {"interval": list(interval)})
        _write_json(args.output / "calls" / f"intent_probe_{index}" /
                    "sampling_audit.json", sampling)
        preserve_sampled_source_frames(clip, sampling, args.output / "calls" /
                                       f"intent_probe_{index}" /
                                       "sampled_frames")
        probes.append({"source_interval": list(interval), "fps": fps,
                       "observation": observation})
        analysis = _model_call(runner, name=f"intent_revision_{index}",
                               prompt=REVISE_PROMPT,
                               payload={"reference": payload,
                                        "local_observations": probes},
                               output=args.output, max_new_tokens=2048)
        audit = _audit(runner, static, analysis, args.output,
                       args.video, index)
    _write_json(args.output / "analysis.json", analysis)
    _write_json(args.output / "audit.json", audit)
    _write_json(args.output / "probes.json", {"probes": probes})
    calls = [{"name": path.parent.name,
              "metadata_sha": sha256_file(path)} for path in sorted(
                  (args.output / "calls").glob("*/model_call.json"))]
    intent = build_intent(static, analysis, audit,
                          source_run=str(args.output.resolve()),
                          model_calls=calls)
    _write_json(args.output / "reference_intent_v1.json", intent)
    result = {"status": "model_checked_candidate" if intent[
        "story_candidate_ready"] else "blocked",
        "story_candidate_ready": intent["story_candidate_ready"],
        "editing_candidate_ready": False,
        "validation_issues": intent["validation_issues"],
        "audit_verdicts": {row.get("id"): row.get("verdict")
                           for row in audit.get("checks") or []},
        "critical_conflict": audit.get("critical_conflict"),
        "probe_count": len(probes), "artifact_sha": intent["artifact_sha"],
        "production_release_allowed": False}
    _write_json(args.output / "result.json", result)
    return result


def abstract(args: argparse.Namespace) -> dict:
    intent = _read(args.intent)
    reference = load_reference(args.reference)
    if (intent["source_sha"] != reference["source_sha"] or
            intent["source_sha"] != sha256_file(args.video)):
        raise ValueError("source_media_sha_mismatch")
    if not intent.get("story_candidate_ready"):
        raise ValueError("parent_intent_not_ready")
    _empty(args.output)
    runner = _runner(args)
    draft = _model_call(runner, name="story_brief_abstraction",
                        prompt=ABSTRACT_PROMPT,
                        payload={"reference_intent": intent["analysis"]},
                        output=args.output, max_new_tokens=1536)
    _write_json(args.output / "draft.json", draft)
    validate_creative_boundary(draft)
    if _transfer_leaks(draft, reference, args.video):
        raise ValueError("reference_surface_leak")
    audit = _model_call(runner, name="story_brief_audit",
                        prompt=BRIEF_AUDIT_PROMPT,
                        payload={"private_intent": intent["analysis"],
                                 "public_draft": draft}, output=args.output,
                        max_new_tokens=768)
    _write_json(args.output / "brief_audit.json", audit)
    markers = [str(args.video), str(args.video.resolve()),
               reference["source_sha"]]
    markers += [str(row[key]) for collection, key in (
        ("claims", "claim_id"), ("events", "event_id"))
        for row in reference[collection]]
    markers += [str(row["object"]) for row in reference["claims"]
                if row.get("modality") == "T" and len(str(row.get(
                    "object") or "")) >= 4]
    brief = publish_story_brief(draft, intent, audit, markers)
    validate_creative_boundary(brief)
    _write_json(args.output / "creative_story_brief_v1.json", brief)
    result = {"status": "model_checked_candidate",
              "artifact_sha": brief["artifact_sha"],
              "production_release_allowed": False}
    _write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "abstract"),
                        required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.stage == "extract" and args.static is None:
        parser.error("--static is required for extract")
    if args.stage == "abstract" and args.intent is None:
        parser.error("--intent is required for abstract")
    try:
        result = extract(args) if args.stage == "extract" else abstract(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "run_failure.json", {
            "stage": args.stage, "error_type": type(exc).__name__,
            "reason": str(exc), "new_output_required_for_retry": True})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
