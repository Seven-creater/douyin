"""One isolated model shard or deterministic compile; safe to run AV shards in parallel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.reference_readout import _model_call, _write_json
from src.agentic_video.reference_transfer_v2 import (
    build_reference_blueprint, inspect_bgm_asset,
)
from src.agentic_video.reference_transfer_v2_shards import (
    GLOBAL_AUDIT_PROMPT, GLOBAL_STORY_PROMPT, SECTION_AUDIT_PROMPT,
    SECTION_PROMPT, global_audit_ids, global_story_payload,
    section_audit_ids, section_payload,
)
from src.perception.omni_runner import cut_clip


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(args):
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus
    set_visible_gpus(args.gpu_pair)
    return OmniRunner(load_config(args.config).perception["omni"])


def _call(args, name: str, prompt: str, payload: dict,
          *, media: Path | None, fps: float | None,
          max_tokens: int) -> dict:
    output = args.output / name
    if (output / "calls").exists() or (output / "result.json").exists():
        raise FileExistsError("shard_output_not_empty")
    output.mkdir(parents=True, exist_ok=True)
    return _model_call(_runner(args), name=name, prompt=prompt,
                       payload=payload, output=output, media=media,
                       channel="AV" if media is not None else "FUSION",
                       fps=fps, max_new_tokens=max_tokens)


def _section(args, static: dict, local: dict, *, audit: bool) -> None:
    sid = args.section
    payload = section_payload(static, local, sid)
    origin, end = payload["interval"]
    output = args.output / (f"audit_{sid}" if audit else f"analysis_{sid}")
    output.mkdir(parents=True, exist_ok=True)
    clip = cut_clip("ffmpeg", args.video, output / "clip",
                    start_s=origin, end_s=end, include_audio=True)
    payload["interval"] = [0.0, end - origin]
    if audit:
        source = _read(args.output / f"analysis_{sid}" / "result.json")
        payload["analysis"] = source
        payload["audit_ids"] = section_audit_ids(static, sid)
        name, prompt = f"audit_{sid}", SECTION_AUDIT_PROMPT
    else:
        name, prompt = f"analysis_{sid}", SECTION_PROMPT
    result = _call(args, name, prompt, payload, media=clip,
                   fps=4.0 if sid != "section_01" else 2.0,
                   max_tokens=4096 if not audit else 3072)
    _write_json(output / "result.json", result)


def _global_story(args, static: dict) -> None:
    sections = [_read(args.output / f"analysis_section_0{i}" / "result.json")
                for i in range(1, 4)]
    payload = global_story_payload(static, sections)
    result = _call(args, "global_story", GLOBAL_STORY_PROMPT, payload,
                   media=None, fps=None, max_tokens=3072)
    _write_json(args.output / "global_story" / "result.json", result)


def _global_audit(args, static: dict) -> None:
    story = _read(args.output / "global_story" / "result.json")
    payload = global_story_payload(static, [
        _read(args.output / f"analysis_section_0{i}" / "result.json")
        for i in range(1, 4)])
    payload.update({"interval": [0.0, static["duration_s"]],
                    "global_story": story,
                    "audit_ids": global_audit_ids(static)})
    result = _call(args, "global_audit", GLOBAL_AUDIT_PROMPT, payload,
                   media=args.video, fps=1.5, max_tokens=2048)
    _write_json(args.output / "global_audit" / "result.json", result)


def _compile(args, static: dict, local: dict) -> None:
    sections = [_read(args.output / f"analysis_section_0{i}" / "result.json")
                for i in range(1, 4)]
    global_story = _read(args.output / "global_story" / "result.json")
    shots = [row for section in sections for row in section.get("shots") or []]
    edges = ([row for section in sections for row in section.get("edges") or []] +
             list(global_story.get("cross_edges") or []))
    shot_map = {row["shot_id"]: row for row in shots}
    edge_map = {(row["from_shot_id"], row["to_shot_id"]): row for row in edges}
    ordered_shots = [shot_map[row["shot_id"]] for row in static["shots"]]
    ordered_edges = [edge_map[(a["shot_id"], b["shot_id"])]
                     for a, b in zip(static["shots"], static["shots"][1:])]
    analysis = {"schema_version": "reference_blueprint_analysis_v2",
                **{key: global_story.get(key) for key in (
                    "theme_stance", "viewer_change", "ending", "narrative_units")},
                "shots": ordered_shots, "edges": ordered_edges,
                "unknowns": [item for section in sections for item in
                             section.get("unknowns") or []] +
                            list(global_story.get("unknowns") or [])}
    checks = []
    checks += _read(args.output / "global_audit" / "result.json").get("checks") or []
    for i in range(1, 4):
        checks += _read(args.output / f"audit_section_0{i}" /
                        "result.json").get("checks") or []
    audit = {"schema_version": "reference_blueprint_audit_v2",
             "checks": checks, "media_truth_guaranteed": False}
    blueprint = build_reference_blueprint(static, local, analysis, audit)
    blueprint["audio"]["bgm_asset"] = inspect_bgm_asset(args.video, args.bgm)
    blueprint["parent_source_run"] = str(args.source_run)
    blueprint["parent_source_run_sha256"] = sha256_file(
        args.source_run / "run_failure.json")
    blueprint["artifact_sha"] = json_hash({key: value for key, value in
                                           blueprint.items() if key != "artifact_sha"})
    _write_json(args.output / "analysis_compiled.json", analysis)
    _write_json(args.output / "audit_compiled.json", audit)
    _write_json(args.output / "reference_blueprint_v2.json", blueprint)
    _write_json(args.output / "result.json", {
        "status": "model_checked_candidate" if blueprint[
            "story_candidate_ready"] else "blocked",
        "story_candidate_ready": blueprint["story_candidate_ready"],
        "editing_candidate_ready": blueprint["editing_candidate_ready"],
        "validation_issues": blueprint["validation_issues"],
        "artifact_sha": blueprint["artifact_sha"],
        "production_release_allowed": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("section-analysis", "section-audit",
                                         "global-story", "global-audit", "compile"),
                        required=True)
    parser.add_argument("--section", choices=("section_01", "section_02",
                                              "section_03"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bgm", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.job.startswith("section") and not args.section:
        parser.error("--section is required for section jobs")
    static = _read(args.static)
    local = _read(args.local)
    if static["source_sha"] != sha256_file(args.video):
        raise ValueError("source_media_sha_mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.job == "section-analysis":
            _section(args, static, local, audit=False)
        elif args.job == "section-audit":
            _section(args, static, local, audit=True)
        elif args.job == "global-story":
            _global_story(args, static)
        elif args.job == "global-audit":
            _global_audit(args, static)
        else:
            _compile(args, static, local)
    except Exception as exc:
        _write_json(args.output / f"failure_{args.job}_{args.section or 'all'}.json",
                    {"error_type": type(exc).__name__, "reason": str(exc),
                     "source_run_unchanged": True})
        raise
    print(json.dumps({"job": args.job, "section": args.section,
                      "status": "completed"}))


if __name__ == "__main__":
    main()
