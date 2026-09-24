"""One-off v4 understanding -> abstract brief -> existing Theme contract trial.

This does not publish to the R2-D workspace or continue to Story/Screenplay.
Run --stage abstract, inspect brief.json, then run --stage theme.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary, validate_theme_structure,
)
from src.agentic_video.creative_pipeline.real_text_baseline import THEME_PROMPT, _ask
from src.agentic_video.creative_structure_v1.freeze import validate_frozen_spec
from src.agentic_video.manifest import json_hash
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference,
)


ABSTRACT_PROMPT = """Extract a transferable creative brief from this
provisional reference reading. The reading can contain mistaken visual details,
so abstract the relationships between initial information, later evidence,
viewer interpretation and the ending; do not promote inferred action, cause or
formal outcome to verified fact. The editing data only measures temporal
organization; audio onset is not a verified beat. Do not assume a particular
story template or set of roles. Do not include source identities, bodily
traits, sport/activity, places, objects, quoted text, source IDs, exact times,
paths or specific original examples. A different domain must be possible.
Return JSON only with exactly:
{"schema_version":"reference_transfer_brief_trial_v1",
"communicative_goal":"...","information_sequence":["..."],
"ending_relation":"... or unknown","tone":"... or unknown",
"editing_organization":["..."],"free_slots":["..."],
"uncertainties":["..."]}
Input: """

BRIEF_KEYS = {"schema_version", "communicative_goal", "information_sequence",
              "ending_relation", "tone", "editing_organization",
              "free_slots", "uncertainties"}


def brief_for_theme(brief: dict) -> dict:
    if set(brief) != BRIEF_KEYS or brief.get("schema_version") != (
            "reference_transfer_brief_trial_v1"):
        raise ValueError("brief_schema_invalid")
    if not isinstance(brief["information_sequence"], list) or not brief[
            "information_sequence"]:
        raise ValueError("brief_information_sequence_missing")
    value = {"communicative_goal": brief["communicative_goal"],
             "information_sequence": brief["information_sequence"],
             "ending_relation": brief["ending_relation"],
             "tone": brief["tone"],
             "editing_organization": brief["editing_organization"],
             "free_slots": brief["free_slots"],
             "instruction": "Use a different domain and causal events; preserve the abstract communicative and temporal relationships."}
    validate_creative_boundary(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("abstract", "theme"), required=True)
    parser.add_argument("--v4-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    reference = load_reference(args.reference)
    result = json.loads((args.v4_run / "reference_reading_v4.json").read_text(
        encoding="utf-8"))
    if (result["source_sha"] != reference["source_sha"] or
            result["production_release_allowed"] is not False):
        raise ValueError("v4_source_or_status_invalid")
    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    validate_frozen_spec(structure)
    set_visible_gpus(args.gpu_pair)
    runner = OmniRunner(load_config(args.config).perception["omni"])
    if args.stage == "abstract":
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output_directory_not_empty")
        reading = json.loads((args.v4_run / "reading_after_local.json").read_text(
            encoding="utf-8"))
        static = json.loads((args.v4_run / "static_review.json").read_text(
            encoding="utf-8"))
        value = _model_call(runner, name="abstract_brief", prompt=ABSTRACT_PROMPT,
                            payload={"reading": reading,
                                     "measured_editing": static["measured_editing"][
                                         "structural_grammar"],
                                     "audio_beat_status": static["audio"][
                                         "beat_synced_status"]},
                            output=args.output, max_new_tokens=2048)
        brief_for_theme(value)
        leaks = _transfer_leaks(value, reference, args.video)
        _write_json(args.output / "brief_validation.json", {
            "status": "BLOCKED" if leaks else "CANDIDATE",
            "exact_source_leak_rule_indices": leaks,
            "human_surface_review_required": True})
        _write_json(args.output / "brief.json", value)
        print(json.dumps({"stage": "abstract", "status": "CANDIDATE" if not leaks
                          else "BLOCKED", "brief_sha": json_hash(value)}))
        return
    validation = json.loads((args.output / "brief_validation.json").read_text(
        encoding="utf-8"))
    if validation["status"] != "CANDIDATE" or (
            args.output / "01_theme" / "request.json").exists():
        raise ValueError("brief_blocked_or_theme_already_called")
    brief = json.loads((args.output / "brief.json").read_text(encoding="utf-8"))
    if _transfer_leaks(brief, reference, args.video):
        raise ValueError("reference_surface_leak")
    user_brief = brief_for_theme(brief)
    draft = _ask(runner, args.output, "01_theme", THEME_PROMPT,
                 {"creative_structure_spec": structure,
                  "user_brief": user_brief}, 2048)
    theme = {"schema_version": "theme_candidate_v1",
             "theme_id": "V4_TRIAL_THEME_01",
             "parent_structure_sha": structure["artifact_sha"], **draft}
    validate_theme_structure(theme, structure_sha=structure["artifact_sha"])
    leaks = _transfer_leaks(theme, reference, args.video)
    _write_json(args.output / "01_theme" / "candidate.json", theme)
    _write_json(args.output / "01_theme" / "validation.json", {
        "status": "BLOCKED" if leaks else "CANDIDATE",
        "structure_contract_passed": True,
        "exact_source_leak_rule_indices": leaks,
        "semantic_quality_review_required": True,
        "production_committed": False})
    print(json.dumps({"stage": "theme", "status": "BLOCKED" if leaks
                      else "CANDIDATE", "theme_sha": json_hash(theme)}))


if __name__ == "__main__":
    main()
