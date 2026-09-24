"""Isolated Theme -> Story trial from a public story-intent brief."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.creative_pipeline.intent_trial import (
    SOURCE_COPY_PROMPT, has_reference_surface, prompts_for_brief,
    review_stories, select_themes,
    validate_source_copy_audit, validate_story, validate_themes,
)
from src.agentic_video.creative_pipeline.real_text_baseline import _ask
from src.agentic_video.manifest import json_hash
from src.agentic_video.transfer_eval_v2 import validate_acceptance
from src.agentic_video.reference_readout import (
    _model_call, _write_json, load_reference,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(args):
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus
    set_visible_gpus(args.gpu_pair)
    return OmniRunner(load_config(args.config).perception["omni"])


def _markers(args) -> list[str]:
    reference = load_reference(args.reference)
    values = [reference["source_sha"], str(args.video),
              str(args.video.resolve())]
    values += [str(row[key]) for collection, key in (
        ("claims", "claim_id"), ("events", "event_id"))
        for row in reference[collection]]
    values += [str(row["shot_id"]) for row in reference[
        "shot_storyboard"]["shot_cards"]]
    values += [str(row["object"]) for row in reference["claims"]
               if row.get("modality") == "T" and len(str(row.get(
                   "object") or "")) >= 4]
    return values


def _fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("trial_output_not_empty")
    path.mkdir(parents=True, exist_ok=True)


def _private_kernel(args, brief: dict) -> dict | None:
    if brief["schema_version"] != "creative_story_brief_v2":
        return None
    if args.private_kernel is None:
        raise ValueError("private_kernel_required_for_v2_source_audit")
    record = _read(args.private_kernel)
    if (record.get("schema_version") != "transfer_kernel_record_v1" or
            record.get("artifact_sha") != brief["parent_kernel_sha"] or
            record["artifact_sha"] != json_hash({
                key: value for key, value in record.items()
                if key != "artifact_sha"}) or
            record.get("source_media_sha") != load_reference(
                args.reference)["source_sha"] or
            record.get("status") != "private_model_checked"):
        raise ValueError("private_kernel_lineage_invalid")
    return record


def _trial_acceptance(args, brief: dict, record: dict | None) -> dict | None:
    if brief["schema_version"] != "creative_story_brief_v2":
        return None
    if args.trial_acceptance is None or record is None:
        raise ValueError("trial_acceptance_required_for_v2")
    path = args.trial_acceptance
    acceptance = _read(path)
    parent = path.parent
    result = _read(parent / "result.json")
    attempt = result.get("attempt")
    if (result.get("status") != "model_checked_candidate" or
            type(attempt) is not int or attempt not in (1, 2) or
            result.get("acceptance_sha") != acceptance.get("artifact_sha")):
        raise ValueError("trial_acceptance_result_invalid")
    prefix = f"attempt_{attempt:02d}"
    validate_acceptance(
        brief, acceptance, kernel_sha=record["artifact_sha"],
        audit=record["audit"],
        calibration=_read(parent / "calibration_report.json"),
        frozen_badcase=_read(parent / "frozen_badcase_regression.json"),
        mapping=_read(parent / f"{prefix}_mapping.json"),
        stance=_read(parent / f"{prefix}_stance.json"),
        regression=_read(parent / f"{prefix}_regression.json"))
    return acceptance


def _source_audit(runner, output: Path, record: dict,
                  candidates: list[dict]) -> dict:
    audit = _model_call(runner, name="source_copy_audit",
                        prompt=SOURCE_COPY_PROMPT,
                        payload={"private_source_bindings": [
                            {key: row[key] for key in (
                                "role_id", "source_binding")}
                            for row in record["kernel"]["roles"]],
                            "candidates": candidates}, output=output,
                        max_new_tokens=1024)
    _write_json(output / "source_copy_audit.json", audit)
    errors = validate_source_copy_audit(audit, [
        row["candidate_id"] for row in candidates])
    if errors:
        raise ValueError(",".join(errors))
    return audit


def themes(args) -> dict:
    brief = _read(args.brief)
    theme_prompt, critique_prompt, _, _ = prompts_for_brief(brief)
    record = _private_kernel(args, brief)
    acceptance = _trial_acceptance(args, brief, record)
    markers = _markers(args)
    _fresh(args.output)
    runner = _runner(args)
    batch = _ask(runner, args.output, "01_themes", theme_prompt,
                 {"creative_story_brief": brief}, 3072)
    problems = validate_themes(batch)
    validate_creative_boundary(batch)
    if has_reference_surface(batch, markers):
        problems.append("reference_surface_leak")
    if record is not None and not problems and len({
            row["domain"].strip().casefold() for row in batch["themes"]}) != 3:
        problems.append("theme_domains_not_distinct")
    _write_json(args.output / "theme_batch.json", batch)
    if problems:
        raise ValueError(",".join(problems))
    critique = _ask(runner, args.output, "02_theme_critique",
                    critique_prompt,
                    {"creative_story_brief": brief,
                     "theme_batch": batch}, 1536)
    _write_json(args.output / "theme_critique.json", critique)
    select_themes(critique)
    combined = copy.deepcopy(critique)
    if record is not None:
        audit = _source_audit(runner, args.output, record, [
            {"candidate_id": row["theme_id"], "candidate": row}
            for row in batch["themes"]])
        copied = {row["candidate_id"] for row in audit["checks"]
                  if row["surface_copy"]}
        for row in combined["checks"]:
            if row["theme_id"] in copied:
                row["original"] = False
    selected = select_themes(combined)
    result = {"schema_version": "intent_theme_selection_v1",
              "brief_sha": brief["artifact_sha"],
              "theme_batch_sha": json_hash(batch),
              "selected_theme_ids": selected,
              "status": "model_checked_candidate" if selected else "blocked",
              "production_committed": False}
    if acceptance is not None:
        result["trial_acceptance_sha"] = acceptance["artifact_sha"]
    _write_json(args.output / "selection.json", result)
    return result


def story(args) -> dict:
    brief, selection = _read(args.brief), _read(args.selection)
    _, _, story_prompt, _ = prompts_for_brief(brief)
    acceptance = _trial_acceptance(args, brief, _private_kernel(args, brief))
    if acceptance is not None and selection.get(
            "trial_acceptance_sha") != acceptance["artifact_sha"]:
        raise ValueError("selection_trial_acceptance_invalid")
    if selection["brief_sha"] != brief["artifact_sha"] or args.theme_id not in (
            selection["selected_theme_ids"]):
        raise ValueError("theme_not_selected")
    batch = _read(args.selection.parent / "theme_batch.json")
    theme = next(row for row in batch["themes"] if row["theme_id"] ==
                 args.theme_id)
    _fresh(args.output)
    candidate = _ask(_runner(args), args.output, "story", story_prompt,
                     {"creative_story_brief": brief,
                      "selected_theme": theme}, 3072)
    _write_json(args.output / "candidate.json", candidate)
    errors = validate_story(candidate, args.theme_id)
    if has_reference_surface(candidate, _markers(args)):
        errors.append("reference_surface_leak")
    result = {"status": "candidate" if not errors else "blocked",
              "theme_id": args.theme_id,
              "candidate_sha": json_hash(candidate), "issues": errors,
              "production_committed": False}
    _write_json(args.output / "result.json", result)
    return result


def review(args) -> dict:
    brief, selection = _read(args.brief), _read(args.selection)
    _, _, _, critique_prompt = prompts_for_brief(brief)
    record = _private_kernel(args, brief)
    acceptance = _trial_acceptance(args, brief, record)
    if acceptance is not None and selection.get(
            "trial_acceptance_sha") != acceptance["artifact_sha"]:
        raise ValueError("selection_trial_acceptance_invalid")
    if selection["brief_sha"] != brief["artifact_sha"]:
        raise ValueError("selection_parent_invalid")
    theme_ids = selection["selected_theme_ids"]
    if len(args.story_dirs) != len(theme_ids):
        raise ValueError("story_count_invalid")
    stories = []
    for theme_id, directory in zip(theme_ids, args.story_dirs):
        result = _read(directory / "result.json")
        candidate = _read(directory / "candidate.json")
        if result["status"] != "candidate" or candidate["theme_id"] != theme_id:
            raise ValueError("story_candidate_invalid")
        stories.append(candidate)
    _fresh(args.output)
    runner = _runner(args)
    critique = _ask(runner, args.output, "story_critique",
                    critique_prompt,
                    {"creative_story_brief": brief,
                     "stories": stories}, 1536)
    _write_json(args.output / "story_critique.json", critique)
    review_stories(critique, theme_ids)
    combined = copy.deepcopy(critique)
    if record is not None:
        audit = _source_audit(runner, args.output, record, [
            {"candidate_id": row["theme_id"], "candidate": row}
            for row in stories])
        copied = {row["candidate_id"] for row in audit["checks"]
                  if row["surface_copy"]}
        for row in combined["checks"]:
            if row["theme_id"] in copied:
                row["original"] = False
    result = review_stories(combined, theme_ids)
    result.update({"schema_version": "intent_story_trial_v1",
                   "brief_sha": brief["artifact_sha"],
                   "story_shas": [json_hash(item) for item in stories]})
    _write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("themes", "story", "review"),
                        required=True)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--private-kernel", type=Path)
    parser.add_argument("--trial-acceptance", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--theme-id")
    parser.add_argument("--story-dir", dest="story_dirs", type=Path,
                        action="append", default=[])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.stage != "themes" and args.selection is None:
        parser.error("--selection is required for story/review")
    if args.stage == "story" and not args.theme_id:
        parser.error("--theme-id is required for story")
    try:
        result = {"themes": themes, "story": story,
                  "review": review}[args.stage](args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "run_failure.json", {
            "stage": args.stage, "error_type": type(exc).__name__,
            "reason": str(exc), "new_output_required_for_retry": True})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
