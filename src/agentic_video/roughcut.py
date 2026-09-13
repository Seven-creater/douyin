"""V5 w15 素材侧最小闭环。

One source scope, explicit stages, one content master, two audio derivatives,
and a final acceptance gate.  This module intentionally does not invoke the
general four-slot story template.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path

from src.config import AppConfig, repo_root

logger = logging.getLogger(__name__)
ROUGHCAST_PROGRAM_SOURCE = "roughcut_spec"
FAILURE_CLASSES = {"content", "scope", "verification", "infrastructure", "audio", "budget"}


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_spec(spec: Path | dict) -> tuple[dict, str]:
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.is_absolute():
            path = repo_root() / path
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = deepcopy(spec)
    if not isinstance(payload, dict) or payload.get("spec_version") != "roughcut_v2":
        raise ValueError("roughcut spec_version must be roughcut_v2")
    required = {"source", "source_scope", "focus_utterance", "stages", "duration", "audio_variants"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"roughcut spec missing: {missing}")
    scope = payload["source_scope"]
    if not isinstance(scope, dict) or float(scope.get("end_s", 0)) <= float(scope.get("start_s", 0)):
        raise ValueError("roughcut source_scope invalid")
    duration = payload["duration"]
    if float(duration.get("max_s", 0)) <= float(duration.get("min_s", 0)):
        raise ValueError("roughcut duration policy invalid")
    if set(payload["audio_variants"]) != {"source_only", "bgm_mix"}:
        raise ValueError("roughcut must declare source_only and bgm_mix")
    if not isinstance(payload["stages"], list) or not payload["stages"]:
        raise ValueError("roughcut stages must be a non-empty list")
    for index, stage in enumerate(payload["stages"]):
        if not isinstance(stage, dict) or not str(stage.get("id") or "").strip() \
                or not str(stage.get("purpose") or "").strip():
            raise ValueError(f"roughcut stage[{index}] requires id and purpose")
    return payload, _sha256_json(payload)


def _interval(row: dict) -> tuple[float, float]:
    try:
        return float(row.get("start_s", row.get("source_start_s", 0)) or 0), float(row.get("end_s", row.get("source_end_s", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _overlap(row: dict, scope: dict) -> bool:
    start, end = _interval(row)
    return end > float(scope["start_s"]) and start < float(scope["end_s"]) and end > start


def _contained(interval: list | tuple | None, scope: dict) -> bool:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return False
    a, b = float(interval[0]), float(interval[1])
    return float(scope["start_s"]) - 1e-6 <= a < b <= float(scope["end_s"]) + 1e-6


def _text(line: dict) -> str:
    return str(line.get("original") or line.get("translation_zh") or line.get("text") or "").strip()


def _line_interval(line: dict) -> tuple[float, float] | None:
    value = line.get("utterance_interval") or line.get("required_evidence_interval")
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        value = [line.get("start_s"), line.get("end_s")]
    try:
        a, b = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (a, b) if b > a else None


def _contentful(line: dict) -> bool:
    import unicodedata
    value = _text(line)
    return bool(value) and any(ch.isalnum() or unicodedata.category(ch).startswith("L") for ch in value)


def _row_dialogue(row: dict) -> list[dict]:
    return [dict(line) for line in row.get("dialogue") or []
            if isinstance(line, dict) and _line_interval(line)]


def _pick_stage_rows(rows: list[dict], spec: dict, scope: dict) -> dict[str, dict | None]:
    """Select stage evidence deterministically; event containers may cross scope."""
    eligible = [row for row in rows if _overlap(row, scope)]
    eligible.sort(key=lambda row: (_interval(row)[0], _interval(row)[1], str(row.get("event_id") or "")))
    focus = spec.get("focus_utterance") or {}
    candidate = focus.get("candidate_interval") or [scope["start_s"], scope["end_s"]]
    core = []
    for row in eligible:
        if any(_contentful(line) and _line_interval(line)[1] > float(candidate[0]) and _line_interval(line)[0] < float(candidate[1]) for line in _row_dialogue(row)):
            core.append(row)
    if not core:
        query = str(focus.get("query") or "")
        core = [row for row in eligible if query and query in str(row.get("caption") or row.get("event_summary") or "")]
    core_row = core[0] if core else None
    core_start = _interval(core_row)[0] if core_row else float(candidate[0])
    core_end = _interval(core_row)[1] if core_row else float(candidate[1])
    before = [row for row in eligible if _interval(row)[1] <= core_start and row is not core_row]
    after = [row for row in eligible if _interval(row)[0] >= core_end and row is not core_row]
    return {"setup": before[-1] if before else None, "core_statement": core_row, "response": after[0] if after else None}


def _stage_role(stage_id: str, index: int) -> str:
    return {"setup": "hook", "core_statement": "conflict", "response": "consequence"}.get(stage_id, ("hook", "context", "consequence")[min(index, 2)])


def _source_from_row(row: dict | None, scope: dict, stage: dict) -> dict:
    if not row:
        return {"video": "", "start_s": 0.0, "end_s": 0.0, "dialogue": [], "container_interval": None, "required_evidence_interval": None}
    raw_start, raw_end = _interval(row)
    start, end = max(raw_start, float(scope["start_s"])), min(raw_end, float(scope["end_s"]))
    dialogue = _row_dialogue(row)
    valid_lines = [line for line in dialogue if _line_interval(line) and _contained(_line_interval(line), scope)]
    evidence = None
    if stage.get("id") == "core_statement":
        intervals = [_line_interval(line) for line in valid_lines if _contentful(line)]
        if intervals:
            evidence = [min(item[0] for item in intervals), max(item[1] for item in intervals)]
            start, end = min(start, evidence[0]), max(end, evidence[1])
    source = {
        "video": str(row.get("video") or ""), "video_stem": str(row.get("video_stem") or ""),
        "window_idx": row.get("window_idx"), "shot_idx": row.get("shot_idx"), "shot_indices": list(row.get("shot_indices") or []),
        "start_s": round(start, 3), "end_s": round(end, 3), "container_interval": [raw_start, raw_end],
        "event_id": str(row.get("event_id") or f"stage_{stage.get('id')}"), "caption": str(row.get("caption") or row.get("event_summary") or ""),
        "entity_ids": list(row.get("entity_ids") or []), "entity_names": list(row.get("entity_names") or []),
        "bindings": [dict(item) for item in row.get("bindings") or [] if isinstance(item, dict)], "focus_x": float(row.get("focus_x", 0.5) or 0.5),
        "dialogue": dialogue, "anchor": "coarse",
    }
    if evidence:
        # This is the immutable ASR utterance, not yet the semantic evidence
        # span.  Omni/人工核验 is the only step allowed to promote a sub-span
        # to ``required_evidence_interval``.
        source["utterance_interval"] = [round(evidence[0], 3), round(evidence[1], 3)]
        source["candidate_evidence_interval"] = [round(evidence[0], 3), round(evidence[1], 3)]
    return source


def _build_plan(rows: list[dict], spec: dict, spec_hash: str, *, theme: str, video: str) -> dict:
    source_hashes = {str(row.get("video_sha256")) for row in rows if row.get("video_sha256")}
    transcript_hashes = {str(row.get("transcript_sha256")) for row in rows if row.get("transcript_sha256")}
    if len(source_hashes) > 1 or len(transcript_hashes) > 1:
        raise ValueError("input hash mismatch across scoped index rows")
    scope = spec["source_scope"]
    selected = _pick_stage_rows(rows, spec, scope)
    slots = []
    preferred = float(spec["duration"].get("preferred_s", 22.0))
    for index, stage in enumerate(spec["stages"]):
        stage_id = str(stage.get("id") or f"stage_{index}")
        row = selected.get(stage_id)
        source = _source_from_row(row, scope, {**stage, "id": stage_id})
        required = bool(stage.get("required"))
        status = "supported" if row and source["end_s"] > source["start_s"] else "unsupported"
        slots.append({
            "slot_idx": index, "stage_id": stage_id, "role": _stage_role(stage_id, index), "required": required,
            "target_interval": [0.0, 0.0], "source": source, "status": status,
            "reason": "" if status == "supported" else ("required_stage_no_evidence" if required else "optional_stage_no_evidence"),
            "transition_reason": "opening" if index == 0 else ("cutaway" if not required else "entity_continuity"),
            "need_spec": {"required": required, "need": stage.get("purpose") or stage_id, "must_have": [stage.get("purpose") or stage_id], "must_not": [], "evidence_mode": stage.get("evidence_type") or "both", "entity_requirements": {}, "stage_id": stage_id},
        })
    active = [slot for slot in slots if slot["status"] in {"supported", "uncertain"}]
    each = preferred / max(1, len(active))
    cursor = 0.0
    for slot in slots:
        if slot["status"] == "unsupported":
            slot["target_interval"] = [round(cursor, 3), round(cursor, 3)]
        else:
            slot["target_interval"] = [round(cursor, 3), round(cursor + each, 3)]
            cursor += each
    transcript_payload = [{"utterance_id": line.get("utterance_id"), "interval": line.get("utterance_interval") or [line.get("start_s"), line.get("end_s")], "text": _text(line)}
                         for row in rows for line in _row_dialogue(row)]
    source_hash = next((str(row.get("video_sha256")) for row in rows if row.get("video_sha256")), None)
    if source_hash is None and Path(video).exists():
        digest = hashlib.sha256()
        with Path(video).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        source_hash = digest.hexdigest()
    source_hash = source_hash or _sha256_json({"video": video, "scope": scope})
    transcript_hash = next((str(row.get("transcript_sha256")) for row in rows if row.get("transcript_sha256")), _sha256_json(transcript_payload))
    return {
        "story_plan_version": "1.1", "plan_kind": "roughcut", "theme": theme.strip(), "library": str(spec["source"]), "reference_id": "roughcut_w15",
        "target_duration_s": round(preferred, 3), "duration_policy": {"preferred_s": preferred, "min_s": float(spec["duration"].get("min_s", 12.0)), "max_s": float(spec["duration"].get("max_s", 26.4)), "content_preserving": True},
        "source_scope": deepcopy(scope), "slots": slots,
        "required_unsupported": [slot["slot_idx"] for slot in slots if slot["status"] == "unsupported" and slot["required"]],
        "plan_complete": not any(slot["status"] == "unsupported" and slot["required"] for slot in slots),
        "entity_contract": {"protagonist": None, "persistence": "local_scene_ids", "relaxed": True}, "narrative_form": {"name": "explicit_stages", "slot_functions": [slot["stage_id"] for slot in slots]},
        "roughcut_spec": deepcopy(spec), "roughcut_spec_sha256": spec_hash,
        "provenance": {"program_source": ROUGHCAST_PROGRAM_SOURCE, "roughcut_spec_sha256": spec_hash, "source_sha256": source_hash, "transcript_sha256": transcript_hash, "input_hash": _sha256_json([{k: row.get(k) for k in ("video", "window_idx", "start_s", "end_s", "event_id")} for row in rows]), "stage_schema": "roughcut_v2"},
        "source_video": video,
    }


def build_roughcut_narrative(rows: list[dict], window_idx: int, reference_uri: str, spec: dict | None = None) -> dict:
    """Compatibility adapter; it no longer emits assertion_visual_payoff."""
    stages = deepcopy((spec or {}).get("stages") or [{"id": "setup"}, {"id": "core_statement", "required": True}])
    return {"program_version": "roughcut_v2", "reference": {"id": f"roughcut_w{window_idx}", "uri": reference_uri, "sha256": "", "duration_s": 40.0, "fps": 24.0}, "stages": stages, "roughcut_spec": deepcopy(spec) if spec else {"stages": stages}, "provenance": {"program_source": ROUGHCAST_PROGRAM_SOURCE}}


def _failure(output_dir: Path, failure_class: str, reasons: list[str], *, debug: Path | None = None) -> None:
    payload = {"passed": False, "failure_class": failure_class if failure_class in FAILURE_CLASSES else "infrastructure", "reasons": reasons, "delivery": "blocked", "debug_preview": str(debug) if debug else None}
    (output_dir / "acceptance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "render_manifest.json").write_text(json.dumps({"delivery": "blocked", "acceptance": payload}, ensure_ascii=False, indent=2), encoding="utf-8")


def run_roughcut(cfg: AppConfig, source: str | None = None, window_idx: int | None = None, *, theme: str = "", output_dir: Path, target_duration_s: float | None = None, runner=None, force: bool = False, spec: Path | dict | None = None, allow_unverified: bool = False) -> Path:
    """Run the complete V5 roughcut gate. ``allow_unverified`` is test-only."""
    from src.agentic_video.recipe_v2 import new_recipe
    from src.agentic_video.renderer import derive_audio_variants, render_recipe
    from src.agentic_video.story_planner import fit_slot_intervals, story_plan_execution_inputs, validate_story_plan
    from src.agentic_video.verify_slots import blind_video_check, deterministic_story_check, localize_coarse_slots, verify_slots
    from src.library.build_index import load_index

    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if not force and (output_dir / "rendered.mp4").exists():
        try:
            prior = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
        if prior.get("passed") is True:
            return output_dir / "rendered.mp4"
    spec = spec or repo_root() / "config" / "roughcuts" / "lxh1_w15.json"
    rough_spec, spec_hash = _read_spec(spec)
    if target_duration_s is not None:
        requested = float(target_duration_s)
        if not float(rough_spec["duration"].get("min_s", 12.0)) <= requested <= float(rough_spec["duration"].get("max_s", 26.4)):
            _failure(output_dir, "budget", ["target_duration_outside_spec"])
            raise ValueError("target_duration_s outside RoughcutSpec duration policy")
        rough_spec["duration"]["preferred_s"] = requested
        spec_hash = _sha256_json(rough_spec)
    source = str(rough_spec["source"])
    if window_idx is not None and int(window_idx) != int(rough_spec.get("base_window", window_idx)):
        raise ValueError("window argument conflicts with RoughcutSpec base_window")
    if runner is None and not allow_unverified:
        _failure(output_dir, "infrastructure", ["runner_missing"]); raise RuntimeError("roughcut blocked: runner_missing")
    try:
        rows, _embeddings = load_index(cfg)
    except Exception as exc:
        _failure(output_dir, "infrastructure", [f"index_load:{type(exc).__name__}:{exc}"])
        raise
    scope = rough_spec["source_scope"]
    scoped = [row for row in rows if str(row.get("video_stem") or "").startswith(f"{source}__") and _overlap(row, scope)]
    if not scoped:
        _failure(output_dir, "content", ["no_scoped_candidates"]); raise RuntimeError("roughcut blocked: no scoped candidates")
    scoped.sort(key=lambda row: _interval(row)[0]); video = str(scoped[0].get("video") or "")
    try:
        plan = _build_plan(scoped, rough_spec, spec_hash, theme=theme or str((rough_spec.get("focus_utterance") or {}).get("query") or ""), video=video)
    except Exception as exc:
        _failure(output_dir, "infrastructure", [f"input_hash:{type(exc).__name__}:{exc}"])
        raise
    plan["story_plan_sha256"] = _sha256_json(plan)
    (output_dir / "roughcut_spec.json").write_text(json.dumps(rough_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "story_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_errors = validate_story_plan(plan)
    if plan_errors:
        _failure(output_dir, "scope" if any("scope" in e for e in plan_errors) else "content", plan_errors); raise RuntimeError("roughcut plan invalid")
    try:
        localize_coarse_slots(cfg, plan, runner=runner, output_dir=output_dir)
    except Exception as exc:
        _failure(output_dir, "infrastructure", [f"localize_error:{type(exc).__name__}:{exc}"]); raise
    localize_log = output_dir / "localize_log.json"
    if localize_log.exists():
        try:
            rows_log = json.loads(localize_log.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows_log = []
        infra_rows = [row for row in rows_log if row.get("error") and "ambiguous" not in str(row.get("error"))]
        if infra_rows:
            _failure(output_dir, "infrastructure", ["localize_runner_error"])
            raise RuntimeError("roughcut blocked: localize infrastructure failure")
    for slot in plan["slots"]:
        if slot["status"] in {"supported", "uncertain"}:
            src = slot["source"]; src["actual_rendered_source_interval"] = [src.get("start_s"), src.get("end_s")]
            evidence = src.get("required_evidence_interval") or src.get("evidence_interval")
            if evidence and not _contained(evidence, scope):
                slot["status"], slot["reason"] = "unsupported", "required_evidence_outside_scope"
    if any(slot["status"] == "unsupported" and slot["required"] for slot in plan["slots"]):
        required_slots = [slot for slot in plan["slots"] if slot["status"] == "unsupported" and slot["required"]]
        reasons = [f"required_stage:{slot['stage_id']}:{slot.get('reason')}" for slot in required_slots]
        failure_class = "verification" if any("timebase" in str(slot.get("reason")) or "interval_outside" in str(slot.get("reason")) for slot in required_slots) else "content"
        _failure(output_dir, failure_class, reasons); raise RuntimeError("roughcut blocked: required evidence unavailable")
    try:
        verification = verify_slots(cfg, plan, runner=runner, context_pad_s=0.0,
                                    clip_root=output_dir / "verify_clips")
    except Exception as exc:
        _failure(output_dir, "infrastructure", [f"verify_error:{type(exc).__name__}:{exc}"]); raise
    (output_dir / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    if verification.get("failed_slots") or verification.get("uncertain_slots"):
        _failure(output_dir, "verification", [f"verify_failed:{verification.get('failed_slots')}", f"verify_uncertain:{verification.get('uncertain_slots')}"])
        raise RuntimeError("roughcut blocked: verification")
    try:
        fit_slot_intervals(plan, mode="content_preserving")
    except ValueError as exc:
        _failure(output_dir, "budget", [str(exc)]); raise
    plan_errors = validate_story_plan(plan)
    if plan_errors:
        _failure(output_dir, "scope", plan_errors); raise RuntimeError("roughcut blocked after duration fit")
    plan_for_hash = deepcopy(plan)
    plan_for_hash.pop("story_plan_sha256", None)
    plan_for_hash.pop("retrieval_sha256", None)
    plan["story_plan_sha256"] = _sha256_json(plan_for_hash)
    det = deterministic_story_check(plan)
    (output_dir / "deterministic_check.json").write_text(json.dumps(det, ensure_ascii=False, indent=2), encoding="utf-8")
    if not det.get("passed"):
        _failure(output_dir, "content", det.get("violations") or ["deterministic_story_check"]); raise RuntimeError("roughcut blocked: deterministic check")
    if allow_unverified:
        _failure(output_dir, "infrastructure", ["allow_unverified_offline_test"]); raise RuntimeError("roughcut is not deliverable when allow_unverified=true")
    (output_dir / "story_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    recipe = new_recipe(reference_id=plan["reference_id"], reference_uri=video,
                        sha256=str(plan["provenance"].get("source_sha256") or ""),
                        duration_s=float(plan["target_duration_s"]), fps=24.0)
    bgm_setting = str((cfg.library.get("narrative_render") or {}).get("bgm_path") or "data/library/bgm/template_7682.m4a")
    bgm_path = Path(bgm_setting) if Path(bgm_setting).is_absolute() else repo_root() / bgm_setting
    bgm_hash = hashlib.sha256(bgm_path.read_bytes()).hexdigest() if bgm_path.exists() else None
    asset_plan, retrieval = story_plan_execution_inputs(plan, recipe, audio_policy={"audio_mode": "source", "bgm_volume": 0.25, "bgm_path": str(bgm_path), "bgm_sha256": bgm_hash})
    retrieval_hash = _sha256_json(retrieval)
    plan["retrieval_sha256"] = retrieval_hash
    asset_plan["story_plan_sha256"] = plan["story_plan_sha256"]
    asset_plan["retrieval_sha256"] = retrieval_hash
    asset_plan["source_sha256"] = plan["provenance"].get("source_sha256")
    asset_plan["transcript_sha256"] = plan["provenance"].get("transcript_sha256")
    try:
        content_master = render_recipe(cfg, recipe, asset_plan, retrieval,
                                       output_dir / "content_master_render", force=True)
    except Exception as exc:
        _failure(output_dir, "infrastructure", [f"render_master:{type(exc).__name__}:{exc}"])
        raise
    master_path = output_dir / "content_master.mp4"; shutil.copy2(content_master, master_path)
    try:
        variants = derive_audio_variants(cfg, master_path, output_dir / "variants", bgm_path=bgm_path, mix_volume=0.25, duration_s=float(plan["target_duration_s"]))
    except FileNotFoundError as exc:
        _failure(output_dir, "audio", [f"bgm_missing:{exc}"]); raise
    except Exception as exc:
        _failure(output_dir, "audio", [f"audio_derivation:{type(exc).__name__}:{exc}"]); raise
    blinds = {}
    for name, path in variants.items():
        try:
            blinds[name] = blind_video_check(path, runner=runner, roughcut=True)
        except Exception as exc:
            _failure(output_dir, "infrastructure", [f"blind_{name}:{type(exc).__name__}:{exc}"]); raise
    (output_dir / "blind_review.json").write_text(json.dumps(blinds, ensure_ascii=False, indent=2), encoding="utf-8")
    reasons = []
    for name, report in blinds.items():
        if not report.get("parsed") or report.get("consistent_protagonist") is not True:
            reasons.append(f"blind_{name}_not_comprehensible")
        if report.get("speech_clear") is not True:
            reasons.append(f"blind_{name}_speech_unclear")
        if name == "bgm_mix" and report.get("music_present") is not True:
            reasons.append("blind_bgm_mix_music_missing")
    if reasons:
        _failure(output_dir, "content", reasons); raise RuntimeError("roughcut blocked: blind acceptance")
    from src.agentic_video.renderer import _sha256_file
    audio_manifest = json.loads((output_dir / "variants" / "audio_variants_manifest.json").read_text(encoding="utf-8"))
    shared = {"story_plan_sha256": plan["story_plan_sha256"], "retrieval_sha256": retrieval_hash,
              "content_master_sha256": _sha256_file(master_path),
              "content_frames_framemd5": audio_manifest["source_only"]["frame_md5"]}
    for variant_name in ("source_only", "bgm_mix"):
        variant_manifest = {**shared, "audio_variant": variant_name,
                            "audio_sha256": audio_manifest[variant_name]["sha256"]}
        (output_dir / "variants" / variant_name / "manifest.json").write_text(
            json.dumps(variant_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    acceptance = {"passed": True, "failure_class": None, "reasons": [], "delivery": "passed", "variants": {key: str(value) for key, value in variants.items()}, "content_master": str(master_path), **shared}
    (output_dir / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(variants["bgm_mix"], output_dir / "rendered.mp4")
    (output_dir / "render_manifest.json").write_text(json.dumps({"delivery": "passed", "primary_variant": "bgm_mix", "variants": {key: str(value) for key, value in variants.items()}, **shared, "acceptance": str(output_dir / "acceptance.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir / "rendered.mp4"
