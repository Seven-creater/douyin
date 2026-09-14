"""V6.1 dual-resolution Evidence diagnostic experiment."""
from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.agentic_video.evidence_matcher import (
    build_v61_edit_plan, link_subject_tracks, match_evidence_to_pattern,
    oracle_pattern_matches, validate_pattern_contract,
)
from src.agentic_video.evidence_pipeline import blind_review_variants
from src.agentic_video.evidence_units import EvidenceUnitV2
from src.agentic_video.long_video import resolve_source_video
from src.agentic_video.renderer import render_micro_montage
from src.config import AppConfig, repo_root
from src.perception.detect_shots import detect_scoped_shots
from src.perception.dual_resolution_miner import run_dual_resolution_mining


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def read_diagnostic_spec(spec: Path | dict) -> tuple[dict[str, Any], str]:
    payload = (json.loads(_resolve_path(spec).read_text(encoding="utf-8"))
               if isinstance(spec, (str, Path)) else deepcopy(spec))
    if not isinstance(payload, dict) or payload.get("spec_version") != "evidence_diagnostic_v1":
        raise ValueError("diagnostic spec_version must be evidence_diagnostic_v1")
    for key in ("source", "reference", "controls", "pattern", "perception", "duration"):
        if key not in payload:
            raise ValueError(f"diagnostic spec missing: {key}")
    controls = payload["controls"]
    for name in ("w15", "action"):
        scope = controls.get(name) or {}
        if float(scope.get("end_s", 0)) <= float(scope.get("start_s", -1)):
            raise ValueError(f"diagnostic control scope invalid: {name}")
    reasons = validate_pattern_contract(payload["pattern"])
    if reasons:
        raise ValueError("policy_contract_conflict:" + ",".join(reasons))
    perception = payload["perception"]
    if not 0 < float(perception.get("coarse_fps", 0)) < float(perception.get("dense_fps", 0)):
        raise ValueError("diagnostic coarse/dense fps invalid")
    duration = payload["duration"]
    if not 0 < float(duration.get("min_s", 0)) <= float(duration.get("max_s", 0)):
        raise ValueError("diagnostic duration policy invalid")
    return payload, _sha256_json(payload)


def oracle_bank(oracle: dict[str, Any], *, source_video: Path,
                scope: tuple[float, float]) -> dict[str, Any]:
    units = []
    for row in oracle.get("facts") or []:
        core = tuple(float(value) for value in row["core_interval"])
        container = tuple(float(value) for value in
                          (row.get("container_interval") or core))
        if not scope[0] <= core[0] < core[1] <= scope[1]:
            raise ValueError(f"oracle fact outside action scope: {row.get('id')}")
        unit = EvidenceUnitV2(
            id=str(row["id"]), core_interval=core, container_interval=container,
            observation=dict(row.get("observation") or {}),
            source_form=str(row["source_form"]),
            attributes=tuple(str(value) for value in row.get("attributes") or []),
            visual_strength=float(row.get("visual_strength", 1.0)),
            confidence=float(row.get("confidence", 1.0)),
            subject_track_id=str(row.get("subject_track_id") or "") or None,
            source_video=str(source_video),
            sampling_provenance={"oracle": True},
            metadata={"oracle": True},
        )
        units.append(unit.to_dict())
    return {
        "schema_version": "evidence_bank_v2", "arm": "oracle_planner",
        "source_video": str(source_video), "scope_interval": list(scope),
        "units": units, "oracle": True,
    }


def _interval_hit(predicted: list[float], truth: list[float]) -> bool:
    left, right = map(float, predicted)
    gt_left, gt_right = map(float, truth)
    center = (left + right) / 2.0
    overlap = max(0.0, min(right, gt_right) - max(left, gt_left))
    union = max(right, gt_right) - min(left, gt_left)
    return gt_left <= center <= gt_right or (union > 0 and overlap / union > 0.3)


def evaluate_arm(bank: dict[str, Any], matches: dict[str, Any], plan: dict[str, Any],
                 oracle: dict[str, Any], *, top_k: int = 5) -> dict[str, Any]:
    units = sorted(bank.get("units") or [], key=lambda row: (
        -float(row.get("visual_strength", 0)), -float(row.get("confidence", 0))))
    facts = oracle.get("facts") or []
    fact_hits = {}
    fact_boundary_errors = {}
    for fact in facts:
        candidates = units[:top_k]
        hit_rows = [row for row in candidates if _interval_hit(
            row.get("core_interval") or [], fact["core_interval"])]
        fact_hits[str(fact["id"])] = bool(hit_rows)
        if hit_rows:
            gt_left, gt_right = map(float, fact["core_interval"])
            fact_boundary_errors[str(fact["id"])] = round(min(
                (abs(float(row["core_interval"][0]) - gt_left) +
                 abs(float(row["core_interval"][1]) - gt_right)) / 2.0
                for row in hit_rows), 4)
        else:
            fact_boundary_errors[str(fact["id"])] = None
    temporal_recall = (sum(fact_hits.values()) / len(fact_hits)) if fact_hits else 0.0
    score_map = {str(row.get("evidence_id")): row.get("slot_fit") or {}
                 for row in matches.get("matches") or []}
    slot_rows = {}
    for slot in sorted({str(value) for fact in facts
                        for value in fact.get("expected_slots") or []}):
        ranked = sorted(units, key=lambda row: -float(
            score_map.get(str(row.get("id")), {}).get(slot, 0.0)))[:top_k]
        expected = [fact for fact in facts if slot in fact.get("expected_slots", [])]
        hits = [row for row in ranked if any(_interval_hit(
            row.get("core_interval") or [], fact["core_interval"])
            for fact in expected)]
        recalled_truth = sum(any(_interval_hit(
            row.get("core_interval") or [], fact["core_interval"])
            for row in ranked) for fact in expected)
        slot_rows[slot] = {
            "recall_at_k": (round(recalled_truth / len(expected), 4)
                            if expected else 0.0),
            "precision_at_k": len(hits) / len(ranked) if ranked else 0.0,
            "top_evidence_ids": [row.get("id") for row in ranked],
        }
    selected = plan.get("segments") or []
    correct = 0
    for segment in selected:
        if any(slot in fact.get("expected_slots", []) and _interval_hit(
                segment.get("core_interval") or [], fact["core_interval"])
               for slot in segment.get("assigned_slots") or [] for fact in facts):
            correct += 1
    arc_subjects = {row.get("subject_track_id") for row in selected
                    if set(row.get("assigned_slots") or []).intersection(
                        {"struggle", "reversal", "payoff"})}
    arc_subjects.discard(None)
    return {
        "temporal_hit_at_k": fact_hits,
        "temporal_recall_at_k": round(temporal_recall, 4),
        "boundary_error_s": fact_boundary_errors,
        "mean_boundary_error_s": round(sum(
            value for value in fact_boundary_errors.values() if value is not None) /
            max(1, sum(value is not None for value in fact_boundary_errors.values())), 4),
        "slot_metrics": slot_rows,
        "planner_slot_accuracy": round(correct / len(selected), 4) if selected else 0.0,
        "same_subject_consistent": len(arc_subjects) <= 1 and bool(arc_subjects),
        "selected_evidence_count": len(selected),
        "irrelevant_selected_count": len(selected) - correct,
    }


def _blind_pass(reviews: dict[str, dict]) -> tuple[bool, list[str]]:
    reasons = []
    for name in ("source_only", "bgm_mix"):
        review = reviews.get(name) or {}
        if not review.get("parsed"):
            reasons.append(f"blind_{name}_unparsed")
            continue
        if review.get("hook_clear") is not True:
            reasons.append(f"blind_{name}_hook_unclear")
        if review.get("montage_coherent") is not True:
            reasons.append(f"blind_{name}_incoherent")
        if review.get("functionless_span_present") is not False:
            reasons.append(f"blind_{name}_functionless_span")
        if review.get("audible_dialogue_present") is True and review.get("speech_clear") is not True:
            reasons.append(f"blind_{name}_speech_unclear")
    if (reviews.get("bgm_mix") or {}).get("music_present") is not True:
        reasons.append("blind_bgm_mix_music_missing")
    return not reasons, reasons


def _run_arm(cfg: AppConfig, *, arm_id: str, bank: dict[str, Any],
             pattern: dict[str, Any], runner, source: Path, bgm: Path,
             output_dir: Path, duration: dict[str, Any],
             oracle: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def blocked_infrastructure(stage: str, reason_code: str) -> dict[str, Any]:
        plan = {
            "schema_version": "evidence_edit_plan_v61", "passed": False,
            "failure_class": "infrastructure", "failure_stage": stage,
            "reason_code": reason_code, "reasons": [reason_code],
            "segments": [],
        }
        acceptance = {
            "automated_passed": False, "passed": False, "delivery": "blocked",
            "failure_class": "infrastructure", "failure_stage": stage,
            "reason_code": reason_code, "reasons": [reason_code],
            "human_acceptance_required": True,
        }
        _write_json(output_dir / "evidence_bank.json", bank)
        _write_json(output_dir / "edit_plan.json", plan)
        _write_json(output_dir / "acceptance.json", acceptance)
        return {"arm_id": arm_id, "bank": bank, "matches": {
            "schema_version": "pattern_matches_v1", "parsed": False,
            "matches": []}, "plan": plan, "acceptance": acceptance}

    if oracle is None:
        bank = link_subject_tracks(
            bank, runner=runner, output=output_dir / "subject_tracks.json")
        if not (bank.get("subject_linking") or {}).get("parsed"):
            return blocked_infrastructure(
                "subject_linking", "subject_linking_response_invalid")
        matches = match_evidence_to_pattern(
            bank, pattern, runner=runner, output=output_dir / "pattern_matches.json")
        if not matches.get("parsed"):
            return blocked_infrastructure("matching", "pattern_match_response_invalid")
    else:
        matches = oracle_pattern_matches(bank, oracle)
        _write_json(output_dir / "subject_tracks.json", {
            "schema_version": "subject_tracks_v1", "oracle": True,
            "links": [{"evidence_id": row.get("id"),
                       "subject_track_id": row.get("subject_track_id")}
                      for row in bank.get("units") or []],
        })
        _write_json(output_dir / "pattern_matches.json", matches)
    _write_json(output_dir / "evidence_bank.json", bank)
    plan = build_v61_edit_plan(
        bank, pattern, matches,
        min_duration_s=float(duration["min_s"]),
        max_duration_s=float(duration["max_s"]),
        max_segment_s=float(duration.get("max_segment_s", 4.0)),
        pre_roll_s=float(duration.get("pre_roll_s", 0.3)),
        post_roll_s=float(duration.get("post_roll_s", 0.2)))
    _write_json(output_dir / "edit_plan.json", plan)
    acceptance = {
        "automated_passed": False, "passed": False, "delivery": "blocked",
        "failure_class": plan.get("failure_class"),
        "failure_stage": plan.get("failure_stage"),
        "reason_code": plan.get("reason_code"), "reasons": plan.get("reasons") or [],
        "human_acceptance_required": True,
    }
    if plan.get("passed"):
        rendered = render_micro_montage(
            cfg, plan, output_dir / "render", source_video=source,
            bgm_path=bgm, force=True)
        preview = output_dir / "preview.mp4"
        shutil.copy2(rendered["content_master"], preview)
        reviews = blind_review_variants(rendered["variants"], runner=runner)
        _write_json(output_dir / "blind_review.json", reviews)
        blind_ok, blind_reasons = _blind_pass(reviews)
        blind_unparsed = any(not (reviews.get(name) or {}).get("parsed")
                             for name in ("source_only", "bgm_mix"))
        acceptance.update({
            "automated_passed": blind_ok,
            "failure_class": (None if blind_ok else
                              "infrastructure" if blind_unparsed else "verification"),
            "failure_stage": "human" if blind_ok else "blind",
            "reason_code": ("human_acceptance_pending" if blind_ok else
                            "blind_review_response_invalid" if blind_unparsed else
                            "blind_review_failed"),
            "reasons": ["human_acceptance_pending"] if blind_ok else blind_reasons,
        })
    _write_json(output_dir / "acceptance.json", acceptance)
    return {"arm_id": arm_id, "bank": bank, "matches": matches,
            "plan": plan, "acceptance": acceptance}


def _diagnosis(coarse: dict[str, Any], dense: dict[str, Any], oracle: dict[str, Any]) \
        -> dict[str, Any]:
    a, b = coarse["metrics"], dense["metrics"]
    notes = []
    if b["temporal_recall_at_k"] > a["temporal_recall_at_k"] and oracle["plan"]["passed"]:
        notes.append("targeted_dense_rewatch_improved_recall")
    if b["temporal_recall_at_k"] < 1.0 and oracle["plan"]["passed"]:
        notes.append("mining_gap_remains")
    if b["temporal_recall_at_k"] >= 0.8 and any(
            row["recall_at_k"] < 1.0 for row in b["slot_metrics"].values()):
        notes.append("pattern_matcher_gap")
    if not oracle["plan"]["passed"]:
        notes.append("planner_failed_oracle_control")
    if oracle["plan"]["passed"] and not oracle["acceptance"]["automated_passed"]:
        notes.append("render_or_blind_acceptance_gap")
    return {"schema_version": "module_diagnosis_v1", "findings": notes or
            ["no_single_failure_stage_identified"],
            "coarse_temporal_recall": a["temporal_recall_at_k"],
            "dense_temporal_recall": b["temporal_recall_at_k"],
            "oracle_plan_passed": bool(oracle["plan"]["passed"])}


def _run_evidence_diagnostic(cfg: AppConfig, experiment_spec: Path | dict,
                             output_dir: Path, *, runner,
                             oracle_path: Path | dict,
                             source_video: Path | None = None) -> dict[str, Any]:
    """Run w15, coarse/dense A/B, and the Oracle Planner control."""
    spec, spec_hash = read_diagnostic_spec(experiment_spec)
    oracle = (json.loads(_resolve_path(oracle_path).read_text(encoding="utf-8"))
              if isinstance(oracle_path, (str, Path)) else deepcopy(oracle_path))
    if oracle.get("schema_version") != "evidence_oracle_v1":
        raise ValueError("oracle schema_version must be evidence_oracle_v1")
    if oracle.get("control_scope_valid") is not True:
        raise ValueError("control_scope_invalid")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_source_video(cfg, spec["source"], source_video)
    reference = _resolve_path(spec["reference"]).resolve()
    bgm_value = str(spec.get("bgm_path") or
                     (cfg.library.get("narrative_render") or {}).get("bgm_path") or "")
    bgm = _resolve_path(bgm_value).resolve()
    for path in (source, reference, bgm):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": "evidence_diagnostic_run_v1", "status": "running",
        "delivery": "blocked", "spec_sha256": spec_hash,
        "oracle_sha256": _sha256_json(oracle),
        "input_hashes": {"source": _sha256_file(source),
                         "reference": _sha256_file(reference),
                         "bgm": _sha256_file(bgm)},
    }
    _write_json(output_dir / "experiment_spec.json", spec)
    _write_json(output_dir / "oracle_ground_truth.json", oracle)
    _write_json(output_dir / "experiment_manifest.json", manifest)
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    perception = spec["perception"]
    scopes = spec["controls"]
    mined = {}
    for control_id, scope_row in scopes.items():
        scope = (float(scope_row["start_s"]), float(scope_row["end_s"]))
        control_dir = output_dir / "controls" / control_id
        shots = detect_scoped_shots(
            source, ffmpeg_bin=ffmpeg,
            threshold=float(perception.get("source_scene_threshold", 0.3)),
            min_shot_len_s=float(perception.get("min_shot_duration_s", 0.15)),
            start_s=scope[0], end_s=scope[1])
        _write_json(control_dir / "source_shots.json", shots)
        mined[control_id] = run_dual_resolution_mining(
            source, shots, runner=runner, scope_interval=scope,
            output_dir=control_dir / "mining",
            coarse_fps=float(perception.get("coarse_fps", 2.0)),
            dense_fps=float(perception.get("dense_fps", 12.0)),
            max_watch_s=float(perception.get("max_watch_s", 6.0)),
            overlap_s=float(perception.get("watch_overlap_s", 0.5)),
            dense_padding_s=float(perception.get("dense_padding_s", 0.5)),
            max_dense_regions=int(perception.get("max_dense_regions", 8)))
        if not mined[control_id]["sampling_audit"]["passed"]:
            _write_json(output_dir / "sampling_audit.json", {
                "schema_version": "sampling_audit_v1",
                "controls": {key: value["sampling_audit"]
                             for key, value in mined.items()},
                "passed": False,
            })
            first_error = mined[control_id]["sampling_audit"]["errors"][0]
            failure_stage = str(first_error.get("failure_stage") or "sampling")
            reason_code = ("sampling_audit_failed" if failure_stage == "sampling"
                           else "omni_response_parse_failed")
            manifest.update({"status": "blocked", "failure_class": "infrastructure",
                             "failure_stage": failure_stage,
                             "reason_code": reason_code})
            _write_json(output_dir / "experiment_manifest.json", manifest)
            acceptance = {**manifest, "passed": False, "automated_passed": False}
            _write_json(output_dir / "acceptance.json", acceptance)
            return {"output_dir": output_dir, "acceptance": acceptance}
    _write_json(output_dir / "sampling_audit.json", {
        "schema_version": "sampling_audit_v1",
        "controls": {key: value["sampling_audit"] for key, value in mined.items()},
        "passed": True,
    })
    action_scope = (float(scopes["action"]["start_s"]),
                    float(scopes["action"]["end_s"]))
    negative = _run_arm(
        cfg, arm_id="w15_coarse_dense", bank=mined["w15"]["dense_bank"],
        pattern=spec["pattern"], runner=runner, source=source, bgm=bgm,
        output_dir=output_dir / "controls" / "w15" / "arm",
        duration=spec["duration"])
    coarse = _run_arm(
        cfg, arm_id="coarse_only", bank=mined["action"]["coarse_bank"],
        pattern=spec["pattern"], runner=runner, source=source, bgm=bgm,
        output_dir=output_dir / "arms" / "coarse_only",
        duration=spec["duration"])
    dense = _run_arm(
        cfg, arm_id="coarse_dense", bank=mined["action"]["dense_bank"],
        pattern=spec["pattern"], runner=runner, source=source, bgm=bgm,
        output_dir=output_dir / "arms" / "coarse_dense",
        duration=spec["duration"])
    oracle_evidence = oracle_bank(oracle, source_video=source, scope=action_scope)
    _write_json(output_dir / "arms" / "oracle_planner" /
                "oracle_evidence_bank.json", oracle_evidence)
    _write_json(output_dir / "arms" / "oracle_planner" /
                "oracle_slot_candidates.json",
                oracle_pattern_matches(oracle_evidence, oracle))
    oracle_arm = _run_arm(
        cfg, arm_id="oracle_planner", bank=oracle_evidence,
        pattern=spec["pattern"], runner=runner, source=source, bgm=bgm,
        output_dir=output_dir / "arms" / "oracle_planner",
        duration=spec["duration"], oracle=oracle)
    for arm in (coarse, dense, oracle_arm):
        arm["metrics"] = evaluate_arm(
            arm["bank"], arm["matches"], arm["plan"], oracle,
            top_k=int(spec.get("metrics", {}).get("top_k", 5)))
        _write_json(output_dir / "arms" / arm["arm_id"] / "metrics.json",
                    arm["metrics"])
    comparison = {
        "schema_version": "evidence_ab_v1",
        "coarse_only": coarse["metrics"], "coarse_dense": dense["metrics"],
        "dense_recall_gain": round(
            dense["metrics"]["temporal_recall_at_k"] -
            coarse["metrics"]["temporal_recall_at_k"], 4),
        "mean_boundary_error_change_s": round(
            dense["metrics"]["mean_boundary_error_s"] -
            coarse["metrics"]["mean_boundary_error_s"], 4),
    }
    diagnosis = _diagnosis(coarse, dense, oracle_arm)
    _write_json(output_dir / "metrics" / "ab_comparison.json", comparison)
    _write_json(output_dir / "metrics" / "module_diagnosis.json", diagnosis)
    manifest.update({"status": "complete", "diagnostic_completed": True,
                     "negative_control": negative["acceptance"],
                     "arms": {arm["arm_id"]: arm["acceptance"]
                              for arm in (coarse, dense, oracle_arm)},
                     "diagnosis": diagnosis})
    _write_json(output_dir / "experiment_manifest.json", manifest)
    acceptance = {
        "diagnostic_completed": True, "automated_passed": False, "passed": False,
        "failure_class": "verification", "failure_stage": "human",
        "reason_code": "human_diagnostic_review_pending",
        "reasons": ["human_diagnostic_review_pending"], "delivery": "blocked",
    }
    _write_json(output_dir / "acceptance.json", acceptance)
    return {"output_dir": output_dir, "acceptance": acceptance,
            "comparison": comparison, "diagnosis": diagnosis}


def run_evidence_diagnostic(cfg: AppConfig, experiment_spec: Path | dict,
                            output_dir: Path, *, runner,
                            oracle_path: Path | dict,
                            source_video: Path | None = None) -> dict[str, Any]:
    """Run the diagnostic and always leave a stable blocked artifact on failure."""
    output_dir = Path(output_dir).resolve()
    try:
        return _run_evidence_diagnostic(
            cfg, experiment_spec, output_dir, runner=runner,
            oracle_path=oracle_path, source_video=source_video)
    except Exception as exc:  # CLI boundary: preserve the concrete exception in evidence
        output_dir.mkdir(parents=True, exist_ok=True)
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if "control_scope_invalid" in lowered:
            failure_class, failure_stage, reason_code = (
                "verification", "policy", "control_scope_invalid")
        elif "bgm" in lowered and isinstance(exc, FileNotFoundError):
            failure_class, failure_stage, reason_code = "audio", "policy", "bgm_missing"
        elif "policy_contract_conflict" in lowered:
            failure_class, failure_stage, reason_code = (
                "verification", "policy", "policy_contract_conflict")
        elif "render" in lowered or "ffmpeg" in lowered:
            failure_class, failure_stage, reason_code = (
                "infrastructure", "render", "render_execution_failed")
        else:
            failure_class, failure_stage, reason_code = (
                "infrastructure", "mining", "runner_or_pipeline_error")
        acceptance = {
            "diagnostic_completed": False, "automated_passed": False,
            "passed": False, "delivery": "blocked",
            "failure_class": failure_class, "failure_stage": failure_stage,
            "reason_code": reason_code, "reasons": [reason_code],
            "exception": message,
        }
        manifest_path = output_dir / "experiment_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {"schema_version": "evidence_diagnostic_run_v1"}
        manifest.update({"status": "blocked", "delivery": "blocked",
                         **{key: value for key, value in acceptance.items()
                            if key not in {"status", "delivery"}}})
        _write_json(manifest_path, manifest)
        _write_json(output_dir / "acceptance.json", acceptance)
        return {"output_dir": output_dir, "acceptance": acceptance}
