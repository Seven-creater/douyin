"""V6 evidence-centric editing control pipeline.

The pipeline is deliberately scoped: one 40-second source interval, one
reference editing grammar, fine-grained evidence units, one visual master,
two audio derivatives, independent blind review, and explicit human release.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.agentic_video.evidence_planner import (build_evidence_edit_plan,
                                                write_evidence_edit_plan)
from src.agentic_video.evidence_units import interval_contained
from src.agentic_video.long_video import resolve_source_video
from src.agentic_video.reference_pattern import extract_reference_pattern
from src.agentic_video.renderer import render_micro_montage
from src.config import AppConfig, repo_root
from src.perception import common
from src.perception.detect_shots import detect_scoped_shots
from src.perception.evidence_miner import mine_evidence, split_analysis_intervals
from src.template.schema import extract_json_block


EVIDENCE_BLIND_PROMPT = """你是第一次看到这条短视频的观众，不知道素材来源、参考视频、
Evidence Unit 或剪辑计划。只根据实际画面和可听声音，输出严格 JSON：
{"core_message":"一句话说明这条视频让你看到了什么",
 "hook_clear":true,"montage_coherent":true,"functionless_span_present":false,
 "visible_evidence_roles":["困境","反击","结果"],
 "audible_dialogue_present":true,"speech_clear":true,"music_present":true,
 "too_short_intervals":[],"redundant_intervals":[],
 "subject_relation_clear":true,"supported_result":"画面实际支持的最强结果",
 "unsupported_claims":[]}
规则：看不懂或不确定时写 false；不要猜官方角色名；visible_evidence_roles 只写画面实际
承担的不同功能，不要因为有切点就默认有功能。逐帧能确认动作存在，不代表正常播放时
观众能看清；太短或冗余时必须返回具体成片时间段。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def read_evidence_spec(spec: Path | dict) -> tuple[dict, str]:
    if isinstance(spec, (str, Path)):
        path = _resolve_repo_path(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = deepcopy(spec)
    if not isinstance(payload, dict) or payload.get("spec_version") != "evidence_v6":
        raise ValueError("evidence spec_version must be evidence_v6")
    for key in ("source", "source_scope", "reference", "evidence", "audio_variants"):
        if key not in payload:
            raise ValueError(f"evidence spec missing: {key}")
    scope = payload["source_scope"]
    start, end = float(scope.get("start_s", -1)), float(scope.get("end_s", -1))
    if start < 0 or end <= start:
        raise ValueError("evidence source_scope invalid")
    policy = payload["evidence"]
    minimum = float(policy.get("min_duration_s", 0))
    preferred = float(policy.get("preferred_duration_s", 0))
    maximum = float(policy.get("max_duration_s", 0))
    if not (0 < minimum <= preferred <= maximum):
        raise ValueError("evidence duration policy invalid")
    if set(payload["audio_variants"]) != {"source_only", "bgm_mix"}:
        raise ValueError("evidence_v6 requires source_only and bgm_mix")
    return payload, _sha256_json(payload)


def _transcript_context(cfg: AppConfig, source: str, shots: list[dict], *,
                        max_watch_s: float, overlap_s: float) \
        -> tuple[dict[str, str], str | None]:
    path = cfg.paths.library_dir / "sources" / source / "narrative_transcript.json"
    if not path.is_file():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    for row in payload.get("segments") or []:
        try:
            start = float(row.get("start_ms")) / 1000.0
            end = float(row.get("end_ms")) / 1000.0
        except (TypeError, ValueError):
            continue
        text = str(row.get("text") or "").strip()
        if end > start and text:
            segments.append((start, end, text))
    mapping = {}
    for shot in shots:
        start, end = float(shot["start_s"]), float(shot["end_s"])
        parts = split_analysis_intervals(
            start, end, max_watch_s=max_watch_s, overlap_s=overlap_s)
        for index, (left, right) in enumerate(parts):
            shot_id = (str(shot["id"]) if len(parts) == 1
                       else f"{shot['id']}_part_{index:02d}")
            texts = [text for seg_start, seg_end, text in segments
                     if seg_end > left and seg_start < right]
            mapping[shot_id] = " ".join(texts)
    return mapping, _sha256_file(path)


def _parse_blind_answer(answer: Any) -> dict[str, Any]:
    raw = str(getattr(answer, "text", answer) or "")
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else None
    except (TypeError, ValueError):
        payload = None
    if not isinstance(payload, dict):
        return {"parsed": False, "raw_head": raw[:500], "raw_response": raw}
    roles = [str(role).strip() for role in payload.get("visible_evidence_roles") or []
             if str(role).strip()]
    return {
        "parsed": True,
        "core_message": str(payload.get("core_message") or "").strip(),
        "hook_clear": payload.get("hook_clear") if isinstance(payload.get("hook_clear"), bool) else None,
        "montage_coherent": (payload.get("montage_coherent")
                             if isinstance(payload.get("montage_coherent"), bool) else None),
        "functionless_span_present": (
            payload.get("functionless_span_present")
            if isinstance(payload.get("functionless_span_present"), bool) else None),
        "visible_evidence_roles": roles,
        "audible_dialogue_present": (
            payload.get("audible_dialogue_present")
            if isinstance(payload.get("audible_dialogue_present"), bool) else None),
        "speech_clear": payload.get("speech_clear") if isinstance(payload.get("speech_clear"), bool) else None,
        "music_present": payload.get("music_present") if isinstance(payload.get("music_present"), bool) else None,
        "too_short_intervals": (payload.get("too_short_intervals")
                                if isinstance(payload.get("too_short_intervals"), list)
                                else None),
        "redundant_intervals": (payload.get("redundant_intervals")
                                if isinstance(payload.get("redundant_intervals"), list)
                                else None),
        "subject_relation_clear": (
            payload.get("subject_relation_clear")
            if isinstance(payload.get("subject_relation_clear"), bool) else None),
        "supported_result": str(payload.get("supported_result") or "").strip(),
        "unsupported_claims": (payload.get("unsupported_claims")
                               if isinstance(payload.get("unsupported_claims"), list)
                               else None),
        "gpu_pair": getattr(answer, "gpu_pair", None),
        "raw_response": raw,
    }


def blind_review_variants(variants: dict[str, Path], *, runner) -> dict[str, dict]:
    ordered = [(name, Path(variants[name])) for name in ("source_only", "bgm_mix")]
    requests = [{"video_path": common.prepare_watch_copy(path),
                 "prompt": EVIDENCE_BLIND_PROMPT,
                 "kwargs": {"max_new_tokens": 1024}}
                for _, path in ordered]
    if hasattr(runner, "watch_many"):
        answers = runner.watch_many(requests)
    else:
        answers = [runner.watch(row["video_path"], row["prompt"], **row["kwargs"])
                   for row in requests]
    return {name: _parse_blind_answer(answer)
            for (name, _), answer in zip(ordered, answers)}


def _message_similarity(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        compact = "".join(ch.lower() for ch in str(value or "")
                          if ch.isalnum() or unicodedata.category(ch).startswith("L"))
        return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}
    a, b = grams(left), grams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def evaluate_evidence_acceptance(*, spec: dict, evidence: dict, plan: dict,
                                 audio_manifest: dict, blind_reviews: dict) -> dict:
    scope = (float(spec["source_scope"]["start_s"]),
             float(spec["source_scope"]["end_s"]))
    reasons: list[str] = []
    failure_class = None
    for unit in evidence.get("units") or []:
        if not interval_contained(unit.get("interval") or [], scope):
            failure_class = "scope"
            reasons.append(f"evidence_outside_scope:{unit.get('id')}")
    for segment in plan.get("segments") or []:
        if not interval_contained(segment.get("source_interval") or [], scope):
            failure_class = "scope"
            reasons.append(f"render_interval_outside_scope:{segment.get('unit_id')}")
    if not evidence.get("passed") or not plan.get("passed"):
        failure_class = failure_class or "content"
        reasons.extend(evidence.get("errors") or [])
        reasons.extend(plan.get("reasons") or [])
    duration = float(plan.get("duration_s") or 0)
    minimum = float(spec["evidence"]["min_duration_s"])
    maximum = float(spec["evidence"]["max_duration_s"])
    if not minimum <= duration <= maximum:
        failure_class = failure_class or "budget"
        reasons.append(f"duration_outside_budget:{duration:g}")
    roles = {str(row.get("editing_role") or "") for row in plan.get("segments") or []}
    if "hook" not in roles or len(roles - {"hook"}) < 2:
        failure_class = failure_class or "content"
        reasons.append("hook_plus_two_evidence_roles_missing")
    if audio_manifest.get("video_identical") is not True:
        failure_class = failure_class or "audio"
        reasons.append("audio_variants_changed_video")
    for name in ("source_only", "bgm_mix"):
        review = blind_reviews.get(name) or {}
        if not review.get("parsed"):
            failure_class = failure_class or "verification"
            reasons.append(f"blind_{name}_unparsed")
            continue
        if not review.get("core_message"):
            failure_class = failure_class or "verification"
            reasons.append(f"blind_{name}_core_message_missing")
        for key, expected in (("hook_clear", True), ("montage_coherent", True),
                              ("functionless_span_present", False)):
            if review.get(key) is not expected:
                failure_class = failure_class or "verification"
                reasons.append(f"blind_{name}_{key}_failed")
        if review.get("audible_dialogue_present") is True \
                and review.get("speech_clear") is not True:
            failure_class = "audio"
            reasons.append(f"blind_{name}_speech_unclear")
        if name == "bgm_mix" and review.get("music_present") is not True:
            failure_class = "audio"
            reasons.append("blind_bgm_mix_music_missing")
    similarity = _message_similarity(
        (blind_reviews.get("source_only") or {}).get("core_message", ""),
        (blind_reviews.get("bgm_mix") or {}).get("core_message", ""))
    if similarity < 0.08:
        failure_class = failure_class or "verification"
        reasons.append(f"blind_variant_message_mismatch:{similarity:.3f}")
    return {"automated_passed": not reasons, "passed": False,
            "failure_class": failure_class if reasons else "verification",
            "reasons": reasons or ["human_acceptance_pending"],
            "human_acceptance_required": True,
            "blind_message_similarity": round(similarity, 4),
            "delivery": "blocked"}


def _blocked(output_dir: Path, failure_class: str, reasons: list[Any],
             manifest: dict) -> dict:
    clean = [str(reason) for reason in reasons]
    acceptance = {"automated_passed": False, "passed": False,
                  "failure_class": failure_class, "reasons": clean,
                  "human_acceptance_required": True, "delivery": "blocked"}
    manifest["status"] = "blocked"
    manifest["failure_class"] = failure_class
    manifest["reasons"] = clean
    formal = output_dir / "rendered.mp4"
    if formal.exists():
        formal.unlink()
    _write_json(output_dir / "acceptance.json", acceptance)
    _write_json(output_dir / "run_manifest.json", manifest)
    return acceptance


def finalize_evidence_delivery(output_dir: Path, human_acceptance: Path | dict) -> Path:
    output_dir = Path(output_dir).resolve()
    automated = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    if not automated.get("automated_passed"):
        raise RuntimeError("automated evidence acceptance has not passed")
    human = (json.loads(Path(human_acceptance).read_text(encoding="utf-8"))
             if isinstance(human_acceptance, (str, Path)) else dict(human_acceptance))
    required = ("approved", "core_message_clear", "hook_clear", "montage_coherent",
                "source_only_audio_ok", "bgm_mix_audio_ok")
    if any(human.get(key) is not True for key in required):
        automated.update({"passed": False, "failure_class": "verification",
                          "reasons": ["human_acceptance_failed"],
                          "human_acceptance": human, "delivery": "blocked"})
        _write_json(output_dir / "acceptance.json", automated)
        raise RuntimeError("human evidence acceptance failed")
    source = output_dir / "variants" / "bgm_mix" / "rendered.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_dir / "rendered.mp4"
    shutil.copy2(source, destination)
    automated.update({"passed": True, "failure_class": None, "reasons": [],
                      "human_acceptance": human, "delivery": "passed",
                      "output": str(destination)})
    _write_json(output_dir / "acceptance.json", automated)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "passed", "delivery": "passed",
                     "rendered": str(destination)})
    _write_json(output_dir / "run_manifest.json", manifest)
    return destination


def run_evidence_pipeline(cfg: AppConfig, spec: Path | dict, output_dir: Path, *,
                          runner, source_video: Path | None = None,
                          human_acceptance: Path | dict | None = None,
                          force: bool = False) -> dict[str, Any]:
    """Run the complete V6 control loop; formal delivery always needs a human."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formal = output_dir / "rendered.mp4"
    if formal.exists():
        formal.unlink()
    payload, spec_hash = read_evidence_spec(spec)
    source = resolve_source_video(cfg, str(payload["source"]), source_video)
    reference = _resolve_repo_path(payload["reference"]).resolve()
    bgm_value = str(payload.get("bgm_path") or
                    (cfg.library.get("narrative_render") or {}).get("bgm_path") or "")
    bgm = _resolve_repo_path(bgm_value).resolve() if bgm_value else Path("")
    scope = (float(payload["source_scope"]["start_s"]),
             float(payload["source_scope"]["end_s"]))
    manifest = {
        "schema_version": "evidence_v6_run_v1", "status": "running",
        "delivery": "blocked", "spec_sha256": spec_hash,
        "source": str(source), "reference": str(reference),
        "scope_interval": list(scope), "stages": {},
    }
    _write_json(output_dir / "roughcut_spec.json", payload)
    _write_json(output_dir / "run_manifest.json", manifest)
    try:
        if runner is None:
            return {"acceptance": _blocked(
                output_dir, "infrastructure", ["runner_missing"], manifest),
                "output_dir": output_dir}
        if not reference.is_file():
            return {"acceptance": _blocked(
                output_dir, "infrastructure", [f"reference_missing:{reference}"], manifest),
                "output_dir": output_dir}
        if not bgm.is_file():
            return {"acceptance": _blocked(
                output_dir, "audio", [f"bgm_missing:{bgm}"], manifest),
                "output_dir": output_dir}
        manifest["input_hashes"] = {
            "source_sha256": _sha256_file(source),
            "reference_sha256": _sha256_file(reference),
            "bgm_sha256": _sha256_file(bgm),
        }
        analysis = payload.get("analysis") or {}
        pattern = extract_reference_pattern(
            reference, runner=runner, output=output_dir / "editorial_pattern.json",
            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            ffprobe_bin=cfg.perception.get("ffprobe_bin", "ffprobe"),
            threshold=float(analysis.get("reference_scene_threshold", 0.1)),
            min_shot_len_s=float(analysis.get("min_shot_duration_s", 0.15)))
        manifest["stages"]["reference_pattern"] = {
            "status": "complete", "pattern_type": pattern.get("pattern_type"),
            "shot_count": pattern.get("shot_count")}
        source_shots = detect_scoped_shots(
            source, ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            threshold=float(analysis.get("source_scene_threshold", 0.3)),
            min_shot_len_s=float(analysis.get("min_shot_duration_s", 0.15)),
            start_s=scope[0], end_s=scope[1])
        for index, shot in enumerate(source_shots["shots"]):
            shot["id"] = f"scope_shot_{index:04d}"
        _write_json(output_dir / "source_shots.json", source_shots)
        max_watch_s = float(analysis.get("max_watch_s", 6.0))
        watch_overlap_s = float(analysis.get("watch_overlap_s", 0.5))
        policy = payload["evidence"]
        transcript_by_shot, transcript_hash = _transcript_context(
            cfg, str(payload["source"]), source_shots["shots"],
            max_watch_s=max_watch_s, overlap_s=watch_overlap_s)
        manifest["input_hashes"]["transcript_sha256"] = transcript_hash
        evidence = mine_evidence(
            source, source_shots, runner=runner, scope_interval=scope,
            output=output_dir / "evidence_units.json",
            clip_dir=output_dir / "omni_clips", transcript_by_shot=transcript_by_shot,
            max_watch_s=max_watch_s, overlap_s=watch_overlap_s,
            min_unit_duration_s=float(policy.get("min_unit_duration_s", 0.15)),
            max_unit_duration_s=float(policy.get("max_unit_duration_s", 2.0)))
        manifest["stages"]["evidence_mining"] = {
            "status": "complete" if evidence.get("passed") else "blocked",
            "units": len(evidence.get("units") or []),
            "calls": len(evidence.get("calls") or []),
            "workers": (evidence.get("analysis_policy") or {}).get("parallel_workers")}
        if not evidence.get("passed"):
            return {"acceptance": _blocked(
                output_dir, str(evidence.get("failure_class") or "content"),
                evidence.get("errors") or ["no_verified_evidence"], manifest),
                "output_dir": output_dir}
        plan = build_evidence_edit_plan(
            evidence, pattern,
            preferred_duration_s=float(policy["preferred_duration_s"]),
            min_duration_s=float(policy["min_duration_s"]),
            max_duration_s=float(policy["max_duration_s"]))
        write_evidence_edit_plan(plan, output_dir / "evidence_edit_plan.json")
        manifest["stages"]["edit_plan"] = {
            "status": "complete" if plan.get("passed") else "blocked",
            "segments": len(plan.get("segments") or []),
            "duration_s": plan.get("duration_s")}
        if not plan.get("passed"):
            failure = "budget" if plan.get("failure_class") == "budget" else "content"
            return {"acceptance": _blocked(
                output_dir, failure, plan.get("reasons") or ["edit_plan_failed"], manifest),
                "output_dir": output_dir}
        rendered = render_micro_montage(
            cfg, plan, output_dir, source_video=source, bgm_path=bgm, force=force)
        audio_manifest = json.loads(
            (output_dir / "variants" / "audio_variants_manifest.json").read_text(
                encoding="utf-8"))
        manifest["stages"]["render"] = {
            "status": "complete", "content_master": str(rendered["content_master"]),
            "content_master_sha256": audio_manifest["content_master_sha256"],
            "video_identical": audio_manifest["video_identical"]}
        blinds = blind_review_variants(rendered["variants"], runner=runner)
        blind_raw_dir = output_dir / "blind_review_raw"
        blind_raw_dir.mkdir(parents=True, exist_ok=True)
        for name, review in blinds.items():
            (blind_raw_dir / f"{name}.txt").write_text(
                str(review.get("raw_response") or ""), encoding="utf-8")
        _write_json(output_dir / "blind_review.json", blinds)
        acceptance = evaluate_evidence_acceptance(
            spec=payload, evidence=evidence, plan=plan,
            audio_manifest=audio_manifest, blind_reviews=blinds)
        _write_json(output_dir / "acceptance.json", acceptance)
        manifest["stages"]["blind_review"] = {
            "status": "complete", "automated_passed": acceptance["automated_passed"]}
        manifest["status"] = ("human_acceptance_pending"
                              if acceptance["automated_passed"] else "blocked")
        manifest["hashes"] = {
            "editorial_pattern_sha256": _sha256_json(pattern),
            "evidence_units_sha256": _sha256_json(evidence),
            "evidence_edit_plan_sha256": _sha256_json(plan),
            "content_master_sha256": audio_manifest["content_master_sha256"],
            "content_frames_framemd5": audio_manifest["source_only"]["frame_md5"],
        }
        _write_json(output_dir / "run_manifest.json", manifest)
        if acceptance["automated_passed"] and human_acceptance is not None:
            final = finalize_evidence_delivery(output_dir, human_acceptance)
            acceptance = json.loads(
                (output_dir / "acceptance.json").read_text(encoding="utf-8"))
        else:
            final = None
        return {"output_dir": output_dir, "acceptance": acceptance,
                "content_master": rendered["content_master"],
                "variants": rendered["variants"], "final": final}
    except Exception as exc:
        _blocked(output_dir, "infrastructure",
                 [f"{type(exc).__name__}:{exc}"], manifest)
        raise
