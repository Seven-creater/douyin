"""P0-R1 staged runner: isolated perception -> validated evidence -> DNA v2 RC."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.agentic_video.creative_dna_v2 import (
    run_director, run_independent_analyses, write_dna_artifacts)
from src.agentic_video.manifest import json_hash
from src.agentic_video.migration_eval import (
    freeze_candidate, release_decision, run_suite)
from src.agentic_video.modality_isolation import (
    audit_request, create_visual_only_copy, write_request_audit)
from src.agentic_video.provenance import build_lineage_report
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.reference_claims import (
    CLAIM_LEDGER_VERSION, ClaimLedger,
    build_event_draft, build_validated_reference, import_legacy_montage)
from src.agentic_video.workspace import Workspace

FULL_VISUAL_PROMPT = """You are a visual evidence recorder. Inspect only pixels
in the supplied silent, text-masked video. Do not infer speech, written claims,
motives, identity, personal attributes, social meaning, or narrative function. Return one
JSON object with:
{"entities":[{"entity_id":"E1","visible_descriptor":"non-identifying visual descriptor"}],
"claims":[{"claim_id":"C1","subject":"E1 or E1.body-region","predicate":"appears_in|visible_body_region|initiates_action|performs_action|contacts|precedes|results_in|scene_changes_to|transition_type|same_entity_as|different_entity_from","object":"anonymous entity, visible region, action or state","interval":[0.0,0.1],"polarity":"POSITIVE|NEGATIVE","visibility":"VISIBLE|PARTIAL|OCCLUDED|OFFSCREEN|UNKNOWN","confidence":0.0}],
"events":[{"event_id":"EV1","participants":["E1"],"action_claim_ids":["C1"],"object_ids":[],"ordering":"observed temporal order","outcome_claim_ids":[],"interval":[0.0,0.1]}],
"coverage":{"observed_intervals":[[0.0,0.1]],"masked_or_unjudgeable":[{"interval":[0.0,0.1],"reason_code":"MASK_OVERLAP|OCCLUSION|OFFSCREEN|MOTION_BLUR"}]}}
Use a NEGATIVE claim only if the entire stated interval is continuously visible;
otherwise report the uncertainty in masked_or_unjudgeable and omit the negative
claim. Use actual clip-local times, not the example numbers. Return at most 20
non-duplicate claims and 10 events; omit low-information UNKNOWN claims rather
than padding the list. Keep the JSON compact and complete. JSON only."""

LOCAL_VISUAL_PROMPT = FULL_VISUAL_PROMPT + """
This is a local evidence probe. Prioritize visible body regions, action
initiators, contact/no-contact only when continuous coverage permits it, scene
changes, observable action order and visible results. The complete schema above,
including entities, claims, events and coverage, is mandatory."""


class P0R1Blocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{stage}:{reason_code}:{detail}")
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail


def _parse(raw: str, stage: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise P0R1Blocked(stage, "json_invalid") from exc
    if not isinstance(value, dict):
        raise P0R1Blocked(stage, "not_object")
    return value


def _next_attempt_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    first = directory / f"{stem}{suffix}"
    if not first.exists():
        return first
    count = len(list(directory.glob(f"{stem}*{suffix}"))) + 1
    return directory / f"{stem}_attempt_{count:03d}{suffix}"


def _absolute_response_times(value: dict[str, Any], *, start_s: float,
                             duration_s: float, stage: str) -> dict[str, Any]:
    """Validate clip-local model times, then apply a deterministic origin."""
    if len(value.get("claims") or []) > 20 or len(value.get("events") or []) > 10:
        raise P0R1Blocked(stage, "response_exceeds_atomic_budget")
    seen_claims: set[str] = set()
    seen_events: set[str] = set()

    def convert(row: dict[str, Any], key: str = "interval") -> None:
        interval = row.get(key)
        if not isinstance(interval, list) or len(interval) != 2:
            raise P0R1Blocked(stage, "interval_invalid")
        left, right = float(interval[0]), float(interval[1])
        if left < -0.001 or right <= left or right > duration_s + 0.05:
            raise P0R1Blocked(stage, "clip_local_interval_out_of_range")
        row[key] = [round(start_s + left, 6), round(start_s + right, 6)]

    for row in value.get("claims") or []:
        claim_id = str(row.get("claim_id") or "")
        if not claim_id or claim_id in seen_claims:
            raise P0R1Blocked(stage, "claim_id_missing_or_duplicate")
        seen_claims.add(claim_id)
        convert(row)
    for row in value.get("events") or []:
        event_id = str(row.get("event_id") or "")
        if not event_id or event_id in seen_events:
            raise P0R1Blocked(stage, "event_id_missing_or_duplicate")
        seen_events.add(event_id)
        convert(row)
    coverage = value.get("coverage") or {}
    converted = []
    for interval in coverage.get("observed_intervals") or []:
        holder = {"interval": interval}
        convert(holder)
        converted.append(holder["interval"])
    coverage["observed_intervals"] = converted
    for row in coverage.get("masked_or_unjudgeable") or []:
        convert(row)
    value["coverage"] = coverage
    return value


def _validated_visual_response(raw: str, *, start_s: float, duration_s: float,
                               stage: str) -> dict[str, Any]:
    value = _parse(raw, stage)
    for key in ("claims", "events", "coverage"):
        if key not in value:
            raise P0R1Blocked(stage, "field_missing", key)
    return _absolute_response_times(
        value, start_s=start_s, duration_s=duration_s, stage=stage)


def _cached_call(output: Path, call_id: str, *, prompt_sha: str,
                 start_s: float, duration_s: float
                 ) -> tuple[dict[str, Any], Path, dict[str, Any]] | None:
    raw_dir = output / "raw"
    candidates = sorted(raw_dir.glob(f"{call_id}*.txt"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for raw_path in candidates:
        audit_path = output / "request_audits" / f"{raw_path.stem}.json"
        if not audit_path.is_file():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("prompt_sha") != prompt_sha:
            continue
        raw = raw_path.read_text(encoding="utf-8")
        try:
            parsed = _validated_visual_response(
                raw, start_s=start_s, duration_s=duration_s,
                stage=f"visual_{call_id}")
        except P0R1Blocked:
            continue
        return parsed, raw_path, audit
    return None


def deterministic_timeline(p04e_dir: Path) -> dict[str, Any]:
    """Keep accepted section boundaries and per-segment ordering, not semantics."""
    root = Path(p04e_dir)
    normalized = json.loads((root / "shot_normalization.json").read_text(
        encoding="utf-8"))
    montage = json.loads((root / "montage_shot_observations.json").read_text(
        encoding="utf-8"))
    montage_by_id = {
        str(row.get("segment_id")): row
        for section in montage.get("sections") or []
        for row in section.get("segments") or []}
    segments = []
    sections = []
    for section in normalized.get("sections") or []:
        section_id = str(section.get("section_id"))
        sections.append({"section_id": section_id,
                         "interval": list(section.get("interval") or [])})
        for row in section.get("content_shots") or []:
            segment_id = str(row.get("shot_id"))
            segments.append({"segment_id": segment_id,
                             "section_id": section_id,
                             "segment_kind": "content",
                             "interval": list(row.get("interval") or [])})
        for row in section.get("transition_segments") or []:
            segment_id = str(row.get("transition_id") or row.get("segment_id")
                             or row.get("shot_id"))
            segments.append({"segment_id": segment_id,
                             "section_id": section_id,
                             "segment_kind": "transition",
                             "transition_type": (row.get("transition_type") or
                                                 (montage_by_id.get(segment_id) or {}).get(
                                                     "transition_type")),
                             "interval": list(row.get("interval") or [])})
    segments.sort(key=lambda row: (float(row["interval"][0]),
                                   float(row["interval"][1])))
    return {"schema_version": "deterministic_timeline_v1",
            "sections": sections, "segments": segments,
            "timing_owner": "deterministic_program",
            "source_files": ["shot_normalization.json",
                             "montage_shot_observations.json"]}


def _visual_payload(request_id: str, interval: list[float], mask_sha: str,
                    *, fps: float, dimensions: list[str]) -> dict[str, Any]:
    return {"request_id": request_id, "channel": "V", "interval": interval,
            "anonymous_subject_ids": ["E1", "E2", "E3"],
            "observation_dimensions": dimensions,
            "sampling": {"fps": fps}, "mask_artifact_sha": mask_sha}


def run_visual_perception(source: Path, p04e_dir: Path, output_dir: Path, *,
                          runner: Any, mask_regions: list[dict[str, Any]],
                          forbidden_markers: list[str],
                          mask_version: str = "visual_mask_v1",
                          ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mask = create_visual_only_copy(
        Path(source), output / "media", mask_regions=mask_regions,
        mask_version=mask_version,
        ffmpeg_bin=ffmpeg_bin)
    visual_path = Path(mask["artifact_path"])
    timeline = deterministic_timeline(p04e_dir)
    intervals = {
        "full": [0.0, max(float(row["interval"][1])
                          for row in timeline["segments"])],
        "s1": next(row["interval"] for row in timeline["sections"]
                   if row["section_id"] == "section_01"),
        "s3": next(row["interval"] for row in timeline["sections"]
                   if row["section_id"] == "section_03"),
    }
    calls = [
        ("full", FULL_VISUAL_PROMPT, intervals["full"], 4.0,
         ["entities", "actions", "contacts", "scene_changes", "event_order"]),
        ("s1", LOCAL_VISUAL_PROMPT, intervals["s1"], 12.0,
         ["visible_body_regions", "action_initiator", "contact_process"]),
        ("s3", LOCAL_VISUAL_PROMPT, intervals["s3"], 12.0,
         ["scene_changes", "action_order", "visible_results", "transitions"]),
    ]
    results = {}
    audits = []
    for call_id, prompt, interval, fps, dimensions in calls:
        payload = _visual_payload(
            f"p0r1_{call_id}", list(map(float, interval)),
            str(mask["artifact_sha"]), fps=fps, dimensions=dimensions)
        duration = float(interval[1]) - float(interval[0])
        call_prompt = (prompt + f"\nThe clip-local duration is {duration:.6f} "
                       "seconds; all returned intervals must be within "
                       f"[0.0, {duration:.6f}].")
        audit = audit_request(
            channel="V", prompt=call_prompt, payload=payload,
            media_paths=[visual_path], forbidden_markers=forbidden_markers)
        cached = _cached_call(
            output, call_id, prompt_sha=str(audit["prompt_sha"]),
            start_s=float(interval[0]), duration_s=duration)
        if cached is not None:
            parsed, raw_path, prior_audit = cached
            results[call_id] = {
                "response": parsed, "raw_path": str(raw_path),
                "raw_sha": sha256_file(raw_path), "request_audit": prior_audit,
                "reused_validated_response": True}
            audits.append(prior_audit)
            continue
        prior_attempts = list((output / "raw").glob(f"{call_id}*.txt"))
        if len(prior_attempts) >= 2:
            raise P0R1Blocked(f"visual_{call_id}", "probe_budget_exhausted")
        audit_path = _next_attempt_path(
            output / "request_audits", call_id, ".json")
        write_request_audit(audit_path, audit)
        answer = runner.watch(
            visual_path, call_prompt, start_s=float(interval[0]),
            end_s=float(interval[1]),
            clip_dir=output / "clips" / call_id,
            duration_s=duration, fps=fps,
            use_audio_in_video=False, max_new_tokens=3072,
            stop_after_json_object=True)
        raw = str(getattr(answer, "text", answer))
        raw_path = _next_attempt_path(output / "raw", call_id, ".txt")
        raw_path.write_text(raw, encoding="utf-8")
        parsed = _validated_visual_response(
            raw, start_s=float(interval[0]), duration_s=duration,
            stage=f"visual_{call_id}")
        results[call_id] = {
            "response": parsed, "raw_path": str(raw_path),
            "raw_sha": sha256_file(raw_path), "request_audit": audit,
            "reused_validated_response": False}
        audits.append(audit)
    evidence = json.loads((Path(p04e_dir) / "reference_evidence.json").read_text(
        encoding="utf-8"))
    audio_channel = {
        "schema_version": "audio_channel_v1", "channel": "A",
        "status": "CAPABILITY_UNAVAILABLE",
        "reason_code": "audio_only_backend_unavailable",
        "fallback_used": False, "media_sent": False,
    }
    text_events = list((evidence.get("ocr") or {}).get("text_events") or [])
    text_channel = {
        "schema_version": "text_channel_v1", "channel": "T",
        "status": ("AVAILABLE" if text_events else "NO_VERIFIED_TEXT_EVENTS"),
        "source_provenance": evidence.get("ocr_provenance") or {},
        "text_events": text_events, "media_sent": False,
        "visual_inference_allowed": False,
    }
    (output / "a_channel.json").write_text(
        json.dumps(audio_channel, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "t_channel.json").write_text(
        json.dumps(text_channel, ensure_ascii=False, indent=2), encoding="utf-8")
    value = {
        "schema_version": "visual_perception_draft_v1",
        "source_sha": sha256_file(Path(source)), "visual_mask": mask,
        "deterministic_timeline": timeline, "calls": results,
        "audio_channel": audio_channel, "text_channel": text_channel,
        "isolation_report": {
            "passed": all(row.get("contract_passed") for row in audits),
            "call_count": len(audits),
            "channels": sorted({row["channel"] for row in audits}),
            "audio_only_backend": "not_used_unavailable",
            "text_channel": text_channel["status"],
            "visual_fallback_for_audio_or_text": False,
        },
    }
    value["artifact_sha"] = json_hash(value)
    (output / "visual_perception_draft.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def _claim_from_model(row: dict[str, Any], *, claim_id: str, call_id: str,
                      perception: dict[str, Any], review: dict[str, Any]
                      ) -> dict[str, Any]:
    decision = review.get(claim_id) or {}
    accepted = decision.get("decision") == "supports"
    interval = [float(value) for value in row.get("interval") or []]
    raw_object = row.get("object")
    support_refs = [{
        "ref_id": f"{call_id}:{row.get('claim_id')}", "kind": "frame_range",
        "path": perception["visual_mask"]["artifact_path"],
        "sha": perception["visual_mask"]["artifact_sha"],
        "coverage": {"kind": decision.get("coverage_kind", "sampled"),
                     "interval": interval},
    }]
    support_refs.extend(list(decision.get("additional_support_refs") or []))
    return {
        "claim_id": claim_id, "subject": str(row.get("subject") or "E_UNKNOWN"),
        "predicate": str(row.get("predicate") or "unmapped_observation"),
        "object": str(raw_object if raw_object not in (None, "") else
                      "UNSPECIFIED"),
        "raw_object": raw_object, "interval": interval,
        "modality": "V", "polarity": str(row.get("polarity") or "POSITIVE"),
        "visibility": str(row.get("visibility") or "UNKNOWN"),
        "epistemic_status": "SUPPORTED" if accepted else "INSUFFICIENT",
        "support_refs": support_refs,
        "producer": f"omni_isolated_visual:{call_id}",
        "source_sha": perception["source_sha"],
        "schema_version": CLAIM_LEDGER_VERSION,
        "record_status": "ACTIVE", "supersedes": decision.get("supersedes"),
        "semantic_review": ({"decision": "supports",
                             "reviewer": decision.get("reviewer"),
                             "basis": decision.get("basis", "isolated visual review")}
                            if accepted else None),
        "limitation": decision.get("limitation") or (
            None if accepted else "not accepted by semantic review"),
    }


def compile_validated_reference(perception_path: Path, p04e_dir: Path,
                                review_path: Path, output_dir: Path
                                ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    perception = json.loads(Path(perception_path).read_text(encoding="utf-8"))
    review_file = json.loads(Path(review_path).read_text(encoding="utf-8"))
    decisions = review_file.get("claim_decisions") or {}
    critical = list(map(str, review_file.get("critical_claim_ids") or []))
    if not critical:
        raise P0R1Blocked("validation", "critical_claim_ids_missing")
    ledger = ClaimLedger(source_sha=perception["source_sha"])
    event_rows = []
    for call_id, call in perception.get("calls", {}).items():
        local_map = {}
        for row in call["response"].get("claims") or []:
            local = str(row.get("claim_id") or f"C{len(local_map)+1}")
            claim_id = f"{call_id.upper()}_{local}"
            local_map[local] = claim_id
            ledger.add_claim(_claim_from_model(
                row, claim_id=claim_id, call_id=call_id,
                perception=perception, review=decisions))
        for event in call["response"].get("events") or []:
            event_rows.append({
                "event_id": f"{call_id.upper()}_{event.get('event_id')}",
                "participants": list(event.get("participants") or []),
                "action_claim_ids": [local_map.get(str(ref), str(ref)) for ref in
                                     event.get("action_claim_ids") or []],
                "object_ids": list(event.get("object_ids") or []),
                "ordering": str(event.get("ordering") or "observed"),
                "outcome_claim_ids": [local_map.get(str(ref), str(ref)) for ref in
                                      event.get("outcome_claim_ids") or []],
                "interval": list(event.get("interval") or []),
            })
    for row in review_file.get("manual_claims") or []:
        claim = dict(row)
        claim.setdefault("modality", "V")
        claim.setdefault("polarity", "POSITIVE")
        claim.setdefault("visibility", "VISIBLE")
        claim.setdefault("epistemic_status", "SUPPORTED")
        claim.setdefault("producer", "human_frame_review")
        claim["source_sha"] = perception["source_sha"]
        claim["schema_version"] = CLAIM_LEDGER_VERSION
        claim.setdefault("record_status", "ACTIVE")
        claim.setdefault("supersedes", None)
        if not claim.get("support_refs"):
            support = dict(review_file.get("manual_support_template") or {})
            support["ref_id"] = f"human:{claim.get('claim_id')}"
            support.setdefault("coverage", {
                "kind": "sampled", "interval": claim.get("interval")})
            claim["support_refs"] = [support]
        claim.setdefault("semantic_review", {
            "decision": "supports", "reviewer": review_file.get("reviewer"),
            "basis": "manual review of source-bound frames"})
        ledger.add_claim(claim)
    event_rows.extend(list(review_file.get("manual_events") or []))
    legacy_value = json.loads((Path(p04e_dir) /
                               "montage_shot_observations.json").read_text(
                                   encoding="utf-8"))
    legacy = import_legacy_montage(legacy_value, source_sha=perception["source_sha"])
    for claim in legacy.claims:
        ledger.add_claim(claim, verify_files=False)
    for edge in review_file.get("edges") or []:
        ledger.add_edge(str(edge.get("type")), str(edge.get("source")),
                        str(edge.get("target")), basis=str(edge.get("basis") or ""))
    conflicts = ledger.find_conflicts()
    event_draft = build_event_draft(event_rows, ledger)
    coverage = {call_id: call["response"].get("coverage") or {}
                for call_id, call in perception.get("calls", {}).items()}
    validated = build_validated_reference(
        ledger, event_draft,
        deterministic_timeline=perception["deterministic_timeline"],
        coverage=coverage, critical_claim_ids=critical)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_value = ledger.to_dict()
    for name, value in (("claim_ledger_v1.json", ledger_value),
                        ("event_draft_v1.json", event_draft),
                        ("validated_reference.json", validated)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    diff = {
        "schema_version": "s1_s3_difference_report_v1",
        "old_authorities": [
            {"path": "section_observations.json", "status": "superseded_for_content"},
            {"path": "montage_shot_observations.json", "status": "legacy_mixed"}],
        "new_authority": "validated_reference.json",
        "s1_claim_ids": [row["claim_id"] for row in validated["accepted_claims"]
                         if row["claim_id"].startswith("S1_")],
        "s3_claim_ids": [row["claim_id"] for row in validated["accepted_claims"]
                         if row["claim_id"].startswith("S3_")],
        "legacy_claim_count": len(legacy.claims),
        "legacy_promoted_count": 0, "conflicts": conflicts,
    }
    (output / "s1_s3_difference_report.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    return validated, ledger_value, diff


def build_release_candidate(validated_path: Path, perception_path: Path,
                            output_dir: Path, *, runner: Any,
                            model_config: dict[str, Any], repo_root: Path,
                            holdout_path: Path | None = None) -> dict[str, Any]:
    validated = json.loads(Path(validated_path).read_text(encoding="utf-8"))
    perception = json.loads(Path(perception_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prior_files = [path for path in output.glob("*.json") if path.is_file()]
    if prior_files:
        attempts = output / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        index = 1
        while (attempts / f"attempt_{index:03d}").exists():
            index += 1
        archive = attempts / f"attempt_{index:03d}"
        archive.mkdir()
        for path in prior_files:
            shutil.copy2(path, archive / path.name)
    # First formal commit: evidence-only validated_reference. No interpretation
    # or director request is allowed before this immutable parent exists.
    workspace = Workspace(output / "workspace")
    workspace.state["goal"] = {"target_artifact": "creative_dna_v2_candidate",
                               "required_status": "tested"}
    workspace.set_dependency("reference_interpretation", "validated_reference")
    workspace.set_dependency("editing_analysis", "validated_reference")
    workspace.set_dependency("creative_dna_v2_candidate", "validated_reference")
    workspace.set_dependency("creative_dna_v2_candidate", "reference_interpretation")
    workspace.set_dependency("creative_dna_v2_candidate", "editing_analysis")
    workspace.write_draft("validated_reference", validated,
                          metadata={"created_by": "p0_r1_validator"})
    workspace.commit("validated_reference")
    trace_root = output / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_index = 1
    while (trace_root / f"attempt_{trace_index:03d}").exists():
        trace_index += 1
    trace_dir = trace_root / f"attempt_{trace_index:03d}"
    interpretation, editing = run_independent_analyses(
        validated, runner=runner, trace_dir=trace_dir)
    for name, value in (("interpretation.json", interpretation),
                        ("editing_analysis.json", editing)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    audit, publish = run_director(
        validated, interpretation, editing, runner=runner,
        trace_dir=trace_dir)
    dev_path = Path(repo_root) / "config" / "migration_dev_v1.json"
    regression_path = Path(repo_root) / "config" / "migration_regression_v1.json"
    dev = run_suite(publish, dev_path, runner=runner, tier="dev")
    regression = run_suite(publish, regression_path, runner=runner,
                           tier="regression")
    holdout = (run_suite(publish, holdout_path, runner=runner, tier="holdout")
               if holdout_path else None)
    decision = release_decision(
        dev_report=dev, regression_report=regression,
        isolation_report=perception["isolation_report"],
        holdout_report=holdout)
    write_dna_artifacts(output, audit, publish,
                        release_status=decision["status"])
    for name, value in (("migration_dev_report.json", dev),
                        ("migration_regression_report.json", regression),
                        ("release_decision.json", decision)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    if holdout:
        (output / "migration_holdout_report.json").write_text(
            json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")
    freeze = freeze_candidate(
        dna_publish=publish, model_config=model_config,
        template_paths=[Path(__file__)],
        judge_path=Path(__file__).with_name("migration_eval.py"),
        rules_path=Path(__file__).with_name("creative_dna_v2.py"),
        dev_suite=dev_path, regression_suite=regression_path,
        holdout_suite=holdout_path, release_status=decision["status"])
    (output / "release_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, value in (("reference_interpretation", interpretation),
                        ("editing_analysis", editing)):
        workspace.write_draft(name, value, metadata={
            "created_by": "independent_text_analysis",
            "derived_from": [{"artifact_id": "validated_reference",
                              "version": "v1",
                              "sha": workspace.get_sha("validated_reference")}],
        })
        workspace.commit(name)
    workspace.write_draft("creative_dna_v2_candidate", publish, metadata={
        "created_by": "director_dna_v2",
        "derived_from": [
            {"artifact_id": name, "version": "v1", "sha": workspace.get_sha(name)}
            for name in ("validated_reference", "reference_interpretation",
                         "editing_analysis")],
    })
    if decision["status"] != "blocked":
        workspace.mark_tested("creative_dna_v2_candidate")
    lineage = build_lineage_report(workspace)
    (output / "lineage_report.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"interpretation": interpretation, "editing_analysis": editing,
            "dna_audit": audit, "dna_publish": publish,
            "dev_report": dev, "regression_report": regression,
            "holdout_report": holdout, "release_decision": decision,
            "freeze": freeze, "lineage": lineage}
