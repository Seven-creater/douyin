"""Versioned, candidate-only bridge from reference evidence to creative inputs.

The accepted timeline is authoritative for time; model prose is never used to
create, move, or remove a shot or transition.  No v1 creative artifact changes.
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file


BLUEPRINT_VERSION = "reference_blueprint_v2"
TRANSFER_VERSION = "creative_transfer_spec_v2"
MAX_LOCAL_PROBES = 8
MAX_PROBES_PER_ISSUE = 2

ANALYZE_PROMPT = """Watch the original audiovisual reference with the supplied
accepted timeline and provisional local observations. Combine picture, sound,
and attributed on-screen statements; a statement is what the video says, not
independent visual proof. Describe the video's communicative position, how
later evidence may change an initial audience interpretation, and how the
ending relates to the preceding information. Do not assume any fixed story
shape, motive, formal contest result, or causal link not shown. For EACH shot,
say what new information it contributes; different short shots must not be
collapsed into one generic action. For EACH adjacent pair, describe only a
supported information change and a possible editing function. Use unknown
when sampled media cannot decide. Cite source shot, statement, and probe IDs.
Return JSON only:
{"schema_version":"reference_blueprint_analysis_v2",
"theme_stance":{"position":"... or unknown","shot_ids":[],"statement_ids":[],"alternative":"...","status":"provisional|insufficient"},
"viewer_change":{"prior":"... or unknown","later":"... or unknown","shot_ids":[],"statement_ids":[],"status":"provisional|insufficient"},
"ending":{"relation":"... or unknown","tone":"... or unknown","shot_ids":[],"statement_ids":[],"status":"provisional|insufficient"},
"narrative_units":[{"unit_id":"...","shot_ids":[],"contribution":"...","status":"provisional|insufficient"}],
"shots":[{"shot_id":"...","information_added":"... or unknown","action_phase":"... or unknown","shot_size":null,"camera_angle":null,"camera_motion":null,"motion_speed":null,"support_probe_ids":[],"statement_ids":[],"status":"provisional|insufficient","technique_status":"provisional|insufficient"}],
"edges":[{"from_shot_id":"...","to_shot_id":"...","information_added":"... or unknown","function_hypothesis":"... or unknown","support_probe_ids":[],"status":"provisional|insufficient"}],
"unknowns":[]}
Input: """

AUDIT_PROMPT = """Independently check the analysis against the original
audiovisual media, accepted timeline, attributed text, and actual sampled
local observations. A matching ID or sampled frame does not prove a semantic
claim. In particular, reject an unseen physical cause, formal outcome, or
textual statement presented as visual fact. Mark insufficient when media does
not decide. Audit every supplied ID once; do not add a salience or required
story-role test. This is model self-check, not human ground truth. Return JSON:
{"schema_version":"reference_blueprint_audit_v2","checks":[
{"id":"theme_stance|viewer_change|ending|shot:<id>|technique:<id>|edge:<from>-><to>",
"verdict":"supported|contested|insufficient","reason":"..."}],
"media_truth_guaranteed":false}
Input: """

ABSTRACT_PROMPT = """Transform the model-checked reference blueprint into
an original-work transfer specification. Preserve the communicative stance,
audience information change, ending relationship, exact supplied slot times,
and only supported editing techniques. Every content slot needs an abstract
information task, not the source action. Every transition slot retains its
measured type and time. Do not include original people, anatomy, activity,
objects, setting, quotations, claim/event/shot/probe IDs, paths, source media,
or reference-specific examples. Do not impose a universal reversal template.
Output JSON only:
{"schema_version":"creative_transfer_spec_v2",
"theme_contract":{"stance":"...","audience_prior":"...","audience_update":"...","ending_relation":"...","tone":"..."},
"story_beats":[{"beat_id":"B01","slot_ids":[],"information_task":"..."}],
"edit_slots":[{"slot_id":"slot_01","start_s":0.0,"end_s":0.0,"information_task":"...","verified_techniques":[],"unresolved":[]}],
"edit_edges":[{"from_slot_id":"slot_01","to_slot_id":"slot_02","information_relation":"... or unknown","editing_function":"... or unknown","status":"supported|unknown"}],
"transition_slots":[{"transition_id":"transition_01","start_s":0.0,"end_s":0.0,"type":"..."}],
"free_slots":["..."],"audio_policy":"reuse_reference_bgm_replace_voice",
"limitations":[]}
Input: """

TRANSFER_AUDIT_PROMPT = """Check the proposed public creative transfer spec
against the private reference blueprint. Reject source-specific characters,
physical traits, actions, setting, objects, quotations, paths, and evidence
IDs, including paraphrased source details. Check that the abstract theme
retains the communicative position and ending relation without turning an
unverified claim into a fact. Check that every shot information task and
adjacent edit relation is abstract and usable in a different domain, not a
rewritten source caption. No unsupported editing label may become verified.
Do not rewrite the draft or return any leaked passage. Return JSON only:
{"schema_version":"creative_transfer_audit_v2","stance_preserved":false,
"ending_preserved":false,"source_surface_absent":false,
"slot_tasks_abstract":false,"reason_codes":[]}
Input: """


def _pairs(shots: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(a["shot_id"], b["shot_id"]) for a, b in zip(shots, shots[1:])]


def expected_audit_ids(static: dict[str, Any]) -> list[str]:
    return (["theme_stance", "viewer_change", "ending"] +
            [f"shot:{row['shot_id']}" for row in static["shots"]] +
            [f"technique:{row['shot_id']}" for row in static["shots"]] +
            [f"edge:{left}->{right}" for left, right in _pairs(static["shots"])])


def inspect_bgm_asset(video: Path, bgm: Path | None,
                      *, ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """A differently encoded file is still only a candidate music stem."""
    if bgm is None or not Path(bgm).is_file():
        return {"status": "missing", "bgm_sha256": None,
                "identical_audio_stream": None}

    def stream_md5(path: Path) -> str:
        result = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-i", str(path), "-map", "0:a:0",
             "-c", "copy", "-f", "md5", "-"], capture_output=True,
            text=True, check=True)
        return result.stdout.strip()

    identical = stream_md5(Path(video)) == stream_md5(Path(bgm))
    return {"status": ("identical_to_source_mix" if identical else
                       "candidate_needs_voice_contamination_review"),
            "bgm_sha256": sha256_file(bgm),
            "identical_audio_stream": identical}


def _local_view(local: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Expose provisional observations, with actual sampling provenance."""
    by_shot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probe_ids: set[str] = set()
    for record in local.get("observations") or []:
        probe_id = str(record["probe_id"])
        probe_ids.add(probe_id)
        findings = record.get("validation", {}).get("change_findings") or []
        allowed = {(row.get("shot_id"), row.get("description"))
                   for row in findings if row.get("effective_status") == "supported"}
        sampled = record.get("sampling", {}).get(
            "actual_frame_timestamps_absolute_s") or []
        for row in record.get("response", {}).get("shots") or []:
            shot_id = str(row.get("shot_id"))
            changes = [dict(change) for change in row.get("changes") or []
                       if (shot_id, change.get("description")) in allowed]
            by_shot[shot_id].append({
                "probe_id": probe_id, "sampling_verified": bool(
                    record.get("sampling", {}).get("sampling_verified")),
                "sampled_at_s": sampled, "entry_state": row.get("entry_state"),
                "visible_action_candidate": row.get("visible_action"),
                "exit_state": row.get("exit_state"), "supported_changes": changes,
                "limitations": row.get("limitations") or [],
            })
    return [{"shot_id": sid, "observations": rows}
            for sid, rows in by_shot.items()], probe_ids


def build_analysis_payload(static: dict[str, Any], local: dict[str, Any]
                           ) -> dict[str, Any]:
    observations, _ = _local_view(local)
    return {
        "interval": [0.0, static["duration_s"]],
        "observation_dimensions": ["story_information", "adjacent_edit",
                                   "attributed_text", "audible", "unknown"],
        "shots": [{"shot_id": row["shot_id"], "section_id": row["section_id"],
                   "start_s": row["start_s"], "end_s": row["end_s"],
                   "claim_ids": {"V": row.get("visual_claim_ids") or [],
                                 "T": row.get("text_claim_ids") or [],
                                 "A": row.get("audio_claim_ids") or []}}
                  for row in static["shots"]],
        "transitions": [{"segment_id": row["segment_id"],
                         "type": row["transition_type"],
                         "interval": row["interval"]}
                        for row in static["transitions"]],
        "text_statements": [{"statement_id": row["claim_id"],
                             "source_type": "attributed_on_screen_text",
                             "observed_text": row["observed_text"],
                             "interval": row["interval"],
                             "timing_status": row["timing_status"]}
                            for row in static["text_timeline"]],
        "audio": {"candidate_onsets_s": static["audio"]["candidate_onsets_s"],
                  "music_beat_status": static["audio"]["music_beat_status"],
                  "beat_synced_status": static["audio"]["beat_synced_status"],
                  "transcript_candidate": static.get("audio_transcript_candidate")},
        "local_observations": observations,
    }


def plan_gap_probes(static: dict[str, Any], local: dict[str, Any],
                    *, limit: int = MAX_LOCAL_PROBES) -> list[dict[str, Any]]:
    """Probe uncovered or invalidated edges without sending prior conclusions."""
    if limit <= 0:
        return []
    covered: set[tuple[str, str]] = set()
    explicit_issue_shots: set[str] = set()
    insufficient_shots: set[str] = set()
    for record in local.get("observations") or []:
        ids = record.get("shot_ids") or []
        if record.get("sampling", {}).get("sampling_verified"):
            covered.update(zip(ids, ids[1:]))
        for code in record.get("validation", {}).get("issue_codes") or []:
            explicit_issue_shots.add(str(code).split(":", 1)[0])
        for finding in record.get("validation", {}).get("change_findings") or []:
            if finding.get("effective_status") == "insufficient":
                insufficient_shots.add(str(finding.get("shot_id")))
    shots = static["shots"]
    ranked = []
    for index, (left, right) in enumerate(zip(shots, shots[1:])):
        pair = left["shot_id"], right["shot_id"]
        boundary = left["section_id"] != right["section_id"]
        explicit = bool(set(pair) & explicit_issue_shots)
        insufficient = bool(set(pair) & insufficient_shots)
        invalid = explicit or insufficient
        if pair in covered and not invalid:
            continue
        rank = 0 if boundary else 1 if explicit else 2 if insufficient else 3
        ranked.append((rank, index, left, right))
    probes = []
    for _, _, left, right in sorted(ranked):
        probes.append({"probe_id": f"gap_{len(probes)+1:02d}",
                       "issue_id": f"edge_{len(probes)+1:02d}",
                       "purpose": "missing_or_invalid_adjacent_information",
                       "channel": "AV", "shot_ids": [left["shot_id"],
                                                      right["shot_id"]],
                       "interval": [left["start_s"], right["end_s"]],
                       "fps": 12.0})
        if len(probes) >= min(limit, MAX_LOCAL_PROBES):
            break
    return probes


def _analysis_errors(analysis: dict[str, Any], static: dict[str, Any],
                     probe_ids: set[str]) -> list[str]:
    errors = []
    if analysis.get("schema_version") != "reference_blueprint_analysis_v2":
        return ["analysis_schema_invalid"]
    shots = [row["shot_id"] for row in static["shots"]]
    pairs = _pairs(static["shots"])
    if [row.get("shot_id") for row in analysis.get("shots") or []] != shots:
        errors.append("shot_coverage_invalid")
    if [(row.get("from_shot_id"), row.get("to_shot_id"))
            for row in analysis.get("edges") or []] != pairs:
        errors.append("edge_coverage_invalid")
    statement_ids = {row["claim_id"] for row in static["text_timeline"]}
    for row in [analysis.get("theme_stance") or {},
                analysis.get("viewer_change") or {}, analysis.get("ending") or {},
                *(analysis.get("narrative_units") or []),
                *(analysis.get("shots") or []), *(analysis.get("edges") or [])]:
        if set(row.get("shot_ids") or []) - set(shots):
            errors.append("unknown_shot_ref")
        if set(row.get("statement_ids") or []) - statement_ids:
            errors.append("unknown_statement_ref")
        if set(row.get("support_probe_ids") or []) - probe_ids:
            errors.append("unknown_probe_ref")
        if row.get("status") not in {"provisional", "insufficient"}:
            errors.append("analysis_status_invalid")
    return sorted(set(errors))


def build_reference_blueprint(static: dict[str, Any], local: dict[str, Any],
                              analysis: dict[str, Any], audit: dict[str, Any]
                              ) -> dict[str, Any]:
    """Join immutable measured time with separately auditable model claims."""
    _, probe_ids = _local_view(local)
    errors = _analysis_errors(analysis, static, probe_ids)
    if audit.get("schema_version") != "reference_blueprint_audit_v2" or audit.get(
            "media_truth_guaranteed") is not False:
        errors.append("audit_schema_invalid")
    expected = expected_audit_ids(static)
    checks = audit.get("checks") or []
    if (len(checks) != len(expected) or
            {row.get("id") for row in checks} != set(expected)):
        errors.append("audit_coverage_invalid")
    if any(row.get("verdict") not in {"supported", "contested", "insufficient"}
           for row in checks):
        errors.append("audit_verdict_invalid")
    verdicts = {row.get("id"): row.get("verdict") for row in checks}
    story_errors = set(errors) - {"edge_coverage_invalid",
                                  "audit_coverage_invalid"}
    story_ready = not story_errors and all(
        verdicts.get(key) == "supported" and
        analysis.get(key, {}).get("status") == "provisional"
        for key in ("theme_stance", "viewer_change", "ending"))
    semantic_fields = (("theme_stance", "position"), ("viewer_change", "prior"),
                       ("viewer_change", "later"), ("ending", "relation"),
                       ("ending", "tone"))
    if any(str((analysis.get(parent) or {}).get(field) or "").strip().lower()
           in {"", "unknown", "unclear", "undetermined"}
           for parent, field in semantic_fields):
        story_ready = False
    if not analysis.get("narrative_units"):
        story_ready = False
    analyzed_shots = {row.get("shot_id"): row for row in analysis.get("shots") or []}
    shot_rows = []
    technique_keys = ("shot_size", "camera_angle", "camera_motion", "motion_speed")
    for index, row in enumerate(static["shots"], 1):
        semantic = analyzed_shots.get(row["shot_id"]) or {}
        model_technique_accepted = (semantic.get("technique_status") == "provisional"
                                    and verdicts.get(f"technique:{row['shot_id']}") ==
                                    "supported")
        techniques = {key: row.get(key) if row.get(key) is not None else (
            semantic.get(key) if model_technique_accepted else None)
            for key in technique_keys}
        shot_rows.append({"slot_index": index, "shot_id": row["shot_id"],
                          "start_s": row["start_s"], "end_s": row["end_s"],
                          "section_id": row["section_id"], **techniques,
                          "semantic": semantic})
    editing_ready = (story_ready and not errors and all(verdicts.get(key) == "supported"
                                         for key in expected[3:]) and all(
        all(row.get(key) is not None for key in technique_keys)
        for row in shot_rows))
    probe_refs = [{"probe_id": record["probe_id"],
                   "media_sha": record.get("media_sha"),
                   "interval": record.get("interval"),
                   "sampled_at_s": record.get("sampling", {}).get(
                       "actual_frame_timestamps_absolute_s") or [],
                   "sampling_verified": bool(record.get("sampling", {}).get(
                       "sampling_verified"))}
                  for record in local.get("observations") or []]
    result = {
        "schema_version": BLUEPRINT_VERSION, "source_sha": static["source_sha"],
        "static_review_sha": static["artifact_sha"],
        "duration_s": static["duration_s"],
        "shots": shot_rows,
        "transitions": [{"transition_id": row["segment_id"],
                         "type": row["transition_type"],
                         "start_s": row["interval"][0],
                         "end_s": row["interval"][1]}
                        for row in static["transitions"]],
        "edges": analysis.get("edges") or [],
        "measured_editing": static["measured_editing"],
        "story": {key: analysis.get(key) for key in (
            "theme_stance", "viewer_change", "ending", "narrative_units")},
        "text_statements": static["text_timeline"],
        "probe_refs": probe_refs,
        "audio": {"mixed_source_sha": static["audio"]["source_sha"],
                  "bgm_stem_status": "not_verified",
                  "music_beat_status": static["audio"]["music_beat_status"],
                  "beat_synced_status": static["audio"]["beat_synced_status"]},
        "unresolved": analysis.get("unknowns") or [],
        "audit": audit, "validation_issues": sorted(set(errors)),
        "story_candidate_ready": story_ready,
        "editing_candidate_ready": editing_ready,
        "production_release_allowed": False,
    }
    result["artifact_sha"] = json_hash(result)
    return result


def abstraction_payload(blueprint: dict[str, Any]) -> dict[str, Any]:
    if not blueprint.get("story_candidate_ready"):
        raise ValueError("story_not_model_checked")
    shot_to_slot = {row["shot_id"]: f"slot_{row['slot_index']:02d}"
                    for row in blueprint["shots"]}
    edge_verdicts = {row["id"]: row["verdict"]
                     for row in blueprint["audit"]["checks"]}
    return {"interval": [0.0, blueprint["duration_s"]],
            "theme_stance": blueprint["story"]["theme_stance"],
            "viewer_change": blueprint["story"]["viewer_change"],
            "ending": blueprint["story"]["ending"],
            "narrative_units": blueprint["story"]["narrative_units"],
            "slots": [{"slot_id": f"slot_{row['slot_index']:02d}",
                       "start_s": row["start_s"], "end_s": row["end_s"],
                       "source_information_task": (row["semantic"] or {}).get(
                           "information_added")}
                      for row in blueprint["shots"]],
            "transitions": [{"transition_id": f"transition_{index:02d}",
                             "start_s": row["start_s"], "end_s": row["end_s"],
                             "type": row["type"]}
                            for index, row in enumerate(blueprint["transitions"], 1)],
            "editing_edges": [{"from_slot_id": shot_to_slot[row["from_shot_id"]],
                               "to_slot_id": shot_to_slot[row["to_shot_id"]],
                               "source_information_added": row.get("information_added"),
                               "source_function_hypothesis": row.get(
                                   "function_hypothesis"),
                               "evidence_status": edge_verdicts.get(
                                   f"edge:{row['from_shot_id']}->{row['to_shot_id']}")}
                              for row in blueprint["edges"]],
            "measured_pace_curve": blueprint["measured_editing"]["pace_curve"],
            "audio_policy": "reuse_reference_bgm_replace_voice",
            "beat_status": blueprint["audio"]["beat_synced_status"]}


def validate_transfer_spec(spec: dict[str, Any], blueprint: dict[str, Any],
                           forbidden_markers: list[str]) -> list[str]:
    errors = []
    if spec.get("schema_version") != TRANSFER_VERSION:
        errors.append("schema_invalid")
    if set(spec) != {"schema_version", "theme_contract", "story_beats",
                     "edit_slots", "edit_edges", "transition_slots", "free_slots",
                     "audio_policy", "limitations"}:
        errors.append("public_fields_invalid")
    expected_slots = [(f"slot_{row['slot_index']:02d}", row["start_s"], row["end_s"])
                      for row in blueprint["shots"]]
    actual_slots = [(row.get("slot_id"), row.get("start_s"), row.get("end_s"))
                    for row in spec.get("edit_slots") or []]
    if expected_slots != actual_slots:
        errors.append("shot_slots_not_exact")
    expected_edges = list(zip((row[0] for row in expected_slots),
                              (row[0] for row in expected_slots[1:])))
    actual_edges = [(row.get("from_slot_id"), row.get("to_slot_id"))
                    for row in spec.get("edit_edges") or []]
    if actual_edges != expected_edges:
        errors.append("edit_edges_not_exact")
    edge_verdicts = {row["id"]: row["verdict"]
                     for row in blueprint["audit"]["checks"]}
    for edge, source in zip(spec.get("edit_edges") or [], blueprint["edges"]):
        verdict = edge_verdicts.get(
            f"edge:{source['from_shot_id']}->{source['to_shot_id']}")
        if edge.get("status") not in {"supported", "unknown"} or (
                edge.get("status") == "supported" and verdict != "supported"):
            errors.append("edit_edge_support_overclaimed")
    expected_transitions = [(f"transition_{index:02d}", row["start_s"],
                             row["end_s"], row["type"])
                            for index, row in enumerate(blueprint["transitions"], 1)]
    actual_transitions = [(row.get("transition_id"), row.get("start_s"),
                           row.get("end_s"), row.get("type"))
                          for row in spec.get("transition_slots") or []]
    if expected_transitions != actual_transitions:
        errors.append("transition_slots_not_exact")
    if spec.get("audio_policy") != "reuse_reference_bgm_replace_voice":
        errors.append("audio_policy_invalid")
    slots = {row[0] for row in expected_slots}
    if any(not set(row.get("slot_ids") or []) <= slots
           for row in spec.get("story_beats") or []):
        errors.append("story_beat_slot_unknown")
    covered = {sid for row in spec.get("story_beats") or []
               for sid in row.get("slot_ids") or []}
    if covered != slots:
        errors.append("story_beat_slot_coverage_invalid")
    for slot, source in zip(spec.get("edit_slots") or [], blueprint["shots"]):
        allowed = {f"shot_size:{source['shot_size']}"} if source.get(
            "shot_size") else set()
        for key in ("camera_angle", "camera_motion", "motion_speed"):
            if source.get(key):
                allowed.add(f"{key}:{source[key]}")
        if set(slot.get("verified_techniques") or []) - allowed:
            errors.append("technique_not_verified")
    if not set(spec.get("theme_contract") or {}) == {
            "stance", "audience_prior", "audience_update",
            "ending_relation", "tone"}:
        errors.append("theme_contract_invalid")
    if not blueprint.get("story_candidate_ready"):
        errors.append("parent_story_not_ready")
    serialized = str(spec).casefold()
    if any(marker and marker.casefold() in serialized for marker in forbidden_markers):
        errors.append("reference_surface_leak")
    if blueprint["audio"]["beat_synced_status"] != "verified" and any(
            "beat" in str(technique).lower() or "卡点" in str(technique)
            for row in spec.get("edit_slots") or []
            for technique in row.get("verified_techniques") or []):
        errors.append("unverified_beat_claim")
    return sorted(set(errors))


def publish_candidate_spec(spec: dict[str, Any], blueprint: dict[str, Any],
                           forbidden_markers: list[str]) -> dict[str, Any]:
    errors = validate_transfer_spec(spec, blueprint, forbidden_markers)
    if errors:
        raise ValueError("transfer_spec_invalid:" + ",".join(errors))
    result = {**spec, "parent_blueprint_sha": blueprint["artifact_sha"],
              "status": "model_checked_candidate",
              "editing_candidate_ready": blueprint["editing_candidate_ready"],
              "production_release_allowed": False}
    result["artifact_sha"] = json_hash(result)
    return result


def validate_transfer_audit(audit: dict[str, Any]) -> list[str]:
    if audit.get("schema_version") != "creative_transfer_audit_v2":
        return ["transfer_audit_schema_invalid"]
    required = ("stance_preserved", "ending_preserved",
                "source_surface_absent", "slot_tasks_abstract")
    return [key for key in required if audit.get(key) is not True]
