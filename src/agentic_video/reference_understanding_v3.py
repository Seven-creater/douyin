"""Isolated, evidence-limited reference understanding experiment.

Nothing here publishes to the creative production DAG. Local media readings
remain hypotheses even when the same model agrees with itself twice.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.reference_readout import (
    _model_call, _transfer_leaks, _write_json, load_reference, prepare_readout,
)
from src.agentic_video.reference_understanding_v2 import build_review_issues
from src.perception.omni_runner import cut_clip

VERSION = "reference_understanding_v3"
MAX_LOCAL_MEDIA_CALLS = 8
MAX_CALLS_PER_ISSUE = 2
MOTION_MIN_SAMPLES = 2

V_PROMPT = """Inspect this silent, text-masked local clip. The supplied shot
intervals are clip-relative. For EACH shot describe entry state, directly
visible action, exit state, and only changes whose first visible evidence can
be located in actual sampled frames. Report times in clip-relative seconds.
If a short shot or mask prevents a motion judgment, say insufficient. Across
each adjacent pair choose same_action, new_action, possible_ellipsis, or
unknown. A hard cut is never a single uninterrupted shot. Do not infer text,
audio, identity, motive, causation, victory, or unseen action. In particular,
not seeing a change in sampled frames does not prove it never occurred.
Return JSON only:
{"schema_version":"reference_local_observation_v3","shots":[
{"shot_id":"...","entry_state":"... or unknown","visible_action":"... or unknown",
"exit_state":"... or unknown","changes":[{"description":"...",
"kind":"action|posture|contact|identity|causal|other","first_visible_s":0.0,
"epistemic_status":"supported|contested|insufficient"}],"limitations":[]}],
"connections":[{"from_shot_id":"...","to_shot_id":"...",
"continuity":"same_action|new_action|possible_ellipsis|unknown",
"information_added":"... or unknown","limitations":[]}],"unresolved":[]}
Input: """

AV_PROMPT = """Inspect this short original-sound clip. The shot intervals are
clip-relative. Observe each shot's entry state, visible action, exit state and
first visible change time. Compare adjacent shots without treating an edit as
continuous uncut footage. On-screen statements listed below are attributed to
the video, not independent visual facts; assign only their supplied IDs to
matching shots. Do not quote speech without an A-channel transcript. Audible
event type may be music, speech, sound_effect, other, or unknown. An audio
energy onset is not a verified musical beat. Do not invent causation or a
story role. Output the same reference_local_observation_v3 schema as below,
with attributed_statement_ids and audible_event_type additionally on each
shot. Report change times relative to this clip.
{"schema_version":"reference_local_observation_v3","shots":[
{"shot_id":"...","entry_state":"... or unknown","visible_action":"... or unknown",
"exit_state":"... or unknown","changes":[{"description":"...",
"kind":"action|posture|contact|identity|causal|other","first_visible_s":0.0,
"epistemic_status":"supported|contested|insufficient"}],
"attributed_statement_ids":[],"audible_event_type":"unknown","limitations":[]}],
"connections":[{"from_shot_id":"...","to_shot_id":"...",
"continuity":"same_action|new_action|possible_ellipsis|unknown",
"information_added":"... or unknown","limitations":[]}],"unresolved":[]}
Input: """

VERIFY_PROMPT = """Inspect the original media afresh. Do not assume a prior
description is correct; it is deliberately withheld. For every supplied shot
state only what is directly visible and locate any first visible change in
clip-relative seconds. If the sampled frames do not establish a change, say
insufficient. This is a neutral independent local observation, not an answer
check. Return the reference_local_observation_v3 JSON schema with shots,
connections and unresolved, exactly as in this template:
{"schema_version":"reference_local_observation_v3","shots":[
{"shot_id":"...","entry_state":"...","visible_action":"...",
"exit_state":"...","changes":[],"limitations":[]}],
"connections":[{"from_shot_id":"...","to_shot_id":"...",
"continuity":"same_action|new_action|possible_ellipsis|unknown",
"information_added":"...","limitations":[]}],"unresolved":[]}
Input: """

STORY_PROMPT = """Form an open-ended story graph from accepted evidence and
provisional local observations. Organize shot changes into temporal events,
then propose cross-event relations and possible communicative stance. Explain
when a visible result FIRST appears; distinguish observed state from a claimed
cause or formal outcome. Screen text is what the work states, never a visual
fact. The ending may alter an earlier reading, or not. Do not assume reversal,
fixed section roles, a moral, or a specific subject. Keep an alternative and
uncertainty for each stance hypothesis. A local observation with insufficient
sampling cannot be sole support for a motion claim. Cite accepted claim/event
IDs or local probe:shot IDs; use unknown when necessary. Return JSON only:
{"schema_version":"reference_story_graph_v3","events":[{"event_id":"...",
"shot_ids":[],"description":"...","first_visible_s":null,
"support_ids":[],"local_refs":[],"status":"provisional|insufficient"}],
"relations":[{"relation_id":"...","source":"...","target":"...",
"description":"...","support_ids":[],"local_refs":[],"status":"provisional|insufficient"}],
"stance_hypotheses":[{"hypothesis_id":"...","stance":"...",
"viewer_information_path":"...","ending_relation":"...",
"support_ids":[],"local_refs":[],"alternative":"...","uncertainty":"..."}],
"unresolved":[]}
Input: """

EDIT_FUNCTION_PROMPT = """For each observed adjacent edit, hypothesize what
new information the next shot contributes and why retaining that edit may
matter. Keep the measured cut/transition and the inferred function separate.
Use unknown where evidence is too sparse. A rapid sequence is not by itself
montage, and an energy onset is not a verified beat. Do not claim a film-grammar
term without content from both sides. Return JSON only:
{"schema_version":"reference_editing_functions_v3","edges":[
{"from_shot_id":"...","to_shot_id":"...","information_added":"...",
"function_hypothesis":"... or unknown","alternative":"...",
"support_probe_ids":[],"status":"provisional|insufficient"}],
"limitations":[]}
Input: """

COHERENCE_PROMPT = """Check only internal consistency of the story graph
against the supplied attributed text, accepted evidence and local observation
statuses. Identify unsupported jumps, temporal contradictions, and instances
where a text statement is promoted to visual reality. This text-only check
CANNOT verify whether a media observation is visually true. Do not require a
particular theme or relation. Return JSON only:
{"schema_version":"reference_story_coherence_v3","internally_consistent":false,
"issues":[{"code":"...","node_id":"...","reason":"..."}],
"visual_truth_verified":false}
Input: """

COUNTERFACTUAL_PROMPT = """Read these ordered, anonymous evidence bundles
as a separate hypothetical edit. For each bundle, say what information it
contributes, then describe any supported cross-bundle relations and possible
communicative stance. Text is an attributed statement, not visual fact.
There is no reference story, answer key, or expected relation. Do not refer
to material absent from the input. Relations may be empty. Return JSON only:
{"schema_version":"reference_counterfactual_reading_v3",
"bundle_readings":[{"ordinal":1,"contribution":"..."}],
"relations":[{"from_ordinal":1,"to_ordinal":2,"description":"..."}],
"stance_hypothesis":"... or unknown","ending_reading":"... or unknown",
"limitations":[]}
Input: """


def _shots(static: dict[str, Any]) -> list[dict[str, Any]]:
    return static["shots"]


def plan_probes(static: dict[str, Any], reference: dict[str, Any]
                ) -> tuple[list[dict[str, Any]], list[str]]:
    """Coverage-first schedule; no case-specific answer or human gold."""
    shots = _shots(static)
    positions = {row["shot_id"]: i for i, row in enumerate(shots)}
    issues = build_review_issues(static, reference)
    selected: list[dict[str, Any]] = []
    omitted: list[str] = []
    seen: set[tuple[str, str, tuple[str, ...], float]] = set()
    per_issue: Counter[str] = Counter()

    def add(issue_id: str, channel: str, ids: list[str], fps: float,
            purpose: str) -> None:
        if len(ids) < 2:
            omitted.append(f"{issue_id}:fewer_than_two_shots")
            return
        key = (issue_id, channel, tuple(ids), fps)
        if key in seen:
            omitted.append(f"{issue_id}:duplicate_scope_mode")
            return
        # Leave one call for a neutral, differently sampled verification of
        # a high-impact claim discovered during observation.
        if len(selected) >= MAX_LOCAL_MEDIA_CALLS - 1 or per_issue[issue_id] >= MAX_CALLS_PER_ISSUE:
            omitted.append(f"{issue_id}:budget")
            return
        indices = [positions[value] for value in ids]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("probe_shots_not_consecutive")
        seen.add(key)
        per_issue[issue_id] += 1
        selected.append({"probe_id": f"probe_{len(selected)+1:02d}",
                         "issue_id": issue_id, "channel": channel,
                         "shot_ids": ids, "fps": fps, "purpose": purpose,
                         "interval": [float(shots[indices[0]]["start_s"]),
                                      float(shots[indices[-1]]["end_s"])]})

    for issue in issues:
        if issue["reason_code"] != "one_action_claim_spans_multiple_shots":
            continue
        ids = issue["shot_ids"]
        # At most two overlapping windows: cover the full coarse claim first.
        windows = ([ids] if len(ids) <= 4 else [ids[:4], ids[-4:]])
        for window in windows:
            add(issue["issue_id"], "V", window, 12.0, "coarse_action_coverage")
        # Link the final action shot to later result-state shots through
        # accepted event IDs, without pre-declaring what the result means.
        final = positions[ids[-1]]
        related_event_ids = set(shots[final].get("event_ids") or [])
        if related_event_ids:
            end = final
            while end + 1 < len(shots) and end - final < 2 and (
                    related_event_ids & set(shots[end + 1].get("event_ids") or [])):
                end += 1
            if end > final:
                add(f"outcome_link_{issue['issue_id']}", "V",
                    [row["shot_id"] for row in shots[max(0, final-1):end+1]],
                    12.0, "action_to_later_state")
    for issue in issues:
        if issue["reason_code"] == "final_text_relation_unverified":
            add(issue["issue_id"], "AV", issue["shot_ids"], 8.0,
                "terminal_statement_connection")
    for issue in issues:
        if issue["reason_code"] == "cross_section_connection_unverified":
            add(issue["issue_id"], "AV", issue["shot_ids"], 8.0,
                "cross_section_edit")
    # The coarse-action section was already sampled densely; spend the
    # remaining local capacity on an edit sequence not yet covered.
    visually_covered = {sid for probe in selected if probe["channel"] == "V"
                        for sid in probe["shot_ids"]}
    for issue in reversed(issues):
        if issue["reason_code"] == "adjacent_edit_meaning_unverified":
            ids = issue["shot_ids"]
            if len(set(ids) & visually_covered) >= len(ids) / 2:
                continue
            windows = ([ids] if float(shots[positions[ids[-1]]]["end_s"]) -
                       float(shots[positions[ids[0]]]["start_s"]) <= 8.0 else
                       [ids[i:i+4] for i in range(0, len(ids)-1, 3)])
            for window in windows:
                add(issue["issue_id"], "AV", window, 8.0,
                    "dense_edit_connection")
    return selected, omitted


def validate_probe_plan(probes: list[dict[str, Any]],
                        static: dict[str, Any]) -> None:
    if len(probes) > MAX_LOCAL_MEDIA_CALLS:
        raise ValueError("media_budget_exceeded")
    ordered = [row["shot_id"] for row in _shots(static)]
    signatures = set()
    counts: Counter[str] = Counter()
    for probe in probes:
        ids = probe["shot_ids"]
        indices = [ordered.index(value) for value in ids]
        if len(ids) < 2 or indices != list(range(indices[0], indices[0]+len(ids))):
            raise ValueError("probe_scope_invalid")
        signature = (probe["issue_id"], probe["channel"], tuple(ids), probe["fps"])
        if signature in signatures:
            raise ValueError("duplicate_probe")
        signatures.add(signature)
        counts[probe["issue_id"]] += 1
        if counts[probe["issue_id"]] > MAX_CALLS_PER_ISSUE:
            raise ValueError("issue_budget_exceeded")


def select_neutral_verification(observations: list[dict[str, Any]],
                                static: dict[str, Any]
                                ) -> dict[str, Any] | None:
    """Spend at most one reserved call on a localized, high-impact change."""
    if len(observations) >= MAX_LOCAL_MEDIA_CALLS:
        return None
    counts = Counter(row["issue_id"] for row in observations)
    ordered = [row["shot_id"] for row in static["shots"]]
    index = {row["shot_id"]: row for row in static["shots"]}
    candidates = []
    for record in observations:
        if record["channel"] != "V" or counts[record["issue_id"]] >= 2:
            continue
        for finding in record["validation"]["change_findings"]:
            if finding["kind"] not in {"posture", "contact", "identity", "causal"}:
                continue
            if finding["declared_status"] == "insufficient":
                continue
            candidates.append((0 if record["purpose"] == "action_to_later_state"
                               else 1, 0 if finding["kind"] == "posture" else 1,
                               ordered.index(finding["shot_id"]), record, finding))
    if not candidates:
        return None
    _, _, position, source, finding = min(candidates, key=lambda row: row[:3])
    lo = max(0, position-1)
    hi = min(len(ordered), position+2)
    if hi-lo < 2:
        return None
    ids = ordered[lo:hi]
    interval = [float(index[ids[0]]["start_s"]), float(index[ids[-1]]["end_s"])]
    return {"probe_id": f"probe_{len(observations)+1:02d}",
            "issue_id": source["issue_id"], "channel": "V",
            "shot_ids": ids, "interval": interval, "fps": 16.0,
            "purpose": "neutral_verify", "prior_probe_id": source["probe_id"],
            "selected_change_kind": finding["kind"],
            "selected_shot_id": finding["shot_id"]}


def _probe_payload(probe: dict[str, Any], static: dict[str, Any]
                   ) -> dict[str, Any]:
    index = {row["shot_id"]: row for row in _shots(static)}
    origin = probe["interval"][0]
    payload: dict[str, Any] = {
        "interval": probe["interval"],
        "shots": [{"shot_id": sid, "relative_interval": [
            round(float(index[sid]["start_s"])-origin, 6),
            round(float(index[sid]["end_s"])-origin, 6)],
            "boundary_type_before": ("transition" if index[sid].get("transition_in")
                                     else "cut_or_section_start")}
                  for sid in probe["shot_ids"]],
    }
    if probe["channel"] == "AV":
        text = {row["claim_id"]: row for row in static["text_timeline"]}
        payload["text_statements"] = [
            {"shot_id": sid, "claim_id": cid,
             "observed_text": text[cid]["observed_text"],
             "source_type": "attributed_on_screen_text"}
            for sid in probe["shot_ids"]
            for cid in index[sid].get("text_claim_ids") or [] if cid in text]
    return payload


def sampling_from_runner(runner: Any, probe: dict[str, Any]
                         ) -> dict[str, Any]:
    raw = dict(getattr(runner, "_last_sampling", {}) or {})
    relative = raw.get("actual_frame_timestamps_relative_s") or []
    origin = float(probe["interval"][0])
    raw["source_time_origin_s"] = origin
    raw["actual_frame_timestamps_absolute_s"] = [round(origin+float(t), 6)
                                                  for t in relative]
    raw["sampling_verified"] = bool(raw.get("sampling_verified") and relative)
    return raw


def preserve_sampled_source_frames(clip: Path, sampling: dict[str, Any],
                                   directory: Path) -> dict[str, Any]:
    """Reconstruct the audited sampled source frames, not the model tensor."""
    indices = sampling.get("actual_frame_indices") or []
    if not sampling.get("sampling_verified") or not indices:
        manifest = {"status": "sampling_not_verified", "frames": [],
                    "limitation": "exact_model_tensor_not_persisted"}
        _write_json(directory / "manifest.json", manifest)
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{int(i)})" for i in indices)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
         f"select={expression},scale=360:-2", "-vsync", "0", "-start_number", "0",
         "-q:v", "3", str(directory / "frame_%04d.jpg")],
        check=True, capture_output=True)
    files = sorted(directory.glob("frame_*.jpg"))
    relative = sampling["actual_frame_timestamps_relative_s"]
    absolute = sampling["actual_frame_timestamps_absolute_s"]
    rows = [{"frame_index": index, "clip_time_s": relative[n],
             "source_time_s": absolute[n], "file": str(path),
             "sha256": sha256_file(path)}
            for n, (index, path) in enumerate(zip(indices, files))]
    manifest = {"status": "reconstructed_source_frames" if len(files) == len(indices)
                else "frame_count_mismatch",
                "frames": rows,
                "limitation": "FFmpeg reconstruction of audited indices; not byte-identical model tensor"}
    _write_json(directory / "manifest.json", manifest)
    return manifest


def validate_local_observation(value: dict[str, Any], probe: dict[str, Any],
                               payload: dict[str, Any], sampling: dict[str, Any]
                               ) -> dict[str, Any]:
    if value.get("schema_version") != "reference_local_observation_v3":
        raise ValueError("local_schema_invalid")
    ids = probe["shot_ids"]
    rows = value.get("shots") or []
    edges = value.get("connections") or []
    if [row.get("shot_id") for row in rows] != ids:
        raise ValueError("shot_coverage_invalid")
    if [(row.get("from_shot_id"), row.get("to_shot_id")) for row in edges] != list(
            zip(ids, ids[1:])):
        raise ValueError("edge_coverage_invalid")
    samples = [float(t) for t in sampling.get("actual_frame_timestamps_relative_s") or []]
    text_ids = {(r["shot_id"], r["claim_id"])
                for r in payload.get("text_statements") or []}
    findings: list[dict[str, Any]] = []
    for row, scope in zip(rows, payload["shots"]):
        sid = row["shot_id"]
        lo, hi = map(float, scope["relative_interval"])
        # Never borrow a neighbouring shot's sample to establish motion in a
        # very short shot. Only numeric rounding gets a small tolerance.
        in_shot = [t for t in samples if lo-0.001 <= t <= hi+0.001]
        if probe["channel"] == "V" and (row.get("attributed_statement_ids") or
                                          row.get("audible_event_type")):
            raise ValueError("visual_cross_modality")
        if probe["channel"] == "AV":
            if row.get("audible_event_type") not in {
                    "music", "speech", "sound_effect", "other", "unknown"}:
                raise ValueError("audio_type_invalid")
            if any((sid, cid) not in text_ids for cid in
                   row.get("attributed_statement_ids") or []):
                raise ValueError("text_wrong_shot")
        for change in row.get("changes") or []:
            when = change.get("first_visible_s")
            status = change.get("epistemic_status")
            if status not in {"supported", "contested", "insufficient"}:
                raise ValueError("change_status_invalid")
            if change.get("kind") not in {"action", "posture", "contact", "identity",
                                           "causal", "other"}:
                raise ValueError("change_kind_invalid")
            if when is not None and not lo-0.04 <= float(when) <= hi+0.04:
                raise ValueError("change_time_outside_shot")
            aligned = (when is not None and any(abs(float(when)-t) <= 0.09
                                                  for t in in_shot))
            effective = (status if sampling.get("sampling_verified") and
                         len(in_shot) >= MOTION_MIN_SAMPLES and aligned else
                         "insufficient")
            # Causal and identity claims require separate verification; one
            # model observation never promotes them to visual fact.
            if change["kind"] in {"causal", "identity"}:
                effective = "contested" if effective == "supported" else effective
            findings.append({"shot_id": sid, "description": change.get("description"),
                             "kind": change["kind"], "first_visible_s": when,
                             "declared_status": status, "effective_status": effective,
                             "sample_aligned": aligned,
                             "sample_count_in_shot": len(in_shot)})
    for edge in edges:
        if edge.get("continuity") not in {
                "same_action", "new_action", "possible_ellipsis", "unknown"}:
            raise ValueError("continuity_invalid")
    return {"sampling_verified": bool(sampling.get("sampling_verified")),
            "change_findings": findings}


def build_editing_graph(static: dict[str, Any], observations: list[dict[str, Any]]
                        ) -> dict[str, Any]:
    by_edge: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in observations:
        for row in record["response"].get("connections") or []:
            key = row["from_shot_id"], row["to_shot_id"]
            by_edge.setdefault(key, []).append({**row, "probe_id": record["probe_id"],
                                                  "sampling_verified": record[
                                                      "sampling"]["sampling_verified"]})
    edges = []
    for left, right in zip(_shots(static), _shots(static)[1:]):
        key = left["shot_id"], right["shot_id"]
        obs = by_edge.get(key, [])
        labels = {row["continuity"] for row in obs if row["sampling_verified"]}
        status = ("insufficient" if not labels or labels == {"unknown"} else
                  "contested" if len(labels) > 1 else "provisional")
        transition = right.get("transition_in")
        edges.append({"from_shot_id": key[0], "to_shot_id": key[1],
                      "edit_start_s": (float(transition["start_s"]) if transition
                                       else float(right["start_s"])),
                      "next_content_start_s": float(right["start_s"]),
                      "transition": transition, "observations": obs,
                      "semantic_status": status})
    result = {"schema_version": "reference_editing_graph_v3",
              "source_sha": static["source_sha"],
              "static_review_sha": static["artifact_sha"],
              "shots": static["shots"], "transitions": static["transitions"],
              "measured_editing": static["measured_editing"],
              "audio_measurement_status": {
                  "music_beat_status": static["audio"]["music_beat_status"],
                  "beat_synced_status": static["audio"]["beat_synced_status"]},
              "adjacent_edges": edges}
    result["artifact_sha"] = json_hash(result)
    return result


def validate_story_graph(story: dict[str, Any], reference: dict[str, Any],
                         static: dict[str, Any], observations: list[dict[str, Any]]
                         ) -> list[str]:
    errors: list[str] = []
    if story.get("schema_version") != "reference_story_graph_v3":
        return ["schema_invalid"]
    known = {row["claim_id"] for row in reference["claims"]} | {
        row["event_id"] for row in reference["events"]}
    shots = {row["shot_id"] for row in static["shots"]}
    local = {f"{record['probe_id']}:{sid}" for record in observations
             for sid in record["shot_ids"]}
    events = story.get("events") or []
    event_ids = {row.get("event_id") for row in events}
    if None in event_ids or len(event_ids) != len(events):
        errors.append("event_ids_invalid")
    for kind, rows in (("event", events), ("relation", story.get("relations") or []),
                       ("stance", story.get("stance_hypotheses") or [])):
        for row in rows:
            if any(cid not in known for cid in row.get("support_ids") or []):
                errors.append(f"{kind}_support_unknown")
            if any(cid not in local for cid in row.get("local_refs") or []):
                errors.append(f"{kind}_local_ref_unknown")
            if not row.get("support_ids") and not row.get("local_refs"):
                errors.append(f"{kind}_unsupported")
            if kind == "event" and any(sid not in shots for sid in
                                       row.get("shot_ids") or []):
                errors.append("event_shot_unknown")
            if kind == "relation" and (row.get("source") not in event_ids or
                                        row.get("target") not in event_ids):
                errors.append("relation_endpoint_unknown")
    return sorted(set(errors))


def validate_edit_functions(value: dict[str, Any], editing: dict[str, Any]
                            ) -> list[str]:
    if value.get("schema_version") != "reference_editing_functions_v3":
        return ["editing_function_schema_invalid"]
    edges = {(row["from_shot_id"], row["to_shot_id"]): row
             for row in editing["adjacent_edges"]}
    errors = []
    for row in value.get("edges") or []:
        key = row.get("from_shot_id"), row.get("to_shot_id")
        if key not in edges:
            errors.append("editing_edge_unknown")
        elif any(pid not in {obs["probe_id"] for obs in edges[key]["observations"]}
                 for pid in row.get("support_probe_ids") or []):
            errors.append("editing_probe_unknown")
    return sorted(set(errors))


def _observation_view(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep semantics and sampled-frame validity, omit transport metadata."""
    view = []
    for row in observations:
        findings = row["validation"]["change_findings"]
        by_shot: dict[str, list[dict[str, Any]]] = {}
        for finding in findings:
            by_shot.setdefault(finding["shot_id"], []).append({
                "description": finding["description"],
                "kind": finding["kind"],
                "first_visible_s": finding["first_visible_s"],
                "status": finding["effective_status"],
                "sample_count_in_shot": finding["sample_count_in_shot"]})
        shot_rows = []
        for original in row["response"].get("shots") or []:
            shot_rows.append({"shot_id": original["shot_id"],
                              "entry_state": original.get("entry_state"),
                              "visible_action": original.get("visible_action"),
                              "exit_state": original.get("exit_state"),
                              "changes": by_shot.get(original["shot_id"], []),
                              "attributed_statement_ids": original.get(
                                  "attributed_statement_ids") or [],
                              "audible_event_type": original.get("audible_event_type"),
                              "limitations": original.get("limitations") or []})
        view.append({"probe_id": row["probe_id"], "channel": row["channel"],
                     "shot_ids": row["shot_ids"], "interval": row["interval"],
                     "sampling_verified": row["sampling"]["sampling_verified"],
                     "shots": shot_rows,
                     "connections": row["response"].get("connections") or [],
                     "unresolved": row["response"].get("unresolved") or []})
    return view


def _counterfactual_sections(static: dict[str, Any], *,
                             mask_terminal_text: bool = False,
                             reverse_order: bool = False
                             ) -> list[dict[str, Any]]:
    """Evidence-only projection, with no original IDs, times, or model prose."""
    bundles = static["section_bundles"]
    terminal = None
    if mask_terminal_text:
        text = bundles[-1].get("text_statements") or []
        terminal = text[-1].get("observed_text") if text else None
    rows = []
    for bundle in (reversed(bundles) if reverse_order else bundles):
        statements = [
            {"source_type": "attributed_on_screen_text",
             "observed_text": item.get("observed_text")}
            for item in bundle.get("text_statements") or []
            if item.get("observed_text") != terminal]
        rows.append({"ordinal": len(rows)+1,
                     "visible_observations": [
                         {"predicate": item.get("predicate"),
                          "object": item.get("object"),
                          "visibility": item.get("visibility")}
                         for item in bundle.get("visual_observations") or []],
                     "attributed_statements": statements})
    return rows


def run_counterfactual_diagnostics(runner: Any, static: dict[str, Any],
                                   output: Path) -> dict[str, Any]:
    """Text-only sensitivity probes; never counted as media truth or release."""
    readings = {}
    errors = {}
    for name, kwargs in (("base", {}),
                         ("terminal_text_masked", {"mask_terminal_text": True}),
                         ("section_order_reversed", {"reverse_order": True})):
        try:
            readings[name] = _model_call(
                runner, name=f"counterfactual_{name}",
                prompt=COUNTERFACTUAL_PROMPT,
                payload={"ordered_bundles": _counterfactual_sections(static,
                                                                      **kwargs)},
                output=output, max_new_tokens=2048)
        except (ValueError, RuntimeError) as exc:
            errors[name] = f"{type(exc).__name__}:{exc}"
    terminal_rows = static["section_bundles"][-1].get("text_statements") or []
    terminal_text = str(terminal_rows[-1].get("observed_text")) if terminal_rows else ""
    masked = readings.get("terminal_text_masked")
    base = readings.get("base")
    reversed_reading = readings.get("section_order_reversed")
    result = {"schema_version": "reference_counterfactual_diagnostics_v3",
              "status": "text_only_sensitivity_not_accuracy",
              "model_call_errors": errors,
              "terminal_text_exact_reappeared_after_mask": (
                  terminal_text.casefold() in json.dumps(masked, ensure_ascii=False).casefold()
                  if masked is not None and terminal_text else None),
              "base_reading_sha": json_hash(base) if base else None,
              "masked_reading_sha": json_hash(masked) if masked else None,
              "reversed_reading_sha": json_hash(reversed_reading)
              if reversed_reading else None,
              "order_change_output_identical": (base == reversed_reading
                                                if base and reversed_reading else None),
              "limitation": "Exact-text and whole-output checks do not establish semantic accuracy"}
    _write_json(output / "counterfactual_diagnostics.json", result)
    return result


def _call_metrics(directory: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((directory / "calls").glob("*/model_call.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return {"model_call_count": len(rows),
            "input_tokens": (sum(row["input_tokens"] for row in rows)
                             if rows and all(isinstance(row.get("input_tokens"), int)
                                             for row in rows) else None),
            "output_tokens": (sum(row["output_tokens"] for row in rows)
                              if rows and all(isinstance(row.get("output_tokens"), int)
                                              for row in rows) else None),
            "elapsed_s": (round(sum(row["elapsed_s"] for row in rows), 3)
                          if rows and all(isinstance(row.get("elapsed_s"), (int, float))
                                          for row in rows) else None)}


def build_baseline_comparison(baseline_path: Path, output: Path,
                              static: dict[str, Any], reference: dict[str, Any],
                              observations: list[dict[str, Any]],
                              editing: dict[str, Any], story: dict[str, Any],
                              counterfactual: dict[str, Any]
                              ) -> dict[str, Any]:
    baseline_dir = baseline_path.parent
    baseline_obs_path = baseline_dir / "local_observations.json"
    baseline_edit_path = baseline_dir / "editing_graph_v2.json"
    baseline_story_path = baseline_dir / "story_graph_v2.json"
    base_obs = json.loads(baseline_obs_path.read_text(encoding="utf-8"))[
        "observations"] if baseline_obs_path.exists() else []
    base_edit = (json.loads(baseline_edit_path.read_text(encoding="utf-8"))
                 if baseline_edit_path.exists() else {})
    base_story = (json.loads(baseline_story_path.read_text(encoding="utf-8"))
                  if baseline_story_path.exists() else {})
    issues = build_review_issues(static, reference)
    action_shots = {sid for issue in issues if issue["reason_code"] ==
                    "one_action_claim_spans_multiple_shots"
                    for sid in issue["shot_ids"]}
    def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
        seen = {sid for row in rows if row.get("channel") == "V"
                for sid in row["shot_ids"]}
        return {"covered_count": len(seen & action_shots),
                "total_count": len(action_shots),
                "uncovered_shot_ids": sorted(action_shots - seen)}
    def duplicate_count(rows: list[dict[str, Any]]) -> int:
        signatures = [(row["issue_id"], row["channel"],
                       tuple(row["shot_ids"]), row.get("fps")) for row in rows]
        return len(signatures) - len(set(signatures))
    localizations = [f for row in observations
                     for f in row["validation"]["change_findings"]
                     if f["effective_status"] != "insufficient" and
                     f["first_visible_s"] is not None]
    def edge_counts(graph: dict[str, Any]) -> dict[str, int]:
        return dict(Counter(row.get("semantic_status", "unknown")
                            for row in graph.get("adjacent_edges") or []))
    result = {"schema_version": "reference_understanding_v3_comparison",
              "status": "model_consistency_comparison_not_accuracy",
              "known_shots_and_cuts_are_inputs_not_new_detections": True,
              "baseline_v2": {"media_call_count": len(base_obs),
                              "call_metrics": _call_metrics(baseline_dir),
                              "coarse_action_shot_coverage": coverage(base_obs),
                              "duplicate_requests": duplicate_count(base_obs),
                              "edge_status_counts": edge_counts(base_edit),
                              "narrative_unit_count": len(base_story.get(
                                  "narrative_units") or []),
                              "relation_count": len(base_story.get("relations") or [])},
              "v3": {"media_call_count": len(observations),
                     "call_metrics": _call_metrics(output),
                     "coarse_action_shot_coverage": coverage(observations),
                     "duplicate_requests": duplicate_count(observations),
                     "localized_change_count": len(localizations),
                     "edge_status_counts": edge_counts(editing),
                     "event_count": len(story.get("events") or []),
                     "relation_count": len(story.get("relations") or []),
                     "counterfactual_diagnostics_sha": json_hash(counterfactual)},
              "limitation": "No independent human ground truth; no accuracy claim"}
    _write_json(output / "v2_v3_comparison.json", result)
    return result


def _call_media(runner: Any, probe: dict[str, Any], static: dict[str, Any],
                video: Path, masked_video: Path, output: Path
                ) -> dict[str, Any]:
    payload = _probe_payload(probe, static)
    clip = cut_clip("ffmpeg", masked_video if probe["channel"] == "V" else video,
                    output / "clips" / probe["probe_id"],
                    start_s=probe["interval"][0], end_s=probe["interval"][1],
                    include_audio=probe["channel"] == "AV")
    value = _model_call(runner, name=probe["probe_id"],
                        prompt=(VERIFY_PROMPT if probe["purpose"] == "neutral_verify"
                                else V_PROMPT if probe["channel"] == "V" else AV_PROMPT),
                        payload=payload, output=output, media=clip,
                        channel=probe["channel"], fps=probe["fps"],
                        max_new_tokens=3072)
    sampling = sampling_from_runner(runner, probe)
    _write_json(output / "calls" / probe["probe_id"] / "sampling_audit.json",
                sampling)
    frame_manifest = preserve_sampled_source_frames(
        clip, sampling, output / "calls" / probe["probe_id"] / "sampled_frames")
    if frame_manifest["status"] == "frame_count_mismatch":
        sampling["sampling_verified"] = False
        sampling["sampling_error"] = "source_frame_reconstruction_count_mismatch"
        _write_json(output / "calls" / probe["probe_id"] / "sampling_audit.json",
                    sampling)
    validation = validate_local_observation(value, probe, payload, sampling)
    validation["sampled_frames_status"] = frame_manifest["status"]
    _write_json(output / "calls" / probe["probe_id"] / "validation.json",
                {"status": "PASS", **validation})
    return {**probe, "media_sha": sha256_file(clip), "sampling": sampling,
            "validation": validation, "response": value}


def run_reference_understanding_v3(
        reference_path: Path, video: Path, masked_video: Path,
        baseline_v2_path: Path, output: Path, runner: Any,
        *, asr_path: Path | None = None) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_directory_not_empty")
    reference = load_reference(reference_path)
    baseline = json.loads(Path(baseline_v2_path).read_text(encoding="utf-8"))
    if baseline.get("schema_version") != "reference_understanding_v2" or (
            baseline.get("source_sha") != reference["source_sha"]):
        raise ValueError("baseline_v2_source_mismatch")
    mask = json.loads((masked_video.parent / "visual_mask_audit.json")
                      .read_text(encoding="utf-8"))
    if (mask.get("source_sha") != reference["source_sha"] or
            mask.get("artifact_sha") != sha256_file(masked_video) or
            mask.get("streams") != ["video"]):
        raise ValueError("masked_media_contract_invalid")
    static = prepare_readout(reference_path, video, output, asr_path=asr_path)
    probes, omitted = plan_probes(static, reference)
    validate_probe_plan(probes, static)
    _write_json(output / "probe_plan.json", {
        "schema_version": "reference_probe_plan_v3", "probes": probes,
        "omitted": omitted, "max_local_media_calls": MAX_LOCAL_MEDIA_CALLS,
        "baseline_v2_sha": sha256_file(baseline_v2_path),
        "human_gold_in_model_requests": False})
    observations = []
    for probe in probes:
        observations.append(_call_media(runner, probe, static, video,
                                        masked_video, output))
        _write_json(output / "local_observations.json", {
            "schema_version": "reference_local_observations_v3",
            "observations": observations})
    neutral = select_neutral_verification(observations, static)
    if neutral is not None:
        validate_probe_plan(probes + [neutral], static)
        _write_json(output / "neutral_verification_selection.json", neutral)
        observations.append(_call_media(runner, neutral, static, video,
                                        masked_video, output))
        _write_json(output / "local_observations.json", {
            "schema_version": "reference_local_observations_v3",
            "observations": observations})
    editing = build_editing_graph(static, observations)
    _write_json(output / "editing_graph_v3.json", editing)
    story = _model_call(runner, name="story_graph", prompt=STORY_PROMPT,
                        payload={"section_bundles": static["section_bundles"],
                                 "accepted_events": reference["events"],
                                 "text_timeline": static["text_timeline"],
                                 "local_observations": _observation_view(observations),
                                 "editing_edges": editing["adjacent_edges"]},
                        output=output, max_new_tokens=4096)
    story_errors = validate_story_graph(story, reference, static, observations)
    _write_json(output / "story_graph_v3.json", story)
    _write_json(output / "story_validation.json", {
        "status": "PASS" if not story_errors else "FAIL", "issues": story_errors})
    edit_functions = _model_call(
        runner, name="editing_functions", prompt=EDIT_FUNCTION_PROMPT,
        payload={"measured_editing": static["measured_editing"],
                 "audio_status": editing["audio_measurement_status"],
                 "adjacent_edges": editing["adjacent_edges"],
                 "story_events": story.get("events") or []},
        output=output, max_new_tokens=3072)
    function_errors = validate_edit_functions(edit_functions, editing)
    _write_json(output / "editing_functions_v3.json", edit_functions)
    _write_json(output / "editing_function_validation.json", {
        "status": "PASS" if not function_errors else "FAIL",
        "issues": function_errors})
    coherence = _model_call(
        runner, name="story_coherence", prompt=COHERENCE_PROMPT,
        payload={"accepted_evidence": static["section_bundles"],
                 "text_timeline": static["text_timeline"],
                 "local_observations": _observation_view(observations),
                 "story_graph": story},
        output=output, max_new_tokens=2048)
    _write_json(output / "story_coherence_v3.json", coherence)
    coherence_ok = (coherence.get("schema_version") ==
                    "reference_story_coherence_v3" and
                    coherence.get("internally_consistent") is True and
                    coherence.get("visual_truth_verified") is False)
    counterfactual = run_counterfactual_diagnostics(runner, static, output)
    # A theme brief is a private, non-production candidate only. Direct media
    # text and identifiers are never copied into this whitelist structure.
    brief = None
    if not story_errors and coherence_ok and story.get("stance_hypotheses"):
        hypothesis = story["stance_hypotheses"][0]
        brief = {"schema_version": "reference_theme_brief_candidate_v3",
                 "status": "model_checked_candidate_only",
                 "communicative_stance": hypothesis.get("stance"),
                 "viewer_information_change": hypothesis.get(
                     "viewer_information_path"),
                 "ending_relation": hypothesis.get("ending_relation"),
                 "relative_story_timing": [{
                     "start_fraction": round(row["interval"][0]/static["duration_s"], 4),
                     "end_fraction": round(row["interval"][1]/static["duration_s"], 4)}
                     for row in static["section_bundles"]],
                 "free_slots": ["domain", "character_relation", "activity",
                                "setting", "evidence_form"],
                 "production_release_allowed": False}
        leak_rules = _transfer_leaks(brief, reference, video)
        if leak_rules:
            brief = None
            _write_json(output / "theme_brief_rejection.json", {
                "status": "BLOCKED", "rule_codes": leak_rules})
        else:
            _write_json(output / "theme_brief_candidate_v3.json", brief)
    comparison = build_baseline_comparison(
        Path(baseline_v2_path), output, static, reference, observations,
        editing, story, counterfactual)
    result = {"schema_version": VERSION, "status": "model_review_candidate",
              "source_sha": reference["source_sha"],
              "baseline_v2_sha": sha256_file(baseline_v2_path),
              "static_review_sha": static["artifact_sha"],
              "local_media_call_count": len(observations),
              "media_budget": MAX_LOCAL_MEDIA_CALLS,
              "probe_omissions": omitted,
              "story_validation_issues": story_errors,
              "editing_function_validation_issues": function_errors,
              "coherence_positive": coherence_ok,
              "counterfactual_diagnostics_sha": json_hash(counterfactual),
              "comparison_sha": json_hash(comparison),
              "theme_brief_candidate_generated": brief is not None,
              "story_ready": False, "editing_ready": False,
              "production_release_allowed": False,
              "human_truth_confirmed": False}
    result["artifact_sha"] = json_hash(result)
    _write_json(output / "reference_understanding_v3.json", result)
    return result
