"""Question-driven reference review, isolated from the production creative DAG.

The accepted timeline and claims remain immutable. Model observations in this
module are hypotheses until a separate human review accepts them.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.reference_readout import (
    _model_call, _write_json, load_reference, prepare_readout,
)
from src.perception.omni_runner import cut_clip

VERSION = "reference_understanding_v2"
MAX_LOCAL_PROBES = 8
MAX_PROBES_PER_ISSUE = 2

PLAN_PROMPT = """You are planning a small number of local observations of an
edited reference video. Select questions from the supplied unresolved issue
IDs; do not answer them. Prefer issues that affect the story interpretation or
the meaning of a key edit, using the supplied priority numbers. Select at most
four probes in this round. For each
probe, choose two to four consecutive shot IDs from that issue. The same issue
may be inspected at most twice across the entire run. Do not request a full
video replay. No human answer key is available. Return JSON only:
{"schema_version":"reference_probe_selection_v2","probes":[
{"issue_id":"...","shot_ids":["...","..."],"reason":"..."}]}
Input: """

VISUAL_PROBE_PROMPT = """Inspect only this silent, text-masked video clip.
Describe the visible action in each supplied shot, then compare adjacent shots.
Say whether motion visibly continues, differs, repeats, or cannot be decided.
Do not infer dialogue, text, identity, intent, causation, or unseen activity.
Masked regions are unavailable and cannot support absence claims.
The shot intervals below are clip-relative; do not output numerical times.
Return JSON only:
{"schema_version":"reference_local_observation_v2","observations":[
{"shot_id":"...","visible_action":"... or null",
"visible_result":"... or null","uncertainty":"..."}],"connections":[
{"from_shot_id":"...","to_shot_id":"...",
"observed_continuity":"... or unknown",
"information_added":"... or unknown",
"possible_function":null,"uncertainty":"..."}],
"unresolved":["..."]}
Input: """

AV_PROBE_PROMPT = """Inspect this short original-sound clip and its supplied
adjacent shot IDs. Distinguish visible actions, attributed on-screen statements,
audible events, and inferred editing functions. An edit can omit time, but do
not claim an omission or direct causation unless the media supports it. A fast
sequence alone does not prove montage; an audio-energy peak does not prove a
musical beat. Do not use a predetermined story template or output numerical
times. Return JSON only:
{"schema_version":"reference_local_observation_v2","observations":[
{"shot_id":"...","visible_action":"... or null",
"visible_result":"... or null","attributed_statement":"... or null",
"audible_event":"... or null","uncertainty":"..."}],"connections":[
{"from_shot_id":"...","to_shot_id":"...",
"observed_continuity":"... or unknown",
"information_added":"... or unknown",
"possible_function":"... or null","uncertainty":"..."}],
"unresolved":["..."]}
Input: """

REVISE_PROMPT = """Re-evaluate the preliminary reading using the accepted
evidence and the new local observations. The preliminary reading is a fallible
hypothesis, not a fact. Build section/event-level narrative units and only
supported relations. Explain what the work may communicate, how the viewer's
information changes, and how the ending affects the earlier reading. The
video's on-screen text is attributed speech by the work, not independent
physical proof. Separate visible facts, stated claims, and interpretations.
Do not force a reversal, three-act shape, moral, or specific topic. Cite
accepted claim/event IDs; local observations may be cited by probe and shot ID.
Write each local observation reference as "<probe_id>:<shot_id>".
If uncertainty remains, return the relevant unresolved issue IDs from the
provided issue list. Return JSON only:
{"schema_version":"reference_story_graph_v2","narrative_units":[
{"unit_id":"...","section_ids":[],"summary":"...","support_ids":[],
"local_observation_refs":[]}],"relations":[
{"relation_id":"...","source":"...","target":"...",
"description":"...","support_ids":[],"local_observation_refs":[]}],
"theme_hypotheses":[{"hypothesis_id":"...","topic":"...",
"stance":"...","viewer_information_path":"...",
"ending_tone":"...","support_ids":[],"alternative":"...",
"uncertainty":"..."}],"unresolved_issue_ids":[],"limitations":[]}
Input: """

AUDIT_PROMPT = """Independently check whether the revised narrative units,
relations, and theme hypotheses are supported by the cited accepted evidence
and local media observations. Check attribution and temporal scope. This is
grounding review, not a salience contest: do not require any particular theme
or relation. Treat intent and tone as hypotheses, not directly observed facts.
Return JSON only:
{"schema_version":"reference_story_audit_v2","checks":[
{"kind":"unit|relation|theme","id":"...","grounded":false,
"attribution_preserved":false,"temporal_scope_valid":false,
"reason":"..."}],"limitations":[]}
Input: """


def _shot_index(static: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["shot_id"]): row for row in static["shots"]}


def build_review_issues(static: dict[str, Any],
                        reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Find generic evidence gaps; never encode an expected story answer."""
    shots = static["shots"]
    claims = {str(row["claim_id"]): row for row in reference["claims"]}
    issues: list[dict[str, Any]] = []
    visual_counts = Counter(str(value) for shot in shots
                            for value in shot.get("visual_claim_ids") or [])
    for claim_id, count in sorted(visual_counts.items()):
        if count < 3 or claims.get(claim_id, {}).get("predicate") != "performs_action":
            continue
        scope = [shot["shot_id"] for shot in shots
                 if claim_id in shot.get("visual_claim_ids", [])]
        issues.append({"issue_id": f"coarse_action_{claim_id}",
                       "reason_code": "one_action_claim_spans_multiple_shots",
                       "priority": 1, "channel": "V", "shot_ids": scope,
                       "claim_ids": [claim_id]})
    sections = static["section_bundles"]
    if sections:
        last = sections[-1]
        final_shots = [shot["shot_id"] for shot in shots
                       if shot["section_id"] == last["section_id"]]
        if final_shots and any(row["interval"][0] >= last["interval"][0]
                               for row in static["text_timeline"]):
            issues.append({"issue_id": "terminal_statement_relation",
                           "reason_code": "final_text_relation_unverified",
                           "priority": 1, "channel": "AV", "shot_ids": (
                               final_shots[-4:] if len(final_shots) >= 2 else
                               [shot["shot_id"] for shot in shots[-2:]]),
                           "claim_ids": []})
    for left, right in zip(sections, sections[1:]):
        left_shots = [shot["shot_id"] for shot in shots
                      if shot["section_id"] == left["section_id"]]
        right_shots = [shot["shot_id"] for shot in shots
                       if shot["section_id"] == right["section_id"]]
        if left_shots and right_shots:
            issues.append({"issue_id": f"boundary_{left['section_id']}_to_"
                                      f"{right['section_id']}",
                           "reason_code": "cross_section_connection_unverified",
                           "priority": 2, "channel": "AV",
                           "shot_ids": left_shots[-1:] + right_shots[:1],
                           "claim_ids": []})
    for section in sections:
        section_shots = [shot["shot_id"] for shot in shots
                         if shot["section_id"] == section["section_id"]]
        if len(section_shots) >= 4:
            issues.append({"issue_id": f"dense_sequence_{section['section_id']}",
                           "reason_code": "adjacent_edit_meaning_unverified",
                           "priority": 2, "channel": "AV", "shot_ids": section_shots,
                           "claim_ids": []})
    return issues


def validate_probe_selection(selection: dict[str, Any],
                             issues: list[dict[str, Any]],
                             static: dict[str, Any],
                             prior_counts: Counter[str] | None = None
                             ) -> list[dict[str, Any]]:
    if selection.get("schema_version") != "reference_probe_selection_v2":
        raise ValueError("probe_selection_schema_invalid")
    rows = selection.get("probes")
    if not isinstance(rows, list) or len(rows) > 4:
        raise ValueError("probe_selection_count_invalid")
    issue_index = {row["issue_id"]: row for row in issues}
    shot_index = _shot_index(static)
    ordered = [row["shot_id"] for row in static["shots"]]
    counts = Counter(prior_counts or {})
    accepted = []
    for row in rows:
        issue_id = str(row.get("issue_id") or "")
        issue = issue_index.get(issue_id)
        ids = row.get("shot_ids")
        if issue is None or not isinstance(ids, list) or not 2 <= len(ids) <= 4:
            raise ValueError("probe_issue_or_scope_invalid")
        if any(value not in issue["shot_ids"] or value not in shot_index
               for value in ids):
            raise ValueError("probe_shot_outside_issue")
        positions = [ordered.index(value) for value in ids]
        if positions != list(range(positions[0], positions[0] + len(ids))):
            raise ValueError("probe_shots_not_consecutive")
        counts[issue_id] += 1
        if counts[issue_id] > MAX_PROBES_PER_ISSUE:
            raise ValueError("probe_issue_budget_exceeded")
        interval = [float(shot_index[ids[0]]["start_s"]),
                    float(shot_index[ids[-1]]["end_s"])]
        accepted.append({"probe_id": f"probe_{sum(counts.values()):02d}",
                         "issue_id": issue_id, "channel": issue["channel"],
                         "shot_ids": ids, "interval": interval,
                         "planner_reason": str(row.get("reason") or "")})
    if sum(counts.values()) > MAX_LOCAL_PROBES:
        raise ValueError("probe_total_budget_exceeded")
    return accepted


def _clip_context(probe: dict[str, Any], static: dict[str, Any]) -> dict[str, Any]:
    index = _shot_index(static)
    origin = probe["interval"][0]
    return {
        "interval": probe["interval"],
        "clip_duration_s": round(probe["interval"][1] - origin, 6),
        "shots": [{"shot_id": shot_id,
                   "relative_interval": [
                       round(float(index[shot_id]["start_s"]) - origin, 6),
                       round(float(index[shot_id]["end_s"]) - origin, 6)]}
                  for shot_id in probe["shot_ids"]],
    }


def validate_local_observation(value: dict[str, Any],
                               probe: dict[str, Any]) -> None:
    if value.get("schema_version") != "reference_local_observation_v2":
        raise ValueError("local_observation_schema_invalid")
    ids = probe["shot_ids"]
    observed = value.get("observations")
    connections = value.get("connections")
    if not isinstance(observed, list) or not isinstance(connections, list):
        raise ValueError("local_observation_shape_invalid")
    if [row.get("shot_id") for row in observed] != ids:
        raise ValueError("local_observation_shot_coverage_invalid")
    if [(row.get("from_shot_id"), row.get("to_shot_id")) for row in
            connections] != list(zip(ids, ids[1:])):
        raise ValueError("local_observation_connection_coverage_invalid")
    if probe["channel"] == "V" and any(
            row.get("attributed_statement") or row.get("audible_event")
            for row in observed):
        raise ValueError("visual_observation_crosses_modality")


def build_editing_graph(static: dict[str, Any],
                        observations: list[dict[str, Any]]) -> dict[str, Any]:
    shots = static["shots"]
    observed_connections: dict[tuple[str, str], list[dict[str, Any]]] = {}
    observed_shots: dict[str, list[dict[str, Any]]] = {}
    for record in observations:
        for row in record["response"].get("observations") or []:
            observed_shots.setdefault(row["shot_id"], []).append({
                **row, "probe_id": record["probe_id"],
                "media_sha": record["media_sha"]})
        for row in record["response"].get("connections") or []:
            key = (row["from_shot_id"], row["to_shot_id"])
            observed_connections.setdefault(key, []).append({
                **row, "probe_id": record["probe_id"],
                "media_sha": record["media_sha"]})
    edges = []
    for left, right in zip(shots, shots[1:]):
        key = (left["shot_id"], right["shot_id"])
        candidates = observed_connections.get(key, [])
        transition = right.get("transition_in")
        edges.append({
            "from_shot_id": key[0], "to_shot_id": key[1],
            "edit_start_s": (float(transition["start_s"]) if transition else
                             float(right["start_s"])),
            "next_content_start_s": float(right["start_s"]),
            "from_duration_s": float(left["duration_s"]),
            "to_duration_s": float(right["duration_s"]),
            "transition": transition,
            "observations": candidates,
            "semantic_status": ("unobserved" if not candidates else
                                "contested" if len({
                                    str(row.get("observed_continuity"))
                                    for row in candidates}) > 1 else "provisional"),
        })
    sections = [{"section_id": row["section_id"],
                 "interval": row["interval"],
                 "duration_s": round(row["interval"][1] - row["interval"][0], 6),
                 "relative_duration": round(
                     (row["interval"][1] - row["interval"][0]) /
                     static["duration_s"], 6)}
                for row in static["section_bundles"]]
    result = {"schema_version": "reference_editing_graph_v2",
              "source_sha": static["source_sha"],
              "static_review_sha": static["artifact_sha"],
              "sections": sections, "shots": shots,
              "shot_action_observations": [
                  {"shot_id": shot["shot_id"],
                   "observations": observed_shots.get(shot["shot_id"], []),
                   "status": ("provisional" if observed_shots.get(shot["shot_id"])
                              else "unobserved")}
                  for shot in shots],
              "measured_editing": static["measured_editing"],
              "audio_measurement_status": {
                  "music_beat_status": static["audio"]["music_beat_status"],
                  "beat_synced_status": static["audio"]["beat_synced_status"]},
              "adjacent_edges": edges}
    result["artifact_sha"] = json_hash(result)
    return result


def validate_story_graph(story: dict[str, Any],
                         reference: dict[str, Any],
                         issues: list[dict[str, Any]],
                         observations: list[dict[str, Any]]) -> None:
    if story.get("schema_version") != "reference_story_graph_v2":
        raise ValueError("story_graph_schema_invalid")
    known_support = ({str(row["claim_id"]) for row in reference["claims"]} |
                     {str(row["event_id"]) for row in reference["events"]})
    sections = {str(row["section_id"]) for row in
                reference["deterministic_timeline"].get("sections") or []}
    if not sections:
        sections = {str(row["section_id"]) for row in
                    reference["shot_storyboard"]["shot_cards"]}
    local_refs = {f"{record['probe_id']}:{shot_id}"
                  for record in observations for shot_id in record["shot_ids"]}
    units = story.get("narrative_units")
    relations = story.get("relations")
    themes = story.get("theme_hypotheses")
    if not all(isinstance(value, list) for value in (units, relations, themes)):
        raise ValueError("story_graph_shape_invalid")
    unit_ids = {str(row.get("unit_id") or "") for row in units}
    if "" in unit_ids or len(unit_ids) != len(units):
        raise ValueError("story_unit_ids_invalid")
    for kind, rows in (("unit", units), ("relation", relations), ("theme", themes)):
        for row in rows:
            support = row.get("support_ids") or []
            if not support or any(str(value) not in known_support for value in support):
                raise ValueError(f"story_{kind}_support_invalid")
            if any(str(value) not in local_refs for value in
                   row.get("local_observation_refs") or []):
                raise ValueError(f"story_{kind}_local_ref_invalid")
            if kind == "unit" and (not row.get("section_ids") or any(
                    str(value) not in sections for value in row["section_ids"])):
                raise ValueError("story_unit_section_invalid")
            if kind == "relation" and (row.get("source") not in unit_ids or
                                       row.get("target") not in unit_ids):
                raise ValueError("story_relation_unit_invalid")
    issue_ids = {row["issue_id"] for row in issues}
    if any(value not in issue_ids for value in
           story.get("unresolved_issue_ids") or []):
        raise ValueError("story_unknown_unresolved_issue")


def validate_story_audit(audit: dict[str, Any], story: dict[str, Any]) -> bool:
    if audit.get("schema_version") != "reference_story_audit_v2":
        raise ValueError("story_audit_schema_invalid")
    expected = ({("unit", str(row["unit_id"])) for row in story["narrative_units"]} |
                {("relation", str(row["relation_id"])) for row in story["relations"]} |
                {("theme", str(row["hypothesis_id"])) for row in story["theme_hypotheses"]})
    checks = audit.get("checks") or []
    actual = [(str(row.get("kind")), str(row.get("id"))) for row in checks]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("story_audit_coverage_invalid")
    return all(all(row.get(field) is True for field in (
        "grounded", "attribution_preserved", "temporal_scope_valid"))
        for row in checks)


def run_reference_understanding_v2(
        reference_path: Path, video: Path, masked_video: Path,
        baseline_readout_path: Path, output: Path, runner: Any,
        *, asr_path: Path | None = None) -> dict[str, Any]:
    """Review one reference with at most two question-driven local rounds."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_directory_not_empty")
    reference = load_reference(reference_path)
    baseline = json.loads(Path(baseline_readout_path).read_text(encoding="utf-8"))
    if baseline.get("source_sha") != reference["source_sha"]:
        raise ValueError("baseline_source_sha_mismatch")
    mask_audit = json.loads((masked_video.parent / "visual_mask_audit.json")
                            .read_text(encoding="utf-8"))
    if (mask_audit.get("source_sha") != reference["source_sha"] or
            mask_audit.get("artifact_sha") != sha256_file(masked_video) or
            mask_audit.get("streams") != ["video"]):
        raise ValueError("masked_video_contract_invalid")
    static = prepare_readout(reference_path, video, output, asr_path=asr_path)
    issues = build_review_issues(static, reference)
    _write_json(output / "review_issues.json", {
        "schema_version": "reference_review_issues_v2", "issues": issues,
        "baseline_readout_sha": sha256_file(baseline_readout_path),
        "human_gold_in_model_requests": False})
    prior = baseline.get("narrative_reading") or {}
    observations: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    story = None
    for round_index in (1, 2):
        pending = issues if story is None else [
            issue for issue in issues if issue["issue_id"] in
            story.get("unresolved_issue_ids", []) and
            counts[issue["issue_id"]] < MAX_PROBES_PER_ISSUE]
        if not pending or sum(counts.values()) >= MAX_LOCAL_PROBES:
            break
        plan_payload = {
            "issues": pending, "prior_probe_counts": dict(counts),
            "remaining_total_budget": MAX_LOCAL_PROBES - sum(counts.values()),
            "unverified_preliminary_reading": prior,
            "latest_revision": story,
        }
        selection = _model_call(
            runner, name=f"round_{round_index}_plan", prompt=PLAN_PROMPT,
            payload=plan_payload, output=output)
        probes = validate_probe_selection(selection, pending, static, counts)
        if not probes:
            break
        for probe in probes:
            counts[probe["issue_id"]] += 1
            context = _clip_context(probe, static)
            clip = cut_clip(
                "ffmpeg", masked_video if probe["channel"] == "V" else video,
                output / "clips" / probe["probe_id"],
                start_s=probe["interval"][0], end_s=probe["interval"][1],
                include_audio=probe["channel"] == "AV")
            fps = 12.0 if probe["channel"] == "V" else 8.0
            value = _model_call(
                runner, name=probe["probe_id"],
                prompt=(VISUAL_PROBE_PROMPT if probe["channel"] == "V"
                        else AV_PROBE_PROMPT), payload=context, output=output,
                media=clip, channel=probe["channel"], fps=fps)
            validate_local_observation(value, probe)
            observations.append({**probe, "media_sha": sha256_file(clip),
                                 "response": value})
        _write_json(output / "local_observations.json", {
            "schema_version": "reference_local_observations_v2",
            "observations": observations})
        revision_payload = {
            "section_bundles": static["section_bundles"],
            "accepted_claim_ids": [row["claim_id"] for row in reference["claims"]],
            "accepted_event_ids": [row["event_id"] for row in reference["events"]],
            "unverified_preliminary_reading": prior,
            "previous_revision": story,
            "local_observations": observations,
            "issue_ids": [row["issue_id"] for row in issues],
        }
        story = _model_call(
            runner, name=f"round_{round_index}_story_revision",
            prompt=REVISE_PROMPT, payload=revision_payload,
            output=output, max_new_tokens=4096)
        validate_story_graph(story, reference, issues, observations)
        _write_json(output / "story_graph_v2.json", story)
    if story is None:
        raise ValueError("no_story_revision_generated")
    editing = build_editing_graph(static, observations)
    _write_json(output / "editing_graph_v2.json", editing)
    audit = _model_call(
        runner, name="story_grounding_audit", prompt=AUDIT_PROMPT,
        payload={"accepted_evidence": static["section_bundles"],
                 "local_observations": observations, "story": story},
        output=output, max_new_tokens=3072)
    audit_positive = validate_story_audit(audit, story)
    _write_json(output / "story_grounding_audit_v2.json", audit)
    result = {
        "schema_version": VERSION,
        "status": "human_review_pending",
        "source_sha": reference["source_sha"],
        "agent_reference_sha": reference.get("artifact_sha"),
        "baseline_readout_sha": sha256_file(baseline_readout_path),
        "static_review_sha": static["artifact_sha"],
        "story_graph_sha": json_hash(story),
        "editing_graph_sha": editing["artifact_sha"],
        "local_probe_count": len(observations),
        "probe_counts_by_issue": dict(counts),
        "grounding_audit_positive": audit_positive,
        "story_ready": False, "editing_ready": False,
        "creation_brief_generated": False,
        "production_release_allowed": False,
        "unresolved_issue_ids": story.get("unresolved_issue_ids") or [],
    }
    result["artifact_sha"] = json_hash(result)
    _write_json(output / "reference_understanding_v2.json", result)
    return result
