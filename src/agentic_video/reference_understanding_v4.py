"""Isolated, candidate-only audiovisual reference-understanding comparison.

No artifact from this experiment is published to the creative production DAG.
The two global calls use the same prompt, media and model configuration; the
only intended input difference is independently collected local observations.
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
from src.agentic_video.reference_understanding_v2 import build_review_issues
from src.agentic_video.reference_understanding_v3 import (
    _editing_view, _observation_view, _probe_payload, build_editing_graph,
    preserve_sampled_source_frames, sampling_from_runner,
    validate_local_observation,
)
from src.perception.omni_runner import cut_clip

VERSION = "reference_understanding_v4"
MAX_LOCAL_CALLS = 8
INITIAL_LOCAL_CALLS = 6
MAX_CALLS_PER_ISSUE = 2
MAX_TOTAL_CALLS = 12

# The wording and output contract are intentionally identical in both passes.
GLOBAL_PROMPT = """Watch the whole original audiovisual video and read the
time-aligned on-screen statements. Synthesize, rather than catalogue, the
information contributed by shots and their relationships across time. Keep
four sources distinct: directly visible, stated by the video, audible, and
inferred. An on-screen statement can express the video's position but is not
independently verified visual or real-world fact. A visible later state does
not by itself prove a particular earlier action caused it or establish a
formal outcome. For each narrative unit and relationship cite time/shot IDs
and any relevant attributed statement IDs. Explain how the final item relates
to the preceding information, or say unknown. Allow alternative readings,
any number of units, empty relations and unknown intent. Do not assume fixed
section roles, a reversal, a moral, or a specific subject. Local observations,
when supplied, are provisional evidence, not answers; reject them when the
whole media conflicts. Return JSON only:
{"schema_version":"reference_reading_v4","narrative_units":[
{"unit_id":"...","shot_ids":[],"time_range_s":[0,0],"contribution":"...",
"visible":[],"video_stated":[],"audible":[],"inferred":[],
"statement_ids":[],"local_refs":[],"uncertainty":"..."}],
"relations":[{"relation_id":"...","source":"...","target":"...",
"description":"...","shot_ids":[],"statement_ids":[],
"local_refs":[],"status":"provisional|insufficient"}],
"message_hypotheses":[{"hypothesis_id":"...","position":"... or unknown",
"viewer_information_change":"... or unknown",
"ending_relation":"... or unknown","shot_ids":[],"statement_ids":[],
"local_refs":[],"alternative":"...","uncertainty":"..."}],
"unresolved":[]}
Input: """

LOCAL_PROMPT = """Inspect this original-sound local clip afresh. Shot
intervals and any on-screen statements are supplied with clip-relative time.
For each shot record entry state, directly visible change, exit state and the
first visible time of each change. Compare each adjacent pair: what does the
later shot newly show or state? A cut is not uninterrupted action; do not
assert unseen causation or a formal outcome. A statement is content attributed
to the video, not a directly visible physical fact. Do not quote unverified
speech. For motion in a sparsely sampled short shot, say insufficient. Do not
infer the video's theme or an edit's rhetorical function. Return JSON only:
{"schema_version":"reference_local_observation_v3","shots":[
{"shot_id":"...","entry_state":"... or unknown",
"visible_action":"... or unknown","exit_state":"... or unknown",
"changes":[{"description":"...","kind":"action|posture|contact|identity|causal|other",
"first_visible_s":0.0,"epistemic_status":"supported|contested|insufficient"}],
"attributed_statement_ids":[],"audible_event_type":"music|speech|sound_effect|other|unknown",
"limitations":[]}],"connections":[{"from_shot_id":"...","to_shot_id":"...",
"continuity":"same_action|new_action|possible_ellipsis|unknown",
"information_added":"... or unknown","limitations":[]}],"unresolved":[]}
Input: """

EDIT_PROMPT = """For each supplied adjacent edge, separate a measured edit
from a hypothesis about what information the following shot adds and why that
edit may matter. Use only the supplied local observations and the provisional
whole-video reading. A short shot is not automatically montage; an audio
energy onset is not a verified beat. If either side is unobserved, report
unknown rather than a film-grammar label. Do not invent a missing shot or
formal event outcome. Return JSON only:
{"schema_version":"reference_editing_functions_v4","edges":[
{"from_shot_id":"...","to_shot_id":"...","information_added":"... or unknown",
"function_hypothesis":"... or unknown","alternative":"...",
"support_probe_ids":[],"status":"provisional|insufficient"}],
"limitations":[]}
Input: """

AUDIT_PROMPT = """Independently check the supplied narrative units, relations,
message hypotheses and edit-function hypotheses against their cited evidence.
Check that IDs and time ranges match, statements remain attributed to the
video, and inferred cause/outcome is not promoted to observation. A local
observation marked insufficient cannot independently support a motion claim.
This is a grounding and attribution audit, not a test of whether the most
important theme was found. Text-only evidence cannot prove a media observation
visually true. Do not require any specific story relation. Return JSON only:
{"schema_version":"reference_grounding_audit_v4","checks":[
{"kind":"unit|relation|message|edit","id":"...","grounded":false,
"attribution_preserved":false,"temporal_scope_valid":false,
"inference_within_evidence":false,"reason":"..."}],
"visual_truth_verified":false,"limitations":[]}
Input: """


def _shot_payload(static: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"shot_id": row["shot_id"], "section_id": row["section_id"],
             "start_s": row["start_s"], "end_s": row["end_s"],
             "transition_in": row.get("transition_in")}
            for row in static["shots"]]


def global_payload(static: dict[str, Any],
                   local_observations: list[dict[str, Any]]
                   ) -> dict[str, Any]:
    """No prior reading, gold, or legacy story prose enters either pass."""
    return {"interval": [0.0, static["duration_s"]],
            "observation_dimensions": ["visible", "video_stated", "audible",
                                       "inferred", "cross_shot_relation"],
            "shots": _shot_payload(static),
            "text_statements": [{"statement_id": row["claim_id"],
                                 "source_type": "attributed_on_screen_text",
                                 "observed_text": row["observed_text"],
                                 "interval": row["interval"],
                                 "timing_status": row["timing_status"]}
                                for row in static["text_timeline"]],
            "audio_transcript": static.get("audio_transcript_candidate"),
            "local_observations": local_observations}


def validate_reading(value: dict[str, Any], static: dict[str, Any],
                     local_ids: set[str]) -> list[str]:
    if value.get("schema_version") != "reference_reading_v4":
        return ["schema_invalid"]
    shots = {row["shot_id"] for row in static["shots"]}
    statements = {row["claim_id"] for row in static["text_timeline"]}
    units = value.get("narrative_units") or []
    unit_ids = {row.get("unit_id") for row in units}
    errors = []
    if None in unit_ids or len(unit_ids) != len(units):
        errors.append("unit_ids_invalid")
    for kind, rows in (("unit", units), ("relation", value.get("relations") or []),
                       ("message", value.get("message_hypotheses") or [])):
        for row in rows:
            if set(row.get("shot_ids") or []) - shots:
                errors.append(f"{kind}_shot_unknown")
            if set(row.get("statement_ids") or []) - statements:
                errors.append(f"{kind}_statement_unknown")
            if set(row.get("local_refs") or []) - local_ids:
                errors.append(f"{kind}_local_ref_unknown")
            if kind == "relation" and (row.get("source") not in unit_ids or
                                       row.get("target") not in unit_ids):
                errors.append("relation_endpoint_unknown")
            if kind == "unit":
                span = row.get("time_range_s") or []
                if len(span) != 2 or not 0 <= float(span[0]) <= float(span[1]) <= float(static["duration_s"]):
                    errors.append("unit_time_invalid")
    return sorted(set(errors))


def _make_probe(static: dict[str, Any], ids: list[str], issue_id: str,
                purpose: str, fps: float, ordinal: int) -> dict[str, Any]:
    index = {row["shot_id"]: row for row in static["shots"]}
    ordered = [row["shot_id"] for row in static["shots"]]
    positions = [ordered.index(sid) for sid in ids]
    if len(ids) < 2 or positions != list(range(positions[0], positions[0]+len(ids))):
        raise ValueError("probe_shots_not_consecutive")
    return {"probe_id": f"probe_{ordinal:02d}", "issue_id": issue_id,
            "purpose": purpose, "channel": "AV", "shot_ids": ids,
            "interval": [float(index[ids[0]]["start_s"]),
                         float(index[ids[-1]]["end_s"])], "fps": fps}


def plan_initial_probes(static: dict[str, Any], reference: dict[str, Any]
                        ) -> tuple[list[dict[str, Any]], list[str]]:
    """Generic coverage of long coarse actions, result edges and final text."""
    shots = static["shots"]
    ordered = [row["shot_id"] for row in shots]
    issues = build_review_issues(static, reference)
    selected: list[dict[str, Any]] = []
    omitted: list[str] = []
    def add(ids: list[str], issue: str, purpose: str, fps: float) -> None:
        if len(selected) >= INITIAL_LOCAL_CALLS:
            omitted.append(f"{issue}:initial_budget")
            return
        if any(row["issue_id"] == issue and row["shot_ids"] == ids and row["fps"] == fps
               for row in selected):
            omitted.append(f"{issue}:duplicate")
            return
        selected.append(_make_probe(static, ids, issue, purpose, fps, len(selected)+1))
    for issue in issues:
        if issue["reason_code"] != "one_action_claim_spans_multiple_shots":
            continue
        ids = issue["shot_ids"]
        windows = [ids] if len(ids) <= 4 else [ids[:4], ids[-4:]]
        for window in windows:
            add(window, issue["issue_id"], "coarse_action_coverage", 12.0)
        end = ordered.index(ids[-1])
        for offset in range(2):
            if end + offset + 1 < len(shots) and shots[end+offset+1]["section_id"] == shots[end]["section_id"]:
                add(ordered[end+offset:end+offset+2],
                    f"outcome_link_{issue['issue_id']}", "action_to_later_state", 12.0)
    for issue in issues:
        if issue["reason_code"] == "final_text_relation_unverified":
            add(issue["shot_ids"], issue["issue_id"], "terminal_statement_connection", 12.0)
    # A contiguous fast section is observed as a sequence, not inferred to
    # have a particular rhetorical role. It may be omitted by the budget.
    if len(selected) < INITIAL_LOCAL_CALLS:
        sections = static["section_bundles"]
        covered = {sid for probe in selected for sid in probe["shot_ids"]}
        ranked = sorted(sections, key=lambda section: (-len([
            shot for shot in shots if shot["section_id"] == section["section_id"]
            and shot["shot_id"] not in covered]), -len([
            shot for shot in shots if shot["section_id"] == section["section_id"]])))
        for section in ranked:
            ids = [shot["shot_id"] for shot in shots
                   if shot["section_id"] == section["section_id"]]
            if len(ids) >= 2:
                add(ids, f"dense_sequence_{section['section_id']}",
                    "adjacent_sequence", 12.0)
                break
    validate_probe_plan(selected, static)
    return selected, omitted


def validate_probe_plan(probes: list[dict[str, Any]], static: dict[str, Any]) -> None:
    if len(probes) > MAX_LOCAL_CALLS:
        raise ValueError("media_budget_exceeded")
    ordered = [row["shot_id"] for row in static["shots"]]
    signatures = set()
    per_issue: Counter[str] = Counter()
    for row in probes:
        positions = [ordered.index(sid) for sid in row["shot_ids"]]
        if (row["channel"] != "AV" or len(positions) < 2 or
                positions != list(range(positions[0], positions[0]+len(positions)))):
            raise ValueError("probe_scope_invalid")
        signature = (row["issue_id"], tuple(row["shot_ids"]), row["fps"])
        if signature in signatures:
            raise ValueError("duplicate_probe")
        signatures.add(signature)
        per_issue[row["issue_id"]] += 1
        if per_issue[row["issue_id"]] > MAX_CALLS_PER_ISSUE:
            raise ValueError("issue_budget_exceeded")


def plan_verification(static: dict[str, Any], probes: list[dict[str, Any]],
                      observations: list[dict[str, Any]], first: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    """Only a changed sampling method may revisit a high-impact local gap."""
    if len(probes) >= MAX_LOCAL_CALLS:
        return []
    unresolved_shots = {sid for row in first.get("unresolved") or []
                        if isinstance(row, dict)
                        for sid in (row.get("shot_ids") or [])}
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for observation in observations:
        missing = sum(row["effective_status"] == "insufficient"
                      for row in observation["validation"]["change_findings"])
        if missing or observation["response"].get("unresolved"):
            ranked.append((len(unresolved_shots & set(observation["shot_ids"])),
                           missing, observation))
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    result = []
    for _, _, record in ranked:
        if len(probes) + len(result) >= MAX_LOCAL_CALLS or len(result) >= 2:
            break
        # The first global reading may reveal a gap, but its interpretation
        # is never sent to the observer; only this prior probe's neutral scope.
        source = record
        fps = 16.0 if float(source["fps"]) != 16.0 else 12.0
        issue_id = f"verify_{source['probe_id']}"
        result.append(_make_probe(static, source["shot_ids"], issue_id,
                                  "different_sampling_verification", fps,
                                  len(probes)+len(result)+1))
    validate_probe_plan(probes + result, static)
    return result


def _call_full(runner: Any, name: str, video: Path, static: dict[str, Any],
               local_view: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    payload = global_payload(static, local_view)
    value = _model_call(runner, name=name, prompt=GLOBAL_PROMPT, payload=payload,
                        output=output, media=video, channel="AV", fps=2.0,
                        max_new_tokens=4096)
    probe = {"interval": payload["interval"]}
    sampling = sampling_from_runner(runner, probe)
    _write_json(output / "calls" / name / "sampling_audit.json", sampling)
    preserve_sampled_source_frames(video, sampling,
                                   output / "calls" / name / "sampled_frames")
    return value


def _call_local(runner: Any, probe: dict[str, Any], static: dict[str, Any],
                video: Path, output: Path) -> dict[str, Any]:
    payload = _probe_payload(probe, static)
    text_by_id = {row["claim_id"]: row for row in static["text_timeline"]}
    for row in payload.get("text_statements") or []:
        source = text_by_id[row["claim_id"]]
        row["source_interval_s"] = source["interval"]
        row["relative_interval_s"] = [
            round(float(value) - float(probe["interval"][0]), 6)
            for value in source["interval"]]
        row["timing_status"] = source["timing_status"]
    clip = cut_clip("ffmpeg", video, output / "clips" / probe["probe_id"],
                    start_s=probe["interval"][0], end_s=probe["interval"][1],
                    include_audio=True)
    value = _model_call(runner, name=probe["probe_id"], prompt=LOCAL_PROMPT,
                        payload=payload, output=output, media=clip,
                        channel="AV", fps=probe["fps"], max_new_tokens=3072)
    sampling = sampling_from_runner(runner, probe)
    _write_json(output / "calls" / probe["probe_id"] / "sampling_audit.json",
                sampling)
    manifest = preserve_sampled_source_frames(
        clip, sampling, output / "calls" / probe["probe_id"] / "sampled_frames")
    if manifest["status"] == "frame_count_mismatch":
        sampling["sampling_verified"] = False
        sampling["sampling_error"] = "source_frame_reconstruction_count_mismatch"
        _write_json(output / "calls" / probe["probe_id"] / "sampling_audit.json",
                    sampling)
    validation = validate_local_observation(value, probe, payload, sampling)
    validation["sampled_frames_status"] = manifest["status"]
    _write_json(output / "calls" / probe["probe_id"] / "validation.json",
                {"status": "WARN" if validation["issue_codes"] or any(
                    row["effective_status"] == "insufficient"
                    for row in validation["change_findings"]) else "PASS",
                 **validation})
    return {**probe, "media_sha": sha256_file(clip), "sampling": sampling,
            "validation": validation, "response": value}


def validate_editing(value: dict[str, Any], editing: dict[str, Any]) -> list[str]:
    if value.get("schema_version") != "reference_editing_functions_v4":
        return ["schema_invalid"]
    expected = [(row["from_shot_id"], row["to_shot_id"])
                for row in editing["adjacent_edges"]]
    actual = [(row.get("from_shot_id"), row.get("to_shot_id"))
              for row in value.get("edges") or []]
    errors = [] if actual == expected else ["edge_coverage_invalid"]
    observed = {(row["from_shot_id"], row["to_shot_id"]): {
        obs["probe_id"] for obs in row["observations"]}
        for row in editing["adjacent_edges"]}
    for row in value.get("edges") or []:
        key = row.get("from_shot_id"), row.get("to_shot_id")
        if set(row.get("support_probe_ids") or []) - observed.get(key, set()):
            errors.append("edit_probe_unknown")
        if not observed.get(key) and row.get("status") != "insufficient":
            errors.append("unobserved_edge_not_insufficient")
    return sorted(set(errors))


def validate_audit(value: dict[str, Any], reading: dict[str, Any],
                   editing: dict[str, Any]) -> list[str]:
    if value.get("schema_version") != "reference_grounding_audit_v4":
        return ["schema_invalid"]
    expected = {(kind, row[key]) for kind, rows, key in (
        ("unit", reading.get("narrative_units") or [], "unit_id"),
        ("relation", reading.get("relations") or [], "relation_id"),
        ("message", reading.get("message_hypotheses") or [], "hypothesis_id"))
        for row in rows}
    expected |= {("edit", f"{row['from_shot_id']}->{row['to_shot_id']}")
                 for row in editing.get("edges") or []}
    actual = [(row.get("kind"), row.get("id")) for row in value.get("checks") or []]
    errors = []
    if set(actual) != expected or len(actual) != len(set(actual)):
        errors.append("audit_coverage_invalid")
    if value.get("visual_truth_verified") is not False:
        errors.append("audit_overclaims_visual_truth")
    return errors


def _blind_packet(first: dict[str, Any], second: dict[str, Any],
                  source_sha: str, output: Path) -> None:
    """Create a reviewer-facing packet without giving the model a gold answer."""
    # Stable per-media order, concealed in a separate mapping. Reviewers
    # receive only blind_packet.json, not blind_key.json or raw call traces.
    swapped = int(source_sha[-1], 16) % 2 == 1
    variants = [second, first] if swapped else [first, second]
    _write_json(output / "human_review" / "first_watch_form.json", {
              "schema_version": "reference_human_first_watch_v4",
              "instruction": "Watch the original at normal speed with sound; record this before opening blind_variants.json.",
              "source_sha": source_sha,
              "observations": {
                  "initial_claim_and_speaker": None,
                  "visible_action_and_result_times": [],
                  "later_statements_vs_visible_actions": [],
                  "ending_relation": None,
                  "important_edit_edges": [], "uncertainties": []},
              "review_status": "human_review_pending"})
    packet = {"schema_version": "reference_blind_review_packet_v4",
              "instruction": "After completing first_watch_form.json, compare A and B without assuming either is improved.",
              "source_sha": source_sha,
              "variants": {"A": variants[0], "B": variants[1]},
              "review_status": "human_review_pending"}
    _write_json(output / "human_review" / "blind_variants.json", packet)
    _write_json(output / "human_review" / "blind_key_private.json", {
        "A": "after_local" if swapped else "before_local",
        "B": "before_local" if swapped else "after_local",
        "do_not_share_before_review": True})


def run_reference_understanding_v4(
        reference_path: Path, video: Path, output: Path, runner: Any,
        *, asr_path: Path | None = None) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_directory_not_empty")
    reference = load_reference(reference_path)
    static = prepare_readout(reference_path, video, output, asr_path=asr_path)
    if static["source_sha"] != reference["source_sha"]:
        raise ValueError("source_sha_mismatch")
    first = _call_full(runner, "full_before_local", video, static, [], output)
    _write_json(output / "reading_before_local.json", first)
    initial, omitted = plan_initial_probes(static, reference)
    _write_json(output / "probe_plan.json", {
        "schema_version": "reference_probe_plan_v4", "initial": initial,
        "initial_omitted": omitted, "max_local_calls": MAX_LOCAL_CALLS,
        "first_reading_withheld_from_local_executor": True})
    observations = []
    for probe in initial:
        observations.append(_call_local(runner, probe, static, video, output))
        _write_json(output / "local_observations.json", {
            "schema_version": "reference_local_observations_v4",
            "observations": observations})
    verification = plan_verification(static, initial, observations, first)
    _write_json(output / "verification_plan.json", {
        "schema_version": "reference_verification_plan_v4",
        "probes": verification,
        "first_reading_interpretation_not_in_executor_request": True})
    for probe in verification:
        observations.append(_call_local(runner, probe, static, video, output))
        _write_json(output / "local_observations.json", {
            "schema_version": "reference_local_observations_v4",
            "observations": observations})
    view = _observation_view(observations)
    editing_graph = build_editing_graph(static, observations)
    _write_json(output / "editing_graph_v4.json", editing_graph)
    second = _call_full(runner, "full_after_local", video, static, view, output)
    _write_json(output / "reading_after_local.json", second)
    local_ids = {f"{row['probe_id']}:{sid}" for row in observations
                 for sid in row["shot_ids"]}
    checks = {"before": validate_reading(first, static, set()),
              "after": validate_reading(second, static, local_ids)}
    _write_json(output / "reading_validation.json", checks)
    edit = _model_call(runner, name="editing_function", prompt=EDIT_PROMPT,
                       payload={"measured_editing": static["measured_editing"],
                                "audio_status": editing_graph["audio_measurement_status"],
                                "adjacent_edges": _editing_view(editing_graph, view),
                                "reading": second}, output=output,
                       max_new_tokens=4096)
    _write_json(output / "editing_functions_v4.json", edit)
    edit_errors = validate_editing(edit, editing_graph)
    _write_json(output / "editing_validation.json", {"issues": edit_errors})
    audit = _model_call(runner, name="grounding_audit", prompt=AUDIT_PROMPT,
                        payload={"shots": _shot_payload(static),
                                 "text_statements": static["text_timeline"],
                                 "accepted_claims": reference["claims"],
                                 "accepted_events": reference["events"],
                                 "local_observations": view,
                                 "reading": second, "editing_functions": edit},
                        output=output, max_new_tokens=4096)
    _write_json(output / "grounding_audit_v4.json", audit)
    audit_errors = validate_audit(audit, second, edit)
    _write_json(output / "audit_validation.json", {"issues": audit_errors})
    _blind_packet(first, second, static["source_sha"], output)
    call_meta = [json.loads(path.read_text(encoding="utf-8"))
                 for path in (output / "calls").glob("*/model_call.json")]
    if len(call_meta) > MAX_TOTAL_CALLS:
        raise ValueError("total_call_budget_exceeded")
    full_meta = [json.loads((output / "calls" / name / "model_call.json")
                            .read_text(encoding="utf-8"))
                 for name in ("full_before_local", "full_after_local")]
    if (full_meta[0]["prompt_sha"] != full_meta[1]["prompt_sha"] or
            full_meta[0]["model_config_sha"] != full_meta[1]["model_config_sha"] or
            full_meta[0]["media_sha"] != full_meta[1]["media_sha"]):
        raise ValueError("full_call_control_mismatch")
    result = {"schema_version": VERSION, "status": "candidate_human_review_pending",
              "source_sha": static["source_sha"],
              "static_review_sha": static["artifact_sha"],
              "local_call_count": len(observations),
              "model_call_count": len(call_meta), "call_budget": MAX_TOTAL_CALLS,
              "same_full_prompt_model_media": True,
              "reading_validation_issues": checks,
              "editing_validation_issues": edit_errors,
              "audit_validation_issues": audit_errors,
              "blind_review_pending": True,
              "story_ready": False, "editing_ready": False,
              "production_release_allowed": False,
              "human_truth_confirmed": False,
              "brief_candidate_generated": False}
    result["artifact_sha"] = json_hash(result)
    _write_json(output / "reference_reading_v4.json", result)
    return result
