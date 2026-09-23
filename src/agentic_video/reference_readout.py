"""Evidence-linked reference readout pilot; never publishes to the creative DAG."""
from __future__ import annotations

import json
import math
import subprocess
from array import array
from pathlib import Path
from typing import Any

from src.agentic_video.editing_grammar import build_measured_editing_facts
from src.agentic_video.manifest import json_hash
from src.agentic_video.modality_isolation import audit_request
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.reference_interpretation import build_interpretation_payload
from src.agentic_video.reference_storyboard import build_agent_reference
from src.perception.omni_runner import cut_clip

READOUT_VERSION = "reference_readout_v1"
SAMPLE_RATE = 16000
WINDOW_S = 0.02


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_reference(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") == "agent_reference_v2":
        return value
    if value.get("accepted_claims") is not None:
        return build_agent_reference(value)
    raise ValueError("expected agent_reference_v2 or validated_reference")


def _pcm_audio(video: Path, *, ffmpeg_bin: str = "ffmpeg") -> array:
    completed = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"],
        capture_output=True, check=True)
    samples = array("h")
    samples.frombytes(completed.stdout)
    if not samples:
        raise ValueError("reference audio stream has no decoded samples")
    return samples


def measure_audio(video: Path, cuts_s: list[float], duration_s: float,
                  *, ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """Report sound energy and candidate onsets, never assert musical beats."""
    samples = _pcm_audio(video, ffmpeg_bin=ffmpeg_bin)
    width = round(SAMPLE_RATE * WINDOW_S)
    envelope = []
    for start in range(0, len(samples), width):
        chunk = samples[start:start + width]
        rms = math.sqrt(sum(value * value for value in chunk) / len(chunk)) / 32768
        envelope.append(round(rms, 6))
    positive_changes = [max(0.0, envelope[index] - envelope[index - 1])
                        for index in range(1, len(envelope))]
    ordered = sorted(positive_changes)
    threshold = ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0.0
    candidates = []
    for index in range(1, len(envelope) - 1):
        change = positive_changes[index - 1]
        if change <= 0 or change < threshold:
            continue
        if index >= 2 and change < positive_changes[index - 2]:
            continue
        if index < len(positive_changes) and change < positive_changes[index]:
            continue
        at_s = round(index * WINDOW_S, 3)
        if candidates and at_s - candidates[-1] < 0.1:
            continue
        candidates.append(at_s)

    def alignment(onsets: list[float]) -> tuple[int, list[dict[str, float | None]]]:
        rows = []
        for cut in cuts_s:
            nearest = min(onsets, key=lambda t: abs(t - cut)) if onsets else None
            delta = round(cut - nearest, 3) if nearest is not None else None
            rows.append({"cut_s": cut, "nearest_onset_s": nearest,
                         "signed_offset_s": delta})
        return sum(row["signed_offset_s"] is not None and
                   abs(row["signed_offset_s"]) <= 0.08 for row in rows), rows

    observed, offsets = alignment(candidates)
    shifted_counts = []
    for step in range(1, 21):
        shift = duration_s * step / 21
        shifted = sorted((onset + shift) % duration_s for onset in candidates)
        shifted_counts.append(alignment(shifted)[0])
    return {
        "schema_version": "reference_audio_measurements_v1",
        "source_sha": sha256_file(video), "sample_rate_hz": SAMPLE_RATE,
        "window_s": WINDOW_S, "rms_envelope": envelope,
        "candidate_onsets_s": candidates,
        "onset_detection": "95th percentile of positive 20 ms RMS changes; 100 ms spacing",
        "cut_onset_tolerance_s": 0.08,
        "cut_onset_offsets": offsets,
        "observed_near_onset_count": observed,
        "shifted_control_counts": shifted_counts,
        "music_beat_status": "unverified",
        "beat_synced_status": "unverified",
    }


def build_static_readout(agent_reference: dict[str, Any], video: Path,
                         *, ffmpeg_bin: str = "ffmpeg",
                         asr_path: Path | None = None) -> dict[str, Any]:
    source_sha = sha256_file(video)
    if source_sha != agent_reference.get("source_sha"):
        raise ValueError("reference source SHA does not match video")
    cards = (agent_reference.get("shot_storyboard") or {}).get("shot_cards") or []
    if not cards:
        raise ValueError("reference has no content shots")
    duration = max(float(card["end_s"]) for card in cards)
    cuts = sorted({round(float(card["start_s"]), 6) for card in cards
                   if float(card["start_s"]) > 0})
    measured = build_measured_editing_facts(agent_reference)
    timeline = agent_reference.get("deterministic_timeline") or {}
    transition_rows = [row for row in timeline.get("segments") or []
                       if row.get("segment_kind") == "transition"]
    claims = agent_reference.get("claims") or []
    text_rows = [{"claim_id": row["claim_id"],
                  "observed_text": row.get("object"),
                  "interval": row.get("interval"),
                  "source_type": "on_screen_text",
                  "timing_status": "accepted_claim_interval_not_frame_verified"}
                 for row in claims if row.get("modality") == "T"]
    asr = None
    if asr_path is not None:
        raw_asr = json.loads(asr_path.read_text(encoding="utf-8"))
        asr = {"status": "candidate_not_human_verified",
               "channel": "A", "producer": "whisper_base_audio_only",
               "source_file_sha": sha256_file(asr_path),
               "segments": [{"start_s": row["start"], "end_s": row["end"],
                             "transcript": row["text"].strip()}
                            for row in raw_asr.get("segments") or []]}
    result = {
        "schema_version": "reference_static_review_v1",
        "source_sha": source_sha,
        "agent_reference_sha": agent_reference.get("artifact_sha"),
        "duration_s": duration,
        "shot_count": len(cards), "content_boundary_count": len(cuts),
        "transition_count": len(transition_rows),
        "shots": cards, "transitions": transition_rows,
        "text_timeline": text_rows,
        "audio": measure_audio(video, cuts, duration, ffmpeg_bin=ffmpeg_bin),
        "audio_transcript_candidate": asr,
        "measured_editing": measured,
        "section_bundles": build_interpretation_payload(agent_reference)[
            "section_bundles"],
        "unresolved": [
            "screen_text_entrance_exit_and_animation_unverified",
            "music_beat_grid_unverified",
            "audio_events_and_speech_content_unverified",
        ],
    }
    result["artifact_sha"] = json_hash(result)
    return result


def prepare_readout(reference_path: Path, video: Path, output: Path,
                    *, asr_path: Path | None = None) -> dict[str, Any]:
    reference = load_reference(reference_path)
    result = build_static_readout(reference, video, asr_path=asr_path)
    _write_json(output / "static_review.json", result)
    extract_review_frames(video, result, output / "review_frames")
    return result


def extract_review_frames(video: Path, static: dict[str, Any],
                          directory: Path) -> dict[str, Any]:
    """Save native-time frame evidence around every accepted edit boundary."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = []
    for index, card in enumerate(static["shots"][1:], start=1):
        cut = float(card["start_s"])
        for side, at in (("before", cut - 1 / 30), ("after", cut + 1 / 30)):
            timestamps.append((f"cut_{index:02d}_{side}", max(0.0, at)))
    for index, segment in enumerate(static["transitions"], start=1):
        start, end = map(float, segment["interval"])
        for position, at in (("start", start), ("mid", (start + end) / 2),
                             ("end", end)):
            timestamps.append((f"transition_{index:02d}_{position}", at))
    records = []
    for label, at in timestamps:
        path = directory / f"{label}.jpg"
        if not path.exists():
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video),
                 "-ss", f"{at:.6f}", "-frames:v", "1", "-vf",
                 "scale=360:-2", "-q:v", "2", "-y", str(path)],
                check=True, capture_output=True)
        records.append({"label": label, "time_s": round(at, 6),
                        "file": str(path), "sha256": sha256_file(path)})
    manifest = {"schema_version": "reference_review_frames_v1",
                "source_sha": static["source_sha"], "frames": records}
    _write_json(directory / "manifest.json", manifest)
    return manifest


VISUAL_DYNAMICS_PROMPT = """Inspect only the silent, text-masked video clip.
For each supplied shot ID, report visible framing and motion. Keep camera
motion distinct from subject motion; use null when undecidable. Do not infer
speech, written content, anatomy, identity, story purpose, or unseen actions.
The supplied shot times are relative to this clip. Return JSON only:
{"schema_version":"shot_dynamics_observation_v1","shots":[{"shot_id":"...",
"shot_size":null,"camera_angle":null,"camera_motion":null,
"visible_motion_speed":null,"support_interval":[0.0,0.0],
"limitations":[]}],"limitations":[]}
Input: """

GLOBAL_READING_PROMPT = """Watch the entire original video with sound and
read the supplied evidence. The evidence channels have already been separated.
Propose a concise, open-ended interpretation of what the video communicates.
Do not restate every claim. Do not assume a reversal, fixed section roles, or a
particular moral. On-screen text is a statement by the video, not independent
proof of its content. Keep what is directly shown, stated, heard, and inferred
separate. Cite supplied claim/event IDs for each unit, relation, and message
hypothesis. If the video leaves intent uncertain, give an alternative reading.
Do not infer motive, real-world biography, anatomy, or ability beyond evidence.
Return JSON only:
{"schema_version":"reference_message_reading_v1",
"narrative_units":[{"unit_id":"N1","section_ids":[],"summary":"...",
"support_ids":[],"attributed_statement_ids":[]}],
"relations":[{"relation_id":"R1","source":"N1","target":"N2",
"description":"...","support_ids":[]}],
"message_hypotheses":[{"hypothesis_id":"H1",
"communicative_goal":"...","viewer_information_path":"...",
"tone_and_ending":"...","support_ids":[],"alternative":"...",
"uncertainty":"..."}],"limitations":[]}
The arrays may be empty. Input: """

LOCAL_EDITING_PROMPT = """Watch this original-sound clip and inspect how the
listed adjacent shots connect. Measured durations, cuts, transitions, and
audio onsets are supplied as facts. Report only content-dependent editing
patterns supported by the visible and audible clip; use an empty list when
none is clear. A fast sequence alone does not prove montage. A cut near an
estimated audio onset alone does not prove music synchronization. Do not infer
the story's overall purpose in this recognition step. Pick exactly one type
per pattern from: rapid_montage, reaction_cut, contrast_cut, delayed_reveal,
cut_on_action, match_cut, shot_reverse_shot, parallel_editing, insert_shot,
audio_bridge, other. Use at least two adjacent shot IDs. The input interval
is the absolute source span; output media_interval uses seconds relative to
the start of this clip, from zero to clip_duration_s. Return JSON only:
{"schema_version":"reference_local_editing_v1","patterns":[{
"pattern_id":"EP1","type":"one allowed type",
"shot_ids":[],"observed_connection":"...","support_ids":[],
"media_interval":[0.0,0.0],"uncertainty":"..."}],"limitations":[]}
Use only supplied shot/evidence IDs.
Input: """

ENDING_PROMPT = """Watch the supplied final video span with sound. Describe
what is visibly shown, what the video states in text or speech, and how the
last item relates to the immediately preceding items. Keep observed content
separate from a possible tone or meaning; allow more than one reading when
ambiguous. Do not infer an unseen life history or a predetermined moral.
Return JSON only:
{"schema_version":"reference_ending_observation_v1","ending_id":"END1",
"observations":"...","relation_to_preceding":"...",
"tone_hypothesis":"...","alternative":"...","support_ids":[],
"limitations":[]}
Use supplied evidence IDs. Input: """

EDITING_FUNCTION_PROMPT = """Infer what each observed local editing pattern
may accomplish in the video, using the independently proposed narrative units.
Keep an observation separate from its possible function. Do not invent a
function for a pattern with insufficient evidence. Do not add new patterns.
Return JSON only:
{"schema_version":"reference_editing_function_v1","functions":[{
"pattern_id":"EP1","function_hypothesis":"...",
"narrative_unit_ids":[],"support_ids":[],"alternative":"...",
"uncertainty":"..."}],"limitations":[]}
Input: """

GROUNDING_AUDIT_PROMPT = """Independently check whether every narrative
unit, cross-unit relation, message hypothesis, editing pattern, and function
or ending observation is supported by its cited evidence. Check temporal scope and attribution of
on-screen statements. Do not judge whether it is the most salient reading and
do not demand a particular theme or narrative relation. Media-dependent
claims not verifiable from the supplied evidence must be marked uncertain.
Return JSON only:
{"schema_version":"reference_grounding_audit_v1","checks":[{
"kind":"unit|relation|message|pattern|function|ending","id":"...",
"grounded":false,"attribution_preserved":false,
"temporal_scope_valid":false,"reason":"..."}],"limitations":[]}
Input: """

TRANSFER_DRAFT_PROMPT = """Abstract the reviewed reference reading into a
draft control interface for creating a different story. Preserve only the
communicative objective, the relation between presented information and
changed interpretation, tone/ending relation, and editing organization.
Change all character, domain, activity, setting, and wording bindings.
No names, quoted text, video paths, claim/event/shot IDs, or examples from the
reference may appear in the result. Do not assert a universal story template.
This is a review draft, not a production release. Return JSON only:
{"schema_version":"reference_transfer_draft_v1","communicative_goal":"...",
"information_relation":"...","tone_and_ending_relation":"...",
"editing_relations":[],"free_slots":[],"limitations":[]}
Input: """


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def _model_call(runner: Any, *, name: str, prompt: str,
                payload: dict[str, Any], output: Path,
                media: Path | None = None, channel: str = "FUSION",
                fps: float | None = None,
                max_new_tokens: int = 3072) -> dict[str, Any]:
    stage = output / "calls" / name
    stage.mkdir(parents=True, exist_ok=True)
    request = prompt + json.dumps(payload, ensure_ascii=False,
                                  separators=(",", ":"))
    audit_payload: dict[str, Any] = {"request_id": name, "channel": channel}
    if channel == "V":
        audit_payload.update({"interval": payload["interval"],
                              "anonymous_subject_ids": [],
                              "observation_dimensions": ["framing", "motion"],
                              "sampling": {"fps": fps},
                              "mask_artifact_sha": sha256_file(media)})
    elif channel == "AV":
        audit_payload.update({"interval": payload["interval"],
                              "claim_ids": payload.get("claim_ids") or [],
                              "event_ids": payload.get("event_ids") or [],
                              "observation_dimensions": payload.get(
                                  "observation_dimensions") or [],
                              "sampling": {"fps": fps}})
    else:
        audit_payload.update({"claim_ids": payload.get("claim_ids") or [],
                              "event_ids": payload.get("event_ids") or []})
    audit = audit_request(channel=channel, prompt=request,
                          payload=audit_payload,
                          media_paths=[media] if media is not None else [])
    _write_json(stage / "request_audit.json", audit)
    (stage / "request.txt").write_text(request, encoding="utf-8")
    if media is not None:
        answer = runner.watch(media, request, fps=fps,
                              duration_s=payload["interval"][1] -
                              payload["interval"][0],
                              use_audio_in_video=channel == "AV",
                              max_new_tokens=max_new_tokens,
                              stop_after_json_object=True)
    else:
        answer = runner.ask(request, max_new_tokens=max_new_tokens,
                            stop_after_json_object=True)
    raw = str(getattr(answer, "text", answer))
    (stage / "raw_response.txt").write_text(raw, encoding="utf-8")
    meta = {"request_sha": sha256_file(stage / "request.txt"),
            "prompt_sha": json_hash(prompt),
            "response_sha": sha256_file(stage / "raw_response.txt"),
            "media_sha": sha256_file(media) if media else None,
            "model_config_sha": json_hash(getattr(runner, "cfg", {}) or {}),
            "model_path": (getattr(runner, "cfg", {}) or {}).get("model_path"),
            "input_tokens": getattr(answer, "input_tokens", None),
            "output_tokens": getattr(answer, "output_tokens", None),
            "elapsed_s": getattr(answer, "elapsed_s", None)}
    _write_json(stage / "model_call.json", meta)
    try:
        parsed = _json_object(raw)
    except (ValueError, IndexError) as exc:
        _write_json(stage / "validation.json",
                    {"status": "FAIL", "error": str(exc)})
        raise
    _write_json(stage / "parsed.json", parsed)
    return parsed


def select_local_probes(static: dict[str, Any]) -> list[dict[str, Any]]:
    """Spend the small probe budget on unobserved motion and busy edits."""
    sections = [(row["section_id"], row["interval"])
                for row in static["section_bundles"]]
    probes: list[dict[str, Any]] = []
    for section_id, interval in sections:
        cards = [row for row in static["shots"]
                 if row["section_id"] == section_id]
        if any(row.get("unresolved") for row in cards):
            probes.append({"probe_id": f"visual_{section_id}",
                           "channel": "V", "interval": interval,
                           "fps": 4.0 if interval[1] - interval[0] > 7 else 8.0,
                           "question": "framing_and_visible_motion"})
    for index in range(1, len(sections)):
        boundary = float(sections[index][1][0])
        probes.append({"probe_id": f"edit_boundary_{index}",
                       "channel": "AV",
                       "interval": [max(0.0, boundary - 0.85),
                                    min(static["duration_s"], boundary + 0.85)],
                       "fps": 12.0, "question": "adjacent_shot_connection"})
    structural = static["measured_editing"].get("structural_grammar") or {}
    profile = structural.get("transition_profile") or []
    transition_section = max(profile, key=lambda row: row.get(
        "transition_count", 0), default=None)
    if transition_section and transition_section.get("transition_count", 0) >= 3:
        section_id = transition_section["section_id"]
        interval = next(interval for sid, interval in sections if sid == section_id)
        probes.append({"probe_id": f"edit_sequence_{section_id}",
                       "channel": "AV", "interval": interval,
                       "fps": 8.0, "question": "transition_heavy_sequence"})
    hard_cut_section = max(
        structural.get("transition_profile") or [],
        key=lambda row: row.get("hard_cut_count", 0), default=None)
    if hard_cut_section and hard_cut_section.get("hard_cut_count", 0) >= 3:
        section_id = hard_cut_section["section_id"]
        if not any(row["probe_id"] == f"edit_sequence_{section_id}"
                   for row in probes):
            interval = next(interval for sid, interval in sections
                            if sid == section_id)
            probes.append({"probe_id": f"edit_sequence_{section_id}",
                           "channel": "AV", "interval": interval,
                           "fps": 4.0, "question": "cut_sequence"})
    final_section_id, final_interval = sections[-1]
    if any(row["interval"][0] >= final_interval[0] for row in
           static["text_timeline"]):
        probes.append({"probe_id": f"ending_{final_section_id}",
                       "channel": "AV", "interval": final_interval,
                       "fps": 6.0, "question": "ending_relationship"})
    if len(probes) > 8:
        raise ValueError("local probe budget exceeded")
    if len({row["question"] + str(row["interval"]) for row in probes}) != len(probes):
        raise ValueError("duplicate local probe question")
    return probes


def _overlap(left: list[float], right: list[float]) -> bool:
    return max(float(left[0]), float(right[0])) < min(
        float(left[1]), float(right[1]))


def _probe_context(probe: dict[str, Any], static: dict[str, Any],
                   reference: dict[str, Any]) -> dict[str, Any]:
    interval = probe["interval"]
    cards = [row for row in static["shots"] if _overlap(
        [row["start_s"], row["end_s"]], interval)]
    claim_ids = sorted({str(value) for row in cards for field in (
        "visual_claim_ids", "text_claim_ids", "audio_claim_ids")
                        for value in row.get(field) or []})
    event_ids = sorted({str(value) for row in cards
                        for value in row.get("event_ids") or []})
    claim_index = {str(row["claim_id"]): row for row in reference["claims"]}
    event_index = {str(row["event_id"]): row for row in reference["events"]}
    shot_rows = [{"shot_id": row["shot_id"],
                  "relative_interval": [
                      round(max(float(row["start_s"]), interval[0]) - interval[0], 6),
                      round(min(float(row["end_s"]), interval[1]) - interval[0], 6)],
                  "transition_in": row.get("transition_in"),
                  "transition_out": row.get("transition_out")}
                 for row in cards]
    return {
        "interval": interval, "clip_duration_s": round(interval[1] - interval[0], 6),
        "shots": shot_rows, "claim_ids": claim_ids, "event_ids": event_ids,
        "claims": [claim_index[value] for value in claim_ids],
        "events": [event_index[value] for value in event_ids],
        "measured_editing": {
            "structural_grammar": static["measured_editing"].get(
                "structural_grammar"),
            "audio_onsets_s": [round(value - interval[0], 3)
                               for value in static["audio"]["candidate_onsets_s"]
                               if interval[0] <= value <= interval[1]],
            "audio_onset_status": "candidate_energy_onsets_not_verified_beats",
        },
    }


def _known_ids(reference: dict[str, Any]) -> set[str]:
    return {str(row["claim_id"]) for row in reference["claims"]} | {
        str(row["event_id"]) for row in reference["events"]}


EDITING_TYPES = {"rapid_montage", "reaction_cut", "contrast_cut",
                 "delayed_reveal", "cut_on_action", "match_cut",
                 "shot_reverse_shot", "parallel_editing", "insert_shot",
                 "audio_bridge", "other"}


def validate_local_patterns(response: dict[str, Any], context: dict[str, Any]
                            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check one media-relative contract without deciding narrative function."""
    valid = []
    rejected = []
    shots = {row["shot_id"]: row["relative_interval"]
             for row in context["shots"]}
    known_support = set(context["claim_ids"] + context["event_ids"])
    duration = float(context["clip_duration_s"])
    for index, row in enumerate(response.get("patterns") or [], start=1):
        reasons = []
        ids = row.get("shot_ids") or []
        interval = row.get("media_interval")
        if response.get("schema_version") != "reference_local_editing_v1":
            reasons.append("schema_version_invalid")
        if row.get("type") not in EDITING_TYPES:
            reasons.append("pattern_type_invalid")
        if (len(ids) < 2 or any(not isinstance(value, str) for value in ids)
                or len(ids) != len(set(map(str, ids))) or any(
                    value not in shots for value in ids)):
            reasons.append("shot_scope_invalid")
        try:
            start, end = map(float, interval)
            if not 0 <= start < end <= duration + 0.001:
                reasons.append("clip_local_interval_invalid")
            elif sum(_overlap([start, end], shots[value])
                     for value in ids if value in shots) < 2:
                reasons.append("interval_does_not_cover_shot_relation")
        except (TypeError, ValueError):
            reasons.append("clip_local_interval_invalid")
        if any(str(value) not in known_support
               for value in row.get("support_ids") or []):
            reasons.append("support_id_outside_clip")
        if reasons:
            rejected.append({"response_index": index, "reason_codes": reasons,
                             "raw_pattern_id": row.get("pattern_id")})
        else:
            valid.append(row)
    return valid, rejected


def _citation_issues(reading: dict[str, Any], patterns: list[dict[str, Any]],
                     functions: dict[str, Any], reference: dict[str, Any],
                     ending: dict[str, Any] | None = None) -> list[str]:
    known = _known_ids(reference)
    units = {str(row.get("unit_id")) for row in reading.get("narrative_units") or []}
    pattern_ids = {str(row.get("pattern_id")) for row in patterns}
    issues = []
    rows = [("unit", row) for row in reading.get("narrative_units") or []]
    rows += [("relation", row) for row in reading.get("relations") or []]
    rows += [("message", row) for row in reading.get("message_hypotheses") or []]
    rows += [("pattern", row) for row in patterns]
    rows += [("function", row) for row in functions.get("functions") or []]
    if ending is not None:
        rows.append(("ending", ending))
    for kind, row in rows:
        for value in row.get("support_ids") or []:
            if str(value) not in known:
                issues.append(f"{kind}:unknown_support:{value}")
        if kind == "relation" and (row.get("source") not in units or
                                   row.get("target") not in units):
            issues.append(f"relation:unknown_unit:{row.get('relation_id')}")
        if kind == "function" and row.get("pattern_id") not in pattern_ids:
            issues.append(f"function:unknown_pattern:{row.get('pattern_id')}")
    return issues


def _transfer_leaks(draft: dict[str, Any], reference: dict[str, Any],
                    video: Path) -> list[str]:
    serialized = json.dumps(draft, ensure_ascii=False).casefold()
    forbidden = [str(video), str(video.resolve()), reference["source_sha"]]
    forbidden += [str(row["claim_id"]) for row in reference["claims"]]
    forbidden += [str(row["event_id"]) for row in reference["events"]]
    forbidden += [str(row["object"]) for row in reference["claims"]
                  if row.get("modality") == "T" and
                  len(str(row.get("object") or "")) >= 4]
    forbidden += [str(row["shot_id"]) for row in
                  reference["shot_storyboard"]["shot_cards"]]
    return [str(index) for index, marker in enumerate(forbidden)
            if marker and marker.casefold() in serialized]


def run_model_readout(reference_path: Path, video: Path, masked_video: Path,
                      output: Path, runner: Any,
                      *, asr_path: Path | None = None) -> dict[str, Any]:
    """Run a single evidence-grounded pilot with a bounded local probe plan."""
    output = Path(output)
    if (output / "calls").exists():
        raise ValueError("model run directory already contains calls")
    reference = load_reference(reference_path)
    mask_audit_path = masked_video.parent / "visual_mask_audit.json"
    mask_audit = json.loads(mask_audit_path.read_text(encoding="utf-8"))
    if (mask_audit.get("source_sha") != reference.get("source_sha") or
            mask_audit.get("artifact_sha") != sha256_file(masked_video) or
            mask_audit.get("streams") != ["video"]):
        raise ValueError("masked visual media does not match accepted source")
    static = prepare_readout(reference_path, video, output,
                             asr_path=asr_path)
    probes = select_local_probes(static)
    _write_json(output / "probe_plan.json", {
        "schema_version": "reference_probe_plan_v1", "max_local_calls": 8,
        "max_calls_per_question": 2, "selected": probes,
        "gold_or_expected_answers_in_requests": False})
    visual_dynamics = []
    for probe in probes:
        if probe["channel"] != "V":
            continue
        context = _probe_context(probe, static, reference)
        clip = cut_clip("ffmpeg", masked_video,
                        output / "clips" / probe["probe_id"],
                        start_s=float(probe["interval"][0]),
                        end_s=float(probe["interval"][1]),
                        include_audio=False)
        payload = {"interval": probe["interval"],
                   "shots": context["shots"]}
        value = _model_call(
            runner, name=probe["probe_id"],
            prompt=VISUAL_DYNAMICS_PROMPT, payload=payload,
            output=output, media=clip, channel="V", fps=probe["fps"])
        expected = {row["shot_id"] for row in context["shots"]}
        for row in value.get("shots") or []:
            if row.get("shot_id") in expected:
                visual_dynamics.append({**row, "probe_id": probe["probe_id"],
                                        "media_sha": sha256_file(clip),
                                        "status": "provisional"})
    _write_json(output / "shot_dynamics_provisional.json", {
        "schema_version": "shot_dynamics_observation_v1",
        "source_sha": reference["source_sha"],
        "observations": visual_dynamics,
        "unresolved_shot_ids": sorted({row["shot_id"] for row in static["shots"]} -
                                      {row["shot_id"] for row in visual_dynamics}),
    })

    global_payload = {
        "interval": [0.0, static["duration_s"]],
        "claim_ids": sorted(str(row["claim_id"]) for row in reference["claims"]),
        "event_ids": sorted(str(row["event_id"]) for row in reference["events"]),
        "observation_dimensions": ["cross_shot_narrative", "message_hypotheses"],
        "section_bundles": static["section_bundles"],
        "provisional_visual_dynamics": visual_dynamics,
        "audio_measurement": {
            "candidate_onsets_s": static["audio"]["candidate_onsets_s"],
            "music_beat_status": "unverified"},
        "audio_transcript_candidate": static["audio_transcript_candidate"],
    }
    reading = _model_call(
        runner, name="global_reading", prompt=GLOBAL_READING_PROMPT,
        payload=global_payload, output=output, media=video,
        channel="AV", fps=2.0, max_new_tokens=4096)
    _write_json(output / "narrative_reading.json", reading)

    patterns = []
    rejected_patterns = []
    ending = None
    for probe in probes:
        if probe["channel"] != "AV":
            continue
        context = _probe_context(probe, static, reference)
        clip = cut_clip("ffmpeg", video,
                        output / "clips" / probe["probe_id"],
                        start_s=float(probe["interval"][0]),
                        end_s=float(probe["interval"][1]),
                        include_audio=True)
        if probe["question"] == "ending_relationship":
            ending = _model_call(
                runner, name=probe["probe_id"], prompt=ENDING_PROMPT,
                payload=context, output=output, media=clip,
                channel="AV", fps=probe["fps"])
            _write_json(output / "ending_observation.json", ending)
            continue
        context["observation_dimensions"] = ["adjacent_shot_semantics",
                                             "editing_pattern"]
        response = _model_call(
            runner, name=probe["probe_id"], prompt=LOCAL_EDITING_PROMPT,
            payload=context, output=output, media=clip,
            channel="AV", fps=probe["fps"])
        accepted, rejected = validate_local_patterns(response, context)
        rejected_patterns.extend({**row, "probe_id": probe["probe_id"]}
                                 for row in rejected)
        _write_json(output / "calls" / probe["probe_id"] /
                    "pattern_validation.json", {
                        "accepted_count": len(accepted), "rejected": rejected})
        for index, row in enumerate(accepted, start=1):
            patterns.append({**row,
                             "pattern_id": f"{probe['probe_id']}:"
                                           f"{row.get('pattern_id') or index}",
                             "original_pattern_id": row.get("pattern_id"),
                             "probe_id": probe["probe_id"],
                             "media_sha": sha256_file(clip),
                             "media_origin_s": probe["interval"][0],
                             "evidence_status": (
                                 "provisional" if row.get("support_ids") else
                                 "media_only_pending_review")})
    _write_json(output / "semantic_editing_patterns.json", {
        "schema_version": "reference_local_editing_v1",
        "patterns": patterns, "rejected_patterns": rejected_patterns,
        "local_probe_count": sum(
            row["channel"] == "AV" for row in probes)})

    function_payload = {
        "claim_ids": global_payload["claim_ids"],
        "event_ids": global_payload["event_ids"],
        "narrative_units": reading.get("narrative_units") or [],
        "relations": reading.get("relations") or [],
        "ending_observation": ending,
        "patterns": patterns,
        "measured_structural_grammar": static["measured_editing"].get(
            "structural_grammar"),
    }
    functions = _model_call(
        runner, name="editing_functions", prompt=EDITING_FUNCTION_PROMPT,
        payload=function_payload, output=output, max_new_tokens=2048)
    _write_json(output / "editing_functions.json", functions)
    citation_issues = _citation_issues(
        reading, patterns, functions, reference, ending)

    audit_payload = {
        "claim_ids": global_payload["claim_ids"],
        "event_ids": global_payload["event_ids"],
        "evidence_bundles": static["section_bundles"],
        "narrative_reading": reading,
        "ending_observation": ending,
        "editing_patterns": patterns,
        "editing_functions": functions,
        "citation_issues": citation_issues,
    }
    audit = _model_call(
        runner, name="grounding_audit", prompt=GROUNDING_AUDIT_PROMPT,
        payload=audit_payload, output=output, max_new_tokens=3072)
    expected_keys = (
        [("unit", str(row.get("unit_id"))) for row in
         reading.get("narrative_units") or []] +
        [("relation", str(row.get("relation_id"))) for row in
         reading.get("relations") or []] +
        [("message", str(row.get("hypothesis_id"))) for row in
         reading.get("message_hypotheses") or []] +
        [("pattern", str(row.get("pattern_id"))) for row in patterns] +
        [("function", str(row.get("pattern_id"))) for row in
         functions.get("functions") or []] +
        ([("ending", str(ending.get("ending_id")))] if ending else []))
    checks = audit.get("checks") or []
    actual_keys = [(str(row.get("kind")), str(row.get("id"))) for row in checks]
    audit_complete = sorted(actual_keys) == sorted(expected_keys) and all(
        all(row.get(field) is True for field in (
            "grounded", "attribution_preserved", "temporal_scope_valid"))
        for row in checks)
    _write_json(output / "grounding_audit.json", {
        "model_response": audit, "expected_check_count": len(expected_keys),
        "actual_check_count": len(checks), "citation_issues": citation_issues,
        "audit_complete_and_positive": audit_complete and not citation_issues})

    transfer_payload = {
        "claim_ids": [], "event_ids": [],
        "reviewed_message_hypotheses": reading.get("message_hypotheses") or [],
        "reviewed_narrative_relations": reading.get("relations") or [],
        "reviewed_ending_observation": ending,
        "editing_functions": functions.get("functions") or [],
        "measured_structural_grammar": static["measured_editing"].get(
            "structural_grammar"),
        "audit_status": "positive" if audit_complete and not citation_issues
                        else "unresolved",
    }
    draft = _model_call(
        runner, name="transfer_draft", prompt=TRANSFER_DRAFT_PROMPT,
        payload=transfer_payload, output=output, max_new_tokens=1536)
    leaks = _transfer_leaks(draft, reference, video)
    transfer = {
        "schema_version": "reference_transfer_draft_v1",
        "status": "human_review_pending" if not leaks else "blocked_leakage",
        "public_candidate": draft if not leaks else None,
        "leak_marker_indexes": leaks,
        "source_readout_sha": None,
        "production_release_allowed": False,
    }

    readout = {
        "schema_version": READOUT_VERSION,
        "status": "candidate_human_review_pending",
        "source_sha": reference["source_sha"],
        "agent_reference_sha": reference.get("artifact_sha"),
        "static_review_sha": static["artifact_sha"],
        "masked_video_sha": sha256_file(masked_video),
        "probe_plan": probes,
        "shot_dynamics": visual_dynamics,
        "narrative_reading": reading,
        "ending_observation": ending,
        "semantic_editing_patterns": patterns,
        "rejected_editing_patterns": rejected_patterns,
        "editing_functions": functions,
        "grounding_audit": audit,
        "citation_issues": citation_issues,
        "unresolved": static["unresolved"] + [
            f"editing_pattern_rejected:{row['probe_id']}:{row['raw_pattern_id']}"
            for row in rejected_patterns] + list(
            reading.get("limitations") or []) + list(
            functions.get("limitations") or []),
        "human_gold_evaluated": False,
    }
    readout["artifact_sha"] = json_hash(readout)
    transfer["source_readout_sha"] = readout["artifact_sha"]
    transfer["artifact_sha"] = json_hash(transfer)
    _write_json(output / "reference_readout_v1.json", readout)
    _write_json(output / "transfer_draft_v1.json", transfer)
    return readout
