"""Small, candidate-only bridge from audiovisual reference to story intent.

This path does not change the accepted reference, v2 blueprint, or R2-D v1.
"""
from __future__ import annotations

from typing import Any

from src.agentic_video.manifest import json_hash


INTENT_VERSION = "reference_intent_v1"
BRIEF_VERSION = "creative_story_brief_v1"
CORE_AUDIT_IDS = ("communicative_goal", "audience_path",
                  "evidence_mechanism", "attribution")
UNKNOWN = {"", "unknown", "unclear", "undetermined", "not_applicable"}

INTENT_PROMPT = """Watch this original audiovisual short video together with
its time-aligned accepted observations and attributed on-screen statements.
Infer what understanding the work invites its audience to take away, how
the audience's earlier and later understandings relate, and which concrete
information supports that reading. Do not restate every observation. Do not
assume a reversal, prejudice, competition, triumph, or any fixed story form.
On-screen text is a statement made by the video, not independent physical
proof. Avoid claims of unseen causation or a formal outcome. If a central
point cannot be inferred, say unknown. Cite only supplied claim/event IDs.
The ending and tone may remain unknown without making the central reading
unknown. Return one JSON object with these fields:
schema_version="reference_intent_analysis_v1";
communicative_goal={statement:string,support_ids:string[]};
audience_prior={interpretation:string,support_ids:string[]};
evidence_mechanism={description:string,support_ids:string[]};
audience_update={interpretation:string,support_ids:string[]};
ending={effect:string,support_ids:string[]};
tone={description:string};
beat_preferences=[{function:string,support_ids:string[]}];
limitations=string[]. Beat preferences are observed information functions,
not required roles for other videos. Return JSON only. Input: """

AUDIT_PROMPT = """Independently inspect the original audiovisual video,
time-aligned observations and attributed text, and the proposed intent.
Check only the four IDs in audit_ids: communicative_goal, audience_path,
evidence_mechanism, attribution. Return exactly one check per audit_ids item
in the same order; copy each ID verbatim and never combine IDs. The verdict
must be exactly one of supported, contested, insufficient, with a short
reason. A cited ID alone is not semantic support. Text stated by the
video is not an independently observed physical fact. Say whether any
important ending or other information contradicts the central reading.
If a core issue needs local media review, optionally request one interval
of no more than 5 seconds and state a neutral observation question; do not
include a proposed answer. Do not judge story salience or camera technique.
Return JSON only with schema_version="reference_intent_audit_v1",
checks:[{id,verdict,reason}], critical_conflict:{present:boolean,reason},
probe_request:null or {interval:[start_s,end_s],question:string}. Input: """

PROBE_PROMPT = """Observe this original audiovisual interval without
interpreting the full video's theme. Report only visible changes, audible
information, and attributed on-screen text, with relative times and any
uncertainty. Do not adopt or rebut a previous interpretation. Return JSON
only with schema_version="reference_intent_probe_v1", observations as an
array of {modality,time_s,description}, and limitations as an array of
strings. Modality is exactly one of V, A, T. Input: """

REVISE_PROMPT = """Reconsider the short video's communicative intent from
the accepted time-aligned observations and fresh local media observations.
The local observations may be incomplete. Do not see the earlier proposed
interpretation as evidence. Use the exact reference_intent_analysis_v1
field shape described below, citing only supplied claim/event IDs; new local
observations may be noted in limitations but do not create new accepted IDs.
On-screen text remains attributed. Return JSON only.
Fields: schema_version, communicative_goal{statement,support_ids},
audience_prior{interpretation,support_ids}, evidence_mechanism
{description,support_ids}, audience_update{interpretation,support_ids},
ending{effect,support_ids}, tone{description}, beat_preferences
[{function,support_ids}], limitations. Input: """

ABSTRACT_PROMPT = """Turn this private reference intent into an abstract
creative story brief for a different domain. Preserve the communicative
goal, the audience's information change, and the role of concrete evidence.
The beat list is a preference, not a mandatory structure. Do not copy or
paraphrase reference-specific people, physical attributes, activity, setting,
objects, quotations, exact times, source IDs, or media paths. Do not invent
unsupported camera or music techniques. Return JSON only:
{"schema_version":"creative_story_brief_v1","communicative_goal":"...",
"audience_prior":"...","evidence_mechanism":"...",
"audience_update":"...","tone":null,"beat_preferences":[],
"free_slots":[],"limitations":[]}. Input: """

BRIEF_AUDIT_PROMPT = """Compare the private intent with the proposed public
creative brief. Check that the communicative position and the role of
evidence survive, while reference-specific surface content, quotes, paths,
IDs and exact times do not. Beat preferences must not be mandatory roles.
Do not rewrite the draft or quote a leak in feedback. Return JSON only:
{"schema_version":"creative_story_brief_audit_v1",
"goal_preserved":false,"evidence_logic_preserved":false,
"source_surface_absent":false,"structure_optional":false,
"reason_codes":[]}. Input: """


def intent_payload(static: dict[str, Any]) -> dict[str, Any]:
    """Use accepted V/event/T observations, not old narrative conclusions."""
    return {"interval": [0.0, static["duration_s"]],
            "sections": [{"section_id": section["section_id"],
                          "interval": section["interval"],
                          "events": section.get("events") or [],
                          "visual_observations": section.get(
                              "visual_observations") or [],
                          "text_statements": [{**row,
                              "source_type": "attributed_on_screen_text"}
                              for row in section.get("text_statements") or []],
                          "audio_observations": section.get(
                              "audio_observations") or []}
                         for section in static["section_bundles"]],
            "audio_transcript_candidate": static.get(
                "audio_transcript_candidate"),
            "audio_transcript_verified": False}


def _known_ids(static: dict[str, Any]) -> set[str]:
    return {str(row[key]) for section in static["section_bundles"]
            for collection, key in (("events", "event_id"),
                                    ("visual_observations", "claim_id"),
                                    ("text_statements", "claim_id"),
                                    ("audio_observations", "claim_id"))
            for row in section.get(collection) or [] if key in row}


def _is_known(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() not in UNKNOWN


def validate_intent(analysis: dict[str, Any], static: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if analysis.get("schema_version") != "reference_intent_analysis_v1":
        errors.append("intent_schema_invalid")
    known = _known_ids(static)
    fields = (("communicative_goal", "statement"),
              ("audience_prior", "interpretation"),
              ("evidence_mechanism", "description"),
              ("audience_update", "interpretation"))
    for parent, field in fields:
        item = analysis.get(parent)
        if not isinstance(item, dict) or not _is_known(item.get(field)):
            errors.append(f"{parent}_missing")
            continue
        refs = item.get("support_ids")
        if not isinstance(refs, list) or not refs or set(refs) - known:
            errors.append(f"{parent}_support_invalid")
    for item in [analysis.get("ending"), *(analysis.get("beat_preferences") or [])]:
        if isinstance(item, dict) and set(item.get("support_ids") or []) - known:
            errors.append("optional_support_invalid")
    return sorted(set(errors))


def validate_intent_audit(audit: dict[str, Any]) -> list[str]:
    errors = []
    if audit.get("schema_version") != "reference_intent_audit_v1":
        errors.append("audit_schema_invalid")
    checks = audit.get("checks") or []
    if ([row.get("id") for row in checks] != list(CORE_AUDIT_IDS) or
            any(row.get("verdict") not in {"supported", "contested",
                                               "insufficient"} for row in checks)):
        errors.append("audit_coverage_or_verdict_invalid")
    conflict = audit.get("critical_conflict")
    if not isinstance(conflict, dict) or not isinstance(
            conflict.get("present"), bool):
        errors.append("critical_conflict_invalid")
    return errors


def build_intent(static: dict[str, Any], analysis: dict[str, Any],
                 audit: dict[str, Any], *, source_run: str,
                 model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    errors = validate_intent(analysis, static) + validate_intent_audit(audit)
    checks = audit.get("checks") or []
    ready = not errors and all(row.get("verdict") == "supported"
                               for row in checks) and audit.get(
                                   "critical_conflict", {}).get("present") is False
    result = {"schema_version": INTENT_VERSION,
              "source_sha": static["source_sha"],
              "static_review_sha": static["artifact_sha"],
              "source_run": source_run,
              "analysis": analysis, "audit": audit,
              "validation_issues": sorted(set(errors)),
              "story_candidate_ready": ready,
              "editing_candidate_ready": False,
              "production_release_allowed": False,
              "model_calls": model_calls}
    result["artifact_sha"] = json_hash(result)
    return result


def validate_story_brief(brief: dict[str, Any], intent: dict[str, Any],
                         forbidden_markers: list[str]) -> list[str]:
    errors = []
    expected = {"schema_version", "communicative_goal", "audience_prior",
                "evidence_mechanism", "audience_update", "tone",
                "beat_preferences", "free_slots", "limitations"}
    if brief.get("schema_version") != BRIEF_VERSION or set(brief) != expected:
        errors.append("brief_schema_invalid")
    for field in ("communicative_goal", "audience_prior",
                  "evidence_mechanism", "audience_update"):
        if not _is_known(brief.get(field)):
            errors.append(f"brief_{field}_missing")
    if not isinstance(brief.get("beat_preferences"), list) or not isinstance(
            brief.get("free_slots"), list) or not isinstance(
                brief.get("limitations"), list):
        errors.append("brief_lists_invalid")
    if not intent.get("story_candidate_ready"):
        errors.append("parent_intent_not_ready")
    serialized = str(brief).casefold()
    if any(marker and marker.casefold() in serialized
           for marker in forbidden_markers):
        errors.append("reference_surface_leak")
    return sorted(set(errors))


def publish_story_brief(brief: dict[str, Any], intent: dict[str, Any],
                        audit: dict[str, Any],
                        forbidden_markers: list[str]) -> dict[str, Any]:
    errors = validate_story_brief(brief, intent, forbidden_markers)
    if (audit.get("schema_version") != "creative_story_brief_audit_v1" or
            not all(audit.get(key) is True for key in (
                "goal_preserved", "evidence_logic_preserved",
                "source_surface_absent", "structure_optional"))):
        errors.append("brief_audit_not_passed")
    if errors:
        raise ValueError("story_brief_blocked:" + ",".join(sorted(set(errors))))
    result = {**brief, "parent_intent_sha": intent["artifact_sha"],
              "brief_audit_sha": json_hash(audit),
              "status": "model_checked_candidate",
              "production_release_allowed": False}
    result["artifact_sha"] = json_hash(result)
    return result
