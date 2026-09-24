"""Isolated audience-message and cross-domain theme experiment."""
from __future__ import annotations

from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.transfer_kernel_v1 import normalize_private_intent


AUDIENCE_PROMPT = """Watch the original audiovisual short video with its
time-aligned accepted observations. Separate what the work specifically says
or shows from what understanding it invites an audience to take away. What,
if anything, should a viewer retain, change, or question after watching?
Identify the concrete information that supports that reading and its limits.
Do not assume a reversal, prejudice, triumph, or a moral lesson. If there is
no supported change in understanding, use "unknown". On-screen text is an
attributed statement by the video, not independent physical proof; a sound
or an image must not be credited with information it does not contain. Do not
infer unseen action, motive, or a formal result. Cite only supplied IDs.
Return JSON only, with exactly:
{"schema_version":"reference_message_v4",
"source_statement":{"text":"...","support_ids":["..."]},
"audience_before":{"text":"...","support_ids":["..."]},
"evidence_path":{"text":"...","support_ids":["..."]},
"audience_after":{"text":"...","support_ids":["..."]},
"takeaway_candidate":{"text":"...","support_ids":["..."]},
"ending":{"text":"...","support_ids":["..."]},
"limitations":[]}.
Input: """

ABSTRACTION_PROMPT = """Analyze this private account of a short video's
audience message. Distinguish the source-specific statement from the
relationship between an initial interpretation, the concrete information
that bears on it, and the later interpretation. The public message must
preserve the work's communicative position and the limits of its evidence;
it must not imply that one example proves unlimited ability or worth.
Do not merely replace a source noun with a broader noun from the same
domain. First propose a portable wording, then privately test its FULL
literal wording in three substantially different human settings. The three
settings must use different initial cues and different kinds of observable
evidence, and must differ from the source setting and one another. Report
each test, including failures; do not revise silently. These settings are
diagnostics, not creative themes. The source-specific statement and tests
remain private. The information order and ending are optional preferences,
not requirements for a different story. No reference-specific examples or
evaluation cases are supplied. Return JSON only, with exactly:
{"schema_version":"message_kernel_v4",
"source_claim":{"statement":"...","support_ids":["..."]},
"audience_takeaway":{"statement":"...","support_ids":["..."]},
"misjudgment_mechanism":{"statement":"...","support_ids":["..."]},
"corrective_evidence_role":{"statement":"...","support_ids":["..."]},
"revised_judgment":{"statement":"...","support_ids":["..."]},
"optional_sequence":["..."],"free_dimensions":["..."],
"private_substitutions":[{"context":"...","initial_cue":"...",
"observable_evidence":"...","literal_fit":true,"reason":"..."}],
"limitations":[]}.
Input: """

GROUNDING_PROMPT = """Check source grounding only. Compare each private
kernel statement with the accepted evidence and the private audience
reading. A cited ID does not itself establish semantic support. Preserve
attribution of text stated by the video and do not treat an image of a
person as proof of a caption's claim. A far-domain message may be an
interpretation, but its position and evidence mechanism must follow from
the source without adding unseen causes or outcomes. Do not judge domain
portability or candidate themes. Return exactly one check for each supplied
field path, in order, with verdict supported, contested, or insufficient:
{"schema_version":"message_grounding_audit_v4","checks":[
{"path":"source_claim","verdict":"supported","reason":"..."}]}.
Input: """

PORTABILITY_PROMPT = """Check only the domain portability of EVERY supplied
PUBLIC field, including optional strings. Compare with the private source
claim. A field is portable only when its complete literal wording remains
true after substantially different human situations replace the source
person, the initial cue, and the corrective activity. Use the three private
substitutions as tests, but assess them independently; their self-reported
literal_fit values are not proof. A source-specific assertion, a broader
version of its source domain, or a euphemism for it is not portable. Do not
rewrite fields or infer evaluation labels. Return one check per dotted path
in the supplied order, copying each path exactly:
{"schema_version":"message_portability_audit_v4","checks":[
{"path":"audience_takeaway","verdict":"portable","reason":"..."}]}.
Allowed verdicts: portable, source_domain_required, uncertain. Input: """

THEME_PROMPT = """Use only the public audience-message brief. Generate
exactly six original, filmable short-video themes. Vary the BASIS of the
initial judgment and the kind of corrective action, not merely the person's
profession or setting; use at least three distinct domains. Preserve every
hard message field; the optional
sequence may be changed. Each theme concerns one person's evaluation and
must give concrete, observable evidence for its revised judgment. Do not
invent a reference video, require one character per information function,
or turn a surprise alone into the message. Return JSON only:
{"schema_version":"message_theme_batch_v4","themes":[
{"theme_id":"T1","domain":"...","premise":"...",
"initial_cue":"...","mistaken_inference":"...",
"observable_evidence":"...","audience_update":"..."}]}.
Use IDs T1 through T6 exactly. Input: """

THEME_CRITIC_PROMPT = """Compare each diagnostic theme with the public
brief. Check separately whether its intended audience conclusion preserves
the complete message, its concrete evidence actually changes the initial
inference, and the action is filmable. Optional information order cannot
veto a theme. Do not infer an unavailable reference video. Return exactly
six checks in theme order as JSON only:
{"schema_version":"message_theme_critique_v4","checks":[
{"theme_id":"T1","message_match":false,"evidence_logic":false,
"filmable":false,"reason":"..."}]}.
Input: """

PRIVATE_FIELDS = ("source_claim", "audience_takeaway",
                  "misjudgment_mechanism", "corrective_evidence_role",
                  "revised_judgment")
HARD_FIELDS = PRIVATE_FIELDS[1:]
UNKNOWN = {"", "unknown", "unclear", "undetermined"}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _refs(value: Any, known_ids: set[str]) -> bool:
    return (isinstance(value, list) and bool(value) and
            all(isinstance(ref, str) for ref in value) and
            not (set(value) - known_ids))


def old_message_input(intent: dict[str, Any]) -> dict[str, Any]:
    """Give old and new readings the same abstraction input shape."""
    analysis = normalize_private_intent(intent)["analysis"]
    pairs = (("source_statement", "communicative_goal", "statement"),
             ("audience_before", "audience_prior", "interpretation"),
             ("evidence_path", "evidence_mechanism", "description"),
             ("audience_after", "audience_update", "interpretation"),
             ("takeaway_candidate", "communicative_goal", "statement"),
             ("ending", "ending", "effect"))
    return {"schema_version": "private_message_input_v4",
            **{dst: {"text": analysis[src][key],
                     "support_ids": analysis[src]["support_ids"]}
               for dst, src, key in pairs},
            "limitations": analysis["limitations"]}


def new_message_input(reading: dict[str, Any]) -> dict[str, Any]:
    return {**reading, "schema_version": "private_message_input_v4"}


def validate_message(reading: dict[str, Any], known_ids: set[str]) -> list[str]:
    expected = {"schema_version", "source_statement", "audience_before",
                "evidence_path", "audience_after", "takeaway_candidate",
                "ending", "limitations"}
    if set(reading) != expected or reading.get("schema_version") != \
            "reference_message_v4":
        return ["message_schema_invalid"]
    issues = []
    for key in expected - {"schema_version", "limitations"}:
        row = reading[key]
        if (not isinstance(row, dict) or set(row) != {"text", "support_ids"}
                or not _text(row["text"]) or not isinstance(
                    row["support_ids"], list) or
                any(not isinstance(ref, str) for ref in
                    row["support_ids"]) or
                set(row["support_ids"]) - known_ids):
            issues.append(f"{key}_invalid")
    if (not isinstance(reading["limitations"], list) or
            any(not _text(row) for row in reading["limitations"])):
        issues.append("limitations_invalid")
    return sorted(issues)


def validate_kernel(kernel: dict[str, Any], known_ids: set[str]) -> list[str]:
    expected = {"schema_version", *PRIVATE_FIELDS, "optional_sequence",
                "free_dimensions", "private_substitutions", "limitations"}
    if set(kernel) != expected or kernel.get("schema_version") != \
            "message_kernel_v4":
        return ["kernel_schema_invalid"]
    issues = []
    for key in PRIVATE_FIELDS:
        row = kernel[key]
        if (not isinstance(row, dict) or set(row) != {"statement", "support_ids"}
                or not _text(row["statement"]) or not _refs(
                    row["support_ids"], known_ids)):
            issues.append(f"{key}_invalid")
    for key in ("optional_sequence", "free_dimensions", "limitations"):
        if (not isinstance(kernel[key], list) or
                any(not _text(row) for row in kernel[key])):
            issues.append(f"{key}_invalid")
    substitutions = kernel["private_substitutions"]
    if not isinstance(substitutions, list) or len(substitutions) != 3:
        issues.append("private_substitutions_invalid")
    else:
        for row in substitutions:
            if (not isinstance(row, dict) or set(row) != {
                    "context", "initial_cue", "observable_evidence",
                    "literal_fit", "reason"} or
                    any(not _text(row.get(key)) for key in (
                        "context", "initial_cue", "observable_evidence",
                        "reason")) or type(row.get("literal_fit")) is not bool):
                issues.append("private_substitutions_invalid")
                break
        if len({row.get("initial_cue", "").casefold() for row in substitutions
                if isinstance(row, dict)}) != 3:
            issues.append("private_substitutions_not_distinct")
        if any(isinstance(row, dict) and row.get("literal_fit") is False
               for row in substitutions):
            issues.append("private_substitution_failed")
    return sorted(set(issues))


def public_brief(kernel: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "creative_message_brief_v4",
            **{key: kernel[key]["statement"] for key in HARD_FIELDS},
            "optional_sequence": kernel["optional_sequence"],
            "free_dimensions": kernel["free_dimensions"]}


def public_fields(brief: dict[str, Any], *, include_optional: bool
                  ) -> list[dict[str, str]]:
    fields = [{"path": key, "text": brief[key]} for key in HARD_FIELDS]
    if include_optional:
        fields += [{"path": f"{key}.{index}", "text": value}
                   for key in ("optional_sequence", "free_dimensions")
                   for index, value in enumerate(brief[key])]
    return fields


def validate_brief(brief: dict[str, Any]) -> list[str]:
    expected = {"schema_version", *HARD_FIELDS, "optional_sequence",
                "free_dimensions"}
    if set(brief) != expected or brief.get("schema_version") != \
            "creative_message_brief_v4":
        return ["brief_schema_invalid"]
    if (any(not _text(brief[key]) for key in HARD_FIELDS) or
            any(not isinstance(brief[key], list) or any(not _text(row)
                for row in brief[key]) for key in (
                    "optional_sequence", "free_dimensions"))):
        return ["brief_fields_invalid"]
    return []


def validate_audit(audit: dict[str, Any], fields: list[dict[str, str]],
                   *, kind: str) -> bool:
    checks = audit.get("checks")
    version = ("message_grounding_audit_v4" if kind == "grounding" else
               "message_portability_audit_v4")
    allowed = ({"supported", "contested", "insufficient"} if kind ==
               "grounding" else {"portable", "source_domain_required",
                                  "uncertain"})
    good = "supported" if kind == "grounding" else "portable"
    return (audit.get("schema_version") == version and
            isinstance(checks, list) and len(checks) == len(fields) and
            [row.get("path") for row in checks if isinstance(row, dict)] ==
            [row["path"] for row in fields] and
            all(row.get("verdict") in allowed and _text(row.get("reason"))
                for row in checks) and
            all(row["verdict"] == good for row in checks))


def validate_themes(batch: dict[str, Any]) -> list[str]:
    themes = batch.get("themes")
    if (batch.get("schema_version") != "message_theme_batch_v4" or
            not isinstance(themes, list) or len(themes) != 6 or
            [row.get("theme_id") for row in themes if isinstance(row, dict)]
            != [f"T{i}" for i in range(1, 7)]):
        return ["themes_schema_or_count_invalid"]
    expected = {"theme_id", "domain", "premise", "initial_cue",
                "mistaken_inference", "observable_evidence",
                "audience_update"}
    if any(set(row) != expected or any(not _text(row[key]) for key in
           expected) for row in themes):
        return ["theme_fields_invalid"]
    if len({row["initial_cue"].strip().casefold() for row in themes}) < 3:
        return ["theme_cues_not_distinct"]
    if len({row["domain"].strip().casefold() for row in themes}) < 3:
        return ["theme_domains_not_distinct"]
    return []


def validate_theme_critique(value: dict[str, Any]) -> list[str]:
    checks = value.get("checks")
    if (value.get("schema_version") != "message_theme_critique_v4" or
            not isinstance(checks, list) or len(checks) != 6 or
            [row.get("theme_id") for row in checks if isinstance(row, dict)]
            != [f"T{i}" for i in range(1, 7)]):
        return ["theme_critique_schema_invalid"]
    if any(set(row) != {"theme_id", "message_match", "evidence_logic",
                        "filmable", "reason"} or
           any(type(row[key]) is not bool for key in (
               "message_match", "evidence_logic", "filmable")) or
           not _text(row["reason"]) for row in checks):
        return ["theme_critique_fields_invalid"]
    return []


def artifact_with_sha(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "artifact_sha": json_hash(value)}
