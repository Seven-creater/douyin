"""Bounded AV analysis/audit shards for the reference transfer v2 experiment."""
from __future__ import annotations

from typing import Any

from src.agentic_video.reference_transfer_v2 import _pairs, _local_view


SECTION_PROMPT = """Inspect this original-sound VIDEO SECTION and the
time-aligned on-screen statements. The shot boundaries are accepted input,
not a model discovery. For every listed content shot, report its new visible,
stated, and audible information separately. For every listed adjacent pair,
report what the following shot newly adds and a possible editing function.
Do not merge short shots, invent unseen causation or a formal result, or treat
on-screen statements as independently observed physical facts. Distinguish
camera motion from subject motion; null means it is not clear. Descriptions
of editing function and cinematic technique remain provisional until audit.
Return JSON with exactly:
{"schema_version":"reference_section_analysis_v2","section_id":"...",
"section_contribution":"... or unknown",
"shots":[{"shot_id":"...","information_added":"... or unknown",
"action_phase":"... or unknown","shot_size":null,"camera_angle":null,
"camera_motion":null,"motion_speed":null,"support_probe_ids":[],
"statement_ids":[],"status":"provisional|insufficient",
"technique_status":"provisional|insufficient"}],
"edges":[{"from_shot_id":"...","to_shot_id":"...",
"information_added":"... or unknown",
"function_hypothesis":"... or unknown","support_probe_ids":[],
"status":"provisional|insufficient"}],"unknowns":[]}
Input: """

SECTION_AUDIT_PROMPT = """Independently check every listed shot and its
cinematic-technique fields and every adjacent edge against this original-
sound media section, attributed on-screen text, and cited high-rate local
observations. A text statement is not visual proof; discrete sampled frames
do not prove an entire action or unseen causal chain. An ID match alone is
not semantic support. Mark contested or insufficient when needed. Audit each
supplied ID exactly once, with no added importance or required-story-role
test. The input audit_ids is an exact ordered list. Output exactly one check
PER audit_ids item, in that order. Copy each ID character-for-character.
Never combine IDs with a vertical bar. Keep each reason under 20 words.
Return JSON only, with schema_version="reference_section_audit_v2" and a
checks array. Each check has id (copied from audit_ids), verdict (exactly
one of supported, contested, insufficient), and a brief evidence reason.
Input: """

GLOBAL_STORY_PROMPT = """Synthesize the three temporally ordered section
observations into an open-ended account of the video's communicative stance,
the viewer's earlier and later interpretation, and the contribution of the
final item. Pay attention to attributed on-screen statements, including the
last statement; explain whether it qualifies, jokes about, reinforces, or
does something else to the preceding content, or say unknown. Do not force a
reversal or predetermined ending, and do not turn a textual claim into a
physical fact. For the two CROSS-SECTION adjacent edges only, describe the
new information and a possible editing function. Cite supplied IDs. Return
JSON only:
{"schema_version":"reference_global_story_v2",
"theme_stance":{"position":"... or unknown","shot_ids":[],
"statement_ids":[],"alternative":"...","status":"provisional|insufficient"},
"viewer_change":{"prior":"... or unknown","later":"... or unknown",
"shot_ids":[],"statement_ids":[],"status":"provisional|insufficient"},
"ending":{"relation":"... or unknown","tone":"... or unknown",
"shot_ids":[],"statement_ids":[],"status":"provisional|insufficient"},
"narrative_units":[{"unit_id":"...","shot_ids":[],
"contribution":"...","status":"provisional|insufficient"}],
"cross_edges":[{"from_shot_id":"...","to_shot_id":"...",
"information_added":"... or unknown",
"function_hypothesis":"... or unknown","support_probe_ids":[],
"status":"provisional|insufficient"}],"unknowns":[]}
Input: """

GLOBAL_AUDIT_PROMPT = """Check this global story and the two cross-section
edit edges against the original audiovisual video and attributed on-screen
statements. Especially check that the final item's semantic content is not
omitted from the claimed ending relation. Do not accept an apparent formal
result or causal action unsupported by media. A reference ID match alone is
not semantic proof. The input audit_ids is an exact ordered list. Output
exactly one check PER audit_ids item, in that order; copy the ID literally.
Never combine IDs with a vertical bar. Keep each reason under 20 words.
Return JSON only, with schema_version="reference_global_audit_v2" and a
checks array. Each check has id (copied from audit_ids), verdict (exactly
one of supported, contested, insufficient), and a brief evidence reason.
Input: """


def section_payload(static: dict[str, Any], local: dict[str, Any],
                    section_id: str) -> dict[str, Any]:
    shots = [row for row in static["shots"] if row["section_id"] == section_id]
    if not shots:
        raise ValueError("section_not_found")
    ids = {row["shot_id"] for row in shots}
    view, _ = _local_view(local)
    statements = [row for row in static["text_timeline"] if any(
        row["claim_id"] in shot.get("text_claim_ids", []) for shot in shots)]
    return {"interval": [shots[0]["start_s"], shots[-1]["end_s"]],
            "section_id": section_id,
            "shots": [{"shot_id": row["shot_id"],
                       "relative_interval": [row["start_s"] - shots[0]["start_s"],
                                             row["end_s"] - shots[0]["start_s"]],
                       "visual_claim_ids": row.get("visual_claim_ids") or [],
                       "text_claim_ids": row.get("text_claim_ids") or []}
                      for row in shots],
            "expected_edges": [{"from_shot_id": a, "to_shot_id": b}
                               for a, b in _pairs(shots)],
            "text_statements": [{"statement_id": row["claim_id"],
                                 "source_type": "attributed_on_screen_text",
                                 "observed_text": row["observed_text"],
                                 "source_interval": row["interval"]}
                                for row in statements],
            "local_observations": [row for row in view if row["shot_id"] in ids]}


def section_audit_ids(static: dict[str, Any], section_id: str) -> list[str]:
    shots = [row for row in static["shots"] if row["section_id"] == section_id]
    return ([f"shot:{row['shot_id']}" for row in shots] +
            [f"technique:{row['shot_id']}" for row in shots] +
            [f"edge:{a}->{b}" for a, b in _pairs(shots)])


def global_story_payload(static: dict[str, Any], sections: list[dict[str, Any]]
                         ) -> dict[str, Any]:
    shots = static["shots"]
    cross = [(a, b) for a, b in _pairs(shots) if next(
        row["section_id"] for row in shots if row["shot_id"] == a) != next(
        row["section_id"] for row in shots if row["shot_id"] == b)]
    return {"sections": sections, "cross_edges": [{"from_shot_id": a,
                                                   "to_shot_id": b} for a, b in cross],
            "text_statements": [{"statement_id": row["claim_id"],
                                 "observed_text": row["observed_text"],
                                 "source_type": "attributed_on_screen_text",
                                 "interval": row["interval"]}
                                for row in static["text_timeline"]]}


def global_audit_ids(static: dict[str, Any]) -> list[str]:
    shots = static["shots"]
    section_of = {row["shot_id"]: row["section_id"] for row in shots}
    return ["theme_stance", "viewer_change", "ending"] + [
        f"edge:{a}->{b}" for a, b in _pairs(shots)
        if section_of[a] != section_of[b]]
