"""Private source bindings and public relational message for the v5 trial."""
from __future__ import annotations

from typing import Any

from src.agentic_video.reference_message_v4 import (
    new_message_input as v4_new_input,
    old_message_input as v4_old_input,
)


ABSTRACTION_PROMPT = """Analyze this private account of a short video's
audience message. First identify distinct source information units and what
each contributes to the work. Then express their relationships using
anonymous role IDs and transferable functions. Finally state the audience
effect and the evidential limit of the relationship.

Keep the work's concrete statement, people, traits, activities, wording and
their evidence IDs in private source bindings. Do not make a source noun
portable by replacing it with a broader noun in the same domain. A public
relationship must be meaningful after its roles are rebound in a different
human setting. Preserve who asserts a claim: on-screen text is a statement
by the work, not independently established physical fact. Do not turn one
example into a universal claim. A tone or information order may be noted as
optional, not as a necessary relationship. The number and kinds of roles
and relationships are determined by the input; do not force a reversal,
character archetype or fixed act count. If a relationship is uncertain,
state that in limitations rather than inventing a cause. No target story or
evaluation example is supplied.

Return JSON only, with exactly this shape:
{"schema_version":"message_kernel_v5",
"source_claim":{"statement":"...","support_ids":["..."]},
"roles":[{"role_id":"R1","function":"...",
"source_binding":"...","support_ids":["..."]}],
"relations":[{"relation_id":"E1","from_role":"R1",
"to_role":"R2","relation":"...","scope":"..."}],
"audience_goal":"...","evidence_scope":"...",
"free_slots":["..."],"optional_tone":null,"limitations":[]}.
Input: """

GROUNDING_PROMPT = """Check the PRIVATE kernel against the accepted source
evidence and private reading. A valid ID alone is not semantic support.
Check each source binding and the audience interpretation it contributes,
including attribution of on-screen claims, causal direction and evidence
scope. An abstract relationship need only be a defensible interpretation
of the work, not a universally proven real-world proposition. Do not judge
target-domain portability or propose a rewrite. Return exactly one check
per supplied path in order, with verdict supported, contested or
insufficient. JSON only:
{"schema_version":"message_grounding_audit_v5","checks":[
{"path":"source_claim","verdict":"supported","reason":"..."}]}.
Input: """

PORTABILITY_PROMPT = """Independently audit the PUBLIC relational brief.
For every supplied field, ask whether its complete meaning depends on a
concrete person attribute, activity, setting or claim from the private
source. A source noun replaced by a broader noun in the same domain is
still source-bound. A relationship may transfer when anonymous roles can
be rebound while preserving the direction of inference, the evidence's
function and its limits. Do not demand that the source sentence itself
remain literally true in a new domain. Do not infer an unseen target story,
rewrite fields or use a self-reported pass/fail from the generator.
Return exactly one check per supplied path in order. Verdicts are portable,
source_domain_required or uncertain. JSON only:
{"schema_version":"message_portability_audit_v5","checks":[
{"path":"audience_goal","verdict":"portable","reason":"..."}]}.
Input: """

THEME_PROMPT = """Use only the public relational message. Generate exactly
six original filmable short-video themes. Preserve its audience goal,
direction of relationships and evidential limit, but freely rebind the
anonymous roles. Vary the BASIS of the initial judgment and the kind of
observable information, not just a job title or location. A surprising
ending alone is not enough. Optional tone is a preference. You do not know
the reference video; do not reconstruct one. Return JSON only:
{"schema_version":"message_theme_batch_v5","themes":[
{"theme_id":"T1","domain":"...","judgment_basis":"...",
"premise":"...","initial_cue":"...","mistaken_inference":"...",
"observable_evidence":"...","audience_update":"..."}]}.
Use IDs T1 through T6 exactly. Input: """

THEME_REVIEW_PROMPT = """Privately review each generated theme against the
public relational brief and the supplied source bindings. Check separately:
same audience position, whether concrete visible evidence really changes
the stated inference, filmability, and whether it merely copies source
person traits, activities or their distinctive combination. Classify if
the basis of judgment stays in the source's domain; a different profession
in the same source domain is still overlap. Report the basis category as a
short phrase, not just a yes/no. Do not revise a theme. Return six checks
in order as JSON only:
{"schema_version":"message_theme_review_v5","checks":[
{"theme_id":"T1","message_match":false,"evidence_logic":false,
"filmable":false,"surface_copy":false,"source_domain_overlap":false,
"basis_category":"...","reason":"..."}]}.
Input: """


def private_input(reading: dict[str, Any], *, old: bool) -> dict[str, Any]:
    normalized = v4_old_input(reading) if old else v4_new_input(reading)
    return {"schema_version": "private_message_input_v5",
            **{key: normalized[key] for key in (
                "source_statement", "audience_before", "evidence_path",
                "audience_after", "ending", "limitations")}}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _refs(value: Any, known_ids: set[str]) -> bool:
    return (isinstance(value, list) and bool(value) and
            all(isinstance(ref, str) for ref in value) and
            not (set(value) - known_ids))


def validate_kernel(value: dict[str, Any], known_ids: set[str]) -> list[str]:
    keys = {"schema_version", "source_claim", "roles", "relations",
            "audience_goal", "evidence_scope", "free_slots",
            "optional_tone", "limitations"}
    if not isinstance(value, dict) or set(value) != keys or value.get(
            "schema_version") != "message_kernel_v5":
        return ["kernel_schema_invalid"]
    issues = []
    source = value["source_claim"]
    if (not isinstance(source, dict) or set(source) != {
            "statement", "support_ids"} or not _text(source["statement"]) or
            not _refs(source["support_ids"], known_ids)):
        issues.append("source_claim_invalid")
    roles = value["roles"]
    if not isinstance(roles, list) or not roles or any(
            not isinstance(row, dict) or set(row) != {
                "role_id", "function", "source_binding", "support_ids"} or
            not all(_text(row[key]) for key in (
                "role_id", "function", "source_binding")) or
            not _refs(row["support_ids"], known_ids) for row in roles):
        issues.append("roles_invalid")
        role_ids = set()
    else:
        role_ids = {row["role_id"] for row in roles}
        if len(role_ids) != len(roles):
            issues.append("role_ids_duplicate")
    relations = value["relations"]
    if not isinstance(relations, list) or not relations or any(
            not isinstance(row, dict) or set(row) != {
                "relation_id", "from_role", "to_role", "relation", "scope"}
            or not all(_text(item) for item in row.values()) or
            row["from_role"] not in role_ids or row["to_role"] not in role_ids
            for row in relations):
        issues.append("relations_invalid")
    elif len({row["relation_id"] for row in relations}) != len(relations):
        issues.append("relation_ids_duplicate")
    for key in ("audience_goal", "evidence_scope"):
        if not _text(value[key]):
            issues.append(f"{key}_invalid")
    for key in ("free_slots", "limitations"):
        if not isinstance(value[key], list) or any(
                not _text(item) for item in value[key]):
            issues.append(f"{key}_invalid")
    if value["optional_tone"] is not None and not _text(value["optional_tone"]):
        issues.append("optional_tone_invalid")
    return issues


def public_brief(kernel: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "creative_message_brief_v5",
            "audience_goal": kernel["audience_goal"],
            "roles": [{"role_id": row["role_id"],
                       "function": row["function"]}
                      for row in kernel["roles"]],
            "relations": kernel["relations"],
            "evidence_scope": kernel["evidence_scope"],
            "free_slots": kernel["free_slots"],
            "optional_tone": kernel["optional_tone"]}


def public_fields(brief: dict[str, Any], *, include_optional: bool) -> list[dict]:
    fields = [{"path": "audience_goal", "text": brief["audience_goal"]}]
    functions = {row["role_id"]: row["function"] for row in brief["roles"]}
    fields += [{"path": f"roles.{row['role_id']}.function",
                "text": row["function"]} for row in brief["roles"]]
    fields += [{"path": f"relations.{row['relation_id']}.relation",
                "text": f"{functions[row['from_role']]} {row['relation']} "
                        f"{functions[row['to_role']]}; scope: {row['scope']}"}
               for row in brief["relations"]]
    fields.append({"path": "evidence_scope", "text": brief["evidence_scope"]})
    if include_optional:
        fields += [{"path": f"free_slots.{index}", "text": item}
                   for index, item in enumerate(brief["free_slots"])]
        if brief["optional_tone"] is not None:
            fields.append({"path": "optional_tone",
                           "text": brief["optional_tone"]})
    return fields


def validate_audit(value: dict[str, Any], fields: list[dict], *,
                   kind: str) -> list[str]:
    version = ("message_grounding_audit_v5" if kind == "grounding" else
               "message_portability_audit_v5")
    allowed = ({"supported", "contested", "insufficient"} if kind ==
               "grounding" else {"portable", "source_domain_required",
                                    "uncertain"})
    checks = value.get("checks") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema_version") != version or
            not isinstance(checks, list) or len(checks) != len(fields) or
            any(not isinstance(row, dict) or set(row) != {
                "path", "verdict", "reason"} for row in checks) or
            [row["path"] for row in checks] != [row["path"] for row in fields]
            or any(row["verdict"] not in allowed or not _text(row["reason"])
                   for row in checks)):
        return [f"{kind}_audit_invalid"]
    good = "supported" if kind == "grounding" else "portable"
    return [row["path"] for row in checks if row["verdict"] != good]


def validate_themes(batch: dict[str, Any]) -> list[str]:
    rows = batch.get("themes") if isinstance(batch, dict) else None
    expected = {"theme_id", "domain", "judgment_basis", "premise",
                "initial_cue", "mistaken_inference", "observable_evidence",
                "audience_update"}
    if (not isinstance(batch, dict) or batch.get(
            "schema_version") != "message_theme_batch_v5" or
            not isinstance(rows, list) or len(rows) != 6 or any(
                not isinstance(row, dict) or set(row) != expected or
                not all(_text(item) for item in row.values())
                for row in rows) or
            [row["theme_id"] for row in rows] != [f"T{i}" for i in range(1, 7)]):
        return ["themes_invalid"]
    return []


def validate_theme_review(review: dict[str, Any]) -> list[str]:
    rows = review.get("checks") if isinstance(review, dict) else None
    expected = {"theme_id", "message_match", "evidence_logic",
                "filmable", "surface_copy", "source_domain_overlap",
                "basis_category", "reason"}
    if (not isinstance(review, dict) or review.get(
            "schema_version") != "message_theme_review_v5" or
            not isinstance(rows, list) or len(rows) != 6 or any(
                not isinstance(row, dict) or set(row) != expected or
                any(type(row[key]) is not bool for key in (
                    "message_match", "evidence_logic", "filmable",
                    "surface_copy", "source_domain_overlap")) or
                not _text(row["basis_category"]) or not _text(row["reason"])
                for row in rows) or
            [row["theme_id"] for row in rows] != [f"T{i}" for i in range(1, 7)]):
        return ["theme_review_invalid"]
    return []
