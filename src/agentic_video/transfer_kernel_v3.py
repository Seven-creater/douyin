"""Source-grounded story units and a separately checked far-domain brief."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.manifest import json_hash


KERNEL_PROMPT = """Extract a source-grounded account of this reference intent.
Keep the video's specific claim separate from a communicative goal that a
different-domain story could enact. The video does not prove a universal
factual claim about every domain. Identify nonredundant story units. For each,
record the concrete source binding privately, a near concept, and its FAR
narrative function and audience effect. Preserve supported temporal and causal
relations; do not invent them or require a reversal. The far function and goal
must remain literally applicable when every concrete binding changes domain.
Merely replacing a source object by its parent category is not far abstraction.
For example, "paper stamp" -> "paper authorization" remains near, whereas
"an eligibility rule affects access" can be far; a changed bus route may
instead have the far function "new conditions change a plan". These generic
examples are not required story roles. On-screen text is attributed video
speech, not an independently verified physical fact. Return JSON only:
{"schema_version":"transfer_kernel_v3",
"source_claim":{"statement":"...","support_ids":["..."]},
"transfer_goal":{"statement":"...","mapping_rationale":"...",
"support_ids":["..."]},
"units":[{"unit_id":"U1","source_binding":"...",
"near_concept":"...","transfer_function":"...",
"audience_effect":"...","support_ids":["..."]}],
"relations":[{"relation_id":"R1","from_unit":"U1","to_unit":"U2",
"relation":"...","support_ids":["..."]}],
"optional_order":["U1"],"uncertainties":[]}.
Do not use evaluation examples or labels. Input: """

GROUNDING_PROMPT = """Audit only source grounding. Compare the source claim,
transfer goal, each unit, and each relation with the accepted evidence and
private intent. Check whether the proposed cross-domain goal faithfully
preserves the video's communicative position without presenting the video's
specific assertion as a universally established fact. A cited ID alone is
not semantic support. Preserve attribution of on-screen text. Do NOT judge
whether a candidate story fits and do not rewrite the kernel. Return exactly
one check for source_claim, transfer_goal, each unit ID, and each relation ID:
{"schema_version":"transfer_grounding_audit_v3","checks":[
{"id":"source_claim","verdict":"supported","reason":"..."}]}.
Allowed verdicts: supported, contested, insufficient. Input: """

PORTABILITY_PROMPT = """Audit only domain portability of the PUBLIC brief.
Compare it with the private source bindings, but do not use evaluation stories
or labels. The input contains an ordered fields array with exact dotted path
and text. Return one check for EVERY field, in the SAME order, copying each
path exactly; never invent a slash-delimited path. This includes both a
role's function and its audience_effect. A field is portable only when its
literal wording does not
require the source domain or a broader version of that same domain. Test two
substantially different, non-source-domain bindings for each field, from
different domains from each other, and explain whether the full field still
holds. Two variations inside the source domain do not demonstrate portability.
Do not excuse a restricted phrase by attending only to its generic clause.
Do not rewrite the brief. Return one check per supplied dotted field path:
{"schema_version":"transfer_portability_audit_v3","checks":[
{"path":"goal","verdict":"portable","reason":"...",
"tested_bindings":["...","..."]},
{"path":"roles.U1.audience_effect","verdict":"source_domain_required",
"reason":"...","tested_bindings":["...","..."]}]}.
Allowed verdicts: portable, source_domain_required, uncertain. Input: """

THEME_PROMPT = """Use only this public far-domain story brief. Propose
exactly three original, filmable short-video themes in distinct domains.
Preserve the communicative goal, the audience effects and the relations
among roles. Give every free_role_id a new concrete binding; do not assume
the original reference domain. A surprise alone is not a match. The optional
order is a preference, not an exact act, shot, or timing constraint.
You have no reference video and must not reconstruct one. JSON only:
{"schema_version":"transfer_theme_batch_v3","themes":[
{"theme_id":"T1","domain":"...","premise":"...",
"communicative_goal":"...","audience_prior":"...",
"evidence_mechanism":"...","audience_update":"...","tone":"...",
"role_bindings":[{"role_id":"U1","new_binding":"..."}]}]}.
Return T1, T2, T3 in order. Input: """

THEME_CRITIC_PROMPT = """Review each proposed theme using only the public
brief. Check the complete communicative goal, credible evidence mechanism,
every role's new-domain binding and its relations, originality, and whether
the events are filmable. A surprising ending or similar topic alone is not
enough. Optional order similarity is a soft 0-3 ranking, never a veto.
Return exactly one check for each theme in input order. JSON only:
{"schema_version":"intent_theme_critique_v1","checks":[
{"theme_id":"T1","goal_match":false,"evidence_logic":false,
"original":false,"filmable":false,"structure_preference":0,
"reason_codes":[]}]}.
Do not use an unavailable reference video. Input: """


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _refs(value: Any, known_ids: set[str]) -> bool:
    return isinstance(value, list) and bool(value) and all(
        _text(item) and item in known_ids for item in value)


def validate_kernel(kernel: dict[str, Any], known_ids: set[str]) -> list[str]:
    if (not isinstance(kernel, dict) or set(kernel) != {
            "schema_version", "source_claim", "transfer_goal", "units",
            "relations", "optional_order", "uncertainties"} or
            kernel.get("schema_version") != "transfer_kernel_v3"):
        return ["kernel_shape_invalid"]
    issues = []
    claim, goal = kernel["source_claim"], kernel["transfer_goal"]
    if (not isinstance(claim, dict) or set(claim) != {
            "statement", "support_ids"} or not _text(claim["statement"]) or
            not _refs(claim["support_ids"], known_ids)):
        issues.append("source_claim_invalid")
    if (not isinstance(goal, dict) or set(goal) != {
            "statement", "mapping_rationale", "support_ids"} or
            not _text(goal["statement"]) or not _text(goal["mapping_rationale"])
            or not _refs(goal["support_ids"], known_ids)):
        issues.append("transfer_goal_invalid")
    units = kernel["units"]
    if not isinstance(units, list) or not units:
        return issues + ["units_invalid"]
    unit_ids = []
    for unit in units:
        if (not isinstance(unit, dict) or set(unit) != {
                "unit_id", "source_binding", "near_concept",
                "transfer_function", "audience_effect", "support_ids"} or
                any(not _text(unit[key]) for key in (
                    "unit_id", "source_binding", "near_concept",
                    "transfer_function", "audience_effect")) or
                not _refs(unit["support_ids"], known_ids)):
            issues.append("unit_invalid")
        else:
            unit_ids.append(unit["unit_id"])
    if len(unit_ids) != len(set(unit_ids)):
        issues.append("unit_ids_invalid")
    relations = kernel["relations"]
    if not isinstance(relations, list):
        return sorted(set(issues + ["relations_invalid"]))
    relation_ids = []
    for relation in relations:
        if (not isinstance(relation, dict) or set(relation) != {
                "relation_id", "from_unit", "to_unit", "relation",
                "support_ids"} or not _text(relation["relation_id"]) or
                not _text(relation["relation"]) or
                relation["from_unit"] not in unit_ids or
                relation["to_unit"] not in unit_ids or
                not _refs(relation["support_ids"], known_ids)):
            issues.append("relation_invalid")
        else:
            relation_ids.append(relation["relation_id"])
    if len(relation_ids) != len(set(relation_ids)):
        issues.append("relation_ids_invalid")
    order = kernel["optional_order"]
    if (not isinstance(order, list) or any(item not in unit_ids for item in order)
            or len(order) != len(set(order))):
        issues.append("optional_order_invalid")
    if (not isinstance(kernel["uncertainties"], list) or any(
            not _text(item) for item in kernel["uncertainties"])):
        issues.append("uncertainties_invalid")
    return sorted(set(issues))


def public_brief_draft(kernel: dict[str, Any]) -> dict[str, Any]:
    brief = {"schema_version": "creative_story_brief_v3",
             "goal": kernel["transfer_goal"]["statement"],
             "roles": [{"role_id": row["unit_id"],
                        "function": row["transfer_function"],
                        "audience_effect": row["audience_effect"]}
                       for row in kernel["units"]],
             "relations": [{"relation_id": row["relation_id"],
                            "from_role": row["from_unit"],
                            "to_role": row["to_unit"],
                            "relation": row["relation"]}
                           for row in kernel["relations"]],
             "optional_order": list(kernel["optional_order"]),
             "free_role_ids": [row["unit_id"] for row in kernel["units"]]}
    validate_public_brief(brief, published=False)
    return brief


def validate_public_brief(brief: dict[str, Any], *, published: bool) -> None:
    expected = {"schema_version", "goal", "roles", "relations",
                "optional_order", "free_role_ids"}
    if published:
        expected |= {"status", "parent_kernel_sha", "grounding_audit_sha",
                     "portability_audit_sha", "mapping_eval_sha",
                     "stance_eval_sha", "regression_sha", "artifact_sha",
                     "production_release_allowed"}
    if (not isinstance(brief, dict) or set(brief) != expected or
            brief.get("schema_version") != "creative_story_brief_v3" or
            not _text(brief.get("goal"))):
        raise ValueError("public_brief_shape_invalid")
    roles = brief["roles"]
    if (not isinstance(roles, list) or not roles or any(
            not isinstance(row, dict) or set(row) != {
                "role_id", "function", "audience_effect"} or any(
                    not _text(value) for value in row.values()) for row in roles)):
        raise ValueError("public_roles_invalid")
    ids = [row["role_id"] for row in roles]
    if len(ids) != len(set(ids)):
        raise ValueError("public_role_ids_invalid")
    relations = brief["relations"]
    if (not isinstance(relations, list) or any(
            not isinstance(row, dict) or set(row) != {
                "relation_id", "from_role", "to_role", "relation"} or
            not all(_text(value) for value in row.values()) or
            row["from_role"] not in ids or row["to_role"] not in ids
            for row in relations)):
        raise ValueError("public_relations_invalid")
    if len({row["relation_id"] for row in relations}) != len(relations):
        raise ValueError("public_relation_ids_invalid")
    if (not isinstance(brief["optional_order"], list) or any(
            item not in ids for item in brief["optional_order"]) or
            len(brief["optional_order"]) != len(set(brief["optional_order"]))
            or brief["free_role_ids"] != ids):
        raise ValueError("public_order_or_slots_invalid")
    if published:
        if (brief["status"] != "model_checked_candidate" or
                brief["production_release_allowed"] is not False or
                any(not _text(brief[key]) for key in (
                    "parent_kernel_sha", "grounding_audit_sha",
                    "portability_audit_sha", "mapping_eval_sha",
                    "stance_eval_sha", "regression_sha")) or
                brief["artifact_sha"] != json_hash({
                    key: value for key, value in brief.items()
                    if key != "artifact_sha"})):
            raise ValueError("public_published_invalid")
    validate_creative_boundary(brief)


def audit_paths(brief: dict[str, Any]) -> list[str]:
    return (["goal"] + [f"roles.{row['role_id']}.{field}"
                        for row in brief["roles"]
                        for field in ("function", "audience_effect")] +
            [f"relations.{row['relation_id']}.relation"
             for row in brief["relations"]])


def public_fields(brief: dict[str, Any]) -> list[dict[str, str]]:
    fields = [{"path": "goal", "text": brief["goal"]}]
    for role in brief["roles"]:
        for key in ("function", "audience_effect"):
            fields.append({"path": f"roles.{role['role_id']}.{key}",
                           "text": role[key]})
    fields += [{"path": f"relations.{row['relation_id']}.relation",
                "text": f"{row['from_role']} {row['relation']} {row['to_role']}"}
               for row in brief["relations"]]
    return fields


def validate_themes(batch: dict[str, Any], brief: dict[str, Any]) -> list[str]:
    themes = batch.get("themes")
    if (batch.get("schema_version") != "transfer_theme_batch_v3" or
            not isinstance(themes, list) or
            [row.get("theme_id") for row in themes] != ["T1", "T2", "T3"]):
        return ["theme_count_or_schema_invalid"]
    expected = {"theme_id", "domain", "premise", "communicative_goal",
                "audience_prior", "evidence_mechanism", "audience_update",
                "tone", "role_bindings"}
    for theme in themes:
        if (set(theme) != expected or any(not _text(theme[key])
                for key in expected - {"role_bindings"}) or
                not isinstance(theme["role_bindings"], list) or
                [row.get("role_id") for row in theme["role_bindings"]] !=
                brief["free_role_ids"] or any(
                    not isinstance(row, dict) or set(row) != {
                        "role_id", "new_binding"} or
                    not _text(row["new_binding"])
                    for row in theme["role_bindings"])):
            return ["theme_fields_or_role_bindings_invalid"]
    if len({row["domain"].strip().casefold() for row in themes}) != 3:
        return ["theme_domains_not_distinct"]
    validate_creative_boundary(batch)
    return []


def validate_grounding(audit: dict[str, Any], kernel: dict[str, Any]) -> bool:
    expected = (["source_claim", "transfer_goal"] +
                [row["unit_id"] for row in kernel["units"]] +
                [row["relation_id"] for row in kernel["relations"]])
    checks = audit.get("checks")
    if (audit.get("schema_version") != "transfer_grounding_audit_v3" or
            not isinstance(checks, list) or
            [row.get("id") for row in checks] != expected or any(
                row.get("verdict") not in {
                    "supported", "contested", "insufficient"} or
                not _text(row.get("reason")) for row in checks)):
        raise ValueError("grounding_audit_invalid")
    return all(row["verdict"] == "supported" for row in checks)


def validate_portability(audit: dict[str, Any], brief: dict[str, Any]) -> bool:
    checks = audit.get("checks")
    if (audit.get("schema_version") != "transfer_portability_audit_v3" or
            not isinstance(checks, list) or
            [row.get("path") for row in checks] != audit_paths(brief) or any(
                row.get("verdict") not in {
                    "portable", "source_domain_required", "uncertain"} or
                not _text(row.get("reason")) or
                not isinstance(row.get("tested_bindings"), list) or
                len(row["tested_bindings"]) != 2 or
                any(not _text(item) for item in row["tested_bindings"]) or
                len(set(row["tested_bindings"])) != 2
                for row in checks)):
        raise ValueError("portability_audit_invalid")
    return all(row["verdict"] == "portable" for row in checks)


def publish_brief(draft: dict[str, Any], *, kernel_sha: str,
                  grounding: dict[str, Any], portability: dict[str, Any],
                  mapping: dict[str, Any], stance: dict[str, Any],
                  regression: dict[str, Any]) -> dict[str, Any]:
    if regression.get("passed") is not True:
        raise ValueError("regression_not_passed")
    brief = dict(draft)
    brief.update({"status": "model_checked_candidate",
                  "parent_kernel_sha": kernel_sha,
                  "grounding_audit_sha": json_hash(grounding),
                  "portability_audit_sha": json_hash(portability),
                  "mapping_eval_sha": json_hash(mapping),
                  "stance_eval_sha": json_hash(stance),
                  "regression_sha": json_hash(regression),
                  "production_release_allowed": False})
    brief["artifact_sha"] = json_hash(brief)
    validate_public_brief(brief, published=True)
    return brief


def make_acceptance(brief: dict[str, Any], *, record: dict[str, Any],
                    grounding: dict[str, Any], portability: dict[str, Any],
                    calibration: dict[str, Any], mapping: dict[str, Any],
                    stance: dict[str, Any], regression: dict[str, Any]
                    ) -> dict[str, Any]:
    validate_public_brief(brief, published=True)
    if (record.get("artifact_sha") != brief["parent_kernel_sha"] or
            record.get("artifact_sha") != json_hash({
                key: value for key, value in record.items()
                if key != "artifact_sha"}) or
            not validate_grounding(grounding, record["kernel"]) or
            not validate_portability(portability, brief) or
            calibration.get("passed") is not True or
            regression.get("passed") is not True or
            any(brief[field] != json_hash(artifact) for field, artifact in (
                ("grounding_audit_sha", grounding),
                ("portability_audit_sha", portability),
                ("mapping_eval_sha", mapping), ("stance_eval_sha", stance),
                ("regression_sha", regression)))):
        raise ValueError("trial_acceptance_checks_failed")
    value = {"schema_version": "trial_acceptance_v3",
             "brief_sha": brief["artifact_sha"],
             "kernel_sha": record["artifact_sha"],
             "source_intent_sha": record["source_intent_sha"],
             "source_media_sha": record["source_media_sha"],
             "grounding_audit_sha": json_hash(grounding),
             "portability_audit_sha": json_hash(portability),
             "calibration_sha": json_hash(calibration),
             "mapping_eval_sha": json_hash(mapping),
             "stance_eval_sha": json_hash(stance),
             "regression_sha": json_hash(regression),
             "auto_trial_allowed": True,
             "production_release_allowed": False}
    value["artifact_sha"] = json_hash(value)
    return value


def validate_acceptance(value: dict[str, Any], brief: dict[str, Any],
                        **artifacts: dict[str, Any]) -> None:
    expected = {"schema_version", "brief_sha", "kernel_sha",
                "source_intent_sha", "source_media_sha",
                "grounding_audit_sha", "portability_audit_sha",
                "calibration_sha", "mapping_eval_sha", "stance_eval_sha",
                "regression_sha", "auto_trial_allowed",
                "production_release_allowed", "artifact_sha"}
    if (set(value) != expected or value.get("schema_version") !=
            "trial_acceptance_v3" or value.get("brief_sha") != brief.get(
                "artifact_sha") or value.get("auto_trial_allowed") is not True
            or value.get("production_release_allowed") is not False or
            value.get("artifact_sha") != json_hash({
                key: item for key, item in value.items()
                if key != "artifact_sha"})):
        raise ValueError("trial_acceptance_invalid")
    try:
        expected_value = make_acceptance(brief, **artifacts)
    except ValueError as exc:
        raise ValueError("trial_acceptance_invalid") from exc
    if value != expected_value:
        raise ValueError("trial_acceptance_invalid")
