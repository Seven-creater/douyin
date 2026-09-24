"""Private relation abstraction and source-blind story brief trial."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_creative_boundary,
)
from src.agentic_video.manifest import json_hash


KERNEL_PROMPT = """Extract a transferable relational account from this
private, evidence-checked reference intent. Separate what must remain true
about the work's communicative position from source-specific bindings.
Give abstract roles and relationships; for each role, put the original
binding and evidence IDs ONLY in source_binding and support_ids. A binding
can be a character attribute, reason for an inference, activity, object,
setting, or concrete action. The invariant_function and stance must remain
true when every source binding changes to a different domain. Do not simply
rename a binding with a metaphor or broader category. Relations and optional
order are discovered from this reference, not a required template; either
may be empty where the evidence supports that. Do not infer unseen facts.
Return JSON only with exactly:
{"schema_version":"transfer_kernel_v1",
"stance":{"statement":"...","support_ids":["..."]},
"roles":[{"role_id":"V1","invariant_function":"...",
"source_binding":"...","rebindable_dimension":"...",
"support_ids":["..."]}],
"relations":[{"relation_id":"R1","from_role":"V1","to_role":"V2",
"relation":"...","support_ids":["..."]}],
"optional_order":["V1"],"uncertainties":[]}.
The input may contain generic feedback codes from a prior blocked draft;
they are not evidence or expected answers. Input: """

KERNEL_AUDIT_PROMPT = """Independently compare the proposed private kernel
with the evidence-checked intent. Check the stance, every role, and every
relation by ID. For each, decide supported, contested, or insufficient.
An evidence ID alone is not semantic support. In particular, identify any
field where a source binding, or a euphemism for it, was placed in stance,
invariant_function, relation, or rebindable_dimension. A free slot must be
able to take a substantially different-domain value. Do not reward a
pleasant-sounding but narrower message. Return exactly one check for stance
and each role/relation ID, with a nonempty reason. For any non-supported
check give a generic issue_code and field_path. Report all leaked fields
with their paths; do not rewrite the kernel. Return JSON only:
{"schema_version":"transfer_kernel_audit_v1",
"checks":[{"id":"stance","verdict":"supported","reason":"...",
"issue_code":null,"field_path":null}],
"binding_leaks":[{"field_path":"...","reason":"..."}]}.
Input: """

CONTRAST_PROMPT = """Evaluate each independent short-story candidate
against ONLY the public abstract story brief. Do not infer an unavailable
reference video. Judge separately whether it preserves the communicative
stance, whether concrete events enact the stated evidence/relationship
logic, and whether it merely copies a concrete surface binding specified
by the brief. Similar topic or a surprise alone is not enough. The optional
order is a soft preference, not a pass condition. Return one check for each
case ID in input order, with a short reason, as JSON only:
{"schema_version":"transfer_contrast_eval_v1","checks":[
{"case_id":"C1","stance_match":false,"evidence_logic":false,
"surface_copy":false,"soft_order_similarity":0,"reason":"..."}]}.
Input: """


def normalize_private_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional model-shape drift without changing the source run."""
    if not intent.get("story_candidate_ready") or not isinstance(
            intent.get("analysis"), dict):
        raise ValueError("parent_intent_not_ready")
    analysis = deepcopy(intent["analysis"])
    changes: dict[str, Any] = {}
    if isinstance(analysis.get("tone"), str):
        changes["tone"] = analysis["tone"]
        analysis["tone"] = {"description": analysis["tone"]}
    if isinstance(analysis.get("limitations"), str):
        changes["limitations"] = analysis["limitations"]
        analysis["limitations"] = [analysis["limitations"]]
    if (not isinstance(analysis.get("tone"), dict) or
            not isinstance(analysis["tone"].get("description"), str) or
            not isinstance(analysis.get("limitations"), list) or
            any(not isinstance(row, str) for row in analysis["limitations"])):
        raise ValueError("parent_optional_fields_invalid")
    return {"analysis": analysis, "normalization_originals": changes,
            "source_intent_sha": intent["artifact_sha"]}


def validate_kernel(kernel: dict[str, Any], known_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if set(kernel) != {"schema_version", "stance", "roles", "relations",
                       "optional_order", "uncertainties"} or kernel.get(
                           "schema_version") != "transfer_kernel_v1":
        return ["kernel_schema_invalid"]
    stance = kernel["stance"]
    if (not isinstance(stance, dict) or set(stance) != {"statement", "support_ids"}
            or not _text(stance["statement"]) or not _refs(
                stance["support_ids"], known_ids)):
        errors.append("stance_invalid")
    roles, relations = kernel["roles"], kernel["relations"]
    if not isinstance(roles, list) or not roles:
        return errors + ["roles_invalid"]
    role_ids = []
    for role in roles:
        if (not isinstance(role, dict) or set(role) != {
                "role_id", "invariant_function", "source_binding",
                "rebindable_dimension", "support_ids"} or any(
                    not _text(role[key]) for key in (
                        "role_id", "invariant_function", "source_binding",
                        "rebindable_dimension")) or not _refs(
                            role["support_ids"], known_ids)):
            errors.append("role_invalid")
        else:
            role_ids.append(role["role_id"])
    if len(set(role_ids)) != len(roles):
        errors.append("role_ids_invalid")
    if not isinstance(relations, list):
        return sorted(set(errors + ["relations_invalid"]))
    relation_ids = []
    for relation in relations:
        if (not isinstance(relation, dict) or set(relation) != {
                "relation_id", "from_role", "to_role", "relation",
                "support_ids"} or not _text(relation["relation_id"]) or
                not _text(relation["relation"]) or relation["from_role"] not in
                role_ids or relation["to_role"] not in role_ids or not _refs(
                    relation["support_ids"], known_ids)):
            errors.append("relation_invalid")
        else:
            relation_ids.append(relation["relation_id"])
    if len(set(relation_ids)) != len(relations):
        errors.append("relation_ids_invalid")
    if (not isinstance(kernel["optional_order"], list) or any(
            role_id not in role_ids for role_id in kernel["optional_order"]) or
            len(set(kernel["optional_order"])) != len(kernel["optional_order"])):
        errors.append("optional_order_invalid")
    if (not isinstance(kernel["uncertainties"], list) or any(
            not _text(item) for item in kernel["uncertainties"])):
        errors.append("uncertainties_invalid")
    return sorted(set(errors))


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _refs(value: Any, known_ids: set[str]) -> bool:
    return (isinstance(value, list) and bool(value) and all(
        _text(item) and item in known_ids for item in value))


def validate_kernel_audit(audit: dict[str, Any], kernel: dict[str, Any]
                          ) -> list[str]:
    expected = ["stance"] + [role["role_id"] for role in kernel["roles"]] + [
        relation["relation_id"] for relation in kernel["relations"]]
    checks = audit.get("checks")
    if (audit.get("schema_version") != "transfer_kernel_audit_v1" or
            not isinstance(checks, list) or any(
                not isinstance(row, dict) for row in checks) or
            [row.get("id") for row in checks] !=
            expected or not isinstance(audit.get("binding_leaks"), list)):
        return ["kernel_audit_shape_invalid"]
    errors = []
    for row in checks:
        if (row.get("verdict") not in {"supported", "contested", "insufficient"}
                or not _text(row.get("reason")) or (row["verdict"] != "supported"
                and (not _text(row.get("issue_code")) or not _text(
                    row.get("field_path"))))):
            errors.append("kernel_audit_check_invalid")
    if any(not isinstance(row, dict) or not _text(row.get("field_path")) or
           not _text(row.get("reason")) for row in audit["binding_leaks"]):
        errors.append("kernel_audit_leak_invalid")
    return sorted(set(errors))


def kernel_audit_passes(audit: dict[str, Any]) -> bool:
    return not audit["binding_leaks"] and all(
        row["verdict"] == "supported" for row in audit["checks"])


def audit_feedback_codes(audit: dict[str, Any]) -> list[dict[str, str]]:
    """Never include source phrases or hidden contrast labels in feedback."""
    result = [{"code": row["issue_code"], "field_path": row["field_path"]}
              for row in audit["checks"] if row["verdict"] != "supported"]
    result += [{"code": "binding_in_invariant",
                "field_path": row["field_path"]}
               for row in audit["binding_leaks"]]
    return result


def public_brief_draft(kernel: dict[str, Any]) -> dict[str, Any]:
    brief = {"schema_version": "creative_story_brief_v2",
             "stance": kernel["stance"]["statement"],
             "roles": [{"role_id": row["role_id"],
                        "function": row["invariant_function"]}
                       for row in kernel["roles"]],
             "relations": [{key: row[key] for key in (
                 "relation_id", "from_role", "to_role", "relation")}
                 for row in kernel["relations"]],
             "optional_order": list(kernel["optional_order"]),
             "free_slots": [{"role_id": row["role_id"],
                             "dimension": row["rebindable_dimension"]}
                            for row in kernel["roles"]]}
    validate_public_brief(brief, published=False)
    return brief


def validate_public_brief(brief: dict[str, Any], *, published: bool) -> None:
    expected = {"schema_version", "stance", "roles", "relations",
                "optional_order", "free_slots"}
    if published:
        expected |= {"status", "parent_kernel_sha", "kernel_audit_sha",
                     "contrast_eval_sha", "artifact_sha",
                     "production_release_allowed"}
    if set(brief) != expected or brief.get("schema_version") != (
            "creative_story_brief_v2") or not _text(brief.get("stance")):
        raise ValueError("public_brief_schema_invalid")
    roles = brief.get("roles")
    if not isinstance(roles, list) or not roles or any(
            not isinstance(row, dict) or set(row) != {"role_id", "function"}
            or not all(_text(value) for value in row.values()) for row in roles):
        raise ValueError("public_roles_invalid")
    role_ids = [row["role_id"] for row in roles]
    if len(set(role_ids)) != len(role_ids):
        raise ValueError("public_role_ids_invalid")
    relations = brief.get("relations")
    if not isinstance(relations, list) or any(
            not isinstance(row, dict) or set(row) != {
                "relation_id", "from_role", "to_role", "relation"} or
            not all(_text(value) for value in row.values()) or
            row["from_role"] not in role_ids or row["to_role"] not in role_ids
            for row in relations):
        raise ValueError("public_relations_invalid")
    if len({row["relation_id"] for row in relations}) != len(relations):
        raise ValueError("public_relation_ids_invalid")
    order, slots = brief.get("optional_order"), brief.get("free_slots")
    if (not isinstance(order, list) or any(not isinstance(item, str)
            for item in order) or len(set(order)) != len(order) or
            any(item not in role_ids for item in order) or
            not isinstance(slots, list) or len(slots) != len(roles) or any(
                not isinstance(row, dict) or set(row) != {
                    "role_id", "dimension"} or row["role_id"] not in role_ids
                or not _text(row["dimension"]) for row in slots)):
        raise ValueError("public_slots_or_order_invalid")
    if len({row["role_id"] for row in slots}) != len(slots):
        raise ValueError("public_slot_ids_invalid")
    validate_creative_boundary(brief)
    if published and (brief["status"] != "model_checked_candidate" or
                      brief["production_release_allowed"] is not False or
                      not all(_text(brief[key]) for key in (
                          "parent_kernel_sha", "kernel_audit_sha",
                          "contrast_eval_sha", "artifact_sha")) or
                      brief["artifact_sha"] != json_hash({
                          key: value for key, value in brief.items()
                          if key != "artifact_sha"})):
        raise ValueError("public_brief_lineage_invalid")


def validate_contrast_eval(evaluation: dict[str, Any], case_ids: list[str]
                           ) -> list[str]:
    checks = evaluation.get("checks")
    if (evaluation.get("schema_version") != "transfer_contrast_eval_v1" or
            not isinstance(checks, list) or any(
                not isinstance(row, dict) for row in checks) or
            [row.get("case_id") for row in
                                              checks] != case_ids):
        return ["contrast_shape_invalid"]
    return ["contrast_check_invalid"] if any(
        not isinstance(row.get(key), bool) for key in (
            "stance_match", "evidence_logic", "surface_copy") or
        type(row.get("soft_order_similarity")) is not int or not 0 <= row[
            "soft_order_similarity"] <= 3 or not _text(row.get("reason"))
        for row in checks) else []


def score_contrast(evaluation: dict[str, Any], labels: dict[str, bool]
                   ) -> dict[str, Any]:
    errors = validate_contrast_eval(evaluation, list(labels))
    if errors:
        raise ValueError(",".join(errors))
    observed = {row["case_id"]: row["stance_match"] and row[
        "evidence_logic"] and not row["surface_copy"] for row in evaluation[
            "checks"]}
    return {"passed": observed == labels,
            "case_results": [{"case_id": key, "expected": expected,
                              "observed": observed[key],
                              "matched": observed[key] == expected}
                             for key, expected in labels.items()]}


def publish_brief(draft: dict[str, Any], *, kernel_sha: str,
                  audit: dict[str, Any], evaluation: dict[str, Any],
                  contrast_score: dict[str, Any]) -> dict[str, Any]:
    validate_public_brief(draft, published=False)
    if not kernel_audit_passes(audit):
        raise ValueError("kernel_audit_blocked")
    if not contrast_score["passed"]:
        raise ValueError("contrast_eval_blocked")
    brief = {**draft, "status": "model_checked_candidate",
             "parent_kernel_sha": kernel_sha,
             "kernel_audit_sha": json_hash(audit),
             "contrast_eval_sha": json_hash(evaluation),
             "production_release_allowed": False}
    brief["artifact_sha"] = json_hash(brief)
    validate_public_brief(brief, published=True)
    return brief
