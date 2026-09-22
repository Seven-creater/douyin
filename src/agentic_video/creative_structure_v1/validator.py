"""Deterministic validation for Creative Structure Plan audit/spec artifacts."""
from __future__ import annotations

import re
from typing import Any, Iterable

from src.agentic_video.creative_structure_v1.minimizer import removable_item_ids
from src.agentic_video.creative_structure_v1.schema import (
    ANTI_INVARIANT_CATEGORIES,
    AUDIT_SCHEMA_VERSION,
    AUDIT_TOP_LEVEL_KEYS,
    BINDING_SLOT_TYPES,
    DIMENSIONS,
    DIMENSION_STATES,
    EDITING_DIMENSIONS,
    EDITING_OBLIGATIONS,
    EDITING_OPERATORS,
    ELEMENT_KINDS,
    GENERATION_CONSTRAINT_TYPES,
    GENERATION_OBLIGATIONS,
    PLAN_STATUSES,
    RELATION_KINDS,
    SOURCE_STRENGTHS,
    SPEC_SCHEMA_VERSION,
    SPEC_TOP_LEVEL_KEYS,
    TRANSFER_POLICIES,
    VALIDATION_RECORD_KEYS,
)
from src.agentic_video.manifest import json_hash

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(
    r"\b(?:N\d+|R\d+|EF\d+|EP\d+|FUNCTION_[A-Za-z0-9_]+|"
    r"PACE(?:_CHANGE)?_\d+|DURATION(?:_OUTLIER)?_\d+|"
    r"TRANSITION(?:_SEQUENCE)?_\d+)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/data/|/home/|\\data\\)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{4,}")

# Generic relation/state language is not a reference surface cue. Everything
# else is derived from the current verified bundle, never a case blacklist.
_ABSTRACT_VOCABULARY = {
    "about", "after", "against", "already", "available", "before", "between",
    "bears", "binding", "boundary", "broad", "causal", "change", "changes",
    "condition", "constraint", "context", "disclosure", "domain", "earlier",
    "effect", "eligible", "element", "evidence", "from", "function",
    "information", "initial", "interpretation", "later", "observable",
    "operation", "ordering", "predicate", "precondition", "prior",
    "proposition", "reference", "relation", "resolution", "revision", "scope",
    "state", "statement", "structure", "supported", "than", "that", "their",
    "then", "there", "these", "this", "those", "through", "transition",
    "updated", "what", "when", "where", "which", "while", "with", "without",
}


class StructureValidationError(RuntimeError):
    def __init__(self, reason_code: str, detail: object = "") -> None:
        super().__init__(f"{reason_code}:{detail}" if detail else reason_code)
        self.reason_code = reason_code
        self.detail = detail


def _fail(reason: str, detail: object = "") -> None:
    raise StructureValidationError(reason, detail)


def _object(value: object, reason: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(reason)
    return value


def _list(value: object, reason: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(reason)
    return value


def _exact_keys(value: dict[str, Any], expected: Iterable[str], reason: str) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    if missing or extra:
        _fail(reason, f"missing={missing},extra={extra}")


def _text(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(reason)
    return value


def _id(value: object, reason: str) -> str:
    text = _text(value, reason)
    if not _ID_RE.fullmatch(text):
        _fail(reason, text)
    return text


def _enum(value: object, allowed: Iterable[str], reason: str) -> str:
    if not isinstance(value, str) or value not in set(allowed):
        _fail(reason, value)
    return value


def _string_list(value: object, reason: str, *, nonempty: bool = False) -> list[str]:
    rows = _list(value, reason)
    if any(not isinstance(item, str) or not item.strip() for item in rows):
        _fail(reason)
    if nonempty and not rows:
        _fail(reason)
    if len(rows) != len(set(rows)):
        _fail(reason, "duplicate")
    return rows


def _confidence(value: object, reason: str = "confidence_invalid") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not 0.0 <= float(value) <= 1.0:
        _fail(reason, value)


def _hash_matches(value: dict[str, Any], reason: str) -> None:
    claimed = value.get("artifact_sha")
    payload = {key: item for key, item in value.items() if key != "artifact_sha"}
    if not isinstance(claimed, str) or claimed != json_hash(payload):
        _fail(reason)


def _support_refs(row: dict[str, Any], known: set[str], label: str) -> None:
    refs = _string_list(row.get("support_refs"),
                        f"{label}_support_refs_invalid", nonempty=True)
    unknown = sorted(set(refs) - known)
    if unknown:
        _fail("support_ref_unresolved", f"{label}:{unknown}")


def available_dimensions(value: dict[str, Any]) -> list[str]:
    return [
        dimension for dimension in DIMENSIONS
        if value["dimension_status"][dimension] == "supported"
    ]


def _content_presence(audit: dict[str, Any]) -> dict[str, bool]:
    structural = audit.get("structural_schema")
    editing = (audit.get("editing_schema") or {}).get("constraints") or []
    return {
        "structural_schema": bool(structural and (
            structural.get("elements") or structural.get("relations"))),
        "information_state_arc": bool(audit.get("information_state_arc")),
        "generation_constraints": bool(audit.get("generation_constraints")),
        "editing_structure": any(
            row.get("source_strength") == "measured" for row in editing),
        "editing_function": any(
            row.get("source_strength") != "measured" for row in editing),
        "binding_slots": bool(audit.get("binding_slots")),
    }


def _audit_items(audit: dict[str, Any]) -> dict[str, str]:
    structural = audit.get("structural_schema") or {"elements": [], "relations": []}
    rows = [
        *((str(row["element_id"]), "element") for row in structural["elements"]),
        *((str(row["relation_id"]), "relation") for row in structural["relations"]),
        *((str(row["state_id"]), "information_state")
          for row in audit["information_state_arc"]),
        *((str(row["constraint_id"]), "generation_constraint")
          for row in audit["generation_constraints"]),
        *((str(row["constraint_id"]), "editing_constraint")
          for row in audit["editing_schema"]["constraints"]),
        *((str(row["slot_id"]), "binding_slot") for row in audit["binding_slots"]),
    ]
    result: dict[str, str] = {}
    for item_id, item_type in rows:
        if item_id in result:
            _fail("abstract_id_collision", item_id)
        result[item_id] = item_type
    return result


def validate_structure_audit(value: dict[str, Any]) -> None:
    audit = _object(value, "audit_not_object")
    _exact_keys(audit, AUDIT_TOP_LEVEL_KEYS, "audit_keys_invalid")
    if audit["schema_version"] != AUDIT_SCHEMA_VERSION:
        _fail("audit_schema_invalid")
    status = _enum(audit["plan_status"], PLAN_STATUSES, "plan_status_invalid")
    _confidence(audit["abstraction_confidence"], "abstraction_confidence_invalid")

    dimensions = _object(audit["dimension_status"], "dimension_status_invalid")
    _exact_keys(dimensions, DIMENSIONS, "dimension_status_keys_invalid")
    for dimension in DIMENSIONS:
        _enum(dimensions[dimension], DIMENSION_STATES,
              "dimension_state_invalid")

    artifacts = _list(audit["input_artifacts"], "input_artifacts_invalid")
    if not artifacts:
        _fail("input_artifacts_empty")
    artifact_ids: set[str] = set()
    for value_row in artifacts:
        row = _object(value_row, "input_artifact_invalid")
        _exact_keys(row, {"artifact_id", "artifact_type", "sha", "schema_version"},
                    "input_artifact_keys_invalid")
        artifact_id = _id(row["artifact_id"], "input_artifact_id_invalid")
        if artifact_id in artifact_ids:
            _fail("input_artifact_duplicate", artifact_id)
        artifact_ids.add(artifact_id)
        _text(row["artifact_type"], "input_artifact_type_invalid")
        _text(row["schema_version"], "input_artifact_schema_invalid")
        if not isinstance(row["sha"], str) or not _SHA_RE.fullmatch(row["sha"]):
            _fail("input_artifact_sha_invalid", artifact_id)

    annex = _object(audit["audit_annex"], "audit_annex_invalid")
    _exact_keys(annex, {"source_bindings", "anti_invariants"},
                "audit_annex_keys_invalid")
    binding_rows = _list(annex["source_bindings"], "source_bindings_invalid")
    known_refs: set[str] = set()
    binding_ids: set[str] = set()
    for value_row in binding_rows:
        row = _object(value_row, "source_binding_invalid")
        _exact_keys(row, {"abstract_id", "source_refs", "reference_specific_summary"},
                    "source_binding_keys_invalid")
        abstract_id = _id(row["abstract_id"], "source_binding_id_invalid")
        if abstract_id in binding_ids:
            _fail("source_binding_duplicate", abstract_id)
        binding_ids.add(abstract_id)
        known_refs.update(_string_list(
            row["source_refs"], "source_binding_refs_invalid", nonempty=True))
        _text(row["reference_specific_summary"], "source_binding_summary_invalid")

    structural_value = audit["structural_schema"]
    element_ids: set[str] = set()
    relation_ids: set[str] = set()
    if structural_value is not None:
        structural = _object(structural_value, "structural_schema_invalid")
        _exact_keys(structural, {"elements", "relations"},
                    "structural_schema_keys_invalid")
        for value_row in _list(structural["elements"], "structural_elements_invalid"):
            row = _object(value_row, "structural_element_invalid")
            _exact_keys(row, {"element_id", "kind", "abstract_role", "support_refs"},
                        "structural_element_keys_invalid")
            element_id = _id(row["element_id"], "structural_element_id_invalid")
            if element_id in element_ids:
                _fail("structural_element_duplicate", element_id)
            element_ids.add(element_id)
            _enum(row["kind"], ELEMENT_KINDS, "structural_element_kind_invalid")
            _text(row["abstract_role"], "structural_element_role_invalid")
            _support_refs(row, known_refs, element_id)
        for value_row in _list(structural["relations"], "structural_relations_invalid"):
            row = _object(value_row, "structural_relation_invalid")
            _exact_keys(row, {
                "relation_id", "relation_kind", "predicate", "arguments",
                "preconditions", "effects", "support_refs", "confidence",
            }, "structural_relation_keys_invalid")
            relation_id = _id(row["relation_id"], "structural_relation_id_invalid")
            if relation_id in relation_ids:
                _fail("structural_relation_duplicate", relation_id)
            relation_ids.add(relation_id)
            _enum(row["relation_kind"], RELATION_KINDS,
                  "structural_relation_kind_invalid")
            _text(row["predicate"], "structural_predicate_invalid")
            arguments = _list(row["arguments"], "structural_arguments_invalid")
            if not arguments:
                _fail("structural_arguments_invalid", relation_id)
            roles: set[str] = set()
            for value_argument in arguments:
                argument = _object(value_argument, "structural_argument_invalid")
                _exact_keys(argument, {"role", "element_id"},
                            "structural_argument_keys_invalid")
                role = _id(argument["role"], "structural_argument_role_invalid")
                if role in roles:
                    _fail("structural_argument_role_duplicate", relation_id)
                roles.add(role)
                if argument["element_id"] not in element_ids:
                    _fail("relation_element_unresolved", relation_id)
            _string_list(row["preconditions"], "relation_preconditions_invalid")
            _string_list(row["effects"], "relation_effects_invalid")
            _support_refs(row, known_refs, relation_id)
            _confidence(row["confidence"])

    state_ids: set[str] = set()
    states = _list(audit["information_state_arc"], "information_state_arc_invalid")
    for value_row in states:
        row = _object(value_row, "information_state_invalid")
        _exact_keys(row, {
            "state_id", "order", "available_element_ids",
            "transition_from_state_id", "trigger_relation_ids", "support_refs",
        }, "information_state_keys_invalid")
        state_id = _id(row["state_id"], "information_state_id_invalid")
        if state_id in state_ids:
            _fail("information_state_duplicate", state_id)
        state_ids.add(state_id)
        if isinstance(row["order"], bool) or not isinstance(row["order"], int) \
                or row["order"] < 1:
            _fail("information_state_order_invalid", state_id)
        available = _string_list(row["available_element_ids"],
                                 "information_elements_invalid", nonempty=True)
        if set(available) - element_ids:
            _fail("information_element_unresolved", state_id)
        previous = row["transition_from_state_id"]
        if previous is not None:
            _id(previous, "information_previous_state_invalid")
        triggers = _string_list(row["trigger_relation_ids"],
                                "information_trigger_refs_invalid")
        if set(triggers) - relation_ids:
            _fail("information_relation_unresolved", state_id)
        _support_refs(row, known_refs, state_id)

    ordered_states = sorted(states, key=lambda row: row["order"])
    if [row["order"] for row in ordered_states] != list(range(1, len(states) + 1)):
        _fail("information_state_order_invalid")
    prior_by_order: set[str] = set()
    for index, row in enumerate(ordered_states):
        state_id = row["state_id"]
        previous = row["transition_from_state_id"]
        triggers = row["trigger_relation_ids"]
        if index == 0:
            if previous is not None or triggers:
                _fail("information_initial_state_invalid", state_id)
        else:
            if previous not in prior_by_order:
                _fail("information_previous_state_unresolved", state_id)
            if not triggers:
                _fail("information_transition_trigger_missing", state_id)
        prior_by_order.add(state_id)

    generation_ids: set[str] = set()
    for value_row in _list(audit["generation_constraints"],
                           "generation_constraints_invalid"):
        row = _object(value_row, "generation_constraint_invalid")
        _exact_keys(row, {
            "constraint_id", "obligation", "constraint_type",
            "requires_relation_ids", "rule", "support_refs", "confidence",
        }, "generation_constraint_keys_invalid")
        constraint_id = _id(row["constraint_id"], "generation_constraint_id_invalid")
        if constraint_id in generation_ids:
            _fail("generation_constraint_duplicate", constraint_id)
        generation_ids.add(constraint_id)
        _enum(row["obligation"], GENERATION_OBLIGATIONS,
              "generation_obligation_invalid")
        _enum(row["constraint_type"], GENERATION_CONSTRAINT_TYPES,
              "generation_constraint_type_invalid")
        refs = _string_list(row["requires_relation_ids"],
                            "generation_relation_refs_invalid", nonempty=True)
        if set(refs) - relation_ids:
            _fail("generation_relation_unresolved", constraint_id)
        _text(row["rule"], "generation_rule_invalid")
        _support_refs(row, known_refs, constraint_id)
        _confidence(row["confidence"])

    editing = _object(audit["editing_schema"], "editing_schema_invalid")
    _exact_keys(editing, {"constraints"}, "editing_schema_keys_invalid")
    editing_ids: set[str] = set()
    for value_row in _list(editing["constraints"], "editing_constraints_invalid"):
        row = _object(value_row, "editing_constraint_invalid")
        _exact_keys(row, {
            "constraint_id", "obligation", "dimension", "operator",
            "applies_to_state_ids", "applies_to_relation_ids", "source_strength",
            "transfer_policy", "support_refs", "confidence",
        }, "editing_constraint_keys_invalid")
        constraint_id = _id(row["constraint_id"], "editing_constraint_id_invalid")
        if constraint_id in editing_ids:
            _fail("editing_constraint_duplicate", constraint_id)
        editing_ids.add(constraint_id)
        obligation = _enum(row["obligation"], EDITING_OBLIGATIONS,
                           "editing_obligation_invalid")
        _enum(row["dimension"], EDITING_DIMENSIONS, "editing_dimension_invalid")
        _enum(row["operator"], EDITING_OPERATORS, "editing_operator_invalid")
        state_refs = _string_list(row["applies_to_state_ids"],
                                  "editing_state_refs_invalid")
        relation_refs = _string_list(row["applies_to_relation_ids"],
                                     "editing_relation_refs_invalid")
        if set(state_refs) - state_ids:
            _fail("editing_state_unresolved", constraint_id)
        if set(relation_refs) - relation_ids:
            _fail("editing_relation_unresolved", constraint_id)
        strength = _enum(row["source_strength"], SOURCE_STRENGTHS,
                         "editing_source_strength_invalid")
        policy = _enum(row["transfer_policy"], TRANSFER_POLICIES,
                       "editing_transfer_policy_invalid")
        if strength == "unaudited_semantic" and (
                obligation != "advisory" or policy != "free_realization"):
            _fail("unaudited_semantic_promoted", constraint_id)
        if strength == "measured" and obligation not in {"required", "preferred"}:
            _fail("measured_editing_obligation_invalid", constraint_id)
        _support_refs(row, known_refs, constraint_id)
        _confidence(row["confidence"])

    slot_ids: set[str] = set()
    for value_row in _list(audit["binding_slots"], "binding_slots_invalid"):
        row = _object(value_row, "binding_slot_invalid")
        _exact_keys(row, {"slot_id", "slot_type", "bound_by", "constraints"},
                    "binding_slot_keys_invalid")
        slot_id = _id(row["slot_id"], "binding_slot_id_invalid")
        if slot_id in slot_ids:
            _fail("binding_slot_duplicate", slot_id)
        slot_ids.add(slot_id)
        _enum(row["slot_type"], BINDING_SLOT_TYPES, "binding_slot_type_invalid")
        if row["bound_by"] != "downstream_skill":
            _fail("binding_slot_authority_invalid", slot_id)
        _string_list(row["constraints"], "binding_slot_constraints_invalid",
                     nonempty=True)

    anti_ids: set[str] = set()
    for value_row in _list(annex["anti_invariants"], "anti_invariants_invalid"):
        row = _object(value_row, "anti_invariant_invalid")
        _exact_keys(row, {"binding_id", "category", "source_refs"},
                    "anti_invariant_keys_invalid")
        binding_id = _id(row["binding_id"], "anti_invariant_id_invalid")
        if binding_id in anti_ids:
            _fail("anti_invariant_duplicate", binding_id)
        anti_ids.add(binding_id)
        _enum(row["category"], ANTI_INVARIANT_CATEGORIES,
              "anti_invariant_category_invalid")
        refs = _string_list(row["source_refs"], "anti_invariant_refs_invalid",
                            nonempty=True)
        if set(refs) - known_refs:
            _fail("support_ref_unresolved", binding_id)

    items = _audit_items(audit)
    if binding_ids != set(items):
        _fail("source_binding_coverage_invalid",
              f"missing={sorted(set(items) - binding_ids)},"
              f"extra={sorted(binding_ids - set(items))}")

    record = _object(audit["validation_record"], "validation_record_invalid")
    _exact_keys(record, VALIDATION_RECORD_KEYS, "validation_record_keys_invalid")
    if any(not isinstance(record[key], bool) for key in VALIDATION_RECORD_KEYS):
        _fail("validation_record_value_invalid")

    presence = _content_presence(audit)
    for dimension in DIMENSIONS:
        if presence[dimension] != (dimensions[dimension] == "supported"):
            _fail("dimension_content_mismatch", dimension)
    supported = available_dimensions(audit)
    insufficient = [d for d in DIMENSIONS if dimensions[d] == "insufficient"]
    if status == "complete":
        if insufficient:
            _fail("complete_has_insufficient_dimension", insufficient)
        if not supported:
            _fail("complete_has_no_supported_dimension")
        if dimensions["structural_schema"] != "supported" \
                or dimensions["generation_constraints"] != "supported":
            _fail("complete_core_dimensions_missing")
    elif status == "partial":
        if not supported:
            _fail("partial_has_no_supported_dimension")
        if not insufficient:
            _fail("partial_missing_insufficient_dimension")
    else:
        if supported or any(presence.values()):
            _fail("blocked_has_supported_content")

    removable = sorted(removable_item_ids(audit))
    if removable:
        _fail("structure_not_minimal", removable)
    _hash_matches(audit, "audit_sha_invalid")


def _source_text(bundle: dict[str, Any]) -> Iterable[str]:
    narrative = bundle.get("narrative_interpretation") or {}
    for unit in narrative.get("narrative_units") or []:
        yield str(unit.get("summary") or "")
    for relation in narrative.get("relations") or []:
        yield str(relation.get("reason") or "")


def _surface_vocabulary(bundle: dict[str, Any]) -> tuple[set[str], set[str]]:
    english: set[str] = set()
    chinese: set[str] = set()
    for text in _source_text(bundle):
        english.update(
            word.lower() for word in _WORD_RE.findall(text)
            if word.lower() not in _ABSTRACT_VOCABULARY
        )
        chinese.update(_CHINESE_RE.findall(text))
    return english, chinese


def _public_text(audit: dict[str, Any]) -> list[tuple[str, str, str]]:
    structural = audit.get("structural_schema") or {"elements": [], "relations": []}
    rows: list[tuple[str, str, str]] = []
    for element in structural["elements"]:
        rows.append((element["element_id"], "abstract_role", element["abstract_role"]))
    for relation in structural["relations"]:
        rows.append((relation["relation_id"], "predicate", relation["predicate"]))
        for index, argument in enumerate(relation["arguments"]):
            rows.append((relation["relation_id"], f"arguments[{index}].role",
                         argument["role"]))
        for field in ("preconditions", "effects"):
            for index, text in enumerate(relation[field]):
                rows.append((relation["relation_id"], f"{field}[{index}]", text))
    for constraint in audit["generation_constraints"]:
        rows.append((constraint["constraint_id"], "rule", constraint["rule"]))
    for slot in audit["binding_slots"]:
        for index, text in enumerate(slot["constraints"]):
            rows.append((slot["slot_id"], f"constraints[{index}]", text))
    return rows


def validate_abstraction_boundary(audit: dict[str, Any],
                                  bundle: dict[str, Any]) -> None:
    """Reject source-event passthrough without prescribing a story template."""
    source_ids = {
        str(row.get("evidence_id")) for row in bundle.get("evidence_catalog") or []
    }
    reused = sorted(set(_audit_items(audit)).intersection(source_ids))
    if reused:
        _fail("abstract_id_reuses_source_id", reused)

    source_english, source_chinese = _surface_vocabulary(bundle)
    for item_id, field, text in _public_text(audit):
        public_english = {word.lower() for word in _WORD_RE.findall(text)}
        overlap = sorted(public_english.intersection(source_english))
        overlap.extend(sorted(phrase for phrase in source_chinese if phrase in text))
        if overlap:
            _fail("reference_surface_in_public_field",
                  f"{item_id}.{field}:{','.join(overlap)}")


def _walk(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, str):
        yield value


def validate_structure_spec(value: dict[str, Any]) -> None:
    spec = _object(value, "spec_not_object")
    _exact_keys(spec, SPEC_TOP_LEVEL_KEYS, "spec_keys_invalid")
    if spec["schema_version"] != SPEC_SCHEMA_VERSION:
        _fail("spec_schema_invalid")
    _text(spec["spec_id"], "spec_id_invalid")
    _enum(spec["plan_status"], {"complete", "partial"}, "spec_status_invalid")
    dimensions = _object(spec["dimension_status"], "spec_dimension_status_invalid")
    _exact_keys(dimensions, DIMENSIONS, "spec_dimension_keys_invalid")
    for dimension in DIMENSIONS:
        _enum(dimensions[dimension], DIMENSION_STATES, "spec_dimension_state_invalid")
    available = _string_list(spec["available_dimensions"],
                             "available_dimensions_invalid", nonempty=True)
    expected_available = [d for d in DIMENSIONS if dimensions[d] == "supported"]
    if available != expected_available:
        _fail("available_dimensions_mismatch")

    structural = spec["structural_schema"]
    element_ids: set[str] = set()
    relation_ids: set[str] = set()
    if structural is not None:
        structural = _object(structural, "spec_structural_schema_invalid")
        _exact_keys(structural, {"elements", "relations"},
                    "spec_structural_schema_keys_invalid")
        for row in _list(structural["elements"], "spec_elements_invalid"):
            _exact_keys(_object(row, "spec_element_invalid"),
                        {"element_id", "kind", "abstract_role"},
                        "spec_element_keys_invalid")
            element_id = _id(row["element_id"], "spec_element_id_invalid")
            if element_id in element_ids:
                _fail("spec_element_duplicate", element_id)
            element_ids.add(element_id)
            _enum(row["kind"], ELEMENT_KINDS, "spec_element_kind_invalid")
            _text(row["abstract_role"], "spec_element_role_invalid")
        for row in _list(structural["relations"], "spec_relations_invalid"):
            _exact_keys(_object(row, "spec_relation_invalid"), {
                "relation_id", "relation_kind", "predicate", "arguments",
                "preconditions", "effects",
            }, "spec_relation_keys_invalid")
            relation_id = _id(row["relation_id"], "spec_relation_id_invalid")
            if relation_id in relation_ids:
                _fail("spec_relation_duplicate", relation_id)
            relation_ids.add(relation_id)
            _enum(row["relation_kind"], RELATION_KINDS,
                  "spec_relation_kind_invalid")
            _text(row["predicate"], "spec_predicate_invalid")
            for argument in _list(row["arguments"], "spec_arguments_invalid"):
                _exact_keys(_object(argument, "spec_argument_invalid"),
                            {"role", "element_id"}, "spec_argument_keys_invalid")
                _id(argument["role"], "spec_argument_role_invalid")
                if argument["element_id"] not in element_ids:
                    _fail("spec_relation_element_unresolved", relation_id)
            _string_list(row["preconditions"], "spec_preconditions_invalid")
            _string_list(row["effects"], "spec_effects_invalid")

    state_ids: set[str] = set()
    for row in _list(spec["information_state_arc"], "spec_information_arc_invalid"):
        _exact_keys(_object(row, "spec_information_state_invalid"), {
            "state_id", "order", "available_element_ids",
            "transition_from_state_id", "trigger_relation_ids",
        }, "spec_information_state_keys_invalid")
        state_id = _id(row["state_id"], "spec_information_state_id_invalid")
        if state_id in state_ids:
            _fail("spec_information_state_duplicate", state_id)
        state_ids.add(state_id)
        if set(row["available_element_ids"]) - element_ids \
                or set(row["trigger_relation_ids"]) - relation_ids:
            _fail("spec_information_reference_unresolved", state_id)

    generation_ids: set[str] = set()
    for row in _list(spec["generation_constraints"],
                     "spec_generation_constraints_invalid"):
        _exact_keys(_object(row, "spec_generation_constraint_invalid"), {
            "constraint_id", "obligation", "constraint_type",
            "requires_relation_ids", "rule",
        }, "spec_generation_constraint_keys_invalid")
        constraint_id = _id(row["constraint_id"], "spec_generation_id_invalid")
        if constraint_id in generation_ids:
            _fail("spec_generation_duplicate", constraint_id)
        generation_ids.add(constraint_id)
        _enum(row["obligation"], GENERATION_OBLIGATIONS,
              "spec_generation_obligation_invalid")
        _enum(row["constraint_type"], GENERATION_CONSTRAINT_TYPES,
              "spec_generation_type_invalid")
        if set(row["requires_relation_ids"]) - relation_ids:
            _fail("spec_generation_relation_unresolved", constraint_id)
        _text(row["rule"], "spec_generation_rule_invalid")

    editing = _object(spec["editing_schema"], "spec_editing_schema_invalid")
    _exact_keys(editing, {"constraints"}, "spec_editing_schema_keys_invalid")
    editing_ids: set[str] = set()
    for row in _list(editing["constraints"], "spec_editing_constraints_invalid"):
        _exact_keys(_object(row, "spec_editing_constraint_invalid"), {
            "constraint_id", "obligation", "dimension", "operator",
            "applies_to_state_ids", "applies_to_relation_ids", "transfer_policy",
        }, "spec_editing_constraint_keys_invalid")
        constraint_id = _id(row["constraint_id"], "spec_editing_id_invalid")
        if constraint_id in editing_ids:
            _fail("spec_editing_duplicate", constraint_id)
        editing_ids.add(constraint_id)
        _enum(row["obligation"], EDITING_OBLIGATIONS,
              "spec_editing_obligation_invalid")
        _enum(row["dimension"], EDITING_DIMENSIONS, "spec_editing_dimension_invalid")
        _enum(row["operator"], EDITING_OPERATORS, "spec_editing_operator_invalid")
        _enum(row["transfer_policy"], TRANSFER_POLICIES,
              "spec_editing_transfer_policy_invalid")
        if set(row["applies_to_state_ids"]) - state_ids \
                or set(row["applies_to_relation_ids"]) - relation_ids:
            _fail("spec_editing_anchor_unresolved", constraint_id)

    slot_ids: set[str] = set()
    for row in _list(spec["binding_slots"], "spec_binding_slots_invalid"):
        _exact_keys(_object(row, "spec_binding_slot_invalid"),
                    {"slot_id", "slot_type", "bound_by", "constraints"},
                    "spec_binding_slot_keys_invalid")
        slot_id = _id(row["slot_id"], "spec_binding_slot_id_invalid")
        if slot_id in slot_ids:
            _fail("spec_binding_slot_duplicate", slot_id)
        slot_ids.add(slot_id)
        _enum(row["slot_type"], BINDING_SLOT_TYPES,
              "spec_binding_slot_type_invalid")
        if row["bound_by"] != "downstream_skill":
            _fail("spec_binding_slot_authority_invalid", slot_id)
        _string_list(row["constraints"], "spec_binding_constraints_invalid",
                     nonempty=True)

    downstream = _object(spec["downstream_contract"], "downstream_contract_invalid")
    _exact_keys(downstream, {
        "required_constraint_ids", "variable_slot_ids", "authorized_stages",
        "reference_context_allowed",
    }, "downstream_contract_keys_invalid")
    mandatory = set(_string_list(downstream["required_constraint_ids"],
                                 "required_constraint_ids_invalid"))
    if mandatory - generation_ids - editing_ids:
        _fail("required_constraint_unresolved")
    if set(_string_list(downstream["variable_slot_ids"],
                        "variable_slot_ids_invalid")) - slot_ids:
        _fail("variable_slot_unresolved")
    stages = _string_list(downstream["authorized_stages"],
                          "authorized_stages_invalid")
    expected_stages = (["theme", "story", "screenplay"]
                       if dimensions["structural_schema"] == "supported"
                       and dimensions["generation_constraints"] == "supported"
                       else [])
    if stages != expected_stages:
        _fail("authorized_stages_invalid")
    if downstream["reference_context_allowed"] is not False:
        _fail("reference_context_allowed")

    forbidden_keys = {
        "support_refs", "source_refs", "audit_annex", "source_bindings",
        "anti_invariants", "input_artifacts", "source_strength", "confidence",
        "prompt", "model_trace", "reference_specific_summary",
    }
    for token in _walk(spec):
        if token in forbidden_keys:
            _fail("spec_forbidden_field", token)
        if isinstance(token, str) and _SOURCE_ID_RE.search(token):
            _fail("spec_source_id_leak", token)
        if isinstance(token, str) and _PATH_RE.search(token):
            _fail("spec_path_leak", token)
    _hash_matches(spec, "spec_sha_invalid")
