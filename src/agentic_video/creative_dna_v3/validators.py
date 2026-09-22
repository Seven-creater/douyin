"""Deterministic validation for Creative DNA v3 audit and publish artifacts."""
from __future__ import annotations

import re
from typing import Any, Iterable

from src.agentic_video.creative_dna_v3.schema import (
    ANTI_INVARIANT_CATEGORIES,
    AUDIT_SCHEMA_VERSION,
    AUDIT_TOP_LEVEL_KEYS,
    DNA_DIMENSIONS,
    DNA_STATUSES,
    EDITING_DIMENSIONS,
    EDITING_LEVELS,
    EDITING_OBLIGATIONS,
    EDITING_OPERATORS,
    EVENT_OBLIGATIONS,
    EXPERIENCE_STATE_ROLES,
    FREE_SLOT_KINDS,
    MECHANISMS,
    NODE_KINDS,
    SOURCE_STRENGTHS,
    SPEC_SCHEMA_VERSION,
    SPEC_TOP_LEVEL_KEYS,
    TRANSFER_POLICIES,
    VALIDATION_RECORD_KEYS,
)
from src.agentic_video.manifest import json_hash

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Z][A-Z0-9_-]*$")
_SOURCE_ID_RE = re.compile(r"(?<![A-Za-z0-9_])(?:N|R|EF|EP)\d+(?![A-Za-z0-9_])|S\d+_[A-Z0-9_]+")
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|(?:^|\s)/(?:[^\s/]+/)+|\.(?:mp4|mov|json|png|jpg)\b)", re.I)


class DNAV3Error(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _fail(reason: str, detail: object = "") -> None:
    raise DNAV3Error(reason, str(detail))


def _object(value: object, reason: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(reason)
    return value


def _list(value: object, reason: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(reason)
    return value


def _text(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(reason)
    return value


def _enum(value: object, allowed: Iterable[str], reason: str) -> str:
    text = _text(value, reason)
    if text not in allowed:
        _fail(reason, text)
    return text


def _confidence(value: object, reason: str = "confidence_invalid") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(reason)
    if not 0.0 <= float(value) <= 1.0:
        _fail(reason, value)


def _exact_keys(value: dict[str, Any], keys: Iterable[str], reason: str) -> None:
    expected = set(keys)
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        _fail(reason, f"missing={missing},extra={extra}")


def _id(value: object, reason: str) -> str:
    text = _text(value, reason)
    if not _ID_RE.fullmatch(text):
        _fail(reason, text)
    return text


def _string_list(value: object, reason: str, *, nonempty: bool = False) -> list[str]:
    rows = _list(value, reason)
    if nonempty and not rows:
        _fail(reason)
    if any(not isinstance(item, str) or not item.strip() for item in rows):
        _fail(reason)
    if len(rows) != len(set(rows)):
        _fail(reason, "duplicates")
    return rows


def _hash_matches(value: dict[str, Any], reason: str) -> None:
    claimed = value.get("artifact_sha")
    if not isinstance(claimed, str) or not _SHA_RE.fullmatch(claimed):
        _fail(reason, "missing_or_invalid")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha"}
    if claimed != json_hash(unhashed):
        _fail(reason, "mismatch")


def available_dimensions(audit: dict[str, Any]) -> list[str]:
    graph = audit.get("mechanism_graph")
    graph_available = isinstance(graph, dict) and bool(
        graph.get("nodes") or graph.get("edges")
    )
    editing = audit.get("editing_constraints") or []
    present = {
        "narrative_mechanism": graph_available,
        "experience_arc": bool(audit.get("experience_arc")),
        "event_constraints": bool(audit.get("event_constraints")),
        "editing_structure": any(
            isinstance(row, dict) and row.get("level") == "structural"
            for row in editing
        ),
        "editing_function": any(
            isinstance(row, dict) and row.get("level") in {
                "semantic_function", "semantic_pattern"
            }
            for row in editing
        ),
        "free_slots": bool(audit.get("free_slots")),
    }
    return [dimension for dimension in DNA_DIMENSIONS if present[dimension]]


def _validate_support_refs(row: dict[str, Any], known_refs: set[str], label: str) -> None:
    refs = _string_list(row.get("support_refs"), f"{label}_support_refs_invalid",
                        nonempty=True)
    unknown = sorted(set(refs) - known_refs)
    if unknown:
        _fail("support_ref_unresolved", f"{label}:{unknown}")


def validate_dna_audit(value: dict[str, Any]) -> None:
    audit = _object(value, "audit_not_object")
    _exact_keys(audit, AUDIT_TOP_LEVEL_KEYS, "audit_keys_invalid")
    if audit["schema_version"] != AUDIT_SCHEMA_VERSION:
        _fail("audit_schema_version_invalid", audit["schema_version"])
    status = _enum(audit["dna_status"], DNA_STATUSES, "dna_status_invalid")
    _confidence(audit["abstraction_confidence"], "abstraction_confidence_invalid")

    unsupported = _string_list(
        audit["unsupported_dimensions"], "unsupported_dimensions_invalid"
    )
    invalid_dimensions = sorted(set(unsupported) - set(DNA_DIMENSIONS))
    if invalid_dimensions:
        _fail("unsupported_dimensions_invalid", invalid_dimensions)

    artifacts = _list(audit["input_artifacts"], "input_artifacts_invalid")
    if not artifacts:
        _fail("input_artifacts_empty")
    artifact_ids: set[str] = set()
    for row_value in artifacts:
        row = _object(row_value, "input_artifact_invalid")
        _exact_keys(row, {"artifact_id", "artifact_type", "sha", "schema_version"},
                    "input_artifact_keys_invalid")
        artifact_id = _text(row["artifact_id"], "input_artifact_id_invalid")
        if artifact_id in artifact_ids:
            _fail("input_artifact_duplicate", artifact_id)
        artifact_ids.add(artifact_id)
        _text(row["artifact_type"], "input_artifact_type_invalid")
        _text(row["schema_version"], "input_artifact_schema_invalid")
        if not isinstance(row["sha"], str) or not _SHA_RE.fullmatch(row["sha"]):
            _fail("input_artifact_sha_invalid", artifact_id)

    bindings = _list(audit["source_bindings"], "source_bindings_invalid")
    known_refs: set[str] = set()
    binding_ids: set[str] = set()
    for value_row in bindings:
        row = _object(value_row, "source_binding_invalid")
        _exact_keys(row, {"abstract_id", "source_refs", "reference_specific_summary"},
                    "source_binding_keys_invalid")
        abstract_id = _id(row["abstract_id"], "source_binding_id_invalid")
        if abstract_id in binding_ids:
            _fail("source_binding_duplicate", abstract_id)
        binding_ids.add(abstract_id)
        known_refs.update(_string_list(row["source_refs"],
                                       "source_binding_refs_invalid", nonempty=True))
        _text(row["reference_specific_summary"], "source_binding_summary_invalid")

    graph_value = audit["mechanism_graph"]
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    if graph_value is not None:
        graph = _object(graph_value, "mechanism_graph_invalid")
        _exact_keys(graph, {"nodes", "edges"}, "mechanism_graph_keys_invalid")
        for value_row in _list(graph["nodes"], "mechanism_nodes_invalid"):
            row = _object(value_row, "mechanism_node_invalid")
            _exact_keys(row, {"node_id", "kind", "abstract_role", "support_refs"},
                        "mechanism_node_keys_invalid")
            node_id = _id(row["node_id"], "mechanism_node_id_invalid")
            if node_id in node_ids:
                _fail("mechanism_node_duplicate", node_id)
            node_ids.add(node_id)
            _enum(row["kind"], NODE_KINDS, "mechanism_node_kind_invalid")
            _text(row["abstract_role"], "mechanism_node_role_invalid")
            _validate_support_refs(row, known_refs, node_id)
        for value_row in _list(graph["edges"], "mechanism_edges_invalid"):
            row = _object(value_row, "mechanism_edge_invalid")
            _exact_keys(row, {"edge_id", "mechanism", "source", "target", "condition",
                              "support_refs", "confidence"},
                        "mechanism_edge_keys_invalid")
            edge_id = _id(row["edge_id"], "mechanism_edge_id_invalid")
            if edge_id in edge_ids:
                _fail("mechanism_edge_duplicate", edge_id)
            edge_ids.add(edge_id)
            _enum(row["mechanism"], MECHANISMS, "mechanism_invalid")
            if row["source"] not in node_ids or row["target"] not in node_ids:
                _fail("mechanism_edge_node_unresolved", edge_id)
            _text(row["condition"], "mechanism_edge_condition_invalid")
            _validate_support_refs(row, known_refs, edge_id)
            _confidence(row["confidence"])

    state_ids: set[str] = set()
    for value_row in _list(audit["experience_arc"], "experience_arc_invalid"):
        row = _object(value_row, "experience_state_invalid")
        _exact_keys(row, {"state_id", "order", "state_role", "information_state",
                          "caused_by_edge_ids", "support_refs"},
                    "experience_state_keys_invalid")
        state_id = _id(row["state_id"], "experience_state_id_invalid")
        if state_id in state_ids:
            _fail("experience_state_duplicate", state_id)
        state_ids.add(state_id)
        if isinstance(row["order"], bool) or not isinstance(row["order"], int) \
                or row["order"] < 1:
            _fail("experience_state_order_invalid", state_id)
        _enum(row["state_role"], EXPERIENCE_STATE_ROLES,
              "experience_state_role_invalid")
        _text(row["information_state"], "experience_information_state_invalid")
        caused_by = _string_list(row["caused_by_edge_ids"],
                                 "experience_edge_refs_invalid")
        if set(caused_by) - edge_ids:
            _fail("experience_edge_unresolved", state_id)
        _validate_support_refs(row, known_refs, state_id)

    event_ids: set[str] = set()
    event_rows = _list(audit["event_constraints"], "event_constraints_invalid")
    for value_row in event_rows:
        row = _object(value_row, "event_constraint_invalid")
        _exact_keys(row, {"constraint_id", "obligation", "abstract_role",
                          "satisfies_node_ids", "satisfies_edge_ids",
                          "must_precede_constraint_ids", "support_refs", "confidence"},
                    "event_constraint_keys_invalid")
        constraint_id = _id(row["constraint_id"], "event_constraint_id_invalid")
        if constraint_id in event_ids:
            _fail("event_constraint_duplicate", constraint_id)
        event_ids.add(constraint_id)
        _enum(row["obligation"], EVENT_OBLIGATIONS, "event_obligation_invalid")
        _text(row["abstract_role"], "event_role_invalid")
        node_refs = _string_list(row["satisfies_node_ids"], "event_node_refs_invalid")
        edge_refs = _string_list(row["satisfies_edge_ids"], "event_edge_refs_invalid")
        if set(node_refs) - node_ids or set(edge_refs) - edge_ids:
            _fail("event_mechanism_ref_unresolved", constraint_id)
        _string_list(row["must_precede_constraint_ids"], "event_order_refs_invalid")
        _validate_support_refs(row, known_refs, constraint_id)
        _confidence(row["confidence"])
    for row in event_rows:
        if set(row["must_precede_constraint_ids"]) - event_ids:
            _fail("event_order_ref_unresolved", row["constraint_id"])

    editing_ids: set[str] = set()
    for value_row in _list(audit["editing_constraints"],
                           "editing_constraints_invalid"):
        row = _object(value_row, "editing_constraint_invalid")
        _exact_keys(row, {"constraint_id", "level", "obligation", "dimension",
                          "operator", "phase_refs", "source_strength",
                          "transfer_policy", "support_refs", "confidence"},
                    "editing_constraint_keys_invalid")
        constraint_id = _id(row["constraint_id"], "editing_constraint_id_invalid")
        if constraint_id in editing_ids or constraint_id in event_ids:
            _fail("constraint_id_duplicate", constraint_id)
        editing_ids.add(constraint_id)
        _enum(row["level"], EDITING_LEVELS, "editing_level_invalid")
        obligation = _enum(row["obligation"], EDITING_OBLIGATIONS,
                           "editing_obligation_invalid")
        _enum(row["dimension"], EDITING_DIMENSIONS, "editing_dimension_invalid")
        _enum(row["operator"], EDITING_OPERATORS, "editing_operator_invalid")
        phases = _string_list(row["phase_refs"], "editing_phase_refs_invalid")
        if phases and set(phases) - state_ids:
            _fail("editing_phase_ref_unresolved", constraint_id)
        strength = _enum(row["source_strength"], SOURCE_STRENGTHS,
                         "editing_source_strength_invalid")
        policy = _enum(row["transfer_policy"], TRANSFER_POLICIES,
                       "editing_transfer_policy_invalid")
        if strength == "unaudited_semantic" and (
                obligation != "advisory" or policy != "free_realization"):
            _fail("unaudited_semantic_promoted", constraint_id)
        _validate_support_refs(row, known_refs, constraint_id)
        _confidence(row["confidence"])

    slot_ids: set[str] = set()
    for value_row in _list(audit["free_slots"], "free_slots_invalid"):
        row = _object(value_row, "free_slot_invalid")
        _exact_keys(row, {"slot_id", "kind", "constraints"},
                    "free_slot_keys_invalid")
        slot_id = _id(row["slot_id"], "free_slot_id_invalid")
        if slot_id in slot_ids:
            _fail("free_slot_duplicate", slot_id)
        slot_ids.add(slot_id)
        _enum(row["kind"], FREE_SLOT_KINDS, "free_slot_kind_invalid")
        _string_list(row["constraints"], "free_slot_constraints_invalid",
                     nonempty=True)

    anti_ids: set[str] = set()
    for value_row in _list(audit["anti_invariants"], "anti_invariants_invalid"):
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

    record = _object(audit["validation_record"], "validation_record_invalid")
    _exact_keys(record, VALIDATION_RECORD_KEYS, "validation_record_keys_invalid")
    if any(not isinstance(record[key], bool) for key in VALIDATION_RECORD_KEYS):
        _fail("validation_record_value_invalid")

    available = available_dimensions(audit)
    expected_unsupported = [d for d in DNA_DIMENSIONS if d not in available]
    if status == "complete":
        if unsupported:
            _fail("complete_has_unsupported_dimensions")
        if expected_unsupported:
            _fail("complete_missing_dimensions", expected_unsupported)
    elif status == "partial":
        if not unsupported:
            _fail("partial_missing_unsupported_dimensions")
        if not available:
            _fail("partial_has_no_supported_dimensions")
        if unsupported != expected_unsupported:
            _fail("partial_dimension_mismatch",
                  f"expected={expected_unsupported},actual={unsupported}")
    else:
        if available:
            _fail("blocked_has_supported_dimensions", available)
        if unsupported != list(DNA_DIMENSIONS):
            _fail("blocked_dimension_mismatch")

    _hash_matches(audit, "audit_sha_invalid")


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


def validate_creative_spec(value: dict[str, Any]) -> None:
    spec = _object(value, "creative_spec_not_object")
    _exact_keys(spec, SPEC_TOP_LEVEL_KEYS, "creative_spec_keys_invalid")
    if spec["schema_version"] != SPEC_SCHEMA_VERSION:
        _fail("creative_spec_schema_invalid")
    _text(spec["spec_id"], "creative_spec_id_invalid")
    status = _enum(spec["dna_status"], {"complete", "partial"},
                   "creative_spec_status_invalid")
    available = _string_list(spec["available_dimensions"],
                             "available_dimensions_invalid", nonempty=True)
    unsupported = _string_list(spec["unsupported_dimensions"],
                               "unsupported_dimensions_invalid")
    if any(item not in DNA_DIMENSIONS for item in available + unsupported):
        _fail("creative_spec_dimension_invalid")
    if available != [item for item in DNA_DIMENSIONS if item in available]:
        _fail("available_dimensions_order_invalid")
    if unsupported != [item for item in DNA_DIMENSIONS if item in unsupported]:
        _fail("unsupported_dimensions_order_invalid")
    if set(available).intersection(unsupported) or set(available + unsupported) != set(DNA_DIMENSIONS):
        _fail("creative_spec_dimension_partition_invalid")
    if status == "complete" and unsupported:
        _fail("creative_spec_complete_has_unsupported")
    if status == "partial" and not unsupported:
        _fail("creative_spec_partial_missing_unsupported")

    narrative = spec["narrative_control"]
    narrative_available = any(item in available for item in (
        "narrative_mechanism", "experience_arc", "event_constraints"
    ))
    if narrative_available != (narrative is not None):
        _fail("creative_spec_narrative_presence_invalid")
    if narrative is not None:
        _exact_keys(_object(narrative, "narrative_control_invalid"),
                    {"mechanism_graph", "experience_arc", "event_constraints"},
                    "narrative_control_keys_invalid")
    editing = _object(spec["editing_control"], "editing_control_invalid")
    _exact_keys(editing, {"constraints"}, "editing_control_keys_invalid")
    _list(editing["constraints"], "editing_control_constraints_invalid")
    _list(spec["free_slots"], "creative_spec_free_slots_invalid")
    downstream = _object(spec["downstream_contract"], "downstream_contract_invalid")
    _exact_keys(downstream, {"must_preserve_constraint_ids", "may_vary_slot_ids",
                             "reference_context_allowed"},
                "downstream_contract_keys_invalid")
    if downstream["reference_context_allowed"] is not False:
        _fail("reference_context_allowed")

    forbidden_keys = {
        "support_refs", "source_refs", "source_bindings", "anti_invariants",
        "input_artifacts", "source_strength", "confidence", "provenance",
        "prompt", "model_trace", "reference_specific_summary",
    }
    for token in _walk(spec):
        if token in forbidden_keys:
            _fail("creative_spec_forbidden_field", token)
        if isinstance(token, str) and _SOURCE_ID_RE.search(token):
            _fail("creative_spec_source_id_leak", token)
        if isinstance(token, str) and _PATH_RE.search(token):
            _fail("creative_spec_path_leak", token)
    _hash_matches(spec, "creative_spec_sha_invalid")
