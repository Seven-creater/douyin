"""Deterministic guard for the R2 narrative-to-mechanism boundary."""
from __future__ import annotations

import re
from typing import Any, Iterable

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{4,}")
_NARRATIVE_FUNCTION_KEYS = {
    "function_id", "source_unit_ids", "function_type", "abstract_meaning",
    "mechanism_node_ids",
}
_MECHANISM_CORE_KINDS = {
    "information_state", "evidence_role", "resolution_state",
}

# Generic abstraction vocabulary is allowed even when the R2 interpretation
# used the same words. Reference-specific vocabulary is derived at runtime.
_ABSTRACT_VOCABULARY = {
    "ability", "abilities", "abstract", "action", "available", "bears",
    "about", "after", "against", "before", "between", "from", "into",
    "capability", "capabilities", "changes", "constraint", "context",
    "develop", "earlier", "establish", "evidence", "information", "initial",
    "interpretation", "later", "observable", "observation", "presented",
    "proposition", "qualifies", "relation", "resolution", "reveal", "reveals",
    "scope", "state", "statement", "subject", "supports", "than", "that",
    "their", "then", "there", "these", "this", "those", "through", "what",
    "when", "where", "which", "while", "with", "without",
}


class AbstractionBoundaryError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _abstract_items(candidate: dict[str, Any]) -> list[tuple[str, str]]:
    graph = candidate.get("mechanism_graph") or {"nodes": [], "edges": []}
    return [
        *((str(row["node_id"]), "node") for row in graph["nodes"]),
        *((str(row["edge_id"]), "edge") for row in graph["edges"]),
        *((str(row["state_id"]), "experience_state")
          for row in candidate.get("experience_arc") or []),
        *((str(row["constraint_id"]), "event_constraint")
          for row in candidate.get("event_constraints") or []),
        *((str(row["constraint_id"]), "editing_constraint")
          for row in candidate.get("editing_constraints") or []),
        *((str(row["slot_id"]), "free_slot")
          for row in candidate.get("free_slots") or []),
    ]


def _public_text(candidate: dict[str, Any]) -> list[tuple[str, str, str]]:
    graph = candidate.get("mechanism_graph") or {"nodes": [], "edges": []}
    rows = [
        *((str(row["node_id"]), "abstract_role", str(row["abstract_role"]))
          for row in graph["nodes"]),
        *((str(row["edge_id"]), "condition", str(row["condition"]))
          for row in graph["edges"]),
        *((str(row["state_id"]), "information_state", str(row["information_state"]))
          for row in candidate.get("experience_arc") or []),
        *((str(row["constraint_id"]), "abstract_role", str(row["abstract_role"]))
          for row in candidate.get("event_constraints") or []),
    ]
    for slot in candidate.get("free_slots") or []:
        for index, constraint in enumerate(slot["constraints"]):
            rows.append((str(slot["slot_id"]), f"constraints[{index}]",
                         str(constraint)))
    return rows


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


def _has_mechanism_content(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(words) >= 3 or len(chinese_chars) >= 6


def _surface_hits(text: str, source_english: set[str],
                  source_chinese: set[str]) -> list[str]:
    public_english = {word.lower() for word in _WORD_RE.findall(text)}
    overlap = sorted(public_english.intersection(source_english))
    overlap.extend(sorted(phrase for phrase in source_chinese if phrase in text))
    return overlap


def _string_list(value: Any, reason: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value):
        raise AbstractionBoundaryError(reason)
    if nonempty and not value:
        raise AbstractionBoundaryError(reason)
    return value


def validate_narrative_function_bridge(functions: Any,
                                       candidate: dict[str, Any],
                                       bundle: dict[str, Any]) -> None:
    """Validate the audit-only unit -> function -> mechanism mapping."""
    if not isinstance(functions, list):
        raise AbstractionBoundaryError("narrative_functions_invalid")
    graph = candidate.get("mechanism_graph") or {"nodes": []}
    nodes = {
        str(row["node_id"]): row for row in graph.get("nodes") or []
    }
    node_ids = set(nodes)
    abstract_ids = {
        item_id for item_id, _ in _abstract_items(candidate)
    }
    if not node_ids:
        if functions:
            raise AbstractionBoundaryError(
                "narrative_functions_without_mechanism")
        return
    if not functions:
        raise AbstractionBoundaryError("mechanism_without_narrative_functions")

    narrative = bundle.get("narrative_interpretation") or {}
    unit_ids = {
        str(row.get("unit_id"))
        for row in narrative.get("narrative_units") or []
    }
    source_ids = {
        str(row.get("evidence_id"))
        for row in bundle.get("evidence_catalog") or []
    }
    source_english, source_chinese = _surface_vocabulary(bundle)
    function_ids: set[str] = set()
    covered_nodes: set[str] = set()
    for value in functions:
        if not isinstance(value, dict) or set(value) != _NARRATIVE_FUNCTION_KEYS:
            raise AbstractionBoundaryError("narrative_function_shape_invalid")
        function_id = str(value.get("function_id") or "").strip()
        if not function_id or function_id in function_ids:
            raise AbstractionBoundaryError(
                "narrative_function_id_invalid", function_id)
        if function_id in source_ids:
            raise AbstractionBoundaryError(
                "narrative_function_reuses_source_id", function_id)
        if function_id in abstract_ids:
            raise AbstractionBoundaryError(
                "narrative_function_id_collision", function_id)
        function_ids.add(function_id)

        source_units = _string_list(
            value.get("source_unit_ids"),
            "narrative_function_source_invalid", nonempty=True)
        unknown_units = sorted(set(source_units) - unit_ids)
        if unknown_units:
            raise AbstractionBoundaryError(
                "narrative_function_source_unresolved",
                f"{function_id}:{','.join(unknown_units)}")
        mapped_nodes = _string_list(
            value.get("mechanism_node_ids"),
            "narrative_function_node_invalid", nonempty=True)
        unknown_nodes = sorted(set(mapped_nodes) - node_ids)
        if unknown_nodes:
            raise AbstractionBoundaryError(
                "narrative_function_node_unresolved",
                f"{function_id}:{','.join(unknown_nodes)}")
        for node_id in mapped_nodes:
            node_support = set(nodes[node_id].get("support_refs") or [])
            if not node_support.intersection(source_units):
                raise AbstractionBoundaryError(
                    "narrative_function_mapping_ungrounded",
                    f"{function_id}:{node_id}")
        covered_nodes.update(mapped_nodes)

        function_type = str(value.get("function_type") or "").strip()
        abstract_meaning = str(value.get("abstract_meaning") or "").strip()
        if not function_type:
            raise AbstractionBoundaryError(
                "narrative_function_not_abstract",
                f"{function_id}.function_type")
        if not _has_mechanism_content(abstract_meaning):
            raise AbstractionBoundaryError(
                "narrative_function_not_abstract",
                f"{function_id}.abstract_meaning")
        for field, text in (("function_type", function_type),
                            ("abstract_meaning", abstract_meaning)):
            if not text:
                raise AbstractionBoundaryError(
                    "narrative_function_not_abstract",
                    f"{function_id}.{field}")
            overlap = _surface_hits(text, source_english, source_chinese)
            if overlap:
                raise AbstractionBoundaryError(
                    "reference_surface_in_narrative_function",
                    f"{function_id}.{field}:{','.join(overlap)}")

    uncovered = sorted(node_ids - covered_nodes)
    if uncovered:
        raise AbstractionBoundaryError(
            "mechanism_node_without_narrative_function", ",".join(uncovered))


def validate_abstraction_boundary(candidate: dict[str, Any],
                                  bundle: dict[str, Any]) -> None:
    """Reject evidence passthrough without prescribing a narrative mechanism."""
    source_ids = {
        str(row.get("evidence_id"))
        for row in bundle.get("evidence_catalog") or []
    }
    items = _abstract_items(candidate)
    abstract_ids = [item_id for item_id, _ in items]
    if len(abstract_ids) != len(set(abstract_ids)):
        raise AbstractionBoundaryError("abstract_id_collision")
    reused = sorted(set(abstract_ids).intersection(source_ids))
    if reused:
        raise AbstractionBoundaryError("abstract_id_reuses_source_id", ",".join(reused))

    abstract_id_set = set(abstract_ids)
    unknown_bindings = sorted({
        str(row.get("abstract_id")) for row in candidate.get("source_bindings") or []
        if str(row.get("abstract_id")) not in abstract_id_set
    })
    if unknown_bindings:
        raise AbstractionBoundaryError(
            "source_binding_abstract_id_unresolved", ",".join(unknown_bindings))

    graph = candidate.get("mechanism_graph") or {"nodes": []}
    for node in graph["nodes"]:
        role = str(node.get("abstract_role") or "")
        if not _has_mechanism_content(role):
            raise AbstractionBoundaryError(
                "abstract_role_not_mechanism", str(node.get("node_id")))
    if graph["nodes"] and not any(
            str(node.get("kind")) in _MECHANISM_CORE_KINDS
            for node in graph["nodes"]):
        raise AbstractionBoundaryError("mechanism_core_kind_missing")

    for constraint in candidate.get("event_constraints") or []:
        if not constraint.get("satisfies_edge_ids"):
            raise AbstractionBoundaryError(
                "event_constraint_without_mechanism_relation",
                str(constraint.get("constraint_id")))
        if not _has_mechanism_content(str(constraint.get("abstract_role") or "")):
            raise AbstractionBoundaryError(
                "event_constraint_not_generative",
                str(constraint.get("constraint_id")))

    experience = sorted(
        candidate.get("experience_arc") or [], key=lambda row: row.get("order", 0))
    if [row.get("order") for row in experience] != list(
            range(1, len(experience) + 1)):
        raise AbstractionBoundaryError("experience_order_invalid")
    for state in experience[1:]:
        if not state.get("caused_by_edge_ids"):
            raise AbstractionBoundaryError(
                "experience_update_without_mechanism_edge",
                str(state.get("state_id")))

    source_english, source_chinese = _surface_vocabulary(bundle)
    for item_id, field, text in _public_text(candidate):
        overlap = _surface_hits(text, source_english, source_chinese)
        if overlap:
            detail = f"{item_id}.{field}:{','.join(overlap)}"
            raise AbstractionBoundaryError(
                "reference_surface_in_public_field", detail)
