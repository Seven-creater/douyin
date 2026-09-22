"""Deterministic guard for the R2 narrative-to-mechanism boundary."""
from __future__ import annotations

import re
from typing import Any, Iterable

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{4,}")

# Generic abstraction vocabulary is allowed even when the R2 interpretation
# used the same words. Reference-specific vocabulary is derived at runtime.
_ABSTRACT_VOCABULARY = {
    "ability", "abilities", "abstract", "action", "available", "bears",
    "capability", "capabilities", "changes", "constraint", "context",
    "develop", "earlier", "establish", "evidence", "information", "initial",
    "interpretation", "later", "observable", "observation", "presented",
    "proposition", "qualifies", "relation", "resolution", "reveal", "reveals",
    "scope", "state", "statement", "subject", "supports",
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

    source_english, source_chinese = _surface_vocabulary(bundle)
    for item_id, field, text in _public_text(candidate):
        public_english = {word.lower() for word in _WORD_RE.findall(text)}
        overlap = sorted(public_english.intersection(source_english))
        chinese_overlap = sorted(
            phrase for phrase in source_chinese if phrase in text
        )
        if overlap or chinese_overlap:
            detail = f"{item_id}.{field}:{','.join(overlap + chinese_overlap)}"
            raise AbstractionBoundaryError(
                "reference_surface_in_public_field", detail)
