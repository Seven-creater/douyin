"""Whitelist-only publisher from reference-bound DNA audit to Creative Spec."""
from __future__ import annotations

import re
from typing import Any

from src.agentic_video.creative_dna_v3.schema import (
    DNA_DIMENSIONS,
    SPEC_SCHEMA_VERSION,
    VALIDATION_RECORD_KEYS,
)
from src.agentic_video.creative_dna_v3.validators import (
    DNAV3Error,
    available_dimensions,
    validate_creative_spec,
    validate_dna_audit,
)
from src.agentic_video.manifest import json_hash

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{4,}|[\u4e00-\u9fff]{2,}")
_SURFACE_STOPWORDS = {
    "about", "after", "again", "earlier", "example", "information", "later",
    "provides", "sequence", "source", "state", "story", "their", "there",
    "these", "using", "video", "which", "作品", "信息", "参考", "来源", "示例",
}


def _public_graph(graph: dict[str, Any] | None) -> dict[str, Any]:
    graph = graph or {"nodes": [], "edges": []}
    return {
        "nodes": [
            {key: row[key] for key in ("node_id", "kind", "abstract_role")}
            for row in graph["nodes"]
        ],
        "edges": [
            {key: row[key] for key in (
                "edge_id", "mechanism", "source", "target", "condition"
            )}
            for row in graph["edges"]
        ],
    }


def _public_experience(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "state_id", "order", "state_role", "information_state",
            "caused_by_edge_ids",
        )}
        for row in rows
    ]


def _public_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "constraint_id", "obligation", "abstract_role", "satisfies_node_ids",
            "satisfies_edge_ids", "must_precede_constraint_ids",
        )}
        for row in rows
    ]


def _public_editing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "constraint_id", "obligation", "dimension", "operator", "phase_refs",
            "transfer_policy",
        )}
        for row in rows
    ]


def _public_slots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in ("slot_id", "kind", "constraints")}
        for row in rows
    ]


def _surface_terms(audit: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for binding in audit["source_bindings"]:
        summary = binding["reference_specific_summary"]
        for match in _WORD_RE.findall(summary):
            lowered = match.lower()
            if lowered not in _SURFACE_STOPWORDS:
                terms.add(lowered)
    return terms


def _assert_no_reference_surface(spec: dict[str, Any], audit: dict[str, Any]) -> None:
    text = repr(spec).lower()
    leaked = sorted(term for term in _surface_terms(audit) if term in text)
    if leaked:
        raise DNAV3Error("reference_surface_leak", ",".join(leaked))


def publish_creative_spec(audit: dict[str, Any], *, spec_id: str) -> dict[str, Any]:
    """Construct the only downstream-visible reference derivative.

    The function never rewrites source-bound fields. It selects an explicit
    allowlist, validates the result, and refuses blocked audits.
    """
    validate_dna_audit(audit)
    if audit["dna_status"] == "blocked":
        raise DNAV3Error("dna_blocked")
    if not spec_id.strip():
        raise DNAV3Error("creative_spec_id_invalid")
    failed = [key for key in VALIDATION_RECORD_KEYS
              if not audit["validation_record"][key]]
    if failed:
        raise DNAV3Error("audit_validation_incomplete", ",".join(sorted(failed)))

    available = available_dimensions(audit)
    unsupported = [item for item in DNA_DIMENSIONS if item not in available]
    has_narrative = any(item in available for item in (
        "narrative_mechanism", "experience_arc", "event_constraints"
    ))
    events = _public_events(audit["event_constraints"])
    editing = _public_editing(audit["editing_constraints"])
    slots = _public_slots(audit["free_slots"])
    spec: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "spec_id": spec_id,
        "dna_status": audit["dna_status"],
        "available_dimensions": available,
        "unsupported_dimensions": unsupported,
        "narrative_control": ({
            "mechanism_graph": _public_graph(audit["mechanism_graph"]),
            "experience_arc": _public_experience(audit["experience_arc"]),
            "event_constraints": events,
        } if has_narrative else None),
        "editing_control": {"constraints": editing},
        "free_slots": slots,
        "downstream_contract": {
            "must_preserve_constraint_ids": [
                row["constraint_id"] for row in events + editing
                if row["obligation"] in {"required", "prohibited"}
            ],
            "may_vary_slot_ids": [row["slot_id"] for row in slots],
            "reference_context_allowed": False,
        },
    }
    _assert_no_reference_surface(spec, audit)
    spec["artifact_sha"] = json_hash(spec)
    validate_creative_spec(spec)
    return spec
