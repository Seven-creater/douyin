"""Theme Candidate v1 fixtures and schema entry point."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_pipeline.contracts import (
    validate_theme_candidate,
)


THEME_GENERATE_N = 20
THEME_SELECT_K = 5

_SURFACE_BINDINGS = (
    ("system diagnosis", "reviewer", "diagnostic record", "operations room"),
    ("learning assessment", "assessor", "later work sample", "review session"),
    ("cooking process", "cook", "temperature record", "test kitchen"),
    ("archive research", "researcher", "newly indexed document", "archive"),
    ("ecological survey", "field observer", "repeat observation", "field site"),
    ("product quality", "inspector", "measurement result", "workshop"),
    ("music rehearsal", "section leader", "recorded performance", "studio"),
    ("museum attribution", "curator", "material analysis", "conservation lab"),
    ("logistics planning", "dispatcher", "route telemetry", "control room"),
    ("building maintenance", "technician", "sensor history", "service floor"),
    ("gardening diagnosis", "gardener", "soil reading", "greenhouse"),
    ("translation review", "editor", "source comparison", "editorial room"),
    ("inventory audit", "auditor", "transaction record", "warehouse"),
    ("weather analysis", "forecaster", "updated observation", "forecast desk"),
    ("software testing", "tester", "reproduction trace", "test lab"),
    ("map correction", "surveyor", "field coordinate", "mapping office"),
    ("restoration planning", "restorer", "layer scan", "restoration studio"),
    ("sports analysis", "analyst", "timing data", "analysis room"),
    ("event planning", "coordinator", "attendance update", "planning office"),
    ("equipment inspection", "operator", "maintenance log", "inspection bay"),
)


def _relation(statement: str) -> dict[str, str]:
    return {
        "prior": "I0_PRIOR_INTERPRETATION",
        "evidence": "E1_NEW_INFORMATION",
        "updated": "I1_UPDATED_INTERPRETATION",
        "statement": statement,
    }


def build_fake_theme_candidates(structure_sha: str) -> list[dict[str, Any]]:
    """Build 20 deterministic surface bindings; this is not creative scoring."""
    candidates = []
    for index, (domain, entity, evidence, setting) in enumerate(
            _SURFACE_BINDINGS, start=1):
        candidates.append({
            "schema_version": "theme_candidate_v1",
            "theme_id": f"FAKE_THEME_{index:02d}",
            "parent_structure_sha": structure_sha,
            "domain": domain,
            "premise": (
                f"In {domain}, new information changes an earlier "
                "interpretation."),
            "audience_promise": (
                "The meaning of an earlier judgment becomes clearer."),
            "tone": "observational",
            "binding_slots": {
                "B_DOMAIN": domain,
                "B_ENTITY_ROLE": entity,
                "B_EVIDENCE_FORM": evidence,
                "B_SETTING": setting,
            },
            "structure_bindings": {
                "I0_PRIOR_INTERPRETATION": (
                    f"the {entity} holds an initial interpretation"),
                "E1_NEW_INFORMATION": f"the {evidence} becomes available",
                "I1_UPDATED_INTERPRETATION": (
                    f"the {entity} revises the interpretation"),
                "R1_INFORMATION_UPDATE": _relation(
                    f"the {evidence} updates the initial interpretation"),
            },
        })
    if len(candidates) != THEME_GENERATE_N:
        raise AssertionError("theme fixture count drift")
    return candidates


def validate_theme_candidate_schema(value: dict[str, Any], *,
                                    structure_sha: str) -> None:
    validate_theme_candidate(value, structure_sha=structure_sha)
