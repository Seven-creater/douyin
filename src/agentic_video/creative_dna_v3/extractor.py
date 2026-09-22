"""One-shot structural abstraction from verified R2 artifacts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.agentic_video.creative_dna_v3.abstraction_prompt import (
    EXTRACTION_MAX_NEW_TOKENS,
    STRUCTURAL_ABSTRACTION_PROMPT,
)
from src.agentic_video.creative_dna_v3.abstraction_validator import (
    validate_abstraction_boundary,
    validate_narrative_function_bridge,
)
from src.agentic_video.creative_dna_v3.schema import (
    AUDIT_SCHEMA_VERSION,
    VALIDATION_RECORD_KEYS,
)
from src.agentic_video.creative_dna_v3.validators import (
    DNAV3Error,
    validate_dna_audit,
)
from src.agentic_video.manifest import json_hash

_DRAFT_KEYS = {
    "dna_status", "abstraction_confidence", "unsupported_dimensions",
    "narrative_functions",
    "mechanism_graph", "experience_arc", "event_constraints",
    "editing_constraints", "free_slots", "source_bindings", "anti_invariants",
}


class DNAExtractionError(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _parse_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DNAExtractionError("extraction_response_json_invalid") from exc
    if not isinstance(value, dict):
        raise DNAExtractionError("extraction_response_not_object")
    return value


def _artifact_hash_valid(value: dict[str, Any]) -> bool:
    claimed = value.get("artifact_sha")
    payload = {key: item for key, item in value.items() if key != "artifact_sha"}
    return isinstance(claimed, str) and claimed == json_hash(payload)


def _input_artifacts(narrative: dict[str, Any], editing: dict[str, Any]
                     ) -> list[dict[str, str]]:
    return [
        {
            "artifact_id": "reference_narrative",
            "artifact_type": "narrative_interpretation",
            "sha": str(narrative["artifact_sha"]),
            "schema_version": str(narrative["schema_version"]),
        },
        {
            "artifact_id": "editing_grammar",
            "artifact_type": "editing_grammar",
            "sha": str(editing["artifact_sha"]),
            "schema_version": str(editing["schema_version"]),
        },
    ]


def _editing_evidence(structural: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = (
        ("PACE", "pace_profile"),
        ("PACE_CHANGE", "pace_changes"),
        ("DURATION", "duration_profile"),
        ("DURATION_OUTLIER", "shot_duration_outliers"),
        ("TRANSITION", "transition_profile"),
        ("TRANSITION_SEQUENCE", "transition_sequences"),
    )
    for prefix, key in groups:
        for index, fact in enumerate(structural.get(key) or [], start=1):
            evidence_id = str(fact.get("fact_id") or f"{prefix}_{index:02d}")
            rows.append({
                "evidence_id": evidence_id,
                "kind": key,
                "source_strength": "measured",
                "fact": copy.deepcopy(fact),
            })
    return rows


def build_r2_bundle(narrative: dict[str, Any], editing: dict[str, Any]
                    ) -> dict[str, Any]:
    """Build the only extraction view; raw claims and identity data are absent."""
    if not _artifact_hash_valid(narrative):
        raise DNAExtractionError("narrative_artifact_sha_invalid")
    if not _artifact_hash_valid(editing):
        raise DNAExtractionError("editing_artifact_sha_invalid")
    if narrative.get("verification_status") != "SUPPORTED":
        raise DNAExtractionError("narrative_not_verified")
    units = narrative.get("narrative_units") or []
    relations = narrative.get("relations") or []
    if any(row.get("verification_status") != "SUPPORTED" for row in units):
        raise DNAExtractionError("narrative_unit_not_verified")
    if any(row.get("verification_status") != "SUPPORTED" for row in relations):
        raise DNAExtractionError("narrative_relation_not_verified")

    structural = editing.get("structural_grammar") or {}
    semantic_patterns = []
    for pattern in editing.get("semantic_patterns") or []:
        semantic_patterns.append({
            **copy.deepcopy(pattern),
            "source_strength": "unaudited_semantic",
        })
    functions = []
    for function in editing.get("functions") or []:
        functions.append({
            **copy.deepcopy(function),
            "evidence_id": f"FUNCTION_{function.get('pattern_id')}",
            "source_strength": "unaudited_semantic",
        })
    bundle = {
        "schema_version": "r2_verified_bundle_for_dna_v1",
        "input_artifacts": _input_artifacts(narrative, editing),
        "narrative_interpretation": {
            "narrative_units": copy.deepcopy(units),
            "relations": copy.deepcopy(relations),
            "limitations": copy.deepcopy(narrative.get("limitations") or []),
        },
        "editing_grammar": {
            "measured_evidence": _editing_evidence(structural),
            "semantic_patterns": semantic_patterns,
            "functions": functions,
            "limitations": {
                "patterns": copy.deepcopy(editing.get("pattern_limitations") or []),
                "functions": copy.deepcopy(editing.get("function_limitations") or []),
            },
        },
    }
    catalog = []
    for row in units:
        catalog.append({"evidence_id": row["unit_id"], "kind": "narrative_unit",
                        "source_strength": "audited_semantic"})
    for row in relations:
        catalog.append({"evidence_id": row["relation_id"],
                        "kind": "narrative_relation",
                        "source_strength": "audited_semantic"})
    for row in bundle["editing_grammar"]["measured_evidence"]:
        catalog.append({key: row[key] for key in (
            "evidence_id", "kind", "source_strength")})
    for row in semantic_patterns:
        catalog.append({"evidence_id": row["pattern_id"],
                        "kind": "semantic_editing_pattern",
                        "source_strength": "unaudited_semantic"})
    for row in functions:
        catalog.append({"evidence_id": row["evidence_id"],
                        "kind": "editing_function",
                        "source_strength": "unaudited_semantic"})
    bundle["evidence_catalog"] = catalog
    bundle["artifact_sha"] = json_hash(bundle)
    return bundle


def _model_metadata(answer: Any) -> dict[str, Any]:
    return {
        "input_tokens": getattr(answer, "input_tokens", None),
        "output_tokens": getattr(answer, "output_tokens", None),
        "elapsed_s": getattr(answer, "elapsed_s", None),
    }


def extract_creative_dna(runner: Any, narrative: dict[str, Any],
                         editing: dict[str, Any], *, trace_dir: Path
                         ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Call the abstraction model exactly once and validate without repair."""
    root = Path(trace_dir)
    stage = root / "01_extraction"
    stage.mkdir(parents=True, exist_ok=True)
    bundle = build_r2_bundle(narrative, editing)
    payload = {"r2_bundle": bundle}
    prompt = STRUCTURAL_ABSTRACTION_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    request = {
        "stage": "structural_abstraction",
        "attempt": 1,
        "max_new_tokens": EXTRACTION_MAX_NEW_TOKENS,
        "stop_after_json_object": True,
        "prompt": prompt,
        "payload": payload,
    }
    _write_json(stage / "request.json", request)
    (stage / "prompt.txt").write_text(prompt, encoding="utf-8")
    answer = runner.ask(prompt, max_new_tokens=EXTRACTION_MAX_NEW_TOKENS,
                        stop_after_json_object=True)
    raw = str(getattr(answer, "text", answer))
    (stage / "raw_response.txt").write_text(raw, encoding="utf-8")
    try:
        draft = _parse_object(raw)
        if set(draft) != _DRAFT_KEYS:
            raise DNAExtractionError("extraction_response_keys_invalid")
        narrative_functions = draft.pop("narrative_functions")
        _write_json(stage / "narrative_functions.json", narrative_functions)
        candidate = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            **copy.deepcopy(draft),
            "input_artifacts": _input_artifacts(narrative, editing),
            "validation_record": {key: False for key in VALIDATION_RECORD_KEYS},
        }
        candidate["artifact_sha"] = json_hash(candidate)
        validate_dna_audit(candidate)
        validate_narrative_function_bridge(
            narrative_functions, candidate, bundle)
        validate_abstraction_boundary(candidate, bundle)
        known = {row["evidence_id"] for row in bundle["evidence_catalog"]}
        used = {
            ref for binding in candidate["source_bindings"]
            for ref in binding["source_refs"]
        }
        unknown = sorted(used - known)
        if unknown:
            raise DNAV3Error("candidate_source_ref_not_in_r2_bundle", str(unknown))
    except BaseException as exc:
        _write_json(stage / "validation.json", {
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc), "model_call": _model_metadata(answer),
        })
        raise
    _write_json(stage / "candidate.json", candidate)
    metadata = _model_metadata(answer)
    _write_json(stage / "validation.json", {
        "status": "PASS", "candidate_sha": candidate["artifact_sha"],
        "r2_bundle_sha": bundle["artifact_sha"], "model_call": metadata,
    })
    return candidate, bundle, metadata
