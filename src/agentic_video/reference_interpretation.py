"""One-pass narrative relation inference for the P0-R2 reference track."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash

INTERPRETATION_VERSION = "reference_interpretation_v1"
RELATION_AUDIT_VERSION = "reference_relation_audit_v1"
RELATION_TYPES = {
    "supports", "contradicts", "qualifies", "reframes", "causes",
    "enables", "precedes", "contrasts", "accumulates", "reveals",
    "contextualizes",
}

INTERPRETATION_PROMPT = """You are an evidence-grounded relation analyst.
Read only the supplied ShotCards, accepted atomic claims, and accepted events.
Identify propositions that are supported by supplied IDs, then record only the
relations that actually follow. It is valid to return an empty relations list.
Do not assign preset narrative roles. Do not infer motives, audience reactions,
unseen activity, anatomy, or facts absent from the input.

Allowed relation types:
supports, contradicts, qualifies, reframes, causes, enables, precedes,
contrasts, accumulates, reveals, contextualizes.

Return exactly one JSON object:
{"schema_version":"reference_interpretation_v1",
"propositions":[{"proposition_id":"P1","statement":"...",
"supported_by":["claim or event ID"]}],
"relations":[{"relation_id":"R1","type":"supports","source":"P1",
"target":"P2","reason":"...","supported_by":["claim or event ID"]}],
"limitations":["..."]}

Every support ID must exist in the input. A relation may not introduce an
unstated proposition in its reason. JSON only. Input:
"""

RELATION_AUDIT_PROMPT = """You are an independent relation auditor. Do not
rewrite the candidate. For each supplied relation, check whether its source and
target propositions are entailed by their cited evidence, whether the named
relation holds, and whether proposition scope is preserved. When no relations
are supplied, return checks=[] and pass=true.

Return exactly one JSON object:
{"schema_version":"reference_relation_audit_v1",
"checks":[{"relation_id":"R1","source_entailed":true,
"target_entailed":true,"relation_valid":true,"scope_valid":true,
"reason_codes":[]}],"pass":true}

Include every relation ID exactly once. Set pass=true only when every boolean
in every check is true. JSON only. Input:
"""


class ReferenceInterpretationError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _answer_text(answer: Any) -> str:
    return str(getattr(answer, "text", answer))


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
        raise ReferenceInterpretationError("response_json_invalid") from exc
    if not isinstance(value, dict):
        raise ReferenceInterpretationError("response_not_object")
    return value


def _source_ids(agent_reference: dict[str, Any]) -> set[str]:
    return {
        str(row.get("claim_id"))
        for row in agent_reference.get("claims") or []
        if row.get("claim_id")
    } | {
        str(row.get("event_id"))
        for row in agent_reference.get("events") or []
        if row.get("event_id")
    }


def build_interpretation_payload(agent_reference: dict[str, Any]) -> dict[str, Any]:
    claim_fields = (
        "claim_id", "subject", "predicate", "object", "interval", "modality",
        "polarity", "visibility",
    )
    claims = [
        {key: row.get(key) for key in claim_fields if key in row}
        for row in agent_reference.get("claims") or []
    ]
    return {
        "reference_sha": agent_reference.get("artifact_sha"),
        "shot_cards": list(
            (agent_reference.get("shot_storyboard") or {}).get("shot_cards") or []),
        "claims": claims,
        "events": list(agent_reference.get("events") or []),
        "unresolved_limitations": list(
            agent_reference.get("unresolved_limitations") or []),
    }


def validate_interpretation(value: dict[str, Any],
                            agent_reference: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != INTERPRETATION_VERSION:
        raise ReferenceInterpretationError("interpretation_schema_invalid")
    propositions = value.get("propositions")
    relations = value.get("relations")
    if not isinstance(propositions, list) or not propositions:
        raise ReferenceInterpretationError("propositions_missing")
    if not isinstance(relations, list):
        raise ReferenceInterpretationError("relations_invalid")
    if not isinstance(value.get("limitations"), list):
        raise ReferenceInterpretationError("limitations_invalid")

    allowed_sources = _source_ids(agent_reference)
    proposition_ids: set[str] = set()
    for proposition in propositions:
        proposition_id = str(proposition.get("proposition_id") or "")
        if not proposition_id or proposition_id in proposition_ids:
            raise ReferenceInterpretationError(
                "proposition_id_invalid", proposition_id)
        proposition_ids.add(proposition_id)
        if not str(proposition.get("statement") or "").strip():
            raise ReferenceInterpretationError(
                "proposition_statement_missing", proposition_id)
        support = proposition.get("supported_by")
        if not isinstance(support, list) or not support:
            raise ReferenceInterpretationError(
                "proposition_support_missing", proposition_id)
        if not set(map(str, support)).issubset(allowed_sources):
            raise ReferenceInterpretationError(
                "proposition_support_invalid", proposition_id)

    relation_ids: set[str] = set()
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        if not relation_id or relation_id in relation_ids:
            raise ReferenceInterpretationError("relation_id_invalid", relation_id)
        relation_ids.add(relation_id)
        if relation.get("type") not in RELATION_TYPES:
            raise ReferenceInterpretationError("relation_type_invalid", relation_id)
        if str(relation.get("source") or "") not in proposition_ids:
            raise ReferenceInterpretationError("relation_source_invalid", relation_id)
        if str(relation.get("target") or "") not in proposition_ids:
            raise ReferenceInterpretationError("relation_target_invalid", relation_id)
        if not str(relation.get("reason") or "").strip():
            raise ReferenceInterpretationError("relation_reason_missing", relation_id)
        support = relation.get("supported_by")
        if not isinstance(support, list) or not support:
            raise ReferenceInterpretationError("relation_support_missing", relation_id)
        if not set(map(str, support)).issubset(allowed_sources):
            raise ReferenceInterpretationError("relation_support_invalid", relation_id)
    return value


def validate_relation_audit(audit: dict[str, Any],
                            interpretation: dict[str, Any]) -> dict[str, Any]:
    if audit.get("schema_version") != RELATION_AUDIT_VERSION:
        raise ReferenceInterpretationError("relation_audit_schema_invalid")
    checks = audit.get("checks")
    if not isinstance(checks, list) or not isinstance(audit.get("pass"), bool):
        raise ReferenceInterpretationError("relation_audit_shape_invalid")
    expected = {
        str(row["relation_id"]) for row in interpretation.get("relations") or []
    }
    actual: set[str] = set()
    all_valid = True
    for check in checks:
        relation_id = str(check.get("relation_id") or "")
        if not relation_id or relation_id in actual:
            raise ReferenceInterpretationError(
                "relation_audit_id_invalid", relation_id)
        actual.add(relation_id)
        for key in (
            "source_entailed", "target_entailed", "relation_valid", "scope_valid",
        ):
            if not isinstance(check.get(key), bool):
                raise ReferenceInterpretationError(
                    "relation_audit_boolean_invalid", f"{relation_id}:{key}")
            all_valid = all_valid and bool(check[key])
        if not isinstance(check.get("reason_codes"), list):
            raise ReferenceInterpretationError(
                "relation_audit_reason_codes_invalid", relation_id)
    if actual != expected:
        raise ReferenceInterpretationError("relation_audit_coverage_invalid")
    if bool(audit["pass"]) != all_valid:
        raise ReferenceInterpretationError("relation_audit_pass_invalid")
    return audit


def finalize_interpretation(interpretation: dict[str, Any],
                            audit: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(interpretation)
    checks = {
        str(row["relation_id"]): row for row in audit.get("checks") or []
    }
    for relation in result.get("relations") or []:
        check = checks[str(relation["relation_id"])]
        relation["verification_status"] = (
            "SUPPORTED" if all(bool(check[key]) for key in (
                "source_entailed", "target_entailed", "relation_valid", "scope_valid",
            )) else "UNRESOLVED"
        )
    result["verification_status"] = (
        "SUPPORTED" if audit.get("pass") else "UNRESOLVED")
    result["relation_audit_sha"] = json_hash(audit)
    result["artifact_sha"] = json_hash(result)
    return result


def _write_trace(trace_dir: Path | None, name: str, content: str) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / name).write_text(content, encoding="utf-8")


def run_reference_interpretation(
        interpreter_runner: Any, audit_runner: Any,
        agent_reference: dict[str, Any], *,
        trace_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate once, audit once, and never retry unchanged evidence."""
    payload = build_interpretation_payload(agent_reference)
    request = INTERPRETATION_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    _write_trace(trace_dir, "interpretation_request.txt", request)
    answer = interpreter_runner.ask(
        request, max_new_tokens=4096, stop_after_json_object=True)
    raw = _answer_text(answer)
    _write_trace(trace_dir, "interpretation_response.txt", raw)
    interpretation = _parse_object(raw)
    validate_interpretation(interpretation, agent_reference)

    audit_payload = {
        "evidence": payload,
        "candidate_interpretation": interpretation,
    }
    audit_request = RELATION_AUDIT_PROMPT + json.dumps(
        audit_payload, ensure_ascii=False, separators=(",", ":"))
    _write_trace(trace_dir, "relation_audit_request.txt", audit_request)
    audit_answer = audit_runner.ask(
        audit_request, max_new_tokens=3072, stop_after_json_object=True)
    audit_raw = _answer_text(audit_answer)
    _write_trace(trace_dir, "relation_audit_response.txt", audit_raw)
    audit = _parse_object(audit_raw)
    validate_relation_audit(audit, interpretation)
    result = finalize_interpretation(interpretation, audit)
    result["agent_reference_sha"] = agent_reference.get("artifact_sha")
    result.pop("artifact_sha", None)
    result["artifact_sha"] = json_hash(result)
    audit["agent_reference_sha"] = agent_reference.get("artifact_sha")
    audit["interpretation_sha"] = result["artifact_sha"]
    audit["artifact_sha"] = json_hash(audit)
    return result, audit


def run_reference_interpretation_checkpointed(
        interpreter_runner: Any, audit_runner: Any,
        agent_reference: dict[str, Any], *, checkpoint_dir: Path,
        trace_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse semantic output until its evidence artifact SHA changes."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    interpretation_path = checkpoint_dir / "reference_interpretation.json"
    audit_path = checkpoint_dir / "reference_relation_audit.json"
    existing = (interpretation_path.exists(), audit_path.exists())
    if any(existing) and not all(existing):
        raise ReferenceInterpretationError("interpretation_checkpoint_incomplete")
    if all(existing):
        try:
            interpretation = json.loads(
                interpretation_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceInterpretationError(
                "interpretation_checkpoint_invalid") from exc
        evidence_sha = agent_reference.get("artifact_sha")
        if (interpretation.get("agent_reference_sha") == evidence_sha and
                audit.get("agent_reference_sha") == evidence_sha):
            validate_interpretation(interpretation, agent_reference)
            validate_relation_audit(audit, interpretation)
            return interpretation, audit

    interpretation, audit = run_reference_interpretation(
        interpreter_runner, audit_runner, agent_reference, trace_dir=trace_dir)
    interpretation_path.write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return interpretation, audit
