"""Atomic reference claims, event drafts, conflict checks and validation gate."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file

CLAIM_LEDGER_VERSION = "claim_ledger_v1"
EVENT_DRAFT_VERSION = "event_draft_v1"
VALIDATED_REFERENCE_VERSION = "validated_reference_v1"

EPISTEMIC_STATUSES = {"SUPPORTED", "REFUTED", "CONTESTED", "INSUFFICIENT"}
RECORD_STATUSES = {"ACTIVE", "SUPERSEDED"}
MODALITIES = {"V", "A", "T", "FUSION", "LEGACY_MIXED"}
POLARITIES = {"POSITIVE", "NEGATIVE"}
VISIBILITIES = {"VISIBLE", "PARTIAL", "OCCLUDED", "OFFSCREEN", "UNKNOWN"}

# Deliberately small v1 vocabulary. Unknown observations remain preserved but
# cannot enter validated_reference until a vocabulary revision normalizes them.
PREDICATE_VOCAB_V1 = {
    "appears_in", "visible_body_region", "initiates_action", "performs_action",
    "contacts", "precedes", "results_in", "scene_changes_to",
    "transition_type", "contains_text", "audio_event", "speech_claim",
    "same_entity_as", "different_entity_from",
}
EDGE_TYPES = {"supports", "contradicts", "supersedes", "derived_from"}


class ClaimValidationError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _interval(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        result = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return result if result[1] > result[0] >= 0 else None


def _negative_coverage_sufficient(claim: dict[str, Any]) -> bool:
    interval = _interval(claim.get("interval"))
    if interval is None or claim.get("visibility") != "VISIBLE":
        return False
    for ref in claim.get("support_refs") or []:
        coverage = ref.get("coverage") or {}
        span = _interval(coverage.get("interval"))
        if (coverage.get("kind") == "continuous" and span is not None and
                span[0] <= interval[0] and span[1] >= interval[1]):
            return True
    return False


def validate_support_ref(ref: dict[str, Any], *, verify_files: bool = True) -> list[str]:
    problems: list[str] = []
    if not str(ref.get("ref_id") or ""):
        problems.append("support_ref_id_missing")
    if ref.get("kind") not in {"frame", "frame_range", "audio_span", "text_box",
                               "human_review", "claim"}:
        problems.append("support_ref_kind_invalid")
    path = ref.get("path")
    expected = str(ref.get("sha") or "")
    if path and verify_files:
        candidate = Path(path)
        if not candidate.is_file():
            problems.append("support_ref_file_missing")
        elif expected and sha256_file(candidate) != expected:
            problems.append("support_ref_sha_mismatch")
    return problems


def validate_claim(claim: dict[str, Any], *, verify_files: bool = True,
                   require_semantic_review: bool = False) -> list[str]:
    problems: list[str] = []
    for key in ("claim_id", "subject", "predicate", "object", "producer",
                "source_sha", "schema_version"):
        if claim.get(key) in (None, ""):
            problems.append(f"{key}_missing")
    if _interval(claim.get("interval")) is None:
        problems.append("interval_invalid")
    if claim.get("modality") not in MODALITIES:
        problems.append("modality_invalid")
    if claim.get("polarity") not in POLARITIES:
        problems.append("polarity_invalid")
    if claim.get("visibility") not in VISIBILITIES:
        problems.append("visibility_invalid")
    if claim.get("epistemic_status") not in EPISTEMIC_STATUSES:
        problems.append("epistemic_status_invalid")
    if claim.get("record_status") not in RECORD_STATUSES:
        problems.append("record_status_invalid")
    if claim.get("predicate") not in PREDICATE_VOCAB_V1:
        problems.append("predicate_pending_normalization")
    if claim.get("modality") == "LEGACY_MIXED" and \
            claim.get("epistemic_status") == "SUPPORTED":
        problems.append("legacy_mixed_cannot_be_supported")
    refs = claim.get("support_refs")
    if not isinstance(refs, list):
        problems.append("support_refs_invalid")
    else:
        for ref in refs:
            problems.extend(validate_support_ref(ref, verify_files=verify_files))
    if (claim.get("polarity") == "NEGATIVE" and
            claim.get("epistemic_status") == "SUPPORTED" and
            not _negative_coverage_sufficient(claim)):
        problems.append("negative_claim_lacks_continuous_visible_coverage")
    if require_semantic_review and claim.get("epistemic_status") == "SUPPORTED":
        review = claim.get("semantic_review") or {}
        if review.get("decision") != "supports" or not review.get("reviewer"):
            problems.append("semantic_support_not_reviewed")
    return problems


@dataclass
class ClaimLedger:
    source_sha: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    schema_version: str = CLAIM_LEDGER_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClaimLedger":
        return cls(source_sha=str(value.get("source_sha") or ""),
                   claims=list(value.get("claims") or []),
                   edges=list(value.get("edges") or []),
                   schema_version=str(value.get("schema_version") or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version,
                "source_sha": self.source_sha,
                "claims": self.claims, "edges": self.edges,
                "ledger_sha": json_hash({"claims": self.claims,
                                           "edges": self.edges})}

    def add_claim(self, claim: dict[str, Any], *, verify_files: bool = True) -> None:
        if any(row.get("claim_id") == claim.get("claim_id") for row in self.claims):
            raise ClaimValidationError("duplicate_claim_id", str(claim.get("claim_id")))
        problems = validate_claim(claim, verify_files=verify_files)
        # Pending predicates are retained as explicitly insufficient records.
        hard = [row for row in problems if row != "predicate_pending_normalization"]
        if hard:
            raise ClaimValidationError("claim_invalid", ",".join(hard))
        if "predicate_pending_normalization" in problems:
            claim["normalization_status"] = "PENDING"
            claim["epistemic_status"] = "INSUFFICIENT"
        else:
            claim["normalization_status"] = "NORMALIZED"
        self.claims.append(claim)

    def add_edge(self, edge_type: str, source: str, target: str,
                 *, basis: str = "") -> None:
        if edge_type not in EDGE_TYPES:
            raise ClaimValidationError("edge_type_invalid", edge_type)
        ids = {str(row.get("claim_id")) for row in self.claims}
        if source not in ids or target not in ids:
            raise ClaimValidationError("edge_claim_missing")
        self.edges.append({"type": edge_type, "source": source,
                           "target": target, "basis": basis})
        if edge_type == "supersedes":
            new = next(row for row in self.claims if row["claim_id"] == source)
            old = next(row for row in self.claims if row["claim_id"] == target)
            old["record_status"] = "SUPERSEDED"
            new["supersedes"] = target

    def find_conflicts(self) -> list[dict[str, Any]]:
        """Flag only aligned, comparable propositions with opposite polarity."""
        conflicts: list[dict[str, Any]] = []
        active = [row for row in self.claims if row.get("record_status") == "ACTIVE"]
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                if left.get("modality") != right.get("modality"):
                    continue
                if left.get("modality") == "LEGACY_MIXED":
                    continue
                same = all(left.get(key) == right.get(key)
                           for key in ("subject", "predicate", "object", "interval"))
                if same and left.get("polarity") != right.get("polarity"):
                    left["epistemic_status"] = "CONTESTED"
                    right["epistemic_status"] = "CONTESTED"
                    conflicts.append({"claim_ids": [left["claim_id"],
                                                     right["claim_id"]],
                                      "alignment": "exact", "resolved": False})
        return conflicts


_EVENT_FORBIDDEN = {"motivation", "audience_takeaway", "rhetorical_function",
                    "cognition_change", "emotion", "theme"}


def validate_event(event: dict[str, Any], claim_ids: set[str]) -> list[str]:
    problems = []
    if set(event).intersection(_EVENT_FORBIDDEN):
        problems.append("event_contains_interpretation")
    for key in ("event_id", "participants", "action_claim_ids", "object_ids",
                "ordering", "outcome_claim_ids", "interval"):
        if key not in event:
            problems.append(f"event_{key}_missing")
    if _interval(event.get("interval")) is None:
        problems.append("event_interval_invalid")
    for ref in (list(event.get("action_claim_ids") or []) +
                list(event.get("outcome_claim_ids") or []) +
                list(event.get("context_claim_ids") or [])):
        if str(ref) not in claim_ids:
            problems.append("event_claim_ref_missing")
    return problems


def build_event_draft(events: list[dict[str, Any]], ledger: ClaimLedger) -> dict[str, Any]:
    ids = {str(row.get("claim_id")) for row in ledger.claims}
    problems = {str(row.get("event_id")): validate_event(row, ids) for row in events}
    problems = {key: value for key, value in problems.items() if value}
    if problems:
        raise ClaimValidationError("event_draft_invalid", json.dumps(problems))
    return {"schema_version": EVENT_DRAFT_VERSION, "source_sha": ledger.source_sha,
            "events": events, "event_draft_sha": json_hash(events)}


def validate_ledger(ledger: ClaimLedger, *, verify_files: bool = True,
                    require_semantic_review: bool = False) -> list[dict[str, Any]]:
    results = []
    for claim in ledger.claims:
        problems = validate_claim(claim, verify_files=verify_files,
                                  require_semantic_review=require_semantic_review)
        if problems:
            results.append({"claim_id": claim.get("claim_id"),
                            "problems": problems})
    return results


def build_validated_reference(
        ledger: ClaimLedger, event_draft: dict[str, Any], *,
        deterministic_timeline: dict[str, Any], coverage: dict[str, Any],
        critical_claim_ids: Iterable[str]) -> dict[str, Any]:
    """Publish evidence only; interpretation and DNA fields are impossible."""
    failures = validate_ledger(ledger, require_semantic_review=True)
    accepted = [row for row in ledger.claims
                if row.get("record_status") == "ACTIVE" and
                row.get("epistemic_status") == "SUPPORTED" and
                row.get("normalization_status", "NORMALIZED") == "NORMALIZED" and
                not validate_claim(row, require_semantic_review=True)]
    accepted_ids = {str(row["claim_id"]) for row in accepted}
    critical_missing = sorted(set(map(str, critical_claim_ids)) - accepted_ids)
    if critical_missing:
        raise ClaimValidationError("critical_claims_unresolved",
                                   ",".join(critical_missing))
    event_ids = {str(row.get("event_id")) for row in event_draft.get("events") or []}
    if not event_ids:
        raise ClaimValidationError("event_draft_empty")
    accepted_events = []
    for event in event_draft.get("events") or []:
        refs = set(map(str, (event.get("action_claim_ids") or []) +
                       (event.get("outcome_claim_ids") or []) +
                       (event.get("context_claim_ids") or [])))
        if refs.issubset(accepted_ids):
            accepted_events.append(event)
    problem_by_id = {str(row.get("claim_id")): row.get("problems") or []
                     for row in failures}
    limitations = [{"claim_id": row.get("claim_id"),
                    "status": row.get("epistemic_status"),
                    "reason": row.get("limitation") or "not accepted",
                    "record_issues": problem_by_id.get(
                        str(row.get("claim_id")), [])}
                   for row in ledger.claims if row not in accepted and
                   row.get("record_status") == "ACTIVE"]
    value = {
        "schema_version": VALIDATED_REFERENCE_VERSION,
        "source_sha": ledger.source_sha,
        "accepted_claims": accepted,
        "accepted_events": accepted_events,
        "deterministic_timeline": deterministic_timeline,
        "coverage": coverage,
        "unresolved_limitations": limitations,
    }
    forbidden = {"interpretation", "creative_dna", "rhetorical_function", "theme"}
    if forbidden.intersection(value):
        raise ClaimValidationError("validated_reference_layer_violation")
    value["artifact_sha"] = json_hash(value)
    return value


def import_legacy_montage(value: dict[str, Any], *, source_sha: str) -> ClaimLedger:
    """Preserve mixed montage observations without promoting them to V facts."""
    ledger = ClaimLedger(source_sha=source_sha)
    counter = 0
    for section in value.get("sections") or []:
        for segment in section.get("segments") or []:
            counter += 1
            claim = {
                "claim_id": f"LEGACY_{counter:04d}",
                "subject": str(segment.get("segment_id") or f"SEG{counter}"),
                "predicate": "appears_in",
                "object": str(segment.get("visible_content") or "unresolved_observation"),
                "interval": list(segment.get("interval") or []),
                "modality": "LEGACY_MIXED", "polarity": "POSITIVE",
                "visibility": "UNKNOWN", "epistemic_status": "INSUFFICIENT",
                "support_refs": [{
                    "ref_id": str(segment.get("segment_id") or counter),
                    "kind": "claim", "sha": str(segment.get("raw_response_sha256") or "")
                }],
                "producer": "legacy_mixed_montage", "source_sha": source_sha,
                "schema_version": CLAIM_LEDGER_VERSION,
                "record_status": "ACTIVE", "supersedes": None,
                "limitation": "legacy request jointly exposed visual and text inputs",
            }
            ledger.add_claim(claim, verify_files=False)
    return ledger
