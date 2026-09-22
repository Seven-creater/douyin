"""Typed, deterministic contracts for R2-D Wave 1 artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from src.agentic_video.creative_structure_v1.freeze import CORE_COMPONENT_IDS
from src.agentic_video.manifest import json_hash


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_STATUSES = frozenset({"draft", "tested", "committed"})
CANDIDATE_STATUSES = frozenset({
    "generated", "schema_valid", "eligible", "scored", "selected",
    "rejected", "superseded",
})


class ContractError(ValueError):
    def __init__(self, reason_code: str, detail: object = "") -> None:
        super().__init__(f"{reason_code}:{detail}" if detail else reason_code)
        self.reason_code = reason_code
        self.detail = detail


def _require(condition: bool, reason: str, detail: object = "") -> None:
    if not condition:
        raise ContractError(reason, detail)


def _exact_keys(value: dict[str, Any], expected: Iterable[str], reason: str) -> None:
    _require(set(value) == set(expected), reason,
             f"missing={sorted(set(expected) - set(value))},"
             f"extra={sorted(set(value) - set(expected))}")


def _text(value: object, reason: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), reason)
    return str(value)


def _sha(value: object, reason: str) -> str:
    _require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
             reason, value)
    return str(value)


def artifact_ref(artifact_id: str, sha: str) -> dict[str, str]:
    return {"artifact_id": _text(artifact_id, "artifact_ref_id_invalid"),
            "sha": _sha(sha, "artifact_ref_sha_invalid")}


@dataclass(frozen=True)
class ArtifactEnvelope:
    artifact_id: str
    artifact_type: str
    schema_version: str
    version: str
    payload: dict[str, Any]
    producer: dict[str, Any]
    derived_from: tuple[dict[str, str], ...] = ()
    policy_versions: dict[str, str] = field(default_factory=dict)
    status: str = "draft"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": "creative_artifact_envelope_v1",
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "payload_schema_version": self.schema_version,
            "version": self.version,
            "content_sha": json_hash(self.payload),
            "status": self.status,
            "producer": dict(self.producer),
            "derived_from": [dict(row) for row in self.derived_from],
            "policy_versions": dict(self.policy_versions),
            "payload": dict(self.payload),
        }
        validate_artifact_envelope(result)
        return result


def validate_artifact_envelope(value: dict[str, Any]) -> None:
    _exact_keys(value, {
        "schema_version", "artifact_id", "artifact_type",
        "payload_schema_version", "version", "content_sha", "status",
        "producer", "derived_from", "policy_versions", "payload",
    }, "artifact_envelope_keys_invalid")
    _require(value["schema_version"] == "creative_artifact_envelope_v1",
             "artifact_envelope_schema_invalid")
    for key in ("artifact_id", "artifact_type", "payload_schema_version",
                "version"):
        _text(value[key], f"artifact_{key}_invalid")
    _require(value["status"] in ARTIFACT_STATUSES, "artifact_status_invalid")
    _sha(value["content_sha"], "artifact_content_sha_invalid")
    _require(value["content_sha"] == json_hash(value["payload"]),
             "artifact_content_sha_mismatch")
    producer = value["producer"]
    _require(isinstance(producer, dict), "artifact_producer_invalid")
    _exact_keys(producer, {
        "skill_id", "skill_version", "package_sha",
        "prompt_or_instruction_sha",
    }, "artifact_producer_keys_invalid")
    _text(producer["skill_id"], "artifact_skill_id_invalid")
    _text(producer["skill_version"], "artifact_skill_version_invalid")
    _sha(producer["package_sha"], "artifact_package_sha_invalid")
    _sha(producer["prompt_or_instruction_sha"],
         "artifact_instruction_sha_invalid")
    refs = value["derived_from"]
    _require(isinstance(refs, list), "artifact_derived_from_invalid")
    seen: set[tuple[str, str]] = set()
    for row in refs:
        _require(isinstance(row, dict), "artifact_parent_invalid")
        _exact_keys(row, {"artifact_id", "sha"}, "artifact_parent_keys_invalid")
        pair = (_text(row["artifact_id"], "artifact_parent_id_invalid"),
                _sha(row["sha"], "artifact_parent_sha_invalid"))
        _require(pair not in seen, "artifact_parent_duplicate", pair[0])
        seen.add(pair)
    _require(isinstance(value["policy_versions"], dict),
             "artifact_policy_versions_invalid")


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    artifact_id: str
    artifact_type: str
    artifact_sha: str
    parent_shas: tuple[str, ...]
    status: str = "eligible"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "candidate_id": self.candidate_id,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "artifact_sha": self.artifact_sha,
            "parent_shas": list(self.parent_shas),
            "status": self.status,
        }
        validate_candidate_record(result)
        return result


def validate_candidate_record(value: dict[str, Any]) -> None:
    _exact_keys(value, {
        "candidate_id", "artifact_id", "artifact_type", "artifact_sha",
        "parent_shas", "status",
    }, "candidate_record_keys_invalid")
    for key in ("candidate_id", "artifact_id", "artifact_type"):
        _text(value[key], f"candidate_{key}_invalid")
    _sha(value["artifact_sha"], "candidate_artifact_sha_invalid")
    _require(value["status"] in CANDIDATE_STATUSES,
             "candidate_status_invalid")
    parents = value["parent_shas"]
    _require(isinstance(parents, list) and bool(parents),
             "candidate_parent_shas_invalid")
    for parent in parents:
        _sha(parent, "candidate_parent_sha_invalid")


@dataclass(frozen=True)
class CandidatePool:
    pool_id: str
    stage: str
    candidate_artifact_type: str
    parent_refs: tuple[dict[str, str], ...]
    candidates: tuple[CandidateRecord, ...]
    selection_policy_version: str

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": "creative_candidate_pool_v1",
            "pool_id": self.pool_id,
            "stage": self.stage,
            "candidate_artifact_type": self.candidate_artifact_type,
            "parent_refs": [dict(row) for row in self.parent_refs],
            "candidates": [row.to_dict() for row in self.candidates],
            "selection_policy_version": self.selection_policy_version,
        }
        validate_candidate_pool(result)
        return result


def validate_candidate_pool(value: dict[str, Any]) -> None:
    _exact_keys(value, {
        "schema_version", "pool_id", "stage", "candidate_artifact_type",
        "parent_refs", "candidates", "selection_policy_version",
    }, "candidate_pool_keys_invalid")
    _require(value["schema_version"] == "creative_candidate_pool_v1",
             "candidate_pool_schema_invalid")
    for key in ("pool_id", "stage", "candidate_artifact_type",
                "selection_policy_version"):
        _text(value[key], f"candidate_pool_{key}_invalid")
    refs = value["parent_refs"]
    _require(isinstance(refs, list) and bool(refs),
             "candidate_pool_parent_refs_invalid")
    for row in refs:
        artifact_ref(row.get("artifact_id"), row.get("sha"))
    rows = value["candidates"]
    _require(isinstance(rows, list) and bool(rows), "candidate_pool_empty")
    ids: set[str] = set()
    for row in rows:
        validate_candidate_record(row)
        _require(row["candidate_id"] not in ids,
                 "candidate_pool_duplicate", row["candidate_id"])
        ids.add(row["candidate_id"])
        _require(row["artifact_type"] == value["candidate_artifact_type"],
                 "candidate_pool_artifact_type_mismatch", row["candidate_id"])


@dataclass(frozen=True)
class SelectionArtifact:
    selection_id: str
    pool_ref: dict[str, str]
    selected_candidate_id: str
    selected_candidate_sha: str
    scorecard_shas: tuple[str, ...]
    selection_policy_version: str
    decision_origin: str

    def to_dict(self, pool: dict[str, Any]) -> dict[str, Any]:
        result = {
            "schema_version": "creative_selection_v1",
            "selection_id": self.selection_id,
            "pool_ref": dict(self.pool_ref),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_sha": self.selected_candidate_sha,
            "scorecard_shas": list(self.scorecard_shas),
            "selection_policy_version": self.selection_policy_version,
            "decision_origin": self.decision_origin,
        }
        validate_selection_artifact(result, pool)
        return result


def validate_selection_artifact(value: dict[str, Any],
                                pool: dict[str, Any]) -> None:
    _exact_keys(value, {
        "schema_version", "selection_id", "pool_ref",
        "selected_candidate_id", "selected_candidate_sha", "scorecard_shas",
        "selection_policy_version", "decision_origin",
    }, "selection_keys_invalid")
    _require(value["schema_version"] == "creative_selection_v1",
             "selection_schema_invalid")
    validate_candidate_pool(pool)
    for key in ("selection_id", "selected_candidate_id",
                "selection_policy_version", "decision_origin"):
        _text(value[key], f"selection_{key}_invalid")
    artifact_ref(value["pool_ref"].get("artifact_id"),
                 value["pool_ref"].get("sha"))
    _sha(value["selected_candidate_sha"], "selection_candidate_sha_invalid")
    scorecards = value["scorecard_shas"]
    _require(isinstance(scorecards, list), "selection_scorecards_invalid")
    for scorecard in scorecards:
        _sha(scorecard, "selection_scorecard_sha_invalid")
    candidates = {row["candidate_id"]: row for row in pool["candidates"]}
    selected = candidates.get(value["selected_candidate_id"])
    _require(selected is not None, "selection_candidate_unknown")
    _require(selected["artifact_sha"] == value["selected_candidate_sha"],
             "selection_candidate_sha_mismatch")
    _require(value["selection_policy_version"]
             == pool["selection_policy_version"],
             "selection_policy_mismatch")


def build_validation_report(*, stage: str, validator_id: str,
                            candidates: Iterable[CandidateRecord]
                            ) -> dict[str, Any]:
    rows = [{
        "candidate_id": candidate.candidate_id,
        "artifact_sha": candidate.artifact_sha,
        "passed": True,
        "reason_codes": [],
    } for candidate in candidates]
    result = {
        "schema_version": "creative_validation_report_v1",
        "stage": stage,
        "validator_ids": [validator_id],
        "status": "PASS",
        "candidate_results": rows,
    }
    validate_validation_report(result)
    return result


def validate_validation_report(value: dict[str, Any]) -> None:
    _exact_keys(value, {
        "schema_version", "stage", "validator_ids", "status",
        "candidate_results",
    }, "validation_report_keys_invalid")
    _require(value["schema_version"] == "creative_validation_report_v1",
             "validation_report_schema_invalid")
    _text(value["stage"], "validation_stage_invalid")
    validators = value["validator_ids"]
    _require(isinstance(validators, list) and validators,
             "validation_validator_ids_invalid")
    for validator in validators:
        _text(validator, "validation_validator_id_invalid")
    rows = value["candidate_results"]
    _require(isinstance(rows, list) and rows, "validation_results_empty")
    for row in rows:
        _exact_keys(row, {"candidate_id", "artifact_sha", "passed",
                          "reason_codes"}, "validation_result_keys_invalid")
        _text(row["candidate_id"], "validation_candidate_id_invalid")
        _sha(row["artifact_sha"], "validation_candidate_sha_invalid")
        _require(row["passed"] is True, "validation_candidate_failed")
        _require(row["reason_codes"] == [], "validation_reason_codes_invalid")
    _require(value["status"] == "PASS", "validation_status_invalid")


def validate_theme_candidate(value: dict[str, Any], *,
                             structure_sha: str) -> None:
    _exact_keys(value, {
        "schema_version", "theme_id", "parent_structure_sha", "domain",
        "premise", "audience_promise", "tone", "binding_slots",
        "structure_bindings",
    }, "theme_candidate_keys_invalid")
    _require(value["schema_version"] == "theme_candidate_v1",
             "theme_candidate_schema_invalid")
    for key in ("theme_id", "domain", "premise", "audience_promise", "tone"):
        _text(value[key], f"theme_{key}_invalid")
    _require(value["parent_structure_sha"] == structure_sha,
             "theme_parent_structure_sha_mismatch")
    slots = value["binding_slots"]
    _require(isinstance(slots, dict), "theme_binding_slots_invalid")
    _exact_keys(slots, {"B_DOMAIN", "B_ENTITY_ROLE", "B_EVIDENCE_FORM",
                        "B_SETTING"}, "theme_binding_slot_keys_invalid")
    for key, item in slots.items():
        _text(item, f"theme_binding_slot_invalid:{key}")
    _validate_structure_bindings(value["structure_bindings"], "theme")


def _validate_structure_bindings(value: object, prefix: str) -> None:
    _require(isinstance(value, dict), f"{prefix}_structure_bindings_invalid")
    _exact_keys(value, CORE_COMPONENT_IDS,
                f"{prefix}_structure_binding_keys_invalid")
    for component_id in CORE_COMPONENT_IDS[:3]:
        _text(value[component_id], f"{prefix}_structure_binding_invalid")
    relation = value["R1_INFORMATION_UPDATE"]
    _require(isinstance(relation, dict), f"{prefix}_relation_binding_invalid")
    _exact_keys(relation, {"prior", "evidence", "updated", "statement"},
                f"{prefix}_relation_binding_keys_invalid")
    _require(relation["prior"] == "I0_PRIOR_INTERPRETATION"
             and relation["evidence"] == "E1_NEW_INFORMATION"
             and relation["updated"] == "I1_UPDATED_INTERPRETATION",
             f"{prefix}_relation_binding_invalid")
    _text(relation["statement"], f"{prefix}_relation_statement_invalid")


def validate_story_blueprint(value: dict[str, Any], *,
                             structure_sha: str,
                             theme_id: str) -> None:
    _exact_keys(value, {
        "schema_version", "blueprint_id", "theme_id", "parent_structure_sha",
        "logline", "characters", "setting", "goal", "stakes", "events",
        "event_relations", "structure_bindings", "production_assumptions",
    }, "story_blueprint_keys_invalid")
    _require(value["schema_version"] == "story_blueprint_v1",
             "story_blueprint_schema_invalid")
    for key in ("blueprint_id", "logline", "setting", "goal", "stakes"):
        _text(value[key], f"story_{key}_invalid")
    _require(value["theme_id"] == theme_id, "story_theme_id_mismatch")
    _require(value["parent_structure_sha"] == structure_sha,
             "story_parent_structure_sha_mismatch")
    _require(isinstance(value["characters"], list) and value["characters"],
             "story_characters_invalid")
    events = value["events"]
    _require(isinstance(events, list) and len(events) >= 3,
             "story_events_invalid")
    event_ids = [row.get("event_id") for row in events]
    _require(all(isinstance(item, str) and item for item in event_ids)
             and len(event_ids) == len(set(event_ids)), "story_event_ids_invalid")
    relations = value["event_relations"]
    _require(isinstance(relations, list) and len(relations) == 1,
             "story_event_relations_invalid")
    relation = relations[0]
    _exact_keys(relation, {"relation_id", "type", "prior_event_id",
                           "evidence_event_id", "updated_event_id"},
                "story_event_relation_keys_invalid")
    _require(relation["type"] == "information_update",
             "story_event_relation_type_invalid")
    for key in ("prior_event_id", "evidence_event_id", "updated_event_id"):
        _require(relation[key] in event_ids, "story_event_relation_unresolved")
    _require(len({relation["prior_event_id"], relation["evidence_event_id"],
                  relation["updated_event_id"]}) == 3,
             "story_information_roles_not_distinct")
    bindings = value["structure_bindings"]
    _validate_structure_bindings(bindings, "story")
    _require(bindings["I0_PRIOR_INTERPRETATION"] == relation["prior_event_id"]
             and bindings["E1_NEW_INFORMATION"] == relation["evidence_event_id"]
             and bindings["I1_UPDATED_INTERPRETATION"]
             == relation["updated_event_id"],
             "story_structure_event_binding_mismatch")
    _require(bindings["R1_INFORMATION_UPDATE"]["statement"]
             == relation["relation_id"], "story_relation_binding_mismatch")
    _require(isinstance(value["production_assumptions"], list),
             "story_production_assumptions_invalid")
