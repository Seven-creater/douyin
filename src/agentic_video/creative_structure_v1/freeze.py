"""R2-C.5 freeze for the selected minimal creative-structure contract.

This module turns the explicitly recorded R2-C.4 internal simulation decision
into a public, reference-free production artifact.  It does not run a model,
generate a story, or claim that the internal simulation is human evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.agentic_video.creative_structure_v1.validator import (
    StructureValidationError,
    validate_structure_spec,
)
from src.agentic_video.manifest import json_hash


CORE_COMPONENT_IDS = (
    "I0_PRIOR_INTERPRETATION",
    "E1_NEW_INFORMATION",
    "I1_UPDATED_INTERPRETATION",
    "R1_INFORMATION_UPDATE",
)

SPEC_ID = "CREATIVE-STRUCTURE-V1"


def _with_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha"] = json_hash(result)
    return result


def _build_spec(spec_id: str) -> dict[str, Any]:
    return _with_hash({
        "schema_version": "creative_structure_spec_v1",
        "spec_id": spec_id,
        "plan_status": "complete",
        "available_dimensions": [
            "structural_schema",
            "information_state_arc",
            "generation_constraints",
            "binding_slots",
        ],
        "dimension_status": {
            "structural_schema": "supported",
            "information_state_arc": "supported",
            "generation_constraints": "supported",
            "editing_structure": "not_applicable",
            "editing_function": "not_applicable",
            "binding_slots": "supported",
        },
        "structural_schema": {
            "elements": [
                {
                    "element_id": "I0_PRIOR_INTERPRETATION",
                    "kind": "information_state",
                    "abstract_role": "a prior interpretation is available",
                },
                {
                    "element_id": "E1_NEW_INFORMATION",
                    "kind": "evidence_role",
                    "abstract_role": "new information becomes available",
                },
                {
                    "element_id": "I1_UPDATED_INTERPRETATION",
                    "kind": "information_state",
                    "abstract_role": "an updated interpretation becomes available",
                },
            ],
            "relations": [
                {
                    "relation_id": "R1_INFORMATION_UPDATE",
                    "relation_kind": "evidential",
                    "predicate": (
                        "new information updates a prior interpretation into "
                        "an updated interpretation"
                    ),
                    "arguments": [
                        {
                            "role": "prior",
                            "element_id": "I0_PRIOR_INTERPRETATION",
                        },
                        {
                            "role": "evidence",
                            "element_id": "E1_NEW_INFORMATION",
                        },
                        {
                            "role": "updated",
                            "element_id": "I1_UPDATED_INTERPRETATION",
                        },
                    ],
                    "preconditions": ["a prior interpretation is available"],
                    "effects": ["an updated interpretation becomes available"],
                }
            ],
        },
        "information_state_arc": [
            {
                "state_id": "STATE_PRIOR",
                "order": 1,
                "available_element_ids": ["I0_PRIOR_INTERPRETATION"],
                "transition_from_state_id": None,
                "trigger_relation_ids": [],
            },
            {
                "state_id": "STATE_UPDATED",
                "order": 2,
                "available_element_ids": [
                    "I0_PRIOR_INTERPRETATION",
                    "E1_NEW_INFORMATION",
                    "I1_UPDATED_INTERPRETATION",
                ],
                "transition_from_state_id": "STATE_PRIOR",
                "trigger_relation_ids": ["R1_INFORMATION_UPDATE"],
            },
        ],
        "generation_constraints": [
            {
                "constraint_id": "GC_INFORMATION_UPDATE",
                "obligation": "required",
                "constraint_type": "relational",
                "requires_relation_ids": ["R1_INFORMATION_UPDATE"],
                "rule": (
                    "The generated story must contain an initial interpretation "
                    "that is updated by newly introduced information."
                ),
            }
        ],
        # Editing was not selected by the R2-C.4 minimality decision.  Keeping
        # the field empty is different from inventing an unsupported rule.
        "editing_schema": {"constraints": []},
        "binding_slots": [
            {
                "slot_id": "B_DOMAIN",
                "slot_type": "domain",
                "bound_by": "downstream_skill",
                "constraints": ["must not alter the required relation"],
            },
            {
                "slot_id": "B_ENTITY_ROLE",
                "slot_type": "entity_role",
                "bound_by": "downstream_skill",
                "constraints": ["must remain independent of the reference entity"],
            },
            {
                "slot_id": "B_EVIDENCE_FORM",
                "slot_type": "evidence_form",
                "bound_by": "downstream_skill",
                "constraints": ["must supply the new-information role"],
            },
            {
                "slot_id": "B_SETTING",
                "slot_type": "setting",
                "bound_by": "downstream_skill",
                "constraints": ["must not alter the required relation"],
            },
        ],
        "downstream_contract": {
            "required_constraint_ids": ["GC_INFORMATION_UPDATE"],
            "variable_slot_ids": [
                "B_DOMAIN",
                "B_ENTITY_ROLE",
                "B_EVIDENCE_FORM",
                "B_SETTING",
            ],
            "authorized_stages": ["theme", "story", "screenplay"],
            "reference_context_allowed": False,
        },
    })


def validate_frozen_spec(value: dict[str, Any]) -> None:
    """Validate both the generic v1 schema and this exact frozen contract."""
    validate_structure_spec(value)
    spec_id = value.get("spec_id")
    expected = _build_spec(spec_id) if isinstance(spec_id, str) else {}
    if value != expected:
        raise StructureValidationError("frozen_spec_contract_mismatch")


def build_frozen_spec(*, spec_id: str = SPEC_ID) -> dict[str, Any]:
    spec = _build_spec(spec_id)
    validate_frozen_spec(spec)
    return spec


def _validate_transfer_inputs(cases: dict[str, Any],
                              bindings: dict[str, Any]) -> None:
    if cases.get("schema_version") != "creative_structure_transfer_cases_v1":
        raise StructureValidationError("transfer_cases_schema_invalid")
    if bindings.get("schema_version") \
            != "creative_structure_transfer_bindings_v1":
        raise StructureValidationError("transfer_bindings_schema_invalid")
    case_rows = cases.get("cases")
    binding_rows = bindings.get("cases")
    if not isinstance(case_rows, list) or not isinstance(binding_rows, list):
        raise StructureValidationError("transfer_cases_invalid")
    case_ids = [row.get("case_id") for row in case_rows]
    binding_ids = [row.get("case_id") for row in binding_rows]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(binding_ids):
        raise StructureValidationError("transfer_case_coverage_invalid")


def run_transfer_validation(spec: dict[str, Any], cases: dict[str, Any],
                            bindings: dict[str, Any]) -> dict[str, Any]:
    """Validate frozen, human-authored bindings without semantic inference."""
    validate_frozen_spec(spec)
    _validate_transfer_inputs(cases, bindings)
    case_by_id = {row["case_id"]: row for row in cases["cases"]}
    binding_by_id = {row["case_id"]: row for row in bindings["cases"]}
    required = set(CORE_COMPONENT_IDS)
    results: list[dict[str, Any]] = []
    for case_id in [row["case_id"] for row in cases["cases"]]:
        case = case_by_id[case_id]
        fixture = binding_by_id[case_id]
        structural_edits = fixture.get("structural_edits")
        if case["case_type"] == "positive":
            component_bindings = fixture.get("component_bindings")
            passed = (
                fixture.get("expected") == "preserve"
                and structural_edits == []
                and isinstance(component_bindings, dict)
                and set(component_bindings) == required
                and all(isinstance(item, str) and item.strip()
                        for item in component_bindings.values())
                and fixture.get("missing_component_ids") == []
            )
            reason = "all frozen components bind without structural edits"
        else:
            missing = fixture.get("missing_component_ids")
            passed = (
                fixture.get("expected") == "reject"
                and structural_edits == []
                and isinstance(missing, list)
                and bool(set(missing) & required)
                and set(missing) <= required
                and "R1_INFORMATION_UPDATE" in missing
                and fixture.get("component_bindings") == {}
            )
            reason = "negative control lacks the required information-update relation"
        results.append({
            "case_id": case_id,
            "case_type": case["case_type"],
            "domain": case["domain"],
            "passed": passed,
            "reason": reason if passed else "frozen binding fixture is invalid",
        })
    positive_results = [row for row in results if row["case_type"] == "positive"]
    negative_results = [row for row in results if row["case_type"] == "negative"]
    return _with_hash({
        "schema_version": "creative_structure_transfer_report_v1",
        "spec_id": spec["spec_id"],
        "spec_sha": spec["artifact_sha"],
        "status": "PASS" if all(row["passed"] for row in results) else "FAIL",
        "checks": {
            "structural_schema_valid": True,
            "reference_context_excluded": (
                spec["downstream_contract"]["reference_context_allowed"] is False),
            "positive_transfer_passed": all(
                row["passed"] for row in positive_results),
            "surface_lure_rejected": all(
                row["passed"] for row in negative_results),
        },
        "case_results": results,
    })


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_markdown(review: dict[str, Any]) -> str:
    selected = review["selected_component_ids"]
    removed = review["removed_component_ids"]
    return "\n".join([
        "# R2-C.4 Result Report",
        "",
        "Status: `SELECTED_FOR_ENGINEERING_FREEZE`",
        "",
        "> This result comes from two simulated internal reviewer perspectives. "
        "It is not an independent human-evaluation result and must not be reported as one.",
        "",
        "## Selected structure",
        "",
        *[f"- `{item}`" for item in selected],
        "",
        "## Removed from the minimal production contract",
        "",
        *[f"- `{item}`" for item in removed],
        "",
        "## Decision",
        "",
        "Candidate A (`minimal information update`) is the engineering freeze input. "
        "Candidates B and C were not selected because they add domain-specific "
        "capability/scope or social-transformation commitments.",
        "",
        "The original human-evaluation form remains empty and `NOT_READY`. The "
        "first-level deletion analyzer was not rewritten after seeing this result.",
        "",
    ])


def _transfer_markdown(report: dict[str, Any]) -> str:
    rows = [
        "# R2-C.5 Transfer Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "This is deterministic validation of frozen, human-authored binding fixtures; "
        "it does not generate stories or perform semantic judgment.",
        "",
        "| Case | Domain | Expected behavior | Result |",
        "|---|---|---|---|",
    ]
    for result in report["case_results"]:
        expected = "bind" if result["case_type"] == "positive" else "reject"
        rows.append(
            f"| `{result['case_id']}` | {result['domain']} | {expected} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    rows.extend([
        "",
        "Editing constraints are intentionally empty: the R2-C.4 decision did not "
        "select an editing component for the minimal structure.",
        "",
    ])
    return "\n".join(rows)


def write_freeze_artifacts(experiment_root: Path) -> dict[str, Any]:
    output = experiment_root / "r2_c5"
    review_path = output / "internal_review.json"
    bindings_path = output / "transfer_bindings.json"
    full_path = experiment_root / "candidates/full_structure.json"
    candidates_path = experiment_root / "candidates/screening_candidates.json"
    cases_path = experiment_root / "bindings/cases.json"
    review = _read(review_path)
    if review.get("review_mode") != "internal_simulation" \
            or review.get("human_evaluation_claim") is not False \
            or tuple(review.get("selected_component_ids", [])) != CORE_COMPONENT_IDS:
        raise StructureValidationError("internal_review_invalid")

    spec = build_frozen_spec()
    report = run_transfer_validation(spec, _read(cases_path), _read(bindings_path))
    _write_json(output / "creative_structure_spec_v1.json", spec)
    _write_json(output / "transfer_report.json", report)
    (output / "R2_C4_RESULT_REPORT.md").write_text(
        _result_markdown(review), encoding="utf-8")
    (output / "R2_C5_TRANSFER_REPORT.md").write_text(
        _transfer_markdown(report), encoding="utf-8")

    freeze_record = _with_hash({
        "schema_version": "creative_structure_freeze_record_v1",
        "freeze_id": "R2-C.5",
        "status": "FROZEN" if report["status"] == "PASS" else "BLOCKED",
        "decision_origin": "internal_simulation",
        "human_evaluation_claim": False,
        "selected_component_ids": list(CORE_COMPONENT_IDS),
        "source_files": [
            {"path": str(path.relative_to(experiment_root)), "sha256": _file_sha(path)}
            for path in (full_path, candidates_path, cases_path, review_path,
                         bindings_path)
        ],
        "spec_sha": spec["artifact_sha"],
        "transfer_report_sha": report["artifact_sha"],
        "next_authorized_stage": (
            "R2-D_SKILL_INTEGRATION" if report["status"] == "PASS" else None),
    })
    _write_json(output / "freeze_record.json", freeze_record)
    return {
        "spec": spec,
        "transfer_report": report,
        "freeze_record": freeze_record,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=repo_root / "experiments/creative_structure_minimality",
    )
    args = parser.parse_args()
    result = write_freeze_artifacts(args.experiment_root.resolve())
    print(json.dumps({
        "status": result["freeze_record"]["status"],
        "spec_sha": result["spec"]["artifact_sha"],
        "transfer_status": result["transfer_report"]["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
