"""Deterministic tooling for the R2-C.4 human-evaluation experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ExperimentDataError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentDataError(message)


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label}_keys_invalid")


def validate_screening_candidates(value: dict[str, Any]) -> None:
    _exact_keys(value, {"schema_version", "purpose", "candidates"}, "screening")
    _require(value["schema_version"] == "creative_structure_screening_v1",
             "screening_schema_invalid")
    rows = value["candidates"]
    _require(isinstance(rows, list) and rows, "screening_candidates_empty")
    ids: set[str] = set()
    for row in rows:
        _exact_keys(row, {"candidate_id", "label", "components", "relations"},
                    "screening_candidate")
        candidate_id = row["candidate_id"]
        _require(isinstance(candidate_id, str) and candidate_id not in ids,
                 "screening_candidate_id_invalid")
        ids.add(candidate_id)
        _require(isinstance(row["label"], str) and row["label"].strip(),
                 "screening_candidate_label_invalid")
        _require(isinstance(row["components"], list) and row["components"],
                 "screening_components_invalid")
        _require(isinstance(row["relations"], list) and row["relations"],
                 "screening_relations_invalid")


def validate_full_structure(value: dict[str, Any]) -> None:
    _exact_keys(value, {"schema_version", "candidate_id", "components"},
                "full_structure")
    _require(value["schema_version"] == "creative_structure_full_candidate_v1",
             "full_structure_schema_invalid")
    _require(isinstance(value["candidate_id"], str)
             and value["candidate_id"].strip(), "full_candidate_id_invalid")
    rows = value["components"]
    _require(isinstance(rows, list) and rows, "full_components_empty")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        _exact_keys(row, {"component_id", "kind", "statement", "depends_on"},
                    "full_component")
        component_id = row["component_id"]
        _require(isinstance(component_id, str) and component_id
                 and component_id not in by_id, "component_id_invalid")
        _require(row["kind"] in {
            "state", "evidence", "specialization", "qualifier", "context",
            "relation",
        }, "component_kind_invalid")
        _require(isinstance(row["statement"], str) and row["statement"].strip(),
                 "component_statement_invalid")
        dependencies = row["depends_on"]
        _require(isinstance(dependencies, list)
                 and len(dependencies) == len(set(dependencies))
                 and all(isinstance(item, str) for item in dependencies),
                 "component_dependencies_invalid")
        by_id[component_id] = row
    for component_id, row in by_id.items():
        _require(component_id not in row["depends_on"], "component_self_dependency")
        _require(not (set(row["depends_on"]) - set(by_id)),
                 "component_dependency_unresolved")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(component_id: str) -> None:
        _require(component_id not in visiting, "component_dependency_cycle")
        if component_id in visited:
            return
        visiting.add(component_id)
        for dependency in by_id[component_id]["depends_on"]:
            visit(dependency)
        visiting.remove(component_id)
        visited.add(component_id)

    for component_id in by_id:
        visit(component_id)


def validate_cases(value: dict[str, Any]) -> None:
    _exact_keys(value, {"schema_version", "cases"}, "cases")
    _require(value["schema_version"] == "creative_structure_transfer_cases_v1",
             "cases_schema_invalid")
    rows = value["cases"]
    _require(isinstance(rows, list) and rows, "cases_empty")
    ids: set[str] = set()
    for row in rows:
        _exact_keys(row, {
            "case_id", "case_type", "domain", "description", "control",
            "available_binding_material",
        }, "case")
        case_id = row["case_id"]
        _require(isinstance(case_id, str) and case_id not in ids,
                 "case_id_invalid")
        ids.add(case_id)
        _require(row["case_type"] in {"positive", "negative"},
                 "case_type_invalid")
        for key in ("domain", "description", "control"):
            _require(isinstance(row[key], str) and row[key].strip(),
                     f"case_{key}_invalid")
        _require(isinstance(row["available_binding_material"], list),
                 "case_binding_material_invalid")
    _require(any(row["case_type"] == "positive" for row in rows),
             "positive_cases_missing")
    _require(any(row["case_type"] == "negative" for row in rows),
             "negative_cases_missing")


def build_single_deletion_variants(
        full: dict[str, Any]) -> list[dict[str, Any]]:
    validate_full_structure(full)
    components = full["components"]
    ordered_ids = [row["component_id"] for row in components]
    dependencies = {row["component_id"]: set(row["depends_on"])
                    for row in components}
    variants: list[dict[str, Any]] = []
    total = len(ordered_ids)
    for root in ordered_ids:
        removed = {root}
        changed = True
        while changed:
            changed = False
            for component_id in ordered_ids:
                if component_id not in removed \
                        and dependencies[component_id].intersection(removed):
                    removed.add(component_id)
                    changed = True
        removed_ordered = [item for item in ordered_ids if item in removed]
        retained = [item for item in ordered_ids if item not in removed]
        variants.append({
            "variant_id": f"DELETE_{root}",
            "parent_candidate_id": full["candidate_id"],
            "removed_root_id": root,
            "removed_component_ids": removed_ordered,
            "retained_component_ids": retained,
            "compression_ratio": round(len(removed_ordered) / total, 6),
        })
    return variants


def _result_complete(result: dict[str, Any], case_ids: set[str]) -> bool:
    required = {
        "variant_id", "source_sufficient", "case_decisions",
        "binding_additions", "unresolved_binding_decisions",
        "reference_bound_commitments", "reviewer_disagreements",
    }
    if set(result) != required or set(result.get("case_decisions") or {}) != case_ids:
        return False
    scalar_keys = required - {"variant_id", "case_decisions"}
    return all(result[key] is not None for key in scalar_keys) \
        and all(value is not None for value in result["case_decisions"].values())


def analyze_results(full: dict[str, Any], variants: list[dict[str, Any]],
                    cases: dict[str, Any], evaluation: dict[str, Any]
                    ) -> dict[str, Any]:
    validate_full_structure(full)
    validate_cases(cases)
    _exact_keys(evaluation, {"schema_version", "results"}, "evaluation")
    _require(evaluation["schema_version"]
             == "creative_structure_minimality_evaluation_v1",
             "evaluation_schema_invalid")
    _require(isinstance(evaluation["results"], list), "evaluation_results_invalid")

    expected_ids = [full["candidate_id"], *(row["variant_id"] for row in variants)]
    by_id: dict[str, dict[str, Any]] = {}
    for result in evaluation["results"]:
        variant_id = result.get("variant_id") if isinstance(result, dict) else None
        _require(isinstance(variant_id, str) and variant_id not in by_id,
                 "evaluation_variant_id_invalid")
        _require(variant_id in expected_ids, "evaluation_variant_unknown")
        by_id[variant_id] = result
    missing = [item for item in expected_ids if item not in by_id]
    case_ids = {row["case_id"] for row in cases["cases"]}
    incomplete = [item for item, result in by_id.items()
                  if not _result_complete(result, case_ids)]
    if missing or incomplete:
        return {
            "schema_version": "creative_structure_minimality_report_v1",
            "status": "NOT_READY",
            "selected_variant_id": None,
            "selected_compression_ratio": None,
            "missing_result_ids": missing,
            "incomplete_result_ids": incomplete,
            "component_necessity": {},
            "variant_results": [],
        }

    positive_ids = {row["case_id"] for row in cases["cases"]
                    if row["case_type"] == "positive"}
    negative_ids = case_ids - positive_ids
    ratio_by_id = {full["candidate_id"]: 0.0}
    retained_count = {full["candidate_id"]: len(full["components"])}
    ratio_by_id.update({row["variant_id"]: row["compression_ratio"]
                        for row in variants})
    retained_count.update({row["variant_id"]: len(row["retained_component_ids"])
                           for row in variants})
    rows = []
    for variant_id in expected_ids:
        result = by_id[variant_id]
        decisions = result["case_decisions"]
        gate_pass = (
            result["source_sufficient"] is True
            and all(decisions[item] is True for item in positive_ids)
            and all(decisions[item] is True for item in negative_ids)
            and result["reference_bound_commitments"] == 0
        )
        rows.append({
            "variant_id": variant_id,
            "gate_pass": gate_pass,
            "compression_ratio": ratio_by_id[variant_id],
            "retained_component_count": retained_count[variant_id],
            "binding_additions": result["binding_additions"],
            "unresolved_binding_decisions": result[
                "unresolved_binding_decisions"],
            "reviewer_disagreements": result["reviewer_disagreements"],
        })
    passing = [row for row in rows if row["gate_pass"]]
    passing.sort(key=lambda row: (
        -row["compression_ratio"],
        row["binding_additions"],
        row["unresolved_binding_decisions"],
        row["retained_component_count"],
        row["variant_id"],
    ))
    full_passed = next(row for row in rows
                       if row["variant_id"] == full["candidate_id"])["gate_pass"]
    pass_by_id = {row["variant_id"]: row["gate_pass"] for row in rows}
    necessity = {
        row["removed_root_id"]: bool(
            full_passed and not pass_by_id[row["variant_id"]])
        for row in variants
    }
    selected = passing[0] if passing else None
    return {
        "schema_version": "creative_structure_minimality_report_v1",
        "status": "SELECTED" if selected else "INCONCLUSIVE",
        "selected_variant_id": selected["variant_id"] if selected else None,
        "selected_compression_ratio": (
            selected["compression_ratio"] if selected else None),
        "missing_result_ids": [],
        "incomplete_result_ids": [],
        "component_necessity": necessity,
        "variant_results": rows,
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"validate", "generate-deletions", "analyze"})
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    full = _read(root / "candidates/full_structure.json")
    cases = _read(root / "bindings/cases.json")
    candidates = _read(root / "candidates/screening_candidates.json")
    validate_full_structure(full)
    validate_cases(cases)
    validate_screening_candidates(candidates)
    variants = build_single_deletion_variants(full)
    if args.command == "validate":
        print("VALID")
    elif args.command == "generate-deletions":
        payload = {"schema_version": "creative_structure_ablation_plan_v1",
                   "full_candidate_id": full["candidate_id"], "variants": variants}
        _write(args.output or root / "deletion_tests/ablation_plan.json", payload)
    else:
        evaluation_path = args.evaluation or root / "evaluation_forms/minimality_evaluation.json"
        report = analyze_results(full, variants, cases, _read(evaluation_path))
        if args.output:
            _write(args.output, report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
