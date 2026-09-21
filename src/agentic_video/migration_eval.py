"""Three-tier structural migration evaluation with sealed holdout support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.creative_dna_v2 import DNA_PUBLISH_VERSION
from src.agentic_video.manifest import json_hash
from src.agentic_video.recipe_v2 import sha256_file

MIGRATION_EVAL_VERSION = "migration_eval_v1"
FREEZE_VERSION = "dna_release_freeze_v1"

JUDGE_PROMPT = """You are an independent structural-relation judge. You receive
only an abstract Creative DNA and one candidate transfer. You never receive the
reference instance or the expected label. Return one JSON object:
{"variable_mapping":[{"variable_id":"...","candidate_binding":"..."}],
"preserved_relation_ids":["..."],"violated_constraint_ids":["..."],
"unsupported_relation_ids":["..."],"accept":true,"reason_codes":["..."]}
Judge relational and causal correspondence, not shared topic words or literary
quality. A surface-similar candidate with a different mechanism must fail; a
surface-different candidate preserving the relations may pass. JSON only. Input:
"""


class MigrationEvalError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def _parse(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MigrationEvalError("judge_json_invalid") from exc
    if not isinstance(value, dict):
        raise MigrationEvalError("judge_output_invalid")
    return value


def judge_case(dna_publish: dict[str, Any], case: dict[str, Any], *,
               runner: Any) -> dict[str, Any]:
    if dna_publish.get("schema_version") != DNA_PUBLISH_VERSION:
        raise MigrationEvalError("dna_not_publish_schema")
    # expected_accept and family metadata are intentionally not serialized.
    payload = {"creative_dna": dna_publish,
               "candidate": case.get("candidate")}
    answer = runner.ask(JUDGE_PROMPT + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")),
        max_new_tokens=1536, stop_after_json_object=True)
    result = _parse(str(getattr(answer, "text", answer)))
    for key in ("variable_mapping", "preserved_relation_ids",
                "violated_constraint_ids", "unsupported_relation_ids",
                "accept", "reason_codes"):
        if key not in result:
            raise MigrationEvalError("judge_field_missing", key)
    valid_relations = {str(row.get("relation_id"))
                       for row in dna_publish.get("relations") or []}
    valid_constraints = {str(row.get("constraint_id"))
                         for row in dna_publish.get("constraints") or []}
    returned_relations = (set(map(str, result["preserved_relation_ids"])) |
                          set(map(str, result["unsupported_relation_ids"])))
    if not returned_relations.issubset(valid_relations):
        raise MigrationEvalError("judge_relation_id_invalid")
    if not set(map(str, result["violated_constraint_ids"])).issubset(
            valid_constraints):
        raise MigrationEvalError("judge_constraint_id_invalid")
    result["case_id"] = case.get("case_id")
    result["judge_input_sha"] = json_hash(payload)
    return result


def run_suite(dna_publish: dict[str, Any], suite_path: Path, *, runner: Any,
              tier: str) -> dict[str, Any]:
    if tier not in {"dev", "regression", "holdout"}:
        raise MigrationEvalError("tier_invalid", tier)
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    cases = suite.get("cases") or []
    results = []
    for case in cases:
        verdict = judge_case(dna_publish, case, runner=runner)
        expected = bool(case.get("expected_accept"))
        verdict["expected_accept"] = expected
        verdict["correct"] = bool(verdict["accept"]) == expected
        verdict["relation_family"] = case.get("relation_family")
        verdict["contrast_type"] = case.get("contrast_type")
        results.append(verdict)
    positives = [row for row in results if row["expected_accept"]]
    negatives = [row for row in results if not row["expected_accept"]]
    report = {
        "schema_version": MIGRATION_EVAL_VERSION,
        "tier": tier,
        "suite_sha": sha256_file(Path(suite_path)),
        "dna_sha": dna_publish.get("artifact_sha"),
        "case_count": len(results),
        "required_positive_passed": all(row["accept"] for row in positives),
        "structure_keep_rate": (sum(bool(row["accept"]) for row in positives) /
                                len(positives) if positives else None),
        "surface_lure_false_accept_rate": (
            sum(bool(row["accept"]) for row in negatives) / len(negatives)
            if negatives else None),
        "all_expected_passed": all(row["correct"] for row in results),
        "results": results,
    }
    report["report_sha"] = json_hash(report)
    return report


def freeze_candidate(*, dna_publish: dict[str, Any], model_config: dict[str, Any],
                     template_paths: list[Path], judge_path: Path,
                     rules_path: Path, dev_suite: Path, regression_suite: Path,
                     holdout_suite: Path | None = None) -> dict[str, Any]:
    def file_row(path: Path) -> dict[str, str]:
        path = Path(path)
        return {"path": str(path), "sha": sha256_file(path)}

    freeze = {
        "schema_version": FREEZE_VERSION,
        "release_status": "release_candidate" if holdout_suite is None else "frozen",
        "dna_sha": dna_publish.get("artifact_sha"),
        "model_config_sha": json_hash(model_config),
        "templates": [file_row(path) for path in template_paths],
        "judge": file_row(judge_path),
        "rules": file_row(rules_path),
        "dev_suite": file_row(dev_suite),
        "regression_suite": file_row(regression_suite),
        "holdout_suite": (file_row(holdout_suite) if holdout_suite else None),
        "holdout_policy": (
            "User-custodied. A disclosed suite becomes regression data and "
            "cannot support an independent release claim."),
    }
    freeze["freeze_sha"] = json_hash(freeze)
    return freeze


def release_decision(*, dev_report: dict[str, Any],
                     regression_report: dict[str, Any],
                     isolation_report: dict[str, Any],
                     holdout_report: dict[str, Any] | None) -> dict[str, Any]:
    base_pass = (bool(dev_report.get("all_expected_passed")) and
                 bool(regression_report.get("all_expected_passed")) and
                 bool(isolation_report.get("passed")))
    if holdout_report is None:
        return {"status": "release_candidate" if base_pass else "blocked",
                "second_commit_allowed": False,
                "reason": "holdout_not_provided" if base_pass else
                          "development_gates_failed"}
    holdout_pass = (bool(holdout_report.get("required_positive_passed")) and
                    float(holdout_report.get("surface_lure_false_accept_rate") or 0) == 0)
    return {"status": "release" if base_pass and holdout_pass else "blocked",
            "second_commit_allowed": bool(base_pass and holdout_pass),
            "reason": "all_gates_passed" if base_pass and holdout_pass else
                      "release_gate_failed"}

