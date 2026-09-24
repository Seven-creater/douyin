"""Literal field mapping and calibrated acceptance for story-brief trials."""
from __future__ import annotations

from typing import Any

from src.agentic_video.manifest import json_hash


MAPPING_PROMPT = """Compare every supplied brief field with each candidate story.
Interpret each field literally, including every qualifier and domain restriction.
For each field copy its FULL text as brief_quote. Give an exact substring of the
story as candidate_quote when available; use null if no exact span captures a
negative finding. A mapped field must have an exact story quote. A broad
paraphrase must not erase a narrower requirement. Mark
mapped only if the candidate satisfies the whole field, conflict if it violates
it, and unmapped if no event realizes it. Do not infer a hidden reference or
rewrite the brief. Return JSON only:
{"schema_version":"transfer_mapping_eval_v2","checks":[
{"case_id":"C1","mappings":[{"path":"stance","brief_quote":"...",
"candidate_quote":"...","verdict":"mapped","reason":"..."}]}]}.
Input: """

STANCE_PROMPT = """Independently compare the COMPLETE communicative stance
of each brief to its candidate story. Retain every qualifier. A similar event
or surprising ending is not enough if the intended audience conclusion differs.
Return one verdict per case in input order, with a short reason. JSON only:
{"schema_version":"transfer_stance_eval_v2","checks":[
{"case_id":"C1","stance_match":false,"reason":"..."}]}.
Input: """


def public_fields(brief: dict[str, Any]) -> list[dict[str, str]]:
    """Every creative invariant must be represented in the mapping task."""
    fields = [{"path": "stance", "text": brief["stance"]}]
    fields += [{"path": f"roles.{row['role_id']}.function",
                "text": row["function"]} for row in brief["roles"]]
    fields += [{"path": f"relations.{row['relation_id']}.relation",
                "text": f"{row['from_role']} {row['relation']} {row['to_role']}"}
               for row in brief["relations"]]
    return fields


def validate_mapping(value: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    checks = value.get("checks")
    if (value.get("schema_version") != "transfer_mapping_eval_v2" or
            not isinstance(checks, list) or len(checks) != len(cases) or
            any(not isinstance(row, dict) for row in checks) or
            [row.get("case_id") for row in checks] != [
                row["case_id"] for row in cases]):
        return ["mapping_shape_invalid"]
    for check, case in zip(checks, cases):
        mappings = check.get("mappings")
        fields = case["fields"]
        if (not isinstance(mappings, list) or len(mappings) != len(fields) or
                any(not isinstance(row, dict) for row in mappings) or
                [row.get("path") for row in mappings] != [
                    row["path"] for row in fields]):
            return ["mapping_coverage_invalid"]
        for row, field in zip(mappings, fields):
            quote = row.get("candidate_quote")
            if (row.get("brief_quote") != field["text"] or
                    row.get("verdict") not in {
                        "mapped", "conflict", "unmapped"} or
                    not isinstance(row.get("reason"), str) or
                    not row["reason"].strip() or
                    (row["verdict"] == "mapped" and (
                        not isinstance(quote, str) or not quote.strip())) or
                    (quote is not None and (
                        not isinstance(quote, str) or not quote.strip() or
                        quote not in case["story"]))):
                return ["mapping_quote_or_verdict_invalid"]
    return []


def validate_stance(value: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    checks = value.get("checks")
    if (value.get("schema_version") != "transfer_stance_eval_v2" or
            not isinstance(checks, list) or len(checks) != len(cases) or
            any(not isinstance(row, dict) for row in checks) or
            [row.get("case_id") for row in checks] != [
                row["case_id"] for row in cases] or any(
                type(row.get("stance_match")) is not bool or
                not isinstance(row.get("reason"), str) or
                not row["reason"].strip() for row in checks)):
        return ["stance_eval_invalid"]
    return []


def decisions(mapping: dict[str, Any], stance: dict[str, Any],
              cases: list[dict[str, Any]]) -> dict[str, bool]:
    errors = validate_mapping(mapping, cases) + validate_stance(stance, cases)
    if errors:
        raise ValueError(",".join(errors))
    return {case["case_id"]: all(row["verdict"] == "mapped" for row in
            mapped["mappings"]) and judged["stance_match"]
            for case, mapped, judged in zip(cases, mapping["checks"],
                                           stance["checks"])}


def score(decided: dict[str, bool], labels: dict[str, bool]) -> dict[str, Any]:
    if set(decided) != set(labels):
        raise ValueError("eval_case_coverage_invalid")
    return {"passed": decided == labels,
            "case_results": [{"case_id": case_id, "expected": expected,
                              "observed": decided[case_id],
                              "matched": decided[case_id] == expected}
                             for case_id, expected in labels.items()]}


def feedback_codes(mapping: dict[str, Any], cases: list[dict[str, Any]],
                   labels: dict[str, bool]) -> list[dict[str, str]]:
    """Only generic issue classes and paths reach a second abstraction call."""
    result = []
    for case, check in zip(cases, mapping["checks"]):
        if labels[case["case_id"]]:
            result += [{"code": "invariant_not_transferable",
                        "field_path": row["path"]}
                       for row in check["mappings"]
                       if row["verdict"] != "mapped"]
    return list({(row["code"], row["field_path"]): row for row in result}.values())


def make_acceptance(*, brief: dict[str, Any], kernel_sha: str,
                    audit: dict[str, Any], calibration: dict[str, Any],
                    mapping: dict[str, Any], stance: dict[str, Any],
                    regression: dict[str, Any],
                    frozen_badcase: dict[str, Any]) -> dict[str, Any]:
    if (not calibration["passed"] or not regression["passed"] or
            not frozen_badcase["passed"]):
        raise ValueError("trial_acceptance_checks_failed")
    value = {"schema_version": "trial_acceptance_v2",
             "evaluation_policy_version": "literal_mapping_v2",
             "brief_sha": brief["artifact_sha"],
             "kernel_sha": kernel_sha,
             "kernel_audit_sha": json_hash(audit),
             "calibration_sha": json_hash(calibration),
             "frozen_badcase_sha": json_hash(frozen_badcase),
             "mapping_eval_sha": json_hash(mapping),
             "stance_eval_sha": json_hash(stance),
             "regression_sha": json_hash(regression),
             "auto_trial_allowed": True,
             "production_release_allowed": False}
    value["artifact_sha"] = json_hash(value)
    return value


def validate_acceptance(brief: dict[str, Any], value: dict[str, Any],
                        *, kernel_sha: str, audit: dict[str, Any],
                        calibration: dict[str, Any], mapping: dict[str, Any],
                        stance: dict[str, Any], regression: dict[str, Any],
                        frozen_badcase: dict[str, Any]
                        ) -> None:
    expected = {"schema_version", "evaluation_policy_version", "brief_sha",
                "kernel_sha", "kernel_audit_sha", "calibration_sha",
                "mapping_eval_sha", "stance_eval_sha", "regression_sha",
                "frozen_badcase_sha",
                "auto_trial_allowed", "production_release_allowed",
                "artifact_sha"}
    if (set(value) != expected or value.get("schema_version") !=
            "trial_acceptance_v2" or value.get("evaluation_policy_version") !=
            "literal_mapping_v2" or value.get("brief_sha") != brief.get(
                "artifact_sha") or value.get("kernel_sha") != kernel_sha or
            value.get("auto_trial_allowed") is not True or
            value.get("production_release_allowed") is not False or
            calibration.get("passed") is not True or
            regression.get("passed") is not True or
            frozen_badcase.get("passed") is not True or
            value.get("kernel_audit_sha") != json_hash(audit) or
            value.get("calibration_sha") != json_hash(calibration) or
            value.get("frozen_badcase_sha") != json_hash(frozen_badcase) or
            value.get("mapping_eval_sha") != json_hash(mapping) or
            value.get("stance_eval_sha") != json_hash(stance) or
            value.get("regression_sha") != json_hash(regression) or
            brief.get("contrast_eval_sha") != json_hash({
                "mapping_sha": json_hash(mapping),
                "stance_sha": json_hash(stance)}) or
            any(not isinstance(value.get(key), str) or not value[key] for key in (
                "kernel_audit_sha", "calibration_sha", "mapping_eval_sha",
                "stance_eval_sha", "regression_sha")) or
            value.get("artifact_sha") != json_hash({key: item for key, item in
                                                       value.items() if key !=
                                                       "artifact_sha"})):
        raise ValueError("trial_acceptance_invalid")
