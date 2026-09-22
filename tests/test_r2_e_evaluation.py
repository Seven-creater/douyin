from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_r2_d_wave5 import _payload, _runtime


def _first_candidate(workspace, wave5_result: dict, kind: str = "image"):
    shot = wave5_result["shot_results"][0]
    pool = _payload(workspace, shot[f"{kind}_pool_ref"]["artifact_id"])
    record = pool["candidates"][0]
    return record, _payload(workspace, record["artifact_id"]), shot


def test_r2e_materializes_records_without_changing_wave5(
        tmp_path: Path) -> None:
    workspace, orchestrator, wave5 = _runtime(tmp_path, wave5=True)
    upstream = {
        name: (workspace.get_sha(name),
               workspace.state["artifacts"][name]["active_version"],
               workspace.effective_status(name))
        for name in workspace.state["artifacts"]}

    result = orchestrator.run_r2e()

    assert result["status"] == "PASS"
    assert result["fixture_only"] is True
    assert result["model_calls"] == 0
    assert result["media_generated"] is False
    assert result["visual_evaluation"] == "not_run"
    assert result["creative_evaluation"] == "not_run"
    assert len(result["candidate_results"]) == 12
    assert upstream == {
        name: (workspace.get_sha(name),
               workspace.state["artifacts"][name]["active_version"],
               workspace.effective_status(name))
        for name in upstream}

    for row in result["candidate_results"]:
        experiment = _payload(
            workspace, row["experiment_ref"]["artifact_id"])
        structure = _payload(
            workspace, row["structure_evaluation_ref"]["artifact_id"])
        visual = _payload(
            workspace, row["visual_consistency_ref"]["artifact_id"])
        scorecard = _payload(
            workspace, row["scorecard_ref"]["artifact_id"])
        assert experiment["execution_mode"] == "fake_metadata_only"
        assert experiment["model_id"] == "none"
        assert experiment["generation_time_ms"] == 0
        assert len(experiment["model_sha"]) == 64
        assert len(experiment["config_sha"]) == 64
        assert len(experiment["prompt_sha"]) == 64
        assert structure["deterministic_status"] == "PASS"
        assert structure["semantic_realization"] == "not_run"
        assert structure["status"] == "PARTIAL"
        assert set(visual["dimensions"].values()) == {"not_run"}
        assert visual["status"] == "PARTIAL"
        assert scorecard["deterministic_status"] == "PASS"
        assert scorecard["overall_status"] == "INCOMPLETE"
        assert scorecard["selection_eligible"] is False
        assert set(scorecard["quality_dimensions"].values()) == {"not_run"}


def test_fake_evaluator_separates_three_failure_surfaces(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.evaluation.r2e import (
        FakeGenerationEvaluator,
    )

    workspace, _, wave5 = _runtime(tmp_path, wave5=True)
    record, candidate, shot = _first_candidate(workspace, wave5)
    contract_ref = shot["shot_contract_ref"]
    contract = _payload(workspace, contract_ref["artifact_id"])
    candidate_ref = {
        "artifact_id": record["artifact_id"], "sha": record["artifact_sha"]}
    asset_ref = {
        "artifact_id": "creative:asset_graph",
        "sha": workspace.get_sha("creative:asset_graph")}
    evaluator = FakeGenerationEvaluator()

    bad_structure = deepcopy(candidate)
    bad_structure["structure_trace"]["structure_roles"] = []
    structure = evaluator.evaluate_structure(
        candidate=bad_structure, candidate_ref=candidate_ref,
        shot_contract=contract, shot_contract_ref=contract_ref)
    assert structure["status"] == "FAIL"
    assert next(row for row in structure["checks"]
                if row["check"] == "role_binding")["passed"] is False

    bad_assets = deepcopy(candidate)
    bad_assets["asset_refs"] = []
    visual = evaluator.evaluate_visual_consistency(
        candidate=bad_assets, candidate_ref=candidate_ref,
        shot_contract=contract, shot_contract_ref=contract_ref,
        asset_graph_ref=asset_ref)
    assert visual["status"] == "FAIL"
    assert next(row for row in visual["checks"]
                if row["check"] == "asset_reference_binding")[
                    "passed"] is False

    valid_structure = evaluator.evaluate_structure(
        candidate=candidate, candidate_ref=candidate_ref,
        shot_contract=contract, shot_contract_ref=contract_ref)
    valid_visual = evaluator.evaluate_visual_consistency(
        candidate=candidate, candidate_ref=candidate_ref,
        shot_contract=contract, shot_contract_ref=contract_ref,
        asset_graph_ref=asset_ref)
    bad_shot = deepcopy(candidate)
    bad_shot["shot_id"] = "WRONG_SHOT"
    scorecard = evaluator.build_quality_scorecard(
        candidate=bad_shot, candidate_ref=candidate_ref,
        shot_contract=contract, shot_contract_ref=contract_ref,
        asset_graph_ref=asset_ref,
        experiment_ref={"artifact_id": "creative:experiment:X",
                        "sha": "1" * 64},
        structure_evaluation=valid_structure,
        structure_evaluation_ref={
            "artifact_id": "creative:structure:X", "sha": "2" * 64},
        visual_consistency=valid_visual,
        visual_consistency_ref={
            "artifact_id": "creative:visual:X", "sha": "3" * 64})
    assert scorecard["deterministic_status"] == "FAIL"
    assert scorecard["overall_status"] == "FAIL"
    assert scorecard["selection_eligible"] is False


def test_generation_experiment_rejects_candidate_provenance_drift(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.generation.experiment import (
        build_generation_experiment_record,
        validate_generation_experiment_record,
    )

    workspace, _, wave5 = _runtime(tmp_path, wave5=True)
    record, candidate, _ = _first_candidate(workspace, wave5)
    value = build_generation_experiment_record(
        candidate=candidate,
        candidate_ref={"artifact_id": record["artifact_id"],
                       "sha": record["artifact_sha"]})
    value["seed"] += 1
    with pytest.raises(ContractError,
                       match="generation_experiment_candidate_mismatch"):
        validate_generation_experiment_record(value, candidate=candidate)


@pytest.mark.parametrize(("failure_class", "target_id", "target_stage"), [
    ("structure_failure", "creative:storyboard", "storyboard"),
    ("asset_failure", "creative:asset_graph", "asset_graph"),
    ("shot_failure", "creative:shot_contract:SHOT", "shot_contract"),
    ("generation_failure", "creative:video_candidate:VID",
     "video_candidate"),
])
def test_repair_classification_routes_explicit_failure_class(
        failure_class: str, target_id: str, target_stage: str) -> None:
    from src.agentic_video.creative_pipeline.repair.router import (
        build_repair_classification, failure_from_repair_classification,
        route_repair,
    )

    classification = build_repair_classification(
        failure_class=failure_class,
        source_evaluation_ref={
            "artifact_id": "creative:generation_scorecard:X",
            "sha": "1" * 64},
        failed_artifact_ref={
            "artifact_id": "creative:video_candidate:VID",
            "sha": "2" * 64},
        repair_target_ref={"artifact_id": target_id, "sha": "3" * 64},
        reason_codes=[f"{failure_class}_detected"])
    failure = failure_from_repair_classification(classification)
    plan = route_repair([failure])

    assert classification["failure_class"] == failure_class
    assert classification["repair_target_stage"] == target_stage
    assert plan["target_ref"] == classification["repair_target_ref"]
    assert plan["target_stage"] == target_stage


def test_repair_classification_rejects_wrong_target_layer() -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.repair.router import (
        build_repair_classification,
    )

    with pytest.raises(ContractError,
                       match="repair_classification_target_invalid"):
        build_repair_classification(
            failure_class="asset_failure",
            source_evaluation_ref={
                "artifact_id": "creative:generation_scorecard:X",
                "sha": "1" * 64},
            failed_artifact_ref={
                "artifact_id": "creative:video_candidate:VID",
                "sha": "2" * 64},
            repair_target_ref={
                "artifact_id": "creative:shot_contract:SHOT",
                "sha": "3" * 64},
            reason_codes=["asset_failure_detected"])


def test_r2e_blocks_on_stale_wave5_candidate_pool(tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.orchestrator import PipelineBlocked

    workspace, orchestrator, wave5 = _runtime(tmp_path, wave5=True)
    record, candidate, _ = _first_candidate(workspace, wave5)
    replacement = deepcopy(candidate)
    replacement["backend"]["seed"] = 999
    workspace.write_draft(record["artifact_id"], replacement)
    workspace.commit(record["artifact_id"])

    with pytest.raises(PipelineBlocked, match="r2e_parent_not_committed"):
        orchestrator.run_r2e()
