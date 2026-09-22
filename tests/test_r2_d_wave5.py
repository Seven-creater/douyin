from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest


def _runtime(tmp_path: Path, *, wave5: bool = False):
    from src.agentic_video.creative_pipeline.fake_skills import (
        build_fake_registry,
    )
    from src.agentic_video.creative_pipeline.orchestrator import (
        CreativePipelineOrchestrator,
    )
    from src.agentic_video.workspace import Workspace

    workspace = Workspace(tmp_path / "wave5")
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    _seed_wave4(workspace, orchestrator)
    result = orchestrator.run_wave5() if wave5 else None
    return workspace, orchestrator, result


def _seed_wave4(workspace, orchestrator) -> None:
    """Seed the exact Wave 5 boundary without rebuilding Waves 2 and 3."""
    from src.agentic_video.creative_pipeline.contracts import artifact_ref
    from src.agentic_video.creative_pipeline.orchestrator import (
        WAVE4_PRODUCTION_POLICY_VERSION,
    )
    from src.agentic_video.manifest import json_hash

    assets = [{
        "asset_id": "CHAR_01", "type": "character", "tier": "A",
        "description": "canonical subject",
        "immutable": {"identity": "canonical CHAR_01"}, "mutable": {},
        "usage": ["BEAT_01", "BEAT_02", "BEAT_03"], "status": "active",
        "source_requirement": "CHAR_01",
        "identity_constraints": ["preserve canonical identity"],
        "visual_constraints": [],
    }, {
        "asset_id": "ENV_01", "type": "location", "tier": "A",
        "description": "test location", "immutable": {}, "mutable": {},
        "usage": ["BEAT_01", "BEAT_02", "BEAT_03"], "status": "active",
        "source_requirement": "test location", "identity_constraints": [],
        "visual_constraints": ["preserve spatial layout"],
    }]
    asset_graph = {
        "schema_version": "asset_graph_v1", "asset_graph_id": "ASSET_TEST",
        "screenplay_sha": "1" * 64, "assets": assets,
    }
    asset_name = "creative:asset_graph"
    asset_ref = orchestrator._write_envelope(
        asset_name, artifact_type="asset_graph",
        payload_schema_version="asset_graph_v1", payload=asset_graph,
        dependencies=[], parent_refs=[],
        producer=orchestrator._runtime_producer("wave5_test_asset"),
        selection_policy_version=WAVE4_PRODUCTION_POLICY_VERSION)

    roles = [
        "I0_PRIOR_INTERPRETATION", "E1_NEW_INFORMATION",
        "I1_UPDATED_INTERPRETATION",
    ]
    shots = []
    for index, role in enumerate(roles, 1):
        beat_id = f"BEAT_{index:02d}"
        shots.append({
            "shot_id": f"PROD_SHOT_{index:03d}", "beat_id": beat_id,
            "narrative_role": role, "duration_budget_s": [5.0, 5.0],
            "characters": ["CHAR_01"], "location": "ENV_01", "props": [],
            "wardrobe": [], "motion_refs": [],
            "camera": {"shot_size": "medium", "angle": "eye_level",
                       "movement": "static"},
            "camera_goal": "test", "start_state": {"CHAR_01": "before"},
            "end_state": {"CHAR_01": "after"},
            "visual_events": [f"perform {beat_id}"],
            "structure_trace": {
                "source_beat_id": beat_id, "structure_roles": [role],
                "relation_ids": ["R1_INFORMATION_UPDATE"],
            },
        })
    shot_plan = {
        "schema_version": "shot_plan_v1", "shot_plan_id": "SHOT_TEST",
        "screenplay_sha": "1" * 64, "asset_graph_sha": asset_ref["sha"],
        "shots": shots,
    }
    shot_name = "creative:shot_plan"
    shot_ref = orchestrator._write_envelope(
        shot_name, artifact_type="shot_plan",
        payload_schema_version="shot_plan_v1", payload=shot_plan,
        dependencies=[asset_name],
        parent_refs=[artifact_ref(asset_name, asset_ref["sha"])],
        producer=orchestrator._runtime_producer("wave5_test_shot"),
        selection_policy_version=WAVE4_PRODUCTION_POLICY_VERSION)
    storyboard = {
        "schema_version": "storyboard_v1", "storyboard_id": "BOARD_TEST",
        "shot_plan_sha": shot_ref["sha"], "asset_graph_sha": asset_ref["sha"],
        "boards": [{
            "shot_id": shot["shot_id"], "source_shot_sha": json_hash(shot),
            "frame_description": f"frame for {shot['shot_id']}",
            "camera": deepcopy(shot["camera"]), "lighting": "neutral",
            "composition": "contracted content",
            "reference_asset_ids": ["CHAR_01", "ENV_01"],
            "structure_trace": deepcopy(shot["structure_trace"]),
        } for shot in shots],
    }
    orchestrator._write_envelope(
        "creative:storyboard", artifact_type="storyboard",
        payload_schema_version="storyboard_v1", payload=storyboard,
        dependencies=[shot_name, asset_name],
        parent_refs=[artifact_ref(shot_name, shot_ref["sha"]),
                     artifact_ref(asset_name, asset_ref["sha"])],
        producer=orchestrator._runtime_producer("wave5_test_storyboard"),
        selection_policy_version=WAVE4_PRODUCTION_POLICY_VERSION)


def _payload(workspace, name: str) -> dict:
    return workspace.read_artifact(name)["payload"]


def test_wave5_builds_metadata_only_candidate_loops(tmp_path: Path) -> None:
    workspace, _, result = _runtime(tmp_path, wave5=True)

    assert result["status"] == "PASS"
    assert result["fixture_only"] is True
    assert result["model_calls"] == 0
    assert result["media_generated"] is False
    assert result["visual_quality"] == result["creative_quality"] == "not_run"
    assert len(result["shot_results"]) == 3
    for shot_result in result["shot_results"]:
        shot_id = shot_result["shot_id"]
        contract = _payload(workspace, f"creative:shot_contract:{shot_id}")
        assert contract["schema_version"] == "generation_shot_contract_v1"
        for kind in ("image", "video"):
            pool = _payload(workspace, f"creative:{kind}_pool:{shot_id}")
            selection = _payload(
                workspace, f"creative:{kind}_selection:{shot_id}")
            assert len(pool["candidates"]) == 2
            assert len(selection["scorecard_shas"]) == 1
            selected = next(row for row in pool["candidates"]
                            if row["candidate_id"]
                            == selection["selected_candidate_id"])
            candidate = _payload(workspace, selected["artifact_id"])
            assert candidate["backend"]["model_id"] == "none"
            assert candidate["placeholder"] == {
                **candidate["placeholder"],
                "materialization": "metadata_only", "generated": False,
            }
    coverage = _payload(workspace, "creative:wave5_structure_coverage")
    assert coverage["status"] == "PASS"
    assert set(coverage["actual_roles"]) == {
        "I0_PRIOR_INTERPRETATION", "E1_NEW_INFORMATION",
        "I1_UPDATED_INTERPRETATION",
    }


def test_wave5_preserves_wave4_artifact_versions_and_shas(tmp_path: Path) -> None:
    workspace, orchestrator, _ = _runtime(tmp_path)
    upstream = {
        name: (workspace.get_sha(name),
               workspace.state["artifacts"][name]["active_version"])
        for name in ("creative:asset_graph", "creative:shot_plan",
                     "creative:storyboard")}

    orchestrator.run_wave5()

    assert upstream == {
        name: (workspace.get_sha(name),
               workspace.state["artifacts"][name]["active_version"])
        for name in upstream}


@pytest.mark.parametrize(("mutation", "failed_check"), [
    ("shot", "shot_contract"),
    ("asset", "asset_sha"),
    ("parent", "parent_sha"),
    ("trace", "structure_trace_coverage"),
])
def test_media_evaluation_reports_each_deterministic_failure(
        tmp_path: Path, mutation: str, failed_check: str) -> None:
    from src.agentic_video.creative_pipeline.evaluation.media import (
        evaluate_media_candidate,
    )
    from src.agentic_video.creative_pipeline.generation.adapter import (
        build_image_generation_request,
    )
    from src.agentic_video.creative_pipeline.generation.image import (
        FakeImageGenerator,
    )

    workspace, _, _ = _runtime(tmp_path, wave5=True)
    shot_id = "PROD_SHOT_001"
    contract_name = f"creative:shot_contract:{shot_id}"
    contract = _payload(workspace, contract_name)
    request = build_image_generation_request(
        shot_contract=contract,
        shot_contract_ref={"artifact_id": contract_name,
                           "sha": workspace.get_sha(contract_name)},
        asset_graph_ref={"artifact_id": "creative:asset_graph",
                         "sha": workspace.get_sha("creative:asset_graph")})
    candidate = FakeImageGenerator().generate(request)[0]
    candidate = deepcopy(candidate)
    if mutation == "shot":
        candidate["shot_id"] = "WRONG_SHOT"
    elif mutation == "asset":
        candidate["asset_graph_sha"] = "0" * 64
    elif mutation == "parent":
        candidate["parent_refs"][0]["sha"] = "0" * 64
    else:
        candidate["structure_trace"]["structure_roles"] = []
    report = evaluate_media_candidate(
        candidate=candidate,
        candidate_ref={"artifact_id": "creative:image_candidate:TEST",
                       "sha": "f" * 64},
        request=request)

    assert report["status"] == "FAIL"
    checks = {row["check"]: row for row in report["checks"]}
    assert checks[failed_check]["passed"] is False
    assert report["visual_quality"] == "not_run"
    assert report["creative_quality"] == "not_run"


def test_structure_coverage_requires_all_selected_roles(tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.evaluation.media import (
        build_structure_coverage_report,
    )

    workspace, _, result = _runtime(tmp_path, wave5=True)
    contracts = [_payload(
        workspace, row["shot_contract_ref"]["artifact_id"])
        for row in result["shot_results"]]
    candidates = []
    for row in result["shot_results"]:
        selection = _payload(
            workspace, row["video_selection_ref"]["artifact_id"])
        pool = _payload(workspace, row["video_pool_ref"]["artifact_id"])
        selected = next(item for item in pool["candidates"]
                        if item["candidate_id"]
                        == selection["selected_candidate_id"])
        candidates.append(_payload(workspace, selected["artifact_id"]))

    report = build_structure_coverage_report(
        selected_candidates=candidates[:-1], shot_contracts=contracts)
    assert report["status"] == "FAIL"
    assert report["reason_codes"] == [
        "selected_structure_trace_coverage_incomplete"]


def test_wave5_blocks_before_writing_when_wave4_parent_revoked(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.orchestrator import PipelineBlocked

    workspace, orchestrator, _ = _runtime(tmp_path)
    workspace.revoke("creative:storyboard", "test")
    with pytest.raises(PipelineBlocked,
                       match="generation_parent_not_committed"):
        orchestrator.run_wave5()
    assert not any(name.startswith("creative:shot_contract:")
                   for name in workspace.state["artifacts"])


def test_repair_creates_new_version_and_stales_descendants(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.generation.image import (
        placeholder_content_sha, validate_image_candidate_schema,
    )
    from src.agentic_video.creative_pipeline.repair.router import (
        commit_repair_version, route_repair,
    )

    workspace, _, result = _runtime(tmp_path, wave5=True)
    shot_id = result["shot_results"][0]["shot_id"]
    pool = _payload(workspace, f"creative:image_pool:{shot_id}")
    selection = _payload(workspace, f"creative:image_selection:{shot_id}")
    selected = next(row for row in pool["candidates"]
                    if row["candidate_id"]
                    == selection["selected_candidate_id"])
    target = selected["artifact_id"]
    failure = {
        "schema_version": "repair_failure_v1",
        "failed_artifact_ref": {
            "artifact_id": target, "sha": workspace.get_sha(target)},
        "stage": "image_candidate",
        "reason_codes": ["image_candidate_external_quality_failure"],
    }
    plan = route_repair([failure])
    replacement = deepcopy(_payload(workspace, target))
    replacement["backend"]["seed"] = 99
    replacement["placeholder"]["content_sha"] = placeholder_content_sha(
        media_kind="image", request_sha=replacement["request_sha"],
        candidate_id=replacement["candidate_id"], seed=99)

    committed = commit_repair_version(
        workspace, plan, replacement,
        validator=validate_image_candidate_schema)

    assert committed["version"] == "v2"
    assert workspace.effective_status(target) == "committed"
    assert workspace.effective_status(
        f"creative:image_selection:{shot_id}") == "stale"
    assert workspace.effective_status(
        f"creative:video_selection:{shot_id}") == "stale"
    assert f"creative:video_selection:{shot_id}" \
        in committed["downstream_stale"]


def test_repair_router_selects_earliest_reported_stage() -> None:
    from src.agentic_video.creative_pipeline.repair.router import route_repair

    failures = [{
        "schema_version": "repair_failure_v1",
        "failed_artifact_ref": {
            "artifact_id": "creative:video_candidate:V", "sha": "a" * 64},
        "stage": "video_candidate", "reason_codes": ["video_failed"],
    }, {
        "schema_version": "repair_failure_v1",
        "failed_artifact_ref": {
            "artifact_id": "creative:shot_contract:S", "sha": "b" * 64},
        "stage": "shot_contract", "reason_codes": ["contract_failed"],
    }]
    plan = route_repair(failures)
    assert plan["target_ref"] == failures[1]["failed_artifact_ref"]
    assert plan["action"] == "create_new_version"
    assert plan["downstream_policy"] == "mark_stale"
