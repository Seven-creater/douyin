from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPEC = (
    ROOT / "experiments/creative_structure_minimality/r2_c5/"
    "creative_structure_spec_v1.json")


def _spec() -> dict:
    return json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))


def _format() -> dict:
    return {
        "schema_version": "screenplay_format_constraints_v1",
        "target_duration_s": 30.0,
        "min_duration_s": 25.0,
        "max_duration_s": 35.0,
        "max_scenes": 3,
        "max_characters": 4,
        "max_props": 3,
    }


def _runtime(tmp_path: Path, *, wave4: bool = False):
    from src.agentic_video.creative_pipeline.fake_skills import (
        build_fake_registry,
    )
    from src.agentic_video.creative_pipeline.orchestrator import (
        CreativePipelineOrchestrator,
    )
    from src.agentic_video.workspace import Workspace

    workspace = Workspace(tmp_path / "wave4")
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    orchestrator.run_wave2(_spec())
    orchestrator.run_wave3(_spec(), format_constraints=_format())
    result = orchestrator.run_wave4(_spec()) if wave4 else None
    return workspace, orchestrator, result


def _payloads(tmp_path: Path):
    workspace, _, _ = _runtime(tmp_path, wave4=True)
    screenplay_pool = workspace.read_artifact(
        "creative:screenplay_pool")["payload"]
    selection = workspace.read_artifact(
        "creative:screenplay_selection")["payload"]
    record = next(row for row in screenplay_pool["candidates"]
                  if row["candidate_id"]
                  == selection["selected_candidate_id"])
    screenplay = workspace.read_artifact(record["artifact_id"])["payload"]
    asset_graph = workspace.read_artifact("creative:asset_graph")["payload"]
    shot_plan = workspace.read_artifact("creative:shot_plan")["payload"]
    storyboard = workspace.read_artifact("creative:storyboard")["payload"]
    return workspace, record, screenplay, asset_graph, shot_plan, storyboard


def test_wave4_builds_text_only_production_chain(tmp_path: Path) -> None:
    workspace, _, result = _runtime(tmp_path, wave4=True)

    assert result["status"] == "PASS"
    assert result["fixture_only"] is True
    assert result["model_calls"] == 0
    assert result["media_generated"] is False
    expected = {
        "asset_graph_ref": "creative:asset_graph",
        "shot_plan_ref": "creative:shot_plan",
        "storyboard_ref": "creative:storyboard",
    }
    for key, name in expected.items():
        assert result[key] == {
            "artifact_id": name, "sha": workspace.get_sha(name)}
        assert workspace.effective_status(name) == "committed"

    asset_graph = workspace.read_artifact("creative:asset_graph")["payload"]
    shot_plan = workspace.read_artifact("creative:shot_plan")["payload"]
    storyboard = workspace.read_artifact("creative:storyboard")["payload"]
    assert asset_graph["schema_version"] == "asset_graph_v1"
    assert shot_plan["schema_version"] == "shot_plan_v1"
    assert storyboard["schema_version"] == "storyboard_v1"
    assert len(shot_plan["shots"]) == len(storyboard["boards"]) == 3
    assert {role for shot in shot_plan["shots"]
            for role in shot["structure_trace"]["structure_roles"]} == {
        "I0_PRIOR_INTERPRETATION", "E1_NEW_INFORMATION",
        "I1_UPDATED_INTERPRETATION",
    }
    assert all("R1_INFORMATION_UPDATE"
               in shot["structure_trace"]["relation_ids"]
               for shot in shot_plan["shots"])
    traces = workspace.recent_trace(3)
    assert [row["action"]["adapter"] for row in traces] == [
        "fake_asset_graph_adapter", "fake_shot_plan_adapter",
        "fake_storyboard_adapter",
    ]
    assert all(row["tests"]["model_calls"] == 0
               and row["tests"]["media_generated"] is False
               for row in traces)


def test_wave4_lineage_binds_each_immediate_parent_sha(tmp_path: Path) -> None:
    workspace, record, _, _, _, _ = _payloads(tmp_path)

    screenplay_name = record["artifact_id"]
    selection_name = "creative:screenplay_selection"
    expected = {
        "creative:asset_graph": {
            (screenplay_name, workspace.get_sha(screenplay_name)),
            (selection_name, workspace.get_sha(selection_name)),
        },
        "creative:shot_plan": {
            (screenplay_name, workspace.get_sha(screenplay_name)),
            ("creative:asset_graph",
             workspace.get_sha("creative:asset_graph")),
        },
        "creative:storyboard": {
            ("creative:shot_plan", workspace.get_sha("creative:shot_plan")),
            ("creative:asset_graph",
             workspace.get_sha("creative:asset_graph")),
        },
    }
    for name, parents in expected.items():
        actual = {(row["artifact_id"], row["sha"])
                  for row in workspace.get_provenance(name)["derived_from"]}
        assert actual == parents


@pytest.mark.parametrize(("mutation", "reason"), [
    ("asset_missing", "production_location_completeness"),
    ("identity", "production_character_identity_invalid"),
    ("shot_trace", "production_shot_trace_mismatch"),
    ("shot_parent", "production_shot_asset_sha_mismatch"),
    ("storyboard_source", "storyboard_source_shot_sha_mismatch"),
    ("storyboard_assets", "storyboard_asset_refs_mismatch"),
])
def test_wave4_validators_reject_contract_drift(
        tmp_path: Path, mutation: str, reason: str) -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.production.asset_validator import (
        validate_production_asset_graph,
    )
    from src.agentic_video.creative_pipeline.production.shot_validator import (
        validate_production_shot_plan,
    )
    from src.agentic_video.creative_pipeline.production.storyboard import (
        validate_storyboard,
    )

    workspace, record, screenplay, asset_graph, shot_plan, storyboard = \
        _payloads(tmp_path)
    asset_graph = deepcopy(asset_graph)
    shot_plan = deepcopy(shot_plan)
    storyboard = deepcopy(storyboard)
    if mutation == "asset_missing":
        asset_graph["assets"] = [row for row in asset_graph["assets"]
                                 if row["type"] != "location"]
    elif mutation == "identity":
        character = next(row for row in asset_graph["assets"]
                         if row["type"] == "character")
        character["immutable"] = {}
    elif mutation == "shot_trace":
        shot_plan["shots"][0]["structure_trace"]["structure_roles"] = []
    elif mutation == "shot_parent":
        shot_plan["asset_graph_sha"] = "0" * 64
    elif mutation == "storyboard_source":
        storyboard["boards"][0]["source_shot_sha"] = "0" * 64
    else:
        storyboard["boards"][0]["reference_asset_ids"] = ["UNKNOWN_ASSET"]

    with pytest.raises(ContractError, match=reason):
        if mutation.startswith("asset") or mutation == "identity":
            validate_production_asset_graph(
                asset_graph, screenplay=screenplay,
                screenplay_sha=record["artifact_sha"])
        elif mutation.startswith("shot"):
            validate_production_shot_plan(
                shot_plan, screenplay=screenplay,
                screenplay_sha=record["artifact_sha"],
                asset_graph=asset_graph,
                asset_graph_sha=workspace.get_sha("creative:asset_graph"))
        else:
            validate_storyboard(
                storyboard, screenplay=screenplay,
                screenplay_sha=record["artifact_sha"],
                asset_graph=asset_graph,
                asset_graph_sha=workspace.get_sha("creative:asset_graph"),
                shot_plan=shot_plan,
                shot_plan_sha=workspace.get_sha("creative:shot_plan"))


def test_asset_boundary_rejects_reference_context_without_echo(tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.production.asset_validator import (
        validate_production_asset_graph,
    )

    _, record, screenplay, asset_graph, _, _ = _payloads(tmp_path)
    asset_graph["accepted_claims"] = ["private source wording"]
    with pytest.raises(ContractError) as error:
        validate_production_asset_graph(
            asset_graph, screenplay=screenplay,
            screenplay_sha=record["artifact_sha"])
    assert error.value.reason_code == "reference_boundary_field"
    assert "private source wording" not in str(error.value)


def test_screenplay_change_stales_entire_production_chain(tmp_path: Path) -> None:
    workspace, record, _, _, _, _ = _payloads(tmp_path)
    workspace.write_draft(record["artifact_id"], {"changed": True})
    workspace.commit(record["artifact_id"])

    assert workspace.effective_status("creative:asset_graph") == "stale"
    assert workspace.effective_status("creative:shot_plan") == "stale"
    assert workspace.effective_status("creative:storyboard") == "stale"


def test_stale_screenplay_selection_blocks_before_production_write(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.orchestrator import PipelineBlocked

    workspace, orchestrator, _ = _runtime(tmp_path)
    pool = workspace.read_artifact("creative:screenplay_pool")["payload"]
    selection = workspace.read_artifact(
        "creative:screenplay_selection")["payload"]
    selected = next(row for row in pool["candidates"]
                    if row["candidate_id"]
                    == selection["selected_candidate_id"])
    workspace.write_draft(selected["artifact_id"], {"changed": True})
    workspace.commit(selected["artifact_id"])

    with pytest.raises(PipelineBlocked,
                       match="production_parent_not_committed"):
        orchestrator.run_wave4(_spec())
    assert "creative:asset_graph" not in workspace.state["artifacts"]
    assert "creative:shot_plan" not in workspace.state["artifacts"]
    assert "creative:storyboard" not in workspace.state["artifacts"]
