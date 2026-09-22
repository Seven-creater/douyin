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


def _runtime(tmp_path: Path):
    from src.agentic_video.creative_pipeline.fake_skills import (
        build_fake_registry,
    )
    from src.agentic_video.creative_pipeline.orchestrator import (
        CreativePipelineOrchestrator,
    )
    from src.agentic_video.workspace import Workspace

    workspace = Workspace(tmp_path / "wave2")
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    return workspace, orchestrator


def _fixtures() -> tuple[dict, dict]:
    from src.agentic_video.creative_pipeline.planning.story import (
        build_fake_story_blueprints,
    )
    from src.agentic_video.creative_pipeline.planning.theme import (
        build_fake_theme_candidates,
    )

    theme = build_fake_theme_candidates(_spec()["artifact_sha"])[0]
    story = build_fake_story_blueprints(_spec()["artifact_sha"], theme)[0]
    return theme, story


def test_wave2_materializes_20_to_5_to_25_to_3_without_models(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)

    result = orchestrator.run_wave2(_spec())

    assert result["status"] == "PASS"
    assert result["model_calls"] == 0
    assert result["call_counts"] == {"fake_theme": 1, "fake_story": 5}
    assert len(result["theme_selections"]) == 5
    assert len(result["story_blueprint_selections"]) == 3

    theme_pool = workspace.read_artifact("creative:theme_pool")["payload"]
    story_pool = workspace.read_artifact(
        "creative:story_blueprint_pool")["payload"]
    assert len(theme_pool["candidates"]) == 20
    assert len(story_pool["candidates"]) == 25
    assert [row["selected_candidate_id"]
            for row in result["theme_selections"]] == [
        f"FAKE_THEME_{index:02d}" for index in range(1, 6)]
    assert [row["selected_candidate_id"]
            for row in result["story_blueprint_selections"]] == [
        f"FAKE_BLUEPRINT_FAKE_THEME_{index:02d}_01"
        for index in range(1, 4)]

    for index in range(1, 6):
        name = "creative:theme_selection" if index == 1 \
            else f"creative:theme_selection:{index:02d}"
        assert workspace.effective_status(name) == "committed"
    for index in range(1, 4):
        name = "creative:story_blueprint_selection" if index == 1 \
            else f"creative:story_blueprint_selection:{index:02d}"
        assert workspace.effective_status(name) == "committed"


def test_each_blueprint_has_exact_selected_theme_lineage(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)
    orchestrator.run_wave2(_spec())

    for theme_index in range(1, 6):
        theme_id = f"FAKE_THEME_{theme_index:02d}"
        theme_name = f"creative:theme_candidate:{theme_id}"
        selection_name = "creative:theme_selection" if theme_index == 1 \
            else f"creative:theme_selection:{theme_index:02d}"
        for story_index in range(1, 6):
            blueprint_id = (
                f"FAKE_BLUEPRINT_{theme_id}_{story_index:02d}")
            name = f"creative:story_blueprint:{blueprint_id}"
            parents = {
                (row["artifact_id"], row["sha"])
                for row in workspace.get_provenance(name)["derived_from"]
            }
            assert (theme_name, workspace.get_sha(theme_name)) in parents
            assert (selection_name,
                    workspace.get_sha(selection_name)) in parents
            assert (_spec()["spec_id"], _spec()["artifact_sha"]) in parents


def test_wave2_structure_and_boundary_validators_reject_drift() -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.evaluation.structure import (
        validate_story_structure,
        validate_theme_structure,
    )

    theme, story = _fixtures()

    missing_relation = deepcopy(theme)
    del missing_relation["structure_bindings"]["R1_INFORMATION_UPDATE"]
    with pytest.raises(ContractError,
                       match="theme_structure_binding_keys_invalid"):
        validate_theme_structure(
            missing_relation, structure_sha=_spec()["artifact_sha"])

    same_states = deepcopy(theme)
    same_states["structure_bindings"]["I1_UPDATED_INTERPRETATION"] = \
        same_states["structure_bindings"]["I0_PRIOR_INTERPRETATION"]
    with pytest.raises(ContractError,
                       match="theme_information_roles_not_distinct"):
        validate_theme_structure(
            same_states, structure_sha=_spec()["artifact_sha"])

    reference_leak = deepcopy(theme)
    reference_leak["accepted_claims"] = ["S1_V01"]
    with pytest.raises(ContractError, match="reference_boundary_field"):
        validate_theme_structure(
            reference_leak, structure_sha=_spec()["artifact_sha"])

    drift = deepcopy(story)
    relation = drift["event_relations"][0]
    relation["updated_event_id"] = relation["prior_event_id"]
    with pytest.raises(ContractError,
                       match="story_information_roles_not_distinct"):
        validate_story_structure(
            drift,
            structure_sha=_spec()["artifact_sha"],
            theme_id=theme["theme_id"],
        )

    wrong_theme = deepcopy(story)
    wrong_theme["theme_id"] = "UNSELECTED_THEME"
    with pytest.raises(ContractError, match="story_theme_id_mismatch"):
        validate_story_structure(
            wrong_theme,
            structure_sha=_spec()["artifact_sha"],
            theme_id=theme["theme_id"],
        )


def test_wave2_selection_binds_pool_and_candidate_shas(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)
    result = orchestrator.run_wave2(_spec())

    pool_name = "creative:theme_pool"
    pool = workspace.read_artifact(pool_name)["payload"]
    candidates = {row["candidate_id"]: row for row in pool["candidates"]}
    for selection in result["theme_selections"]:
        selected = candidates[selection["selected_candidate_id"]]
        assert selection["pool_ref"] == {
            "artifact_id": pool_name,
            "sha": workspace.get_sha(pool_name),
        }
        assert selection["selected_candidate_sha"] == \
            selected["artifact_sha"]


def test_wave2_structure_change_invalidates_all_selections(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)
    orchestrator.run_wave2(_spec())

    workspace.write_draft("creative:structure_spec", {"changed": True})
    workspace.commit("creative:structure_spec")

    selection_names = [
        name for name in workspace.state["artifacts"]
        if "selection" in name
    ]
    assert len(selection_names) == 8
    assert all(workspace.effective_status(name) == "stale"
               for name in selection_names)
