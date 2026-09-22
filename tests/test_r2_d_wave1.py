from __future__ import annotations

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

    workspace = Workspace(tmp_path / "wave1")
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    return workspace, orchestrator


def test_skill_spec_extension_is_backward_compatible_and_enforced(
        tmp_path: Path) -> None:
    from src.agentic_video.skills.registry import (
        SkillBlocked,
        SkillRegistry,
        SkillSpec,
    )
    from src.agentic_video.workspace import Workspace

    legacy = SkillSpec("legacy", "legacy skill")
    assert legacy.skill_version == "legacy"
    assert legacy.permission_profile == "workspace_legacy"
    assert legacy.max_calls is None

    registry = SkillRegistry()
    registry.register(SkillSpec(
        "bounded", "bounded fixture", permission_profile="payload_only",
        max_calls=1, skill_version="1.0.0",
        input_schema_versions=["input_v1"],
        output_schema_version="output_v1",
        package_sha="a" * 64,
        prompt_or_instruction_sha="b" * 64,
    ))
    workspace = Workspace(tmp_path / "guards")
    registry.validate_action(
        {"skill": "bounded"}, workspace,
        allowed_permission_profiles={"payload_only"}, calls_made=0)
    with pytest.raises(SkillBlocked, match="permission_profile"):
        registry.validate_action(
            {"skill": "bounded"}, workspace,
            allowed_permission_profiles={"network"}, calls_made=0)
    with pytest.raises(SkillBlocked, match="call_budget"):
        registry.validate_action(
            {"skill": "bounded"}, workspace,
            allowed_permission_profiles={"payload_only"}, calls_made=1)


def test_wave1_runs_spec_to_story_selection_with_zero_model_calls(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)

    result = orchestrator.run_wave1(
        _spec(), user_brief={"purpose": "contract smoke test"})

    assert result["status"] == "PASS"
    assert result["model_calls"] == 0
    assert result["structure_public_sha"] == _spec()["artifact_sha"]
    assert result["call_counts"] == {"fake_theme": 1, "fake_story": 1}
    assert result["theme_selection"]["selected_candidate_id"] \
        == "FAKE_THEME_01"
    assert result["story_blueprint_selection"]["selected_candidate_id"] \
        == "FAKE_BLUEPRINT_FAKE_THEME_01_01"

    expected = {
        "creative:structure_spec",
        "creative:theme_validation",
        "creative:theme_pool",
        "creative:theme_selection",
        "creative:story_blueprint_validation",
        "creative:story_blueprint_pool",
        "creative:story_blueprint_selection",
    }
    expected.update(
        f"creative:theme_candidate:FAKE_THEME_{index:02d}"
        for index in range(1, 21))
    expected.update(
        "creative:story_blueprint:"
        f"FAKE_BLUEPRINT_FAKE_THEME_01_{index:02d}"
        for index in range(1, 6))
    assert set(workspace.state["artifacts"]) == expected
    assert all(workspace.effective_status(name) == "committed"
               for name in expected)
    assert {row["name"] for row in result["lineage_report"]["artifacts"]} \
        == expected

    serialized = json.dumps(
        {name: workspace.read_artifact(name) for name in expected},
        ensure_ascii=False,
    )
    for forbidden in (
        "reference_video", "accepted_claims", "narrative_interpretation",
        "audit_annex", "creative_dna",
    ):
        assert forbidden not in serialized


def test_wave1_candidates_have_distinct_files_and_exact_lineage(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)
    result = orchestrator.run_wave1(_spec())

    first = "creative:theme_candidate:FAKE_THEME_01"
    second = "creative:theme_candidate:FAKE_THEME_02"
    first_path = workspace._stage_dir(first) / "v1.json"
    second_path = workspace._stage_dir(second) / "v1.json"
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    assert workspace.read_artifact(first) != workspace.read_artifact(second)

    selection = result["theme_selection"]
    pool_envelope = workspace.read_artifact("creative:theme_pool")
    pool = pool_envelope["payload"]
    selected = next(row for row in pool["candidates"]
                    if row["candidate_id"]
                    == selection["selected_candidate_id"])
    assert selection["selected_candidate_sha"] == selected["artifact_sha"]
    provenance = workspace.get_provenance("creative:theme_selection")
    parent_pairs = {(row["artifact_id"], row["sha"])
                    for row in provenance["derived_from"]}
    assert ("creative:theme_pool", workspace.get_sha(
        "creative:theme_pool")) in parent_pairs
    assert (selected["artifact_id"], selected["artifact_sha"]) in parent_pairs


def test_wave1_call_budget_blocks_a_second_skill_call(tmp_path: Path) -> None:
    from src.agentic_video.skills.registry import SkillBlocked

    _, orchestrator = _runtime(tmp_path)
    orchestrator.run_wave1(_spec())

    with pytest.raises(SkillBlocked, match="call_budget"):
        orchestrator._execute_skill(
            "fake_theme", creative_structure_spec=_spec(), user_brief={})


def test_structure_change_invalidates_entire_creative_chain(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)
    orchestrator.run_wave1(_spec())

    workspace.write_draft("creative:structure_spec", {"changed": True})
    workspace.commit("creative:structure_spec")

    assert workspace.effective_status("creative:theme_selection") == "stale"
    assert workspace.effective_status(
        "creative:story_blueprint_selection") == "stale"


def test_story_validator_rejects_non_distinct_information_roles() -> None:
    from src.agentic_video.creative_pipeline.contracts import (
        ContractError,
        validate_story_blueprint,
    )
    from src.agentic_video.creative_pipeline.fake_skills import (
        build_fake_registry,
    )

    registry = build_fake_registry()
    skill = registry.get("fake_story")
    # Use the deterministic function directly only to construct a mutation
    # fixture; no model or external dependency is involved.
    theme_skill = registry.get("fake_theme")
    theme = theme_skill.execute_fn(
        None, creative_structure_spec=_spec(), user_brief={})["candidates"][0]
    blueprint = skill.execute_fn(
        None, creative_structure_spec=_spec(), selected_theme=theme
    )["candidates"][0]
    relation = blueprint["event_relations"][0]
    relation["updated_event_id"] = relation["prior_event_id"]

    with pytest.raises(ContractError,
                       match="story_information_roles_not_distinct"):
        validate_story_blueprint(
            blueprint,
            structure_sha=_spec()["artifact_sha"],
            theme_id=theme["theme_id"],
        )
