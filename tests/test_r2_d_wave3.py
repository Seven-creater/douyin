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


def _runtime(tmp_path: Path):
    from src.agentic_video.creative_pipeline.fake_skills import (
        build_fake_registry,
    )
    from src.agentic_video.creative_pipeline.orchestrator import (
        CreativePipelineOrchestrator,
    )
    from src.agentic_video.workspace import Workspace

    workspace = Workspace(tmp_path / "wave3")
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    orchestrator.run_wave2(_spec())
    return workspace, orchestrator


def _candidate_fixture(tmp_path: Path):
    from src.agentic_video.creative_pipeline.writing.adapter import (
        FakeScreenplaySkill,
        build_screenplay_request,
    )

    workspace, _ = _runtime(tmp_path)
    pool = workspace.read_artifact(
        "creative:story_blueprint_pool")["payload"]
    selection = workspace.read_artifact(
        "creative:story_blueprint_selection")["payload"]
    record = next(row for row in pool["candidates"]
                  if row["candidate_id"]
                  == selection["selected_candidate_id"])
    envelope = workspace.read_artifact(record["artifact_id"])
    request = build_screenplay_request(
        creative_structure_spec=_spec(),
        blueprint_envelope=envelope,
        blueprint_sha=record["artifact_sha"],
        format_constraints=_format(),
    )
    return request, FakeScreenplaySkill().run(request)


def test_wave3_compiles_three_blueprints_and_selects_one_without_models(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)

    result = orchestrator.run_wave3(
        _spec(), format_constraints=_format())

    assert result["status"] == "PASS"
    assert result["fixture_only"] is True
    assert result["model_calls"] == 0
    assert result["quality_evaluation"] == "not_run"
    assert result["semantic_entailment"] == "not_run"
    assert result["call_counts"] == {
        "fake_theme": 1, "fake_story": 5, "fake_screenplay": 3}
    pool = workspace.read_artifact("creative:screenplay_pool")["payload"]
    assert len(pool["candidates"]) == 3
    assert result["screenplay_selection"]["selected_candidate_id"] \
        == "FAKE_SCREENPLAY_FAKE_BLUEPRINT_FAKE_THEME_01_01"
    assert workspace.effective_status("creative:screenplay_selection") \
        == "committed"
    assert result["committed_screenplay_ref"]["sha"] \
        == result["screenplay_selection"]["selected_candidate_sha"]

    traces = workspace.recent_trace(3)
    assert len(traces) == 3
    assert all(row["tests"]["quality_evaluation"] == "not_run"
               and row["tests"]["model_calls"] == 0 for row in traces)
    blind = traces[0]["result"]["blind_evaluation_request"]
    assert set(blind["screenplay"]) == {
        "target_duration_s", "characters", "scenes", "beats",
        "dialogue_or_text_cues", "production_requirements",
    }
    assert "screenplay_id" not in blind["screenplay"]
    assert "blueprint_event_ids" not in blind["screenplay"]["beats"][0]
    serialized = json.dumps(traces, ensure_ascii=False)
    for forbidden in ("reference_video", "accepted_claims", "creative_dna",
                      "narrative_interpretation", "audit_annex"):
        assert forbidden not in serialized


def test_each_screenplay_binds_exact_blueprint_selection_and_format_shas(
        tmp_path: Path) -> None:
    workspace, orchestrator = _runtime(tmp_path)
    orchestrator.run_wave3(_spec(), format_constraints=_format())

    pool = workspace.read_artifact("creative:screenplay_pool")["payload"]
    constraints_name = "creative:screenplay_format_constraints"
    for index, record in enumerate(pool["candidates"], 1):
        screenplay = workspace.read_artifact(record["artifact_id"])["payload"]
        blueprint_name = (
            f"creative:story_blueprint:{screenplay['blueprint_id']}")
        selection_name = "creative:story_blueprint_selection" + (
            "" if index == 1 else f":{index:02d}")
        parents = {
            (row["artifact_id"], row["sha"])
            for row in workspace.get_provenance(
                record["artifact_id"])["derived_from"]
        }
        assert (_spec()["spec_id"], _spec()["artifact_sha"]) in parents
        assert (blueprint_name, workspace.get_sha(blueprint_name)) in parents
        assert (selection_name, workspace.get_sha(selection_name)) in parents
        assert (constraints_name,
                workspace.get_sha(constraints_name)) in parents
        assert screenplay["story_blueprint_sha"] \
            == workspace.get_sha(blueprint_name)


@pytest.mark.parametrize(("mutation", "reason"), [
    ("coverage", "screenplay_blueprint_coverage_missing"),
    ("trace", "screenplay_trace_roles_invalid"),
    ("duration", "screenplay_duration_sum_mismatch"),
    ("production", "screenplay_production_coverage_mismatch"),
    ("reference", "reference_boundary_field"),
])
def test_screenplay_validator_rejects_invalid_contracts(
        tmp_path: Path, mutation: str, reason: str) -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.writing.validator import (
        validate_screenplay,
    )

    request, candidate = _candidate_fixture(tmp_path)
    candidate = deepcopy(candidate)
    if mutation == "coverage":
        candidate["beats"].pop()
        candidate["scenes"][0]["beat_ids"].pop()
    elif mutation == "trace":
        candidate["structure_trace"]["I1_UPDATED_INTERPRETATION"] = \
            candidate["structure_trace"]["I0_PRIOR_INTERPRETATION"]
    elif mutation == "duration":
        candidate["beats"][0]["duration_s"] += 1
    elif mutation == "production":
        candidate["production_requirements"]["locations"] = ["wrong place"]
    else:
        candidate["accepted_claims"] = ["S1_V01"]

    with pytest.raises(ContractError, match=reason):
        validate_screenplay(
            candidate,
            blueprint=request["story_blueprint"],
            blueprint_sha=request["story_blueprint_sha"],
            structure_sha=request[
                "creative_structure_spec"]["artifact_sha"],
            format_constraints=request["format_constraints"],
        )


def test_writer_request_rejects_reference_fields_without_echoing_content(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.contracts import ContractError
    from src.agentic_video.creative_pipeline.writing.adapter import (
        build_screenplay_request,
    )

    workspace, _ = _runtime(tmp_path)
    pool = workspace.read_artifact(
        "creative:story_blueprint_pool")["payload"]
    record = pool["candidates"][0]
    envelope = deepcopy(workspace.read_artifact(record["artifact_id"]))
    envelope["payload"]["accepted_claims"] = ["secret reference wording"]
    # Preserve internal envelope consistency so the boundary is the failing gate.
    from src.agentic_video.manifest import json_hash
    envelope["content_sha"] = json_hash(envelope["payload"])
    envelope_sha = json_hash(envelope)

    with pytest.raises(ContractError) as error:
        build_screenplay_request(
            creative_structure_spec=_spec(),
            blueprint_envelope=envelope,
            blueprint_sha=envelope_sha,
            format_constraints=_format(),
        )
    assert error.value.reason_code == "reference_boundary_field"
    assert "secret reference wording" not in str(error.value)


def test_stale_blueprint_selection_blocks_all_writer_side_effects(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.orchestrator import PipelineBlocked

    workspace, orchestrator = _runtime(tmp_path)
    pool = workspace.read_artifact(
        "creative:story_blueprint_pool")["payload"]
    selected = workspace.read_artifact(
        "creative:story_blueprint_selection")["payload"]
    record = next(row for row in pool["candidates"]
                  if row["candidate_id"]
                  == selected["selected_candidate_id"])
    workspace.write_draft(record["artifact_id"], {"changed": True})
    workspace.commit(record["artifact_id"])

    with pytest.raises(PipelineBlocked,
                       match="screenplay_parent_not_committed"):
        orchestrator.run_wave3(_spec(), format_constraints=_format())
    assert "fake_screenplay" not in orchestrator.call_counts
    assert not any(name.startswith("creative:screenplay")
                   for name in workspace.state["artifacts"])
