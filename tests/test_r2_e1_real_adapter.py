from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPEC = (
    ROOT / "experiments/creative_structure_minimality/r2_c5/"
    "creative_structure_spec_v1.json")
FORMAT_CONSTRAINTS = {
    "schema_version": "screenplay_format_constraints_v1",
    "target_duration_s": 30.0,
    "min_duration_s": 25.0,
    "max_duration_s": 35.0,
    "max_scenes": 3,
    "max_characters": 4,
    "max_props": 3,
}


class StubVideoClient:
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id

    def generate(self, request: dict, output_path: Path) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            f"video:{self.client_id}:{request['seed']}".encode("utf-8"))
        return {"id": f"job-{self.client_id}",
                "output_path": str(output_path)}


class StubImageClient:
    def generate(self, *, prompt: str, seed: int, output_path: Path,
                 config: dict) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            f"image:{seed}:{prompt}:{config}".encode("utf-8"))
        return {"id": f"image-{seed}", "output_path": str(output_path)}


def _runtime(tmp_path: Path):
    from src.agentic_video.creative_pipeline.fake_skills import (
        build_fake_registry,
    )
    from src.agentic_video.creative_pipeline.orchestrator import (
        CreativePipelineOrchestrator,
    )
    from src.agentic_video.workspace import Workspace

    spec = json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    workspace = Workspace(tmp_path / "workspace")
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    orchestrator.run_wave2(spec)
    orchestrator.run_wave3(spec, format_constraints=FORMAT_CONSTRAINTS)
    orchestrator.run_wave4(spec)
    return workspace, orchestrator


def _payload(workspace, name: str) -> dict:
    return workspace.read_artifact(name)["payload"]


def test_real_video_acceptance_generates_three_by_three_without_selection(
        tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.generation.real import (
        RealVideoAdapter, file_sha256,
    )

    workspace, orchestrator = _runtime(tmp_path)
    upstream_names = (
        "creative:asset_graph", "creative:shot_plan", "creative:storyboard")
    upstream = {
        name: (workspace.get_sha(name),
               workspace.state["artifacts"][name]["active_version"],
               workspace.effective_status(name))
        for name in upstream_names
    }
    adapter = RealVideoAdapter(
        [StubVideoClient(str(index)) for index in range(3)],
        model_id="test/real-video", model_version="checkpoint-v1",
        model_sha="a" * 64, inference_steps=4)

    result = orchestrator.run_r2e1_real_video_acceptance(
        adapter, output_dir=tmp_path / "media")

    assert result == {**result, "status": "PASS", "shot_count": 3,
                      "candidate_count": 9, "model_calls": 9,
                      "media_generated": True,
                      "candidate_status": "generated",
                      "evaluation": "not_run", "selection": "not_run",
                      "full_video_generation": False}
    assert upstream == {
        name: (workspace.get_sha(name),
               workspace.state["artifacts"][name]["active_version"],
               workspace.effective_status(name))
        for name in upstream_names
    }
    assert not any(name.startswith("creative:real_video_selection")
                   for name in workspace.state["artifacts"])
    assert not any(name.startswith("creative:structure_evaluation:RVID_")
                   for name in workspace.state["artifacts"])

    roles = set()
    for shot in result["shot_results"]:
        roles.update(shot["structure_roles"])
        pool = _payload(workspace,
                        shot["candidate_pool_ref"]["artifact_id"])
        assert len(pool["candidates"]) == 3
        assert {row["status"] for row in pool["candidates"]} == {"generated"}
        for record in pool["candidates"]:
            candidate = _payload(workspace, record["artifact_id"])
            media = _payload(
                workspace,
                f"creative:generated_media:{candidate['candidate_id']}")
            experiment = _payload(
                workspace,
                f"creative:generation_experiment:{candidate['candidate_id']}")
            assert candidate["placeholder"]["generated"] is True
            assert candidate["placeholder"]["content_sha"] == file_sha256(
                Path(media["path"]))
            assert experiment["execution_mode"] == "real_model"
            assert experiment["model_id"] == "test/real-video"
            assert experiment["model_version"] == "checkpoint-v1"
            assert experiment["model_sha"] == "a" * 64
            assert experiment["input_artifact_refs"] == candidate["parent_refs"]
    assert roles == {
        "I0_PRIOR_INTERPRETATION", "E1_NEW_INFORMATION",
        "I1_UPDATED_INTERPRETATION",
    }


def test_real_image_adapter_preserves_candidate_schema(tmp_path: Path) -> None:
    from src.agentic_video.creative_pipeline.contracts import artifact_ref
    from src.agentic_video.creative_pipeline.generation.image import (
        IMAGE_CANDIDATE_FIELDS,
    )
    from src.agentic_video.creative_pipeline.generation.real import (
        RealImageAdapter, RealVideoAdapter, build_real_generation_request,
    )

    workspace, orchestrator = _runtime(tmp_path)
    video_adapter = RealVideoAdapter(
        [StubVideoClient(str(index)) for index in range(3)],
        model_id="test/real-video", model_version="v1",
        model_sha="b" * 64, inference_steps=4)
    result = orchestrator.run_r2e1_real_video_acceptance(
        video_adapter, output_dir=tmp_path / "video")
    contract_ref = result["shot_results"][0]["shot_contract_ref"]
    contract = _payload(workspace, contract_ref["artifact_id"])
    asset_ref = artifact_ref(
        "creative:asset_graph", str(workspace.get_sha("creative:asset_graph")))
    request = build_real_generation_request(
        media_kind="image", shot_contract=contract,
        shot_contract_ref=contract_ref, asset_graph_ref=asset_ref)
    adapter = RealImageAdapter(
        StubImageClient(), model_id="test/real-image",
        model_version="v1", model_sha="c" * 64)
    rows = adapter.generate(
        request, asset_graph=_payload(workspace, "creative:asset_graph"),
        output_dir=tmp_path / "image")

    assert len(rows) == 3
    assert all(set(row["candidate"]) == IMAGE_CANDIDATE_FIELDS
               for row in rows)
    assert all(row["candidate"]["backend"]["adapter_id"]
               == "real_image_adapter" for row in rows)
