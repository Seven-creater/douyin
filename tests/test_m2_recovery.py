"""Regressions from the real M2-A master failure on 2026-09-21."""
import json
from types import SimpleNamespace

import pytest

from src.agentic_video.agent_v4 import agent_loop
from src.agentic_video.no_progress import NoProgressDetector
from src.agentic_video.skills.registry import SkillRegistry, SkillSpec
from src.agentic_video.workspace import Workspace


MASTER = "asset:C0_master"
FAILURE = {"passed": False, "validators_run": ["test_character_master"],
           "failures": [{"check": "background", "detail": "visible light stand"}]}


def recovery_fixture(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "run")
    ws.state["goal"] = {"target_artifact": MASTER, "required_status": "committed"}
    version = ws.write_draft(MASTER, {"bad": True})
    ws.record_trace(1, {"skill": "generate_character_master"},
                    {"artifact": MASTER, "version": version}, FAILURE)
    reg = SkillRegistry()
    received = []

    def repair(workspace, **kw):
        received.append(kw)
        version = workspace.write_draft(MASTER, {"bad": False})
        return {"artifact": MASTER, "version": version, "action": "master_repaired"}

    reg.register(SkillSpec(
        "repair_character_master", "repair", outputs=[MASTER],
        validators=["test_character_master"], cost_class="expensive_gpu",
        execute_fn=repair))
    reg.register(SkillSpec(
        "inspect_character_asset", "inspect", execute_fn=lambda ws, **kw: {
            "artifact": "asset:C0", "action": "inspected", "master": {"bad": True}}))

    def validate(action, registry, workspace, runner):
        return {"passed": True, "validators_run": (
            ["test_character_master"] if action["skill"].startswith("repair_") else [])}

    monkeypatch.setattr("src.agentic_video.agent_v4._run_validator_for_action", validate)
    return ws, reg, received


def payload(prompt):
    return json.loads(prompt.rsplit("输入：", 1)[1])


def test_failure_survives_inspect_and_resume(tmp_path, monkeypatch):
    ws, reg, received = recovery_fixture(tmp_path, monkeypatch)
    seen = []

    class Runner:
        def ask(self, prompt, **kw):
            data = payload(prompt)
            seen.append(data)
            assert data["recent_failure"]["failures"] == FAILURE["failures"]
            return SimpleNamespace(text=json.dumps({"skill": (
                "inspect_character_asset" if len(seen) == 1 else "repair_character_master")}))

    runner = Runner()
    agent_loop(ws, reg, controller_runner=runner, max_steps=1, budget="gpu")
    assert ws.get_status(MASTER) == "draft"
    result = agent_loop(Workspace(ws.root), reg, controller_runner=runner,
                        max_steps=2, budget="gpu")
    assert result["goal_satisfied"]
    assert received[0]["test_failures"] == FAILURE["failures"]
    assert seen[1]["recent_trace"][-1]["observation"]["master"] == {"bad": True}


@pytest.mark.parametrize("ignore_restriction", [False, True])
def test_stagnation_recovery_is_enforced_and_bounded(
        tmp_path, monkeypatch, ignore_restriction):
    ws, reg, received = recovery_fixture(tmp_path, monkeypatch)
    recovery_calls = []

    class Runner:
        def ask(self, prompt, **kw):
            data = payload(prompt)
            skill = "inspect_character_asset"
            if data["recovery_mode"]:
                recovery_calls.append(data)
                assert [s["name"] for s in data["runnable_skills"]] == [
                    "repair_character_master"]
                if not ignore_restriction:
                    skill = "repair_character_master"
            return SimpleNamespace(text=json.dumps({"skill": skill}))

    result = agent_loop(ws, reg, controller_runner=Runner(), budget="gpu",
                        max_steps=12, no_progress_detector=NoProgressDetector(2))
    assert len(recovery_calls) == 1
    assert result["goal_satisfied"] is (not ignore_restriction)
    if ignore_restriction:
        assert result["stop_reason"] == "no_progress"
        assert received == []
        assert ws.recent_trace(2)[0]["result"]["action"] == "blocked"


def test_recovery_cannot_override_gpu_budget(tmp_path, monkeypatch):
    ws, reg, received = recovery_fixture(tmp_path, monkeypatch)

    class Runner:
        def ask(self, prompt, **kw):
            assert all(s["name"] != "repair_character_master"
                       for s in payload(prompt)["runnable_skills"])
            return SimpleNamespace(text='{"skill":"inspect_character_asset"}')

    result = agent_loop(ws, reg, controller_runner=Runner(), budget="cheap_text",
                        max_steps=10, no_progress_detector=NoProgressDetector(2))
    assert result["stop_reason"] == "no_progress"
    assert received == []


def test_acceptance_draft_and_stale_are_not_pass(tmp_path):
    from src.agentic_video.asset_studio.acceptance import build_acceptance_report
    from src.agentic_video.asset_studio.workspace_setup import prepare_m2a_workspace
    ws = Workspace(tmp_path / "run")
    prepare_m2a_workspace(ws)
    ws.write_draft("asset_graph", {"version": 1})
    ws.commit("asset_graph")
    ws.write_draft(MASTER, {"master": {"sha": "m1"}})
    ws.record_trace(1, {"skill": "repair_character_master", "target": MASTER},
                    {"action": "master_repaired", "target": "front"}, FAILURE)
    ws.write_draft("asset:C0", {"human_review": "approved"})
    report = build_acceptance_report(ws)
    assert report["master"]["status"] == "draft"
    assert report["human_review"] == "pending"
    assert report["repair_history"] == {"front": 1}
    assert report["recommended_reference_roles"] == {}
    ws.commit(MASTER)
    ws.write_draft("asset:C0_views", {"views": {"front": {"sha": "m1"}}})
    ws.commit("asset:C0_views")
    assert build_acceptance_report(ws)["master"]["status"] == "pass"
    # Simulate a missed eager invalidation: lazy dependency checks must still block.
    ws.write_draft("asset_graph", {"version": 2})
    report = build_acceptance_report(ws)
    assert report["master"]["status"] == "stale"
    assert report["views"]["front"]["status"] == "stale"
    assert report["recommended_reference_roles"] == {}
    spec = SkillSpec("views", "generate", preconditions=[MASTER + ":committed"])
    assert not spec.can_run(ws)[0]


def test_master_validation_receives_screenplay_and_explicit_age(tmp_path):
    from src.agentic_video.asset_studio.validators import _expected_spec
    ws = Workspace(tmp_path / "run")
    ws.write_draft("asset_graph", {"assets": [{"asset_id": "C0", "immutable": {
        "face": "老年男性", "hair": "白发"}}]})
    ws.write_draft("screenplay", {"characters": [{"id": "C0",
        "description": "戴黑色眼罩的老年男性"}]})
    spec = json.loads(_expected_spec(ws))
    assert spec["face"] == "老年男性"
    assert spec["screenplay_description"] == "戴黑色眼罩的老年男性"
