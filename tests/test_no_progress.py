# -*- coding: utf-8 -*-
"""Asset schema contract + no-progress detection 回归锚定。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agentic_video.no_progress import NoProgressDetector
from src.agentic_video.skills.asset_schema import (
    ASSET_GRAPH_FORMAT_SPEC, ASSET_GRAPH_SCHEMA_VERSION, validate_asset_graph)
from src.agentic_video.workspace import Workspace


# ---- Shared schema contract ----

def test_valid_asset_graph_passes() -> None:
    graph = {"schema_version": ASSET_GRAPH_SCHEMA_VERSION,
             "assets": [
                 {"asset_id": "C0", "type": "character", "tier": "A",
                  "immutable": {"face": "...", "body_build": "..."},
                  "status": "active"},
                 {"asset_id": "L01", "type": "location", "tier": "A",
                  "status": "active"}]}
    assert validate_asset_graph(graph) == []


def test_invalid_schemas_fail_with_specific_checks() -> None:
    # 缺 schema_version
    graph = {"assets": []}
    failures = validate_asset_graph(graph)
    assert any(f["check"] == "schema_version" for f in failures)
    assert any(f["check"] == "assets" for f in failures)

    # character 缺 immutable
    graph = {"schema_version": ASSET_GRAPH_SCHEMA_VERSION,
             "assets": [{"asset_id": "C0", "type": "character",
                         "tier": "A"}]}
    failures = validate_asset_graph(graph)
    assert any(f["check"] == "immutable" and f["asset_id"] == "C0"
               for f in failures)

    # character immutable 空字典
    graph["assets"][0]["immutable"] = {}
    failures = validate_asset_graph(graph)
    assert any(f["check"] == "immutable" for f in failures)

    # tier 非法
    graph["assets"][0]["immutable"] = {"face": "x"}
    graph["assets"][0]["tier"] = "Z"
    failures = validate_asset_graph(graph)
    assert any(f["check"] == "tier" for f in failures)

    # type 非法
    graph["assets"][0]["tier"] = "A"
    graph["assets"][0]["type"] = "dragon"
    failures = validate_asset_graph(graph)
    assert any(f["check"] == "type" for f in failures)

    # asset_id 重复
    good = {"asset_id": "C0", "type": "character", "tier": "A",
            "immutable": {"face": "x"}, "status": "active"}
    graph = {"schema_version": ASSET_GRAPH_SCHEMA_VERSION,
             "assets": [dict(good), dict(good)]}
    failures = validate_asset_graph(graph)
    assert any(f["check"] == "asset_id" and f["detail"] == "duplicate"
               for f in failures)


def test_format_spec_matches_schema() -> None:
    """Skill prompt 和 validator 共用同一 schema 源。"""
    assert "asset_graph_v1" in ASSET_GRAPH_FORMAT_SPEC
    assert "immutable" in ASSET_GRAPH_FORMAT_SPEC
    assert "identity" in ASSET_GRAPH_FORMAT_SPEC or "face" in ASSET_GRAPH_FORMAT_SPEC


# ---- No-progress detection ----

def _ws_with_screenplay(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "run")
    ws.set_dependency("screenplay", "creative_dna", "committed")
    ws.set_dependency("asset_graph", "screenplay", "committed")
    ws.write_draft("creative_dna", {"x": 1})
    ws.commit("creative_dna")
    ws.write_draft("screenplay", {"title": "A"})
    ws.commit("screenplay")
    return ws


def test_no_progress_detects_stagnation(tmp_path: Path) -> None:
    """连续 N 步世界状态+失败都不变 → stalled=True + 动作空间收缩。"""
    ws = _ws_with_screenplay(tmp_path)
    detector = NoProgressDetector(limit=3)
    failure = {"failures": [{"check": "immutable", "asset_id": "C0"}]}

    # Step 1: 首次（有 last_hashes 为 None → progress=True）
    r1 = detector.step(ws, failure)
    assert r1["progress"] is True

    # Steps 2-4: 什么都不改，同一个 failure
    r2 = detector.step(ws, failure)
    r3 = detector.step(ws, failure)
    r4 = detector.step(ws, failure)
    assert not r2["progress"]
    assert not r3["progress"]
    # 第 3 次连续零增量 → stalled
    assert r4["stalled"]
    assert r4["event"] == "NO_PROGRESS"
    assert r4["consecutive_stagnant_steps"] == 3

    # 动作空间收缩
    restricted = detector.restricted_actions()
    assert "repair_screenplay" in restricted
    assert "stop" in restricted


def test_no_progress_resets_on_world_change(tmp_path: Path) -> None:
    """世界变化（如新 draft）重置零计数。"""
    ws = _ws_with_screenplay(tmp_path)
    detector = NoProgressDetector(limit=3)
    failure = {"failures": [{"check": "x"}]}

    detector.step(ws, failure)  # progress=True（首次）
    r2 = detector.step(ws, failure)  # stagnant=1
    assert not r2["stalled"]

    # 世界变化：写入新版本
    ws.write_draft("asset_graph", {"assets": []})
    r3 = detector.step(ws, failure)
    assert r3["progress"] is True
    assert r3["consecutive_stagnant_steps"] == 0


def test_no_progress_resets_on_failure_change() -> None:
    """失败列表变化（如 3 fail → 1 fail）也算 progress。"""
    import tempfile
    from src.agentic_video.workspace import Workspace as W

    with tempfile.TemporaryDirectory() as td:
        ws = W(Path(td) / "run")
        ws.write_draft("screenplay", {"title": "A"})
        detector = NoProgressDetector(limit=2)
        failure_3 = {"failures": [{"x": 1}, {"x": 2}, {"x": 3}]}
        failure_1 = {"failures": [{"x": 1}]}

        detector.step(ws, failure_3)  # progress=True
        r2 = detector.step(ws, failure_3)  # stagnant=1
        assert not r2["stalled"]

        r3 = detector.step(ws, failure_1)  # failure changed → progress
        assert r3["progress"] is True
        assert r3["consecutive_stagnant_steps"] == 0


def test_no_progress_not_fooled_by_version_bump(tmp_path: Path) -> None:
    """M3-A dry-run 教训：repair 原样吐回相同内容 → 版本号递增但
    sha 不变——这不是 progress，必须计入停滞。"""
    ws = _ws_with_screenplay(tmp_path)
    detector = NoProgressDetector(limit=3)
    failure = {"failures": [{"check": "asset_ref"}]}

    detector.step(ws, failure)  # 首次 progress=True
    ws.write_draft("asset_graph", {"assets": []})  # v1（世界真变了）
    detector.step(ws, failure)  # progress=True（sha 变）
    # 之后 repair 原样吐回：版本号 v2/v3/v4 递增但内容相同
    r3 = detector.step(ws, failure)  # 无变化 → stagnant=1
    assert not r3["progress"]
    r4 = detector.step(ws, failure)  # stagnant=2
    assert not r4["stalled"]
    r5 = detector.step(ws, failure)  # stagnant=3 → stalled
    assert r5["stalled"]


# ---- Run A-v2 三教训回归 ----

class _RecordingRunner:
    """记录 prompt 并按脚本回放的假 Omni 池。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def ask(self, prompt: str, **kw):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else "{}"
        from types import SimpleNamespace
        return SimpleNamespace(text=text)


def test_repair_screenplay_unwraps_mirrored_payload(tmp_path: Path) -> None:
    """Run A-v2 教训：Omni 照抄 payload 外层 {"screenplay": ...} 包装
    时，write_draft 必须存内层对象而非嵌套包装。"""
    from src.agentic_video.skills import build_m1_registry

    inner = {"title": "A", "scenes": [{"scene_id": "SC01", "beats": []}]}
    runner = _RecordingRunner([json.dumps({"screenplay": inner})])
    registry = build_m1_registry(runner=runner)

    ws = _ws_with_screenplay(tmp_path)
    ws.write_draft("screenplay", inner)
    registry.execute({"skill": "repair_screenplay", "target": "B4"}, ws,
                     test_failures=[{"beat_id": "B4",
                                     "check": "visualizability"}],
                     target="B4")
    assert ws.read_artifact("screenplay") == inner  # 无嵌套包装


def test_extract_assets_carries_schema_and_failure_feedback(
        tmp_path: Path) -> None:
    """schema 原生内嵌 prompt；失败后的 re-extract 收到失败明细。"""
    from src.agentic_video.skills import build_m1_registry

    good = {"schema_version": "asset_graph_v1",
            "assets": [{"asset_id": "C0", "type": "character", "tier": "A",
                        "immutable": {"face": "x"}, "status": "active"}]}
    runner = _RecordingRunner([json.dumps(good)])
    registry = build_m1_registry(runner=runner)

    ws = _ws_with_screenplay(tmp_path)
    ws.write_draft("screenplay", {"title": "A"})
    registry.execute({"skill": "extract_assets"}, ws,
                     test_failures=[{"asset_id": "_root",
                                     "check": "schema_version",
                                     "detail": "expected asset_graph_v1"}])
    # prompt 带共享 schema + 失败明细
    assert "asset_graph_v1" in runner.prompts[0]
    assert "schema_version" in runner.prompts[0]
    assert "expected asset_graph_v1" in runner.prompts[0]
    assert runner.prompts[0].index("asset_graph_v1") < \
        runner.prompts[0].index("expected asset_graph_v1")  # schema 在前
    assert ws.read_artifact("asset_graph") == good


def test_asset_graph_validator_never_calls_omni(tmp_path: Path) -> None:
    """确定性 schema 校验是唯一权威——validator 不再调 Omni。"""
    from src.agentic_video.validators import run_test

    class _Boom:
        def ask(self, *a, **kw):  # pragma: no cover
            raise AssertionError("asset_graph validator must not call Omni")

    ws = _ws_with_screenplay(tmp_path)
    ws.write_draft("asset_graph",
                   {"schema_version": "asset_graph_v1",
                    "assets": [{"asset_id": "C0", "type": "character",
                                "tier": "A", "immutable": {"face": "x"},
                                "status": "active"}]})
    report = run_test("test_asset_graph", ws, runner=_Boom())
    assert report["passed"] is True


def test_agent_loop_stops_on_no_progress(tmp_path: Path) -> None:
    """世界+失败连续零增量 → agent_loop 原生 break（stop_reason）。"""
    import tempfile
    from src.agentic_video.agent_v4 import agent_loop
    from src.agentic_video.skills import build_m1_registry
    from src.agentic_video.workspace import Workspace as W
    from types import SimpleNamespace

    class _IdleRunner:
        """choose_action 恒选 inspect；validator 恒返回非法 JSON 体。"""
        def ask(self, prompt: str, **kw):
            return SimpleNamespace(text='{"skill": "inspect_screenplay"}')

    with tempfile.TemporaryDirectory() as td:
        ws = W(Path(td) / "run")
        ws.set_dependency("screenplay", "creative_dna", "committed")
        ws.set_dependency("asset_graph", "screenplay", "committed")
        ws.write_draft("creative_dna", {"narrative_invariants": {"x": 1}})
        ws.record_dependency_snapshot("creative_dna")
        ws.commit("creative_dna")
        ws.write_draft("screenplay", {"title": "A"})
        ws.record_dependency_snapshot("screenplay")
        ws.commit("screenplay")

        registry = build_m1_registry(runner=None)
        result = agent_loop(ws, registry, controller_runner=_IdleRunner(),
                            max_steps=10,
                            no_progress_detector=NoProgressDetector(limit=3))
        assert result["stop_reason"] == "no_progress"
        assert not result["goal_satisfied"]
        assert result["steps_taken"] < 10  # 早停，非跑满
