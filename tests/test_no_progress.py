# -*- coding: utf-8 -*-
"""Asset schema contract + no-progress detection 回归锚定。"""
from __future__ import annotations

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
