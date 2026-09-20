# -*- coding: utf-8 -*-
"""v4 Validators：与 Skills 解耦的生产验收测试（Actor ≠ Test）。

同一 Omni 但独立 prompt/上下文——最终 validator 不收到"我认为已修好"
的 reasoning。
"""
from __future__ import annotations

import json
from typing import Any

from src.agentic_video.workspace import Workspace

VALIDATE_SCREENPLAY_PROMPT = """你是剧本验收员（不是编剧）。输入一部
剧本 JSON。独立逐项检查以下三条，只输出一个 JSON：
{"passed": true, "failures": [{"beat_id": "...", "check": "...",
  "detail": "..."}]}
三条检查：
1. causal：counter_evidence 是否直接反驳 initial_belief（同维度正反命题）
2. visualizability：每个 beat 的 visual_action 是否可观察可拍
   （无内部心理/无文字渲染依赖/actor-action-target 明确）
3. production_feasibility：production_feasibility 字段是否存在且合理
你没有看过任何之前的修复过程——只根据当前剧本本身判断。
输入剧本："""

# 注：asset_graph 不再有 Omni 语义验收 prompt——Run A-v2 教训，
# 确定性 schema 校验（skills/asset_schema.py）是唯一权威。


def run_test(validator_name: str, workspace: Workspace, runner
             ) -> dict[str, Any]:
    """运行指定 validator，返回 {passed, failures, validators_run}。"""
    if validator_name == "test_creative_dna":
        return _test_creative_dna(workspace)
    if validator_name == "test_screenplay":
        return _test_screenplay(workspace, runner)
    if validator_name == "test_asset_graph":
        return _test_asset_graph(workspace, runner)
    if validator_name in ("test_character_master",
                          "test_character_multiview",
                          "test_views_4k", "test_character_asset"):
        from src.agentic_video.asset_studio.validators import run_m2_test
        return run_m2_test(validator_name, workspace, runner)
    return {"passed": True, "failures": [],
            "validators_run": [validator_name], "detail": "unknown validator, auto-pass"}


def _test_creative_dna(workspace: Workspace) -> dict[str, Any]:
    """Deterministic check：creative_dna 存在且非空。"""
    dna = workspace.read_artifact("creative_dna")
    failures = []
    if not dna:
        failures.append({"check": "exists", "detail": "creative_dna is empty"})
    elif not (dna.get("narrative_invariants") or {}):
        failures.append({"check": "invariants",
                         "detail": "narrative_invariants missing"})
    return {"passed": not failures, "failures": failures,
            "validators_run": ["test_creative_dna"]}


def _test_screenplay(workspace: Workspace, runner) -> dict[str, Any]:
    """Omni 独立验收（不收修复 reasoning）。"""
    from src.agentic_video.reference_program_v9 import _parse_one_object
    screenplay = workspace.read_artifact("screenplay")
    if not screenplay:
        return {"passed": False,
                "failures": [{"check": "exists",
                              "detail": "screenplay is empty"}],
                "validators_run": ["test_screenplay"]}
    answer = runner.ask(
        VALIDATE_SCREENPLAY_PROMPT + json.dumps(
            screenplay, ensure_ascii=False, separators=(",", ":")),
        max_new_tokens=2048, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        value = _parse_one_object(text, stage="validator")
    except Exception:
        return {"passed": False,
                "failures": [{"check": "parse", "detail": raw[:200]}],
                "validators_run": ["test_screenplay"]}
    return {"passed": bool(value.get("passed")),
            "failures": value.get("failures") or [],
            "validators_run": ["test_screenplay"]}


def _test_asset_graph(workspace: Workspace, runner) -> dict[str, Any]:
    """P0.4 修复：完全确定性校验（Run A-v2 教训——Omni 语义层会与 schema
    层互相矛盾导致死循环）。immutable 含 face/body_build 是事实判定。"""
    from src.agentic_video.skills.asset_schema import validate_asset_graph
    graph = workspace.read_artifact("asset_graph")
    if not graph:
        return {"passed": False,
                "failures": [{"check": "exists",
                              "detail": "asset_graph is empty"}],
                "validators_run": ["test_asset_graph"]}
    failures = validate_asset_graph(graph)
    return {"passed": not failures, "failures": failures,
            "validators_run": ["test_asset_graph"]}
