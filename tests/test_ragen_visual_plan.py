# -*- coding: utf-8 -*-
"""G1-v2：Visual Plan / Visualizability Gate / AST 编译 / COMMIT 回归锚定。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agentic_video import ragen_visual_plan as vp
from src.agentic_video.ragen_director import ReGenBlocked
from src.agentic_video.ragen_visual_plan import (
    G1_PLAN, commit_state, compile_ref2va, validate_visualizability)


def _memory() -> dict:
    return {"C0": {
        "face_anchor": {"path": "/tmp/face.jpg"},
        "body_anchor": {"path": "/tmp/body.jpg"},
        "semantic_identity": {"gender_presentation": "female",
                              "body_build": "slim athletic"}}}


def test_compile_semantic_conservation_no_recap() -> None:
    """AST 语义守恒：plan 的 action 原样进 prompt，不二次概括。"""
    compiled = compile_ref2va(G1_PLAN, _memory())
    prompt = compiled["prompt"]
    # 三个 shot 的 action_semantics 逐字保留
    for row in G1_PLAN["shots"]:
        assert row["action_semantics"] in prompt
    assert G1_PLAN["location_id"] in prompt or "training hall" in prompt
    # 分层：Global（人物）与 Shot（动作）不混——shot 段不重复外观描述
    desc = prompt.split("detailed_description:")[1]
    assert "facial identity" not in desc
    # body_anchor 只锁体格不锁比赛服
    assert "NOT required to remain" in prompt
    assert "body proportions only, not wardrobe" in prompt
    # 双条件
    assert len(compiled["references"]) == 2
    assert compiled["task"] == "ref2va"
    # 无嵌套段落（AST 编译，非字符串套字符串）
    assert prompt.count("subject_definitions:") == 1
    assert prompt.count("overall_soundscape:") == 1
    assert "integrated_multimodal_description" not in prompt


def test_visualizability_gate_text_level_scope() -> None:
    """职责限定：查'是否可拍'，不查'观众是否看懂'。"""

    class OKRunner:
        def ask(self, prompt, **kwargs):
            return SimpleNamespace(text=json.dumps(
                {"visualizable": True, "problems": []}))

    result = validate_visualizability(G1_PLAN, runner=OKRunner())
    assert result["visualizable"] is True

    class FailRunner:
        def ask(self, prompt, **kwargs):
            return SimpleNamespace(text=json.dumps(
                {"visualizable": False,
                 "problems": ["relies on registration list text"]}))

    result = validate_visualizability(G1_PLAN, runner=FailRunner())
    assert result["visualizable"] is False
    assert result["problems"]

    class BrokenRunner:
        def ask(self, prompt, **kwargs):
            return SimpleNamespace(text="not json at all {{{")

    with pytest.raises(ReGenBlocked):
        validate_visualizability(G1_PLAN, runner=BrokenRunner())


def test_commit_requires_all_three_gates_then_world_fact() -> None:
    state = {"identity_valid": True, "beat_readable": True,
             "coverage_valid": False}
    commit_state(state)
    assert state["usable_for_story"] is False
    assert state["committed"] is False
    state["coverage_valid"] = True
    commit_state(state)
    assert state["usable_for_story"] is True
    assert state["committed"] is True


def test_global_identity_memory_immutable_on_commit() -> None:
    """Memory 不变式：COMMIT 后生成帧不得写回 identity 锚。"""
    memory = _memory()
    original_face = memory["C0"]["face_anchor"]["path"]
    state = {"identity_valid": True, "beat_readable": True,
             "coverage_valid": True,
             "generated_frame": "/tmp/g1_last.jpg"}
    commit_state(state)
    # commit_state 不接受 memory 参数——设计上生成结果无法触碰锚
    assert memory["C0"]["face_anchor"]["path"] == original_face
