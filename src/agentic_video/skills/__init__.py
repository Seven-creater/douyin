# -*- coding: utf-8 -*-
"""v4 M1 五个 Skills：analyze_reference / write_screenplay /
repair_screenplay / extract_assets / inspect_screenplay。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.skills.registry import SkillRegistry, SkillSpec
from src.agentic_video.workspace import Workspace

# ---- Prompts ----

CHOOSE_ACTION_PROMPT = """你是视频生产 Agent 的决策器。输入：当前目标、
workspace 状态、可用 Skills 列表（含 cost/描述/前置条件）、最近失败。
你的任务：从可用 Skills 中选出最合理的下一步动作。只输出一个 JSON：
{"skill": "skill_name", "target": "...", "reason": "一句话为什么选这个",
 "expected_change": "预期产生什么变化"}
选择原则：
1. 优先 cheap_text（诊断/修复），能用文本解决不烧 GPU
2. 最小修复：只修失败部分，不重写全局
3. 如果有测试失败，优先 repair 而非重新 write
输入："""

WRITE_SCREENPLAY_PROMPT = """你是编剧。输入 Creative DNA（一条参考短片
逆向归纳的创作程序：主题、叙事因果约束、剪辑语法、节奏先验、可替换槽位）。
你的任务：据此创作一个全新的 22 秒短视频剧本。只输出一个 JSON：
{"title": "...", "theme": "...", "target_duration": 22,
 "characters": [{"id": "C0", "description": "...", "tier": "A"}],
 "scenes": [
   {"scene_id": "SC01", "role": "situation_setup",
    "location": "L01_description",
    "beats": [
      {"beat_id": "B1", "purpose": "...",
       "visual_action": "具体可拍事件（谁/在哪/做什么/怎么反应）",
       "editing_role": {"strategy": "...", "rhythm": "...",
                        "required_evidence": ["setup", "decisive_action",
                                              "visible_result", "reaction"]},
       "production_requirements": {"characters": ["C0"],
                                   "location": "L01", "props": [],
                                   "motion_refs": []},
       "overlay_text": null}
    ]}
 ],
 "production_feasibility": {
   "visual_readability": "high",
   "tier_a_asset_count": 2, "tier_b_asset_count": 1,
   "location_count": 1, "estimated_asset_complexity": "low"}}
要求：
1. 每个 beat 的 visual_action 必须可观察可拍（无内部心理/无文字渲染依赖）
2. 剧本因果必须满足 Creative DNA 的 narrative_invariants
3. production_feasibility 评估资产成本
输入 Creative DNA："""

REPAIR_SCREENPLAY_PROMPT = """你是剧本医生。输入：当前剧本 + 失败的测试
报告 + 失败位置。你的任务：只修复失败的部分（局部 patch），不重写整部
剧本。只输出修复后的完整 JSON 剧本（结构同输入）。原则：
1. 只改失败 Beat 及其直接关联字段
2. 不得改动已 PASS 的 Beat
3. 修复必须针对失败原因（如 visualizability 失败 → 把抽象心理改为
   可观察动作）
输入："""

EXTRACT_ASSETS_PROMPT = """你是资产管家。输入锁定的剧本。你的任务：抽取
剧本中全部实体，输出 Asset Graph。只输出一个 JSON：
{"assets": [
  {"asset_id": "C0", "type": "character", "tier": "A",
   "canonical": {"description": "..."},
   "states": ["default"],
   "usage": {"B1": "default", "B2": "default"},
   "immutable": {"face": "...", "body_build": "..."},
   "mutable": {"wardrobe": true},
   "status": "active"},
  {"asset_id": "L01", "type": "location", "tier": "A", ...},
  {"asset_id": "A01", "type": "motion", "tier": "A",
   "status": "deferred_to_M4"}
]}
要求：
1. 覆盖全部实体：Character/Location/Prop/Wardrobe/Motion/Style/Audio/
   Graphics/Creature——不生成的标 status=deferred
2. 人物 immutable（脸/体格/发型）与 mutable（服装/表情/姿态）分开
3. tier 按复用频率：A=跨多镜头（完整 pack）/B=2-3 次/C=一次性
输入剧本："""


def build_m1_registry(runner=None) -> SkillRegistry:
    """构建 M1 的五个 skills。runner 为 Omni 池（文本模式 ask）。"""
    registry = SkillRegistry()

    def _ask(prompt: str, max_tokens: int = 4096) -> dict[str, Any]:
        from src.agentic_video.reference_program_v9 import _parse_one_object
        answer = runner.ask(prompt, max_new_tokens=max_tokens,
                            stop_after_json_object=True)
        raw = getattr(answer, "text", str(answer))
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return _parse_one_object(text, stage="v4_skill")

    def run_analyze_reference(workspace: Workspace, **kw) -> dict:
        # 复用已有 creative_dna（ragen_director 产物）或直接读
        existing = workspace.read_artifact("creative_dna")
        if existing:
            workspace.write_draft("creative_dna", existing)
            return {"artifact": "creative_dna", "action": "reused"}
        # 从 p04e 目录读取（由调用者提供路径）
        dna = kw.get("creative_dna") or {}
        if not dna:
            raise ValueError("creative_dna not provided")
        workspace.write_draft("creative_dna", dna)
        return {"artifact": "creative_dna", "action": "written"}

    def run_write_screenplay(workspace: Workspace, **kw) -> dict:
        dna = workspace.read_artifact("creative_dna")
        if not dna:
            raise ValueError("creative_dna not in workspace")
        screenplay = _ask(
            WRITE_SCREENPLAY_PROMPT + json.dumps(
                dna, ensure_ascii=False, separators=(",", ":")))
        version = workspace.write_draft("screenplay", screenplay)
        return {"artifact": "screenplay", "version": version,
                "action": "written"}

    def run_repair_screenplay(workspace: Workspace, **kw) -> dict:
        screenplay = workspace.read_artifact("screenplay")
        if not screenplay:
            raise ValueError("screenplay not in workspace")
        failures = kw.get("test_failures") or []
        target = kw.get("target") or ""
        payload = json.dumps(
            {"screenplay": screenplay, "failed_tests": failures,
             "failure_location": target},
            ensure_ascii=False, separators=(",", ":"))
        repaired = _ask(REPAIR_SCREENPLAY_PROMPT + payload)
        version = workspace.write_draft("screenplay", repaired)
        return {"artifact": "screenplay", "version": version,
                "action": "repaired"}

    def run_extract_assets(workspace: Workspace, **kw) -> dict:
        screenplay = workspace.read_artifact("screenplay")
        if not screenplay:
            raise ValueError("screenplay not in workspace")
        graph = _ask(
            EXTRACT_ASSETS_PROMPT + json.dumps(
                screenplay, ensure_ascii=False, separators=(",", ":")))
        version = workspace.write_draft("asset_graph", graph)
        return {"artifact": "asset_graph", "version": version,
                "action": "extracted"}

    def run_inspect_screenplay(workspace: Workspace, **kw) -> dict:
        screenplay = workspace.read_artifact("screenplay")
        return {"artifact": "screenplay", "action": "inspected",
                "data": screenplay}

    registry.register(SkillSpec(
        name="analyze_reference",
        description="Read/produce Creative DNA from reference video",
        inputs=[], outputs=["creative_dna"],
        preconditions=[], validators=["test_creative_dna"],
        cost_class="cheap_text",
        execute_fn=run_analyze_reference))

    registry.register(SkillSpec(
        name="write_screenplay",
        description="Write a new screenplay from Creative DNA",
        inputs=["creative_dna"], outputs=["screenplay"],
        preconditions=["creative_dna:committed"],
        validators=["test_screenplay"],
        cost_class="cheap_text",
        execute_fn=run_write_screenplay))

    registry.register(SkillSpec(
        name="repair_screenplay",
        description="Locally patch failing beats (not full rewrite)",
        inputs=["screenplay", "test_report"], outputs=["screenplay"],
        preconditions=["screenplay:exists"],
        validators=["test_screenplay"],
        cost_class="cheap_text",
        execute_fn=run_repair_screenplay))

    registry.register(SkillSpec(
        name="extract_assets",
        description="Extract Asset Graph from locked screenplay",
        inputs=["screenplay"], outputs=["asset_graph"],
        preconditions=["screenplay:committed"],
        validators=["test_asset_graph"],
        cost_class="cheap_text",
        execute_fn=run_extract_assets))

    registry.register(SkillSpec(
        name="inspect_screenplay",
        description="Read-only diagnosis of current screenplay",
        inputs=["screenplay"], outputs=[],
        preconditions=["screenplay:exists"],
        validators=[],
        cost_class="cheap_text", idempotent=True,
        execute_fn=run_inspect_screenplay))

    return registry
