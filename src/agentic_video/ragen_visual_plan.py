# -*- coding: utf-8 -*-
"""G1-v2：Beat Visual Plan + Visualizability Gate + AST 编译器 + COMMIT 状态机。

p0622 四项修正并入：
- Visualizability Gate 职责限定（文本级"是否可拍"，不承担"观众是否看懂"）
- body_anchor 只提供体格，retention 写明比赛服不要求延续
- 编译器分层 = Global Story Context（人物是谁）+ Current Shot Plan（拍什么）
- COMMIT 加 usable_for_story 总门；Global Identity Memory 不可变
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.ragen_director import ReGenBlocked

VISUAL_PLAN_VERSION = "beat_visual_plan_v1"

# 用户 p0622 §十七 拍板的三个 shot（视觉鲁棒：不依赖编号/报名表渲染）
G1_PLAN: dict[str, Any] = {
    "schema_version": VISUAL_PLAN_VERSION,
    "span_id": "G1",
    "beats": ["B1", "B2"],
    "location_id": "indoor_taekwondo_training_hall",
    "post_overlay_text": "他们都觉得她只是来体验的。",
    "narrative_claim": "others clearly do not take C0 seriously as a competitor",
    "shots": [
        {"shot_id": "G1_S1", "duration": "0-3s", "actor": "C0",
         "action_semantics": "enters the training hall while other athletes warm up; "
         "the camera clearly establishes C0",
         "camera": "medium shot", "purpose": "establish protagonist"},
        {"shot_id": "G1_S2", "duration": "3-7s", "actor": "coach_and_C0",
         "action_semantics": "a coach glances at C0, shows a skeptical expression, "
         "then waves her to the back of the line; nearby athletes exchange "
         "dismissive glances",
         "camera": "two-shot then reaction close-up",
         "purpose": "make underestimation visible"},
        {"shot_id": "G1_S3", "duration": "7-10s", "actor": "C0",
         "action_semantics": "C0 does not argue; she calmly looks toward the "
         "competition area and prepares her gear",
         "camera": "close-up on C0", "purpose": "prepare transition"},
    ],
}

VISUALIZABILITY_PROMPT = """你是可视化检查员。输入一份 Shot Plan（每个 shot
的 actor / action / camera / location）。逐条检查四项，只输出一个 JSON：
{"visualizable": true, "problems": ["..."]}
四项检查：
1. 每个叙事状态都有**可观察的视觉证据**（动作/表情/位置变化）；
2. 不依赖不可见的心理活动（"不看好她"这种要落到摇头/摆手/眼神上）；
3. 不依赖画面文字渲染（编号/报名表内容不作为叙事载体）；
4. actor / action / target / spatial relation 均明确。
输入："""


def validate_visualizability(plan: dict[str, Any], *, runner) -> dict[str, Any]:
    """文本级门（不承担观众是否看懂——那在 H3 后的 Readability Gate）。"""
    from src.agentic_video.reference_program_v9 import _parse_one_object
    payload = json.dumps(
        {"shots": plan.get("shots"), "location": plan.get("location_id"),
         "overlay": plan.get("post_overlay_text")},
        ensure_ascii=False, separators=(",", ":"))
    answer = runner.ask(VISUALIZABILITY_PROMPT + payload,
                        max_new_tokens=512, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        value = _parse_one_object(text, stage="visualizability")
    except Exception:
        raise ReGenBlocked("visual_plan", "visualizability_parse_failed")
    if not isinstance(value.get("visualizable"), bool):
        raise ReGenBlocked("visual_plan", "visualizability_verdict_invalid")
    return value


def compile_ref2va(plan: dict[str, Any],
                   memory: dict[str, Any]) -> dict[str, Any]:
    """AST 编译（Compiler 无创作权）：字段级装配，语义守恒。

    分层（MultiShotMaster 同款）：Global Story Context（人物是谁，双锚
    identity_only）+ Current Shot Plan（拍什么，不重复人物外观）。
    actor_role / action_semantics / location_id 原样传递——G1_PLAN 里的
    英文 action 直接进 prompt，不做二次概括。
    """
    c0 = (memory or {}).get("C0") or {}
    face = c0.get("face_anchor") or {}
    body = c0.get("body_anchor") or {}
    semantic = c0.get("semantic_identity") or {}
    shots = plan.get("shots") or []
    shot_lines = [
        f"[{row['shot_id']}] {row['duration']}: {row['actor']} "
        f"{row['action_semantics']} ({row['camera']})"
        for row in shots]
    prompt = "\n\n".join([
        "subject_definitions:\n"
        "<Subject 1> is the same adult woman defined jointly by "
        "<Picture 1> and <Picture 2>.\n"
        "<Picture 1> provides her facial identity.\n"
        "<Picture 2> provides her body proportions and overall physical "
        "identity.\n"
        "Her competition wardrobe is NOT required to remain in this clip; "
        "wardrobe follows the shot plan below.\n"
        f"Semantic identity: {json.dumps(semantic, ensure_ascii=False)}",
        "summary:\n[reference generation] Generate one coherent 10-second "
        "clip featuring <Subject 1> as described. Preserve <Subject 1>'s "
        "identity throughout.",
        "retention_analysis:\n"
        "<Subject 1> identity (throughout): fully_preserved - face, gender "
        "presentation, body build from <Picture 1>+<Picture 2>.\n"
        "<Picture 1> (source for <Subject 1>): attribute_transfer - facial "
        "identity only.\n"
        "<Picture 2> (source for <Subject 1>): attribute_transfer - body "
        "proportions only, not wardrobe.",
        "detailed_description:\n" + "\n".join(shot_lines),
        "overall_soundscape:\nNatural indoor training-hall ambience.",
        "non_diegetic_music:\nNone."])
    references = [
        {"type": "image", "uri": Path(face["path"]).resolve().as_uri()},
        {"type": "image", "uri": Path(body["path"]).resolve().as_uri()}]
    return {"prompt": prompt, "references": references, "task": "ref2va"}


def commit_state(state: dict[str, Any]) -> dict[str, Any]:
    """usable_for_story 总门 → committed（素材正式成为世界事实）。"""
    gates = ("identity_valid", "beat_readable", "coverage_valid")
    state["usable_for_story"] = all(state.get(g) for g in gates)
    state["committed"] = state["usable_for_story"]
    return state
