# -*- coding: utf-8 -*-
"""M2-A Character Asset Studio prompt 契约（棚拍资产标准）。

身份主资产不绑定剧情服装（identity master = 人是谁；
wardrobe pack = 这一场穿什么），master 穿中性基础款。

P0-2 纠正：Qwen-Image 支持独立 negative_prompt（配 true_cfg_scale>1
启用 CFG）。身份关键约束保留在正向 prompt；通用劣化/多余元素走
negative prompt 辅助。
"""
from __future__ import annotations

import json
from typing import Any

# 通用劣化负面词（身份关键约束不放这里——放正向语义里）
DEFAULT_NEGATIVE_PROMPT = (
    "extra people, duplicate person, mirrored duplicate, text, "
    "watermark, logo, dramatic pose, distorted anatomy, extra limbs, "
    "cluttered background, scene context")

# 通用棚拍条款（所有视图共用）
STUDIO_BASE = (
    "9:16 vertical portrait, soft even studio lighting, "
    "neutral gray seamless studio backdrop, eye-level camera, "
    "minimal perspective distortion, sharp focus, high detail, "
    "photorealistic")

# master 明确禁项（保留正向描述；negative prompt 做辅助）
MASTER_FORBIDDEN = (
    "no dramatic pose, no scene context, single person only")

# 中性基础款（占位服装，不进身份；可被 character_spec 覆盖）
NEUTRAL_WARDROBE = (
    "wearing plain fitted dark gray short-sleeve top and matching "
    "full-length trousers, plain black soft shoes, no non-canonical "
    "accessories, no logos")

MASTER_VIEW_SPECS = {
    "front": (
        "full-body front view, head facing forward toward the camera, "
        "relaxed natural stance, neutral expression, mouth closed, "
        "eyes open looking at the camera, entire body from head to "
        "feet visible, full body visible"),
    "profile": (
        "full-body strict 90-degree side profile view, head facing "
        "directly to the side, only one eye contour visible, relaxed "
        "natural stance, neutral expression, mouth closed, entire "
        "body from head to feet visible, full body visible"),
    "back": (
        "full-body back view from directly behind, head facing away "
        "from the camera, relaxed natural stance, entire body from "
        "head to feet visible, full body visible"),
    "face": (
        "head-and-shoulders facial close-up portrait, head facing "
        "forward toward the camera, neutral expression, mouth closed, "
        "eyes open looking at the camera, face fully visible and "
        "evenly lit"),
}


def _anatomy_phrase(character_spec: dict[str, Any] | None) -> str:
    """P1-7：解剖学自适应——不把'标准可站立健全成人'当 universal 模板。"""
    if not character_spec:
        return MASTER_VIEW_SPECS["front"].replace(
            "relaxed natural stance",
            "standing straight, arms relaxed at the sides")
    parts = [
        "neutral relaxed pose appropriate to the character's canonical "
        "anatomy"]
    if character_spec.get("anatomy_constraints"):
        parts.append(f"anatomy: {character_spec['anatomy_constraints']}")
    if character_spec.get("mobility_aids"):
        parts.append(f"mobility aids: "
                     f"{character_spec['mobility_aids']} (must be "
                     f"preserved and visible)")
    if character_spec.get("identity_accessories"):
        parts.append(f"identity-defining accessories to preserve: "
                     f"{character_spec['identity_accessories']}")
    if character_spec.get("pose_profile") == "seated_neutral":
        parts.append("seated neutral pose")
    return ", ".join(parts)


def build_master_prompt(identity_description: str,
                        character_spec: dict[str, Any] | None = None,
                        ) -> str:
    """Hero Master（front 全身）生成 prompt。

    identity_description 来自 asset_graph 的 canonical/immutable 描述
    （人物是谁），不含剧情服装。character_spec 可覆盖解剖学假设。
    """
    return (
        f"Character identity reference sheet photo. "
        f"{identity_description}. "
        f"{_anatomy_phrase(character_spec)}, {NEUTRAL_WARDROBE}, "
        f"{STUDIO_BASE}, {MASTER_FORBIDDEN}.")


def build_view_prompt(view: str, identity_description: str) -> str:
    """Edit-2511 迭代视图生成 prompt（以 master 为身份参考）。

    强调 same person + 只改视角，锁身份特征与占位服装。
    """
    spec = MASTER_VIEW_SPECS[view]
    return (
        f"Change the camera viewpoint of this exact same person: "
        f"{spec}. "
        f"Keep it the exact same person: {identity_description}. "
        "Same neutral gray seamless studio backdrop, same soft even "
        f"studio lighting, same plain dark gray outfit, {STUDIO_BASE}, "
        f"{MASTER_FORBIDDEN}.")


def build_repair_view_prompt(view: str, identity_description: str,
                             failures: list[dict[str, Any]],
                             attempt: int) -> str:
    """P1-2：诊断式修复——失败原因 + 递进要求进 prompt（非盲目重抽）。"""
    notes = []
    for failure in failures or []:
        check = str(failure.get("check") or "")
        detail = str(failure.get("detail") or "")
        if check in ("identity", "same_person"):
            notes.append("previous attempt drifted the facial identity — "
                         "this time preserve the exact nose bridge, "
                         "jawline, hairline and age appearance")
        elif check in ("view_angle", "view_correct"):
            notes.append(f"previous viewpoint was wrong ({detail}) — "
                         "rotate the camera to the exact required "
                         f"{view} viewpoint")
        elif check == "extra_person":
            notes.append("previous attempt contained an extra person — "
                         "strictly single person in frame")
        elif check == "background":
            notes.append("previous background was not a clean neutral "
                         "gray seamless backdrop")
        elif detail:
            notes.append(f"previous failure ({check}): {detail}")
    failure_block = "\n".join(f"- {n}" for n in notes) or \
        "- previous attempt failed validation"
    return (
        f"Repair regeneration (attempt {attempt}) of the "
        f"{view} view. Previous validation failures:\n{failure_block}\n"
        f"{build_view_prompt(view, identity_description)}")


def build_wardrobe_prompt(identity_description: str,
                          wardrobe_description: str) -> str:
    """（M2-B+ 预留）剧情服装包：同一身份换装。"""
    return (
        f"Change only the outfit of this exact same person to: "
        f"{wardrobe_description}. Keep it the exact same person: "
        f"{identity_description}. Same face, same hairstyle, same body "
        f"build. {STUDIO_BASE}, {MASTER_FORBIDDEN}.")
