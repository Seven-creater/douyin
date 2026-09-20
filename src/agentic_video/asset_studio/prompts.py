# -*- coding: utf-8 -*-
"""M2-A Character Asset Studio prompt 契约（棚拍资产标准）。

身份主资产不绑定剧情服装（identity master = 人是谁；
wardrobe pack = 这一场穿什么），master 穿中性基础款。
"""
from __future__ import annotations

# 通用棚拍条款（所有视图共用）
STUDIO_BASE = (
    "9:16 vertical portrait, soft even studio lighting, "
    "neutral gray seamless studio backdrop, eye-level camera, "
    "minimal perspective distortion, sharp focus, high detail, "
    "photorealistic")

# master 明确禁项（负面语义直接写进正向描述，Qwen-Image 无独立负_prompt 通道）
MASTER_FORBIDDEN = (
    "no dramatic pose, no scene context, no props, no text, "
    "no watermark, no extra people, single person only")

# 中性基础款（占位服装，不进身份）
NEUTRAL_WARDROBE = (
    "wearing plain fitted dark gray short-sleeve top and matching "
    "full-length trousers, plain black soft shoes, no accessories, "
    "no logos")

MASTER_VIEW_SPECS = {
    "front": (
        "full-body front view, standing straight facing the camera, "
        "head facing forward, relaxed natural stance, arms relaxed "
        "at the sides, neutral expression, mouth closed, eyes open "
        "looking at the camera, entire body from head to feet visible, "
        "full body visible"),
    "profile": (
        "full-body strict 90-degree side profile view, standing "
        "straight, head facing directly to the side, relaxed natural "
        "stance, neutral expression, mouth closed, entire body from "
        "head to feet visible, full body visible"),
    "back": (
        "full-body back view from directly behind, standing straight, "
        "head facing away from the camera, relaxed natural stance, "
        "entire body from head to feet visible, full body visible"),
    "face": (
        "head-and-shoulders facial close-up portrait, head facing "
        "forward toward the camera, neutral expression, mouth closed, "
        "eyes open looking at the camera, face fully visible and "
        "evenly lit"),
}


def build_master_prompt(identity_description: str) -> str:
    """Hero Master（front 全身）生成 prompt。

    identity_description 来自 asset_graph 的 canonical/immutable 描述
    （人物是谁），不含剧情服装。
    """
    return (
        f"Character identity reference sheet photo. {identity_description}. "
        f"{MASTER_VIEW_SPECS['front']}, {NEUTRAL_WARDROBE}, "
        f"{STUDIO_BASE}, {MASTER_FORBIDDEN}.")


def build_view_prompt(view: str, identity_description: str) -> str:
    """Edit-2511 迭代视图生成 prompt（以 master 为身份参考）。

    强调 same person + 只改视角，锁身份特征与占位服装。
    """
    spec = MASTER_VIEW_SPECS[view]
    return (
        f"Change the camera viewpoint of this exact same person: {spec}. "
        f"Keep it the exact same person: {identity_description}. "
        "Same neutral gray seamless studio backdrop, same soft even "
        f"studio lighting, same plain dark gray outfit, {STUDIO_BASE}, "
        f"{MASTER_FORBIDDEN}.")


def build_wardrobe_prompt(identity_description: str,
                          wardrobe_description: str) -> str:
    """（M2-B+ 预留）剧情服装包：同一身份换装。"""
    return (
        f"Change only the outfit of this exact same person to: "
        f"{wardrobe_description}. Keep it the exact same person: "
        f"{identity_description}. Same face, same hairstyle, same body "
        f"build. {STUDIO_BASE}, {MASTER_FORBIDDEN}.")
