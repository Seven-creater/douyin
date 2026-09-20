# -*- coding: utf-8 -*-
"""Storyboard 视觉后端接口（M3-B 接线；今晚只定义不运行）。

start/end pair 的意义：单一 keyframe 不足以规定一个镜头"从什么状态
发展到什么状态"（STAGE）。Reference Router 给后端的 refs 已是
COMMIT 过的资产视图文件——后端不猜人物长什么样。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class StoryboardBackend:
    """固定接口，模型可替换（v1: Qwen-Image-Edit；后续可换任意
    image-conditioned 模型）。"""

    def generate_start_frame(self, shot: dict[str, Any],
                             reference_paths: list[Path],
                             out_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def generate_end_frame(self, shot: dict[str, Any],
                           start_frame_path: Path,
                           reference_paths: list[Path],
                           out_path: Path) -> dict[str, Any]:
        raise NotImplementedError


def storyboard_frame_prompt(shot: dict[str, Any], phase: str,
                            identity_refs_note: str = "") -> str:
    """start/end frame 的生成 prompt（模型无关的描述层）。

    phase: "start" | "end"。成片 16:9 横屏（1920×1080 生产线）。
    """
    size = (shot.get("camera") or {}).get("shot_size", "medium")
    movement = (shot.get("camera") or {}).get("movement", "static")
    state = shot.get("start_state") if phase == "start" \
        else shot.get("end_state")
    anchor = ("This frame continues exactly from the start frame of the "
              "same shot." if phase == "end" else "")
    return (
        f"Storyboard frame, cinematic 16:9 landscape composition, "
        f"{size} shot size, camera {movement}. "
        f"Visual state: {state}. "
        f"The exact same characters as the reference images must appear "
        f"with identical identity, hairstyle and body build. "
        f"{identity_refs_note} {anchor} "
        "No text, no watermark, no extra people.")


def _shot_seed(shot_id: str, offset: int) -> int:
    """确定性 shot seed（crc32，跨进程稳定）。"""
    import zlib
    return 20260920 + offset + (zlib.crc32(str(shot_id).encode()) % 1000)


class QwenEditStoryboardBackend(StoryboardBackend):
    """v1 实现：复用 M2 的 Qwen-Image-Edit-2511 后端。"""

    def __init__(self, edit_backend) -> None:
        self.edit = edit_backend

    def generate_start_frame(self, shot, reference_paths, out_path):
        return self.edit.edit(
            reference_paths[0],
            storyboard_frame_prompt(shot, "start"),
            out_path, width=1920, height=1088,  # 16:9（16 对齐）
            seed=_shot_seed(str(shot.get("shot_id")), 700))

    def generate_end_frame(self, shot, start_frame_path,
                           reference_paths, out_path):
        return self.edit.edit(
            start_frame_path,
            storyboard_frame_prompt(shot, "end"),
            out_path, width=1920, height=1088,
            seed=_shot_seed(str(shot.get("shot_id")), 800))
