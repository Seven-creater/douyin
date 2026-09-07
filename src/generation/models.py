"""生成层数据结构（纯 dataclass，无 I/O）。"""
from __future__ import annotations

from dataclasses import dataclass, field

# MiniMax 帧数约束（serve.py 实测）：num_frames = 17n+5 且 ≥124
FRAME_STEP = 17
FRAME_BASE = 5
MIN_FRAMES = 124
DEFAULT_FPS = 24


@dataclass(frozen=True)
class UnitSegment:
    seg_index: int          # 在 template.timeline 中的下标
    start_s: float          # 模板时间线绝对时刻
    end_s: float
    unit_offset_s: float    # 生成片段内取片起点 = head_pad + 前序段时长和


@dataclass(frozen=True)
class GenerationUnit:
    unit_id: int
    roles: list[str]                    # 覆盖的 timeline role（主 role 取首个）
    timeline_span: tuple[float, float]
    duration_s: float                   # 模板时长（各段和）
    num_frames: int                     # 17n+5 ≥124
    seed_base: int                      # fnv1a(template_id, unit_id)
    visual_brief: str                   # 合并后的中文视觉描述（重写输入）
    speech_lines: list[str]             # 各段 speech 非 null 值
    segments: list[UnitSegment]


@dataclass(frozen=True)
class GenerationPlan:
    template_id: str
    source_duration_s: float
    audio_brief: str | None             # template.audio.bgm
    units: list[GenerationUnit]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str                     # snake_case，作目录名
    label: str
    substitutions: dict[str, str]       # replaceable 元素描述 → 新值


@dataclass(frozen=True)
class UnitPrompt:
    unit_id: int
    prompt: str                         # 最终 t2va prompt
    seed: int


def fnv1a(text: str) -> int:
    """确定性 seed 源（全链路可复现）。"""
    h = 0x811C9DC5
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h
