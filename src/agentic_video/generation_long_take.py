# -*- coding: utf-8 -*-
"""H3-LT0：S2 Long-Take 质量实验（用户拍板 v2 矩阵）。

Generation Unit ≠ 最终 Snippet：H3 一次拍完整事件长 Take（12s，与参考
11.6s 匹配），Omni 后期剪 Moment。本轮只用 S2、4 recipe × 2 seed = 8 条、
H3-Q1 同世界重拍、不做跨 Take 串行继承（全部从同一 Canonical 锚出发）。

Recipe 矩阵（v2，用户修正后）：
  A prompt_only          —— H3 自己能做到什么
  B canonical_only       —— 固定人物/场景参考有没有帮助
  C video_only           —— 完整参考视频（静音视觉版）本身有多强
  D canonical_plus_video —— 图片身份锚能否在视频参考上继续增益
First Frame 是真实第 0 帧端点约束而非语义参考，移到 LT0.1；12 vs 15 的
long-horizon degradation、音视频参考对比、跨身份迁移（LT1/Q2）均不在本轮。

复用（不重写）：SGLangH3Client / build_h3_request（条件顺序与 4–15s 校验）、
select_evidence_moments、P4/P5.1 审计、OmniProcessPool、run_ffmpeg。
"""
from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.agentic_video.generation_v9g import (
    SGLangH3Client, V9GBlocked, build_h3_request, select_evidence_moments,
    _file_hash, _write_json)
from src.perception import common

LT0_VERSION = "long_take_lt0_v2"
RECIPES = ("prompt_only", "canonical_only", "video_only",
           "canonical_plus_video")
DEFAULT_SECONDS = 12.0
DEFAULT_SEEDS = (1001, 1002)
IDENTITY_SAMPLE_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)

# Canonical 候选区间 + 目标裁剪（人工审核 p0510：一张图里太多人不是身份参考，
# 是污染）。arena 在 LT0 Round 1 不作为 Picture condition——候选帧全是
# "垫+两人+裁判+观众"的群像，无法当纯场景参考；场景由 prompt/Video 1 定义。
# 生产化（p0521）：c0_fullbody 重定位为 c0_body——职责从"全身比例"改为
# "清楚展示上肢形态"（morphology 证据，Gloria 式职责单一锚）；morphology
# 参考默认 1 张、最多 2 张互补视角（multi-reference conflict）。
PACK_ROLES: tuple[dict[str, Any], ...] = (
    {"role": "c0_identity", "interval": [14.2, 16.0],
     "purpose": "主角面部清晰帧",
     "crop": {"frac_x": [0.15, 0.85], "frac_y": [0.05, 0.75]},
     "crop_note": "裁成头肩/上半身，去掉背景观众"},
    {"role": "c0_body", "interval": [8.4, 14.0],
     "purpose": "主角上肢形态清晰帧（morphology 证据）",
     "crop": {"frac_x": [0.45, 1.00], "frac_y": [0.0, 0.62]},
     "crop_note": "C0 右侧上半身，聚焦上肢末端形态"},
    {"role": "c1_opponent", "interval": [6.4, 7.2],
     "purpose": "对手清晰帧",
     "crop": {"frac_x": [0.00, 0.58], "frac_y": [0.0, 1.0]},
     "crop_note": "只裁左侧 C1"},
)
# 证据质量门（p0521 修正 1）：critical evidence 自身必须可证明属性，
# 否则整条传递链建立在错误证据上——四环齐全也没意义。
EVIDENCE_QUALITY_REQUIRED = {
    "visibility": "clear",
    "subject_attribution": "unambiguous",
    "critical_region_complete": True,
    "human_approved": True,
}
# phase → S2 内容镜头序号（0 基）的确定性映射（p04e 归一化 8 镜头）。
PHASE_SHOT_MAP: dict[str, list[int]] = {
    "initiation": [0],
    "action": [1, 2],
    "resolution": [3],
    "reflection": [4, 5, 6, 7],
}
# 12s 的 phase 时间箱（生成 prompt 用最小充分事件结构，不机械拷贝全部 shot）。
PHASE_TIME_BUDGET_S: dict[str, float] = {
    "initiation": 2.0, "action": 5.0, "resolution": 2.0, "reflection": 3.0,
}
# 官方 retention markers（Ref2VA Prompt Guide：每个 reference 的保留/迁移关系）。
RETENTION_MARKERS = ("fully_preserved", "partially_preserved",
                     "attribute_transfer", "weak_reference")
# 外观词确定性词表（Prompt Compiler v3：中文证据 → 英文 wire，不调模型）。
WARDROBE_GLOSSARY: tuple[tuple[str, str], ...] = (
    ("白色道服", "a white dobok"),
    ("红色护具", "a red chest protector"),
    ("红白护具", "red-and-white protective gear"),
    ("红色头盔", "red headgear"),
    ("黑色服装", "black clothing"),
    ("蓝色护具", "a blue chest protector"),
)
# wire 时间线模板：phase → 英文 shot 指令（official shot/timeline grammar）。
# subject1_actions = 该 phase 中主角的**肯定动作断言**（主体作用域冲突
# 检查只扫这份结构化清单，不扫成品 prompt 的自由文本）。
PHASE_WIRE_TEMPLATES: dict[str, str] = {
    "initiation": (
        "Two taekwondo athletes prepare to begin the match inside the same "
        "indoor training hall. <Subject 1>, wearing {c0_wardrobe}, takes "
        "the initiative and begins to attack."),
    "action": (
        "<Subject 1> launches a sustained kicking exchange against "
        "<Subject 2>. The interaction remains physically coherent, with "
        "<Subject 2> visibly reacting to the impacts."),
    "resolution": (
        "<Subject 1> delivers a decisive high kick. <Subject 2> loses "
        "balance and falls to the mat."),
    "reflection": (
        "The exchange has ended. <Subject 1> relaxes, turns away from the "
        "bout, and gives a light, confident reaction toward the camera."),
}
PHASE_SUBJECT1_ACTIONS: dict[str, list[str]] = {
    "initiation": ["prepares for the match", "begins to attack"],
    "action": ["launches a kicking exchange"],
    "resolution": ["delivers a decisive high kick"],
    "reflection": ["relaxes and turns toward the camera"],
}


def _wardrobe_english(*texts: str) -> list[str]:
    joined = "".join(str(text or "") for text in texts)
    return [english for chinese, english in WARDROBE_GLOSSARY
            if chinese in joined]


# ---- 生产化：人物约束卡（p0521） ----

SUBJECT_CONSTRAINTS_VERSION = "subject_constraints_v1"

def default_subject_constraints() -> dict[str, Any]:
    """用户已拍板的两条种子属性（原话），确定性装配；不扩写医学故事。"""
    return {
        "schema_version": SUBJECT_CONSTRAINTS_VERSION,
        "subject_id": "C0",
        "role": "protagonist",
        "critical_attributes": [
            {"id": "C0.target_character",
             "description": "本轮生成用户指定的参考女性主角",
             "source": "user_confirmed_target",
             "evidence_asset_ids": ["c0_identity"],
             "applies_to": ["S1", "S2", "S3"]},
            {"id": "C0.hand_morphology",
             "description": "没有双手；具体上肢形态按已确认参考画面保持",
             "source": "user_statement_and_approved_visual_evidence",
             "evidence_asset_ids": ["c0_body"],
             "applies_to": ["S1", "S2", "S3"],
             # 三层 wire（p0521 修正 3：正面形态 + 语义 + 负面 guard；
             # 不发明关节位置——正面层只指向参考画面）
             "wire": {
                 "positive": "Preserve the exact upper-limb morphology "
                             "visible in <Picture {BODY_N}> throughout the "
                             "target video.",
                 "semantic": "This defining morphology includes the absence "
                             "of hands.",
                 "negative": "Do not synthesize hands, fingers, or "
                             "hand-shaped gloves for <Subject 1>."}},
        ],
        "section_appearance": {
            "S2": {"clothing": "与已确认比赛参考一致",
                   "headgear": "与已确认比赛参考一致"}},
    }


def validate_subject_constraints(constraints: dict[str, Any]) -> None:
    """结构校验：subject_id/critical_attributes 每条 id+description+source。"""
    if not constraints.get("subject_id"):
        raise LT0Blocked("constraints", "constraints_subject_missing")
    rows = constraints.get("critical_attributes") or []
    if not rows:
        raise LT0Blocked("constraints", "constraints_attributes_empty")
    for row in rows:
        for key in ("id", "description", "source"):
            if not str(row.get(key) or "").strip():
                raise LT0Blocked("constraints", "constraints_field_missing",
                                 f"{row.get('id') or '?'}:{key}")


def validate_evidence_quality(constraints: dict[str, Any],
                              pack: dict[str, Any]) -> None:
    """证据质量门：critical evidence 必须 clear/unambiguous/完整/人工批准。

    四环传递齐全但 Picture 2 本身证明不了属性（人太小/被挡/混入对手肢体）
    时，整条闭环建立在错误证据上——BLOCK BEFORE H3。
    """
    by_role = {str(entry.get("role")): entry
               for entry in pack.get("entries") or []}
    for row in constraints.get("critical_attributes") or []:
        for asset_id in row.get("evidence_asset_ids") or []:
            entry = by_role.get(str(asset_id))
            if entry is None:
                raise LT0Blocked("constraints", "evidence_asset_missing",
                                 f"{row['id']}:{asset_id}")
            quality = entry.get("evidence_quality") or {}
            for key, expected in EVIDENCE_QUALITY_REQUIRED.items():
                if quality.get(key) != expected:
                    raise LT0Blocked(
                        "constraints", "evidence_quality_gate",
                        f"{asset_id}:{key}={quality.get(key)!r} "
                        f"(need {expected!r}); 人工批准证据后重跑")


# 主体作用域动作校验（p0521 修正 2：不做全局词扫——禁止句与 C1 的合法
# 手部动作都会误触发；只检查主角的**肯定动作断言**）。
HAND_REQUIRED_ACTIONS = (
    "grab", "clench", "punch_with_hands", "catch", "push_with_hands",
    "handstand", "clap", "握拳", "抓住", "抓握", "撑地", "拍手", "握手",
)

def validate_subject_scoped_actions(brief: dict[str, Any],
                                    constraints: dict[str, Any]) -> None:
    """主角动作断言 vs 手部形态约束（只扫 subject1_action_claims，不扫
    negative guard / Subject 2 / retention——它们按构造不进入 claims）。"""
    no_hands = any("hand_morphology" in str(row.get("id") or "")
                   for row in constraints.get("critical_attributes") or [])
    if not no_hands:
        return
    claims = brief.get("subject1_action_claims") or []
    for claim in claims:
        action = str(claim.get("action") or "").lower()
        if any(word.lower() in action for word in HAND_REQUIRED_ACTIONS):
            raise LT0Blocked(
                "plan", "hand_action_contradicts_no_hands",
                f"subject1 claim '{claim.get('action')}' requires hands "
                f"({claim.get('source')})")


class LT0Blocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{stage}:{reason_code}:{detail}")
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail


def _ffprobe_bin_of(ffmpeg_bin: str) -> str:
    if "/" in ffmpeg_bin or "\\" in ffmpeg_bin:
        return str(Path(ffmpeg_bin).with_name("ffprobe"))
    return "ffprobe"


def detect_active_crop(ffmpeg_bin: str, video: Path, *, samples: int = 5,
                       duration_s: float | None = None,
                       dark_limit: int = 24) -> dict[str, int] | None:
    """检测持续黑边（人工审核：参考容器 9:16 但上下各约 130px 黑条，实际
    画面 ≈0.707≈3:4）。

    服务器 ffmpeg 4.2.7 不打印 cropdetect 逐帧行——改为导出灰度原始帧，
    纯 Python 逐行判黑（行平均亮度 < dark_limit 记为黑行），多帧采样取
    中位，左右列不扫（本用例 letterbox 只在上下）。"""
    if duration_s is None:
        duration_s = probe_media_geometry(
            _ffprobe_bin_of(ffmpeg_bin), video)["duration_s"]
    geometry = probe_media_geometry(_ffprobe_bin_of(ffmpeg_bin), video)
    width, height = geometry["width"], geometry["height"]
    tops, bottoms = [], []
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "frame.gray"
        for index in range(samples):
            timestamp = min(duration_s * (index + 0.5) / samples,
                            max(duration_s - 0.05, 0.05))
            result = subprocess.run(
                [ffmpeg_bin, "-y", "-loglevel", "error",
                 "-ss", f"{timestamp:.3f}", "-i", str(video),
                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray",
                 str(raw)], capture_output=True, timeout=120)
            if result.returncode or not raw.is_file():
                continue
            data = raw.read_bytes()
            if len(data) < width * height:
                continue
            def _row_mean(row: int) -> float:
                start = row * width
                return sum(data[start:start + width]) / width
            top = 0
            while top < height // 3 and _row_mean(top) < dark_limit:
                top += 1
            bottom = height
            while bottom > height * 2 // 3 and _row_mean(bottom - 1) < dark_limit:
                bottom -= 1
            if 0 < top and bottom < height:
                tops.append(top)
                bottoms.append(bottom)
            raw.unlink()
    if not tops:
        return None
    tops.sort()
    bottoms.sort()
    top, bottom = tops[len(tops) // 2], bottoms[len(bottoms) // 2]
    if bottom - top >= height - 8:  # 基本无黑边
        return None
    return {"w": width, "h": bottom - top, "x": 0, "y": top}


def _composed_crop_filter(active: dict[str, int] | None,
                          frac: dict[str, Any] | None,
                          full_width: int, full_height: int) -> str | None:
    """active 黑边盒 + 目标人物分数区域 → 单条 crop=w:h:x:y。"""
    base_w = active["w"] if active else full_width
    base_h = active["h"] if active else full_height
    base_x = active["x"] if active else 0
    base_y = active["y"] if active else 0
    if not frac:
        return f"crop={base_w}:{base_h}:{base_x}:{base_y}" if active else None
    left = base_x + int(base_w * float(frac["frac_x"][0]))
    right = base_x + int(base_w * float(frac["frac_x"][1]))
    top = base_y + int(base_h * float(frac["frac_y"][0]))
    bottom = base_y + int(base_h * float(frac["frac_y"][1]))
    return f"crop={max(8, right - left)}:{max(8, bottom - top)}:{left}:{top}"


def _extract_frame(ffmpeg_bin: str, reference: Path, timestamp: float,
                   jpg: Path, *, crop_filter: str | None = None) -> None:
    if not jpg.is_file():
        args = ["-y", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                "-i", str(reference)]
        if crop_filter:
            args += ["-vf", crop_filter]
        args += ["-frames:v", "1", "-q:v", "2", str(jpg)]
        common.run_ffmpeg(ffmpeg_bin, args, timeout_s=120)


def probe_media_geometry(ffprobe_bin: str, video: Path) -> dict[str, Any]:
    probe = common.run_ffprobe_json(ffprobe_bin, video)
    streams = [row for row in probe.get("streams") or []
               if row.get("codec_type") == "video"]
    if not streams:
        raise LT0Blocked("plan", "reference_probe_failed", str(video))
    stream = streams[0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "duration_s": float(probe.get("format", {}).get("duration") or 0.0)}


H3_ASPECT_RATIOS: dict[str, float] = {
    "9:16": 9 / 16, "3:4": 3 / 4, "1:1": 1.0,
    "4:3": 4 / 3, "16:9": 16 / 9, "21:9": 21 / 9,
}


def resolve_reference_aspect_ratio(width: int, height: int) -> str:
    """按参考视频实测比例映射到**最近**的 H3 合法 aspect（禁写死、禁
    auto——SGLang 的 auto 会 fallback 到 16:9）。8 条 Take 必须同一比例。"""
    if height <= 0:
        raise LT0Blocked("plan", "aspect_probe_invalid")
    ratio = width / height
    return min(H3_ASPECT_RATIOS,
               key=lambda name: abs(H3_ASPECT_RATIOS[name] - ratio))


def build_canonical_world_pack(
        reference: Path, output_dir: Path, *, candidate_count: int = 5,
        ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """每个 role 抽 candidate_count 个候选帧 + contact sheet，plan-only 阶段
    人工每 role 选 1 帧（防止八条实验建立在一张糊掉的"身份证照"上）。"""
    output_dir = Path(output_dir)
    geometry = probe_media_geometry(_ffprobe_bin_of(ffmpeg_bin), Path(reference))
    active = detect_active_crop(ffmpeg_bin, Path(reference),
                                duration_s=geometry["duration_s"])
    document: dict[str, Any] = {"schema_version": LT0_VERSION,
                                "reference": str(reference),
                                "container_geometry": geometry,
                                "active_crop": active,
                                "roles": []}
    for spec in PACK_ROLES:
        role_dir = output_dir / "canonical_candidates" / spec["role"]
        role_dir.mkdir(parents=True, exist_ok=True)
        low, high = spec["interval"]
        times = [round(low + (high - low) * (index + 0.5) / candidate_count, 3)
                 for index in range(candidate_count)]
        crop_filter = _composed_crop_filter(
            active, spec.get("crop"), geometry["width"], geometry["height"])
        candidates = []
        for index, timestamp in enumerate(times, 1):
            jpg = role_dir / f"candidate_{index:02d}.jpg"
            _extract_frame(ffmpeg_bin, Path(reference), timestamp, jpg,
                           crop_filter=crop_filter)
            candidates.append({"time_s": timestamp, "path": str(jpg),
                               "sha256": _file_hash(jpg)})
        contact = role_dir / "contact_sheet.jpg"
        inputs: list[str] = []
        for row in candidates:
            inputs += ["-i", row["path"]]
        common.run_ffmpeg(ffmpeg_bin, [
            "-y", "-loglevel", "error", *inputs, "-filter_complex",
            f"hstack=inputs={len(candidates)}", str(contact)], timeout_s=120)
        document["roles"].append({
            "role": spec["role"], "purpose": spec["purpose"],
            "interval": spec["interval"],
            "crop": spec.get("crop"), "crop_note": spec.get("crop_note"),
            "candidates": candidates,
            "contact_sheet": str(contact)})
    _write_json(output_dir / "canonical_pack_candidates.json", document)
    return document


def choose_canonical_pack(candidates: dict[str, Any], output_dir: Path,
                          *, picks: dict[str, float] | None = None
                          ) -> dict[str, Any]:
    """人工选帧落档（缺省取每 role 中位候选）；同一 pack 复用于全部 Take。"""
    picks = picks or {}
    entries = []
    for role_row in candidates.get("roles") or []:
        role = str(role_row["role"])
        chosen = picks.get(role)
        rows = role_row.get("candidates") or []
        if not rows:
            raise LT0Blocked("plan", "canonical_candidates_missing", role)
        if isinstance(chosen, str) and chosen.startswith("candidate_"):
            # 人工审核按编号选帧（p0510：zip 没带候选时间戳，编号比换算秒方便）
            index = int(chosen.split("_", 1)[1])
            if not 1 <= index <= len(rows):
                raise LT0Blocked("plan", "canonical_pick_out_of_candidates",
                                 f"{role}:{chosen}")
            chosen = float(rows[index - 1]["time_s"])
        if chosen is None:
            chosen = sorted(rows, key=lambda row: abs(
                float(row["time_s"]) - sum(
                    float(item["time_s"]) for item in rows) / len(rows)
            ))[0]["time_s"]
        match = [row for row in rows if abs(float(row["time_s"]) - chosen) < 1e-6]
        if not match:
            raise LT0Blocked("plan", "canonical_pick_out_of_candidates",
                             f"{role}:{chosen}")
        row = match[0]
        chosen_path = Path(row["path"])
        if not chosen_path.is_absolute():
            raise LT0Blocked("plan", "canonical_path_not_absolute",
                             f"{role}:{chosen_path}")
        entries.append({"role": role, "chosen_time_s": float(row["time_s"]),
                        "path": row["path"], "sha256": row["sha256"],
                        "uri": chosen_path.as_uri()})
    pack = {"schema_version": LT0_VERSION, "entries": entries}
    _write_json(Path(output_dir) / "canonical_world_pack.json", pack)
    return pack


def prepare_visual_reference(source_video: Path, output_dir: Path, *,
                             ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """H3 会把带音轨的 video 参考的音频也当参考——C/D 一律静音；同时裁掉
    持续黑边（容器 9:16 但上下各 ~130px 黑条，不裁则参考把 letterbox 也
    传给生成）。黑边裁剪需重编码（copy 不能裁）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "section_02_visual_ref.mp4"
    active = detect_active_crop(ffmpeg_bin, Path(source_video))
    args = ["-y", "-loglevel", "error", "-i", str(source_video)]
    if active:
        args += ["-vf",
                 f"crop={active['w']}:{active['h']}:{active['x']}:{active['y']}",
                 "-an", "-c:v", "libx264", "-crf", "18"]
    else:
        args += ["-an", "-c:v", "copy"]
    args.append(str(target))
    common.run_ffmpeg(ffmpeg_bin, args, timeout_s=300)
    return {"path": str(target), "sha256": _file_hash(target),
            "source": str(source_video), "active_crop": active,
            "source_sha256": _file_hash(Path(source_video))}


def compile_long_take_brief(content_program: dict[str, Any],
                            edit_program: dict[str, Any],
                            requirement: dict[str, Any],
                            section_observations: dict[str, Any], *,
                            section_id: str = "section_02",
                            constraints: dict[str, Any] | None = None,
                            output_dir: Path | None = None) -> dict[str, Any]:
    """P0 冻结产物 → H3 导演任务书（确定性模板装配，不再调 Omni，防二次理解漂移）。

    输出两层：structured brief（机器用）与 base_event_prompt（wire 用，各
    recipe 从同一 base 派生，只叠加各自 reference 子句）。
    """
    sections = {str(row.get("section_id")): row
                for row in content_program.get("sections") or []}
    patterns = {str(row.get("section_id")): row
                for row in edit_program.get("editorial_patterns") or []}
    requirements = {str(row.get("section_id")): row
                    for row in requirement.get("requirements") or []}
    observed = {str(row.get("section_id")): row
                for row in section_observations.get("sections") or []}
    for name, bank in (("content", sections), ("edit", patterns),
                       ("requirement", requirements),
                       ("observation", observed)):
        if section_id not in bank:
            raise LT0Blocked("plan", "section_missing_in_p0", f"{name}:{section_id}")
    section = sections[section_id]
    pattern = patterns[section_id]
    req = requirements[section_id]
    shots = (observed[section_id].get("shots") or [])
    phases = [str(item) for item in pattern.get("semantic_phases") or []]
    if not phases or not shots:
        raise LT0Blocked("plan", "brief_inputs_incomplete", section_id)
    phase_rows = []
    for phase in phases:
        indices = PHASE_SHOT_MAP.get(phase)
        if indices is None:  # 未知 phase 均分镜头
            bucket = max(1, math.ceil(len(shots) / len(phases)))
            order = phases.index(phase)
            indices = list(range(order * bucket, min((order + 1) * bucket,
                                                     len(shots))))
        evidence = [str(shots[index].get("information_added") or "").strip()
                    for index in indices if index < len(shots)]
        phase_rows.append({"phase": phase,
                           "source_shot_indices": [i for i in indices
                                                   if i < len(shots)],
                           "evidence_text": [row for row in evidence if row]})
    continuity = (req.get("continuity_requirement") or {}).get("levels") or {}
    snippet_range = ((req.get("presentation_requirement") or {})
                     .get("snippet_count_range") or [1, 1])
    budget = float((req.get("presentation_requirement") or {})
                   .get("target_duration_s") or 11.6)
    fact = str(section.get("reference_specific_fact") or "").strip()
    # 语义 phase 压缩（人工审核 p0510）：Edit Program 保留全部 shot 证据，
    # 生成 prompt 只保留**最小充分事件结构**——每 phase 一条指令（4 个
    # reflection shots → 1 行），否则模型以为"最后要花 7 秒连续微笑"。
    compressed = [{"phase": row["phase"],
                   "line": (row["evidence_text"][0] if row["evidence_text"]
                            else f"按参考推进 {row['phase']} 阶段")}
                  for row in phase_rows]
    weights = [PHASE_TIME_BUDGET_S.get(row["phase"], 1.0)
               for row in phase_rows]
    total_weight = sum(weights) or 1.0
    timeboxes = []
    cursor = 0.0
    for row, weight in zip(compressed, weights):
        span_end = cursor + weight / total_weight
        timeboxes.append({"phase": row["phase"],
                          "span_ratio": [round(cursor, 4),
                                         round(span_end, 4)],
                          "line": row["line"]})
        cursor = span_end
    subject_text = " ".join(
        fact.split("。"))[:400] if fact else "两位跆拳道选手：主角与对手。"
    # Prompt Compiler v3（人工审核 p0520）：subject 定义只写人物外观，不塞
    # 事件（事件归 detailed_description）；外观词确定性词表抽取，不调模型。
    all_texts = [fact] + [row for shot in phase_rows
                          for row in shot["evidence_text"]]
    c0_wardrobe = _wardrobe_english(*all_texts) or \
        ["a taekwondo dobok with protective gear"]
    c1_wardrobe = _wardrobe_english(
        *[row for row in all_texts if "对手" in row or "蓝" in row or "黑" in row]
    ) or ["a taekwondo dobok with protective gear"]
    brief = {
        "schema_version": LT0_VERSION, "section_id": section_id,
        "interval": section.get("interval"),
        "reference_specific_fact": fact,
        "subject_textual_definition": subject_text,
        "subject_wardrobe": {
            "subject_1": ", ".join(c0_wardrobe),
            "subject_2": ", ".join(c1_wardrobe)},
        "phases": phase_rows,
        "compressed_phase_lines": compressed,
        "phase_timeboxes": timeboxes,
        "required_continuity": continuity,
        "prompt_fields": {
            "subject_definitions": "{{SUBJECT_DEFINITIONS}}",
            "summary": ("[reference generation] Generate one coherent "
                        "{{SECONDS}}-second taekwondo sparring event clip "
                        "featuring <Subject 1> (the protagonist) and "
                        "<Subject 2> (the opponent). Preserve both subjects "
                        "throughout the entire clip."),
            "retention_analysis": "{{RETENTION_MARKERS}}",
            "detailed_description": "{{TIMEBOXES}}",
            "overall_soundscape": (
                "Natural ambient sound consistent with the depicted activity "
                "and setting: foot movement, protective-gear impacts, and "
                "restrained crowd reactions. No dialogue is required."),
            "non_diegetic_music": "None.",
        },
        "continuity_clause": (
            "Natural camera movement or internal cuts are allowed, but the "
            "clip must stay one complete, temporally continuous, causally "
            "coherent event: no switching to a different setting, a "
            "different match, or different people."),
        "editability_target": {
            "snippet_count_range": snippet_range,
            "moment_duration_range_s": [1.0, 3.0],
            "target_duration_s": budget,
            "composition_mode": (req.get("presentation_requirement") or {})
            .get("composition_mode")},
        # 生产化（p0521）：全局人物约束进 brief（不依附 Section）；主角
        # 动作断言按模板结构化登记，供主体作用域冲突检查。
        "subject_constraints": constraints or None,
        "subject1_action_claims": [
            {"action": action, "phase": box["phase"],
             "source": "phase_wire_template"}
            for box in timeboxes
            for action in PHASE_SUBJECT1_ACTIONS.get(box["phase"], [])],
    }
    if output_dir is not None:
        _write_json(Path(output_dir) / "h3_generation_brief.json", brief)
    return brief


def _timestamp(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}.000"


def _wire_shot_lines(brief: dict[str, Any], *, seconds: float,
                     with_subject_labels: bool) -> list[str]:
    """官方 shot/timeline grammar：[Shot N] At 00:0X.000 + 英文模板指令。

    模板参数化外观（词表抽取），不机械翻译模型中文证据；中文证据保留在
    brief JSON 里供审计。
    """
    wardrobe = brief.get("subject_wardrobe") or {}
    lines = []
    for index, box in enumerate(brief.get("phase_timeboxes") or [], 1):
        template = PHASE_WIRE_TEMPLATES.get(box["phase"])
        if template is None:
            template = (f"The {box['phase']} stage of the event unfolds "
                        "coherently with both subjects consistent.")
        line = template.replace("{c0_wardrobe}",
                                wardrobe.get("subject_1", "a taekwondo dobok"))
        if not with_subject_labels:
            line = (line.replace("<Subject 1>", "the protagonist")
                        .replace("<Subject 2>", "the opponent"))
        at = _timestamp(box["span_ratio"][0] * seconds)
        lines.append(f"[Shot {index}] At {at}, {line}")
    lines.append(str(brief.get("continuity_clause") or ""))
    return lines


def _retention_marker_lines(brief: dict[str, Any], recipe: str) -> list[str]:
    """官方 retention markers：每个 reference/subject 的保留与迁移关系。

    生产化（p0521）：critical attributes 逐条 fully_preserved；morphology
    证据图职责单一化（attribute_transfer - upper-limb morphology）。
    """
    use_canonical = recipe in {"canonical_only", "canonical_plus_video"}
    use_video = recipe in {"video_only", "canonical_plus_video"}
    wardrobe = brief.get("subject_wardrobe") or {}
    constraints = brief.get("subject_constraints") or {}
    lines = [
        "<Subject 1> (throughout): fully_preserved - preserve the "
        "protagonist's identity, body proportions, and wardrobe "
        f"({wardrobe.get('subject_1', 'a taekwondo dobok')}).",
        "<Subject 2> (throughout): fully_preserved - preserve the "
        "opponent's identity and wardrobe "
        f"({wardrobe.get('subject_2', 'a taekwondo dobok')}).",
        "Setting (throughout): fully_preserved - the same indoor taekwondo "
        "training hall.",
    ]
    for row in constraints.get("critical_attributes") or []:
        lines.append(
            f"{row['id']} (throughout): fully_preserved - "
            f"{row.get('description', '')}")
    if use_canonical:
        body_line = ("<Picture 2> (source for <Subject 1>): "
                     "attribute_transfer - transfer her reference-visible "
                     "upper-limb morphology.")
        lines[1:1] = [
            "<Picture 1> (source for <Subject 1>): attribute_transfer - "
            "transfer facial identity and headgear appearance.",
            body_line,
            "<Picture 3> (source for <Subject 2>): attribute_transfer - "
            "transfer appearance, clothing and protective gear.",
        ]
    if use_video:
        lines.append(
            "<Video 1>: attribute_transfer - transfer the overall event "
            "progression, attacking-defending relationship, major kicking "
            "motion, visible outcome, and camera energy. Do not copy exact "
            "cut timing or frames.")
    return lines


def _wire_prompt(brief: dict[str, Any], *, recipe: str, seconds: float,
                 pack: dict[str, Any] | None) -> str:
    """Prompt Compiler v3：A 用官方 T2VA 三字段 base schema；B/C/D 用官方
    Ref2VA 六段 schema（[reference generation] + retention markers）。
    实验语义相同，prompt grammar 服从对应 checkpoint 的官方格式。
    """
    use_canonical = recipe in {"canonical_only", "canonical_plus_video"}
    use_video = recipe in {"video_only", "canonical_plus_video"}
    wardrobe = brief.get("subject_wardrobe") or {}
    if recipe == "prompt_only":
        # T2VA 三字段（base schema）：无 subject_definitions/retention。
        shot_lines = _wire_shot_lines(brief, seconds=seconds,
                                      with_subject_labels=False)
        return "\n\n".join([
            "integrated_multimodal_description:\n" + "\n\n".join(shot_lines),
            "overall_soundscape:\n" +
            brief["prompt_fields"]["overall_soundscape"],
            "non_diegetic_music:\nN/A"])
    subject_lines = []
    constraints = brief.get("subject_constraints") or {}
    if use_canonical:
        if constraints.get("critical_attributes"):
            # 三层 morphology 块（p0521 修正 3）：正面形态（指向参考画面，
            # 不发明关节位置）+ 语义说明 + 负面 guard。
            subject_lines.append(
                "<Subject 1> is the adult woman defined by <Picture 1> "
                "for facial identity and <Picture 2> for her "
                "reference-visible upper-limb morphology.")
            for row in constraints["critical_attributes"]:
                wire = row.get("wire") or {}
                for key in ("positive", "semantic", "negative"):
                    text = str(wire.get(key) or "").replace("{BODY_N}", "2")
                    if text:
                        subject_lines.append(text)
            subject_lines.append(
                "<Subject 2> is the opponent, defined by <Picture 3>. "
                "Do not transfer <Subject 1>'s body attributes to "
                "<Subject 2>.")
        else:
            subject_lines.append(
                "<Subject 1> is the same protagonist, defined jointly by "
                "<Picture 1> (facial identity and headgear appearance) and "
                "<Picture 2> (full-body proportions, uniform and protective "
                "gear).")
            subject_lines.append(
                "<Subject 2> is the opponent, defined by <Picture 3>.")
    else:
        subject_lines.append(
            "<Subject 1> is the protagonist: an adult taekwondo athlete "
            f"wearing {wardrobe.get('subject_1', 'a taekwondo dobok')}.")
        subject_lines.append(
            "<Subject 2> is the opposing taekwondo athlete wearing "
            f"{wardrobe.get('subject_2', 'a taekwondo dobok')}.")
    if use_video:
        subject_lines.append(
            "<Video 1> provides the overall temporal structure, sparring "
            "interaction, major motion progression, and camera energy of "
            "the reference event.")
    fields = dict(brief["prompt_fields"])
    fields["subject_definitions"] = "\n".join(subject_lines)
    fields["retention_analysis"] = "\n".join(
        _retention_marker_lines(brief, recipe))
    fields["summary"] = fields["summary"].replace(
        "{{SECONDS}}", f"{seconds:g}")
    fields["detailed_description"] = "\n\n".join(
        _wire_shot_lines(brief, seconds=seconds, with_subject_labels=True))
    sections = [f"{key}:\n{fields[key]}" for key in (
        "subject_definitions", "summary", "retention_analysis",
        "detailed_description", "overall_soundscape", "non_diegetic_music")]
    return "\n\n".join(sections)


def build_long_take_request(brief: dict[str, Any], *,
                            pack: dict[str, Any] | None,
                            visual_reference: dict[str, Any] | None,
                            recipe: str, seed: int, seconds: float,
                            aspect_ratio: str) -> dict[str, Any]:
    """recipe ∈ {prompt_only, canonical_only, video_only, canonical_plus_video}；
    A 无 references（t2va，跑 fl2va server），B/C/D 走 ref2va。"""
    if recipe not in RECIPES:
        raise LT0Blocked("plan", "recipe_unknown", recipe)
    references: list[dict[str, Any]] = []
    use_canonical = recipe in {"canonical_only", "canonical_plus_video"}
    use_video = recipe in {"video_only", "canonical_plus_video"}
    if use_canonical:
        if not pack or not (pack.get("entries") or []):
            raise LT0Blocked("plan", "canonical_pack_missing", recipe)
        for entry in pack["entries"]:
            references.append({"type": "image", "uri": entry["uri"]})
    if use_video:
        if not visual_reference:
            raise LT0Blocked("plan", "visual_reference_missing", recipe)
        visual_path = Path(visual_reference["path"])
        if not visual_path.is_absolute():
            raise LT0Blocked("plan", "visual_reference_not_absolute",
                             str(visual_path))
        references.append({"type": "video", "uri": visual_path.as_uri()})
    prompt = _wire_prompt(brief, recipe=recipe, seconds=float(seconds),
                          pack=pack)
    request = build_h3_request(
        prompt=prompt, duration_s=float(seconds),
        references=references or None, first_frame=None, last_frame=None,
        capabilities=None, seed=int(seed), aspect_ratio=aspect_ratio)
    request["_lt0"] = {"recipe": recipe, "seed": int(seed),
                       "seconds": float(seconds), "version": LT0_VERSION,
                       "server_variant": "fl2va" if not references
                       else "ref2va"}
    types = {row.get("type") for row in references}
    if len(types) > 1:
        # 混合条件路由备注（v9g route_h3_mode 取排序首位的 capability 名，
        # 只反映单类；审计按条件组合重标）。
        request["_lt0"]["condition_mix"] = \
            "+".join(sorted(str(item) for item in types))
        request["_lt0"]["route_override_note"] = "ref2va_mixed"
    return request


def validate_lt0_wire_prompt(request: dict[str, Any]) -> None:
    """执行前 sanity check（人工审核 p0520 定稿，替代下一轮 plan 人工审）：

    A：task=t2va、零条件、T2VA 三字段 schema（不得出现 Ref2VA 六段）；
    B/C/D：task=ref2va、六段齐全、summary 含 [reference generation]、
    retention 至少一个合法 marker、<Picture>/<Video> 标签与条件顺序匹配。
    """
    recipe = str((request.get("_lt0") or {}).get("recipe") or "")
    prompt = str(request.get("prompt") or "")
    conditions = request.get("conditions") or []
    if recipe == "prompt_only":
        if request.get("task") != "t2va" or conditions:
            raise LT0Blocked("plan", "wire_check_t2va_contract", recipe)
        if "integrated_multimodal_description:" not in prompt:
            raise LT0Blocked("plan", "wire_check_t2va_schema", recipe)
        if "subject_definitions:" in prompt or "<Picture" in prompt or \
                "<Video" in prompt:
            raise LT0Blocked("plan", "wire_check_t2va_leak", recipe)
        return
    if request.get("task") != "ref2va" or not conditions:
        raise LT0Blocked("plan", "wire_check_ref2va_contract", recipe)
    for section in ("subject_definitions:", "summary:", "retention_analysis:",
                    "detailed_description:", "overall_soundscape:",
                    "non_diegetic_music:"):
        if section not in prompt:
            raise LT0Blocked("plan", "wire_check_section_missing",
                             f"{recipe}:{section}")
    if "[reference generation]" not in prompt:
        raise LT0Blocked("plan", "wire_check_summary_marker", recipe)
    if not any(marker in prompt for marker in RETENTION_MARKERS):
        raise LT0Blocked("plan", "wire_check_retention_marker", recipe)
    if "No reference material" in prompt:
        raise LT0Blocked("plan", "wire_check_contradictory_retention", recipe)
    picture_labels = sorted({
        int(label) for label in re.findall(r"<Picture (\d+)>", prompt)})
    video_labels = sorted({
        int(label) for label in re.findall(r"<Video (\d+)>", prompt)})
    image_count = sum(1 for row in conditions if row.get("type") == "image")
    video_count = sum(1 for row in conditions if row.get("type") == "video")
    if picture_labels != list(range(1, image_count + 1)):
        raise LT0Blocked("plan", "wire_check_picture_label_mismatch", recipe)
    if video_labels != list(range(1, video_count + 1)):
        raise LT0Blocked("plan", "wire_check_video_label_mismatch", recipe)


LONG_TAKE_OBSERVE_PROMPT = """只观察这段视频画面本身可见的内容，不要推断生成
意图或期望结果，不要评价"应该怎样"。只输出一个 JSON 对象：
{"subject_consistency":true,"opponent_consistency":true,"scene_consistency":true,
 "switch_points":[{"at_s":0.0,"what_changed":"..."}],
 "motion_quality":"natural|stiff|erratic",
 "severe_artifacts":["..."],
 "event_timeline":[{"phase":"...","interval":[0.0,1.0],"supported":true}],
 "phase_order_valid":true,
 "usable_moments":[{"start":0.0,"end":1.0,"type":"..."}]}
字段说明：三个 consistency 布尔判断主体/对手/场景是否全程同一（看不清填
null 并在 switch_points 说明）；motion_quality 描述动作自然度；
severe_artifacts 列出畸形/闪烁/肢体错误等严重缺陷（没有就空数组）；
event_timeline 用 phase 名（initiation/action/resolution/reflection 之外可自
定）标出实际发生了什么、interval 用视频内秒数、supported 表示该段画面清晰
可剪；usable_moments 列出 1~3 秒、画面清楚、可直接剪进成片的片段。所有
数值必须落在视频时长内。"""


IDENTITY_CHECK_PROMPT = """前两张是同一位主角 C0 的标准参考图（第 1 张面部、
第 2 张全身），第 3 张起是某条生成视频按时间等距抽取的采样帧。逐张判断采样
帧中的主角是否与 C0 参考（而非对手或其他人）为同一人物。只输出一个 JSON
对象：{"frames":[{"sample_index":1,"c0_visible":"clear|blurred|back_turned|absent",
 "c0_match":true,"notes":"..."}],"c0_consistent":true}
c0_visible 为 blurred/back_turned/absent 时 c0_match 必须为 null（可见性不足
不判身份失败）。c0_consistent 只依据可见帧判定；全部不可见则填 null。"""


def _validate_observation(raw: str, duration_s: float) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    value = json.loads(text)
    timeline = value.get("event_timeline")
    if not isinstance(timeline, list) or not timeline:
        raise LT0Blocked("observe", "observation_timeline_missing")
    for row in timeline:
        interval = row.get("interval")
        if (not isinstance(interval, list) or len(interval) != 2 or
                not 0 <= float(interval[0]) < float(interval[1]) <=
                duration_s + 0.5):
            raise LT0Blocked("observe", "observation_interval_invalid",
                             json.dumps(interval, ensure_ascii=False))
    for key in ("subject_consistency", "opponent_consistency",
                "scene_consistency"):
        if key in value and value[key] is not None and \
                not isinstance(value[key], bool):
            raise LT0Blocked("observe", "observation_field_invalid", key)
    if not isinstance(value.get("usable_moments"), list):
        raise LT0Blocked("observe", "observation_moments_missing")
    return value


def observe_long_take(video_path: Path, output_dir: Path, *, runner,
                      take_id: str, pack: dict[str, Any],
                      ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """盲观察（中性 watch，不看任何参考）+ 跨 Take 身份检查（非盲，对
    canonical C0 锚比，不链式比）。"""
    output_dir = Path(output_dir)
    geometry = probe_media_geometry(_ffprobe_bin_of(ffmpeg_bin), Path(video_path))
    duration_s = geometry["duration_s"]
    answer = runner.watch(
        Path(video_path), LONG_TAKE_OBSERVE_PROMPT, fps=4.0,
        max_new_tokens=2048, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    raw_path = output_dir / f"{take_id}_observe.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")
    observation = _validate_observation(raw, duration_s)
    # 身份采样：5 帧等距 + canonical C0 两图（面部/全身）
    frame_dir = output_dir / f"{take_id}_identity_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = []
    for index, ratio in enumerate(IDENTITY_SAMPLE_RATIOS, 1):
        jpg = frame_dir / f"sample_{index}.jpg"
        _extract_frame(ffmpeg_bin, Path(video_path),
                       min(duration_s * ratio, max(duration_s - 0.05, 0.05)),
                       jpg)
        sample_paths.append(jpg)
    c0_entries = [entry for entry in pack.get("entries") or []
                  if str(entry.get("role", "")).startswith("c0_")]
    if not c0_entries:
        raise LT0Blocked("observe", "canonical_c0_missing")
    identity_answer = runner.inspect_media(
        [Path(entry["path"]) for entry in c0_entries] + sample_paths,
        IDENTITY_CHECK_PROMPT, max_new_tokens=1024,
        stop_after_json_object=True)
    identity_raw = getattr(identity_answer, "text", str(identity_answer))
    (output_dir / f"{take_id}_identity.txt").write_text(
        identity_raw, encoding="utf-8")
    identity_text = identity_raw.strip()
    if identity_text.startswith("```"):
        identity_text = identity_text.strip("`")
        if identity_text.startswith("json"):
            identity_text = identity_text[4:]
    identity = json.loads(identity_text)
    # 模型可能幻觉多余采样行（首跑实测：5 帧输入回了 6+ 行）——丢弃
    # 越界行，但给定样本必须全覆盖，缺号仍硬拦。
    frames = [row for row in (identity.get("frames") or [])
              if isinstance(row, dict)
              and isinstance(row.get("sample_index"), int)
              and 1 <= row["sample_index"] <= len(sample_paths)]
    covered = {row["sample_index"] for row in frames}
    if covered != set(range(1, len(sample_paths) + 1)):
        raise LT0Blocked("observe", "identity_frames_invalid", take_id)
    identity["frames"] = frames
    return {"take_id": take_id, "duration_s": duration_s,
            "observation": observation, "identity": identity}


ATTRIBUTE_CHECK_PROMPT = """前两张起是目标人物的已批准参考图（按输入顺序，
与人物卡中的证据对应），其后是生成视频按时间采样的帧。对人物卡中的每个
critical attribute，逐项核对**画面中归属于 C0 的身体部位**与批准参考：
只输出一个 JSON 对象：
{"attributes":[{"attribute_id":"...","verdict":"pass|fail|unknown",
  "ownership_confirmed":true,"evidence_interval":[0.0,1.0],
  "notes":"..."}],
 "scope_note":"checked spans only"}
判定规则（必须遵守）：
1. 先判归属：画面结构属于谁看不清时不得判 C0 的属性。
2. 看不到不等于确认：关键区域被遮挡/模糊/出画 → verdict=unknown，
   绝不是 pass（也不是 fail）。
3. fail 仅当：清楚看到属于 C0 的、与约束冲突的结构（例如清楚可见的
   手形结构且归属 C0）。
4. 对手（C1）正常有手不构成 C0 的 fail。
5. notes 必须具体到时间与画面；报告只声明已检查区间，不得宣称全片。"""


def check_take_attributes(video_path: Path, output_dir: Path, *, runner,
                           take_id: str, constraints: dict[str, Any],
                           pack: dict[str, Any],
                           sample_ratios: tuple[float, ...] = (0.1, 0.35, 0.55,
                                                               0.75, 0.9),
                           ffmpeg_bin: str = "ffmpeg"
                           ) -> dict[str, Any]:
    """第二段核对（p0521）：人物卡+参考图+采样帧 → 逐属性 pass/fail/unknown。

    采样=前/中/后/动作关键位；五帧不宣称全片证明（scope_note 强制声明）。
    可疑处补看（drill）由生产控制器按需追加，本函数只做基础核对。
    """
    output_dir = Path(output_dir)
    geometry = probe_media_geometry(_ffprobe_bin_of(ffmpeg_bin), Path(video_path))
    duration_s = geometry["duration_s"]
    frame_dir = output_dir / f"{take_id}_attribute_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = []
    for index, ratio in enumerate(sample_ratios, 1):
        jpg = frame_dir / f"sample_{index}.jpg"
        _extract_frame(ffmpeg_bin, Path(video_path),
                       min(duration_s * ratio, max(duration_s - 0.05, 0.05)),
                       jpg)
        sample_paths.append(jpg)
    by_role = {str(entry.get("role")): entry
               for entry in pack.get("entries") or []}
    reference_paths = []
    attributes_payload = []
    for row in constraints.get("critical_attributes") or []:
        attributes_payload.append({
            "attribute_id": row["id"], "description": row["description"]})
        for asset_id in row.get("evidence_asset_ids") or []:
            entry = by_role.get(str(asset_id))
            if entry and Path(str(entry["path"])).is_file():
                reference_paths.append(Path(str(entry["path"])))
    prompt = (ATTRIBUTE_CHECK_PROMPT + "\n人物卡 critical attributes：\n"
              + json.dumps(attributes_payload, ensure_ascii=False,
                           separators=(",", ":")) + "\n输入：")
    answer = runner.inspect_media(
        reference_paths + sample_paths, prompt, max_new_tokens=1536,
        stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    (output_dir / f"{take_id}_attributes.txt").write_text(
        raw, encoding="utf-8")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    value = json.loads(text)
    rows = value.get("attributes") or []
    expected = {row["id"] for row in attributes_payload}
    seen: set[str] = set()
    for row in rows:
        verdict = row.get("verdict")
        if verdict not in {"pass", "fail", "unknown"}:
            raise LT0Blocked("observe", "attribute_verdict_invalid",
                             str(row.get("attribute_id")))
        seen.add(str(row.get("attribute_id")))
    if seen != expected:
        raise LT0Blocked("observe", "attribute_missing",
                         f"{sorted(expected - seen)}")
    return {"take_id": take_id, "duration_s": duration_s,
            "attribute_verdicts": {row["attribute_id"]: row
                                   for row in rows},
            "samples": [str(path) for path in sample_paths],
            "scope_note": value.get("scope_note") or "checked spans only"}


def compare_take_to_brief(result: dict[str, Any],
                          brief: dict[str, Any]) -> dict[str, Any]:
    """三分离结果（VBench 思想：质量/覆盖/可剪不压成一个分）。
    partial success 保留：质量好但缺 phase 的 Take 仍可进素材库。"""
    observation = result["observation"]
    identity = result["identity"]
    required_phases = [row["phase"] for row in brief["phases"]]
    timeline = observation.get("event_timeline") or []
    phases_supported = sorted({str(row.get("phase")) for row in timeline
                               if row.get("supported") is True})
    missing = [phase for phase in required_phases
               if phase not in phases_supported]
    order_valid = observation.get("phase_order_valid") is True
    moments = observation.get("usable_moments") or []
    usable_duration = round(sum(
        max(0.0, float(row.get("end", 0)) - float(row.get("start", 0)))
        for row in moments), 3)
    budget = float((brief.get("editability_target") or {}).get(
        "target_duration_s") or 11.6)
    # 预算取 max(参考预算, 本条 take 实际时长)：12s take 的四 phase 最短
    # 区间合计 12.0s vs 参考 11.6s——0.4s "超支" 是对比框架边界 artifact，
    # 不是覆盖失败（phase 全在且有序；真预算约束在最终 montage 阶段）。
    budget = max(budget, float(result.get("duration_s") or 0.0))
    selection = select_evidence_moments(
        timeline, required_phases=required_phases, duration_budget_s=budget,
        composition_mode="event_compression_montage")
    coverage_pass = (not missing and order_valid and selection["passed"])
    artifacts = observation.get("severe_artifacts") or []
    consistency_fields = [observation.get("subject_consistency"),
                          observation.get("opponent_consistency"),
                          observation.get("scene_consistency")]
    quality_pass = (all(value is True for value in consistency_fields) and
                    not artifacts and
                    observation.get("motion_quality") == "natural")
    frames = identity.get("frames") or []
    visible = [row for row in frames
               if row.get("c0_visible") == "clear"]
    matched = [row for row in visible if row.get("c0_match") is True]
    mismatched = [row for row in visible if row.get("c0_match") is False]
    if visible:
        cross_take_pass = not mismatched and len(matched) >= max(1, len(visible) // 2)
    else:
        cross_take_pass = None  # 全部可见性不足：无法判定，不判失败
    return {
        "quality": {
            "subject_consistency": consistency_fields[0],
            "opponent_consistency": consistency_fields[1],
            "scene_consistency": consistency_fields[2],
            "severe_artifacts": artifacts,
            "motion_quality": observation.get("motion_quality"),
            "cross_take_identity": cross_take_pass,
            "take_quality_pass": quality_pass,
        },
        "coverage": {
            "phases_supported": phases_supported,
            "missing_phases": missing,
            "phase_order_valid": order_valid,
            "moment_selection": {key: selection.get(key) for key in
                                 ("passed", "reason_code", "duration_s")},
            "full_event_coverage_pass": coverage_pass,
        },
        "editability": {
            "usable_moment_count": len(moments),
            "usable_duration_s": usable_duration,
            "usable_for_editing": len(moments) >= 3 and usable_duration >= 5.0,
        },
    }


# -*- coding: utf-8 -*-
"""拼接 LT0 编排层到 generation_long_take.py。"""


def _endpoint_reachable(endpoint: str, *, timeout_s: float = 5.0) -> bool:
    import requests
    try:
        requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=timeout_s)
        return True
    except Exception:
        return False


def _ensure_h3_server(endpoint: str, *, backend: str, variant: str,
                      manage_server: bool, port: int, repo_root: Path,
                      log_path: Path, gpu_set: str = "0,1,6,7",
                      python_bin: str | None = None
                      ) -> subprocess.Popen | None:
    """H3 与 Omni 错峰占同一组卡：必要时由实验自起 server（记录 owned
    进程组，任务后回收）。

    backend=sglang：serve_h3.sh variant fl2va|ref2va（需驱动 ≥580 的 cu13
    栈，本机驱动 575 不可用）；backend=diffusers：统一图 serve
    （t2va/ref2va 同服，官方 /v1/videos 契约）——LT0 现行后端。"""
    if _endpoint_reachable(endpoint):
        return None
    if not manage_server:
        raise LT0Blocked("h3", "h3_endpoint_unreachable", endpoint)
    import os
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    if backend == "diffusers":
        command = [python_bin or sys.executable, "-m",
                   "src.generation.minimax_h3_ref2va_serve",
                   "--port", str(port)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_set}
    else:
        script = repo_root / "scripts" / "v9g" / "serve_h3.sh"
        if not script.is_file():
            raise LT0Blocked("h3", "serve_script_missing", str(script))
        command = ["bash", str(script), variant, str(port)]
        env = None
    process = subprocess.Popen(
        command, cwd=str(repo_root), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env)
    import time
    deadline = time.monotonic() + 3600.0
    while time.monotonic() < deadline:
        if _endpoint_reachable(endpoint, timeout_s=10.0):
            return process
        if process.poll() is not None:
            raise LT0Blocked("h3", "h3_server_died_early",
                             log_path.read_text(encoding="utf-8")[-500:])
        time.sleep(30.0)
    raise LT0Blocked("h3", "h3_server_timeout", endpoint)


def _stop_owned_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    import os
    import signal
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        import os as _os
        _os.killpg(_os.getpgid(process.pid), signal.SIGKILL)


def _load_p04e(p04e_dir: Path) -> dict[str, Any]:
    loaders = {"content_program": "reference_content_program.json",
               "edit_program": "reference_edit_program.json",
               "requirement": "material_requirements.json",
               "section_observations": "section_observations.json"}
    loaded = {}
    for key, name in loaders.items():
        path = Path(p04e_dir) / name
        if not path.is_file():
            raise LT0Blocked("plan", "p04e_artifact_missing", name)
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def run_lt0_experiment(cfg: Any, p04e_dir: Path, output_dir: Path, *,
                       reference: Path,
                       plan_only: bool = False, execute: bool = False,
                       ref2va_endpoint: str = "http://127.0.0.1:30011",
                       fl2va_endpoint: str = "http://127.0.0.1:30011",
                       gpu_set: str = "0,1,6,7",
                       seconds: float = DEFAULT_SECONDS,
                       recipes: tuple[str, ...] = RECIPES,
                       seeds: tuple[int, ...] = DEFAULT_SEEDS,
                       pack_picks: dict[str, float] | None = None,
                       gpu_pairs: str = "0,1;6,7",
                       manage_server: bool = True,
                       h3_backend: str = "diffusers",
                       h3_python_bin: str | None = None,
                       ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """三段顺序 GPU 占用：CPU 计划（--plan-only 止于此）→ H3 生成 → Omni
    盲观察+对比 → lt0_report.json。LT0 为能力实验：P0 冻结与正式 V9-G
    license 门不阻塞本实验，但 --execute 须显式给出。"""
    import time

    from src.agentic_video.generation_p51 import probe_gpu_preflight
    from src.perception.omni_pool import OmniProcessPool

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    p04e = _load_p04e(Path(p04e_dir).resolve())
    section_clip = Path(p04e_dir) / "review_assets" / "section_02.mp4"
    if not section_clip.is_file():
        raise LT0Blocked("plan", "section_clip_missing", str(section_clip))

    # ---- CPU 阶段：canonical 候选 / pack / 静音参考 / brief / 8 请求 ----
    candidates = build_canonical_world_pack(
        Path(reference), output_dir, ffmpeg_bin=ffmpeg_bin)
    pack = choose_canonical_pack(candidates, output_dir, picks=pack_picks)
    visual_reference = prepare_visual_reference(section_clip, output_dir,
                                                ffmpeg_bin=ffmpeg_bin)
    brief = compile_long_take_brief(
        p04e["content_program"], p04e["edit_program"], p04e["requirement"],
        p04e["section_observations"], output_dir=output_dir)
    # aspect 按**有效画面**算（人工审核 p0510：容器 9:16 但上下各 ~130px
    # 黑条，实际 ≈0.707≈3:4；8 条 Take 同一比例，全部像素生成有效内容）。
    geometry = probe_media_geometry(_ffprobe_bin_of(ffmpeg_bin), section_clip)
    active = (visual_reference or {}).get("active_crop")
    active_w = int(active["w"]) if active else geometry["width"]
    active_h = int(active["h"]) if active else geometry["height"]
    aspect_ratio = resolve_reference_aspect_ratio(active_w, active_h)
    jobs = []
    for seed in seeds:
        for recipe in recipes if recipes else RECIPES:
            take_id = f"{recipe}_s{seed}"
            request = build_long_take_request(
                brief, pack=pack, visual_reference=visual_reference,
                recipe=recipe, seed=seed, seconds=seconds,
                aspect_ratio=aspect_ratio)
            validate_lt0_wire_prompt(request)
            jobs.append({"take_id": take_id, "recipe": recipe,
                         "seed": seed, "request": request})
    plan = {"schema_version": LT0_VERSION, "execute_authorized": bool(execute),
            "experiment_note": "LT0 为 H3 能力实验；正式 V9-G license 门另行签署",
            "aspect_ratio": aspect_ratio, "seconds": float(seconds),
            "aspect_basis": {"container": geometry, "active_crop": active,
                             "active": [active_w, active_h]},
            "seeds": list(seeds),
            "recipes": list(recipes or RECIPES),
            "server_routing": {"prompt_only": "fl2va (t2va task)",
                               "others": "ref2va",
                               "note": "A 为 T2VA baseline，非严格同 "
                                       "checkpoint ablation；干净的 "
                                       "ablation 是 B vs C vs D"},
            "canonical_pack": pack, "visual_reference": visual_reference,
            "brief_phases": [row["phase"] for row in brief["phases"]],
            "jobs": [{"take_id": job["take_id"],
                      "request_path": "takes/{}/request.json".format(
                          job["take_id"])}
                     for job in jobs]}
    _write_json(output_dir / "lt0_plan.json", plan)
    for job in jobs:
        job_dir = output_dir / "takes" / job["take_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_json(job_dir / "request.json", job["request"])
        (job_dir / "prompt.txt").write_text(
            str(job["request"].get("prompt") or ""), encoding="utf-8")
    if plan_only or not execute:
        return {"phase": "planned",
                "plan_path": str(output_dir / "lt0_plan.json"),
                "aspect_ratio": aspect_ratio,
                "contact_sheets": [row["contact_sheet"]
                                   for row in candidates["roles"]],
                "take_count": len(jobs)}

    # ---- H3 阶段（错峰占卡；A 与 B/C/D 分别跑 fl2va / ref2va server）----
    required_indices = [int(item) for item in gpu_set.split(",") if item.strip()]
    preflight = probe_gpu_preflight(required_indices)
    if preflight["status"] != "READY":
        raise LT0Blocked("h3", "BLOCKED_GPU_BUSY",
                         json.dumps(preflight.get("gpus") or [],
                                    ensure_ascii=False))
    repo_root = Path(__file__).resolve().parents[2]
    take_results = []
    take_results_lock = threading.Lock()

    def _parse_endpoints(raw: str) -> list[str]:
        return [item.strip().rstrip("/") for item in str(raw).split(",")
                if item.strip()]

    def _gpu_pairs_for(endpoint_count: int) -> list[str]:
        """gpu_set 按序切成每端点一对（2 卡/实例，Phase-4 同款配方）。"""
        indices = [item.strip() for item in gpu_set.split(",") if item.strip()]
        if len(indices) != 2 * endpoint_count:
            raise LT0Blocked(
                "h3", "gpu_endpoint_mismatch",
                f"{len(indices)} GPUs for {endpoint_count} endpoints "
                f"(each diffusers instance needs exactly 2)")
        return [",".join(indices[index:index + 2])
                for index in range(0, len(indices), 2)]

    def _run_job(client: SGLangH3Client, job: dict[str, Any],
                 variant: str) -> dict[str, Any]:
        job_dir = output_dir / "takes" / job["take_id"]
        state = {"take_id": job["take_id"], "recipe": job["recipe"],
                 "seed": job["seed"], "variant": variant,
                 "state": "planned"}
        _write_json(job_dir / "job_state.json", state)
        try:
            payload = client.serialize_payload(job["request"])
            created = client.submit(payload)
            state.update({"state": "submitted", "job_id": created.get("id"),
                          "accepted_conditions": created.get(
                              "accepted_conditions")})
            _write_json(job_dir / "backend_response.json", created)
            completed = client.poll(str(created["id"]))
            state.update({"state": "completed", "status_payload": completed})
            media = client.download(str(created["id"]), job_dir / "take.mp4")
            state.update({"state": "downloaded", **media})
            state["ffprobe"] = common.run_ffprobe_json(
                _ffprobe_bin_of(ffmpeg_bin), Path(media["output_path"]))
            state["state"] = "transport_validated"
        except V9GBlocked as exc:
            state.update({"state": "failed", "reason_code": exc.reason_code,
                          "detail": str(exc)[:500]})
        _write_json(job_dir / "job_state.json", state)
        return state

    def _lane(endpoint: str, work: "queue.Queue[dict[str, Any]]",
              variant: str) -> None:
        client = SGLangH3Client(endpoint)
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            state = _run_job(client, job, variant)
            with take_results_lock:
                take_results.append(state)

    def _run_group(group_jobs: list[dict[str, Any]], endpoints_raw: str,
                   variant: str, *, stop_after: bool = True) -> None:
        endpoints = _parse_endpoints(endpoints_raw)
        # resume 先行：已 transport_validated 且成片在盘的 take 直接复用，
        # 全部可复用时根本不起 serve（观察-only 重跑不再空烧加载）。
        pending: list[dict[str, Any]] = []
        for job in group_jobs:
            prior_path = output_dir / "takes" / job["take_id"] / \
                "job_state.json"
            prior: dict[str, Any] = {}
            if prior_path.is_file():
                try:
                    prior = json.loads(prior_path.read_text(encoding="utf-8"))
                except ValueError:
                    prior = {}
            if (prior.get("state") == "transport_validated" and
                    Path(str(prior.get("output_path") or "")).is_file()):
                prior["variant"] = variant
                take_results.append(prior)
            else:
                pending.append(job)
        if not pending:
            return
        pairs = _gpu_pairs_for(len(endpoints))
        servers: list[tuple[str, subprocess.Popen | None]] = []
        try:
            for endpoint, pair in zip(endpoints, pairs):
                port = endpoint.rsplit(":", 1)[-1].split("/")[0]
                servers.append((endpoint, _ensure_h3_server(
                    endpoint, backend=h3_backend, variant=variant,
                    manage_server=manage_server, port=int(port),
                    repo_root=repo_root,
                    log_path=output_dir /
                    f"h3_server_{variant}_{port}.log",
                    gpu_set=pair, python_bin=h3_python_bin)))
            work: "queue.Queue[dict[str, Any]]" = queue.Queue()
            for job in pending:
                work.put(job)
            if len(endpoints) == 1:
                _lane(endpoints[0], work, variant)
            else:
                threads = [threading.Thread(target=_lane,
                                             args=(endpoint, work, variant))
                           for endpoint in endpoints]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        finally:
            if stop_after:
                for _, process in servers:
                    _stop_owned_server(process)

    if (_parse_endpoints(fl2va_endpoint) == _parse_endpoints(ref2va_endpoint)
            or "prompt_only" not in (recipes or RECIPES)):
        # 统一 diffusers serve 同时服务 t2va 与 ref2va：全部 take 一组跑完
        # （多端点时每实例一对卡并行车道）；定点重跑（无 A）同理。
        _run_group(jobs, ref2va_endpoint, "unified", stop_after=True)
    else:
        _run_group([job for job in jobs if job["recipe"] == "prompt_only"],
                   fl2va_endpoint, "fl2va")
        _run_group([job for job in jobs if job["recipe"] != "prompt_only"],
                   ref2va_endpoint, "ref2va")

    # ---- Omni 阶段：盲观察 + 身份锚比对 + 三分离对比 ----
    omni_cfg = (getattr(cfg, "perception", None) or {}).get("omni") or {}
    observations = []
    with OmniProcessPool(gpu_pairs, omni_cfg, ffmpeg_bin=ffmpeg_bin,
                         response_timeout_s=3600.0) as runner:
        for state in take_results:
            if state.get("state") != "transport_validated":
                continue
            result = observe_long_take(
                Path(state["output_path"]), output_dir, runner=runner,
                take_id=state["take_id"], pack=pack, ffmpeg_bin=ffmpeg_bin)
            comparison = compare_take_to_brief(result, brief)
            row = {"take_id": state["take_id"], "recipe": state["recipe"],
                   "seed": state["seed"], **comparison}
            observations.append(row)
            _write_json(output_dir / "takes" / state["take_id"] /
                        "comparison.json", row)

    matrix: dict[str, Any] = {}
    for row in observations:
        recipe = row["recipe"]
        matrix.setdefault(recipe, {})["seed_{}".format(row["seed"])] = {
            "take_quality_pass": row["quality"]["take_quality_pass"],
            "full_event_coverage_pass": row["coverage"][
                "full_event_coverage_pass"],
            "usable_for_editing": row["editability"]["usable_for_editing"],
            "usable_moment_count": row["editability"]["usable_moment_count"],
            "cross_take_identity": row["quality"]["cross_take_identity"]}
    summary = {}
    for recipe, rows in matrix.items():
        values = list(rows.values())
        summary[recipe] = {
            "take_quality_pass_count": sum(
                1 for value in values if value["take_quality_pass"]),
            "full_event_coverage_pass_count": sum(
                1 for value in values if value["full_event_coverage_pass"]),
            "usable_for_editing_count": sum(
                1 for value in values if value["usable_for_editing"])}
    report = {"schema_version": LT0_VERSION,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "experiment": "LT0-Q1 same-world resynthesis",
              "recipe_labels": {"prompt_only": "T2VA baseline (fl2va "
                              "server)——非严格同 checkpoint ablation；"
                              "干净 ablation 是 B vs C vs D"},
              "take_states": [{key: state.get(key) for key in
                               ("take_id", "recipe", "seed", "state",
                                "variant", "reason_code")}
                              for state in take_results],
              "matrix": matrix, "summary": summary,
              "interpretation_hints": [
                  "若 C≈D：same-world 下完整参考视频已锚牢人物/场景，图片条件可省",
                  "若 D 劣于 C：condition 过多可能互相干扰",
                  "结论仅适用于 same-world resynthesis；跨身份迁移（LT1/Q2）另测"],
              "limits": ["First Frame 归 LT0.1", "音频参考对比归后续",
                         "12 vs 15 时长退化单独实验",
                         "A 组跑在 fl2va checkpoint 上，只作 T2VA baseline"]}
    _write_json(output_dir / "lt0_report.json", report)
    return {"phase": "complete",
            "report_path": str(output_dir / "lt0_report.json"),
            "summary": summary}
