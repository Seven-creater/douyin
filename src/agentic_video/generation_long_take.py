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
import subprocess
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
PACK_ROLES: tuple[dict[str, Any], ...] = (
    {"role": "c0_identity", "interval": [14.2, 16.0],
     "purpose": "主角面部清晰帧",
     "crop": {"frac_x": [0.15, 0.85], "frac_y": [0.05, 0.75]},
     "crop_note": "裁成头肩/上半身，去掉背景观众"},
    {"role": "c0_fullbody", "interval": [5.3, 5.9],
     "purpose": "主角全身（含护具）帧",
     "crop": {"frac_x": [0.50, 1.00], "frac_y": [0.0, 1.0]},
     "crop_note": "只裁右侧 C0（原帧左 C1/中裁判/右 C0）"},
    {"role": "c1_opponent", "interval": [6.4, 7.2],
     "purpose": "对手清晰帧",
     "crop": {"frac_x": [0.00, 0.58], "frac_y": [0.0, 1.0]},
     "crop_note": "只裁左侧 C1"},
)
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
                       duration_s: float | None = None) -> dict[str, int] | None:
    """检测持续黑边（人工审核：参考容器 9:16 但上下各约 130px 黑条，实际
    画面 ≈0.707≈3:4）。ffmpeg cropdetect 采样取中位，返回 crop 盒或 None。"""
    if duration_s is None:
        duration_s = probe_media_geometry(
            _ffprobe_bin_of(ffmpeg_bin), video)["duration_s"]
    boxes = []
    for index in range(samples):
        timestamp = min(duration_s * (index + 0.5) / samples,
                        max(duration_s - 0.05, 0.05))
        result = subprocess.run(
            [ffmpeg_bin, "-ss", f"{timestamp:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", "cropdetect=round=2:limit=24",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        for line in result.stderr.splitlines():
            marker = "crop="
            if marker in line:
                # cropdetect 输出形如 crop=w:h:x:y[:aspect]（裸冒号值）
                values = line.split(marker, 1)[1].strip().split(":")
                try:
                    boxes.append({"w": int(values[0]), "h": int(values[1]),
                                  "x": int(values[2]), "y": int(values[3])})
                except (IndexError, ValueError):
                    continue
                break
    if not boxes:
        return None
    median = {}
    for key in ("w", "h", "x", "y"):
        values = sorted(box[key] for box in boxes)
        median[key] = values[len(values) // 2]
    return median


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
    # 官方 H3-Context-IR 六段结构（人工审核：Ref2VA Prompt Guide 要求
    # subject_definitions/summary/retention_analysis/detailed_description/
    # overall_soundscape/non_diegetic_music；角色引用要写 <Subject N>，
    # 且同一 Subject 可由多个 reference assets 共同定义）。
    brief = {
        "schema_version": LT0_VERSION, "section_id": section_id,
        "interval": section.get("interval"),
        "reference_specific_fact": fact,
        "subject_textual_definition": subject_text,
        "phases": phase_rows,
        "compressed_phase_lines": compressed,
        "phase_timeboxes": timeboxes,
        "required_continuity": continuity,
        "prompt_fields": {
            "subject_definitions": "{{SUBJECT_DEFINITIONS}}",
            "summary": ("Generate one coherent {{SECONDS}}-second taekwondo "
                        "sparring event clip featuring <Subject 1> (the "
                        "protagonist) and <Subject 2> (the opponent). "
                        "Preserve both subjects throughout the entire clip."),
            "retention_analysis": (
                "Preserve:\n"
                "- identity, body proportions, uniforms and protective gear "
                "of both subjects\n"
                "- the same indoor taekwondo training-hall setting\n"
                "{{RETENTION_REFERENCE}}"),
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
    }
    if output_dir is not None:
        _write_json(Path(output_dir) / "h3_generation_brief.json", brief)
    return brief


def _wire_prompt(brief: dict[str, Any], *, recipe: str, seconds: float,
                 pack: dict[str, Any] | None) -> str:
    """按官方 H3-Context-IR 六段结构组装 wire prompt。

    所有 recipe 共用同一 brief 字段，只有 subject_definitions 与
    retention 的 reference 段随 conditioning 变化——比较的只有条件本身。
    <Subject 1> 由 Picture 1+2 共同定义、<Subject 2> 由 Picture 3 定义
    （官方 Guide：同一 Subject 可由多个 reference assets 联合定义）。
    """
    textual = str(brief.get("subject_textual_definition") or "")
    use_canonical = recipe in {"canonical_only", "canonical_plus_video"}
    use_video = recipe in {"video_only", "canonical_plus_video"}
    subject_lines = []
    if use_canonical:
        subject_lines.append(
            "<Subject 1> is the same protagonist, defined jointly by "
            "<Picture 1> (facial identity and headgear appearance) and "
            "<Picture 2> (full-body proportions, uniform and protective "
            "gear).")
        subject_lines.append(
            "<Subject 2> is the opponent, defined by <Picture 3>.")
    else:
        subject_lines.append(
            f"<Subject 1> is the protagonist: {textual}")
        subject_lines.append(
            "<Subject 2> is the opponent competing against <Subject 1> "
            "in the same match.")
    if use_video:
        subject_lines.append(
            "<Video 1> provides the overall temporal structure, sparring "
            "interaction, major motion progression, and camera energy of "
            "the reference event.")
    retention_reference = (
        "Transfer from <Video 1>: " + "; ".join(
            row["phase"] for row in brief.get("phases") or []) +
        ".\nDo not require exact source timing, exact source cuts, or "
        "frame-by-frame reproduction."
        if use_video else
        "No reference material; rely on the textual description alone.")
    timebox_lines = []
    for box in brief.get("phase_timeboxes") or []:
        start = box["span_ratio"][0] * seconds
        end = box["span_ratio"][1] * seconds
        timebox_lines.append(f"{start:.0f}-{end:.0f}s:\n{box['line']}")
    timebox_lines.append(str(brief.get("continuity_clause") or ""))
    fields = dict(brief["prompt_fields"])
    fields["subject_definitions"] = "\n".join(subject_lines)
    fields["retention_analysis"] = fields["retention_analysis"].replace(
        "{{RETENTION_REFERENCE}}", retention_reference)
    fields["summary"] = fields["summary"].replace(
        "{{SECONDS}}", f"{seconds:g}")
    fields["detailed_description"] = "\n\n".join(timebox_lines)
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
    return request


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
    if not isinstance(identity.get("frames"), list) or \
            len(identity["frames"]) != len(sample_paths):
        raise LT0Blocked("observe", "identity_frames_invalid", take_id)
    return {"take_id": take_id, "duration_s": duration_s,
            "observation": observation, "identity": identity}


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


def _ensure_h3_server(endpoint: str, *, variant: str, manage_server: bool,
                      port: int, repo_root: Path, log_path: Path
                      ) -> subprocess.Popen | None:
    """H3 与 Omni 错峰占同一组卡：必要时由实验自起 server（记录 owned
    进程组，任务后回收）。variant ∈ fl2va|ref2va——A 无 references 是
    t2va 任务，必须跑 fl2va server；B/C/D 才是 ref2va（人工审核 p0510）。"""
    if _endpoint_reachable(endpoint):
        return None
    if not manage_server:
        raise LT0Blocked("h3", "h3_endpoint_unreachable", endpoint)
    script = repo_root / "scripts" / "v9g" / "serve_h3.sh"
    if not script.is_file():
        raise LT0Blocked("h3", "serve_script_missing", str(script))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    process = subprocess.Popen(
        ["bash", str(script), variant, str(port)], cwd=str(repo_root),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
                       fl2va_endpoint: str = "http://127.0.0.1:30010",
                       gpu_set: str = "0,1,6,7",
                       seconds: float = DEFAULT_SECONDS,
                       seeds: tuple[int, ...] = DEFAULT_SEEDS,
                       pack_picks: dict[str, float] | None = None,
                       gpu_pairs: str = "0,1;6,7",
                       manage_server: bool = True,
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
        for recipe in RECIPES:
            take_id = f"{recipe}_s{seed}"
            request = build_long_take_request(
                brief, pack=pack, visual_reference=visual_reference,
                recipe=recipe, seed=seed, seconds=seconds,
                aspect_ratio=aspect_ratio)
            jobs.append({"take_id": take_id, "recipe": recipe,
                         "seed": seed, "request": request})
    plan = {"schema_version": LT0_VERSION, "execute_authorized": bool(execute),
            "experiment_note": "LT0 为 H3 能力实验；正式 V9-G license 门另行签署",
            "aspect_ratio": aspect_ratio, "seconds": float(seconds),
            "aspect_basis": {"container": geometry, "active_crop": active,
                             "active": [active_w, active_h]},
            "seeds": list(seeds),
            "recipes": list(RECIPES),
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

    def _run_group(group_jobs: list[dict[str, Any]], endpoint: str,
                   variant: str) -> None:
        port = endpoint.rsplit(":", 1)[-1].split("/")[0]
        server_process = _ensure_h3_server(
            endpoint, variant=variant, manage_server=manage_server,
            port=int(port), repo_root=repo_root,
            log_path=output_dir / f"h3_server_{variant}.log")
        client = SGLangH3Client(endpoint)
        try:
            for job in group_jobs:
                job_dir = output_dir / "takes" / job["take_id"]
                state = {"take_id": job["take_id"], "recipe": job["recipe"],
                         "seed": job["seed"], "variant": variant,
                         "state": "planned"}
                _write_json(job_dir / "job_state.json", state)
                try:
                    payload = client.serialize_payload(job["request"])
                    created = client.submit(payload)
                    state.update({"state": "submitted",
                                  "job_id": created.get("id"),
                                  "accepted_conditions": created.get(
                                      "accepted_conditions")})
                    _write_json(job_dir / "backend_response.json", created)
                    completed = client.poll(str(created["id"]))
                    state.update({"state": "completed",
                                  "status_payload": completed})
                    media = client.download(str(created["id"]),
                                            job_dir / "take.mp4")
                    state.update({"state": "downloaded", **media})
                    state["ffprobe"] = common.run_ffprobe_json(
                        _ffprobe_bin_of(ffmpeg_bin), Path(media["output_path"]))
                    state["state"] = "transport_validated"
                except V9GBlocked as exc:
                    state.update({"state": "failed",
                                  "reason_code": exc.reason_code,
                                  "detail": str(exc)[:500]})
                _write_json(job_dir / "job_state.json", state)
                take_results.append(state)
        finally:
            _stop_owned_server(server_process)

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
