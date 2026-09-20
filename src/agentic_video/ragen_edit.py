# -*- coding: utf-8 -*-
"""ReGen Module 3：Agentic Editing（p0523 修正 5/6 + StoryFlow 原则）。

- **候选 EDL 发散→选择**（From Shots to Stories 原则）：不取"每 phase 最短"
  单一解；生成 3 种合法 EDL（最小充分/最全证据/节奏对齐），按 Editing
  Functional Grammar 确定性打分选择（coverage/redundancy/causal/duration/
  variety）。
- **9:16 竖版画布**（修正 5）：撤横屏默认——canvas 经 cfg 注入 1080×1920，
  3:4 素材由 renderer 既有 scale+pad 保主体上画布。
- **盲看→比对两段终审**（修正 6）：Blind Viewer 不给蓝图（防迎合性 PASS），
  只答"这条片讲了什么/认知怎么变/结尾什么作用"；Comparator 才拿契约比对。
- 有界一次 re-edit：只重选 EDL，不回炉生成。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.ragen_director import ReGenBlocked

REGEN_EDIT_VERSION = "ragen_edit_v1"

BLIND_VIEWER_PROMPT = """只观察这段视频本身。不带任何预设，回答：观众看完
这条视频会怎么理解它？只输出一个 JSON 对象：
{"inferred_theme": "你觉得这条视频想表达什么（一句话）",
 "cognition_arc": [{"at_s": 0.0, "belief": "观众此刻对主人公的判断"}],
 "turning_point_s": 0.0,
 "turning_what": "哪一段让观众改变判断",
 "ending_function": "结尾起了什么作用",
 "emotional_aftertaste": "看完的情绪余味"}"""


COMPARATOR_PROMPT = """你是结构比对器。输入：A) 一位盲看观众对一条成片的
理解（无任何预设的观后答）；B) 该成片 intended 的叙事迁移契约。只输出一个
JSON 对象：{"theme_consistent": true, "counter_evidence_landed": true,
 "ending_function_matched": true, "issues": ["..."], "summary": "..."}
判定标准：观众**自行复述**出的主题/认知反转/结尾作用与契约一致即为 true；
观众没看出来的记 false 并写进 issues。不得因为"制作精良"放水。
A) 盲看结果："""


def build_moment_index(dailies: list[dict[str, Any]],
                       take_roles: dict[str, str] | None = None
                       ) -> list[dict[str, Any]]:
    """dailies 观察 → 扁平 moment 池（p0620 修正 1a：保留 section 归属）。

    take_roles: take_id → section_role——丢了它 _pick 会跨 section 串台
    （badcase 根因：S1 素材一秒未进成片、S2 的 0-1s 被填三段）。
    """
    moments = []
    for row in dailies:
        observation = row.get("observation") or {}
        take_id = str(row.get("take_id"))
        section_role = (take_roles or {}).get(take_id, "unknown")
        for entry in observation.get("event_timeline") or []:
            if entry.get("supported") is not True:
                continue
            interval = entry.get("interval") or []
            if len(interval) != 2:
                continue
            moments.append({
                "take_id": take_id, "phase": str(entry.get("phase")),
                "section_role": section_role,
                "beat_id": f"{section_role}::{entry.get('phase')}",
                "start": float(interval[0]), "end": float(interval[1]),
                "source_interval": [float(interval[0]), float(interval[1])]})
    return moments


def _pick(moments: list[dict[str, Any]], section_role: str, phase: str,
          strategy: str) -> dict[str, Any] | None:
    """只在该 section 自己的 moments 内选（修正 1a：跨 section=illegal）。"""
    rows = [m for m in moments
            if m["phase"] == phase and m["section_role"] == section_role]
    if not rows:
        return None
    if strategy == "longest":
        return max(rows, key=lambda m: m["end"] - m["start"])
    if strategy == "earliest":
        return min(rows, key=lambda m: m["start"])
    return min(rows, key=lambda m: (m["end"] - m["start"], m["start"]))


def build_candidate_edls(moment_index: list[dict[str, Any]],
                         section_patterns: list[list[str]],
                         rhythm: dict[str, Any],
                         section_roles: list[str] | None = None
                         ) -> list[dict[str, Any]]:
    """三种合法 EDL（发散）：min_sufficient / fullest / earliest。

    section_roles 与 section_patterns 平行——moment 按 section 归属匹配，
    杜绝串台。1b/1c 的合法性在 validate_edl 单独做（时长硬门+overlap）。
    """
    if section_roles is None:
        section_roles = [f"section_{i}" for i in range(len(section_patterns))]
    edls = []
    for strategy in ("shortest", "longest", "earliest"):
        segments, legal = [], True
        for section_index, phases in enumerate(section_patterns):
            picks = []
            for phase in phases:
                moment = _pick(moment_index,
                               section_roles[section_index], phase, strategy)
                if moment is None:
                    legal = False
                    break
                picks.append(moment)
            if not legal:
                break
            segments.append({"section_index": section_index,
                             "section_role": section_roles[section_index],
                             "moments": picks})
        if legal:
            edls.append({"strategy": strategy, "sections": segments})
    return edls


def validate_edl(edl: dict[str, Any], rhythm: dict[str, Any],
                 section_roles: list[str],
                 *, target_duration: float = 21.93,
                 tolerance: float = 1.5) -> list[str]:
    """p0620 修正 1b/1c：时长硬门（动态，非魔法数）+ interval overlap
    全局唯一。返回违规清单（空=合法）。"""
    problems = []
    total = sum(row["end"] - row["start"]
                for sec in edl["sections"] for row in sec["moments"])
    if not (target_duration - tolerance <= total
            <= target_duration + tolerance):
        problems.append(f"total_duration_out_of_range:{round(total, 2)}")
    prior = rhythm.get("rhythm_prior") or {}
    for sec in edl["sections"]:
        span = sum(row["end"] - row["start"] for row in sec["moments"])
        bounds = prior.get(sec["section_role"]) or {}
        rng = bounds.get("range")
        if rng and total > 0:
            share = span / total
            if not (rng[0] - 0.05 <= share <= rng[1] + 0.05):
                problems.append(
                    f"section_budget_violation:{sec['section_role']}:"
                    f"{round(share, 3)}")
    # 修正 1c：同 take_id 下任意两 interval 交集 > 0 → 重复（tuple 相同不够）
    by_take: dict[str, list[tuple[float, float]]] = {}
    for sec in edl["sections"]:
        for row in sec["moments"]:
            by_take.setdefault(row["take_id"], []).append(
                (row["start"], row["end"]))
    for take_id, spans in by_take.items():
        ordered = sorted(spans)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] < left[1] - 1e-6:
                problems.append(f"interval_overlap:{take_id}:"
                                f"[{left[0]},{left[1]}]x[{right[0]},{right[1]}]")
    return problems


def score_edl(edl: dict[str, Any], rhythm: dict[str, Any]) -> dict[str, Any]:
    """确定性打分（选择问题）：覆盖已由合法性保证；评冗余/顺序/节奏/变化。"""
    total = sum(row["end"] - row["start"]
                for sec in edl["sections"] for row in sec["moments"])
    # 冗余：同段内时间重叠惩罚
    overlap = 0.0
    for sec in edl["sections"]:
        ordered = sorted(sec["moments"], key=lambda m: m["start"])
        for left, right in zip(ordered, ordered[1:]):
            overlap += max(0.0, left["end"] - right["start"])
    # 节奏：段时长占比与 rhythm ratios 的偏差
    ratios = rhythm.get("section_duration_ratios") or []
    spans = [sum(row["end"] - row["start"] for row in sec["moments"])
             for sec in edl["sections"]]
    section_total = sum(spans) or 1.0
    rhythm_error = sum(abs(spans[i] / section_total - ratios[i])
                       for i in range(min(len(spans), len(ratios))))
    # 变化：moment 数量（更多切点=更高密度，奖励中等）
    density = sum(len(sec["moments"]) for sec in edl["sections"])
    variety = min(density, 12) / 12.0
    score = round(variety * 2.0 - overlap * 1.0 - rhythm_error * 1.5, 4)
    return {"score": score, "total_duration_s": round(total, 2),
            "overlap_s": round(overlap, 2),
            "rhythm_error": round(rhythm_error, 3), "moment_count": density}


def select_edl(edls: list[dict[str, Any]], rhythm: dict[str, Any],
               section_roles: list[str] | None = None,
               *, target_duration: float = 21.93,
               tolerance: float = 1.5) -> dict[str, Any]:
    """硬门（validate_edl）过滤后再打分选择——不满足时长/重叠即非法，
    不是扣分（p0620 修正 1b/1c）。"""
    if not edls:
        raise ReGenBlocked("edit", "no_legal_edl",
                           "moment index cannot cover all section patterns")
    roles = section_roles or [
        f"section_{i}" for i in range(len(edls[0]["sections"]))]
    legal, rejected = [], []
    for edl in edls:
        problems = validate_edl(edl, rhythm, roles,
                                target_duration=target_duration,
                                tolerance=tolerance)
        if problems:
            rejected.append({"strategy": edl.get("strategy"),
                             "problems": problems})
        else:
            legal.append(edl)
    if not legal:
        raise ReGenBlocked(
            "edit", "all_edls_illegal",
            json.dumps(rejected, ensure_ascii=False)[:500])
    scored = [{"edl": edl, **score_edl(edl, rhythm)} for edl in legal]
    scored.sort(key=lambda row: row["score"], reverse=True)
    return {"selected": scored[0], "candidates": scored,
            "rejected": rejected}


def assemble_final(edl: dict[str, Any], dailies_paths: dict[str, Path],
                   cfg: Any, output_dir: Path) -> dict[str, Any]:
    """EDL → render_micro_montage segments；9:16 画布经 cfg 注入（1080×1920）。"""
    from types import SimpleNamespace
    from src.agentic_video.renderer import render_micro_montage
    segments = []
    for section in edl["sections"]:
        for moment in section["moments"]:
            source = dailies_paths.get(moment["take_id"])
            if source is None:
                raise ReGenBlocked("edit", "dailies_path_missing",
                                   moment["take_id"])
            segments.append({
                "render_mode": "micro_clip",
                "duration_s": round(moment["end"] - moment["start"], 3),
                "source_video": str(source),
                "render_interval": [moment["start"], moment["end"]]})
    plan = {"passed": True,
            "duration_s": round(sum(s["duration_s"] for s in segments), 3),
            "segments": segments,
            "editorial_policy_version": REGEN_EDIT_VERSION}
    # 修正 5：竖版 9:16 画布（1080×1920），覆写 renderer 的画布配置读取
    base_generation = getattr(cfg, "generation", None)
    generation_dict = dict(base_generation) if isinstance(
        base_generation, dict) else {}
    generation_dict["assemble"] = {"width": 1080, "height": 1920}
    canvas_cfg = SimpleNamespace(
        **{**vars(cfg), "generation": generation_dict})
    result = render_micro_montage(canvas_cfg, plan, Path(output_dir))
    return {"plan": plan, "render": result}


def blind_view(final_video: Path, output_dir: Path, *, runner,
               ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    from src.perception.common import prepare_watch_copy
    watch_copy = prepare_watch_copy(Path(final_video), ffmpeg_bin=ffmpeg_bin)
    answer = runner.watch(
        Path(watch_copy), BLIND_VIEWER_PROMPT, fps=2.0,
        max_new_tokens=1536, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    (Path(output_dir) / "blind_viewer_raw.txt").write_text(
        raw, encoding="utf-8")
    from src.agentic_video.reference_program_v9 import _parse_one_object
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return _parse_one_object(text, stage="ragen_edit")


def compare_to_contract(blind: dict[str, Any], contract: dict[str, Any],
                        output_dir: Path, *, runner) -> dict[str, Any]:
    payload = json.dumps(
        {"blind": blind,
         "contract": {"theme": contract.get("theme"),
                      "narrative_invariants": contract.get(
                          "narrative_invariants")}},
        ensure_ascii=False, separators=(",", ":"))
    answer = runner.ask(
        COMPARATOR_PROMPT + payload, max_new_tokens=1024,
        stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    (Path(output_dir) / "comparator_raw.txt").write_text(
        raw, encoding="utf-8")
    from src.agentic_video.reference_program_v9 import _parse_one_object
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    value = _parse_one_object(text, stage="ragen_edit")
    for key in ("theme_consistent", "counter_evidence_landed",
                "ending_function_matched"):
        if not isinstance(value.get(key), bool):
            raise ReGenBlocked("edit", "comparator_field_invalid", key)
    return value


def run_ragen_edit(cfg: Any, output_dir: Path, *,
                   contract: dict[str, Any],
                   dailies: list[dict[str, Any]],
                   dailies_paths: dict[str, Path],
                   runner, section_patterns: list[list[str]],
                   take_roles: dict[str, str] | None = None,
                   target_duration: float = 21.93,
                   ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """Moment Index → 候选 EDL（role-aware）→ 硬门选择 → 9:16 装配 →
    盲看 → causal judge → 比对（p0620 三关终审）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = take_roles or {}
    moment_index = build_moment_index(dailies, roles)
    edls = build_candidate_edls(moment_index, section_patterns,
                                contract.get("rhythm_grammar") or {},
                                section_roles=contract.get("section_roles"))
    selection = select_edl(edls, contract.get("rhythm_grammar") or {},
                           contract.get("section_roles"),
                           target_duration=target_duration)
    assembled = assemble_final(selection["selected"]["edl"],
                               dailies_paths, cfg, output_dir)
    blind = blind_view(Path(assembled["render"]["content_master"]),
                       output_dir, runner=runner, ffmpeg_bin=ffmpeg_bin)
    comparison = compare_to_contract(blind, contract, output_dir,
                                     runner=runner)
    # p0620 修正 1g：causal judge（盲看谓词互否）+ identity gate
    negation = blind_causal_judge(blind, contract, runner=runner)
    identity_false = any(
        str((d.get("identity") or {}).get("c0_consistent")) == "False"
        for d in dailies)
    passed = (all(comparison[key] for key in (
        "theme_consistent", "counter_evidence_landed",
        "ending_function_matched")) and negation == "yes"
        and not identity_false)
    report = {
        "schema_version": REGEN_EDIT_VERSION,
        "edl_selection": selection,
        "final_video": str(assembled["render"]["content_master"]),
        "duration_s": assembled["render"].get("duration_s"),
        "blind_viewer": blind, "comparator": comparison,
        "blind_causal_judge": negation,
        "identity_gate_clean": not identity_false,
        "passed": passed,
        "bounded_reedit_available": not passed,
    }
    (output_dir / "ragen_edit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


BLIND_CAUSAL_JUDGE_PROMPT = """只做逻辑判定。A 是观众盲看一条视频后自述的
认知变化；B 是该视频 intended 的叙事契约。问题：观众自述的初始判断与
转折证据，是否构成契约要求的**能力维度上的直接否定**？只输出一个 JSON
对象：{"directly_contradicts": "yes|no|partially", "reason": "一句话"}
注意：性格反差（严肃→活泼）不是能力修正；只有"低估其能力→可见的高能力
证据"才是 yes。A) 盲看自述："""


def blind_causal_judge(blind: dict[str, Any],
                       contract: dict[str, Any], *, runner) -> str:
    """p0620 修正 1g：防 generic-theme 假绿（"某种刻板印象被打破"≠PASS）。"""
    payload = json.dumps(
        {"A": {"inferred_theme": blind.get("inferred_theme"),
               "cognition_arc": blind.get("cognition_arc"),
               "turning_what": blind.get("turning_what")},
         "B": {"theme": contract.get("theme"),
               "invariants": contract.get("narrative_invariants")}},
        ensure_ascii=False, separators=(",", ":"))
    answer = runner.ask(BLIND_CAUSAL_JUDGE_PROMPT + payload,
                        max_new_tokens=512,
                        stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    from src.agentic_video.reference_program_v9 import _parse_one_object
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    value = _parse_one_object(text, stage="ragen_edit_judge")
    verdict = str(value.get("directly_contradicts") or "")
    if verdict not in {"yes", "no", "partially"}:
        raise ReGenBlocked("edit_judge", "blind_judge_verdict_invalid", verdict)
    return verdict
