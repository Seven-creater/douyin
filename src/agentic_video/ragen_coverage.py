# -*- coding: utf-8 -*-
"""ReGen v3：Beat Graph + Coverage Loop（p0620 六项修正）。

- Beat Graph：story 拆连续 beats（beat_id=故事层 ID，phase=素材内部过程，
  两层分离）；take_001 观察映射 covered。
- Beat Span 生成单位：G1=[B1,B2]/G2=[B7,B8]/G3=[B9,B10]——一条 8-12s
  take 承载一个语义连贯的下一段，不是一 beat 一 take。
- Coverage 三态（covered/partial/missing）+ evidence（interval+readability）：
  判定标准=有没有足够清楚、≥1s、能剪进去的证据，不是语义标签存在。
- 纵向优先：Controller 选 missing/partial 最重的 span；横向按需（某 span
  质量差才拍第二版）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.ragen_director import ReGenBlocked

COVERAGE_VERSION = "ragen_coverage_v1"

# 故事 beat 骨架（从 contract 的 section_roles + wrapper 结构确定性推导；
# take_001 已覆盖 B3-B6）
BEAT_SKELETON = [
    {"beat_id": "B1", "section_role": "situation_setup",
     "beat_function": "protagonist_introduction"},
    {"beat_id": "B2", "section_role": "situation_setup",
     "beat_function": "explicit_underestimation"},
    {"beat_id": "B3", "section_role": "counter_evidence",
     "beat_function": "challenge_setup"},
    {"beat_id": "B4", "section_role": "counter_evidence",
     "beat_function": "intense_exchange"},
    {"beat_id": "B5", "section_role": "counter_evidence",
     "beat_function": "decisive_outcome"},
    {"beat_id": "B6", "section_role": "counter_evidence",
     "beat_function": "aftermath_reaction"},
    {"beat_id": "B7", "section_role": "evidence_expansion",
     "beat_function": "additional_capability_evidence"},
    {"beat_id": "B8", "section_role": "evidence_expansion",
     "beat_function": "cross_context_evidence"},
    {"beat_id": "B9", "section_role": "evidence_expansion",
     "beat_function": "humanizing_detail"},
    {"beat_id": "B10", "section_role": "evidence_expansion",
     "beat_function": "light_punchline"},
]

# 生成 span（用户拍板：3 条而非 5；take_001=B3-B6 已有）
BEAT_SPANS = [
    {"span_id": "G1", "beats": ["B1", "B2"],
     "desc": "protagonist introduction then explicit underestimation"},
    {"span_id": "G2", "beats": ["B7", "B8"],
     "desc": "additional capability evidence then cross-context evidence"},
    {"span_id": "G3", "beats": ["B9", "B10"],
     "desc": "humanizing detail then light punchline"},
]

COVERAGE_JUDGE_PROMPT = """你是素材覆盖判定员。输入：故事 Beat 清单（每条
beat 在整个故事中的功能）+ 现有素材的观察结果（每条 take 的 event_timeline，
interval 为该 take 内秒数，supported=画面清晰）。对每个 beat 判定覆盖状态：
- covered：存在**足够清楚、≥1 秒、能直接剪进成片**的证据
- partial：有相关画面但可读性弱/太短/不确定能剪
- missing：无相关画面
只输出一个 JSON 对象：
{"coverage": [{"beat_id": "B1", "status": "covered|partial|missing",
  "evidence": [{"take_id": "...", "interval": [0.0, 1.0],
                "readability": "clear|weak"}]}]}
判定标准是"能不能剪"，不是"语义沾边"：路人瞟一眼不足以 covered 一个
"明确低估"beat。输入："""


def build_beat_graph() -> list[dict[str, Any]]:
    return json.loads(json.dumps(BEAT_SKELETON))


def coverage_payload(beats: list[dict[str, Any]],
                     observations: list[dict[str, Any]]) -> str:
    rows = []
    for obs in observations:
        timeline = (obs.get("observation") or {}).get("event_timeline") or []
        rows.append({
            "take_id": obs.get("take_id"),
            "timeline": [
                {"phase": row.get("phase"),
                 "interval": row.get("interval"),
                 "supported": row.get("supported")}
                for row in timeline]})
    return json.dumps(
        {"beats": [{"beat_id": b["beat_id"],
                    "section_role": b["section_role"],
                    "beat_function": b["beat_function"]} for b in beats],
         "dailies": rows},
        ensure_ascii=False, separators=(",", ":"))


def judge_coverage(beats: list[dict[str, Any]],
                   observations: list[dict[str, Any]], *, runner,
                   output_dir: Path) -> dict[str, Any]:
    """一次 ask 判全部 beat 的覆盖状态（三态+证据）。"""
    from src.agentic_video.reference_program_v9 import _parse_one_object
    answer = runner.ask(
        COVERAGE_JUDGE_PROMPT + coverage_payload(beats, observations),
        max_new_tokens=2048, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    (Path(output_dir) / "coverage_raw.txt").write_text(
        raw, encoding="utf-8")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    value = _parse_one_object(text, stage="ragen_coverage")
    rows = value.get("coverage") or []
    by_id = {str(row.get("beat_id")): row for row in rows}
    for beat in beats:
        row = by_id.get(beat["beat_id"])
        if not row or row.get("status") not in {
                "covered", "partial", "missing"}:
            raise ReGenBlocked("coverage", "coverage_verdict_invalid",
                               str(beat["beat_id"]))
    return {str(row["beat_id"]): row for row in rows}


def next_span_to_shoot(coverage: dict[str, Any]) -> dict[str, Any] | None:
    """选 missing/partial 最重的 span（纵向优先）；全 covered → None。

    partial 的 beat 若带 generation_needed=false（人审改判）不触发。
    """
    best: dict[str, Any] | None = None
    best_score = -1
    for span in BEAT_SPANS:
        statuses = []
        for beat_id in span["beats"]:
            row = coverage.get(beat_id) or {}
            status = str(row.get("status") or "missing")
            if status == "partial" and row.get("generation_needed") is False:
                status = "covered"
            statuses.append(status)
        score = sum({"missing": 2, "partial": 1, "covered": 0}[s]
                    for s in statuses)
        if score > best_score and score > 0:
            best, best_score = span, score
    return best


def span_take_prompt(story: dict[str, Any], span: dict[str, Any],
                     contract: dict[str, Any]) -> str:
    """Beat Span → H3 视觉 prompt（visual_event 具体事件，t2va 三字段）。

    p0620 修正 1e：H3 只吃 visual_event（地点/人物/动作序列），抽象
    narrative_claim 不进 prompt（overlay 后期）；修正 1d：无 image
    condition 时走 t2va 三字段 base grammar。
    """
    sections = {row.get("role"): row for row in story.get("sections") or []}
    role = next((b["section_role"] for b in BEAT_SKELETON
                 if b["beat_id"] == span["beats"][0]), "situation_setup")
    section = sections.get(role) or {}
    events = []
    if role == "situation_setup":
        belief = story.get("initial_belief") or {}
        events.append(
            "the protagonist is introduced in daily context; others "
            f"visibly react with doubt driven by "
            f"{belief.get('source_of_underestimation', 'circumstances')}")
        events.append(
            "a clear visible signal of underestimation: an authority or "
            "peer examines the protagonist's situation, shakes their head, "
            "and waves the protagonist off dismissively")
    elif role == "evidence_expansion":
        for row in story.get("reinforcement") or []:
            event = row.get("visual_event") or row.get("event")
            if event:
                events.append(str(event))
        ending = story.get("ending") or {}
        ending_visual = ending.get("visual_event") or ending.get("statement")
        if ending_visual:
            events.append(str(ending_visual))
    body = ". Then, ".join(events[:3]) + "."
    return "\n\n".join([
        "integrated_multimodal_description:\n"
        "[Shot 1] At 00:00.000, " + body,
        "overall_soundscape:\nNatural ambient sound consistent with the "
        "depicted activity and setting.",
        "non_diegetic_music:\nN/A"])


def build_span_take_request(story: dict[str, Any], span: dict[str, Any],
                            contract: dict[str, Any], *, seconds: float,
                            seed: int,
                            identity_anchor: dict[str, Any] | None = None
                            ) -> dict[str, Any]:
    """Span take 请求。有 Identity Anchor → ref2va 六段；无 → t2va 三字段
    （修正 1d：grammar 跟随真实 conditions 路由，不写死）。"""
    from src.agentic_video.generation_v9g import build_h3_request
    references = None
    prompt: str
    if identity_anchor:
        from src.agentic_video.ragen_generate import (
            ref2va_identity_prompt)
        prompt = ref2va_identity_prompt(
            span_take_prompt(story, span, contract), identity_anchor)
        references = [{"type": "image",
                       "uri": identity_anchor["uri"]}]
    else:
        prompt = span_take_prompt(story, span, contract)
    request = build_h3_request(
        prompt=prompt, duration_s=float(seconds),
        references=references, first_frame=None, last_frame=None,
        capabilities=None, seed=int(seed), aspect_ratio="3:4")
    request["_ragen_span"] = {"span_id": span["span_id"],
                              "beats": span["beats"],
                              "seconds": float(seconds)}
    return request
