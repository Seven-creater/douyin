"""槽级证据验证器（V1 P3）：看实际进成片的区间，回答结构化验证问题。

不是问"是否满足需求"（容易得到宽泛肯定），而是逐条核对 must_have：
结论(通过/不通过/不确定) + 各条件是否满足 + 证据在片段的什么时间 +
是否需要前后文 + 具体失败原因。看的是**实际准备放进成片的区间**；
为理解上下文额外看的部分帮助判断，但"观众看不到的源片情节"不算成片
已表达。需求"保护同伴"时看见挥刀不能直接通过——须定位受威胁对象、
介入动作，或可靠对白/前后文支持保护关系。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.template.schema import extract_json_block

VERIFICATION_PROMPT = """你是素材证据验证员。观看影片 {video} 的 {start:g}~{end:g} 秒区间
（这正是将被放进成片的片段，观众只能看到这段）。对照下列叙事需求逐条验证。

叙事需求：{need}
必须满足：{must_have}
明确排除：{must_not}
证据模式：{evidence_mode}

只输出 JSON：
{{"verdict":"pass|fail|uncertain",
"conditions":[{{"condition":"必须满足项原文","met":true,"evidence_interval":[片段内秒数,片段内秒数]}}],
"missing":["未满足的条件"],
"failure_reason":"不通过时的具体原因：缺什么证据、断在哪",
"needs_context":false,
"what_is_visible":"片段里实际可见的内容一句话"}}

判定纪律：
- 只依据片段内可见/可听内容；片段外剧情不能作为通过理由。
- 每条 met=true 都必须给 evidence_interval；给不出就 met=false。
- 无法判断（画面太暗/太快/被遮挡）→ verdict=uncertain，不要猜。
"""

VERIFICATION_PROMPT_VERSION = "verify_v1"


def _slot_window(slot: dict, *, pad_s: float = 0.0) -> tuple[float, float]:
    source = slot.get("source") or {}
    start = max(0.0, float(source.get("start_s") or 0) - pad_s)
    end = float(source.get("end_s") or 0) + pad_s
    return start, end


def parse_verification(raw: str) -> dict | None:
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else None
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or payload.get("verdict") not in {
            "pass", "fail", "uncertain"}:
        return None
    conditions = []
    for item in payload.get("conditions") or []:
        if isinstance(item, dict) and item.get("condition"):
            conditions.append({
                "condition": str(item["condition"]),
                "met": bool(item.get("met")),
                "evidence_interval": item.get("evidence_interval")
                if isinstance(item.get("evidence_interval"), list) else None,
            })
    return {
        "verdict": payload["verdict"],
        "conditions": conditions,
        "missing": [str(value) for value in payload.get("missing") or []],
        "failure_reason": str(payload.get("failure_reason") or ""),
        "needs_context": bool(payload.get("needs_context")),
        "what_is_visible": str(payload.get("what_is_visible") or "")[:120],
    }


def verify_slots(cfg, story_plan: dict, *, runner, slot_idxs=None,
                 context_pad_s: float = 5.0) -> dict:
    """对 supported 槽逐个看实际区间验证。needs_context 时有界扩展重看一次。

    返回 {"results": [...], "failed_slots": [...], "uncertain_slots": [...]}，
    由 pipeline 决定是否触发 re_search（每轮限一次验证 pass）。
    """
    results = []
    for slot in story_plan.get("slots") or []:
        idx = int(slot.get("slot_idx", 0))
        if slot_idxs is not None and idx not in set(slot_idxs):
            continue
        if slot.get("status") != "supported":
            continue
        source = slot.get("source") or {}
        video = Path(str(source.get("video") or ""))
        if not video.exists():
            results.append({"slot_idx": idx, "verdict": "uncertain",
                            "failure_reason": "source video missing"})
            continue
        spec = slot.get("need_spec") or {}
        start, end = _slot_window(slot)

        def _ask(start_s: float, end_s: float) -> dict | None:
            prompt = VERIFICATION_PROMPT.format(
                video=video.name, start=start_s, end=end_s,
                need=spec.get("need") or slot.get("role"),
                must_have="；".join(spec.get("must_have") or []),
                must_not="；".join(spec.get("must_not") or []),
                evidence_mode=spec.get("evidence_mode") or "visual")
            answer = runner.watch(video, prompt, start_s=start_s, end_s=end_s,
                                  max_new_tokens=1024, duration_s=end_s - start_s)
            return parse_verification(answer.text)

        verdict = _ask(start, end)
        if verdict is None:
            verdict = {"verdict": "uncertain", "conditions": [], "missing": [],
                       "failure_reason": "verification parse failed",
                       "needs_context": False, "what_is_visible": ""}
        elif verdict.get("needs_context"):
            # 有界上下文扩展：帮助判断，但判定仍以原区间为准
            widened = _ask(max(0.0, start - context_pad_s), end + context_pad_s)
            if widened is not None:
                verdict["context_widened"] = True
                verdict = widened
        verdict["slot_idx"] = idx
        results.append(verdict)
    return {
        "results": results,
        "failed_slots": [row["slot_idx"] for row in results
                         if row["verdict"] == "fail"],
        "uncertain_slots": [row["slot_idx"] for row in results
                            if row["verdict"] == "uncertain"],
        "prompt_version": VERIFICATION_PROMPT_VERSION,
    }
