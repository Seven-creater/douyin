"""Structured narrative critic, comprehension score, and guarded story patches."""
from __future__ import annotations

import copy
import json

from src.agentic_video.story_planner import validate_story_plan
from src.template.schema import extract_json_block

STORY_PATCH_OPS = ("replace",)
COMPREHENSION_QUESTIONS = (
    "主角是谁？", "他面对什么问题？", "为什么采取关键行动？",
    "高潮发生了什么？", "最后的结果或表达是什么？",
)


def preflight_story_faults(story_plan: dict) -> list[dict]:
    """Find deterministic story failures before asking the video judge to explain them."""
    faults = []
    slots = story_plan.get("slots") or []
    supported = [slot for slot in slots if slot.get("status") == "supported"]
    for left, right in zip(supported, supported[1:]):
        left_source, right_source = left.get("source") or {}, right.get("source") or {}
        if (set(left_source.get("entity_ids") or [])
                and set(right_source.get("entity_ids") or [])
                and not (set(left_source.get("entity_ids") or [])
                         & set(right_source.get("entity_ids") or []))):
            faults.append({"type": "entity_switch", "slot_idx": right.get("slot_idx"),
                           "evidence": "adjacent supported slots have no shared entity"})
        if (left_source.get("video_stem") == right_source.get("video_stem")
                and float(right_source.get("start_s") or 0)
                < float(left_source.get("start_s") or 0)):
            faults.append({"type": "causal_order", "slot_idx": right.get("slot_idx"),
                           "evidence": "same source timeline runs backwards"})
    if not any(slot.get("role") == "resolution" and slot.get("status") == "supported"
               for slot in slots):
        faults.append({"type": "missing_resolution", "slot_idx": None,
                       "evidence": "no supported resolution slot"})
    if (any(slot.get("role") == "conflict" and slot.get("status") == "supported"
            for slot in slots)
            and not any(slot.get("role") in {"context", "choice"}
                        and slot.get("status") == "supported" for slot in slots)):
        faults.append({"type": "missing_cause", "slot_idx": None,
                       "evidence": "conflict has no supported context or choice"})
    for slot in slots:
        source = slot.get("source") or {}
        start, end = map(float, slot.get("target_interval") or [0.0, 0.0])
        source_start = float(source.get("start_s") or 0.0)
        source_end = float(source.get("end_s") or source_start)
        for line in source.get("dialogue") or []:
            line_start = float(line.get("start_s") or source_start)
            line_end = float(line.get("end_s") or line_start)
            if line_end <= source_start or line_start < source_start or line_end > source_end + 1e-6:
                faults.append({"type": "subtitle_timing", "slot_idx": slot.get("slot_idx"),
                               "evidence": "subtitle lies outside source interval"})
            if line_end - source_start > end - start + 1e-6:
                faults.append({"type": "dialogue_cut", "slot_idx": slot.get("slot_idx"),
                               "evidence": "subtitle would extend beyond target slot"})
    return faults


def validate_story_patch(plan: dict, patch: dict) -> list[str]:
    errors = []
    if patch.get("op") not in STORY_PATCH_OPS:
        errors.append("story patch op invalid")
    path = patch.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        return errors + ["story patch path invalid"]
    parts = path.strip("/").split("/")
    allowed = False
    if len(parts) >= 3 and parts[0] == "slots" and parts[1].isdigit():
        idx = int(parts[1])
        allowed = idx < len(plan.get("slots") or []) and parts[2] in {
            "source", "status", "reason", "target_interval"}
        if parts[2] == "source" and len(parts) > 4:
            allowed = False
    if not allowed:
        errors.append("story patch path outside allowlist")
    if "value" not in patch:
        errors.append("story patch value missing")
    return errors


def _path_parts(path: str) -> list[str]:
    return [part.replace("~1", "/").replace("~0", "~")
            for part in path.strip("/").split("/")]


def apply_story_patches(plan: dict, patches: list[dict]) -> tuple[dict, list[dict]]:
    current = copy.deepcopy(plan)
    audit = []
    for patch in patches:
        errors = validate_story_patch(current, patch)
        if errors:
            audit.append({"patch": patch, "status": "rejected", "errors": errors})
            continue
        candidate = copy.deepcopy(current)
        target = candidate
        parts = _path_parts(patch["path"])
        try:
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            key = parts[-1]
            if isinstance(target, list):
                target[int(key)] = patch["value"]
            else:
                target[key] = patch["value"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            audit.append({"patch": patch, "status": "rejected", "errors": [str(exc)]})
            continue
        errors = validate_story_plan(candidate)
        if errors:
            audit.append({"patch": patch, "status": "rejected", "errors": errors})
            continue
        current = candidate
        audit.append({"patch": patch, "status": "applied"})
    return current, audit


def score_comprehension_answers(answers: list[dict]) -> dict:
    by_id = {int(row.get("question_id", -1)): bool(row.get("correct"))
             for row in answers if isinstance(row, dict)}
    correct = sum(by_id.get(idx, False) for idx in range(len(COMPREHENSION_QUESTIONS)))
    return {"correct": correct, "total": len(COMPREHENSION_QUESTIONS),
            "score": correct / len(COMPREHENSION_QUESTIONS), "passes": correct >= 4}


NARRATIVE_CRITIC_PROMPT = """你是可审计的 Narrative Critic。请只观看成片回答五个理解问题，
并检查人物连续、事件因果、高潮铺垫、结尾收束和字幕同步。参考程序只用于核对答案，不能用来代替看片。

只输出一个 JSON 对象：
{"theme_relevance":0.0,"narrative_coherence":0.0,
 "answers":[{"question_id":0,"answer":"看片得到的答案","correct":true,"evidence":"可见/可听证据"}],
 "issues":[{"type":"entity_switch|causal_order|missing_resolution|subtitle_timing|dialogue_cut",
            "slot_idx":0,"evidence":"具体可见现象"}],
 "patches":[{"op":"replace","path":"/slots/0/source","value":{},"reason":"证据关系"}],
 "re_search":[{"slot_idx":0,"reason":"该槽画面不支持其叙事需求（具体断点）","need_hint":"改写后的检索需求"}],
 "verdict":"一句话结论"}

五个问题依次为：{questions}
只允许替换已有槽的 source/status/reason/target_interval。不能创造事件、对白或素材。
若某槽画面不支持其叙事需求且你无法直接给出替代素材，用 re_search 指令描述断点与
改写后的检索需求——不要自己编造 source。

【Narrative Program】{narrative}
【Story Plan】{story_plan}
"""


def parse_narrative_critique(raw: str) -> dict | None:
    block = extract_json_block(raw)
    if not block:
        return None
    try:
        value = json.loads(block)
    except ValueError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("answers"), list):
        return None
    for key in ("theme_relevance", "narrative_coherence"):
        try:
            value[key] = min(1.0, max(0.0, float(value.get(key, 0))))
        except (TypeError, ValueError):
            value[key] = 0.0
    value["comprehension"] = score_comprehension_answers(value["answers"])
    value["patches"] = [patch for patch in value.get("patches") or []
                        if isinstance(patch, dict)]
    # P3 重搜指令通道：{slot_idx, reason, need_hint?}，槽号必须是整数下标
    value["re_search"] = [
        {"slot_idx": int(directive["slot_idx"]),
         "reason": str(directive.get("reason") or ""),
         "need_hint": str(directive.get("need_hint") or "") or None}
        for directive in value.get("re_search") or []
        if isinstance(directive, dict)
        and str(directive.get("slot_idx", "")).lstrip("-").isdigit()]
    return value


def run_narrative_critic(video, narrative: dict, story_plan: dict, *, runner) -> dict:
    # The prompt contains a literal JSON example.  Replace the three named
    # fields directly so JSON braces can never be interpreted as format keys.
    prompt = NARRATIVE_CRITIC_PROMPT
    replacements = {
        "{questions}": "；".join(f"{idx}. {question}"
                                for idx, question in enumerate(COMPREHENSION_QUESTIONS)),
        "{narrative}": json.dumps(narrative, ensure_ascii=False)[:12000],
        "{story_plan}": json.dumps(story_plan, ensure_ascii=False)[:10000],
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    answer = runner.watch(video, prompt, max_new_tokens=2048)
    result = parse_narrative_critique(answer.text)
    if result is None:
        result = {
            "theme_relevance": 0.0, "narrative_coherence": 0.0,
            "answers": [], "comprehension": score_comprehension_answers([]),
            "issues": [{"type": "critic_parse_failed", "slot_idx": None,
                        "evidence": "Narrative Critic JSON parse failed"}],
            "patches": [], "re_search": [], "verdict": "critic parse failed",
            "raw_head": answer.text[:300],
        }
    result["elapsed_s"] = answer.elapsed_s
    deterministic_faults = preflight_story_faults(story_plan)
    existing = {(row.get("type"), row.get("slot_idx"))
                for row in result.get("issues") or [] if isinstance(row, dict)}
    result["issues"] = (result.get("issues") or []) + [
        row for row in deterministic_faults
        if (row["type"], row.get("slot_idx")) not in existing]
    return result
