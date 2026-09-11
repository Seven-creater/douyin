"""7682 型情感叙事模板的文案轨生成器（MVP，2026-09-10）。

参考片 7682719919410072847 的表达公式是「弱视觉事件 + 强文字轨 + BGM」：
钩子断言（人们常常觉得…）→ 画面实证留白 → 成就/事实卡连发（~1s/张）→
反转梗钉尾。语义由烧录文案承担，剪辑只提供在场证明。

两级生成：Tier1 纯模板（确定性，任何环境可用）；Tier2 单次 OmniRunner.ask
让模型结合主题与已选素材 caption 写文案——失败/无 runner 一律落 Tier1，
绝不阻塞渲染。时序全部规则化（复刻参考片实测节奏），不经素材对白时序映射。
"""
from __future__ import annotations

import json

COPY_VERSION = "copy_v2"          # V3 P4：cue 绑 slot_idx/subject/evidence + grounding 校验

# 模板保底文案：通用但不空洞的断言/收尾，适配逆袭-励志-守护类主题
_FALLBACK_HOOKS = (
    "人们常常觉得，弱者的挣扎毫无意义",
    "人们常说，普通人的努力改变不了什么",
    "人们常常觉得，奇迹只是天才的另一个名字",
)
_FALLBACK_PUNCHLINES = (
    "但故事才刚刚开始",
    "但没人能让他停下",
    "但这不是终点",
)
_HOOK_MAX_CHARS = 20
_CARD_MAX_CHARS = 12
_PUNCHLINE_MAX_CHARS = 14


def _clean(text: str, limit: int) -> str:
    return "".join(ch for ch in str(text or "").strip() if ch not in "\r\n\t")[:limit]


def _template_copy(story_plan: dict) -> dict:
    """Tier1：主题哈希轮换 + 素材 caption 首段截断，确定性输出。"""
    theme = str(story_plan.get("theme") or "逆袭")
    slots = story_plan.get("slots") or []
    hook = _FALLBACK_HOOKS[hash(theme) % len(_FALLBACK_HOOKS)]
    punchline = _FALLBACK_PUNCHLINES[hash(theme + "p") % len(_FALLBACK_PUNCHLINES)]
    cards = []
    for slot in slots[1:]:                                # 首槽是钩子画面，不出卡
        caption = str((slot.get("source") or {}).get("caption") or "").strip()
        head = caption.split("，")[0].split("。")[0] if caption else ""
        if head:
            cards.append(_clean(head, _CARD_MAX_CHARS))
    if not cards:                                         # caption 全空时用角色兜底
        cards = [f"{slot['role']}时刻" for slot in slots[1:3]] or ["全力以赴"]
    return {"hook": hook, "cards": cards[:4], "punchline": punchline}


def _ask_llm(runner, story_plan: dict) -> dict | None:
    """Tier2：单次 ask 生成文案 JSON；解析或校验失败返回 None 走模板。"""
    if runner is None:
        return None
    theme = str(story_plan.get("theme") or "")
    slot_lines = []
    for slot in story_plan.get("slots") or []:
        source = slot.get("source") or {}
        entity_ids = source.get("entity_ids") or []
        entities = "、".join(entity_ids) if entity_ids else "未知"
        slot_lines.append(f"- {slot.get('role')}：{str(source.get('caption') or '')[:60]}"
                          f"（人物：{entities}）")
    prompt = (
        "你是抖音情感叙事爆款文案师。参考表达公式：开头一句颠覆常识的断言钩子"
        "（人们常常觉得…），中段画面实证，结尾成就卡连发后抛一句反转/留白短句。\n"
        f"主题：{theme}\n已选素材段（每行一个槽位，cards 按槽位顺序逐条对应）：\n"
        + "\n".join(slot_lines) +
        "\n证据纪律：每条 card 必须是对应槽位 caption 的写实转写，不得编造画面"
        "外的人物/事件；punchline 如果含动作断言（救/赢/找到等），该动作必须"
        "在末槽 caption 里可见——画面没有的动作一个字不许写。输出 JSON："
        '{"hook": "<=20字断言句", "cards": ["<=12字陈述式事实/成就，3-4条"], '
        '"punchline": "<=14字反转或留白句"}'
    )
    try:
        from src.template.schema import extract_json_block

        answer = runner.ask(prompt, max_new_tokens=1024)
        block = extract_json_block(answer.text)
        parsed = json.loads(block) if block else None
        if not isinstance(parsed, dict):
            return None
        hook = _clean(parsed.get("hook"), _HOOK_MAX_CHARS)
        punchline = _clean(parsed.get("punchline"), _PUNCHLINE_MAX_CHARS)
        cards = [_clean(card, _CARD_MAX_CHARS)
                 for card in parsed.get("cards") or [] if _clean(card, 1)]
        if not hook or not punchline or not cards:
            return None
        return {"hook": hook, "cards": cards[:4], "punchline": punchline}
    except Exception:                                     # noqa: BLE001 - LLM 层任何失败都落模板
        return None


def build_copy_cues(story_plan: dict, runner=None) -> dict:
    """产出成片时间轴的文案 cue 列表 + 音频模式（7682 公式 = 纯文案 + 纯 BGM）。

    时序规则（复刻参考片实测）：钩子 [0.15, min(4.2, 首槽末-0.2)]；
    中段留白（对应参考片记分牌区）；成就卡从末槽起点 +0.2s 起每张 1.0s、
    间隔 0.1s 连发；反转梗钉在 [总长-1.0, 总长-0.05]。
    """
    slots = story_plan.get("slots") or []
    total = float(story_plan.get("target_duration_s") or 0)
    if not slots or total <= 0:
        return {"cues": [], "audio_mode": "dialogue", "source": "none"}

    llm_copy = _ask_llm(runner, story_plan)
    copy = llm_copy or _template_copy(story_plan)
    source_tier = "llm" if llm_copy else "template"

    first_end = float(slots[0]["target_interval"][1])
    last_start = float(slots[-1]["target_interval"][0])
    last_duration = total - last_start

    def _subject(slot: dict) -> str:
        names = (slot.get("source") or {}).get("entity_names") or []
        return "、".join(str(name) for name in names[:2]) if names else ""

    def _evidence(slot: dict) -> str:
        return str((slot.get("source") or {}).get("caption") or "")[:80]

    cues = [{
        "kind": "hook_line",
        "start_s": 0.15,
        "end_s": round(min(4.2, first_end - 0.2), 3),
        "text": copy["hook"],
        "slot_idx": int(slots[0]["slot_idx"]),
        "subject_entity": _subject(slots[0]),
        "evidence": _evidence(slots[0]),
    }]
    # 末槽装得下的卡数：留 1.4s 给反转梗，每张占 1.1s
    n_cards = max(1, min(len(copy["cards"]), int((last_duration - 1.4) // 1.1) or 1))
    t = last_start + 0.2
    for card_idx, card in enumerate(copy["cards"][:n_cards]):
        card_slot = slots[1 + card_idx] if 1 + card_idx < len(slots) else slots[-1]
        cues.append({"kind": "info_card", "start_s": round(t, 3),
                     "end_s": round(t + 1.0, 3), "text": card,
                     "slot_idx": int(card_slot["slot_idx"]),
                     "subject_entity": _subject(card_slot),
                     "evidence": _evidence(card_slot)})
        t += 1.1
    cues.append({
        "kind": "punchline",
        "start_s": round(max(0.0, total - 1.0), 3),
        "end_s": round(max(0.1, total - 0.05), 3),
        "text": copy["punchline"],
        "slot_idx": int(slots[-1]["slot_idx"]),
        "subject_entity": _subject(slots[-1]),
        "evidence": _evidence(slots[-1]),
    })
    return {"cues": cues, "audio_mode": "bgm", "source": source_tier,
            "version": COPY_VERSION}


# 动作断言词：punchline 含这些词时必须有画面证据（V3 P4）
_ACTION_ASSERTION_CHARS = ("救", "赢", "找到", "夺", "守住", "护住", "翻盘",
                           "逆袭", "夺冠", "康复", "打败", "战胜")


def validate_copy_grounding(copy: dict, story_plan: dict) -> dict:
    """V3 P4：无画面证据的文案不烧录——card 必须与绑定槽 caption 有词面重合，
    punchline 的动作断言必须命中末槽画面词，否则丢弃/换无断言兜底。

    lxh_p4_C2 病灶：punchline「可它还是救了他」——22 秒里没有任何"救"的
    画面，文案在替乱剪编造叙事关系。copy 是最后生成的（槽位定稿后），但
    "最后"不等于"免检"。
    """
    from src.agentic_video.story_planner import _text_shingles

    slots = story_plan.get("slots") or []
    by_idx = {int(slot.get("slot_idx", -1)): slot for slot in slots}
    report = {"dropped": [], "replaced_punchline": False, "kept": 0}
    cues = list((copy or {}).get("cues") or [])
    kept = []
    for cue in cues:
        kind = cue.get("kind")
        text = str(cue.get("text") or "")
        if kind == "info_card":
            slot = by_idx.get(int(cue.get("slot_idx", -1)))
            caption = str(((slot or {}).get("source") or {}).get("caption") or "")
            grounded = (slot is not None
                        and slot.get("status") in {"supported", "uncertain"}
                        and caption
                        and bool(_text_shingles(text) & _text_shingles(caption)))
            if not grounded:
                report["dropped"].append({"kind": kind, "text": text,
                                          "reason": "no_visual_evidence"})
                continue
        elif kind == "punchline":
            live = [slot for slot in slots
                    if slot.get("status") in {"supported", "uncertain"}]
            caption = str(((live[-1].get("source") or {}).get("caption") or "")
                          if live else "")
            asserts_action = any(verb in text for verb in _ACTION_ASSERTION_CHARS)
            if asserts_action and not (_text_shingles(text) & _text_shingles(caption)):
                cue = {**cue, "text": _FALLBACK_PUNCHLINES[0],
                       "grounding": "replaced_nonassertive"}
                report["replaced_punchline"] = True
        kept.append(cue)
    copy["cues"] = kept
    report["kept"] = len(kept)
    return report


def write_copy_track(copy: dict, path) -> object:
    """落盘 copy_track（人工审改后经 render 子命令缓存键触发重渲）。"""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(copy, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
