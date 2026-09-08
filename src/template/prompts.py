"""模板抽取 prompts（纯文本模块，无重依赖）。"""
from __future__ import annotations

import re

ROLE_ENUM = ("setup", "buildup", "twist", "climax", "ending", "other")

_TEMPLATE_SKELETON = (
    '{"trend_summary": "一句话：这条视频为什么火", '
    '"core_meme": "核心梗/核心看点", '
    '"timeline": [{"start": 0.0, "end": 3.2, "role": "setup", "visual": "画面内容", '
    '"speech": "语音内容概括或null", "text": "画面字幕原文或null"}], '
    '"audio": {"bgm": "BGM描述或null", "beat_points": [], "speech": []}, '
    '"fixed_elements": ["翻拍必须保留的元素：说明"], '
    '"replaceable_elements": ["可替换的元素：说明"], '
    '"generation_plan": ["给视频生成模型的分段指令"]}'
)

TEMPLATE_PROMPT_HEADER = """你是一名短视频模板拆解专家。下面给你一条抖音热门视频的多源分析材料
（视频整体理解、语音转写、画面字幕 OCR、镜头切分、节拍检测）。请把这些材料综合成一份结构化「热门模板」，
供后续批量翻拍使用。

【硬性规则】
1. 只输出一个 JSON 对象，不要 markdown 代码围栏，不要任何解释文字。
2. timeline 的 start/end 必须取自材料中给出的镜头边界时刻（可合并相邻镜头），不允许自造时刻；
   区间按时间升序且互不重叠（可相接）。
3. timeline[].role 只能取：setup（铺垫）/ buildup（推进）/ twist（反转）/ climax（高潮）/ ending（收尾）/ other。
4. audio.beat_points 只能从材料给出的节拍点列表中选择若干个，禁止自造数字；材料未提供节拍则留空数组 []。
5. 上下文材料里没有依据的内容，对应字段填 null 或字符串 "uncertain"，禁止编造。
6. speech 字段写概括（ASR 转写无时间戳，按叙事时间线推断落位），不要逐字引用长段原文；
   text 字段必须是 OCR 事件里出现过的字幕文字。
7. 标题/热评/相关视频标题是判断「为什么火」的首要外部证据：若其中出现具体模仿对象或 IP
   （游戏/影视/动漫角色、名人、歌曲）或「教学/belike/模仿」等字样，core_meme 与 trend_summary
   必须点明「模仿谁+出处」（如「模仿《王者荣耀》角色安琪拉的语音台词与动作」），并把被模仿的
   台词与标志性动作列在 fixed_elements 最前面——模仿关系是必须保留的梗核心，道具/场景是次要的；
   replaceable_elements 不得包含被模仿的角色/出处本身。
8. timeline 必须连续覆盖全片：相邻区间相接（不留空隙），最后一段 end 距视频总时长 ≤1s，
   禁止只覆盖前半段。

【输出 JSON 骨架】（字段名与类型严格一致）
""" + _TEMPLATE_SKELETON + """

【视频材料】
"""


def build_template_prompt(context: str) -> str:
    return TEMPLATE_PROMPT_HEADER + context


def build_repair_prompt(context: str, raw_output: str, errors: list[str]) -> str:
    return (
        "你上一次的输出未能通过校验。请根据错误清单修正，只返回修正后的完整 JSON 对象"
        "（无围栏、无解释），字段与规则如下。\n\n"
        "【规则摘要】start/end 取自镜头边界；role 只能取 "
        + "/".join(ROLE_ENUM)
        + "；beat_points 只能取自给定列表；无依据内容标 uncertain 或 null；"
        "timeline 连续覆盖全片；标题/热评出现模仿对象时必须点明出处。\n\n"
        "【上次输出】\n" + raw_output[:3000] + "\n\n"
        "【校验错误】\n- " + "\n- ".join(errors[:20]) + "\n\n"
        "【视频材料】\n" + context
    )


# 复读保险（JSON 版）：同一顶层键名第二次出现即截断
_KEY_RE_CACHE: dict[str, re.Pattern] = {}


def dedupe_json_repetition(text: str, first_key: str = '"trend_summary"') -> str:
    pat = _KEY_RE_CACHE.setdefault(first_key, re.compile(re.escape(first_key)))
    first = text.find(first_key)
    if first < 0:
        return text
    second = text.find(first_key, first + len(first_key))
    if second < 0:
        return text
    # 回退到 second 之前最后一个 '{'（截掉复读出来的第二个对象开头）
    cut = text.rfind("{", 0, second)
    return text[: cut if cut > 0 else second].rstrip().rstrip(",")
