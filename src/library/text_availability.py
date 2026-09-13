"""文本可用性统一规则（V5 P0，外审六轮硬修改①）。

"uncertain" 遮蔽对白实锤：库内 389 条对白 translation_zh 全是字面量
"uncertain"（产源 attach_transcript_to_shots 硬写 sentinel），而 294 条
original 是真中文文本；`translation_zh or original` 因 sentinel 非空串短路，
真文本从未进入检索与对白锚定——外审离线回放：修掉后 w15「人和妖一样，
很难定义好坏」立即可锚定。

纪律：
- 判断用**精确集合匹配**（归一 strip+lower 后 `in MISSING_TEXT_MARKERS`），
  绝不做子串包含——真对白可能就含有 "uncertain" 一词；
- 生产端以后写 null（JSON None），本模块只为兼容历史 sentinel 数据；
- 对白真值优先级：translation_zh 可用 → original 可用 → None。
"""
from __future__ import annotations

MISSING_TEXT_MARKERS = frozenset({"", "uncertain", "unknown", "not_applicable"})


def usable_text(value) -> str | None:
    """归一后精确匹配缺失标记；可用则返回原文本，不可用返回 None。"""
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in MISSING_TEXT_MARKERS:
        return None
    return text


def dialogue_text(line: dict) -> str | None:
    """对白行的可用文本：translation_zh 可用 → original 可用 → None。

    取代一切 `line.get("translation_zh") or line.get("original")` 式写法
    ——对 sentinel 字符串，`or` 不会跳到 original，这正是遮蔽根因。"""
    return usable_text((line or {}).get("translation_zh")) \
        or usable_text((line or {}).get("original"))
