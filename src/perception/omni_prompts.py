"""omni 观看 prompt 与小节解析（纯文本模块，无重依赖）。"""
from __future__ import annotations

import re

BASELINE_PROMPT = """你是一名短视频内容分析专家。请仔细观看这段抖音视频（画面与声音都要听），严格按下列六个小节输出，要求：
1. 使用给定的中文小节标题、顺序不变、不增减小节；
2. 每小节 2~6 句话或条目，具体、可验证，不写空话；
3. 时间用「第X秒~第Y秒」格式；
4. 不确定的内容明确写「不确定」，不要编造。

## 1. 内容概述
（一句话：视频主题、主角、核心看点）

## 2. 叙事时间线
（按时间顺序列关键情节节点，标注大致秒数）

## 3. 语音内容归纳
（说话内容要点/对话/口号；若无语音则写「无语音」并描述环境声）

## 4. BGM与音效
（背景音乐风格/情绪/节奏变化、出现的音效与时机、BGM 与内容的关系）

## 5. 画面与镜头变化
（镜头数量与切换节奏、景别与运镜、字幕/贴纸/特效的使用情况）

## 6. 热门模板要素猜测
（若把这条视频当作可复制的「模板」：固定钩子/结构/反转点/情绪曲线是什么，翻拍应保留什么、替换什么；为后续自动化模板抽取提供依据）"""

CLIP_PROMPT_TEMPLATE = """请仔细观看这段视频片段（取自原视频第 {start:.1f} 秒 ~ 第 {end:.1f} 秒，画面与声音都要听），回答以下问题：
{question}
要求：具体、引用画面与声音证据；时间用片段内的相对秒数；不确定就明说。"""

SECTION_KEYS = {
    "内容概述": "content_summary",
    "叙事时间线": "timeline",
    "语音内容归纳": "speech",
    "BGM与音效": "bgm_sfx",
    "画面与镜头变化": "visual_shots",
    "热门模板要素猜测": "template_elements",
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def dedupe_repetition(text: str, section_marker: str = "### 1.") -> str:
    """贪心解码复读保险：同一小节标记第二次出现即截断（实测 30B 在短输入上有复读倾向）。"""
    first = text.find(section_marker)
    if first < 0:
        return text
    second = text.find(section_marker, first + len(section_marker))
    return text[:second].rstrip() if second > 0 else text


def parse_baseline_sections(text: str) -> dict[str, str]:
    """按 '## ' 行切分六小节；缺节容忍（值为空串）；剥离 <think>。"""
    cleaned = _THINK_RE.sub("", text).strip()
    sections = {key: "" for key in SECTION_KEYS.values()}
    parts = re.split(r"^##\s*", cleaned, flags=re.MULTILINE)
    # parts[0] 是第一个 ## 之前的内容（通常为空或总起句）
    if parts and parts[0].strip():
        sections["content_summary"] = parts[0].strip()
    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        for key_cn, key_en in SECTION_KEYS.items():
            if key_cn in title:
                sections[key_en] = body
                break
    return sections


def build_clip_prompt(start_s: float, end_s: float, question: str) -> str:
    return CLIP_PROMPT_TEMPLATE.format(start=start_s, end=end_s, question=question)
