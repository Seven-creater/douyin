"""omni 模块单测：prompt 小节解析、clip prompt、（无 torch 的）纯函数部分。"""
from __future__ import annotations

from src.perception.omni_prompts import (
    BASELINE_PROMPT,
    build_clip_prompt,
    parse_baseline_sections,
)
from src.perception.omni_watch_full import build_prompt

SAMPLE_ANSWER = """## 1. 内容概述
一名男子在湖面表演水上飞板，喷射水柱将他托起。

## 2. 叙事时间线
第0秒~第3秒：男子踩板起步。
第3秒~第22秒：飞板升空表演翻转。

## 3. 语音内容归纳
无语音，仅有环境声。

## 4. BGM与音效
节奏感强的电子乐，升空时刻有重音。

## 5. 画面与镜头变化
约3个镜头，竖屏全景为主。

## 6. 热门模板要素猜测
固定钩子是「静止→突然升空」的反转；翻拍保留节奏，替换主角与场景。"""


def test_baseline_prompt_has_six_sections():
    assert BASELINE_PROMPT.count("## ") == 6
    assert "热门模板要素猜测" in BASELINE_PROMPT


def test_parse_sections_full_answer():
    s = parse_baseline_sections(SAMPLE_ANSWER)
    assert set(s.keys()) == {"content_summary", "timeline", "speech", "bgm_sfx",
                             "visual_shots", "template_elements"}
    assert "水上飞板" in s["content_summary"]
    assert "第0秒" in s["timeline"]
    assert "无语音" in s["speech"]
    assert "电子乐" in s["bgm_sfx"]
    assert "反转" in s["template_elements"]


def test_parse_sections_tolerates_missing_and_think():
    s = parse_baseline_sections("<think>内心戏</think>## 1. 内容概述\n只有概述。")
    assert s["content_summary"] == "只有概述。"
    assert s["timeline"] == ""  # 缺节容忍


def test_parse_sections_preamble_fallback():
    s = parse_baseline_sections("直接一句话概括。\n## 2. 叙事时间线\n第1秒开始。")
    assert s["content_summary"] == "直接一句话概括。"
    assert "第1秒" in s["timeline"]


def test_build_clip_prompt():
    p = build_clip_prompt(3.0, 9.5, "这段在演什么？")
    assert "第 3.0 秒" in p and "第 9.5 秒" in p
    assert "这段在演什么？" in p


def test_build_prompt_with_question():
    p = build_prompt("有没有出现品牌logo？")
    assert "有没有出现品牌logo？" in p and "热门模板要素猜测" in p


def test_build_prompt_without_question_is_baseline():
    assert build_prompt(None) == BASELINE_PROMPT
