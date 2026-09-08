"""外部上下文增强（fetch_context → 模板抽取 → 生成 prompt）单测。

背景（2026-09-08）：纸箱梗模板丢失「王者荣耀·安琪拉模仿」核心 → 标题/相关视频/
热评进上下文 + prompt 硬规则 + 覆盖完整性软校验。
"""
from __future__ import annotations

from src.generation.prompts import REWRITE_PROMPT_HEADER, VARIANT_PROPOSAL_PROMPT_HEADER
from src.template.context import build_context
from src.template.prompts import TEMPLATE_PROMPT_HEADER
from src.template.schema import validate_template


def _valid_template(timeline: list[dict]) -> dict:
    return {
        "trend_summary": "模仿游戏角色制造反差",
        "core_meme": "模仿《王者荣耀》安琪拉的台词与动作",
        "timeline": timeline,
        "audio": {"bgm": None, "beat_points": [], "speech": []},
        "fixed_elements": ["安琪拉台词模仿", "标志性施法动作"],
        "replaceable_elements": ["场景"],
        "generation_plan": ["按 timeline 生成"],
    }


# ---------- context 装配 ----------

def test_build_context_includes_external_signals():
    bundle = {
        "metadata": {"wellbyte": {"title": "你的本命英雄#安琪拉#安琪拉教学",
                                  "stats": {"likes": 10, "views": 100}}},
        "omni_full": {"sections": {}},
        "context": {
            "comments": [{"text": "安琪拉本拉哈哈哈", "digg_count": 5200}],
            "related": [{"title": "你的本命英雄#虞姬 #虞姬教学", "author": "叶子_",
                         "likes": "2.6万", "aweme_id": "1"}],
        },
    }
    ctx, stats = build_context(bundle)
    assert "[热评·观众反应] ♡5200 安琪拉本拉哈哈哈" in ctx
    assert "[相关视频·同款/系列] 你的本命英雄#虞姬 #虞姬教学（作者叶子_ ♡2.6万）" in ctx
    assert "context" in stats["tools_used"]


def test_build_context_without_external_still_works():
    ctx, _ = build_context({"metadata": {"wellbyte": {"title": "t"}}, "omni_full": {"sections": {}}})
    assert "[标题] t" in ctx and "热评" not in ctx


# ---------- 覆盖完整性软校验 ----------

def test_validate_warns_tail_uncovered():
    tl = [{"start": 0.0, "end": 5.0, "role": "setup", "visual": "v", "speech": None, "text": None}]
    res = validate_template(_valid_template(tl), duration_s=16.9,
                            shot_boundaries=[0.0, 5.0], beat_points_known=[])
    assert res.ok  # 软警告不拒收
    assert any("尾部未覆盖" in w for w in res.warnings)


def test_validate_warns_gap():
    tl = [
        {"start": 0.0, "end": 3.5, "role": "setup", "visual": "v", "speech": None, "text": None},
        {"start": 5.0, "end": 16.5, "role": "ending", "visual": "v", "speech": None, "text": None},
    ]
    res = validate_template(_valid_template(tl), duration_s=16.9,
                            shot_boundaries=[0.0, 3.5, 5.0], beat_points_known=[])
    assert res.ok
    assert any("间隔未覆盖" in w for w in res.warnings)


def test_validate_full_coverage_no_warning():
    tl = [{"start": 0.0, "end": 16.5, "role": "setup", "visual": "v", "speech": None, "text": None}]
    res = validate_template(_valid_template(tl), duration_s=16.9,
                            shot_boundaries=[0.0], beat_points_known=[])
    assert res.ok and not any("未覆盖" in w for w in res.warnings)


# ---------- prompt 硬规则 ----------

def test_template_prompt_has_imitation_rules():
    assert "模仿对象或 IP" in TEMPLATE_PROMPT_HEADER
    assert "连续覆盖全片" in TEMPLATE_PROMPT_HEADER


def test_generation_prompts_carry_imitation_core():
    assert "模仿某角色/IP" in REWRITE_PROMPT_HEADER
    assert "不得替换被模仿的角色" in VARIANT_PROPOSAL_PROMPT_HEADER
