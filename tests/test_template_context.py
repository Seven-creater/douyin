"""template context 单测（fixtures 假 bundle，对齐真实 output 结构）。"""
from __future__ import annotations

from src.template.context import (
    beat_points_known_of,
    build_context,
    duration_of,
    estimate_tokens,
    shot_boundaries_of,
)


def fake_bundle() -> dict:
    return {
        "aweme_id": "test1",
        "metadata": {"wellbyte": {"title": "水上飞人 #户外", "stats": {"likes": 130745, "views": 10805641}}},
        "inspect": {"duration_s": 22.0, "width": 576, "height": 1024, "fps": 30.0},
        "shots": {"shot_count": 3, "boundaries_s": [0.0, 3.2, 12.1, 22.0],
                   "shots": [{"duration_s": 3.2}, {"duration_s": 8.9}, {"duration_s": 9.9}]},
        "ocr": {"text_events": [
            {"text": "水上飞人", "t_start_s": 0.0, "t_end_s": 6.0, "occurrences": 7},
            {"text": "关注我", "t_start_s": 18.0, "t_end_s": 20.0, "occurrences": 2},
        ]},
        "transcribe": {"full_text": "无语音内容", "audio_events": ["BGM"], "emotions": []},
        "beats": {"beat_points_s": [0.5, 1.0, 2.1, 3.6]},
        "omni_full": {"sections": {
            "content_summary": "男子湖面飞板表演",
            "timeline": "第0-3秒起步，第3-22秒升空表演",
            "speech": "无语音",
            "bgm_sfx": "高能量电子乐",
            "visual_shots": "三个镜头",
            "template_elements": "固定钩子是静止到升空的反差",
        }},
    }


def test_build_context_contains_all_sources():
    ctx, stats = build_context(fake_bundle())
    assert "水上飞人 #户外" in ctx
    assert "3.2" in ctx                    # 镜头边界
    assert "「水上飞人」" in ctx             # OCR 事件
    assert "无语音内容" in ctx               # ASR
    assert "0.5" in ctx                    # 节拍
    assert "男子湖面飞板表演" in ctx          # omni 概述
    assert set(stats["tools_used"]) == {"metadata", "inspect", "shots", "ocr", "transcribe", "beats", "omni_full"}


def test_build_context_omits_speech_section_when_asr_present():
    ctx, _ = build_context(fake_bundle())
    assert "speech: 无语音" not in ctx       # ASR 有文本时 omni.speech 节跳过


def test_build_context_budget():
    ctx, stats = build_context(fake_bundle(), max_chars=6000)
    assert len(ctx) <= 6200                  # 截断后仍保头尾
    assert stats["chars"] == len(ctx)
    assert stats["est_tokens"] > 0


def test_build_context_truncates_when_tiny_budget():
    ctx, stats = build_context(fake_bundle(), max_chars=300)
    assert "已截断" in ctx and "__context__" in stats["sections_truncated"]


def test_build_context_tolerates_missing_tools():
    b = fake_bundle()
    b["inspect"] = b["shots"] = b["ocr"] = b["transcribe"] = b["beats"] = None
    ctx, stats = build_context(b)
    assert "未知" in ctx and "未提供" in ctx and "未检测" in ctx
    assert "omni_full" in stats["tools_used"]


def test_build_context_deterministic():
    c1, _ = build_context(fake_bundle())
    c2, _ = build_context(fake_bundle())
    assert c1 == c2


def test_accessors():
    b = fake_bundle()
    assert duration_of(b) == 22.0
    assert shot_boundaries_of(b) == [0.0, 3.2, 12.1, 22.0]
    assert beat_points_known_of(b) == [0.5, 1.0, 2.1, 3.6]


def test_accessors_missing():
    b = {"inspect": None, "shots": None, "beats": None,
         "omni_full": {"video": {"duration_s": 18.0}}}
    assert duration_of(b) == 18.0
    assert shot_boundaries_of(b) == [] and beat_points_known_of(b) == []


def test_estimate_tokens_mixed():
    assert estimate_tokens("中文") == 2
    assert 1 <= estimate_tokens("abcd") <= 2   # int(4*0.3)=1
