"""素材库（src/library）单测：镜头切分/故事板校验/检索排序/装配命令（全 mock 零模型）。"""
from __future__ import annotations

import json

import numpy as np

from src.library.caption_shots import CAPTION_PROMPT
from src.library.index_shots import build_shots, parse_scene_log
from src.library.retrieve import rank_segments
from src.library.story import build_story_prompt, capability_summary, validate_storyboard


# ---------- B1 镜头切分 ----------


def test_caption_prompt_is_source_agnostic_and_forbids_character_guessing():
    assert "不得依据 IP 常识" in CAPTION_PROMPT
    assert "蜘蛛侠" not in CAPTION_PROMPT
    assert "章鱼博士" not in CAPTION_PROMPT

def test_parse_scene_log_dedup():
    assert parse_scene_log("pts_time:2.0\npts_time:2.0\npts_time:5.5") == [0.0, 2.0, 5.5]


def test_build_shots_filters_and_caps():
    shots = build_shots([0.0, 1.0, 1.2, 8.0, 20.0], 20.0, min_len_s=0.4, max_shots=10)
    spans = [(s["start_s"], s["end_s"]) for s in shots]
    assert (0.0, 1.0) in spans and (1.2, 8.0) in spans
    assert all(e - b >= 0.4 for b, e in spans)          # 1.0~1.2 过短被滤


# ---------- C2 故事板 ----------

_TPL = {"core_meme": "m", "fixed_elements": ["f1"],
        "timeline": [{"start": 0.0, "end": 2.0, "role": "setup", "visual": "v", "text": None},
                     {"start": 2.0, "end": 5.0, "role": "twist", "visual": "v2", "text": "台词"}]}


def test_validate_storyboard_ok_and_fails():
    ok = [{"start": 0.0, "end": 2.0, "subject": "蜘蛛侠", "action": "摆动",
           "shot_scale": "wide", "mood": "燃", "need_zh": "主体：蜘蛛侠摆动",
           "caption_zh": "摆动"},
          {"start": 2.0, "end": 5.0, "subject": "面具", "action": "摘下",
           "shot_scale": "close", "mood": "震惊", "need_zh": "主体：摘面具",
           "caption_zh": "台词"}]
    assert validate_storyboard(ok, _TPL) == []
    bad = [dict(ok[0], start=0.5), ok[1]]
    assert any("不一致" in e for e in validate_storyboard(bad, _TPL))
    assert validate_storyboard([ok[0]], _TPL) and "段数" in validate_storyboard([ok[0]], _TPL)[0]


def test_story_prompt_and_summary():
    rows = [{"caption": "主体：彼得帕克摘面具；动作：震惊", "video_stem": "t1",
             "duration_s": 2.5} for _ in range(3)]
    s = capability_summary(rows)
    p = build_story_prompt(_TPL, s)
    assert "素材库" in p and "2.0~5.0" in p and "f1" in p


# ---------- C3a 检索排序 ----------

def _rows(n=6):
    return [{"video_stem": "v", "shot_idx": i, "duration_s": 3.0,
             "caption": f"镜头{i}"} for i in range(n)]


def test_rank_segments_picks_and_dedupe():
    rows = _rows()
    # 段0 最优是镜头1，段1 因相邻去重跳过镜头1 取次优
    emb = np.zeros((6, 4), "float32")
    emb[1] = [1, 0, 0, 0]
    emb[2] = [0.9, 0, 0, 0]
    q0 = np.array([1, 0, 0, 0], "float32")
    q1 = np.array([1, 0, 0, 0], "float32")
    out = rank_segments(rows, emb, [q0, q1], [2.0, 2.0], queries=["蜘蛛侠", "蜘蛛侠"],
                        top_k=3, min_cosine=0.1, dedupe_window=2)
    assert out[0]["picked"]["row_idx"] == 1
    assert out[1]["picked"]["row_idx"] == 2          # 去重生效
    assert not out[0]["missing"]


def test_rank_segments_marks_missing_when_low_score():
    rows = _rows(3)
    emb = np.zeros((3, 4), "float32")               # 全零 → 全零分
    q = np.array([1, 0, 0, 0], "float32")
    out = rank_segments(rows, emb, [q], [2.0], queries=["x"], top_k=3,
                        min_cosine=0.18)
    assert out[0]["picked"] is None and "库内不足" in out[0]["missing"]


def test_rank_segments_duration_short_falls_back():
    rows = [{"video_stem": "v", "shot_idx": 0, "duration_s": 0.8, "caption": "短镜头"}]
    emb = np.array([[1.0, 0, 0, 0]], "float32")
    q = np.array([1, 0, 0, 0], "float32")
    out = rank_segments(rows, emb, [q], [5.0], queries=["x"], top_k=3, min_cosine=0.1)
    assert out[0]["picked"] is not None and "时长不足" in out[0]["missing"]


# ---------- C3b 装配命令 ----------

def test_build_seg_cmd_pad_and_crop():
    from src.library.assemble_lib import build_seg_cmd
    from pathlib import Path
    cmd = build_seg_cmd(Path("a.mp4"), Path("o.mp4"), shot_start=1.5, shot_dur=2.0,
                        need_dur=3.0, crf=20)
    s = " ".join(cmd)
    assert "tpad=stop_mode=clone:stop_duration=1" in s    # 不足补冻结
    assert "crop=544:960" in s and "-ss 1.5" in s
    cmd2 = build_seg_cmd(Path("a.mp4"), Path("o.mp4"), shot_start=0, shot_dur=9.0,
                         need_dur=3.0, crf=20)
    assert "tpad" not in " ".join(cmd2) and "-t 3" in " ".join(cmd2)


def test_merge_dialogue_prefers_asr_timing_and_keeps_model_translation():
    """H1 合并侧：ASR 时间戳为准；模型补翻译；模型空数组不再抹掉 ASR 真值。"""
    from src.library.build_index import merge_dialogue

    asr = [{"start_s": 10.0, "end_s": 12.0, "original": "走れ",
            "translation_zh": "uncertain", "confidence": 0.7}]
    model = [{"start_s": 10.2, "end_s": 12.1, "original": "走れ",
              "translation_zh": "快跑！", "confidence": 0.9}]
    merged = merge_dialogue(asr, model)
    assert merged[0]["start_s"] == 10.0 and merged[0]["end_s"] == 12.0   # ASR 时间戳
    assert merged[0]["translation_zh"] == "快跑！"                        # 模型翻译
    assert merged[0]["confidence"] == 0.9
    # 模型空数组（合法无对白）不能覆盖 ASR
    assert merge_dialogue(asr, []) == asr
    # 无 ASR 时用模型对白兜底
    assert merge_dialogue([], model) == model
