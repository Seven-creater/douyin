"""generation planner 单测（纯函数；fixture 复刻纸箱模板时间线形态）。"""
from __future__ import annotations

from src.generation.models import fnv1a
from src.generation.planner import clip_frames_for, merge_segments, plan_units

# 纸箱梗模板的时间线形态（时长与 role 取自真实 Phase 3 产物量级）
CARDBOARD_TL = [
    {"start": 0.0, "end": 2.0, "role": "setup", "visual": "纸箱在地毯上滚动", "speech": None, "text": None},
    {"start": 2.0, "end": 7.0, "role": "buildup", "visual": "镜头展示宿舍，女性跑入", "speech": "这箱子里到底是什么", "text": None},
    {"start": 7.0, "end": 9.0, "role": "buildup", "visual": "女性跪地双手举箱", "speech": None, "text": "知识就是力量"},
    {"start": 9.0, "end": 12.0, "role": "twist", "visual": "纸箱突然打开出现惊喜", "speech": "我有什么特长", "text": None},
    {"start": 12.0, "end": 16.9, "role": "ending", "visual": "魔性舞蹈与文字特效", "speech": None, "text": "关注我"},
]

PLAN_CFG = {"fps": 24, "min_unit_s": 5.2, "max_unit_s": 10.0, "head_pad_s": 0.3, "tail_pad_s": 0.3, "max_frames": 260}


# ---------- 帧数 ----------

def test_frames_minimum():
    assert clip_frames_for(5.0) == 124          # 最小窗口
    assert clip_frames_for(0.1) == 124


def test_frames_step_math():
    # 需要 9.0s → 216 帧 → 向上取 17n+5 = 226
    assert clip_frames_for(9.0) == 226
    # 124 帧容量 5.167s：5.16s 恰好够；5.17s 差 0.08 帧跳到 141
    assert clip_frames_for(5.16) == 124
    assert clip_frames_for(5.17) == 141
    # 帧数恒为 17n+5
    for s in (5.5, 7.0, 9.9):
        f = clip_frames_for(s)
        assert (f - 5) % 17 == 0 and f >= 124


def test_frames_capped():
    assert clip_frames_for(60.0) == 260


# ---------- 合并 ----------

def test_merge_covers_all_no_gaps():
    groups = merge_segments(CARDBOARD_TL, min_unit_s=5.2, max_unit_s=10.0)
    flat = [s for g in groups for s in g]
    assert flat == CARDBOARD_TL                 # 无丢失无重排
    # 相邻组时间相接
    for a, b in zip(groups, groups[1:]):
        assert a[-1]["end"] == b[0]["start"]


def test_merge_min_duration_respected():
    groups = merge_segments(CARDBOARD_TL, min_unit_s=5.2, max_unit_s=10.0)
    for g in groups:
        dur = sum(s["end"] - s["start"] for s in g)
        # 单段自身可小于 min（无相邻可并时），多段组必须 ≥ min
        if len(g) > 1:
            assert dur >= 5.2
        assert dur <= 10.0 + 1e-6


def test_merge_same_role_prefers_grouping():
    # buildup 两段（2-7、7-9）应尽量在同一组
    groups = merge_segments(CARDBOARD_TL, min_unit_s=5.2, max_unit_s=10.0)
    roles = [[s["role"] for s in g] for g in groups]
    assert any("buildup" in r and r.count("buildup") >= 2 for r in roles)


def test_merge_long_single_role_splits():
    tl = [{"start": i * 3.0, "end": (i + 1) * 3.0, "role": "climax", "visual": "x"} for i in range(8)]  # 24s 同 role
    groups = merge_segments(tl, min_unit_s=5.2, max_unit_s=10.0)
    assert all(sum(s["end"] - s["start"] for s in g) <= 10.0 + 1e-6 for g in groups)
    assert len(groups) >= 3


# ---------- plan_units ----------

def test_plan_units_structure_and_offsets():
    template = {"timeline": CARDBOARD_TL, "audio": {"bgm": "电子流行"}}
    plan = plan_units(template, template_id="t1", plan_cfg=PLAN_CFG)
    assert plan.template_id == "t1"
    assert plan.source_duration_s == 16.9
    assert plan.audio_brief == "电子流行"
    assert 2 <= len(plan.units) <= 4
    # 覆盖全部时间线且相接
    assert plan.units[0].timeline_span[0] == 0.0
    assert plan.units[-1].timeline_span[1] == 16.9
    for a, b in zip(plan.units, plan.units[1:]):
        assert a.timeline_span[1] == b.timeline_span[0]
    # 每段取片窗口 = head_pad + 前序段长；首段 = head_pad
    u0 = plan.units[0]
    assert u0.segments[0].unit_offset_s == 0.3
    if len(u0.segments) > 1:
        d = u0.segments[0].end_s - u0.segments[0].start_s
        assert abs(u0.segments[1].unit_offset_s - (0.3 + d)) < 1e-6
    # 生成的时长容量 ≥ duration + pads
    for u in plan.units:
        assert u.num_frames / 24 >= u.duration_s + 0.6 - 1e-6
    # speech_lines 汇总
    all_speech = [s for u in plan.units for s in u.speech_lines]
    assert "这箱子里到底是什么" in all_speech


def test_seed_deterministic():
    assert fnv1a("a|0") == fnv1a("a|0")
    assert fnv1a("a|0") != fnv1a("a|1")
