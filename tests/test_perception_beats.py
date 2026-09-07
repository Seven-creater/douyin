"""detect_beats 单测（postprocess 纯函数，不跑 librosa/ffmpeg）。"""
from __future__ import annotations

from src.perception.detect_beats import postprocess_onsets


def test_postprocess_filters_close_onsets():
    # 最小间隔 0.3s：0.1/0.15/0.4 → 0.1, 0.4
    assert postprocess_onsets([0.1, 0.15, 0.4], duration_s=10, min_interval_s=0.3, max_beats=160) == [0.1, 0.4]


def test_postprocess_drops_out_of_range():
    assert postprocess_onsets([-0.5, 1.0, 12.0], duration_s=10.0,
                              min_interval_s=0.3, max_beats=160) == [1.0]


def test_postprocess_caps_max_beats():
    times = [i * 0.5 for i in range(100)]
    out = postprocess_onsets(times, duration_s=100, min_interval_s=0.3, max_beats=10)
    assert len(out) == 10 and out == times[:10]


def test_postprocess_rounds_and_keeps_order():
    assert postprocess_onsets([1.23456, 2.0], duration_s=5, min_interval_s=0.3, max_beats=160) == [1.235, 2.0]


def test_postprocess_empty():
    assert postprocess_onsets([], duration_s=5, min_interval_s=0.3, max_beats=160) == []
