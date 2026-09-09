from __future__ import annotations

from src.agentic_video.narrative_index import (DEFAULT_QUOTAS,
                                                score_narrative_windows,
                                                select_narrative_windows)


def _samples(duration: int = 240):
    return [{"t_s": float(i), "motion": (i % 10) / 10,
             "cut_density": (i % 7) / 7, "audio_energy": (i % 5) / 5,
             "audio_onset": (i % 3) / 3} for i in range(duration)]


def test_narrative_window_selection_has_type_quotas_and_no_large_overlap():
    transcript = {"segments": [
        {"start_ms": 0, "end_ms": 40000, "text": "对白"},
        {"start_ms": 90000, "end_ms": 130000, "text": "对白"},
    ], "emotions": ["SAD"]}
    rows = score_narrative_windows(_samples(), transcript, 240, window_s=30, stride_s=15)
    selected = select_narrative_windows(rows, 240,
                                        quotas={key: 2 for key in DEFAULT_QUOTAS})
    assert len(selected) <= 8
    assert {row["selection_type"] for row in selected} <= set(DEFAULT_QUOTAS)
    for idx, left in enumerate(selected):
        for right in selected[idx + 1:]:
            overlap = max(0, min(left["end_s"], right["end_s"])
                          - max(left["start_s"], right["start_s"]))
            assert overlap / 30 <= 0.35
