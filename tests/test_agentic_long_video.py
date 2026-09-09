from __future__ import annotations

from src.agentic_video.long_video import (aggregate_action_windows,
                                           robust_normalize,
                                           select_action_windows)


def test_robust_normalize_is_bounded_and_handles_constant():
    assert robust_normalize([2, 2, 2]) == [0.0, 0.0, 0.0]
    values = robust_normalize([0, 1, 2, 100])
    assert all(0 <= value <= 1 for value in values)
    assert values[0] == 0.0 and values[-1] == 1.0


def test_action_window_score_prefers_motion_cuts_and_audio():
    samples = []
    for second in range(180):
        hot = 60 <= second < 105
        samples.append({"t_s": second, "motion": 20 if hot else 1,
                        "cut": 4 if hot else 0, "audio_energy": 1 if hot else 0.1,
                        "audio_onset": 0.8 if hot else 0.01})
    windows = aggregate_action_windows(samples, 180, window_s=45, stride_s=15)
    best = max(windows, key=lambda row: row["action_score"])
    assert best["start_s"] <= 60 and best["end_s"] >= 90


def test_selection_enforces_nms_count_and_partitions():
    windows = []
    for index in range(40):
        start = index * 15.0
        windows.append({"start_s": start, "end_s": start + 45,
                        "action_score": 1 - index / 100,
                        "motion_norm": 1.0, "cut_density_norm": 1.0,
                        "audio_energy_norm": 1.0, "audio_onset_norm": 1.0})
    selected = select_action_windows(windows, 600, max_windows=8, partitions=4)
    assert len(selected) <= 8
    assert all(sum(row["partition"] == part for row in selected) <= 2 for part in range(4))
    assert all(selected[i]["end_s"] <= selected[i + 1]["start_s"] + 9.0
               for i in range(len(selected) - 1))
