"""extract_frames / detect_shots 单测（纯函数，不跑 ffmpeg）。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.perception.detect_shots import (
    boundaries_to_shots, detect_scoped_shots, parse_showinfo_times)
from src.perception.extract_frames import expected_frame_count


# ---------- frames ----------

def test_expected_count_normal():
    n, fps = expected_frame_count(22.0, 1.0, 64)
    assert n == 22 and fps == 1.0


def test_expected_count_caps_and_lowers_fps():
    # 92.6s @1fps = 93 帧 > 64 → 降 fps
    n, fps = expected_frame_count(92.6, 1.0, 64)
    assert n == 64
    assert 92.6 * fps < 64.5  # 降后的 fps 保证不超上限


def test_expected_count_short_video():
    n, fps = expected_frame_count(6.5, 1.0, 64)
    assert n == 7 and fps == 1.0  # ceil(6.5)


# ---------- shots ----------

def test_parse_showinfo_times():
    stderr = "[Parsed_showinfo] n:   0 pts: 123 pts_time:3.12\n" \
             "garbage line\n" \
             "[Parsed_showinfo] n:   1 pts: 456 pts_time:7.4\n" \
             "[Parsed_showinfo] n:   2 pts: 789 pts_time:12.008\n"
    assert parse_showinfo_times(stderr) == [3.12, 7.4, 12.008]


def test_parse_showinfo_empty():
    assert parse_showinfo_times("no matches here") == []


def test_boundaries_basic_shots():
    out = boundaries_to_shots([3.12, 7.4, 12.0], 22.0, min_shot_len_s=0.5)
    assert out["shot_count"] == 4
    assert out["boundaries_s"] == [0.0, 3.12, 7.4, 12.0, 22.0]
    s0 = out["shots"][0]
    assert s0["start_s"] == 0.0 and s0["end_s"] == 3.12 and abs(s0["mid_s"] - 1.56) < 0.01
    # 索引连续
    assert [s["index"] for s in out["shots"]] == [0, 1, 2, 3]


def test_boundaries_merges_short_shot_into_previous():
    # 3.0~3.2 仅 0.2s < min_shot_len → 并入前镜头，切点从边界中消失
    out = boundaries_to_shots([3.0, 3.2], 10.0, min_shot_len_s=0.5)
    assert out["shot_count"] == 2          # [0,3.2] [3.2,10]
    assert out["boundaries_s"] == [0.0, 3.2, 10.0]
    assert out["shots"][0]["duration_s"] == 3.2


def test_boundaries_no_cuts_single_shot():
    out = boundaries_to_shots([], 22.0, min_shot_len_s=0.5)
    assert out["shot_count"] == 1
    assert out["shots"][0]["start_s"] == 0.0 and out["shots"][0]["end_s"] == 22.0


def test_boundaries_dedup_and_order():
    out = boundaries_to_shots([7.4, 3.12, 3.12, 21.99], 22.0, min_shot_len_s=0.5)
    # 21.99~22.0 太短并入前镜头 → 最终边界以 shots 为准
    assert out["shot_count"] == 3
    assert out["boundaries_s"] == [0.0, 3.12, 7.4, 22.0]


def test_scoped_shots_seek_only_scope_and_normalize_to_absolute(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stderr="pts_time:2.5\npts_time:9.0\n",
        )

    monkeypatch.setattr("src.perception.detect_shots.subprocess.run", fake_run)
    out = detect_scoped_shots(
        Path("movie.mkv"), ffmpeg_bin="ffmpeg", threshold=0.3,
        min_shot_len_s=0.5, start_s=2965.0, end_s=3005.0)

    command, kwargs = calls[0]
    assert command[1:4] == ["-ss", "2965", "-i"]
    assert command[command.index("-t") + 1] == "40"
    assert "setpts=PTS-STARTPTS" in command[command.index("-filter:v") + 1]
    assert kwargs["timeout"] == 600
    assert out["scope_interval"] == [2965.0, 3005.0]
    assert out["timebase_origin_s"] == 2965.0
    assert out["relative_boundaries_s"] == [0.0, 2.5, 9.0, 40.0]
    assert out["boundaries_s"] == [2965.0, 2967.5, 2974.0, 3005.0]
    assert [out["shots"][1]["start_s"], out["shots"][1]["end_s"]] == [2967.5, 2974.0]
