from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agentic_video.zones import (exclusion_reason, excluded_row_indices,
                                     load_source_durations, row_video_durations,
                                     zone_config)

MOVIE = "movie.mp4"
DURATION = 9295.8           # 鬼灭无限城实测时长


def _cfg(library_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(library={},
                           paths=SimpleNamespace(library_dir=library_dir))


def _row(start: float, end: float, *, video: str = MOVIE) -> dict:
    return {"video": video, "start_s": start, "end_s": end}


def test_tail_credits_row_excluded_mid_movie_row_kept():
    config = zone_config(_cfg(Path(".")))
    durations = {MOVIE: DURATION}
    assert exclusion_reason(_row(9060, 9105), durations, config) == "tail_credits"  # w034 ED 段
    assert exclusion_reason(_row(7980, 8025), durations, config) == ""              # w032 正片
    assert exclusion_reason(_row(10, 55), durations, config) == "head_credits"      # 片头


def test_short_sources_are_never_zone_excluded():
    config = zone_config(_cfg(Path(".")))
    durations = {"trailer.mp4": 120.0}
    assert exclusion_reason(_row(0, 30, video="trailer.mp4"), durations, config) == ""
    assert exclusion_reason(_row(100, 119, video="trailer.mp4"), durations, config) == ""


def test_durations_fall_back_to_max_row_end(tmp_path):
    rows = [_row(165, 210), _row(9060, 9105), _row(9090, 9135)]
    assert row_video_durations(rows) == {MOVIE: 9135.0}
    assert excluded_row_indices(rows, _cfg(tmp_path)) == {1, 2}   # 8775s 之后全剔


def test_scan_envelope_duration_wins_over_row_fallback(tmp_path):
    (tmp_path / "sources" / "guimie").mkdir(parents=True)
    (tmp_path / "sources" / "guimie" / "narrative_windows.json").write_text(
        json.dumps({"video": MOVIE, "duration_s": DURATION}), encoding="utf-8")
    assert load_source_durations(tmp_path) == {MOVIE: DURATION}
    assert excluded_row_indices([_row(165, 210), _row(9060, 9105)], _cfg(tmp_path)) == {1}


def test_zone_config_merges_defaults_with_overrides():
    cfg = SimpleNamespace(library={"source_zones": {"tail_s": 600}}, paths=None)
    config = zone_config(cfg)
    assert config == {"head_s": 90.0, "tail_s": 600.0, "min_movie_s": 1800.0}
