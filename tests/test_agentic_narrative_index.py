from __future__ import annotations

import json

from src.agentic_video.narrative_index import (DEFAULT_QUOTAS,
                                                build_entity_registry,
                                                namespace_window_events,
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


def test_window_selection_drops_head_and_tail_credit_zones():
    """片头/片尾职员表窗口不进标注与检索池（2026-09-10 首跑选中 ED 段的教训）。"""
    windows = [
        {"start_s": 10.0, "end_s": 55.0, "dialogue_score": 1.0},     # 片头
        {"start_s": 3030.0, "end_s": 3075.0, "dialogue_score": 0.8},  # 正片中段
        {"start_s": 9060.0, "end_s": 9105.0, "dialogue_score": 1.0},  # 片尾 ED
    ]
    selected = select_narrative_windows(windows, 9295.8, quotas={"dialogue": 3},
                                        exclude_head_s=90.0, exclude_tail_s=360.0)
    assert [row["start_s"] for row in selected] == [3030.0]


def test_dialogue_density_ignores_gapless_punctuation_only_segments():
    samples = [{"t_s": float(i), "motion": 0.2, "cut_density": 0.1,
                "audio_energy": 0.3, "audio_onset": 0.1} for i in range(60)]
    transcript = {"segments": [
        {"start_ms": 0, "end_ms": 30000, "text": "。。。"},
        {"start_ms": 30000, "end_ms": 60000, "text": "必ず守る 絶対に諦めない"},
    ]}

    rows = score_narrative_windows(
        samples, transcript, 60, window_s=30, stride_s=30)

    assert rows[0]["dialogue_raw"] == 0
    assert rows[1]["dialogue_raw"] > 0
    assert rows[1]["dialogue_score"] > rows[0]["dialogue_score"]


def test_entity_registry_and_event_ids_are_stable_across_windows():
    registry = build_entity_registry({
        "2": {"entity_ids": ["black_hair_swordsman"],
              "entity_names": ["黑发持刀少年"]},
        "3": {"entity_ids": ["black_hair_swordsman"],
              "entity_names": ["持刀少年"]},
    })
    parsed = namespace_window_events({
        "shots": {"2": {"event_id": "event_1"},
                  "3": {"event_id": "event_1"}},
        "causal_links": [{"from_event": "event_1", "to_event": "event_1"}],
    }, 4)

    assert registry == [{
        "entity_id": "black_hair_swordsman",
        "visible_names": ["黑发持刀少年", "持刀少年"],
        "shot_idxs": [2, 3],
    }]
    assert parsed["shots"]["2"]["event_id"] == "w004_event_1"
    assert parsed["causal_links"][0]["from_event"] == "w004_event_1"


def test_parse_window_annotations_offsets_dialogue_to_movie_timeline():
    """H1：模型对白是切片内坐标（0 起算）——必须 +窗口偏移归一回电影轴并夹范围。"""
    from src.agentic_video.narrative_index import parse_window_annotations

    raw = json.dumps({"shots": [{
        "shot_idx": 7, "entity_ids": ["e000"], "entity_names": ["少年"],
        "event_id": "w003_1", "event_summary": "奔跑", "story_role": "conflict",
        "dialogue": [
            {"start_s": 1.0, "end_s": 2.5, "original": "走れ", "confidence": 0.9},
            {"start_s": 44.0, "end_s": 60.0, "original": "越界句", "confidence": 0.9},
        ]}], "causal_links": []}, ensure_ascii=False)
    parsed = parse_window_annotations(raw, valid_shot_ids={7},
                                      time_offset_s=3700.0, clip_duration_s=45.0)
    lines = parsed["shots"]["7"]["dialogue"]
    assert lines[0]["start_s"] == 3701.0 and lines[0]["end_s"] == 3702.5
    assert lines[1]["end_s"] == 3745.0                     # 夹到窗口末


def test_atomic_write_json_survives_rerun_and_leaves_no_tmp(tmp_path):
    from src.agentic_video.narrative_index import _atomic_write_json

    target = tmp_path / "state.json"
    _atomic_write_json(target, {"shots": {"1": {"a": 1}}, "completed_windows": [1]})
    _atomic_write_json(target, {"shots": {"1": {"a": 2}}, "completed_windows": [1, 2]})
    assert json.loads(target.read_text(encoding="utf-8"))["shots"]["1"]["a"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_run_narrative_annotations_migrates_legacy_clip_coordinates(tmp_path):
    """H1 存量迁移：无 coord_system 标记的旧对白按窗口起点 +offset，且立即落盘。"""
    import json as _json

    from src.agentic_video.narrative_index import run_narrative_annotations
    from src.config import AppConfig, PathsCfg
    from src.perception import common

    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, perception={}, template={},
        generation={}, logging_level="INFO", library={},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "per", generation_dir=tmp_path / "g",
                       library_dir=tmp_path / "lib"))
    shots_dir = cfg.paths.library_dir / "shots" / "src1__narrative"
    shots_dir.mkdir(parents=True)
    common.write_result_json(shots_dir, tool="shots", aweme_id="src1", params={},
                             output={"video": str(tmp_path / "movie.mp4"), "shots": [
                                 {"shot_idx": 0, "start_s": 3700.0, "end_s": 3710.0,
                                  "duration_s": 10.0, "window_idx": 0},
                                 {"shot_idx": 1, "start_s": 3710.0, "end_s": 3720.0,
                                  "duration_s": 10.0, "window_idx": 0}]})
    legacy = {"shots": {"0": {"dialogue": [{"start_s": 1.0, "end_s": 2.0,
                                            "original": "走れ"}]}},
              "causal_links": [], "completed_windows": [0]}
    ann_path = shots_dir / "narrative_annotations.json"
    ann_path.write_text(_json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    class DoneRunner:                                  # 窗口 0 已完成 → 不会再调模型
        def watch(self, *a, **k):
            raise AssertionError("不应再调模型")

    out = run_narrative_annotations(cfg, shots_dir / "result.json", runner=DoneRunner())
    migrated = _json.loads(out.read_text(encoding="utf-8"))
    line = migrated["shots"]["0"]["dialogue"][0]
    assert line["start_s"] == 3701.0 and line["end_s"] == 3702.0   # +窗口起点 3700
    assert migrated["coord_system"] == "movie"
