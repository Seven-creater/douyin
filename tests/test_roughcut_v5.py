import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.roughcut import (_build_plan, _hydrate_transcript_rows,
                                        _overlap, _summary_similarity)
from src.agentic_video.story_planner import fit_slot_intervals
from src.agentic_video.verify_slots import localize_coarse_slots, parse_localization
from src.agentic_video.renderer import write_story_subtitles


def _spec():
    return {
        "spec_version": "roughcut_v2", "source": "luoxiaohei1",
        "source_scope": {"start_s": 2965.0, "end_s": 3005.0},
        "focus_utterance": {"candidate_interval": [2977.5, 2990.63], "query": "好坏"},
        "stages": [
            {"id": "setup", "required": False, "purpose": "语境"},
            {"id": "core_statement", "required": True, "purpose": "核心判断", "evidence_type": "dialogue"},
            {"id": "response", "required": False, "purpose": "回应"},
        ],
        "duration": {"preferred_s": 22.0, "min_s": 12.0, "max_s": 26.4},
        "audio_variants": ["source_only", "bgm_mix"],
    }


def test_cross_scope_container_is_candidate_but_final_evidence_is_inside():
    row = {"start_s": 2955.0, "end_s": 2980.0}
    assert _overlap(row, {"start_s": 2965.0, "end_s": 3005.0})
    plan = _build_plan([
        {"video": "m.mp4", "video_stem": "luoxiaohei1__narrative", "start_s": 2955, "end_s": 2995,
         "event_id": "e", "dialogue": [{"utterance_interval": [2977.5, 2984], "original": "核心判断"}]},
    ], _spec(), "hash", theme="t", video="m.mp4")
    source = plan["slots"][1]["source"]
    assert source["container_interval"] == [2955.0, 2995.0]
    assert source["utterance_interval"] == [2977.5, 2984.0]
    assert source.get("required_evidence_interval") is None


def test_old_window_view_is_hydrated_to_full_transcript_segment(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [{"start_ms": 100000, "end_ms": 115000,
                                                     "text": "完整语义证据"}]}), encoding="utf-8")
    rows = [{"start_s": 95, "end_s": 105, "dialogue": [{"start_s": 100, "end_s": 105,
                                                            "original": "旧窗口截断"}]}]
    _hydrate_transcript_rows(rows, transcript)
    line = rows[0]["dialogue"][0]
    assert line["utterance_interval"] == [100.0, 115.0]
    assert line["overlap_interval"] == [100.0, 105.0]
    assert line["partial"] is True


def test_relative_localization_records_origin_and_normalizes(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"placeholder")

    class Runner:
        def watch(self, *_args, **_kwargs):
            return SimpleNamespace(text=json.dumps({"found": True, "start": 20, "end": 30,
                                                     "timebase": "relative", "evidence": "e", "confidence": .9}))

    plan = {"slots": [{"slot_idx": 0, "status": "supported", "target_interval": [0, 10],
                       "need_spec": {"required": True, "need": "e"},
                       "source": {"video": str(video), "start_s": 2965, "end_s": 3005, "anchor": "coarse"}}]}
    log = localize_coarse_slots(None, plan, runner=Runner(), output_dir=tmp_path)
    assert log[0]["timebase_origin_s"] == 2965
    assert log[0]["normalized_absolute_interval"] == [2985.0, 2995.0]
    assert plan["slots"][0]["source"]["required_evidence_interval"] == [2985.0, 2995.0]


def test_complete_story_shorter_than_preferred_is_not_padded():
    plan = {"plan_kind": "roughcut", "target_duration_s": 22, "duration_policy": {"preferred_s": 22, "min_s": 12, "max_s": 26.4},
            "slots": [{"slot_idx": 0, "status": "supported", "need_spec": {"required": True},
                       "source": {"start_s": 100, "end_s": 117, "required_evidence_interval": [100, 117]},
                       "target_interval": [0, 22]}]}
    fit_slot_intervals(plan, mode="content_preserving")
    assert plan["target_duration_s"] == pytest.approx(17)


def test_subtitles_require_full_verified_utterance_and_skip_punctuation(tmp_path):
    out = tmp_path / "a.srt"
    asset = {"slots": [{"slot_idx": 0, "start_s": 0, "end_s": 5}]}
    retrieval = [{"slot_idx": 0, "picked": {"source_start_s": 100, "source_end_s": 105,
        "actual_rendered_source_interval": [100, 105], "dialogue": [
            {"utterance_interval": [100, 101], "original": "，"},
            {"utterance_interval": [101, 102], "original": "完整证据"},
            {"utterance_interval": [102, 106], "original": "画外尾句"}]}}]
    write_story_subtitles(asset, retrieval, out)
    text = out.read_text(encoding="utf-8")
    assert "完整证据" in text and "画外尾句" not in text and "，" not in text


def test_blind_variant_summary_similarity_tolerates_small_paraphrase():
    assert _summary_similarity("两个人讨论人和妖不能简单判断好坏",
                               "两人讨论不能按人或妖简单判断好坏") >= 0.2
    assert _summary_similarity("两个人讨论好坏", "森林里发生战斗") < 0.2
