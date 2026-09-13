import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.roughcut import (_build_plan, _human_acceptance_reasons,
                                        _failure, _hydrate_transcript_rows, _overlap,
                                        _read_spec, _roughcut_blind_reasons,
                                        _summary_similarity,
                                        finalize_roughcut_delivery)
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


def test_v3_spec_locks_editorial_control_experiment():
    spec = _spec()
    spec["spec_version"] = "roughcut_v3"
    spec["editorial"] = {
        "candidate_min": 2, "candidate_max": 5,
        "variant_ids": ["viewpoint", "question_answer", "core_close"],
        "av_sync": "locked", "source_order": "chronological",
    }
    parsed, digest = _read_spec(spec)
    assert parsed["editorial"]["candidate_min"] == 2
    assert len(digest) == 64
    with pytest.raises(ValueError, match="candidates must be 2..5"):
        _read_spec({**spec, "editorial": {**spec["editorial"], "candidate_min": 3}})


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


def test_roughcut_blind_uses_local_speaker_contract_not_general_protagonist():
    base = {
        "parsed": True,
        "consistent_protagonist": False,
        "story_in_one_sentence": "一位妖怪向小孩解释人与妖都不能简单判断好坏",
        "core_statement": "人和妖一样，很难绝对定义好坏",
        "speech_clear": True,
        "music_present": False,
        "speaker_description": "绿皮肤年长妖怪",
        "addressee_description": "黑发猫耳小孩",
        "speaker_addressee_stable": True,
    }
    mix = {**base, "story_in_one_sentence": "妖怪告诉小孩人和妖的好坏都不是绝对的",
           "core_statement": "人和妖的好坏都不是绝对的", "music_present": True}
    reasons, similarity = _roughcut_blind_reasons(
        {"source_only": base, "bgm_mix": mix})
    assert reasons == []
    assert similarity >= 0.2


def test_editorial_final_blind_requires_opening_function_and_ending():
    base = {
        "parsed": True, "story_in_one_sentence": "人和妖不能简单判断好坏",
        "core_statement": "人和妖的好坏都不是绝对的",
        "speaker_description": "人物A", "addressee_description": "人物B",
        "speaker_addressee_stable": True, "speech_clear": True,
        "music_present": False, "opening_reason_clear": True,
        "functionless_span_present": False,
        "transitions_have_clear_function": True, "ending_intentional": True,
    }
    reasons, _ = _roughcut_blind_reasons(
        {"source_only": base, "bgm_mix": {**base, "music_present": True}},
        editorial_required=True)
    assert reasons == []
    reasons, _ = _roughcut_blind_reasons(
        {"source_only": {**base, "functionless_span_present": True},
         "bgm_mix": {**base, "music_present": True}},
        editorial_required=True)
    assert "blind_source_only_functionless_span" in reasons


def test_human_acceptance_requires_restatement_and_audio_checks():
    assert _human_acceptance_reasons(None) == ["human_acceptance_pending"]
    approved = {
        "approved": True,
        "story_in_one_sentence": "两人讨论人和妖不能简单按类别判断好坏",
        "source_only_speech_clear": True,
        "bgm_mix_speech_clear": True,
        "bgm_mix_music_present": True,
    }
    assert _human_acceptance_reasons(approved) == []
    assert "human_bgm_mix_music_present_not_confirmed" in _human_acceptance_reasons(
        {**approved, "bgm_mix_music_present": False})
    editorial = {**approved, "core_meaning_preserved": True,
                 "opening_reason_clear": True, "no_functionless_shots": True,
                 "all_cuts_have_editorial_reason": True, "ending_intentional": True}
    assert _human_acceptance_reasons(editorial, editorial_required=True) == []
    assert "human_ending_intentional_not_confirmed" in _human_acceptance_reasons(
        {**editorial, "ending_intentional": False}, editorial_required=True)


def test_finalize_delivery_only_after_human_acceptance(tmp_path):
    variants_dir = tmp_path / "variants"
    source = variants_dir / "source_only" / "rendered.mp4"
    mix = variants_dir / "bgm_mix" / "rendered.mp4"
    master = tmp_path / "content_master.mp4"
    source_audio = tmp_path / "source_audio.m4a"
    for path, payload in ((source, b"source"), (mix, b"mix"),
                          (master, b"master"), (source_audio, b"audio")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    automated = {
        "passed": True,
        "variants": {"source_only": str(source), "bgm_mix": str(mix)},
        "content_master": str(master),
        "content_master_sha256": digest(master),
        "source_audio_sha256": digest(source_audio),
        "story_plan_sha256": "story",
        "retrieval_sha256": "retrieval",
        "content_frames_framemd5": "frames",
    }
    (tmp_path / "automated_acceptance.json").write_text(
        json.dumps(automated), encoding="utf-8")
    (variants_dir / "audio_variants_manifest.json").write_text(json.dumps({
        "video_identical": True,
        "source_audio": str(source_audio),
        "source_audio_sha256": digest(source_audio),
        "source_only": {"sha256": digest(source)},
        "bgm_mix": {"sha256": digest(mix)},
    }), encoding="utf-8")
    human = {
        "approved": True,
        "story_in_one_sentence": "两人讨论人和妖不能简单判断好坏",
        "source_only_speech_clear": True,
        "bgm_mix_speech_clear": True,
        "bgm_mix_music_present": True,
    }
    final = finalize_roughcut_delivery(tmp_path, human)
    assert final.read_bytes() == b"mix"
    assert json.loads((tmp_path / "acceptance.json").read_text(encoding="utf-8"))["passed"] is True


def test_blocked_gate_removes_stale_formal_delivery(tmp_path):
    formal = tmp_path / "rendered.mp4"
    formal.write_bytes(b"previously-passed")
    _failure(tmp_path, "content", ["human_acceptance_rejected"])
    assert not formal.exists()
    acceptance = json.loads((tmp_path / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["delivery"] == "blocked"
