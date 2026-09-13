import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.editorial_planner import (
    build_editorial_candidates, edit_plan_execution_inputs, evaluate_edit_plan,
    plan_edit_variants, select_edit_plan)
from src.agentic_video.renderer import render_editorial_preview
from src.agentic_video.roughcut import run_roughcut


def _story(video: Path) -> dict:
    return {
        "theme": "人和妖不能简单按类别判断好坏",
        "library": "luoxiaohei1", "reference_id": "roughcut_w15",
        "source_video": str(video),
        "source_scope": {"start_s": 0.0, "end_s": 40.0},
        "duration_policy": {"preferred_s": 22.0, "min_s": 12.0, "max_s": 26.4},
        "slots": [{
            "required": True, "status": "supported",
            "source": {"video": str(video), "required_evidence_interval": [12.0, 16.0]},
        }],
    }


def _rows():
    return [{"dialogue": [
        {"utterance_id": "u1", "utterance_interval": [1.0, 8.0],
         "original": "人类本来就很可恶。"},
        {"utterance_id": "u2", "utterance_interval": [10.0, 18.0],
         "original": "人和妖一样，很难定义好坏。"},
        {"utterance_id": "u3", "utterance_interval": [25.0, 30.0],
         "original": "为什么一定要跟人类相处呢？"},
    ]}]


class _Runner:
    def watch(self, _video, prompt, **_kwargs):
        if "Editorial Planner" in prompt:
            payload = {"variants": [
                {"plan_id": "viewpoint", "supported": True, "segments": [
                    {"candidate_id": "clip_001", "proposed_function": "setup",
                     "cut_in_reason": "new_information",
                     "cut_out_reason": "next_function_begins", "reason_detail": "给出反方语境"},
                    {"candidate_id": "clip_002", "proposed_function": "core_statement",
                     "cut_in_reason": "core_statement_starts",
                     "cut_out_reason": "answer_complete", "reason_detail": "完整回答"},
                ]},
                {"plan_id": "question_answer", "supported": False,
                 "unsupported_reason": "没有位于核心前的真实问题", "segments": []},
                {"plan_id": "core_close", "supported": True, "segments": [
                    {"candidate_id": "clip_002", "proposed_function": "core_statement",
                     "cut_in_reason": "core_statement_starts",
                     "cut_out_reason": "answer_complete", "reason_detail": "直接给观点"},
                    {"candidate_id": "clip_004", "proposed_function": "ending",
                     "cut_in_reason": "visual_subject_change",
                     "cut_out_reason": "reaction_complete", "reason_detail": "用人物反应收束"},
                ]},
            ]}
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))
        return SimpleNamespace(text=json.dumps({
            "observed_dialogue": "可听对白", "observed_action": "两人对话",
            "speaker": "人物A", "addressee": "人物B",
            "observation_confidence": 0.9,
        }, ensure_ascii=False))


def _build(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    story = _story(video)
    shots = [
        {"shot_idx": 1, "start_s": 20.0, "end_s": 23.0},
        {"shot_idx": 2, "start_s": 24.0, "end_s": 27.0},
    ]
    settings = {"candidate_min": 2, "candidate_max": 5,
                "min_segments": 2, "min_source_gap_s": 0.25}
    candidates = build_editorial_candidates(
        story, _rows(), shots, runner=_Runner(), output_dir=tmp_path,
        editorial=settings)
    return story, settings, candidates


def test_candidates_are_observations_not_self_assigned_functions(tmp_path):
    story, _settings, candidates = _build(tmp_path)
    assert 2 <= len(candidates) <= 5
    core = next(row for row in candidates if row["required_evidence_ids"])
    assert core["source_interval"] == [10.0, 18.0]
    assert core["utterance_intervals"] == [[10.0, 18.0]]
    assert core["observed_action"] == "两人对话"
    assert all("proposed_function" not in row and "roles" not in row
               for row in candidates)
    assert story["slots"][0]["source"]["required_evidence_interval"] == [12.0, 16.0]


def test_candidate_preserves_container_and_usable_intervals(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    story = _story(video)
    story["source_scope"] = {"start_s": 5.0, "end_s": 35.0}
    story["slots"][0]["source"]["required_evidence_interval"] = [12.0, 16.0]
    rows = [{"start_s": 0.0, "end_s": 40.0, "dialogue": [
        {"utterance_id": "core", "utterance_interval": [12.0, 16.0],
         "original": "核心观点。"},
    ]}]
    candidates = build_editorial_candidates(
        story, rows, [{"shot_idx": 1, "start_s": 2.0, "end_s": 8.0}],
        runner=_Runner(), output_dir=tmp_path,
        editorial={"candidate_min": 2, "candidate_max": 5})
    visual = next(row for row in candidates if row["kind"] == "visual")
    dialogue = next(row for row in candidates if row["kind"] == "dialogue")
    assert dialogue["container_interval"] == [0.0, 40.0]
    assert dialogue["source_interval"] == [12.0, 16.0]
    assert visual["container_interval"] == [2.0, 8.0]
    assert visual["source_interval"] == [5.0, 8.0]


def test_two_candidates_are_allowed_but_one_is_blocked(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    rows = [{"dialogue": _rows()[0]["dialogue"][:2]}]
    candidates = build_editorial_candidates(
        _story(video), rows, [], runner=_Runner(), output_dir=tmp_path,
        editorial={"candidate_min": 2, "candidate_max": 5})
    assert len(candidates) == 2
    with pytest.raises(RuntimeError, match="insufficient_editorial_candidates"):
        build_editorial_candidates(
            _story(video), [{"dialogue": rows[0]["dialogue"][1:]}], [],
            runner=_Runner(), output_dir=tmp_path,
            editorial={"candidate_min": 2, "candidate_max": 5})


def test_boundary_dialogue_fragment_and_over_budget_shot_are_not_candidates(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    rows = [{"dialogue": [
        {"utterance_id": "partial", "utterance_interval": [-2.0, 1.0],
         "original": "scope 边界上被截断的半句"},
        {"utterance_id": "core", "utterance_interval": [10.0, 18.0],
         "original": "人和妖一样，很难定义好坏。"},
        {"utterance_id": "close", "utterance_interval": [25.0, 30.0],
         "original": "后续完整回应。"},
    ]}]
    candidates = build_editorial_candidates(
        _story(video), rows,
        [{"shot_idx": 99, "start_s": 0.0, "end_s": 40.0}],
        runner=_Runner(), output_dir=tmp_path,
        editorial={"candidate_min": 2, "candidate_max": 5})
    assert [row["source_interval"] for row in candidates] == [
        [10.0, 18.0], [25.0, 30.0]]


def test_three_fixed_plans_are_materialized_from_candidate_ids(tmp_path):
    story, settings, candidates = _build(tmp_path)
    plans = plan_edit_variants(story, candidates, runner=_Runner(), output_dir=tmp_path)
    assert [row["plan_id"] for row in plans["variants"]] == [
        "viewpoint", "question_answer", "core_close"]
    assert plans["variants"][1]["supported"] is False
    check = evaluate_edit_plan(plans["variants"][0], story, candidates,
                               editorial=settings)
    assert check["passed"] is True
    assert check["metrics"]["editorial_transform_present"] is True
    assert check["metrics"]["omitted_source_duration_s"] == pytest.approx(2.0)


def test_cut_taxonomy_and_candidate_interval_are_hard_constraints(tmp_path):
    story, settings, candidates = _build(tmp_path)
    plan = plan_edit_variants(story, candidates, runner=_Runner(),
                              output_dir=tmp_path)["variants"][0]
    plan["segments"][0]["cut_in_reason"] = "为了节奏"
    plan["segments"][0]["proposed_function"] = "question"
    plan["segments"][1]["source_interval"][0] += 0.5
    errors = evaluate_edit_plan(plan, story, candidates, editorial=settings)["errors"]
    assert "segment[0]:invalid_cut_in_reason" in errors
    assert "segment[1]:candidate_interval_changed" in errors
    assert "viewpoint_functions_invalid" in errors


def test_required_dialogue_cannot_have_an_internal_source_gap(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    story = _story(video)
    story["duration_policy"] = {"min_s": 1.0, "max_s": 26.4}
    candidates = [
        {"candidate_id": "left", "video": str(video),
         "source_interval": [10.0, 14.0], "utterance_intervals": [[10.0, 20.0]],
         "required_evidence_ids": ["required_00"]},
        {"candidate_id": "right", "video": str(video),
         "source_interval": [16.0, 20.0], "utterance_intervals": [[10.0, 20.0]],
         "required_evidence_ids": []},
    ]
    plan = {"plan_id": "viewpoint", "supported": True, "duration_s": 8.0,
            "segments": [
                {"candidate_id": "left", "source_interval": [10.0, 14.0],
                 "output_interval": [0.0, 4.0], "proposed_function": "core_statement",
                 "cut_in_reason": "core_statement_starts",
                 "cut_out_reason": "redundant_content_starts", "reason_detail": "保留观点"},
                {"candidate_id": "right", "source_interval": [16.0, 20.0],
                 "output_interval": [4.0, 8.0], "proposed_function": "ending",
                 "cut_in_reason": "new_information",
                 "cut_out_reason": "semantic_unit_complete", "reason_detail": "收束"},
            ]}
    check = evaluate_edit_plan(
        plan, story, candidates,
        editorial={"min_segments": 2, "min_source_gap_s": 0.25})
    assert "required_dialogue_internal_gap" in check["errors"]


def test_extraction_only_is_recorded_separately_from_content_quality(tmp_path):
    story, settings, candidates = _build(tmp_path)
    core = next(row for row in candidates if row["required_evidence_ids"])
    adjacent = {**core, "candidate_id": "adjacent",
                "source_interval": [18.0, 22.0],
                "required_evidence_ids": [], "utterance_intervals": []}
    candidates = [core, adjacent]
    raw = {"plan_id": "core_close", "supported": True, "duration_s": 12.0,
           "segments": [
               {"candidate_id": core["candidate_id"], "source_interval": [10.0, 18.0],
                "output_interval": [0.0, 8.0], "proposed_function": "core_statement",
                "cut_in_reason": "core_statement_starts", "cut_out_reason": "answer_complete",
                "reason_detail": "观点完整"},
               {"candidate_id": "adjacent", "source_interval": [18.0, 22.0],
                "output_interval": [8.0, 12.0], "proposed_function": "ending",
                "cut_in_reason": "reaction_starts", "cut_out_reason": "reaction_complete",
                "reason_detail": "反应收束"},
           ]}
    check = evaluate_edit_plan(raw, story, candidates, editorial=settings)
    assert check["passed"] is True
    assert check["metrics"]["editorial_transform_present"] is False
    review = {"parsed": True, "core_meaning_preserved": True,
              "opening_reason_clear": True, "functionless_segment_present": False,
              "all_cuts_purposeful": True,
              "scores": {key: 1.0 for key in ("core", "opening", "function", "ending", "continuity")}}
    selected, gate = select_edit_plan([raw], {"core_close": check},
                                      {"core_close": review})
    assert selected is None
    assert gate["rankings"][0]["reasons"] == ["extraction_only"]


def test_scores_only_rank_and_do_not_add_an_arbitrary_threshold(tmp_path):
    story, settings, candidates = _build(tmp_path)
    plan = plan_edit_variants(story, candidates, runner=_Runner(),
                              output_dir=tmp_path)["variants"][0]
    check = evaluate_edit_plan(plan, story, candidates, editorial=settings)
    review = {"parsed": True, "core_meaning_preserved": True,
              "opening_reason_clear": True, "functionless_segment_present": False,
              "all_cuts_purposeful": True,
              "scores": {key: 0.70 for key in ("core", "opening", "function", "ending", "continuity")}}
    selected, gate = select_edit_plan([plan], {"viewpoint": check},
                                      {"viewpoint": review})
    assert selected["plan_id"] == "viewpoint"
    assert gate["rankings"][0]["score"] == pytest.approx(70.0)


def test_preview_is_low_cost_real_multisegment_render(tmp_path, monkeypatch):
    story, _settings, candidates = _build(tmp_path)
    plan = plan_edit_variants(story, candidates, runner=_Runner(),
                              output_dir=tmp_path)["variants"][0]
    asset, retrieval = edit_plan_execution_inputs(plan, story, candidates,
                                                  audio_policy={"audio_mode": "source"})
    calls = []

    def fake_ffmpeg(_binary, args, **_kwargs):
        calls.append(args)
        Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(args[-1]).write_bytes(b"preview")

    monkeypatch.setattr("src.agentic_video.renderer._has_audio_stream",
                        lambda *_args: True)
    monkeypatch.setattr("src.agentic_video.renderer.common.run_ffmpeg", fake_ffmpeg)
    preview = render_editorial_preview(
        SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"},
                        library={"sources": {}}),
        asset, retrieval, tmp_path / "preview")
    assert preview.is_file()
    assert len(calls) == len(plan["segments"]) + 1
    assert all("scale='min(640,iw)':-2,fps=12" in call for call in calls[:-1])
    concat = (tmp_path / "preview" / "work" / "concat.txt").read_text(encoding="utf-8")
    assert concat.count("segment_") == len(plan["segments"])


def test_fake_runner_synthetic_media_full_v3_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    bgm = tmp_path / "bgm.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=320x180:r=24:d=40", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=40", "-shortest", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=220:duration=20", "-c:a", "aac", str(bgm),
    ], check=True)
    library = tmp_path / "library"
    transcript = library / "sources" / "luoxiaohei1" / "narrative_transcript.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"segments": [
        {"start_ms": 1000, "end_ms": 8000, "text": "人类本来就很可恶。"},
        {"start_ms": 10000, "end_ms": 18000,
         "text": "人和妖一样，很难定义好坏。"},
        {"start_ms": 25000, "end_ms": 30000,
         "text": "为什么一定要跟人类相处呢？"},
    ]}, ensure_ascii=False), encoding="utf-8")
    ingestion = library / "shots" / "luoxiaohei1__narrative" / "result.json"
    ingestion.parent.mkdir(parents=True)
    ingestion.write_text(json.dumps({
        "params": {"video_sha256": "a" * 64},
        "output": {"shots": [{"shot_idx": 1, "start_s": 0, "end_s": 40}]},
    }), encoding="utf-8")
    rows = [{"video": str(source), "video_stem": "luoxiaohei1__narrative",
             "start_s": 0.0, "end_s": 40.0, "event_id": "w15",
             "caption": "两人对话", "dialogue": []}]
    monkeypatch.setattr("src.library.build_index.load_index",
                        lambda _cfg: (rows, None))
    monkeypatch.setattr("src.perception.detect_shots.detect_scoped_shots",
                        lambda *_args, **_kwargs: {
                            "shot_count": 1, "scope_interval": [0.0, 40.0],
                            "timebase": "absolute", "timebase_origin_s": 0.0,
                            "relative_boundaries_s": [0.0, 20.0, 23.0, 40.0],
                            "boundaries_s": [0.0, 20.0, 23.0, 40.0],
                            "threshold": 0.3, "min_shot_len_s": 0.4,
                            "shots": [{"index": 1, "start_s": 20.0,
                                       "end_s": 23.0, "duration_s": 3.0,
                                       "relative_interval": [20.0, 23.0]}],
                        })
    spec = {
        "spec_version": "roughcut_v3", "source": "luoxiaohei1", "base_window": 15,
        "source_scope": {"start_s": 0.0, "end_s": 40.0},
        "focus_utterance": {"candidate_interval": [10.0, 18.0],
                            "query": "人和妖不能简单按类别判断好坏"},
        "stages": [
            {"id": "setup", "required": False, "purpose": "建立语境"},
            {"id": "core_statement", "required": True, "purpose": "核心判断",
             "evidence_type": "dialogue"},
            {"id": "response", "required": False, "purpose": "回应"},
        ],
        "duration": {"preferred_s": 22.0, "min_s": 12.0, "max_s": 26.4},
        "editorial": {"candidate_min": 2, "candidate_max": 5,
                      "variant_ids": ["viewpoint", "question_answer", "core_close"],
                      "av_sync": "locked", "source_order": "chronological",
                      "min_segments": 2, "min_source_gap_s": 0.25},
        "audio_variants": ["source_only", "bgm_mix"],
    }

    class PipelineRunner(_Runner):
        def watch(self, video, prompt, **kwargs):
            if "素材证据定位员" in prompt:
                return SimpleNamespace(text=json.dumps({
                    "found": True, "start": 12, "end": 16,
                    "timebase": "absolute", "evidence": "核心观点", "confidence": 1.0}))
            if "素材证据验证员" in prompt:
                return SimpleNamespace(text=json.dumps({
                    "verdict": "pass", "conditions": [{"condition": "核心判断",
                    "met": True, "evidence_interval": [1.5, 5.5]}], "missing": [],
                    "failure_reason": "", "needs_context": False,
                    "what_is_visible": "完整表达观点"}, ensure_ascii=False))
            if "Editorial Planner" in prompt:
                return super().watch(video, prompt, **kwargs)
            if "第一次看到这条短视频" in prompt:
                return SimpleNamespace(text=json.dumps({
                    "core_statement": "人和妖的好坏都不是绝对的",
                    "first_three_seconds_summary": "先给出争议观点",
                    "core_meaning_preserved": True, "opening_reason_clear": True,
                    "functionless_segment_present": False, "all_cuts_purposeful": True,
                    "ending_intentional": True, "transition_coherent": True,
                    "scores": {key: 0.7 for key in
                               ("core", "opening", "function", "ending", "continuity")},
                }, ensure_ascii=False))
            if "素材侧粗剪验收" in prompt:
                return SimpleNamespace(text=json.dumps({
                    "main_character": "人物A", "consistent_protagonist": True,
                    "switch_points": [], "story_in_one_sentence": "人和妖不能简单判断好坏",
                    "event_relations": "连续", "speech_clear": True,
                    "music_present": True, "speaker_description": "人物A",
                    "addressee_description": "人物B", "speaker_addressee_stable": True,
                    "core_statement": "人和妖的好坏都不是绝对的",
                    "first_three_seconds_summary": "先给出争议观点",
                    "opening_reason_clear": True, "functionless_span_present": False,
                    "transitions_have_clear_function": True, "ending_intentional": True,
                }, ensure_ascii=False))
            return super().watch(video, prompt, **kwargs)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(library_dir=library),
        perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"},
        generation={"assemble": {"width": 320, "height": 180}},
        library={"narrative_render": {"width": 320, "height": 180,
                                       "bgm_path": str(bgm)}, "sources": {}},
    )
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="human acceptance pending"):
        run_roughcut(cfg, output_dir=output, spec=spec, runner=PipelineRunner())
    gate = json.loads((output / "editorial_gate.json").read_text(encoding="utf-8"))
    assert gate["selected_plan_id"] == "core_close"
    scoped_shots = json.loads((output / "editorial_shots.json").read_text(
        encoding="utf-8"))
    assert [scoped_shots["shots"][0]["start_s"],
            scoped_shots["shots"][0]["end_s"]] == [20.0, 23.0]
    assert (output / "editorial_previews" / "core_close" / "preview.mp4").is_file()
    assert (output / "content_master.mp4").is_file()
    assert (output / "variants" / "source_only" / "rendered.mp4").is_file()
    assert (output / "variants" / "bgm_mix" / "rendered.mp4").is_file()
    assert not (output / "rendered.mp4").exists()
    concat = (output / "content_master_render" / "work" / "concat.txt").read_text(
        encoding="utf-8")
    assert concat.count("segment_") == 2
    acceptance = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["failure_class"] == "verification"
    assert acceptance["reasons"] == ["human_acceptance_pending"]
