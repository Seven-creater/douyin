import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.evidence_planner import build_evidence_edit_plan
from src.agentic_video.evidence_units import EvidenceUnit
from src.agentic_video.reference_pattern import extract_reference_pattern
from src.perception.evidence_miner import (mine_evidence, parse_evidence_response,
                                           split_analysis_intervals)
from src.perception.omni_pool import parse_gpu_pairs


def test_evidence_unit_rejects_invalid_interval_and_accepts_micro_clip():
    with pytest.raises(ValueError, match="positive duration"):
        EvidenceUnit("bad", (1.0, 1.0), "micro_clip", "x")
    unit = EvidenceUnit("ok", (10.0, 10.5), "micro_clip", "起身",
                        editing_role="reversal", editorial_value=.9, confidence=.8)
    assert unit.duration_s == pytest.approx(.5)
    assert unit.to_dict()["interval"] == [10.0, 10.5]


def test_relative_evidence_interval_is_normalized_to_absolute_and_audited():
    units, audit = parse_evidence_response(json.dumps({
        "timebase": "relative",
        "evidence_units": [
            {"id": "ev", "interval": [20, 30], "unit_type": "micro_clip",
             "description": "瞬间", "editing_role": "proof", "confidence": .9},
            {"id": "outside", "interval": [39, 41], "unit_type": "micro_clip"},
        ]}), input_clip_interval=(2965, 3005), scope_interval=(2965, 3005),
        source_shot_id="shot_1")
    assert units[0].interval == (2985.0, 2995.0)
    assert units[0].metadata["timebase_origin_s"] == 2965.0
    assert audit["accepted"] == 1
    assert audit["rejected"][0]["reason"] == "interval_outside_scope"


def test_keyframe_point_gets_minimum_render_interval():
    units, _ = parse_evidence_response(json.dumps({
        "timebase": "relative",
        "evidence_units": [{"id": "frame", "frame_time": 3.0,
                             "unit_type": "keyframe_hold", "role": "hook"}]}),
        input_clip_interval=(100, 110), scope_interval=(100, 110))
    assert units[0].interval == (103.0, 103.15)
    assert units[0].unit_type == "keyframe_hold"


def test_long_continuous_shot_is_split_into_eight_omni_calls():
    intervals = split_analysis_intervals(2965, 3005, max_watch_s=6, overlap_s=.5)
    assert len(intervals) == 8
    assert intervals[0] == (2965.0, 2971.0)
    assert intervals[-1] == (3003.5, 3005.0)
    assert all(end - start <= 6.0 for start, end in intervals)


def test_eight_cards_are_four_disjoint_two_card_omni_workers():
    assert parse_gpu_pairs("0,1;2,3;4,5;6,7") == ["0,1", "2,3", "4,5", "6,7"]
    with pytest.raises(ValueError, match="disjoint"):
        parse_gpu_pairs("0,1;1,2")


def test_evidence_miner_calls_once_per_scoped_shot_and_writes_raw(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")

    class Runner:
        _last_sampling = {"actual_sampled_frames": 8}

        def __init__(self):
            self.calls = []

        def watch(self, video_path, prompt, **kwargs):
            self.calls.append((video_path, prompt, kwargs))
            return SimpleNamespace(text=json.dumps({
                "timebase": "relative",
                "evidence_units": [{"interval": [1, 2], "type": "micro_clip",
                                     "description": "反击", "role": "reversal",
                                     "editorial_value": .8, "confidence": .9}]}))

    runner = Runner()
    out = tmp_path / "evidence_units.json"
    result = mine_evidence(
        video, {"shots": [{"id": "s1", "start_s": 0, "end_s": 4},
                           {"id": "s2", "start_s": 8, "end_s": 12}]},
        runner=runner, scope_interval=(0, 10), output=out)
    assert len(runner.calls) == 2
    assert len(result["units"]) == 2
    assert (tmp_path / "evidence_units_raw" / "s1.txt").is_file()
    assert result["calls"][0]["sampling"]["actual_sampled_frames"] == 8


def test_evidence_miner_uses_batch_runner_for_long_oner(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")

    class BatchRunner:
        gpu_pairs = ["0,1", "2,3", "4,5", "6,7"]

        def __init__(self):
            self.requests = []

        def watch_many(self, requests):
            self.requests = requests
            return [SimpleNamespace(
                text=json.dumps({"timebase": "relative", "evidence_units": [{
                    "interval": [.2, .5], "unit_type": "micro_clip",
                    "description": f"证据{index}", "editing_role": "proof",
                    "editorial_value": .8, "confidence": .9}]}),
                sampling={"actual_sampled_frames": 12},
                gpu_pair=self.gpu_pairs[index % 4])
                for index, _ in enumerate(requests)]

    runner = BatchRunner()
    result = mine_evidence(
        video, {"shots": [{"id": "oner", "start_s": 2965, "end_s": 3005}]},
        runner=runner, scope_interval=(2965, 3005),
        output=tmp_path / "evidence_units.json")
    assert len(runner.requests) == 8
    assert result["analysis_policy"]["analysis_call_count"] == 8
    assert result["analysis_policy"]["parallel_workers"] == 4
    assert {call["gpu_pair"] for call in result["calls"]} == {
        "0,1", "2,3", "4,5", "6,7"}


def test_evidence_planner_builds_short_montage_without_padding():
    units = [
        EvidenceUnit("hook", (100, 100.5), "micro_clip", "受挫",
                      editing_role="hook", editorial_value=.8, confidence=.9),
        EvidenceUnit("proof", (101, 101.8), "micro_clip", "反击",
                      editing_role="proof", editorial_value=.9, confidence=.9),
        EvidenceUnit("payoff", (102, 102.4), "keyframe_hold", "胜利",
                      editing_role="payoff", editorial_value=.9, confidence=.9),
    ]
    plan = build_evidence_edit_plan(
        units, {"pattern_type": "proof_montage", "beats": [
            {"role": "hook"}, {"role": "proof"}, {"role": "payoff"}],
            "allowed_unit_types": ["micro_clip", "keyframe_hold"]},
        preferred_duration_s=12, min_duration_s=1, max_duration_s=20)
    assert plan["passed"] is True
    assert plan["duration_s"] == pytest.approx(1.7)
    assert [row["editing_role"] for row in plan["segments"]] == ["hook", "proof", "payoff"]


def test_evidence_planner_blocks_when_no_two_units_exist():
    plan = build_evidence_edit_plan([], {"pattern_type": "proof_montage"})
    assert plan["passed"] is False
    assert plan["failure_class"] == "evidence_missing"


def test_reference_pattern_has_deterministic_proof_montage_prior(tmp_path, monkeypatch):
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.agentic_video.reference_pattern.common.video_duration_s", lambda *_: 10.0)
    monkeypatch.setattr(
        "src.agentic_video.reference_pattern.detect_shots",
        lambda *_, **__: {"shot_count": 4, "shots": [
            {"start_s": 0, "end_s": 2.0, "duration_s": 2.0},
            {"start_s": 2, "end_s": 4.0, "duration_s": 2.0},
            {"start_s": 4, "end_s": 6.0, "duration_s": 2.0},
            {"start_s": 6, "end_s": 10.0, "duration_s": 4.0},
        ]})
    pattern = extract_reference_pattern(video)
    assert pattern["pattern_type"] == "proof_montage"
    assert {row["role"] for row in pattern["beats"]} >= {"hook", "proof", "payoff"}


def test_micro_montage_renderer_accepts_keyframe_and_micro_clip(tmp_path, monkeypatch):
    from src.agentic_video.renderer import render_micro_montage

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []

    def fake_ffmpeg(_binary, args, **_kwargs):
        calls.append(args)
        target = Path(args[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"rendered")

    monkeypatch.setattr("src.agentic_video.renderer.common.run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr("src.agentic_video.renderer._has_audio_stream", lambda *_: False)
    cfg = SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"},
                          generation={}, library={})
    plan = {"duration_s": 1.0, "segments": [
        {"unit_id": "a", "unit_type": "micro_clip", "source_video": str(source),
         "source_interval": [0, .5], "duration_s": .5},
        {"unit_id": "b", "unit_type": "keyframe_hold", "source_interval": [1, 1.15],
         "duration_s": .5},
    ]}
    result = render_micro_montage(cfg, plan, tmp_path / "render", source_video=source)
    assert result["content_master"].is_file()
    assert (tmp_path / "render" / "evidence_render_manifest.json").is_file()
    assert any("-loop" in call for call in calls)


def test_evidence_v6_fake_runner_synthetic_media_integration(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from src.agentic_video.evidence_pipeline import run_evidence_pipeline

    source = tmp_path / "source.mp4"
    reference = tmp_path / "reference.mp4"
    bgm = tmp_path / "bgm.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=320x180:rate=24:duration=10", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=10", "-shortest", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)], check=True)
    shutil.copy2(source, reference)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=220:duration=10", "-c:a", "aac", str(bgm)], check=True)

    class FakeRunner:
        gpu_pairs = ["0,1", "2,3", "4,5", "6,7"]

        def watch(self, *_args, **_kwargs):
            return SimpleNamespace(text=json.dumps({
                "pattern_type": "proof_montage",
                "beats": [{"role": "hook"}, {"role": "proof"},
                          {"role": "payoff"}],
                "transition": "hard_cut",
                "allowed_unit_types": ["micro_clip", "keyframe_hold"]}))

        def watch_many(self, requests):
            if requests and "start_s" in (requests[0].get("kwargs") or {}):
                answers = []
                for index, _ in enumerate(requests):
                    rows = ([
                        {"interval": [0.2, 1.7], "unit_type": "micro_clip",
                         "description": "进入困境", "editing_role": "hook",
                         "editorial_value": .9, "confidence": .9},
                        {"interval": [2.0, 3.5], "unit_type": "micro_clip",
                         "description": "展示能力", "editing_role": "proof",
                         "editorial_value": .9, "confidence": .9},
                    ] if index == 0 else [
                        {"interval": [0.5, 2.0], "unit_type": "micro_clip",
                         "description": "得到结果", "editing_role": "payoff",
                         "editorial_value": .9, "confidence": .9},
                    ])
                    answers.append(SimpleNamespace(
                        text=json.dumps({"timebase": "relative",
                                         "evidence_units": rows}),
                        sampling={"actual_sampled_frames": 12},
                        gpu_pair=self.gpu_pairs[index % 4]))
                return answers
            return [SimpleNamespace(text=json.dumps({
                "core_message": "角色从困境中展示能力并得到结果",
                "hook_clear": True, "montage_coherent": True,
                "functionless_span_present": False,
                "visible_evidence_roles": ["困境", "能力", "结果"],
                "audible_dialogue_present": True, "speech_clear": True,
                "music_present": index == 1}), gpu_pair=self.gpu_pairs[index])
                for index, _ in enumerate(requests)]

    cfg = SimpleNamespace(
        perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"},
        generation={"assemble": {"width": 320, "height": 180}},
        library={"narrative_render": {"bgm_path": str(bgm)}},
        paths=SimpleNamespace(library_dir=tmp_path / "library"))
    spec = {
        "spec_version": "evidence_v6", "source": "test",
        "source_scope": {"start_s": 0, "end_s": 10},
        "reference": str(reference), "audio_variants": ["source_only", "bgm_mix"],
        "analysis": {"source_scene_threshold": .3, "reference_scene_threshold": .1,
                     "min_shot_duration_s": .15, "max_watch_s": 6,
                     "watch_overlap_s": .5},
        "evidence": {"preferred_duration_s": 6, "min_duration_s": 4,
                     "max_duration_s": 10}}
    output = tmp_path / "run"
    result = run_evidence_pipeline(
        cfg, spec, output, runner=FakeRunner(), source_video=source, force=True)
    assert result["acceptance"]["automated_passed"] is True
    assert result["acceptance"]["passed"] is False
    assert result["acceptance"]["reasons"] == ["human_acceptance_pending"]
    assert not (output / "rendered.mp4").exists()
    manifest = json.loads(
        (output / "variants" / "audio_variants_manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_identical"] is True
    assert manifest["source_only"]["frame_md5"] == manifest["bgm_mix"]["frame_md5"]
