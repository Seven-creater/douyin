# -*- coding: utf-8 -*-
"""H3-LT0：S2 Long-Take 实验模块的回归锚定（v2 矩阵 / 三分离评估 / 静音参考）。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video import generation_long_take as lt0
from src.agentic_video.generation_long_take import (
    CANONICAL_REFERENCE_CLAUSE, IDENTITY_CHECK_PROMPT, LT0Blocked,
    LONG_TAKE_OBSERVE_PROMPT, RECIPES, VIDEO_REFERENCE_CLAUSE,
    build_long_take_request, choose_canonical_pack, compare_take_to_brief,
    compile_long_take_brief, resolve_reference_aspect_ratio)


def _p04e_fixtures() -> dict:
    shots = [
        {"shot_id": f"section_02.shot_{index:03d}",
         "interval": [5.1 + index, 5.1 + index + 1],
         "information_added": text, "edit_function": "stage",
         "event_relation": "same_event", "supporting_deterministic_ids": []}
        for index, text in enumerate([
            "双方行礼并进入比赛状态", "主角连续高踢攻击对手",
            "主角继续腿部进攻", "对手被击倒在地",
            "主角转身微笑", "主角走向镜头", "主角面对镜头微笑",
            "主角赛后近景放松"])]
    content = {"sections": [{
        "section_id": "section_02", "interval": [5.1, 16.7],
        "reference_specific_fact": "一场跆拳道对抗",
        "transferable_structure": "完整对抗事件反驳成见",
        "audience_takeaway": "主角竞技能力远超预期",
        "continuity": {key: "required" for key in (
            "subject", "opponent", "scene", "event", "actor_role",
            "spatial_orientation")}}]}
    edit = {"editorial_patterns": [{
        "section_id": "section_02", "composition_mode": "micro_montage",
        "semantic_phases": ["initiation", "action", "resolution", "reflection"],
        "snippet_count_range": [5, 8]}]}
    requirement = {"requirements": [{
        "section_id": "section_02",
        "semantic_requirement": {"meaning_to_prove": "对抗能力",
                                 "required_semantic_phases": [
                                     "initiation", "action", "resolution",
                                     "reflection"]},
        "presentation_requirement": {"target_duration_s": 11.6,
                                     "composition_mode": "micro_montage",
                                     "snippet_count_range": [5, 8]},
        "continuity_requirement": {"levels": {
            key: "required" for key in (
                "subject", "opponent", "scene", "event", "actor_role",
                "spatial_orientation")}}}]}
    observations = {"sections": [{
        "section_id": "section_02", "source_interval": [5.1, 16.7],
        "shots": shots}]}
    return {"content_program": content, "edit_program": edit,
            "requirement": requirement, "section_observations": observations}


def _brief() -> dict:
    fixtures = _p04e_fixtures()
    return compile_long_take_brief(
        fixtures["content_program"], fixtures["edit_program"],
        fixtures["requirement"], fixtures["section_observations"])


def _pack() -> dict:
    return {"schema_version": "x", "entries": [
        {"role": "c0_identity", "path": "/a.jpg", "sha256": "a", "uri": "file:///a.jpg"},
        {"role": "c0_fullbody", "path": "/b.jpg", "sha256": "b", "uri": "file:///b.jpg"},
        {"role": "c1_opponent", "path": "/c.jpg", "sha256": "c", "uri": "file:///c.jpg"},
        {"role": "arena", "path": "/d.jpg", "sha256": "d", "uri": "file:///d.jpg"},
    ]}


def test_brief_compiles_from_frozen_p0_with_phase_evidence() -> None:
    brief = _brief()
    assert [row["phase"] for row in brief["phases"]] == [
        "initiation", "action", "resolution", "reflection"]
    evidence = brief["phases"][1]["evidence_text"]
    assert any("高踢" in row for row in evidence)
    # 六维 required 连续性进入结构化 brief
    assert set(brief["required_continuity"].values()) == {"required"}
    assert brief["editability_target"]["target_duration_s"] == 11.6
    # base prompt 只描述事件与约束，不引用任何 Picture/Video 标签（A 无参考）
    assert "<Picture" not in brief["base_event_prompt"]
    assert "<Video" not in brief["base_event_prompt"]


@pytest.mark.parametrize("recipe,ref_count,has_video", [
    ("prompt_only", 0, False),
    ("canonical_only", 4, False),
    ("video_only", 1, True),
    ("canonical_plus_video", 5, True),
])
def test_recipes_share_base_prompt_and_differ_only_in_clauses(
        recipe: str, ref_count: int, has_video: bool) -> None:
    brief = _brief()
    visual_path = str(Path("/ref.mp4").resolve())  # file:// URI 须绝对路径
    request = build_long_take_request(
        brief, pack=_pack(),
        visual_reference={"path": visual_path}, recipe=recipe, seed=1001,
        seconds=12, aspect_ratio="9:16")
    assert request["seconds"] == 12
    assert request["target"]["aspect_ratio"] == "9:16"
    conditions = request["conditions"]
    assert len(conditions) == ref_count
    assert all(row["role"] == "reference" for row in conditions)
    if has_video:
        assert conditions[-1]["type"] == "video"
    assert all(row["type"] == "image" for row in conditions[:max(0, ref_count - (1 if has_video else 0))])
    prompt = request["prompt"]
    base = brief["base_event_prompt"]
    assert prompt.startswith(base)  # 同一 base，不重写
    if recipe == "prompt_only":
        assert prompt == base
    if "canonical" in recipe:
        assert prompt.endswith(CANONICAL_REFERENCE_CLAUSE.strip()) or \
            CANONICAL_REFERENCE_CLAUSE.strip() in prompt
        assert "<Picture 1>" in prompt and "<Picture 4>" in prompt
    if has_video:
        assert VIDEO_REFERENCE_CLAUSE.strip() in prompt
        assert "<Video 1>" in prompt
    assert request["_lt0"]["recipe"] == recipe


def test_video_only_without_visual_reference_blocks() -> None:
    with pytest.raises(LT0Blocked) as excinfo:
        build_long_take_request(
            _brief(), pack=_pack(), visual_reference=None,
            recipe="video_only", seed=1, seconds=12, aspect_ratio="16:9")
    assert excinfo.value.reason_code == "visual_reference_missing"


@pytest.mark.parametrize("width,height,expected", [
    (1080, 1920, "9:16"),
    (720, 960, "3:4"),
    (1080, 1080, "1:1"),
    (1440, 1080, "4:3"),
    (1920, 1080, "16:9"),
    (2560, 1080, "21:9"),
])
def test_aspect_resolution_maps_reference_not_hardcoded(
        width: int, height: int, expected: str) -> None:
    assert resolve_reference_aspect_ratio(width, height) == expected


def test_silent_visual_reference_strips_audio(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(lt0.common, "run_ffmpeg",
                        lambda ffmpeg_bin, args, **kwargs: calls.append(args))
    monkeypatch.setattr(lt0, "_file_hash", lambda path: "hash")
    source = tmp_path / "section_02.mp4"
    source.write_bytes(b"clip")
    row = lt0.prepare_visual_reference(source, tmp_path)
    assert row["sha256"] == "hash"
    flat = [str(item) for item in calls[0]]
    assert "-an" in flat and "-c:v" in flat and "copy" in flat


def _observation(*, artifacts=None, motion="natural", missing_phase=None,
                 moments=None, subject=True, order=True):
    phases = [row["phase"] for row in _brief()["phases"]]
    timeline = [{"phase": phase, "interval": [index * 2.0, index * 2.0 + 2.0],
                 "supported": phase != missing_phase}
                for index, phase in enumerate(phases)]
    return {"observation": {
        "subject_consistency": subject, "opponent_consistency": True,
        "scene_consistency": True, "switch_points": [],
        "motion_quality": motion, "severe_artifacts": artifacts or [],
        "event_timeline": timeline, "phase_order_valid": order,
        "usable_moments": moments if moments is not None else [
            {"start": 0.5, "end": 2.3, "type": "exchange"},
            {"start": 3.0, "end": 5.0, "type": "decisive_action"},
            {"start": 6.0, "end": 7.5, "type": "result"}]},
        "identity": {"frames": [
            {"sample_index": index, "c0_visible": "clear", "c0_match": True}
            for index in range(1, 6)], "c0_consistent": True}}


def test_compare_separates_quality_coverage_editability() -> None:
    result = compare_take_to_brief(_observation(), _brief())
    assert result["quality"]["take_quality_pass"] is True
    assert result["coverage"]["full_event_coverage_pass"] is True
    assert result["editability"]["usable_for_editing"] is True


def test_missing_phase_fails_coverage_but_not_quality() -> None:
    result = compare_take_to_brief(
        _observation(missing_phase="reflection"), _brief())
    assert result["coverage"]["full_event_coverage_pass"] is False
    assert result["coverage"]["missing_phases"] == ["reflection"]
    # 质量好但缺 phase 的 Take 仍可进素材库（partial success 不丢）
    assert result["quality"]["take_quality_pass"] is True
    assert result["editability"]["usable_for_editing"] is True


def test_artifacts_fail_quality_but_keep_coverage() -> None:
    result = compare_take_to_brief(
        _observation(artifacts=["对手面部畸形"]), _brief())
    assert result["quality"]["take_quality_pass"] is False
    assert result["coverage"]["full_event_coverage_pass"] is True


def test_thin_moments_fail_editability_only() -> None:
    result = compare_take_to_brief(
        _observation(moments=[{"start": 1.0, "end": 2.0, "type": "a"}]),
        _brief())
    assert result["editability"]["usable_for_editing"] is False
    assert result["quality"]["take_quality_pass"] is True
    assert result["coverage"]["full_event_coverage_pass"] is True


def test_identity_insufficient_visibility_is_not_failure() -> None:
    payload = _observation()
    payload["identity"]["frames"] = [
        {"sample_index": index, "c0_visible": visibility, "c0_match": None}
        for index, visibility in enumerate(
            ["blurred", "back_turned", "clear", "clear", "absent"], 1)]
    payload["identity"]["frames"][2]["c0_match"] = True
    result = compare_take_to_brief(payload, _brief())
    # 可见性不足不判身份失败（高速运动误杀防护）
    assert result["quality"]["cross_take_identity"] is True


def test_phase_order_broken_fails_coverage() -> None:
    result = compare_take_to_brief(_observation(order=False), _brief())
    assert result["coverage"]["full_event_coverage_pass"] is False
    assert result["quality"]["take_quality_pass"] is True


class FakeRunner:
    def __init__(self, observation: dict, identity: dict) -> None:
        self.observation = observation
        self.identity = identity
        self.watch_calls = []
        self.inspect_calls = []

    def watch(self, video, prompt, **kwargs):
        self.watch_calls.append((video, prompt, kwargs))
        return SimpleNamespace(text=json.dumps(self.observation, ensure_ascii=False))

    def inspect_media(self, images, prompt, **kwargs):
        self.inspect_calls.append((images, prompt, kwargs))
        return SimpleNamespace(text=json.dumps(self.identity, ensure_ascii=False))


def test_observe_long_take_blind_watch_then_anchored_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lt0, "probe_media_geometry",
                        lambda ffprobe, video: {"width": 1080, "height": 1920,
                                                "duration_s": 12.0})
    created = []
    monkeypatch.setattr(
        lt0, "_extract_frame",
        lambda ffmpeg_bin, reference, timestamp, jpg:
        (created.append((timestamp, jpg)), jpg.write_bytes(b"f")) and None)
    observation = _observation()["observation"]
    identity = {"frames": [
        {"sample_index": index, "c0_visible": "clear", "c0_match": True}
        for index in range(1, 6)], "c0_consistent": True}
    runner = FakeRunner(observation, identity)
    video = tmp_path / "take.mp4"
    video.write_bytes(b"v")
    result = lt0.observe_long_take(
        video, tmp_path, runner=runner, take_id="d_s1001", pack=_pack())
    # 盲观察不带任何参考
    assert runner.watch_calls[0][1] is LONG_TAKE_OBSERVE_PROMPT
    assert runner.watch_calls[0][2]["stop_after_json_object"] is True
    # 身份检查对 canonical C0 两图 + 5 采样帧，允许 visibility 不足
    images = runner.inspect_calls[0][0]
    assert len(images) == 7  # 2 canonical + 5 samples
    assert runner.inspect_calls[0][1] is IDENTITY_CHECK_PROMPT
    assert len(created) == 5
    assert result["duration_s"] == 12.0
    assert result["identity"]["c0_consistent"] is True


def test_choose_canonical_pack_requires_pick_within_candidates(
        tmp_path: Path) -> None:
    candidates = {"roles": [{
        "role": "c0_identity",
        "candidates": [
            {"time_s": 14.28, "path": str(tmp_path / "a.jpg"), "sha256": "a"},
            {"time_s": 15.1, "path": str(tmp_path / "b.jpg"), "sha256": "b"}]}]}
    pack = choose_canonical_pack(candidates, tmp_path, picks={"c0_identity": 15.1})
    assert pack["entries"][0]["chosen_time_s"] == 15.1
    assert pack["entries"][0]["uri"].startswith("file://")
    with pytest.raises(LT0Blocked) as excinfo:
        choose_canonical_pack(candidates, tmp_path,
                              picks={"c0_identity": 13.0})
    assert excinfo.value.reason_code == "canonical_pick_out_of_candidates"


def test_plan_only_writes_requests_without_http(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """plan-only 阶段零 HTTP：只产 contact sheet 候选、pack、静音参考与 8 请求。"""
    p04e = _p04e_fixtures()
    p04e_dir = tmp_path / "p04e"
    p04e_dir.mkdir()
    for name, value in p04e.items():
        mapping = {"content_program": "reference_content_program.json",
                   "edit_program": "reference_edit_program.json",
                   "requirement": "material_requirements.json",
                   "section_observations": "section_observations.json"}
        (p04e_dir / mapping[name]).write_text(
            json.dumps(value), encoding="utf-8")
    assets = p04e_dir / "review_assets"
    assets.mkdir()
    (assets / "section_02.mp4").write_bytes(b"clip")
    monkeypatch.setattr(lt0, "probe_media_geometry",
                        lambda ffprobe, video: {"width": 1080, "height": 1920,
                                                "duration_s": 11.6})
    monkeypatch.setattr(
        lt0, "_extract_frame",
        lambda ffmpeg_bin, reference, timestamp, jpg: jpg.write_bytes(b"f"))
    monkeypatch.setattr(lt0, "_file_hash", lambda path: "hash")
    monkeypatch.setattr(lt0.common, "run_ffmpeg", lambda *args, **kwargs: None)
    import src.agentic_video.generation_long_take as module

    summary = module.run_lt0_experiment(
        _NSConfig(), p04e_dir, tmp_path / "out",
        reference=tmp_path / "video.mp4", plan_only=True, execute=False)
    assert summary["phase"] == "planned"
    assert summary["take_count"] == 8
    assert summary["aspect_ratio"] == "9:16"
    plan = json.loads((tmp_path / "out" / "lt0_plan.json").read_text(
        encoding="utf-8"))
    assert plan["execute_authorized"] is False
    assert plan["recipes"] == list(RECIPES)
    take_ids = sorted(row.name for row in (tmp_path / "out" / "takes").iterdir())
    assert len(take_ids) == 8
    for take_dir in (tmp_path / "out" / "takes").iterdir():
        request = json.loads((take_dir / "request.json").read_text(
            encoding="utf-8"))
        assert 4 <= request["seconds"] <= 15
        assert (take_dir / "prompt.txt").is_file()


class _NSConfig:
    perception = {"omni": {}}
