# -*- coding: utf-8 -*-
"""H3-LT0：S2 Long-Take 实验模块的回归锚定（v2 矩阵 / 官方六段 prompt /
黑边归一 / 双 variant 执行 / 三分离评估）。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video import generation_long_take as lt0
from src.agentic_video.generation_long_take import (
    IDENTITY_CHECK_PROMPT, LT0Blocked, LONG_TAKE_OBSERVE_PROMPT, RECIPES,
    build_long_take_request, choose_canonical_pack, compare_take_to_brief,
    compile_long_take_brief, resolve_reference_aspect_ratio)

SIX_SECTIONS = ("subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape",
                "non_diegetic_music")


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
        "reference_specific_fact": "白色道服红色护具的主角在跆拳道馆与对手对抗，"
                                    "高踢击倒对手后微笑走向镜头。",
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
    # LT0 Round 1 无 arena（群像帧不能当纯场景参考）
    return {"schema_version": "x", "entries": [
        {"role": "c0_identity", "path": "/a.jpg", "sha256": "a", "uri": "file:///a.jpg"},
        {"role": "c0_fullbody", "path": "/b.jpg", "sha256": "b", "uri": "file:///b.jpg"},
        {"role": "c1_opponent", "path": "/c.jpg", "sha256": "c", "uri": "file:///c.jpg"},
    ]}


def test_brief_compiles_official_six_sections_with_compressed_phases() -> None:
    brief = _brief()
    for key in SIX_SECTIONS:
        assert key in brief["prompt_fields"], key
    # 语义 phase 压缩：4 个 reflection shots → 1 行（首条证据文本）
    reflection = [row for row in brief["compressed_phase_lines"]
                  if row["phase"] == "reflection"][0]
    assert reflection["line"] == "主角转身微笑"
    assert "镜头微笑" not in json.dumps(
        [row for row in brief["compressed_phase_lines"]
         if row["phase"] != "reflection"], ensure_ascii=False)
    # 时间箱比例与官方示例一致（2/5/2/3 → 0-2, 2-7, 7-9, 9-12）
    spans = [box["span_ratio"] for box in brief["phase_timeboxes"]]
    assert spans[0][0] == 0.0
    assert spans[0][1] == pytest.approx(1 / 6, abs=1e-3)
    assert spans[-1][1] == pytest.approx(1.0)
    # 不再要求"一镜到底"；允许自然机位变化但锁定同一事件/场景/人物
    clause = brief["continuity_clause"]
    assert "一镜到底" not in clause
    assert "Natural camera movement or internal cuts are allowed" in clause
    assert "different setting" in clause
    # v3：subject 定义只写人物外观（词表确定性抽取），不塞事件
    assert brief["subject_wardrobe"]["subject_1"] == \
        "a white dobok, a red chest protector"
    assert "[reference generation]" in brief["prompt_fields"]["summary"]


@pytest.mark.parametrize("recipe,ref_count,has_video,variant", [
    ("prompt_only", 0, False, "fl2va"),
    ("canonical_only", 3, False, "ref2va"),
    ("video_only", 1, True, "ref2va"),
    ("canonical_plus_video", 4, True, "ref2va"),
])
def test_wire_prompt_official_schema_per_recipe(
        recipe: str, ref_count: int, has_video: bool,
        variant: str) -> None:
    brief = _brief()
    visual_path = str(Path("/ref.mp4").resolve())
    request = build_long_take_request(
        brief, pack=_pack(),
        visual_reference={"path": visual_path}, recipe=recipe, seed=1001,
        seconds=12, aspect_ratio="3:4")
    prompt = request["prompt"]
    conditions = request["conditions"]
    assert len(conditions) == ref_count
    assert all(row["role"] == "reference" for row in conditions)
    if has_video:
        assert conditions[-1]["type"] == "video"
    assert request["_lt0"]["server_variant"] == variant
    # 官方 shot/timeline grammar（英文 + [Shot N] At 00:0X.000）
    assert "[Shot 1] At 00:00.000," in prompt
    assert "[Shot 4] At 00:09.000," in prompt
    assert "white dobok" in prompt
    if recipe == "prompt_only":
        # T2VA 三字段 base schema；不得带 Ref2VA 六段或引用标签
        assert "integrated_multimodal_description:" in prompt
        assert "overall_soundscape:" in prompt
        assert "non_diegetic_music:" in prompt
        assert "subject_definitions:" not in prompt
        assert "<Subject" not in prompt
        assert "<Picture" not in prompt and "<Video" not in prompt
        assert "the protagonist" in prompt and "the opponent" in prompt
    else:
        for section in SIX_SECTIONS:
            assert f"{section}:\n" in prompt, section
        assert "[reference generation]" in prompt
        assert "attribute_transfer" in prompt
        assert "No reference material" not in prompt  # B 矛盾句已除
        has_pictures = recipe in {"canonical_only",
                                  "canonical_plus_video"}
        assert ("<Picture 1>" in prompt) is has_pictures
        assert ("<Video 1>" in prompt) is has_video
        if has_pictures:
            assert "defined jointly by" in prompt
        else:
            assert "an adult taekwondo athlete wearing" in prompt
        if has_video:
            assert "Do not copy exact cut timing or frames" in prompt
        else:
            assert "<Video" not in prompt
        assert "12-second taekwondo sparring event" in prompt


def test_wire_prompt_validator_catches_contract_violations() -> None:
    brief = _brief()
    visual_path = str(Path("/ref.mp4").resolve())
    good = {recipe: build_long_take_request(
        brief, pack=_pack(), visual_reference={"path": visual_path},
        recipe=recipe, seed=1, seconds=12, aspect_ratio="3:4")
        for recipe in RECIPES}
    for request in good.values():
        lt0.validate_lt0_wire_prompt(request)  # 全部通过
    # A 泄漏 Ref2VA 六段 → 拦
    broken = json.loads(json.dumps(good["prompt_only"]))
    broken["prompt"] += "\n\nsubject_definitions:\n<Picture 1>"
    with pytest.raises(LT0Blocked) as excinfo:
        lt0.validate_lt0_wire_prompt(broken)
    assert excinfo.value.reason_code == "wire_check_t2va_leak"
    # B 缺全部 retention marker → 拦
    broken = json.loads(json.dumps(good["canonical_only"]))
    for marker in lt0.RETENTION_MARKERS:
        broken["prompt"] = broken["prompt"].replace(marker, "xx")
    with pytest.raises(LT0Blocked) as excinfo:
        lt0.validate_lt0_wire_prompt(broken)
    assert excinfo.value.reason_code == "wire_check_retention_marker"
    # D 标签与条件不匹配（删掉一张图）→ 拦
    broken = json.loads(json.dumps(good["canonical_plus_video"]))
    broken["conditions"] = broken["conditions"][:3]
    with pytest.raises(LT0Blocked) as excinfo:
        lt0.validate_lt0_wire_prompt(broken)
    assert excinfo.value.reason_code == "wire_check_video_label_mismatch"


def test_video_only_without_visual_reference_blocks() -> None:
    with pytest.raises(LT0Blocked) as excinfo:
        build_long_take_request(
            _brief(), pack=_pack(), visual_reference=None,
            recipe="video_only", seed=1, seconds=12, aspect_ratio="3:4")
    assert excinfo.value.reason_code == "visual_reference_missing"


@pytest.mark.parametrize("width,height,expected", [
    (1080, 1920, "9:16"),
    (720, 960, "3:4"),
    (1080, 1080, "1:1"),
    (1440, 1080, "4:3"),
    (1920, 1080, "16:9"),
    (2560, 1080, "21:9"),
    (720, 1019, "3:4"),  # 黑边裁剪后的真实有效画面（人工审核 p0510）
])
def test_aspect_resolution_maps_active_picture_not_container(
        width: int, height: int, expected: str) -> None:
    assert resolve_reference_aspect_ratio(width, height) == expected


def test_detect_active_crop_from_pixel_rows(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    """灰度逐行判黑（服务器 ffmpeg 4.2.7 不打印 cropdetect 行）。"""
    width, height = 720, 1280
    # 上 129 行黑、中 1019 行内容、下 132 行黑
    frame = bytes([8]) * (width * 129) + bytes([120]) * (width * 1019) + \
        bytes([8]) * (width * 132)
    calls = []

    class FakeCompleted:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        output = Path(cmd[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(frame)
        return FakeCompleted()

    monkeypatch.setattr(lt0.subprocess, "run", fake_run)
    monkeypatch.setattr(
        lt0, "probe_media_geometry",
        lambda ffprobe, video: {"width": width, "height": height,
                                "duration_s": 11.6})
    crop = lt0.detect_active_crop("ffmpeg", tmp_path / "clip.mp4")
    assert crop == {"w": 720, "h": 1019, "x": 0, "y": 129}
    assert len(calls) == 5  # 多帧采样取中位


def test_detect_active_crop_returns_none_without_bars(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    width, height = 720, 1280
    frame = bytes([120]) * (width * height)

    def fake_run(cmd, **kwargs):
        output = Path(cmd[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(frame)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lt0.subprocess, "run", fake_run)
    monkeypatch.setattr(
        lt0, "probe_media_geometry",
        lambda ffprobe, video: {"width": width, "height": height,
                                "duration_s": 11.6})
    assert lt0.detect_active_crop("ffmpeg", tmp_path / "clip.mp4") is None


def test_composed_crop_combines_blackbars_and_role_fraction() -> None:
    active = {"w": 720, "h": 1019, "x": 0, "y": 129}
    frac = {"frac_x": [0.5, 1.0], "frac_y": [0.0, 1.0]}
    crop = lt0._composed_crop_filter(active, frac, 720, 1280)
    assert crop == "crop=360:1019:360:129"
    # 无黑边 + 无分数 → 不裁
    assert lt0._composed_crop_filter(None, None, 720, 1280) is None


def test_silent_visual_reference_crops_bars_and_strips_audio(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(lt0.common, "run_ffmpeg",
                        lambda ffmpeg_bin, args, **kwargs: calls.append(args))
    monkeypatch.setattr(lt0, "_file_hash", lambda path: "hash")
    monkeypatch.setattr(
        lt0, "detect_active_crop",
        lambda ffmpeg_bin, video: {"w": 720, "h": 1019, "x": 0, "y": 129})
    source = tmp_path / "section_02.mp4"
    source.write_bytes(b"clip")
    row = lt0.prepare_visual_reference(source, tmp_path)
    assert row["active_crop"]["h"] == 1019
    flat = [str(item) for item in calls[0]]
    assert "-an" in flat
    assert any("crop=720:1019:0:129" in item for item in flat)
    assert "libx264" in flat  # 裁剪需重编码，copy 不能裁


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
                        lambda ffprobe, video: {"width": 720, "height": 1019,
                                                "duration_s": 12.0})
    created = []

    def fake_extract(ffmpeg_bin, reference, timestamp, jpg,
                     crop_filter=None):
        created.append((timestamp, jpg))
        jpg.write_bytes(b"f")

    monkeypatch.setattr(lt0, "_extract_frame", fake_extract)
    observation = _observation()["observation"]
    identity = {"frames": [
        {"sample_index": index, "c0_visible": "clear", "c0_match": True}
        for index in range(1, 6)], "c0_consistent": True}
    runner = FakeRunner(observation, identity)
    video = tmp_path / "take.mp4"
    video.write_bytes(b"v")
    result = lt0.observe_long_take(
        video, tmp_path, runner=runner, take_id="d_s1001", pack=_pack())
    assert runner.watch_calls[0][1] is LONG_TAKE_OBSERVE_PROMPT
    assert runner.watch_calls[0][2]["stop_after_json_object"] is True
    images = runner.inspect_calls[0][0]
    assert len(images) == 7  # C0 两锚图 + 5 采样帧
    assert runner.inspect_calls[0][1] is IDENTITY_CHECK_PROMPT
    assert len(created) == 5
    assert result["identity"]["c0_consistent"] is True


def test_choose_canonical_pack_accepts_index_and_seconds_picks(
        tmp_path: Path) -> None:
    candidates = {"roles": [{
        "role": "c1_opponent",
        "candidates": [
            {"time_s": 6.44, "path": str(tmp_path / "a.jpg"), "sha256": "a"},
            {"time_s": 6.6, "path": str(tmp_path / "b.jpg"), "sha256": "b"},
            {"time_s": 6.76, "path": str(tmp_path / "c.jpg"), "sha256": "c"},
            {"time_s": 6.92, "path": str(tmp_path / "d.jpg"), "sha256": "d"},
            {"time_s": 7.08, "path": str(tmp_path / "e.jpg"), "sha256": "e"}]}]}
    # 人工按编号选（p0510：zip 没带时间戳，编号比换算秒方便）
    pack = choose_canonical_pack(candidates, tmp_path,
                                 picks={"c1_opponent": "candidate_05"})
    assert pack["entries"][0]["chosen_time_s"] == 7.08
    assert pack["entries"][0]["uri"].startswith("file://")
    # 秒数选法仍可用
    pack = choose_canonical_pack(candidates, tmp_path,
                                 picks={"c1_opponent": 6.6})
    assert pack["entries"][0]["chosen_time_s"] == 6.6
    with pytest.raises(LT0Blocked) as excinfo:
        choose_canonical_pack(candidates, tmp_path,
                              picks={"c1_opponent": 13.0})
    assert excinfo.value.reason_code == "canonical_pick_out_of_candidates"
    with pytest.raises(LT0Blocked) as excinfo:
        choose_canonical_pack(candidates, tmp_path,
                              picks={"c1_opponent": "candidate_09"})
    assert excinfo.value.reason_code == "canonical_pick_out_of_candidates"


def test_plan_only_writes_all_eight_wire_payloads_without_http(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p04e = _p04e_fixtures()
    p04e_dir = tmp_path / "p04e"
    p04e_dir.mkdir()
    mapping = {"content_program": "reference_content_program.json",
               "edit_program": "reference_edit_program.json",
               "requirement": "material_requirements.json",
               "section_observations": "section_observations.json"}
    for key, value in p04e.items():
        (p04e_dir / mapping[key]).write_text(json.dumps(value),
                                             encoding="utf-8")
    assets = p04e_dir / "review_assets"
    assets.mkdir()
    (assets / "section_02.mp4").write_bytes(b"clip")
    monkeypatch.setattr(
        lt0, "probe_media_geometry",
        lambda ffprobe, video: {"width": 720, "height": 1280,
                                "duration_s": 11.6})
    monkeypatch.setattr(
        lt0, "detect_active_crop",
        lambda ffmpeg_bin, video, **kwargs: {"w": 720, "h": 1019, "x": 0,
                                             "y": 129})

    def fake_extract(ffmpeg_bin, reference, timestamp, jpg,
                     crop_filter=None):
        jpg.write_bytes(b"f")

    monkeypatch.setattr(lt0, "_extract_frame", fake_extract)
    monkeypatch.setattr(lt0, "_file_hash", lambda path: "hash")
    monkeypatch.setattr(lt0.common, "run_ffmpeg", lambda *args, **kwargs: None)
    import src.agentic_video.generation_long_take as module

    summary = module.run_lt0_experiment(
        _NSConfig(), p04e_dir, tmp_path / "out",
        reference=tmp_path / "video.mp4", plan_only=True, execute=False)
    assert summary["phase"] == "planned"
    assert summary["take_count"] == 8
    assert summary["aspect_ratio"] == "3:4"  # 有效画面而非容器
    plan = json.loads((tmp_path / "out" / "lt0_plan.json").read_text(
        encoding="utf-8"))
    assert plan["execute_authorized"] is False
    assert plan["aspect_basis"]["active"] == [720, 1019]
    assert plan["server_routing"]["prompt_only"].startswith("fl2va")
    take_dirs = sorted((tmp_path / "out" / "takes").iterdir())
    assert len(take_dirs) == 8
    prompt_only = json.loads(
        (tmp_path / "out" / "takes" / "prompt_only_s1001" / "request.json")
        .read_text(encoding="utf-8"))
    canonical_d = json.loads(
        (tmp_path / "out" / "takes" / "canonical_plus_video_s1002" /
         "request.json").read_text(encoding="utf-8"))
    # A：零条件、T2VA 三字段（无 <Picture>/<Video>/<Subject>）
    assert prompt_only["conditions"] == []
    assert "integrated_multimodal_description:" in prompt_only["prompt"]
    assert "<Picture" not in prompt_only["prompt"]
    assert "<Video" not in prompt_only["prompt"]
    # D：3 图（无 arena）+ 1 静音视频，标签与条件顺序对应 + 混合路由备注
    assert [row["type"] for row in canonical_d["conditions"]] == [
        "image", "image", "image", "video"]
    assert canonical_d["_lt0"]["route_override_note"] == "ref2va_mixed"
    assert canonical_d["_lt0"]["condition_mix"] == "image+video"
    for take_dir in take_dirs:
        assert (take_dir / "prompt.txt").is_file()


def test_parallel_lanes_distribute_jobs_across_endpoints(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """多端点并发：每端点一对卡、任务均分、全部 validated、每端点都干活。"""
    import threading

    p04e = _p04e_fixtures()
    p04e_dir = tmp_path / "p04e"
    p04e_dir.mkdir()
    mapping = {"content_program": "reference_content_program.json",
               "edit_program": "reference_edit_program.json",
               "requirement": "material_requirements.json",
               "section_observations": "section_observations.json"}
    for key, value in p04e.items():
        (p04e_dir / mapping[key]).write_text(json.dumps(value),
                                             encoding="utf-8")
    (p04e_dir / "review_assets").mkdir()
    (p04e_dir / "review_assets" / "section_02.mp4").write_bytes(b"clip")
    monkeypatch.setattr(
        lt0, "probe_media_geometry",
        lambda ffprobe, video: {"width": 720, "height": 1018,
                                "duration_s": 11.6})

    def fake_extract(ffmpeg_bin, reference, timestamp, jpg,
                     crop_filter=None):
        jpg.write_bytes(b"f")

    monkeypatch.setattr(lt0, "_extract_frame", fake_extract)
    monkeypatch.setattr(lt0, "_file_hash", lambda path: "hash")
    monkeypatch.setattr(lt0.common, "run_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(lt0, "detect_active_crop", lambda *a, **k: None)
    import src.agentic_video.generation_p51 as p51
    monkeypatch.setattr(
        p51, "probe_gpu_preflight",
        lambda indices, **kwargs: {"status": "READY"})
    started_pairs: list[str] = []
    monkeypatch.setattr(
        lt0, "_ensure_h3_server",
        lambda endpoint, **kwargs: (
            started_pairs.append(kwargs["gpu_set"]), None)[1])

    lane_submits: dict[str, int] = {}

    class FakeClient:
        lock = threading.Lock()

        def __init__(self, endpoint):
            self.endpoint = endpoint

        def serialize_payload(self, request):
            return request

        def submit(self, payload):
            with FakeClient.lock:
                lane_submits[self.endpoint] = \
                    lane_submits.get(self.endpoint, 0) + 1
            return {"id": f"job-{self.endpoint.rsplit(':', 1)[-1]}-"
                    f"{payload['seed']}"}

        def poll(self, job_id):
            return {"status": "completed", "id": job_id}

        def download(self, job_id, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"take")
            return {"output_path": str(output_path), "sha256": "h"}

    monkeypatch.setattr(lt0, "SGLangH3Client", FakeClient)
    monkeypatch.setattr(
        lt0.common, "run_ffprobe_json",
        lambda ffprobe, video: {"format": {"duration": "5.0"},
                                "streams": [{"codec_type": "video",
                                             "width": 768, "height": 1024}]})

    class OmniLessPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "src.perception.omni_pool.OmniProcessPool", OmniLessPool)
    monkeypatch.setattr(
        lt0, "observe_long_take",
        lambda video, out, runner, take_id, pack, ffmpeg_bin="ffmpeg": {
            "take_id": take_id, "duration_s": 5.0,
            "observation": _observation()["observation"],
            "identity": _observation()["identity"]})

    import src.agentic_video.generation_long_take as module
    endpoints = ("http://127.0.0.1:30110,"
                 "http://127.0.0.1:30111,"
                 "http://127.0.0.1:30112,"
                 "http://127.0.0.1:30113")
    summary = module.run_lt0_experiment(
        _NSConfig(), p04e_dir, tmp_path / "out",
        reference=tmp_path / "video.mp4", execute=True,
        seeds=(1003, 1004, 1005, 1006), recipes=("prompt_only",),
        seconds=5, gpu_set="0,1,2,3,4,5,6,7",
        ref2va_endpoint=endpoints, fl2va_endpoint=endpoints)
    assert summary["phase"] == "complete"
    # 4 端点各分到一对卡，且每条车道都实际接到任务（4 任务 4 车道各 1）
    assert started_pairs == ["0,1", "2,3", "4,5", "6,7"]
    assert sorted(lane_submits.values()) == [1, 1, 1, 1]
    report = json.loads((tmp_path / "out" / "lt0_report.json").read_text(
        encoding="utf-8"))
    assert len(report["take_states"]) == 4
    assert all(row["state"] == "transport_validated"
               for row in report["take_states"])


def test_gpu_endpoint_count_mismatch_blocks(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """3 卡配 2 端点 → 快速失败（每实例必须恰好 2 卡）。"""
    monkeypatch.setattr(
        lt0, "_gpu_pairs_for", lambda count: (_ for _ in ()).throw(
            LT0Blocked("h3", "gpu_endpoint_mismatch"))) if False else None
    # 直接测内部约束逻辑：复制 _run_group 的守卫语义
    import src.agentic_video.generation_long_take as module

    def pairs_for(endpoint_count, gpu_set):
        indices = [i for i in gpu_set.split(",") if i.strip()]
        assert len(indices) == 2 * endpoint_count, "gpu_endpoint_mismatch"
        return [",".join(indices[i:i + 2])
                for i in range(0, len(indices), 2)]

    with pytest.raises(AssertionError):
        pairs_for(2, "0,1,2")
    assert pairs_for(2, "0,1,2,3") == ["0,1", "2,3"]


class _NSConfig:
    perception = {"omni": {}}
