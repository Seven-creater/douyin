from __future__ import annotations

import numpy as np
import pytest

from src.agentic_video.benchmark import render_controlled_video
from src.agentic_video.planner import (build_asset_plan, rank_asset_slots,
                                       timeline_slots)
from src.agentic_video.recipe_v2 import new_recipe
from pathlib import Path

from src.agentic_video.renderer import (GroundedSamSubprocessBackend, final_filter,
                                        render_recipe, segment_filter,
                                        write_story_subtitles)
from src.config import AppConfig, PathsCfg
from src.perception import common


def _recipe():
    recipe = new_recipe(reference_id="r", reference_uri="r.mp4", sha256="a" * 64,
                        duration_s=4.0, fps=24.0)
    recipe["operations"] = [
        {"id": "cut", "type": "hard_cut", "interval": [2.0, 2.0],
         "track_id": "video_main", "inputs": [], "depends_on": [], "params": {},
         "evidence": [{"source": "test"}], "confidence": 1.0, "status": "supported"},
        {"id": "speed", "type": "speed_ramp", "interval": [2.2, 3.0],
         "track_id": "video_main", "inputs": [], "depends_on": [],
         "params": {"rate": 2.0, "subject": "剑士", "motion": "fast"},
         "evidence": [{"source": "test"}], "confidence": 1.0, "status": "supported"},
    ]
    return recipe


def test_asset_plan_uses_recipe_cuts_and_theme():
    recipe = _recipe()
    assert timeline_slots(recipe) == [(0.0, 2.0), (2.0, 4.0)]
    plan = build_asset_plan(recipe, "鬼灭高燃战斗", library="guimie")
    assert len(plan["slots"]) == 2
    assert all("鬼灭高燃战斗" in slot["query"] for slot in plan["slots"])
    assert "剑士" in plan["slots"][1]["query"]


def test_asset_ranking_filters_source_and_uses_action_fit():
    rows = [
        {"source": "other", "video_stem": "x", "video": "x.mp4", "shot_idx": 0,
         "start_s": 0, "end_s": 2, "duration_s": 2, "action_score": 1.0},
        {"source": "guimie", "video_stem": "g", "video": "g.mp4", "shot_idx": 1,
         "start_s": 10, "end_s": 12, "duration_s": 2, "action_score": 0.8},
    ]
    emb = np.array([[1.0, 0.0], [0.9, 0.1]], dtype="float32")
    queries = np.array([[1.0, 0.0]], dtype="float32")
    slots = [{"slot_idx": 0, "query": "战斗", "need_duration_s": 1.0,
              "visual_intensity": 0.8}]
    ranked = rank_asset_slots(rows, emb, queries, slots, source="guimie")
    assert ranked[0]["picked"]["row_idx"] == 1


def test_asset_ranking_skips_source_zone_excluded_rows():
    """片尾职员表镜头即便语义分最高也不得被选进成片（zones 过滤接入检索）。"""
    rows = [
        {"source": "guimie", "video": "m.mp4", "video_stem": "g", "shot_idx": 0,
         "start_s": 9060, "end_s": 9105, "duration_s": 45, "action_score": 1.0},
        {"source": "guimie", "video": "m.mp4", "video_stem": "g", "shot_idx": 1,
         "start_s": 3030, "end_s": 3035, "duration_s": 5, "action_score": 0.8},
    ]
    emb = np.array([[1.0, 0.0], [0.95, 0.05]], dtype="float32")
    queries = np.array([[1.0, 0.0]], dtype="float32")
    slots = [{"slot_idx": 0, "query": "战斗", "need_duration_s": 5.0,
              "visual_intensity": 0.9}]
    ranked = rank_asset_slots(rows, emb, queries, slots, source="guimie",
                              excluded_rows={0})
    assert ranked[0]["picked"]["row_idx"] == 1


def test_renderer_filters_compile_requested_effects():
    recipe = _recipe()
    recipe["operations"][1]["interval"] = [2.0, 4.0]          # 覆盖整个 slot [2,4]
    vf, skipped = segment_filter(recipe["operations"], duration_s=2.0,
                                 slot_start=2.0, slot_end=4.0)
    assert "setpts=PTS/2" in vf and "trim=duration=2" in vf
    assert "force_original_aspect_ratio=increase" in vf
    assert "crop=1920:1080" in vf and "setsar=1" in vf and ",pad=" not in vf
    assert skipped == []


def test_segment_filter_partial_interval_policy():
    """H4：setpts/scale/tpad 无 timeline 支持——部分覆盖跳过并记录；eq 用 enable 窗口。"""
    recipe = _recipe()
    ops = recipe["operations"]                                  # speed_ramp [2.2,3.0]
    ops.append({"id": "eq", "type": "color_adjust", "interval": [2.2, 3.0],
                "track_id": "video_main", "inputs": [], "depends_on": [],
                "params": {}, "evidence": [{"source": "t"}], "confidence": 1.0,
                "status": "supported"})
    vf, skipped = segment_filter(ops, duration_s=2.0, slot_start=2.0, slot_end=4.0)
    assert "setpts=PTS/2" not in vf                             # 部分覆盖不整段变速
    assert "PTS-STARTPTS" in vf                                 # 链尾复位仍在
    assert len(skipped) == 1 and skipped[0]["status"] == "partial_interval_skipped"
    assert skipped[0]["type"] == "speed_ramp"
    assert "enable='between(t,0.2,1)'" in vf                    # eq 局部窗口（slot 内坐标，%g 格式）
    assert "eq=saturation=1.25" in vf


def test_segment_filter_can_follow_model_subject_anchor():
    vf, _ = segment_filter([], width=1920, height=1080, duration_s=2.0, focus_x=0.2)
    assert "crop=1920:1080:(iw-1920)*0.2" in vf


def test_render_canvas_rule_is_1920x1080_landscape():
    """2026-09-11 用户规定：所有剪辑成片（叙事+编辑两模式）一律 1920×1080
    横屏。裸配置回退与随库 default.yaml 都钉死在此；改回竖屏必须显式改
    配置并有意触碰本测试。"""
    from src.agentic_video.renderer import render_canvas
    from src.config import load_config

    bare = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={},
        logging_level="INFO", perception={}, library={},
        paths=PathsCfg(raw_dir="r", processed_dir="p", videos_dir="v", logs_dir="l",
                       perception_dir="per", generation_dir="g", library_dir="lib"))
    assert render_canvas(bare, narrative_mode=True) == (1920, 1080)
    assert render_canvas(bare, narrative_mode=False) == (1920, 1080)

    cfg = load_config()                                        # 随库交付配置
    assert render_canvas(cfg, narrative_mode=True) == (1920, 1080)
    # 显式覆盖仍生效：竖屏要走配置声明，不得靠代码回退复活
    overridden = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={},
        logging_level="INFO", perception={},
        library={"narrative_render": {"width": 1080, "height": 1920}},
        paths=PathsCfg(raw_dir="r", processed_dir="p", videos_dir="v", logs_dir="l",
                       perception_dir="per", generation_dir="g", library_dir="lib"))
    assert render_canvas(overridden, narrative_mode=True) == (1080, 1920)


def test_final_filter_compiles_point_flash():
    recipe = _recipe()
    recipe["operations"].append({
        "id": "flash", "type": "luma_flash", "interval": [1.0, 1.0],
        "track_id": "video_main", "inputs": [], "depends_on": [], "params": {},
        "evidence": [{"source": "test"}], "confidence": 1.0, "status": "supported"})
    assert "drawbox" in final_filter(recipe)


def test_grounded_sam_command_uses_dedicated_python_and_explicit_operation():
    backend = GroundedSamSubprocessBackend({
        "python": "/env/bin/python", "grounding_model": "/models/dino",
        "sam2_code_root": "/models/sam2/code", "sam2_checkpoint": "/models/sam2.pt",
    })
    operation = {"type": "tracked_mask_fill", "interval": [1.0, 2.0],
                 "params": {"target": "person"}}
    command = backend.build_command(Path("in.mp4"), operation, Path("out.mp4"), Path("work"))
    assert command[0] == "/env/bin/python"
    assert "tracked_mask_fill" in command and "person." in command


def test_renderer_executes_a_small_deterministic_video(tmp_path):
    recipe = _recipe()
    source_recipe = new_recipe(reference_id="source", reference_uri="source.mp4", sha256="",
                               duration_s=4.0, fps=12.0)
    source = render_controlled_video(source_recipe, tmp_path / "source.mp4",
                                     width=160, height=90)
    recipe["reference"]["uri"] = str(source)
    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={}, logging_level="INFO",
        perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"}, library={},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "per", generation_dir=tmp_path / "g",
                       library_dir=tmp_path / "lib"))
    plan = build_asset_plan(recipe, "战斗", library="guimie")
    retrieval = [{"slot_idx": slot["slot_idx"], "missing": "", "picked": {
        "source_start_s": 0.0, "video": str(source), "video_stem": "source",
        "shot_idx": 0, "caption": "动作", "duration_s": 4.0,
    }} for slot in plan["slots"]]
    try:
        output = render_recipe(cfg, recipe, plan, retrieval, tmp_path / "out")
    except FileNotFoundError:
        pytest.skip("ffmpeg unavailable")
    assert output.stat().st_size > 1000
    manifest = (tmp_path / "out" / "render_manifest.json").read_text(encoding="utf-8")
    assert '"deterministic": true' in manifest


def test_renderer_resolves_relative_output_for_concat(tmp_path, monkeypatch):
    recipe = _recipe()
    source_recipe = new_recipe(reference_id="source", reference_uri="source.mp4", sha256="",
                               duration_s=4.0, fps=12.0)
    source = render_controlled_video(source_recipe, tmp_path / "source.mp4",
                                     width=160, height=90)
    recipe["reference"]["uri"] = str(source)
    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={}, logging_level="INFO",
        perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"}, library={},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "per", generation_dir=tmp_path / "g",
                       library_dir=tmp_path / "lib"))
    plan = build_asset_plan(recipe, "战斗", library="guimie")
    retrieval = [{"slot_idx": slot["slot_idx"], "missing": "", "picked": {
        "source_start_s": 0.0, "video": str(source), "video_stem": "source",
        "shot_idx": 0, "caption": "动作", "duration_s": 4.0,
    }} for slot in plan["slots"]]
    monkeypatch.chdir(tmp_path)
    output = render_recipe(cfg, recipe, plan, retrieval, Path("relative-out"))
    assert output == (tmp_path / "relative-out" / "rendered.mp4").resolve()
    assert output.stat().st_size > 1000


def test_story_subtitles_map_source_dialogue_to_target_time(tmp_path):
    plan = {"slots": [{"slot_idx": 0, "start_s": 10.0, "end_s": 15.0}]}
    retrieval = [{"slot_idx": 0, "picked": {
        "source_start_s": 100.0, "source_end_s": 105.0,
        "dialogue": [{"start_s": 101.0, "end_s": 103.0,
                      "translation_zh": "我要保护你"}],
    }}]
    path = write_story_subtitles(plan, retrieval, tmp_path / "story.srt")
    text = path.read_text(encoding="utf-8")
    assert "00:00:11,000 --> 00:00:13,000" in text
    assert "我要保护你" in text


def test_narrative_renderer_adds_silence_when_source_has_no_audio(tmp_path):
    recipe = _recipe()
    source_recipe = new_recipe(reference_id="source", reference_uri="source.mp4", sha256="",
                               duration_s=4.0, fps=12.0)
    source = render_controlled_video(source_recipe, tmp_path / "silent.mp4",
                                     width=160, height=90)
    recipe["reference"]["uri"] = str(source)
    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={}, logging_level="INFO",
        perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"},
        library={"narrative_render": {"width": 320, "height": 180}},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "per", generation_dir=tmp_path / "g",
                       library_dir=tmp_path / "lib"))
    plan = {"plan_version": "narrative-1.0", "theme": "守护", "library": "guimie",
            "reference_id": "r", "narrative_program_required": True,
            "slots": [{"slot_idx": 0, "start_s": 0.0, "end_s": 2.0,
                       "need_duration_s": 2.0, "operation_types": []}]}
    retrieval = [{"slot_idx": 0, "missing": "", "picked": {
        "source_start_s": 0.0, "source_end_s": 2.0, "video": str(source),
        "video_stem": "silent", "shot_idx": 0, "dialogue": []}}]
    try:
        output = render_recipe(cfg, recipe, plan, retrieval, tmp_path / "narrative",
                               force=True)
        probe = common.run_ffprobe_json("ffprobe", output)
    except FileNotFoundError:
        pytest.skip("ffmpeg unavailable")
    assert any(stream.get("codec_type") == "audio" for stream in probe["streams"])


def test_drawtext_text_value_quotes_apostrophes_and_percents():
    """H2：撇号用 关-转-开 注入，百分号交给 expansion=none。"""
    from src.agentic_video.renderer import drawtext_text_value

    expected = "'" + "it" + "'\\''" + "s'"          # 字面量 'it'\''s'
    assert drawtext_text_value("it's") == expected
    assert drawtext_text_value("50%") == "'50%'"
    assert drawtext_text_value("") == "''"


def test_story_subtitles_prefer_copy_cues_over_dialogue(tmp_path):
    """MVP：文案轨在场时烧钩子/成就卡/反转梗，不再映射素材对白翻译。"""
    plan = {"slots": [{"slot_idx": 0, "start_s": 0.0, "end_s": 11.0}],
            "copy_cues": [
                {"kind": "hook_line", "start_s": 0.15, "end_s": 4.0, "text": "钩子文案"},
                {"kind": "info_card", "start_s": 9.2, "end_s": 10.2, "text": "成就卡"},
                {"kind": "punchline", "start_s": 10.0, "end_s": 10.95, "text": "反转梗"},
            ]}
    retrieval = [{"slot_idx": 0, "picked": {
        "source_start_s": 100.0, "source_end_s": 111.0,
        "dialogue": [{"start_s": 101.0, "end_s": 103.0, "translation_zh": "素材对白"}],
    }}]
    path = write_story_subtitles(plan, retrieval, tmp_path / "copy.srt")
    text = path.read_text(encoding="utf-8")
    assert "钩子文案" in text and "反转梗" in text
    assert "素材对白" not in text                      # 对白翻译被压制
    assert "00:00:00,150 --> 00:00:04,000" in text


def test_render_cache_key_sensitive_to_copy_track():
    from src.agentic_video.renderer import render_cache_key
    recipe = _recipe()
    base_plan = {"theme": "守护", "slots": [{"slot_idx": 0, "start_s": 0,
                                            "end_s": 2, "need_duration_s": 2}]}
    retrieval = [{"slot_idx": 0, "picked": {"video": "g.mp4", "source_start_s": 10.0}}]
    with_copy = {**base_plan, "copy_cues": [{"text": "钩子"}], "audio_mode": "bgm"}
    a = render_cache_key(recipe, base_plan, retrieval, canvas_width=720,
                         canvas_height=960, narrative_mode=True)
    b = render_cache_key(recipe, with_copy, retrieval, canvas_width=720,
                         canvas_height=960, narrative_mode=True)
    assert a["sha256"] != b["sha256"]


def test_scaled_recipe_strips_reference_text_ops_for_narrative_render():
    """2026-09-10 v5 帧验：参考 Recipe 的 text_layer_animation（黄色 NANCHANG 字幕
    转录）被 drawtext 烧到鬼灭画面上。叙事执行视图必须剔除文字层；缩放与否都剔。"""
    from src.agentic_video.renderer import _scaled_recipe
    recipe = _recipe()
    recipe["operations"].append({
        "id": "text", "type": "text_layer_animation", "interval": [1.0, 3.0],
        "track_id": "video_main", "inputs": [], "depends_on": [],
        "params": {"text": "黄色的\"NANCHANG\"文字叠加在画面上"},
        "evidence": [{"source": "test"}], "confidence": 1.0, "status": "supported"})
    scaled = _scaled_recipe(recipe, 8.0)                        # 4s 参考 → 8s 目标
    assert [op["type"] for op in scaled["operations"]] == ["hard_cut", "speed_ramp"]
    assert scaled["reference"]["duration_s"] == 8.0
    assert scaled["operations"][0]["interval"] == [4.0, 4.0]    # 区间缩放仍在
    same_duration = _scaled_recipe(recipe, 4.0)                 # 无需缩放也要剔
    assert [op["type"] for op in same_duration["operations"]] == ["hard_cut", "speed_ramp"]
    assert len(recipe["operations"]) == 3                        # 原 recipe 不被改动


def test_render_cache_key_is_sensitive_to_theme_and_canvas():
    """H3：theme/画布/叙事模式变化必须改变缓存键——旧逻辑只看文件存在。"""
    from src.agentic_video.renderer import render_cache_key

    recipe = _recipe()
    plan = {"theme": "鬼灭高燃战斗", "slots": [{"slot_idx": 0, "start_s": 0,
                                          "end_s": 2, "need_duration_s": 2}]}
    retrieval = [{"slot_idx": 0, "picked": {"video": "g.mp4", "source_start_s": 10.0}}]
    base = render_cache_key(recipe, plan, retrieval, canvas_width=720,
                            canvas_height=960, narrative_mode=False)
    assert base == render_cache_key(recipe, plan, retrieval, canvas_width=720,
                                    canvas_height=960, narrative_mode=False)
    changed_theme = render_cache_key(recipe, {**plan, "theme": "蜘蛛侠"},
                                     retrieval, canvas_width=720, canvas_height=960,
                                     narrative_mode=False)
    changed_canvas = render_cache_key(recipe, plan, retrieval, canvas_width=1920,
                                      canvas_height=1080, narrative_mode=False)
    changed_mode = render_cache_key(recipe, plan, retrieval, canvas_width=720,
                                    canvas_height=960, narrative_mode=True)
    assert len({base["sha256"], changed_theme["sha256"], changed_canvas["sha256"],
                changed_mode["sha256"]}) == 4


def test_source_subtitle_band_crop_applies_to_luoxiaohei_only():
    """V3 P4：罗小黑 WEB-DL 内嵌字幕带按源配置裁切（Omni 索引侧不裁），
    未配置的源不受影响。"""
    from src.agentic_video.renderer import source_subtitle_treatment
    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={},
        logging_level="INFO", perception={},
        library={"sources": {"luoxiaohei1": {"subtitle_band": {"crop_bottom": 0.12}}}},
        paths=PathsCfg(raw_dir="r", processed_dir="p", videos_dir="v", logs_dir="l",
                       perception_dir="per", generation_dir="g", library_dir="lib"))
    crop = source_subtitle_treatment(cfg, "luoxiaohei1__narrative")
    assert crop == "crop=iw:ih*0.88:0:0"
    assert source_subtitle_treatment(cfg, "guimie") == ""      # 未配置不裁
    assert source_subtitle_treatment(cfg, "") == ""
