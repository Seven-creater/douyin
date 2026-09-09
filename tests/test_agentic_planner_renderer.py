from __future__ import annotations

import numpy as np
import pytest

from src.agentic_video.benchmark import render_controlled_video
from src.agentic_video.planner import (build_asset_plan, rank_asset_slots,
                                       timeline_slots)
from src.agentic_video.recipe_v2 import new_recipe
from pathlib import Path

from src.agentic_video.renderer import (GroundedSamSubprocessBackend, final_filter,
                                        render_recipe, segment_filter)
from src.config import AppConfig, PathsCfg


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


def test_renderer_filters_compile_requested_effects():
    vf = segment_filter(_recipe()["operations"], duration_s=2.0)
    assert "setpts=PTS/2" in vf and "trim=duration=2" in vf
    assert "force_original_aspect_ratio=increase" in vf
    assert "crop=720:960" in vf and ",pad=" not in vf
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
