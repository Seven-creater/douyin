from __future__ import annotations

from src.agentic_video.agent import (AgentBudget, answer_operations,
                                     build_recipe_from_agent, choose_probe,
                                     propose_refinement_tasks)
from src.agentic_video.recipe_v2 import validate_recipe_v2
from src.config import AppConfig, PathsCfg
from src.perception import common


def test_answer_operations_supports_old_and_new_shapes():
    assert len(answer_operations({"op_type": "hard_cut"})) == 1
    assert len(answer_operations({"operations": [{"op_type": "hard_cut"},
                                                   {"op_type": "text_overlay"}]})) == 2


def test_probe_selection_targets_layer_and_text():
    assert choose_probe(["tracked_mask_fill"], []) == "region_segmentation_probe"
    assert choose_probe([], [{"what_changed": "text"}]) == "ocr_probe"
    assert choose_probe(["speed_ramp"], []) == "motion_probe"


def test_refinement_tasks_respect_remaining_and_prioritize_missed_layers():
    windows = [
        {"idx": 0, "start": 1.0, "end": 2.0, "confidence": 0.9,
         "hypotheses": ["tracked_mask_fill"]},
        {"idx": 1, "start": 3.0, "end": 4.0, "confidence": 0.5,
         "hypotheses": ["hard_cut"]},
    ]
    results = [
        {"idx": 0, "answer": {"op_type": "hard_cut", "confidence": 0.9}},
        {"idx": 1, "answer": {"op_type": "uncertain", "confidence": 0.2}},
    ]
    tasks = propose_refinement_tasks(windows, results, round_idx=1, remaining=1)
    assert len(tasks) == 1
    assert tasks[0].source_window_idx == 0
    assert tasks[0].probe == "region_segmentation_probe"


def test_build_recipe_from_multi_operation_agent_result(tmp_path):
    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, perception={}, template={}, generation={},
        logging_level="INFO", library={},
        paths=PathsCfg(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
                       videos_dir=tmp_path / "videos", logs_dir=tmp_path / "logs",
                       perception_dir=tmp_path / "perception",
                       generation_dir=tmp_path / "generation",
                       library_dir=tmp_path / "library"))
    vid = "r1"
    video = cfg.paths.videos_dir / vid / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    inspect_dir = cfg.paths.perception_dir / vid / "inspect"
    inspect_dir.mkdir(parents=True)
    common.write_result_json(inspect_dir, tool="inspect", aweme_id=vid, params={},
                             output={"duration_s": 4.0, "fps": 24.0})
    result = {"tool_calls": [{"tool": "omni_window"}], "results": [{
        "idx": 0, "probe": "region_segmentation_probe",
        "task": {"start": 1.0, "end": 1.8},
        "answer": {"operations": [
            {"op_type": "hard_cut", "event_time_original_s": 1.2,
             "confidence": 0.9, "what_changed": "global", "evidence_quote": "切换"},
            {"op_type": "tracked_mask_fill", "event_time_original_s": 1.25,
             "interval_original_s": [1.1, 1.7], "confidence": 0.8,
             "what_changed": "subject_interior", "evidence_quote": "主体内部纹理变换"},
        ]}}]}
    recipe = build_recipe_from_agent(cfg, vid, result)
    assert {op["type"] for op in recipe["operations"]} == {
        "hard_cut", "tracked_mask_fill"}
    assert [op["interval"][0] for op in recipe["operations"]] == [1.1, 1.2]
    assert any(track["type"] == "mask" for track in recipe["tracks"])
    assert validate_recipe_v2(recipe) == []


def test_budget_defaults_match_plan():
    assert AgentBudget().max_initial_windows == 48
    assert AgentBudget().max_rounds == 2
    assert AgentBudget().max_refinement_windows == 16
