from __future__ import annotations

from src.agentic_video.benchmark import (build_suite_specs, compare_ablation,
                                         evaluate_recipes, interval_iou,
                                         render_controlled_video)
from src.agentic_video.recipe_v2 import sha256_file, validate_recipe_v2


def test_controlled_suite_has_exact_composition_and_is_deterministic():
    first = build_suite_specs()
    second = build_suite_specs()
    assert len(first) == 144
    assert sum(r["reference"]["id"].startswith("single_") for r in first) == 96
    assert sum(r["reference"]["id"].startswith("paired_") for r in first) == 24
    assert sum(r["reference"]["id"].startswith("compound_") for r in first) == 24
    assert first == second
    assert all(validate_recipe_v2(recipe) == [] for recipe in first)


def test_controlled_renderer_writes_video(tmp_path):
    recipe = build_suite_specs(duration_s=1.0, fps=8.0)[0]
    output = render_controlled_video(recipe, tmp_path / "case.mp4", width=96, height=64)
    assert output.stat().st_size > 100
    assert len(sha256_file(output)) == 64


def test_interval_iou_handles_point_and_span():
    assert interval_iou([1.0, 1.0], [1.05, 1.05], tolerance_s=0.1) == 1.0
    assert interval_iou([0.0, 1.0], [0.5, 1.5]) == 1 / 3


def test_perfect_predictions_pass_all_recipe_metrics():
    truth = build_suite_specs()
    metrics = evaluate_recipes(truth, truth)
    assert metrics["cut_boundary_f1"] == 1.0
    assert metrics["op_macro_f1"] == 1.0
    assert metrics["median_tiou"] == 1.0
    assert metrics["ungrounded_rate"] == 0.0
    assert metrics["passes_thresholds"] is True


def test_ablation_accepts_layer_gain_or_call_reduction():
    fixed = {"layer_op_f1": 0.5, "op_macro_f1": 0.8, "model_calls": 100}
    by_quality = {"layer_op_f1": 0.61, "op_macro_f1": 0.82, "model_calls": 100}
    by_calls = {"layer_op_f1": 0.5, "op_macro_f1": 0.8, "model_calls": 70}
    assert compare_ablation(fixed, by_quality)["passes_agent_gate"]
    assert compare_ablation(fixed, by_calls)["passes_agent_gate"]
