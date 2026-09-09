from __future__ import annotations

import json

from src.agentic_video.benchmark import build_suite_specs, compare_ablation_variants
from src.agentic_video.discovery import metadata_triage, parse_semantic_audit
from src.agentic_video.story_planner import (merge_event_candidates,
                                               story_plan_execution_inputs)


def test_semantic_event_count_mismatch_is_always_ineligible():
    row = metadata_triage({"aweme_id": "x", "title": "暖心救助",
                           "stats": {}, "extras": {"media_type": 4,
                                                     "duration_ms": 30000}})
    result = parse_semantic_audit(json.dumps({
        "category": "real_story", "event_count": 4,
        "events": [{"start_s": 0, "end_s": 1, "evidence_sources": ["frame"]},
                    {"start_s": 2, "end_s": 3, "evidence_sources": ["asr"]},
                    {"start_s": 4, "end_s": 5, "evidence_sources": ["ocr"]}],
        "has_goal_or_causality": True, "has_resolution": True,
        "evidence_coverage": 1.0, "story_clarity": 1.0,
        "confidence": 1.0, "pure_sensory": False}), row)
    assert result["eligible"] is False
    assert "event_count_mismatch" in result["rejection_reasons"]


def test_story_execution_maps_recipe_timebase_to_target_slots():
    story = {"story_plan_version": "1.0", "theme": "主题", "library": "guimie",
             "target_duration_s": 60.0, "reference_id": "r", "slots": [{
                 "slot_idx": 0, "role": "hook", "target_interval": [0.0, 60.0],
                 "reference_event_ids": [], "source": {"video": "v.mp4",
                 "video_stem": "v", "shot_idx": 0, "start_s": 0.0, "end_s": 1.0,
                 "event_id": "e", "causal_predecessors": [], "caption": "x",
                 "entity_ids": [], "focus_x": 0.5, "dialogue": []},
                 "status": "supported", "reason": "", "transition_reason": "opening"}]}
    recipe = {"reference": {"duration_s": 10.0}, "operations": [{"type": "speed_ramp",
              "interval": [8.0, 9.0]}]}
    story["slots"][0]["target_interval"] = [0.0, 30.0]
    second = json.loads(json.dumps(story["slots"][0]))
    second.update({"slot_idx": 1, "role": "resolution", "target_interval": [30.0, 60.0],
                   "transition_reason": "entity_continuity"})
    story["slots"].append(second)
    asset_plan, _ = story_plan_execution_inputs(story, recipe)
    assert asset_plan["slots"][0]["operation_types"] == []
    assert asset_plan["slots"][1]["operation_types"] == ["speed_ramp"]
    assert recipe["operations"][0]["interval"] == [8.0, 9.0]


def test_three_way_ablation_report_keeps_middle_variant():
    fixed = {"layer_op_f1": 0.5, "op_macro_f1": 0.7, "model_calls": 10}
    signal = {"layer_op_f1": 0.55, "op_macro_f1": 0.72, "model_calls": 9}
    agent = {"layer_op_f1": 0.65, "op_macro_f1": 0.75, "model_calls": 8}
    result = compare_ablation_variants(fixed, signal, agent)
    assert set(("fixed_window", "signal_guided", "agent")) <= result.keys()
    assert result["agent_vs_fixed"]["passes_agent_gate"] is True


def test_adjacent_narrative_shots_merge_into_a_traceable_event_range():
    rows = [
        {"video": "movie.mp4", "video_stem": "movie", "window_idx": 2,
         "event_id": "protect", "story_role": "climax", "shot_idx": 8,
         "source_start_s": 100.0, "source_end_s": 103.0,
         "event_summary": "角色挡在同伴面前", "entity_ids": ["hero"],
         "dialogue": [], "semantic_score": 0.7},
        {"video": "movie.mp4", "video_stem": "movie", "window_idx": 2,
         "event_id": "protect", "story_role": "climax", "shot_idx": 9,
         "source_start_s": 103.0, "source_end_s": 108.0,
         "event_summary": "角色承受攻击", "entity_ids": ["hero", "ally"],
         "dialogue": [{"start_s": 104.0, "end_s": 106.0, "original": "守る"}],
         "semantic_score": 0.8},
    ]
    merged = merge_event_candidates(rows)
    assert len(merged) == 1
    assert merged[0]["source_start_s"] == 100.0
    assert merged[0]["source_end_s"] == 108.0
    assert merged[0]["shot_indices"] == [8, 9]
    assert merged[0]["entity_ids"] == ["ally", "hero"]
