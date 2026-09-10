from __future__ import annotations

from src.agentic_video.narrative import ARC_ROLES
from src.agentic_video.story_planner import (_expand_thin_arc, build_story_plan,
                                             rank_story_path,
                                             story_plan_execution_inputs,
                                             validate_story_plan)
from tests.test_agentic_narrative import valid_program


def _candidate(idx: int, role: str, entity: str, *, start: float,
               score: float = 0.8, event_id: str | None = None) -> dict:
    return {"row_idx": idx, "video": "movie.mp4", "video_stem": "guimie",
            "shot_idx": idx, "source_start_s": start, "source_end_s": start + 3,
            "duration_s": 3.0, "caption": role, "story_role": role,
            "entity_ids": [entity], "event_id": event_id or f"source-{idx}",
            "semantic_score": score, "dialogue": []}


def test_rank_story_path_prefers_entity_and_source_continuity():
    candidates = [
        [_candidate(0, "hook", "tanjiro", start=0),
         _candidate(1, "hook", "other", start=0, score=0.9)],
        [_candidate(2, "conflict", "tanjiro", start=4),
         _candidate(3, "conflict", "other2", start=4, score=0.9)],
        [_candidate(4, "resolution", "tanjiro", start=8)],
    ]
    path = rank_story_path(candidates)
    assert [row["row_idx"] for row in path] == [0, 2, 4]


def test_story_plan_is_traceable_and_rejects_missing_source():
    program = valid_program()
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "resolution", "person", start=9, event_id="e3")]
    plan = build_story_plan(program, rows, theme="守护与牺牲", library="guimie",
                            target_duration_s=45.0)
    assert validate_story_plan(plan) == []
    plan["slots"][0]["source"]["video"] = ""
    assert any("source.video" in e for e in validate_story_plan(plan))


def test_expand_thin_arc_chunks_events_in_time_order():
    program = valid_program()
    program["arc"] = [program["arc"][0]]                     # 只留 hook
    expanded, added = _expand_thin_arc(program, min_slots=3)
    assert added == ["conflict", "climax"]
    roles = [segment["role"] for segment in expanded["arc"]]
    assert roles == ["hook", "conflict", "climax"]
    assert all(segment.get("synthesized") for segment in expanded["arc"][1:])
    # 两个补位段按时序切分事件引用：冲突拿早期、高潮拿后期
    assert expanded["arc"][1]["event_ids"] == ["e1"]
    assert expanded["arc"][2]["event_ids"] == ["e2", "e3"]


def test_thin_reference_arc_never_renders_a_single_slot():
    """2026-09-10 首跑病灶：arc=[hook] 把 60s 全灌进 1 个槽，成片零剪辑。"""
    program = valid_program()
    program["arc"] = [program["arc"][0]]
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "climax", "person", start=8, event_id="e2"),
            _candidate(3, "resolution", "person", start=12, event_id="e3")]
    plan = build_story_plan(program, rows, theme="鬼灭高燃战斗", library="guimie",
                            target_duration_s=60.0)
    assert len(plan["slots"]) >= 3
    assert plan["arc_expanded"]["added_roles"] == ["conflict", "climax"]
    assert validate_story_plan(plan) == []
    order = {role: idx for idx, role in enumerate(ARC_ROLES)}
    roles = [slot["role"] for slot in plan["slots"]]
    assert roles == sorted(roles, key=order.get)
    assert all(slot["target_interval"][1] - slot["target_interval"][0] < 60.0
               for slot in plan["slots"])


def _wide_candidate(idx: int, role: str, start: float) -> dict:
    row = _candidate(idx, role, "person", start=start, event_id=f"e{idx}")
    row.update({"source_end_s": start + 45, "duration_s": 45,
                "dialogue": [
                    {"start_s": start + 1, "end_s": start + 3,
                     "original": "前", "translation_zh": "窗内的对白"},
                    {"start_s": start + 38, "end_s": start + 40,
                     "original": "後", "translation_zh": "会越界的对白"},
                ]})
    return row


def test_oversized_event_is_trimmed_to_slot_not_dropped():
    """2026-09-10 v4 黑屏：45s 合并段 + 越界对白被一票否决 → picked=None →
    渲染器 0 segment 纯黑 60s。现在必须裁剪保槽，只留窗内完整对白。"""
    program = valid_program()
    program["arc"] = program["arc"][:3]                      # hook/conflict/choice
    rows = [_wide_candidate(0, "hook", start=5040),
            _wide_candidate(1, "conflict", start=5370),
            _wide_candidate(2, "choice", start=7819)]
    plan = build_story_plan(program, rows, theme="鬼灭高燃战斗", library="guimie",
                            target_duration_s=60.0)
    assert validate_story_plan(plan) == []
    for slot in plan["slots"]:
        source = slot["source"]
        assert slot["status"] == "supported"                 # 不再一票否决
        assert slot["source_interval_trimmed"] is True
        assert source["end_s"] - source["start_s"] <= 20.0 + 1e-6
        assert [line["translation_zh"] for line in source["dialogue"]] == ["窗内的对白"]


def test_execution_inputs_never_advertise_reference_text_ops():
    """2026-09-10 v5 帧验：参考 Recipe 文字层（黄色 NANCHANG）不得进入叙事
    执行视图的槽操作类型；非文字操作照常映射。"""
    program = valid_program()
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "resolution", "person", start=9, event_id="e3")]
    plan = build_story_plan(program, rows, theme="鬼灭高燃战斗", library="guimie",
                            target_duration_s=60.0)
    recipe = {
        "reference": {"duration_s": 25.3},
        "operations": [
            {"type": "hard_cut", "interval": [0.0, 25.3]},
            {"type": "text_layer_animation", "interval": [2.0, 8.0],
             "params": {"text": "黄色的\"NANCHANG\"文字叠加在画面上"}},
            {"type": "text_overlay", "interval": [10.0, 12.0],
             "params": {"text": "贴纸"}},
        ],
    }
    asset_plan, _retrieval = story_plan_execution_inputs(plan, recipe)
    for slot in asset_plan["slots"]:
        assert "text_layer_animation" not in slot["operation_types"]
        assert "text_overlay" not in slot["operation_types"]
        assert "hard_cut" in slot["operation_types"]
