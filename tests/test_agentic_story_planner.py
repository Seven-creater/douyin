from __future__ import annotations

from src.agentic_video.narrative import ARC_ROLES
from src.agentic_video.story_planner import (_arc_query, _build_specs,
                                             _emotion_peak_hint,
                                             _expand_thin_arc, _hard_continuity,
                                             build_story_plan, rank_story_path,
                                             slot_need_spec,
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


def test_target_duration_allows_short_emotional_template():
    """2026-09-10：7682 型模板天然 20-35s，旧 45..75 下界把整类挡在门外。"""
    program = valid_program()
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "resolution", "person", start=9, event_id="e3")]
    plan = build_story_plan(program, rows, theme="守护", library="guimie",
                            target_duration_s=21.9)
    assert validate_story_plan(plan) == []
    with_static = dict(plan, target_duration_s=15.0)
    assert any("target_duration_s" in e for e in validate_story_plan(with_static))


def test_same_row_in_two_slots_gets_deduplicated():
    """C3：hook/conflict 隔槽引用同一事件时，Viterbi 相邻惩罚拦不住撞段——
    二 pass 必须换掉重复行或标记 dedup_conflict。"""
    program = valid_program()
    program["arc"] = [
        {"role": "hook", "event_ids": ["e1"]},
        {"role": "context", "event_ids": ["e1"]},
        {"role": "conflict", "event_ids": ["e1"]},
    ]
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1", score=0.95),
            _candidate(1, "context", "other", start=4, event_id="e1", score=0.5),
            _candidate(2, "conflict", "third", start=8, event_id="e1", score=0.5)]
    plan = build_story_plan(program, rows, theme="守护", library="guimie",
                            target_duration_s=45.0)
    picked_rows = [slot["source"].get("shot_idx") for slot in plan["slots"]]
    assert len(picked_rows) == len(set(picked_rows)) or \
        any(slot.get("dedup_conflict") for slot in plan["slots"])


def test_execution_inputs_pass_copy_track_through():
    plan = build_story_plan(valid_program(),
                           [_candidate(0, "hook", "person", start=0, event_id="e1"),
                            _candidate(1, "conflict", "person", start=4, event_id="e2"),
                            _candidate(2, "resolution", "person", start=9, event_id="e3")],
                           theme="守护", library="guimie", target_duration_s=45.0)
    plan["copy"] = {"cues": [{"kind": "hook_line", "start_s": 0.15, "end_s": 4.0,
                              "text": "钩子"}],
                    "audio_mode": "bgm"}
    asset_plan, _ = story_plan_execution_inputs(plan, {"reference": {"duration_s": 25.3},
                                                       "operations": []})
    assert asset_plan["copy_cues"] == plan["copy"]["cues"]
    assert asset_plan["audio_mode"] == "bgm"


def test_slot_need_spec_compiles_six_questions_and_bindings():
    """P2：need 是证据规格不是角色中文化——六问字段进 need，绑定从 participants
    推导，unknown/not_applicable 不被当内容引用。"""
    program = valid_program()
    program["intent"].update({"protagonist": "救助者", "problem": "小猫被困",
                              "motivation": "unknown", "outcome": "not_applicable"})
    arc = program["arc"]
    spec = slot_need_spec(program, arc[0], 0)
    assert "救助者" in spec["need"] and "小猫被困" in spec["need"]
    assert "unknown" not in spec["need"]                     # 未呈现的不引用
    assert set(spec["entity_bindings"]) == {"A", "B"}         # person/cat 双绑定
    assert spec["entity_bindings"]["A"]["reference_name"] == "救助者"
    conflict_spec = slot_need_spec(program,
                                   next(s for s in arc if s["role"] == "conflict"), 1)
    assert conflict_spec["required"] is True
    assert conflict_spec["must_have"] and conflict_spec["must_not"]
    choice_spec = slot_need_spec(program,
                                 next(s for s in arc if s["role"] == "choice"), 2)
    assert choice_spec["required"] is False                   # 可选槽


def test_adjacent_slots_share_bindings_depend_on_each_other():
    program = valid_program()
    specs = _build_specs(program)
    for idx in range(1, len(specs)):
        previous_keys = set(specs[idx - 1]["entity_bindings"])
        if previous_keys & set(specs[idx]["entity_bindings"]):
            assert specs[idx]["depends_on"] == idx - 1
    assert _hard_continuity(specs)[0] is False                # 首槽无边


def test_required_slot_unreachable_under_hard_continuity_is_unsupported():
    """V1 P2 红线：共享实体绑定的相邻槽实体不相交 → 转移非法 → 必选槽
    unsupported，plan 如实报未完成；绝不放宽约束凑满。"""
    program = valid_program()
    program["arc"] = [
        {"role": "hook", "event_ids": ["e1"]},
        {"role": "conflict", "event_ids": ["e2"]},            # 与 hook 共享 person/cat
    ]
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1", score=0.9),
            _candidate(1, "conflict", "stranger", start=4, event_id="e2", score=0.9)]
    plan = build_story_plan(program, rows, theme="救助", library="guimie",
                            target_duration_s=45.0)
    conflict_slot = plan["slots"][1]
    assert conflict_slot["status"] == "unsupported"
    assert conflict_slot["reason"] in {"library_insufficient", "continuity_infeasible"}
    assert plan["required_unsupported"] == [1]
    assert plan["plan_complete"] is False
    assert validate_story_plan(plan) == []                    # 未完成但结构合法


def test_disjoint_entities_without_shared_bindings_still_allowed():
    """绑定不重叠的相邻槽（如环境/反应镜头位）不要求实体连续——哪些位置必须
    同主体由模板（绑定共享）决定，不是一刀切。"""
    program = valid_program()
    program["arc"] = [
        {"role": "hook", "event_ids": ["e1"]},
        {"role": "context", "event_ids": ["e2"]},              # 与 hook 共享绑定
        {"role": "resolution", "event_ids": ["e3"]},
    ]
    program["entities"].append({"id": "stranger", "kind": "person",
                                "name_or_role": "路人", "aliases": [],
                                "evidence": [], "confidence": 0.9,
                                "status": "uncertain"})
    program["events"][2]["participants"] = []                  # 无参与实体（环境/收束镜头位）
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1", score=0.9),
            _candidate(1, "context", "person", start=4, event_id="e2", score=0.9),
            _candidate(2, "resolution", "stranger", start=9, event_id="e3", score=0.9)]
    plan = build_story_plan(program, rows, theme="救助", library="guimie",
                            target_duration_s=45.0)
    assert plan["slots"][1]["status"] in {"supported", "uncertain"}
    assert plan["required_unsupported"] == []


def test_arc_query_v3_leads_with_need_not_bare_role():
    program = valid_program()
    program["intent"].update({"protagonist": "救助者", "problem": "小猫被困"})
    spec = slot_need_spec(program, program["arc"][0], 0)
    query = _arc_query(program, program["arc"][0], "救助", spec=spec)
    assert query.index("叙事需求") < query.index("叙事角色")    # 需求优先于角色
    assert "需要：" in query and "排除：" in query
    assert "人物：" in query                                   # 绑定提示进查询


def test_emotion_peak_hint_from_climax():
    program = valid_program()
    program["arc"].append({"role": "climax", "event_ids": ["e2"]})
    hint = _emotion_peak_hint(program)
    assert hint and hint["source"] == "climax"
    assert 0.0 <= hint["peak_ratio"] <= 1.0
