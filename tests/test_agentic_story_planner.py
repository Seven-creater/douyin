from __future__ import annotations

from src.agentic_video.narrative import ARC_ROLES
from src.agentic_video.story_planner import (_arc_query, _build_specs,
                                             _emotion_peak_hint,
                                             _hard_continuity,
                                             build_story_plan, rank_story_path,
                                             slot_need_spec,
                                             story_plan_execution_inputs,
                                             validate_story_plan)
from src.agentic_video.narrative_form import (compile_form_need,
                                              infer_narrative_form,
                                              resolve_slot_sequence)
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


def test_thin_arc_maps_to_form_template_never_synthesizes_conflict():
    """V3 P1：薄参考弧（hook/consequence）不再被 _expand_thin_arc 强补 conflict
    （lxh_p4_C2 选错故事语法的病灶）——Form 模板按实际表达结构给声明式槽序列。"""
    program = valid_program()
    program["arc"] = [program["arc"][0]]                     # 只留 hook
    program["utterances"] = [{"id": f"u{i}", "interval": [0.0, 2.0], "speaker_id": None,
                              "original": "x", "translation_zh": "x",
                              "evidence": [], "confidence": 0.9, "status": "uncertain"}
                             for i in range(4)]              # 文字轨密 → assertion 型
    slots, form_name = resolve_slot_sequence(program)
    assert form_name == "assertion_visual_payoff"
    roles = [slot["role"] for slot in slots]
    assert roles == ["hook", "context", "consequence", "resolution"]
    assert "conflict" not in roles                            # 绝不合成冲突
    assert all(slot["required"] for slot in slots)            # 声明式必选
    assert slots[0]["form_function"] == "premise"
    assert slots[1]["form_function"] == "counter_evidence"


def test_form_needs_carry_zero_reference_facts():
    """form_function 编译的需求零参考事实（V3 P1：Reference Narrative 不直接
    成为 Target Narrative）。"""
    need = compile_form_need("counter_evidence")
    assert "做得到" in need["need"]
    for value in (need["need"], *need["must_have"], *need["must_not"]):
        assert "跆拳道" not in value and "双手" not in value
    assert infer_narrative_form(valid_program()) == "classic_arc"   # 有冲突角色


def test_thin_reference_arc_renders_form_slots_not_single_slot():
    """2026-09-10 首跑病灶回归：arc=[hook] 不得把 60s 灌进 1 个槽；
    V3 后由 Form 模板给 4 槽（不再是自动补弧的 3 槽）。"""
    program = valid_program()
    program["arc"] = [program["arc"][0]]
    program["utterances"] = [{"id": f"u{i}", "interval": [0.0, 2.0], "speaker_id": None,
                              "original": "x", "translation_zh": "x",
                              "evidence": [], "confidence": 0.9, "status": "uncertain"}
                             for i in range(4)]
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "context", "person", start=4, event_id="e2"),
            _candidate(2, "consequence", "person", start=8, event_id="e2"),
            _candidate(3, "resolution", "person", start=12, event_id="e3")]
    plan = build_story_plan(program, rows, theme="鬼灭高燃战斗", library="guimie",
                            target_duration_s=60.0)
    assert len(plan["slots"]) == 4
    assert "arc_expanded" not in plan                          # 补弧机制已删
    assert plan["narrative_form"]["name"] == "assertion_visual_payoff"
    assert validate_story_plan(plan) == []
    order = {role: idx for idx, role in enumerate(ARC_ROLES)}
    roles = [slot["role"] for slot in plan["slots"]]
    assert roles == sorted(roles, key=order.get)
    assert all(slot["target_interval"][1] - slot["target_interval"][0] < 60.0
               for slot in plan["slots"])


_WIDE_DIALOGUE = {"hook": "处境与张力", "conflict": "冲突与危险", "choice": "关键选择与行动",
                  "climax": "高潮峰值", "resolution": "收束", "consequence": "结果",
                  "context": "关系背景"}


def _wide_candidate(idx: int, role: str, start: float) -> dict:
    row = _candidate(idx, role, "person", start=start, event_id=f"e{idx}")
    row.update({"source_end_s": start + 45, "duration_s": 45,
                "dialogue": [
                    {"start_s": start + 1, "end_s": start + 3,
                     "original": "前", "translation_zh": _WIDE_DIALOGUE[role]},
                    {"start_s": start + 38, "end_s": start + 40,
                     "original": "後", "translation_zh": "会越界的对白"},
                ]})
    return row


def test_oversized_event_is_trimmed_to_slot_not_dropped():
    """2026-09-10 v4 黑屏 + V4 B：45s 合并段不再一票否决；有相关对白锚时
    窗口跟着对白走（anchor=dialogue），只留窗内完整对白。"""
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
        assert source["anchor"] == "dialogue"                # 对白锚定而非起点盲切
        assert [line["translation_zh"] for line in source["dialogue"]] == [_WIDE_DIALOGUE[slot["role"]]]


def test_coarse_window_is_never_blind_cut_from_event_start():
    """V4 B3（外审三轮必改②）：无对白锚且 > 2×budget 的粗窗禁止从起点盲切
    ——anchor=coarse、保留完整区间交 localize 步骤（localize or reject）。"""
    program = valid_program()
    program["arc"] = program["arc"][:3]
    rows = []
    for idx, (role, start) in enumerate([("hook", 5040), ("conflict", 5370),
                                         ("choice", 7819)]):
        row = _candidate(idx, role, "person", start=start, event_id=f"e{idx}")
        row.update({"source_end_s": start + 45, "duration_s": 45,
                    "dialogue": [                      # 与需求零重合的对白 → 无锚
                        {"start_s": start + 1, "end_s": start + 3,
                         "original": "闲", "translation_zh": "完全无关的闲聊"}]})
        rows.append(row)
    plan = build_story_plan(program, rows, theme="鬼灭高燃战斗", library="guimie",
                            target_duration_s=60.0)
    for slot in plan["slots"]:
        source = slot["source"]
        assert source["anchor"] == "coarse"                  # 不盲切
        assert source["end_s"] - source["start_s"] > 40.0    # 完整区间留给 localize


def test_select_evidence_window_anchors_single_shot_on_dialogue():
    """V4 B：单镜头超预算时在镜头内按最相关对白行选子窗，锚行完整落入。"""
    from src.agentic_video.story_planner import _select_evidence_window
    member = [{"shot_idx": 0, "start_s": 2940.0, "end_s": 2985.0,
               "caption": "餐桌旁众人交谈",
               "dialogue": [
                   {"start_s": 2941.0, "end_s": 2944.0,
                    "original": "a", "translation_zh": "无关的一句"},
                   {"start_s": 2977.5, "end_s": 2985.0,
                    "original": "b", "translation_zh": "人和妖很难定义好坏"}],
               }]
    window = _select_evidence_window(member, budget_s=7.33,
                                     query_text="人和妖的关系 妖灵与人类")
    assert window["anchor"] == "dialogue"
    assert window["start_s"] <= 2977.5                       # 锚行完整落入
    assert window["end_s"] >= 2985.0 - 1e-6
    # 行比预算长时整行优先（≤1s 轻微超预算由渲染 -t 收口）
    assert window["end_s"] - window["start_s"] <= 7.33 + 1.0


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


def test_slot_need_spec_compiles_bindings_without_reference_facts():
    """P2 + 2026-09-11 C2 验证器实锤：need 只含可迁移结构，参考片六问的具体事实
    （protagonist/problem 原文）不得灌入——否则跨库变成「找穿跆拳道服的角色」。"""
    program = valid_program()
    program["intent"].update({"protagonist": "救助者", "problem": "小猫被困",
                              "motivation": "unknown", "outcome": "not_applicable"})
    arc = program["arc"]
    spec = slot_need_spec(program, arc[0], 0)
    assert "救助者" not in spec["need"] and "小猫被困" not in spec["need"]
    assert "处境" in spec["need"]                              # 结构性需求保留
    assert set(spec["entity_bindings"]) == {"A", "B"}          # 绑定照常从 participants 推导
    assert spec["entity_bindings"]["A"]["reference_name"] == "救助者"
    conflict_spec = slot_need_spec(program,
                                   next(s for s in arc if s["role"] == "conflict"), 1)
    assert conflict_spec["required"] is True
    assert conflict_spec["must_have"] and conflict_spec["must_not"]
    choice_spec = slot_need_spec(program,
                                 next(s for s in arc if s["role"] == "choice"), 2)
    assert choice_spec["required"] is False                    # 可选槽


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
    unsupported，plan 如实报未完成；绝不放宽约束凑满。
    V3 起厚弧（≥3 段）直接走参考弧——本例用厚弧验证（薄弧由 Form 模板接管）。"""
    program = valid_program()
    program["arc"] = [
        {"role": "hook", "event_ids": ["e1"]},
        {"role": "conflict", "event_ids": ["e2"]},            # 与 hook 共享 person/cat
        {"role": "resolution", "event_ids": ["e3"]},
    ]
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1", score=0.9),
            _candidate(1, "conflict", "stranger", start=4, event_id="e2", score=0.9),
            _candidate(2, "resolution", "person", start=8, event_id="e3", score=0.9)]
    plan = build_story_plan(program, rows, theme="救助", library="guimie",
                            target_duration_s=45.0)
    conflict_slot = plan["slots"][1]
    assert conflict_slot["status"] == "unsupported"
    assert conflict_slot["reason"] in {"library_insufficient", "continuity_infeasible"}
    assert 1 in plan["required_unsupported"]
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


def test_library_scope_matches_multiple_film_sources():
    """2026-09-11 罗小黑库：两部电影（luoxiaohei1/2）共用一个检索池
    （--library luoxiaohei1,luoxiaohei2），别名 luoxiaohei 不得误伤。"""
    from src.agentic_video.story_planner import _row_in_library
    rows = [
        {"source": "luoxiaohei1", "video_stem": "luoxiaohei1__narrative"},
        {"source": "luoxiaohei2", "video_stem": "luoxiaohei2__narrative"},
        {"source": "guimie", "video_stem": "guimie__narrative"},
    ]
    assert _row_in_library(rows[0], "luoxiaohei1,luoxiaohei2")
    assert _row_in_library(rows[1], "luoxiaohei1,luoxiaohei2")
    assert not _row_in_library(rows[2], "luoxiaohei1,luoxiaohei2")
    assert not _row_in_library(rows[1], "luoxiaohei1")   # 单源不含片2
    assert _row_in_library(rows[2], "guimie")


def test_entity_contract_locks_protagonist_against_higher_score_impostor():
    """V3 P1/P3 核心断言（外审二轮）：identity constraint > semantic similarity。
    槽0 选小黑后主角锁定；槽1 即使"无限"候选分数更高也不能上位——
    lxh_p4_C2 三槽三主角的病灶直接回归测试。"""
    program = valid_program()
    program["arc"] = [
        {"role": "hook", "event_ids": ["e1"]},
        {"role": "conflict", "event_ids": ["e2"]},
        {"role": "resolution", "event_ids": ["e3"]},
    ]
    xiaohei = {"row_idx": 0, "video": "movie.mp4", "video_stem": "luoxiaohei1__narrative",
               "shot_idx": 0, "source_start_s": 100, "source_end_s": 103, "duration_s": 3,
               "caption": "hook", "story_role": "hook", "entity_ids": ["e5"],
               "entity_names": ["小黑"], "event_id": "e1", "semantic_score": 0.85,
               "dialogue": []}
    impostor = {**xiaohei, "row_idx": 1, "shot_idx": 1, "source_start_s": 200,
                "source_end_s": 203, "entity_ids": ["e9"], "entity_names": ["无限"],
                "story_role": "conflict", "caption": "conflict", "event_id": "e2",
                "semantic_score": 0.99}                    # 分数更高但不是主角
    xiaohei_conflict = {**xiaohei, "row_idx": 2, "shot_idx": 2, "source_start_s": 300,
                        "source_end_s": 303, "story_role": "conflict",
                        "caption": "conflict", "event_id": "e2",
                        "semantic_score": 0.5}
    resolution = {**xiaohei, "row_idx": 3, "shot_idx": 3, "source_start_s": 400,
                  "source_end_s": 403, "story_role": "resolution",
                  "caption": "resolution", "event_id": "e3"}
    plan = _plan_with_groups(program, [[xiaohei], [impostor, xiaohei_conflict],
                                       [resolution]])
    contract = plan["entity_contract"]
    assert contract["protagonist"] == "char:xiaohei"      # 注册表归一（别名 小黑）
    assert contract["relaxed"] is False
    picked_names = [slot["source"].get("entity_names") for slot in plan["slots"]]
    assert "无限" not in {name for names in picked_names if names for name in names}
    assert all(slot["status"] == "supported" for slot in plan["slots"])


def _plan_with_groups(program, groups):
    from src.agentic_video.story_planner import _assemble_story_plan
    return _assemble_story_plan(program, groups, theme="守护",
                                library="luoxiaohei1", target_duration_s=45.0)


def test_entity_registry_unifies_cross_film_entities():
    """V3 P2：两部电影各自的 entity_id 经注册表归一到同一 canonical id——
    画面还是小黑，系统不再以为换人了。"""
    from src.library.entity_registry import (load_entity_registry,
                                             row_identity_keys)
    registry = {"char:xiaohei": {"aliases": ["小黑"],
                                 "source_entities": ["luoxiaohei1/entity_005",
                                                     "luoxiaohei2/entity_017"]}}
    film1 = {"video_stem": "luoxiaohei1__narrative", "entity_ids": ["entity_005"],
             "entity_names": ["小黑"]}
    film2 = {"video_stem": "luoxiaohei2__narrative", "entity_ids": ["entity_017"],
             "entity_names": ["小黑"]}
    keys1, keys2 = row_identity_keys(film1, registry), row_identity_keys(film2, registry)
    assert "char:xiaohei" in keys1 and "char:xiaohei" in keys2
    # 未注册实体回退源内键（同片约束仍成立）
    unknown = {"video_stem": "luoxiaohei1__narrative", "entity_ids": ["entity_042"],
               "entity_names": ["神秘角色"]}
    assert "id:luoxiaohei1/entity_042" in row_identity_keys(unknown, registry)
    # 空注册表不崩
    assert row_identity_keys(film1, {})


def test_identity_keys_are_window_scoped_and_alias_fuzzy():
    """V3 晨跑实锤 + V4 外审三轮修订：
    ①e001 是窗口级编号——窗1 的 e001（黑发少年）≠ 窗7 的 e001（小女孩），
    id 键必须带窗口，否则三槽三主角假等价、det 假绿；
    ②外观描述（黑发持刀）不产生 canonical 键——V3_C3 假绿的直接根因
    （外审反例：删该别名后必选槽主角存在率 100%→0%）；跨片同人的正路是
    身份别名/source_entities/supported bindings；
    ③身份别名子串仍接住同真名的标注变体；proposed binding 不进硬约束。"""
    from src.library.entity_registry import row_identity_keys
    registry = {"char:wuxian": {"aliases": ["无限"],
                                "appearance_aliases": ["黑发持刀"],
                                "source_entities": ["luoxiaohei1/w1/e001",
                                                    "luoxiaohei2/w3/e017"]}}
    w1 = {"video_stem": "luoxiaohei1__narrative", "window_idx": 1,
          "entity_ids": ["e001"], "entity_names": ["黑发持刀少年"]}
    w7 = {"video_stem": "luoxiaohei1__narrative", "window_idx": 7,
          "entity_ids": ["e001"], "entity_names": ["小女孩"]}
    k1, k7 = row_identity_keys(w1, registry), row_identity_keys(w7, registry)
    assert "id:luoxiaohei1/w1/e001" in k1 and "id:luoxiaohei1/w7/e001" in k7
    assert k1 & k7 == set()                        # 窗口隔离：不再假等价
    # 外观别名不进等价表：靠"黑发持刀"绝不能证明是 char:wuxian
    assert "char:wuxian" not in row_identity_keys(
        {"video_stem": "luoxiaohei1__narrative", "window_idx": 2,
         "entity_ids": ["e009"], "entity_names": ["黑发持刀少年"]}, registry)
    # source_entities 是跨片同人的正路（film1/e001 与 film2/e017 同 canonical）
    assert "char:wuxian" in k1                     # 经 source_entities
    film2 = {"video_stem": "luoxiaohei2__narrative", "window_idx": 3,
             "entity_ids": ["e017"], "entity_names": ["黑发持刀男子"]}
    assert "char:wuxian" in row_identity_keys(film2, registry)
    # 身份别名子串：≥3 字的身份别名接住标注变体；2 字别名只保精确命中
    # （len≥3 门槛是防误匹配的设计底线——「小黑」in「小黑猫」这种不能放开）
    assert "char:wuxian" in row_identity_keys(
        {"video_stem": "luoxiaohei2__narrative", "window_idx": 5,
         "entity_ids": ["e021"], "entity_names": ["无限"]}, registry)
    registry3 = {"char:xiaohei": {"aliases": ["罗小黑"]}}
    assert "char:xiaohei" in row_identity_keys(
        {"video_stem": "luoxiaohei1__narrative", "window_idx": 5,
         "entity_ids": ["e021"], "entity_names": ["罗小黑少年"]}, registry3)
    # proposed binding 不进硬约束（外审必改：知识包猜测不能当身份真值）
    assert "char:wuxian" not in row_identity_keys(
        {"video_stem": "luoxiaohei2__narrative", "window_idx": 5,
         "entity_ids": ["e030"],
         "bindings": [{"local_entity_id": "e030",
                       "canonical_entity_id": "char:wuxian",
                       "binding_status": "proposed", "binding_confidence": 0.82}]},
        registry)
    assert "char:wuxian" in row_identity_keys(
        {"video_stem": "luoxiaohei2__narrative", "window_idx": 6,
         "entity_ids": ["e031"],
         "bindings": [{"local_entity_id": "e031",
                       "canonical_entity_id": "char:wuxian",
                       "binding_status": "supported", "binding_confidence": 0.86}]},
        registry)


def test_evidence_window_follows_relevant_shot_not_event_start():
    """V3 P2 断言（lxh_p4_C2 槽1）：事件聚合 caption 说"奔跑"，相关镜头在事件
    后段——裁剪窗口必须跟着证据走，不再从事件起点盲切 7.3s 放出开头的施法段；
    槽 caption 只保留被裁入镜头。"""
    from src.agentic_video.story_planner import _select_evidence_window
    member_shots = [
        {"shot_idx": 0, "start_s": 945.0, "end_s": 952.0,
         "caption": "紫发角色在墓园伸手施法"},
        {"shot_idx": 1, "start_s": 952.0, "end_s": 956.5,
         "caption": "少年在森林中奔跑追赶"},
        {"shot_idx": 2, "start_s": 956.5, "end_s": 960.0,
         "caption": "奔跑中回头张望"},
    ]
    window = _select_evidence_window(member_shots, budget_s=8.0,
                                     query_text="叙事需求：主角奔跑的动作；需要：奔跑可见")
    assert window is not None
    assert window["start_s"] == 952.0                    # 跟着奔跑镜头，不是 945
    assert window["end_s"] == 960.0
    assert "奔跑" in window["caption"] and "施法" not in window["caption"]


def test_merged_events_carry_member_shots_and_preceding():
    """V3 P2：事件行带成员镜头明细（供证据窗口裁剪）与同窗时序前驱
    （替代普遍为空的库内因果图，不造假）。"""
    from src.agentic_video.story_planner import merge_event_candidates
    rows = [
        {"video": "m.mp4", "video_stem": "s", "window_idx": 0, "event_id": "eA",
         "story_role": "conflict", "source_start_s": 100, "source_end_s": 104,
         "duration_s": 4, "caption": "对峙", "entity_ids": ["a"], "entity_names": ["甲"],
         "dialogue": [], "semantic_score": 0.5},
        {"video": "m.mp4", "video_stem": "s", "window_idx": 0, "event_id": "eB",
         "story_role": "climax", "source_start_s": 104, "source_end_s": 108,
         "duration_s": 4, "caption": "爆发", "entity_ids": ["a"], "entity_names": ["甲"],
         "dialogue": [], "semantic_score": 0.6},
    ]
    merged = merge_event_candidates(rows)
    by_event = {row["event_id"]: row for row in merged}
    assert len(by_event["eA"]["member_shots"]) == 1
    assert by_event["eB"]["preceding_event_ids"] == ["eA"]
    assert by_event["eA"]["preceding_event_ids"] == []


def test_compile_form_need_enum_first_then_legacy_keywords():
    """V4 C（外审三轮）：固定 enum 精确命中优先；旧自由中文句走 legacy
    关键词归一——badcase 两条原句必须命中且 need 不含输入句原文（参考
    事实红线）；不可映射回退空（上层走角色模板）。"""
    from src.agentic_video.narrative_form import (compile_form_need,
                                                  normalize_form_function)
    # enum 精确命中
    assert compile_form_need("counter_evidence")["must_have"] \
        == ["主角的关键行动可见", "主角在场"]
    # badcase 原句一：「展示主角在专业领域的成就，强化其能力形象」
    spec = compile_form_need("展示主角在专业领域的成就，强化其能力形象")
    assert spec == compile_form_need("evidence_expansion")      # 归一到模板，零原文透传
    # badcase 原句二：「列举生活技能与局限形成对比」——强扩展信号压过 对比/局限
    assert normalize_form_function("列举生活技能与局限形成对比") \
        == "evidence_expansion"
    assert normalize_form_function("主角以行动反驳刻板印象") == "counter_evidence"
    assert normalize_form_function("收束整个故事的情绪") == "payoff"
    assert normalize_form_function("完全不沾边的句子") is None
    assert compile_form_need("完全不沾边的句子") == \
        {"need": "", "must_have": [], "must_not": [], "evidence_mode": "visual"}


def test_thick_arc_prefers_canonical_form_enum_over_free_text():
    """厚弧段带 enum（V4 双字段）时直接消费；只给自由中文时走归一器，
    不再把整句当 form_function 传给 compile_form_need。"""
    from src.agentic_video.narrative_form import resolve_slot_sequence
    narrative = {
        "arc": [
            {"role": "hook", "event_ids": [],
             "form_function": "premise", "function": "开场断言"},
            {"role": "context", "event_ids": [],
             "function": "展示主角在专业领域的成就，强化其能力形象"},
            {"role": "resolution", "event_ids": [],
             "function": "轻细节收尾"},
        ],
        "events": [], "entities": [], "causal_links": [], "utterances": [],
    }
    slots, _form = resolve_slot_sequence(narrative)
    assert slots[0]["form_function"] == "premise"
    assert slots[1]["form_function"] == "evidence_expansion"
    assert slots[2]["form_function"] == "payoff"
