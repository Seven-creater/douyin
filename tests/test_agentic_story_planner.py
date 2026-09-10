from __future__ import annotations

from src.agentic_video.narrative import ARC_ROLES
from src.agentic_video.story_planner import (_expand_thin_arc, build_story_plan,
                                             rank_story_path, validate_story_plan)
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
