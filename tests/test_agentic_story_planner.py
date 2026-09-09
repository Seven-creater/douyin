from __future__ import annotations

from src.agentic_video.story_planner import (build_story_plan, rank_story_path,
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
