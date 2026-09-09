from __future__ import annotations

from src.agentic_video.discovery import (CONTENT_CATEGORIES,
                                          build_audition_shortlist,
                                          metadata_triage, select_balanced)


def _row(aweme_id: str, title: str, likes: int = 100) -> dict:
    return {"aweme_id": aweme_id, "title": title, "stats": {"likes": likes},
            "extras": {"media_type": 4, "duration_ms": 30000},
            "trend": {"lists": ["1001"], "rank": {"1001": 1}}}


def _audit(row: dict, category: str, *, eligible: bool = True) -> dict:
    return {**row, "category": category, "eligible": eligible,
            "event_count": 3, "has_goal_or_causality": True,
            "has_resolution": True, "evidence_coverage": 0.9,
            "story_clarity": 0.8, "confidence": 0.8,
            "pure_sensory": False}


def test_metadata_triage_rejects_pure_technique_but_keeps_story_candidates():
    rejected = metadata_triage(_row("1", "纯卡点技术流 运镜教学"))
    accepted = metadata_triage(_row("2", "应急迫降 飞机上的暖心瞬间"))
    assert rejected["eligible"] is False
    assert "pure_edit_or_tutorial" in rejected["reasons"]
    assert accepted["eligible"] is True


def test_balanced_selection_never_fills_with_ineligible_rows():
    rows = []
    for category in CONTENT_CATEGORIES:
        rows.extend(_audit(_row(f"{category}-{i}", "内容", likes=100 - i), category)
                    for i in range(3))
    rows.append(_audit(_row("weak", "弱内容", likes=9999), CONTENT_CATEGORIES[0],
                       eligible=False))
    selected, gaps = select_balanced(rows, per_category=4)
    assert len(selected) == 9
    assert all(row["eligible"] for row in selected)
    assert gaps == {category: 1 for category in CONTENT_CATEGORIES}


def test_content_gate_is_not_replaced_by_popularity():
    strong = _audit(_row("strong", "故事", likes=10), "real_story")
    weak = _audit(_row("weak", "热门", likes=1_000_000), "real_story")
    weak["event_count"] = 1
    selected, _ = select_balanced([weak, strong], per_category=1)
    assert [row["aweme_id"] for row in selected] == ["strong"]


def test_shortlist_keeps_unclassified_titles_for_semantic_audition():
    rows = [
        {"aweme_id": "hinted", "eligible": True, "category_hint": "real_story",
         "appeared_in_lists": ["hot"], "ranks": {"hot": 2}},
        {"aweme_id": "unknown", "eligible": True, "category_hint": None,
         "appeared_in_lists": ["hot"], "ranks": {"hot": 1}},
    ]
    shortlisted = build_audition_shortlist(rows, per_category=1, unclassified=1)
    assert {row["aweme_id"] for row in shortlisted} == {"hinted", "unknown"}
