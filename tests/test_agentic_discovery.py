from __future__ import annotations

import json
from types import SimpleNamespace

from src.agentic_video.discovery import (CONTENT_CATEGORIES,
                                          SEMANTIC_AUDIT_PROMPT,
                                          build_audition_shortlist,
                                          has_temporal_story_coverage,
                                          load_rolling_candidates, metadata_triage,
                                          parse_semantic_audit,
                                          select_balanced)


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


def test_semantic_audit_prompt_replaces_only_context_placeholder():
    prompt = SEMANTIC_AUDIT_PROMPT.replace("{context}", "证据")
    assert "\"category\"" in prompt
    assert "证据" in prompt


def test_semantic_audit_counts_only_located_evidence_events():
    row = metadata_triage(_row("event-mismatch", "暖心救助"))
    audit = parse_semantic_audit(json.dumps({
        "category": "real_story", "event_count": 3,
        "events": [{"start_s": 0, "end_s": 1, "evidence_sources": ["frame"]},
                    {"start_s": 2, "end_s": 3, "evidence_sources": ["ocr"]}],
        "has_goal_or_causality": True, "has_resolution": True,
        "evidence_coverage": 1.0, "story_clarity": 1.0,
        "confidence": 1.0, "pure_sensory": False,
    }), row)
    assert audit["event_count"] == 2
    assert audit["eligible"] is False
    assert "event_count_mismatch" in audit["rejection_reasons"]


def test_content_gate_rejects_three_events_confined_to_video_opening():
    row = _audit(_row("front-loaded", "暖心故事"), "real_story")
    row["duration_s"] = 30
    row["events"] = [
        {"start_s": 0, "end_s": 1},
        {"start_s": 1, "end_s": 2},
        {"start_s": 2, "end_s": 3},
    ]

    assert has_temporal_story_coverage(row) is False
    selected, _ = select_balanced([row], per_category=1)
    assert selected == []


def test_rolling_candidates_keep_newest_signed_download_url(tmp_path):
    def payload(url: str) -> dict:
        return {"data": {"objs": [{
            "item_id": "123", "item_title": "真实故事", "nick_name": "作者",
            "item_url": url, "like_cnt": 10, "play_cnt": 100,
            "item_duration": 30_000, "media_type": 4, "image_cnt": 0,
        }]}}

    (tmp_path / "2026-09-07_1001.json").write_text(
        json.dumps(payload("https://cdn/old")), encoding="utf-8")
    (tmp_path / "2026-09-09_1001.json").write_text(
        json.dumps(payload("https://cdn/new")), encoding="utf-8")
    cfg = SimpleNamespace(paths=SimpleNamespace(raw_dir=tmp_path),
                          wellbyte=SimpleNamespace(request_params={"date_window": 24}))
    rows, _ = load_rolling_candidates(cfg, end_date="2026-09-09", days=3)
    assert rows[0]["download_url"] == "https://cdn/new"
    assert rows[0]["download_url_observed_date"] == "2026-09-09"
    assert rows[0]["observed_dates"] == ["2026-09-09", "2026-09-07"]
