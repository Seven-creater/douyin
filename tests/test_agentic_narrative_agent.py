from __future__ import annotations

import json

from src.agentic_video.narrative_agent import (NarrativeBudget,
                                                parse_narrative_program,
                                                plan_narrative_windows)


def test_narrative_windows_cover_duration_with_budget():
    windows = plan_narrative_windows(60.0, max_windows=6, target_window_s=12)
    assert len(windows) == 5
    assert windows[0].start == 0
    assert windows[-1].end == 60
    assert all(a.end == b.start for a, b in zip(windows, windows[1:]))


def test_long_video_windows_remain_short_and_span_the_source():
    windows = plan_narrative_windows(9_295.0, max_windows=24, target_window_s=12)
    assert len(windows) == 24
    assert windows[0].start == 0
    assert windows[-1].end == 9_295.0
    assert all(abs((window.end - window.start) - 12.0) <= 0.002 for window in windows)
    assert all(left.end < right.start for left, right in zip(windows, windows[1:]))


def test_parser_downgrades_unsupported_claim_without_evidence():
    raw = json.dumps({
        "intent": {"topic": "守护", "message": "牺牲", "content_type": "screen_story",
                   "evidence": [], "confidence": 0.9, "status": "supported"},
        "entities": [], "events": [], "causal_links": [], "arc": [],
        "utterances": [], "emotion_curve": [], "uncertainties": [],
    }, ensure_ascii=False)
    program = parse_narrative_program(
        raw, reference_id="r", reference_uri="r.mp4", sha256="a" * 64,
        duration_s=10, fps=24, model="fake", tool_calls=[])
    assert program["intent"]["status"] == "uncertain"
    assert program["status"] == "unsupported"
    assert program["evidence"] == []


def test_default_narrative_budget_matches_contract():
    budget = NarrativeBudget()
    assert (budget.max_initial_windows, budget.max_rounds,
            budget.max_refinement_windows) == (24, 2, 12)
