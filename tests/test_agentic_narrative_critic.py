from __future__ import annotations

from src.agentic_video.narrative_critic import (apply_story_patches,
                                                 preflight_story_faults,
                                                 run_narrative_critic,
                                                 score_comprehension_answers,
                                                 validate_story_patch)
from src.agentic_video.story_planner import build_story_plan
from tests.test_agentic_narrative import valid_program
from tests.test_agentic_story_planner import _candidate


def _plan() -> dict:
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "resolution", "person", start=9, event_id="e3")]
    return build_story_plan(valid_program(), rows, theme="守护与牺牲",
                            library="guimie", target_duration_s=45.0)


def test_story_patch_allowlist():
    plan = _plan()
    assert validate_story_patch(plan, {"op": "replace", "path": "/slots/0/source",
                                       "value": plan["slots"][0]["source"]}) == []
    assert validate_story_patch(plan, {"op": "replace", "path": "/theme",
                                       "value": "篡改"})


def test_invalid_story_patch_is_rejected():
    plan = _plan()
    patched, audit = apply_story_patches(plan, [
        {"op": "replace", "path": "/slots/0/source/video", "value": ""},
        {"op": "remove", "path": "/slots/0"},
    ])
    assert patched == plan
    assert all(item["status"] == "rejected" for item in audit)


def test_comprehension_no_longer_self_grades():
    """V3 P4：理解题对错不再由看过槽位事实的模型自评（C2 实锤 5/5 vs
    coherence 0 自相矛盾）——只统计覆盖（有答案+有证据），passes 恒 False，
    通过与否由独立盲看协议决定。"""
    result = score_comprehension_answers([
        {"question_id": i, "answer": f"答{i}", "evidence": "可见画面" if i < 4 else ""}
        for i in range(5)])
    assert result["covered"] == 4
    assert result["passes"] is False


def test_preflight_story_faults_covers_causal_and_execution_failures():
    plan = _plan()
    plan["slots"][1]["source"]["entity_ids"] = ["other"]
    plan["slots"][1]["source"]["dialogue"] = [{"start_s": 4.0, "end_s": 40.0}]
    plan["slots"][1]["target_interval"] = [15.0, 20.0]
    plan["slots"][2]["role"] = "conflict"
    plan["slots"][2]["status"] = "unsupported"
    for slot in plan["slots"]:
        if slot["role"] == "resolution":
            slot["status"] = "unsupported"
    faults = preflight_story_faults(plan)
    assert {fault["type"] for fault in faults} >= {
        "entity_switch", "missing_resolution", "missing_cause",
        "subtitle_timing", "dialogue_cut"}


def test_narrative_critic_prompt_keeps_literal_json_braces():
    class Runner:
        def watch(self, _video, prompt, **_kwargs):
            assert '"theme_relevance":0.0' in prompt
            assert "{questions}" not in prompt
            return type("Answer", (), {"text": '{"answers":[],"patches":[]}',
                                        "elapsed_s": 0.1})()

    critique = run_narrative_critic("rendered.mp4", valid_program(), _plan(),
                                    runner=Runner())
    assert critique["comprehension"]["total"] == 5


def test_critic_prompt_no_longer_carries_full_plan():
    """V3 P4：critic 只拿槽位事实摘要——Narrative Program/Story Plan 全文是
    自报答对与参考污染的通道（C2：comprehension 5/5 vs coherence 0、edit
    critic 幻觉跆拳道画面）。"""
    from src.agentic_video.narrative_critic import (NARRATIVE_CRITIC_PROMPT,
                                                    _slot_facts_summary)
    assert "{narrative}" not in NARRATIVE_CRITIC_PROMPT
    assert "{story_plan}" not in NARRATIVE_CRITIC_PROMPT
    assert "{slot_facts}" in NARRATIVE_CRITIC_PROMPT
    summary = _slot_facts_summary(_plan(), valid_program())
    assert len(summary) <= 2000
    assert "槽0" in summary


def test_edit_critic_skeleton_and_hallucination_filter():
    from src.agentic_video.critic_v2 import (filter_critic_issues, recipe_skeleton,
                                             reference_terms)
    recipe = {"reference": {"duration_s": 25.3}, "operations": [
        {"type": "text_overlay", "interval": [1.0, 2.0], "status": "supported",
         "params": {"text": "跆拳道全国冠军"}}]}
    skeleton = recipe_skeleton(recipe)
    assert "跆拳道" not in skeleton                     # 参考文字层不再外泄
    assert "text_overlay" in skeleton
    narrative = {"intent": {"protagonist": "一名身穿跆拳道服的女性，未见双手"}}
    refs = reference_terms(narrative, recipe)
    assert "跆拳" in refs                                # 参考词被切成可匹配的二元组
    critique = {"issues": [
        {"aspect": "render_integrity", "evidence": "成片出现跆拳道比赛画面"},
        {"aspect": "render_integrity", "evidence": "中段小黑与敌人对峙画面切换突兀"},
        {"aspect": "rhythm_continuity", "evidence": "字幕与画面不同步"}]}
    filtered = filter_critic_issues(critique, ["小黑"], reference_tokens=refs)
    assert len(filtered["issues"]) == 2                 # 跆拳道条被弃
    assert filtered["hallucinated_issues"][0]["dropped_reason"] == "reference_contamination"
    assert filtered["hallucinated_issues"][0]["evidence"].startswith("成片出现跆拳道")
