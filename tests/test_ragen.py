# -*- coding: utf-8 -*-
"""ReGen 三模块回归锚定（p0523 七项修正）。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video import ragen_director as director
from src.agentic_video import ragen_edit as editor
from src.agentic_video import ragen_story as story
from src.agentic_video.ragen_director import ReGenBlocked
from src.agentic_video.ragen_story import (
    INPUT_LEAK_TERMS, strip_to_transferable, validate_story_causal)


def _contract() -> dict:
    return {
        "schema_version": "transfer_contract_v1",
        "theme": "challenge an initial underestimation through visible evidence",
        "narrative_invariants": {
            "initial_belief": {"type": "underestimation",
                               "belief_predicate": "not capable of X",
                               "capability_dimension": "physical competence",
                               "source_of_underestimation":
                                   "free_instantiation_slot"},
            "counter_evidence": {"required_relation": "direct_negation",
                                 "must_target_same_capability_dimension": True,
                                 "must_be_visually_observable": True,
                                 "must_be_decisive": True},
            "reinforcement": {"must_support_revised_belief": True,
                              "may_use_other_events": True},
            "ending": {"function": "humanizing_contrast",
                       "must_not_reverse_revised_belief": True}},
        "editing_grammar": [
            {"role": "situation_setup", "editing_intent": "slow premise setup",
             "content_pattern": ["premise"], "shot_density": "low",
             "transition_policy": "hard_cut", "ordering": "causal",
             "timing_function": "give time to form the belief"},
            {"role": "counter_evidence",
             "editing_intent": "compress one complete event",
             "content_pattern": ["setup", "decisive_action", "visible_result",
                                 "reaction"],
             "shot_density": "medium_high", "transition_policy": "hard_cut",
             "ordering": "causal", "timing_function": "accelerate"},
            {"role": "evidence_expansion",
             "editing_intent": "rapid evidence montage",
             "content_pattern": ["proof_a", "proof_b"],
             "shot_density": "very_high", "transition_policy": "stylized",
             "ordering": "claim_consistency", "timing_function": "density"}],
        "instantiation_slots": ["protagonist_identity", "capability_domain",
                                "venue", "ending_quirk"],
        "rhythm_grammar": {
            "section_duration_ratios": [0.23, 0.53, 0.24],
            "relative_cut_density": ["low", "medium_high", "very_high"],
            "punchline_release": True, "beat_sync_strength": "preferred"},
        "display_format": {"target_canvas": "9:16",
                           "reference_active_ratio": "3:4"},
        "section_roles": ["situation_setup", "counter_evidence",
                          "evidence_expansion"],
    }


def _annex() -> dict:
    return {"sections": [{
        "hook_text_quote": "人们常常觉得失去了双手就会变成一个废人",
        "final_text_quote": "但不会剪脚指甲",
        "reference_specific_fact":
            "白色道服红色护具的主角在跆拳道馆与对手对抗，"
            "高踢击倒对手后微笑走向镜头。"}]}


def _story(*, capability: str = "competitive ability",
           demonstrates: str | None = None,
           ending_cancels: bool = False) -> dict:
    return {
        "story_id": "story_a", "logline": "x",
        "initial_belief": {"claim": "被认为只是业余新手",
                           "capability": capability,
                           "source_of_underestimation": "外表普通"},
        "counter_evidence": {"demonstrated_capability":
                                 demonstrates or capability,
                             "event": "比赛中连续进攻",
                             "outcome": "决定性取胜"},
        "reinforcement": [{"event": "更多能力证据",
                           "supports_corrected_belief": True}],
        "ending": {"statement": "小缺点反差", "function": "人格化收束",
                   "cancels_corrected_belief": ending_cancels},
        "sections": [
            {"role": "situation_setup", "event": "被低估", "emotion": "怀疑"},
            {"role": "counter_evidence", "event": "反证", "emotion": "惊讶"},
            {"role": "evidence_expansion", "event": "扩展", "emotion": "认可"}],
        "emotion_arc": ["doubt", "surprise", "warmth"]}


# ---- 修正 1：因果检查器 ----

def test_causal_checker_accepts_sound_story() -> None:
    assert validate_story_causal(_story(), _contract()) == []


def test_causal_checker_rejects_dimension_mismatch() -> None:
    """用户反例锚死：预期 A 却证明 B（四 role 齐也不放过）。"""
    problems = validate_story_causal(
        _story(capability="cooking skill",
               demonstrates="athletic prowess"),
        _contract())
    assert "counter_evidence_capability_mismatch" in problems


def test_causal_checker_rejects_ending_cancel_and_role_gaps() -> None:
    problems = validate_story_causal(
        _story(ending_cancels=True), _contract())
    assert "ending_cancels_corrected_belief" in problems
    story = _story()
    story["sections"] = story["sections"][:2]
    assert any(p.startswith("section_roles_mismatch")
               for p in validate_story_causal(story, _contract()))


# ---- 修正 2/3：输入门严格、输出门从宽 ----

def test_input_gate_strips_and_blocks_reference_terms() -> None:
    payload = strip_to_transferable(_contract())
    assert "reference_specific" not in json.dumps(payload)
    # 白名单外的字段被剥离（不进 payload）；白名单字段若携带参考词 → 拦
    poisoned = _contract()
    poisoned["core_expression"] = {
        "reference_specific": "跆拳道馆里没有双手的全国冠军"}
    stripped = strip_to_transferable(poisoned)
    assert "core_expression" not in stripped
    poisoned["theme"] = "taekwondo champion proves 全国冠军 wrong"
    with pytest.raises(ReGenBlocked) as excinfo:
        strip_to_transferable(poisoned)
    assert excinfo.value.reason_code == "input_leak_term"


def test_output_gate_allows_independent_domains_blocks_verbatim() -> None:
    """修正 3：模型独立想到"跆拳道"不算泄漏；逐字 quote / 长串照抄算。"""
    independent = _story()
    independent["counter_evidence"]["statement"] = \
        "她在跆拳道比赛中展现决定性水平并取胜"
    assert story._output_violations(
        json.dumps(independent, ensure_ascii=False), _annex()) == []
    copycat = _story()
    copycat["initial_belief"] = {
        "claim": "人们常常觉得失去了双手就会变成一个废人",
        "capability": "capability", "source_of_underestimation": "外表"}
    violations = story._output_violations(
        json.dumps(copycat, ensure_ascii=False), _annex())
    assert any(v.startswith("verbatim_quote") for v in violations)
    fact_copy = _story()
    fact_copy["logline"] = ("白色道服红色护具的主角在跆拳道馆与对手对抗，"
                            "高踢击倒对手后微笑走向镜头。")
    violations = story._output_violations(
        json.dumps(fact_copy, ensure_ascii=False), _annex())
    assert "reference_fact_copy" in violations


# ---- Module 1：契约校验与节奏推导 ----

def test_contract_validator_requires_logic_not_labels() -> None:
    director.validate_contract(_contract(), _contract()["section_roles"])
    weak = _contract()
    weak["narrative_invariants"]["counter_evidence"][
        "required_relation"] = "weak_reference"
    with pytest.raises(ReGenBlocked) as excinfo:
        director.validate_contract(weak, weak["section_roles"])
    assert excinfo.value.reason_code == "invariant_logic_weak"
    # 抽象禁令（p0524）：身体完整性渗进维度 → 拦（三候选全残障题材的根因）
    poisoned = _contract()
    poisoned["narrative_invariants"]["initial_belief"][
        "capability_dimension"] = "body integrity vs capability"
    with pytest.raises(ReGenBlocked) as excinfo:
        director.validate_contract(poisoned, poisoned["section_roles"])
    assert excinfo.value.reason_code == "dimension_not_abstract"
    missing_role = _contract()
    missing_role["editing_grammar"] = missing_role["editing_grammar"][:2]
    with pytest.raises(ReGenBlocked):
        director.validate_contract(missing_role,
                                   missing_role["section_roles"])


def test_rhythm_grammar_from_p0_data() -> None:
    content = {"sections": [
        {"interval": [0.0, 5.0], "content_function": "situation_setup"},
        {"interval": [5.0, 16.6], "content_function": "counter_evidence"},
        {"interval": [16.6, 22.0],
         "content_function": "evidence_expansion"}]}
    ledger = {"scene_detection": {"cut_candidates": [
        {"pts_s": 1.0}, {"pts_s": 7.0}, {"pts_s": 8.0}, {"pts_s": 9.0},
        {"pts_s": 17.0}, {"pts_s": 17.5}, {"pts_s": 18.0}, {"pts_s": 18.5},
        {"pts_s": 19.0}, {"pts_s": 19.5}, {"pts_s": 20.0}, {"pts_s": 20.5},
        {"pts_s": 21.0}, {"pts_s": 21.5}]}}
    rhythm = director.compute_rhythm_grammar(content, ledger)
    assert rhythm["section_duration_ratios"][0] == pytest.approx(
        5.0 / 22.0, abs=1e-3)
    # sec1 1cut/5s=0.2→medium；sec2 3/11.6≈0.26→medium；sec3 11/5.4≈2.0→very_high
    assert rhythm["relative_cut_density"] == ["medium", "medium",
                                              "very_high"]
    assert rhythm["punchline_release"] is True


# ---- Module 3：EDL 发散→选择 ----

def _moments() -> list[dict]:
    rows = []
    for phase, spans in (("premise", [(0.0, 2.0)]),
                         ("setup", [(2.0, 3.0)]),
                         ("decisive_action", [(3.0, 5.5)]),
                         ("visible_result", [(5.5, 7.0)]),
                         ("reaction", [(7.0, 9.0)]),
                         ("proof_a", [(9.0, 10.0)]),
                         ("proof_b", [(10.0, 11.0)])):
        for start, end in spans:
            rows.append({"take_id": "t1", "phase": phase,
                         "start": start, "end": end,
                         "source_interval": [start, end]})
    return rows


def test_edl_candidates_and_deterministic_selection() -> None:
    patterns = [["premise"], ["setup", "decisive_action", "visible_result",
                              "reaction"], ["proof_a", "proof_b"]]
    edls = editor.build_candidate_edls(
        _moments(), patterns, _contract()["rhythm_grammar"])
    assert {edl["strategy"] for edl in edls} == {
        "shortest", "longest", "earliest"}
    selection = editor.select_edl(edls, _contract()["rhythm_grammar"])
    assert selection["selected"]["score"] >= 0
    assert selection["selected"]["moment_count"] == 7
    # 覆盖不齐 → 无合法 EDL → 拦
    assert editor.build_candidate_edls(
        _moments()[:2], patterns, {}) == []
    with pytest.raises(ReGenBlocked) as excinfo:
        editor.select_edl([], {})
    assert excinfo.value.reason_code == "no_legal_edl"


def test_blind_viewer_then_comparator_two_stage() -> None:
    """修正 6：盲看 prompt 不含契约词；comparator 字段校验。"""
    assert "契约" not in editor.BLIND_VIEWER_PROMPT
    assert "theme" not in editor.BLIND_VIEWER_PROMPT.split("JSON")[0]
    blind = {"inferred_theme": "被低估的人用实力打脸",
             "cognition_arc": [], "turning_point_s": 6.0,
             "ending_function": "幽默收束"}

    class FakeRunner:
        def ask(self, prompt, **kwargs):
            assert "盲看观众" in prompt or "结构比对器" in prompt
            return SimpleNamespace(text=json.dumps({
                "theme_consistent": True, "counter_evidence_landed": True,
                "ending_function_matched": True, "issues": [],
                "summary": "ok"}))

    value = editor.compare_to_contract(blind, _contract(),
                                       Path("data/preview"),
                                       runner=FakeRunner())
    assert value["theme_consistent"] is True


# ---- 防泄露：新 prompt 进 spoiler 测试（此处先钉死常量存在与中性） ----

@pytest.mark.parametrize("prompt", [
    director.DIRECTOR_CONTRACT_PROMPT, story.STORY_INSTANTIATION_PROMPT,
    story.ASSET_STORY_PROMPT, editor.BLIND_VIEWER_PROMPT,
    editor.COMPARATOR_PROMPT])
def test_ragen_prompts_do_not_spoil_reference(prompt: str) -> None:
    for spoiler in ("没有双手", "跆拳道", "全国冠军", "废人", "剪脚指甲"):
        assert spoiler not in prompt, (spoiler, prompt[:40])


def test_negation_judge_and_diversity() -> None:
    """p0524：文本 judge 与多样性检查。"""

    class JudgeRunner:
        def __init__(self, verdict):
            self.verdict = verdict
            self.calls = []

        def ask(self, prompt, **kwargs):
            self.calls.append(prompt)
            return SimpleNamespace(text=json.dumps({
                "directly_contradicts": self.verdict,
                "capability_match": self.verdict == "yes",
                "reason": "r"}))

    good = _story()
    assert story.judge_direct_negation(
        good, runner=JudgeRunner("yes")) == "yes"
    bad = JudgeRunner("banana")
    with pytest.raises(ReGenBlocked) as excinfo:
        story.judge_direct_negation(good, runner=bad)
    assert excinfo.value.reason_code == "judge_verdict_invalid"

    same_source = [_story() for _ in range(3)]
    for row in same_source:
        row["story_id"] = "x"
    problems = story.validate_diversity(same_source)
    assert any("homogeneous" in p for p in problems)
    varied = []
    for index, (source, capability) in enumerate([
            ("youth", "cooking"), ("appearance", "chess"),
            ("seniority", "climbing")]):
        row = _story(capability=capability)
        row["initial_belief"]["source_of_underestimation"] = source
        row["story_id"] = f"s{index}"
        varied.append(row)
    assert story.validate_diversity(varied) == []


def test_rhythm_prior_ranges_not_exact() -> None:
    """p0524 修正 7：节奏是先验区间（target+range），不是精确硬约束。"""
    content = {"sections": [
        {"interval": [0.0, 5.0], "content_function": "situation_setup"},
        {"interval": [5.0, 16.6], "content_function": "counter_evidence"},
        {"interval": [16.6, 22.0],
         "content_function": "evidence_expansion"}]}
    rhythm = director.compute_rhythm_grammar(content, {"scene_detection":
                                                       {"cut_candidates":
                                                        []}})
    prior = rhythm["rhythm_prior"]["counter_evidence"]
    assert prior["range"][0] < prior["target"] < prior["range"][1]
    assert "不复制精确秒数" in rhythm["policy"]
