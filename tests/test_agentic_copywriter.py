from __future__ import annotations

from src.agentic_video.copywriter import build_copy_cues
from tests.test_agentic_narrative import valid_program
from tests.test_agentic_story_planner import _candidate


class _BrokenRunner:
    """返回垃圾文本的假 runner：LLM 层任何失败都必须落模板，不阻塞渲染。"""

    def ask(self, prompt, *, max_new_tokens=None):
        class _A:
            text = "这不是 JSON {{{"
        return _A()


class _GoodRunner:
    def ask(self, prompt, *, max_new_tokens=None):
        class _A:
            text = ('{"hook": "人们常常觉得，刀只是杀人的工具", '
                    '"cards": ["以脚代手的守护", "全集中呼吸", "拼死一战"], '
                    '"punchline": "但他从不是为赢而战"}')
        return _A()


def _plan(target=22.0, roles=("hook", "context", "conflict", "consequence")):
    from src.agentic_video.story_planner import build_story_plan
    program = valid_program()
    from src.agentic_video.narrative import ARC_ROLES
    events = [e["id"] for e in program["events"]]
    program["arc"] = [{"role": role, "event_ids": [events[i % len(events)]]}
                      for i, role in enumerate(roles)
                      if role in ARC_ROLES]
    rows = [_candidate(i, role, "person", start=i * 4, event_id=f"e{i}")
            for i, role in enumerate(("hook", "context", "conflict", "resolution"))]
    return build_story_plan(program, rows, theme="鬼灭炭治郎拼死守护", library="guimie",
                            target_duration_s=target)


def test_template_tier_is_deterministic_and_7682_shaped():
    plan = _plan()
    copy = build_copy_cues(plan, runner=None)
    again = build_copy_cues(plan, runner=None)
    assert copy == again and copy["source"] == "template"
    assert copy["audio_mode"] == "bgm"
    kinds = [cue["kind"] for cue in copy["cues"]]
    assert kinds[0] == "hook_line" and kinds[-1] == "punchline"
    assert kinds.count("info_card") >= 1
    hook = copy["cues"][0]
    assert hook["start_s"] == 0.15 and hook["end_s"] <= 4.2


def test_card_burst_timing_and_punchline_pinned_to_tail():
    plan = _plan(target=22.0)
    total = plan["target_duration_s"]
    copy = build_copy_cues(plan, runner=None)
    cards = [c for c in copy["cues"] if c["kind"] == "info_card"]
    assert all(round(c["end_s"] - c["start_s"], 3) == 1.0 for c in cards)
    gaps = [round(b["start_s"] - a["start_s"], 3) for a, b in zip(cards, cards[1:])]
    assert all(g == 1.1 for g in gaps)                       # 1.0s 卡 + 0.1s 间隔
    punch = copy["cues"][-1]
    assert abs(punch["start_s"] - (total - 1.0)) < 1e-6      # 反转梗钉尾 1.0s
    assert cards[-1]["end_s"] <= punch["start_s"] + 1e-6     # 不与梗重叠


def test_broken_llm_falls_back_to_template():
    plan = _plan()
    copy = build_copy_cues(plan, runner=_BrokenRunner())
    assert copy["source"] == "template"
    assert copy["cues"] and copy["cues"][-1]["kind"] == "punchline"


def test_llm_tier_used_when_json_parses():
    plan = _plan()
    copy = build_copy_cues(plan, runner=_GoodRunner())
    assert copy["source"] == "llm"
    texts = [cue["text"] for cue in copy["cues"]]
    assert any("刀" in t for t in texts) and any("为赢" in t for t in texts)


def test_empty_plan_degrades_gracefully():
    copy = build_copy_cues({"slots": [], "target_duration_s": 0})
    assert copy["cues"] == [] and copy["audio_mode"] == "dialogue"


def _grounding_plan():
    slots = [
        {"slot_idx": 0, "role": "hook", "status": "supported",
         "target_interval": [0.0, 7.0], "source": {
             "caption": "小黑在雨中被困", "entity_names": ["小黑"]}},
        {"slot_idx": 1, "role": "conflict", "status": "supported",
         "target_interval": [7.0, 14.0], "source": {
             "caption": "小黑与敌人对峙", "entity_names": ["小黑"]}},
        {"slot_idx": 2, "role": "resolution", "status": "supported",
         "target_interval": [14.0, 22.0], "source": {
             "caption": "小黑守护同伴倒下", "entity_names": ["小黑"]}},
    ]
    return {"slots": slots, "target_duration_s": 22.0, "theme": "守护"}


def test_cues_carry_slot_and_subject_binding():
    """V3 P4：每条 cue 绑 slot_idx/subject_entity/evidence。"""
    copy = build_copy_cues(_grounding_plan(), runner=None)
    assert copy["source"] == "template"
    for cue in copy["cues"]:
        assert cue.get("slot_idx") is not None
        assert cue.get("subject_entity")
        assert cue.get("evidence")


def test_copy_grounding_drops_unevidenced_lines():
    """V3 P4 负例（lxh_p4_C2 punchline 病灶）：「可它还是救了他」在画面无救助
    时必须被替换为无断言兜底；无 caption 重合的 card 必须被丢弃。"""
    from src.agentic_video.copywriter import validate_copy_grounding
    plan = _grounding_plan()
    copy = {"cues": [
        {"kind": "hook_line", "slot_idx": 0, "text": "人们常常觉得妖怪无情"},
        {"kind": "info_card", "slot_idx": 1, "text": "对峙"},
        {"kind": "info_card", "slot_idx": 2, "text": "妖灵被关进笼子"},  # 无重合→丢
        {"kind": "punchline", "slot_idx": 2, "text": "可它还是救了他"},   # 无救助画面→换
        {"kind": "punchline", "slot_idx": 2, "text": "但故事才刚刚开始"}, # 无断言→保留
    ]}
    report = validate_copy_grounding(copy, plan)
    texts = [cue["text"] for cue in copy["cues"]]
    assert "妖灵被关进笼子" not in texts
    assert "可它还是救了他" not in texts
    assert report["replaced_punchline"] is True
    assert any(row["reason"] == "no_visual_evidence" for row in report["dropped"])
