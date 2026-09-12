from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agentic_video.narrative_critic import parse_narrative_critique
from src.agentic_video.story_planner import re_search_slot
from src.agentic_video.verify_slots import parse_verification, verify_slots
from tests.test_agentic_narrative import valid_program
from tests.test_agentic_story_planner import _candidate, _wide_candidate


def _verify_plan(tmp_path: Path) -> dict:
    program = valid_program()
    program["arc"] = program["arc"][:3]                     # hook/conflict/choice
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "choice", "person", start=9, event_id="e3")]
    from src.agentic_video.story_planner import build_story_plan
    plan = build_story_plan(program, rows, theme="救助", library="guimie",
                            target_duration_s=45.0)
    for slot in plan["slots"]:                              # 指到真实存在的片源文件
        source = slot["source"]
        if source["video"]:
            video = tmp_path / Path(source["video"]).name
            video.write_bytes(b"fake")
            source["video"] = str(video)
    return plan


class _VerifyRunner:
    def __init__(self, verdict: str, needs_context: bool = False):
        self.verdict = verdict
        self.needs_context = needs_context
        self.watches = 0
        self.saw_clip_dir = False

    def watch(self, video, prompt, **kwargs):
        self.watches += 1
        self.saw_clip_dir = kwargs.get("clip_dir") is not None
        return SimpleNamespace(text=json.dumps({
            "verdict": self.verdict,
            "conditions": [{"condition": "主角在场", "met": self.verdict == "pass",
                            "evidence_interval": [0.5, 2.0]}],
            "missing": [] if self.verdict == "pass" else ["冲突/危险可见"],
            "failure_reason": "" if self.verdict == "pass" else "只有环境画面，无冲突",
            "needs_context": self.needs_context,
            "what_is_visible": "少年在室内"}))


def test_verify_slots_structured_verdicts(tmp_path):
    plan = _verify_plan(tmp_path)
    runner = _VerifyRunner("fail")
    report = verify_slots(None, plan, runner=runner)
    assert runner.saw_clip_dir is True                # watch 必须带 clip_dir（夜间实锤坑）
    assert report["failed_slots"] and report["results"][0]["verdict"] == "fail"
    assert report["results"][0]["missing"] == ["冲突/危险可见"]
    assert report["results"][0]["conditions"][0]["met"] is False

    ok = verify_slots(None, _verify_plan(tmp_path), runner=_VerifyRunner("pass"))
    assert ok["failed_slots"] == [] and ok["uncertain_slots"] == []


def test_verify_slots_widens_context_once_when_requested(tmp_path):
    plan = _verify_plan(tmp_path)
    runner = _VerifyRunner("uncertain", needs_context=True)
    report = verify_slots(None, plan, runner=runner, context_pad_s=5.0)
    assert runner.watches >= 2 * sum(1 for s in plan["slots"] if s["status"] == "supported")
    assert report["results"][0]["verdict"] == "uncertain"   # 扩展后仍判断不了就如实报


def test_parse_verification_rejects_garbage_and_guessing():
    assert parse_verification("不是 JSON") is None
    assert parse_verification(json.dumps({"verdict": "yes"})) is None
    good = parse_verification(json.dumps({
        "verdict": "pass",
        "conditions": [{"condition": "主角在场", "met": True, "evidence_interval": [1, 2]}],
        "missing": [], "failure_reason": "", "needs_context": False,
        "what_is_visible": "战斗"}))
    assert good["verdict"] == "pass" and good["conditions"][0]["met"] is True


def test_critic_re_search_directives_parsed_and_bounded():
    raw = json.dumps({
        "theme_relevance": 0.5, "narrative_coherence": 0.5,
        "answers": [{"question_id": 0, "answer": "a", "correct": True, "evidence": "e"}],
        "issues": [],
        "patches": [],
        "re_search": [
            {"slot_idx": 2, "reason": "画面不支持保护关系", "need_hint": "同框互动的保护行动"},
            {"slot_idx": "nine", "reason": "坏下标被丢弃"},
            {"slot_idx": 0},
        ],
        "verdict": "v"}, ensure_ascii=False)
    critique = parse_narrative_critique(raw)
    assert [d["slot_idx"] for d in critique["re_search"]] == [2, 0]
    assert critique["re_search"][0]["need_hint"].startswith("同框")
    assert critique["re_search"][1]["need_hint"] is None


def _research_plan(tmp_path: Path) -> tuple[dict, object]:
    program = valid_program()
    program["arc"] = program["arc"][:3]                     # hook/conflict/choice
    rows = [_candidate(0, "hook", "person", start=0, event_id="e1"),
            _candidate(1, "conflict", "person", start=4, event_id="e2"),
            _candidate(2, "choice", "person", start=9, event_id="e3")]
    from src.agentic_video.story_planner import build_story_plan
    plan = build_story_plan(program, rows, theme="救助", library="guimie",
                            target_duration_s=45.0)
    return plan, program


def test_re_search_level1_uses_pool_and_respects_rejected(tmp_path):
    plan, _program = _research_plan(tmp_path)
    current = dict(plan["slots"][1]["source"])
    pool = [
        _candidate(9, "conflict", "person", start=400, event_id="e9", score=0.7),
        _candidate(8, "conflict", "person", start=800, event_id="e8", score=0.6),
    ]
    # 已拒绝清单挡住最高分的 400s 段（同视频重叠去重）——用其它槽占用它
    others_block = [{"video": pool[0]["video"], "start_s": 399, "end_s": 401}]
    for row in pool:
        row["video"] = str(tmp_path / "movie.mp4")
    report = re_search_slot(None, plan, 1, theme="救助", candidate_pool=pool,
                            need_hint="可见的冲突", rejected=[
                                {"video": pool[0]["video"], "start_s": 398,
                                 "end_s": 402, "reason": "verification_failed"}])
    assert report["status"] == "replaced" and report["level"] == 1
    assert plan["slots"][1]["source"]["start_s"] == 800.0   # 400s 段被拒，落到次优
    assert plan["slots"][1]["reason"].startswith("re_searched:")
    assert plan["slots"][1]["source"]["start_s"] != current["start_s"]


def test_re_search_fails_explicitly_when_nothing_eligible(tmp_path):
    plan, _program = _research_plan(tmp_path)
    pool = [_candidate(9, "conflict", "person", start=400, event_id="e9")]
    pool[0]["video"] = str(tmp_path / "movie.mp4")
    everything_rejected = [{"video": pool[0]["video"], "start_s": 0,
                            "end_s": 10_000, "reason": "verification_failed"}]
    before = json.dumps(plan["slots"][1]["source"], ensure_ascii=False)
    report = re_search_slot(None, plan, 1, theme="救助", candidate_pool=pool,
                            rejected=everything_rejected)
    assert report["status"] == "failed"                     # 明确失败，不放宽凑满
    assert json.dumps(plan["slots"][1]["source"], ensure_ascii=False) == before


def _contract_plan(switched=False):
    """三槽计划：正常=全小黑；switched=槽1 换成无限（C2 病灶形态）。"""
    def slot(idx, role, name, entity_id, status="supported"):
        return {"slot_idx": idx, "role": role, "status": status,
                "transition_reason": "entity_continuity" if idx else "opening",
                "need_spec": {"required": True,
                              "entity_requirements": {"protagonist": "required"}},
                "source": {"video": "a.mkv", "video_stem": "luoxiaohei1__narrative",
                           "start_s": 100 + idx * 50, "end_s": 107 + idx * 50,
                           "entity_ids": [entity_id], "entity_names": [name],
                           "caption": f"{name}行动"}}
    slots = [slot(0, "hook", "小黑", "entity_005")]
    if switched:
        second = slot(1, "conflict", "无限", "entity_009")
        second["transition_reason"] = "unexplained"
        slots.append(second)
    else:
        slots.append(slot(1, "conflict", "小黑", "entity_005"))
    slots.append(slot(2, "resolution", "小黑", "entity_005"))
    return {"slots": slots, "entity_contract": {
        "protagonist": "char:xiaohei", "locked_slots": [0, 1, 2],
        "free_slots": [], "relaxed": False}}


def test_deterministic_check_catches_protagonist_switch_without_omni():
    """V3 P3：Slot0=小黑、Slot1=无限 在渲染前就被确定性拦下（零模型）。"""
    from src.agentic_video.verify_slots import deterministic_story_check
    registry = {"char:xiaohei": {"aliases": ["小黑"]},
                "char:wuxian": {"aliases": ["无限"]}}
    ok = deterministic_story_check(_contract_plan(switched=False), registry=registry)
    assert ok["passed"] is True
    assert ok["metrics"]["protagonist_switch_count"] == 0
    assert ok["metrics"]["required_protagonist_presence"] == 1.0
    bad = deterministic_story_check(_contract_plan(switched=True), registry=registry)
    assert bad["passed"] is False
    assert bad["metrics"]["protagonist_switch_count"] == 1
    assert 1 in bad["re_search_slots"]


def test_blind_video_check_prompt_carries_zero_context():
    """V3 P3：盲看 prompt 只描述任务，不含计划/文案/参考——独立证词。"""
    from src.agentic_video.verify_slots import BLIND_VIDEO_PROMPT
    assert "Story Plan" not in BLIND_VIDEO_PROMPT
    assert "Narrative Program" not in BLIND_VIDEO_PROMPT
    assert "consistent_protagonist" in BLIND_VIDEO_PROMPT


def test_blind_video_check_parses_structured_answer():
    from types import SimpleNamespace
    import json as _json
    from src.agentic_video.verify_slots import blind_video_check

    class _Runner:
        def watch(self, video, prompt, **kwargs):
            assert "背景资料" in prompt
            return SimpleNamespace(text=_json.dumps({
                "main_character": "一只黑猫妖灵",
                "consistent_protagonist": False,
                "switch_points": [{"at_s": 7.0, "what_changed": "人物换成人类男性"}],
                "story_in_one_sentence": "看不懂在讲什么",
                "event_relations": "无关"}, ensure_ascii=False))

    result = blind_video_check("fake.mp4", runner=_Runner())
    assert result["parsed"] is True
    assert result["consistent_protagonist"] is False
    assert result["switch_points"][0]["at_s"] == 7.0


def test_parse_verification_normalizes_inconsistent_pass():
    """V4 E1（外审六节反例）：pass+met=false/missing 非空 → 降级 fail；
    met=true 无合法 evidence_interval → met=false；fail 无原因 → 补 unspecified。"""
    bad = parse_verification(json.dumps({
        "verdict": "pass",
        "conditions": [{"condition": "主角在场", "met": False,
                        "evidence_interval": [1, 2]}],
        "missing": ["冲突可见"], "failure_reason": "", "needs_context": False,
        "what_is_visible": "x"}, ensure_ascii=False))
    assert bad["verdict"] == "fail"
    assert "normalized: pass with unmet conditions" in bad["failure_reason"]

    no_interval = parse_verification(json.dumps({
        "verdict": "pass",
        "conditions": [{"condition": "主角在场", "met": True,
                        "evidence_interval": None}],
        "missing": [], "failure_reason": "", "needs_context": False,
        "what_is_visible": "x"}, ensure_ascii=False))
    assert no_interval["verdict"] == "fail"
    assert no_interval["conditions"][0]["met"] is False

    inverted = parse_verification(json.dumps({
        "verdict": "pass",
        "conditions": [{"condition": "主角在场", "met": True,
                        "evidence_interval": [5, 2]}],
        "missing": [], "failure_reason": "", "needs_context": False,
        "what_is_visible": "x"}, ensure_ascii=False))
    assert inverted["verdict"] == "fail"

    empty_fail = parse_verification(json.dumps({
        "verdict": "fail", "conditions": [], "missing": [],
        "failure_reason": "", "needs_context": False, "what_is_visible": "x"}))
    assert empty_fail["verdict"] == "fail"
    assert empty_fail["failure_reason"] == "unspecified"


def test_overall_verdict_truth_table():
    """V4 E3（外审必改③）：blind_required 时 None/unparsed/inconsistent 全
    blocked；det/plan/文案轨三因子独立命名；B 档（blind_required=False）
    不因盲看缺席而 fail。"""
    from src.agentic_video.pipeline import _overall_verdict

    det_ok = {"passed": True, "violations": []}
    story = {"slots": [{"slot_idx": 0, "status": "supported",
                        "need_spec": {"required": True}}],
             "copy": {"cues": [{"kind": "hook_line"}, {"kind": "punchline"}]}}
    grounding = {"kept": 2, "dropped": []}
    blind_ok = {"parsed": True, "consistent_protagonist": True}
    assert _overall_verdict(det_ok, blind_ok, grounding, story,
                            blind_required=True)["passed"] is True
    # 三态盲看失败
    for blind in (None, {"parsed": False},
                  {"parsed": True, "consistent_protagonist": False}):
        verdict = _overall_verdict(det_ok, blind, grounding, story,
                                   blind_required=True)
        assert verdict["passed"] is False
    assert _overall_verdict(det_ok, None, grounding, story,
                            blind_required=True)["reasons"] == ["blind_missing"]
    # B 档不因盲看缺席 fail
    assert _overall_verdict(det_ok, None, grounding, story,
                            blind_required=False)["passed"] is True
    # det / plan / 文案轨道独立
    assert _overall_verdict({"passed": False, "violations": ["x"]}, blind_ok,
                            grounding, story, blind_required=True)["reasons"] \
        == ["det_violations:x"]
    broken = {"slots": [{"slot_idx": 1, "status": "unsupported",
                         "need_spec": {"required": True}}],
              "copy": {"cues": []}}
    reasons = _overall_verdict(det_ok, blind_ok, grounding, broken,
                               blind_required=False)["reasons"]
    assert any(r.startswith("plan_incomplete") for r in reasons)
    assert any(r.startswith("copy_track_empty") for r in reasons)


def test_parse_localization_multi_object_and_absolute_time():
    """V4 探针实锤：Omni 输出一串 JSON 对象+垃圾后缀；且报绝对时间
    （窗口 4095-4140 里 start=4121）。解析取第一个 found=true；绝对时间
    由 localize 主循环判别（此处只测解析层）。"""
    from src.agentic_video.verify_slots import parse_localization
    raw = ('{"found": true, "start": 4121, "end": 4123, "evidence": "你可能还不了解", "confidence": 1.0} ant\n'
           '{"found": true, "start": 4128, "end": 4131, "evidence": "所到之处就是属于你的世界", "confidence": 1.0} ant\n'
           '{"found": false, "start": 0, "end": 0, "evidence": "", "confidence": 0.0} ant')
    parsed = parse_localization(raw)
    assert parsed is not None and parsed["found"] is True
    assert parsed["interval"] == [4121.0, 4123.0]          # 第一个真候选
    assert "你可能还不了解" in parsed["evidence"]
    # 单对象路径兼容
    single = parse_localization('{"found": false, "start": 0, "end": 0, "evidence": "", "confidence": 0.0}')
    assert single["found"] is False
    assert parse_localization("完全不是 JSON") is None
