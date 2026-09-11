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
