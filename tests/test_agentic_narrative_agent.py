from __future__ import annotations

import json
from types import SimpleNamespace

from src.agentic_video.narrative_agent import (NarrativeBudget,
                                                build_reference_identity,
                                                parse_narrative_program,
                                                plan_narrative_windows,
                                                run_narrative_agent)
from src.agentic_video.narrative import new_narrative_program


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


def test_parser_normalizes_six_questions_importance_and_arc_function():
    """P0 v2：六问默认 unknown、importance 归一、arc.function 透传、外部证据
    source（external_title 等）不再被丢掉。"""
    raw = json.dumps({
        "intent": {"topic": "守护", "message": "牺牲", "content_type": "growth_story",
                   "protagonist": "短发少年", "motivation": "", "outcome": "not_applicable",
                   "evidence": [{"source": "external_title", "interval": [0, 1],
                                 "quote": "标题原文", "confidence": 0.9}],
                   "confidence": 0.8, "status": "supported"},
        "entities": [{"id": "e0", "kind": "person", "name_or_role": "少年",
                      "evidence": [], "confidence": 0.9, "status": "supported"}],
        "events": [{"id": "ev0", "interval": [0.0, 5.0], "participants": ["e0"],
                    "action": "挥刀", "importance": "超高", "evidence": [],
                    "confidence": 0.9, "status": "supported"}],
        "causal_links": [], "utterances": [], "emotion_curve": [],
        "arc": [{"role": "hook", "event_ids": ["ev0"], "function": "抛出悬念"}],
        "uncertainties": [],
    }, ensure_ascii=False)
    program = parse_narrative_program(
        raw, reference_id="r", reference_uri="r.mp4", sha256="a" * 64,
        duration_s=10, fps=24, model="fake", tool_calls=[])
    intent = program["intent"]
    assert intent["protagonist"] == "短发少年"
    assert intent["motivation"] == "unknown"              # 未呈现不补写
    assert intent["outcome"] == "not_applicable"          # 合法值透传
    assert intent["goal"] == "unknown" and intent["problem"] == "unknown"
    assert program["events"][0]["importance"] == "medium" # 非法值归一
    assert program["arc"][0]["function"] == "抛出悬念"
    assert program["provenance"]["prompt_version"] == "narrative_agent_v2"
    sources = {row["source"] for row in intent["evidence"]}
    assert "external_title" in sources


class _FakeRunner:
    def __init__(self):
        self.watches = 0
        self.asks = 0

    def watch(self, video, prompt, **kwargs):
        self.watches += 1
        return SimpleNamespace(text=json.dumps({
            "observations": [{"start_s": kwargs.get("start_s", 0),
                              "end_s": kwargs.get("end_s", 1), "entities": ["少年"],
                              "action": "战斗", "speech": None, "emotion": "uncertain",
                              "story_role": "conflict", "evidence": [
                                  {"source": "frame", "interval": [0, 1],
                                   "quote": "刀光", "confidence": 0.9}],
                              "confidence": 0.9}],
            "uncertainties": []}))

    def ask(self, prompt, *, max_new_tokens=None):
        self.asks += 1
        return SimpleNamespace(text=json.dumps({
            "intent": {"topic": "守护", "message": "牺牲", "content_type": "growth_story",
                       "protagonist": "短发少年", "evidence": [], "confidence": 0.9,
                       "status": "supported"},
            "entities": [{"id": "e0", "kind": "person", "name_or_role": "少年",
                          "evidence": [{"source": "frame", "interval": [0.0, 1.0],
                                        "confidence": 0.9}], "confidence": 0.9,
                          "status": "supported"}],
            "events": [{"id": "ev0", "interval": [0.0, 8.0], "participants": ["e0"],
                        "action": "战斗", "state_before": "平静", "state_after": "负伤",
                        "importance": "high",
                        "evidence": [{"source": "frame", "interval": [0.0, 8.0],
                                      "confidence": 0.9}], "confidence": 0.9,
                        "status": "supported"},
                       {"id": "ev1", "interval": [2.0, 9.0], "participants": ["e0"],
                        "action": "和解", "state_before": "对峙", "state_after": "和解",
                        "evidence": [{"source": "frame", "interval": [2.0, 9.0],
                                      "confidence": 0.9}], "confidence": 0.9,
                        "status": "supported"},
                       {"id": "ev2", "interval": [4.0, 9.0], "participants": ["e0"],
                        "action": "启程", "state_before": "和解", "state_after": "出发",
                        "evidence": [{"source": "frame", "interval": [4.0, 9.0],
                                      "confidence": 0.9}], "confidence": 0.9,
                        "status": "supported"}],
            "causal_links": [], "utterances": [], "emotion_curve": [],
            "arc": [{"role": "conflict", "event_ids": ["ev0"], "function": "危机爆发"},
                    {"role": "climax", "event_ids": ["ev1"], "function": "全力一击"},
                    {"role": "resolution", "event_ids": ["ev2"], "function": "收束"}],
            "uncertainties": []}, ensure_ascii=False))


def _agent_cfg(tmp_path, *, with_inspect=True):
    perception_dir = tmp_path / "perception"
    if with_inspect:
        inspect_dir = perception_dir / "ref" / "inspect"
        inspect_dir.mkdir(parents=True, exist_ok=True)
        (inspect_dir / "result.json").write_text(json.dumps(
            {"output": {"duration_s": 10.0, "fps": 24}}), encoding="utf-8")
    videos_dir = tmp_path / "videos"
    ref_dir = videos_dir / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "video.mp4").write_bytes(b"fake-video-bytes")
    (ref_dir / "metadata.json").write_text(json.dumps(
        {"title": "测试标题", "author": "作者", "comments": [{"text": "评论线索"}],
         "related_videos": [{"title": "相关推荐一"}]}), encoding="utf-8")
    return SimpleNamespace(paths=SimpleNamespace(perception_dir=perception_dir,
                                                 videos_dir=videos_dir),
                           perception={})


def test_synthesis_cache_hit_reuses_program_without_model_calls(tmp_path):
    cfg = _agent_cfg(tmp_path)
    runner = _FakeRunner()
    run_narrative_agent(cfg, "ref", budget=NarrativeBudget(max_initial_windows=2, target_window_s=5.0),
                        runner=runner)
    first_watches, first_asks = runner.watches, runner.asks
    assert first_watches == 2 and first_asks >= 1

    program = run_narrative_agent(cfg, "ref", runner=runner)   # 依赖全同 → 快路径
    assert runner.watches == first_watches and runner.asks == first_asks
    assert program["provenance"]["prompt_version"] == "narrative_agent_v2"


def test_legacy_envelope_without_cache_is_stale_not_reused(tmp_path):
    """旧式 {'program': …} 信封（无依赖指纹）不得再被存在性判断直接复用。"""
    cfg = _agent_cfg(tmp_path)
    root = cfg.paths.perception_dir / "ref" / "narrative_agent"
    root.mkdir(parents=True)
    legacy = new_narrative_program(reference_id="ref", reference_uri="ref.mp4",
                                   sha256="a" * 64, duration_s=10, fps=24)
    (root / "result.json").write_text(json.dumps({"program": legacy}), encoding="utf-8")
    runner = _FakeRunner()
    program = run_narrative_agent(cfg, "ref",
                                  budget=NarrativeBudget(max_initial_windows=2, target_window_s=5.0),
                                  runner=runner)
    assert runner.watches > 0 and runner.asks > 0           # 真的重建了
    assert program != legacy                                 # 不再吃旧程序


def test_window_observation_cache_survives_synthesis_prompt_change(tmp_path):
    """只改归纳 prompt（v2→新键）时窗口观察缓存仍命中：只有 synthesis 重算。"""
    cfg = _agent_cfg(tmp_path)
    runner = _FakeRunner()
    run_narrative_agent(cfg, "ref", budget=NarrativeBudget(max_initial_windows=2, target_window_s=5.0),
                        runner=runner)
    watches_after_first = runner.watches

    # 模拟归纳提示词变更：作废顶层信封（删 result.json），窗口缓存保留
    (cfg.paths.perception_dir / "ref" / "narrative_agent" / "result.json").unlink()
    import src.agentic_video.narrative_agent as agent_mod
    original = agent_mod.SYNTHESIS_PROMPT
    agent_mod.SYNTHESIS_PROMPT = original + "\n【v3 试炼】"
    try:
        run_narrative_agent(cfg, "ref",
                            budget=NarrativeBudget(max_initial_windows=2, target_window_s=5.0),
                            runner=runner)
    finally:
        agent_mod.SYNTHESIS_PROMPT = original
    assert runner.watches == watches_after_first            # 零次重看片
    assert runner.asks > 1                                  # 只重算了归纳


def test_reference_identity_records_sha_title_and_evidence(tmp_path):
    cfg = _agent_cfg(tmp_path)
    perception = cfg.paths.perception_dir / "ref"
    ocr_dir = perception / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "result.json").write_text(json.dumps(
        {"output": {"text_events": [{"t_start_s": 0.0, "t_end_s": 4.0,
                                     "text": "人们常常觉得失去了双手"}]}}),
        encoding="utf-8")
    video = cfg.paths.videos_dir / "ref" / "video.mp4"
    from src.agentic_video.recipe_v2 import sha256_file
    identity = build_reference_identity(cfg, "ref", video=video,
                                        video_sha=sha256_file(video))
    assert identity["reference_id"] == "ref"
    assert identity["metadata_title"] == "测试标题"
    assert identity["source_sha256"] == sha256_file(video)
    assert identity["key_evidence_locations"][0]["text"] == "人们常常觉得失去了双手"


def test_prompt_v2_carries_tiered_evidence_and_reconciliation_rules():
    from src.agentic_video.narrative_agent import (SYNTHESIS_PROMPT,
                                                   build_narrative_material)
    assert "external_title/external_comment/external_related" in SYNTHESIS_PROMPT
    assert "逐项核对" in SYNTHESIS_PROMPT                      # 主角对账
    assert "不同的 function" in SYNTHESIS_PROMPT               # 事件多角色须说明功能
    for field in ("protagonist", "goal", "problem", "motivation", "change", "outcome"):
        assert field in SYNTHESIS_PROMPT                       # 六问
    assert "not_applicable" in SYNTHESIS_PROMPT
