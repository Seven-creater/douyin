"""recipe 单测：BPM 估计 / 证据锚定校验 / texture 反编造 / 规则底稿映射 / repair·回退。"""
from __future__ import annotations

import json

from src.library.recipe import (OP_TYPES, build_rule_draft, bpm_from_beats,
                                recipe_penalty, validate_recipe)


def _meta(**over):
    meta = {"duration_s": 25.0, "beats": [1.0, 1.5, 2.0, 2.5, 3.0],
            "bounds": [0.0, 5.0, 10.0, 25.0],
            "ocr_events": [{"t_start_s": 2.0, "text": "标题"}],
            "candidates": [
                {"t_s": 2.0, "type_hypotheses": ["tracked_mask_fill", "texture_replace"],
                 "signature": {"cells_active": 8}, "confidence": 0.45,
                 "source": "signals"},
                {"t_s": 5.0, "type_hypotheses": ["hard_cut"], "signature": {},
                 "confidence": 0.85, "source": "signals"}],
            "window_confirms": [{"t_s": 2.05, "op_type": "tracked_mask_fill",
                                 "idx": 3, "quote": "人像内部纹理变化", "subject": "人"}],
            "known_ts": [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 25.0],
            "seg_bounds": [0.0, 5.0, 10.0, 25.0], "bpm": 120.0}
    meta.update(over)
    return meta


def _recipe(ops=None, segs=None):
    return {"video": {"aweme_id": "x", "duration_s": 25.0}, "tempo_bpm_est": 120.0,
            "segments": segs if segs is not None else [
                {"index": 0, "start": 0.0, "end": 5.0},
                {"index": 1, "start": 5.0, "end": 10.0},
                {"index": 2, "start": 10.0, "end": 25.0}],
            "operations": ops if ops is not None else [], "global_notes": ""}


def test_bpm_from_beats_median_interval():
    assert bpm_from_beats([0.0, 0.5, 1.0, 1.5]) == 120.0       # 0.5s 间隔
    assert bpm_from_beats([0.0, 0.5, 1.0, 3.0]) == 120.0       # 中位数取多数
    assert bpm_from_beats([0.0, 1.0]) is None                   # 样本不足


def test_validate_rejects_fabricated_timestamp_and_enum():
    errs = validate_recipe(_recipe(ops=[
        {"t_s": 7.77, "type": "hard_cut", "confidence": 0.8}]), _meta())
    assert any("证据时刻" in e for e in errs)                    # 7.77 不在任何锚附近
    errs2 = validate_recipe(_recipe(ops=[
        {"t_s": 5.0, "type": "蒙版擦除", "confidence": 0.8}]), _meta())
    assert any("不在枚举" in e for e in errs2)


def test_validate_rejects_duplicate_anchor_and_unconfirmed_texture():
    errs = validate_recipe(_recipe(ops=[
        {"t_s": 5.0, "type": "hard_cut", "confidence": 0.8},
        {"t_s": 5.02, "type": "hard_cut", "confidence": 0.8}]), _meta())
    assert any("重复" in e for e in errs)
    # texture_sequence 有窗口确认（t=2.05）→ 合法
    ok = validate_recipe(_recipe(ops=[
        {"t_s": 2.0, "type": "tracked_mask_fill", "confidence": 0.8,
         "texture_sequence": ["stone"], "change_anchor": 2.0}]), _meta())
    assert ok == []
    # 无确认的操作挂 texture → 拒
    errs2 = validate_recipe(_recipe(ops=[
        {"t_s": 5.0, "type": "tracked_mask_fill", "confidence": 0.8,
         "texture_sequence": ["stone"]}]), _meta())
    assert any("texture_sequence" in e for e in errs2)


def test_validate_segment_coverage_and_soft_normalize():
    errs = validate_recipe(_recipe(segs=[{"index": 0, "start": 0.0, "end": 10.0}]),
                           _meta())
    assert any("未覆盖" in e for e in errs)
    segs = [{"index": 0, "start": 0.0, "end": 25.0,
             "source": {"camera_motion": "推近", "shot_scale": "特写"},
             "transition_out": {"type": "mask_wipe", "duration": 0.2}}]
    assert validate_recipe(_recipe(segs=segs), _meta()) == []   # 越界软归一不拒


def test_rule_draft_maps_and_caps_confidence():
    draft = build_rule_draft(_meta(), "x")
    ops = {o["type"]: o for o in draft["operations"]}
    assert "tracked_mask_fill" in ops and "texture_replace" not in ops  # 折叠映射
    assert all(o["confidence"] <= 0.4 for o in draft["operations"])     # 底稿压置信
    assert [s["start"] for s in draft["segments"]] == [0.0, 5.0, 10.0]
    assert recipe_penalty(draft, _meta()) == 0.0                        # 底稿零惩罚


def test_penalty_counts_unanchored_ops():
    r = _recipe(ops=[{"t_s": 7.7, "type": "hard_cut", "confidence": 0.8}])
    assert recipe_penalty(r, _meta()) >= 10.0


def _setup_env(tmp_path, answers, meta_extra=None):
    from src.config import AppConfig, PathsCfg
    from src.perception import common

    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, perception={}, template={},
        generation={}, logging_level="INFO", library={"recipe": {}},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "perception",
                       generation_dir=tmp_path / "g", library_dir=tmp_path / "lib"))
    vid = "t1"
    (cfg.paths.videos_dir / vid).mkdir(parents=True)
    for tool, out in (("inspect", {"duration_s": 25.0}),
                      ("beats", {"beat_points_s": _meta()["beats"]}),
                      ("shots", {"boundaries_s": _meta()["bounds"]}),
                      ("ocr", {"text_events": _meta()["ocr_events"]})):
        d = cfg.paths.perception_dir / vid / tool
        d.mkdir(parents=True, exist_ok=True)
        common.write_result_json(d, tool=tool, aweme_id=vid, params={}, output=out)
    sig = _meta()["candidates"]
    d = cfg.paths.library_dir / "editing" / vid / "signals"
    d.mkdir(parents=True, exist_ok=True)
    common.write_result_json(d, tool="signals", aweme_id=vid, params={},
                             output={"candidates": sig})
    wd = cfg.paths.library_dir / "editing" / vid / "windows"
    wd.mkdir(parents=True, exist_ok=True)
    # 生产形态：decompose 汇总是裸 JSON（无信封 output 层）
    (wd / "result.json").write_text(json.dumps({
        "aweme_id": vid, "results": [
            {"idx": 3, "parse": "json", "answer": {
                "op_type": "tracked_mask_fill",
                "event_time_original_s": 2.05,
                "evidence_quote": "人像内部纹理变化",
                "subject": "人"}}]}, ensure_ascii=False), encoding="utf-8")

    class FakeAns:
        def __init__(self, text):
            self.text, self.elapsed_s = text, 1.0

    class FakeRunner:
        def __init__(self):
            self.i = 0

        def ask(self, prompt, *, max_new_tokens=None):
            a = FakeAns(answers[min(self.i, len(answers) - 1)])
            self.i += 1
            return a

    return cfg, vid, FakeRunner()


def test_run_recipe_repairs_then_accepts(tmp_path):
    good = json.dumps(_recipe(ops=[
        {"t_s": 2.0, "type": "tracked_mask_fill", "confidence": 0.8,
         "texture_sequence": ["stone"], "change_anchor": 2.0,
         "evidence": {"source": "window", "window_idx": 3}}]), ensure_ascii=False)
    bad = json.dumps(_recipe(ops=[
        {"t_s": 7.77, "type": "hard_cut", "confidence": 0.8}]), ensure_ascii=False)
    cfg, vid, runner = _setup_env(tmp_path, [bad, good])
    from src.library.recipe import run_recipe
    p = run_recipe(cfg, vid, force=True, runner=runner)
    env = json.loads(p.read_text(encoding="utf-8"))
    assert env["output"]["mode"] == "json" and env["output"]["attempts"] == 2  # repair 生效


def test_run_recipe_falls_back_to_rule_draft(tmp_path):
    garbage = "完全不是 JSON" * 10
    cfg, vid, runner = _setup_env(tmp_path, [garbage, garbage])
    from src.library.recipe import run_recipe
    p = run_recipe(cfg, vid, force=True, runner=runner)
    env = json.loads(p.read_text(encoding="utf-8"))
    assert env["output"]["mode"] == "rule_fallback"
    assert env["output"]["n_operations"] == 2                      # 候选直填
