"""decompose 单测：窗口规划（合并/限额/控制窗）/ 电池 prompt / 答案归一 / FakeRunner 续跑。"""
from __future__ import annotations

import json

from src.library.decompose import (OP_TYPES, build_window_prompt, parse_window_answer,
                                   plan_windows, run_decompose)


def _cand(t, hy=("hard_cut",), conf=0.8, sig=None):
    return {"t_s": t, "type_hypotheses": list(hy), "signature": sig or {},
            "confidence": conf, "source": "signals"}


def test_plan_windows_merges_nearby_candidates():
    wins = plan_windows([_cand(3.0), _cand(3.3), _cand(10.0, ("luma_flash",))], 25.0)
    assert len(wins) == 2                       # 3.0/3.3 合并（<merge_gap 0.5）
    assert wins[0]["start"] == 2.4 and wins[0]["end"] == 3.9
    assert set(wins[0]["hypotheses"]) == {"hard_cut"}
    assert wins[0]["n_candidates"] == 2


def test_plan_windows_caps_with_type_coverage():
    cands = [_cand(i * 0.4, conf=0.6) for i in range(30)]           # 30 个 hard_cut
    cands += [_cand(12.0, ("tracked_mask_fill",), conf=0.45)]
    wins = plan_windows(cands, 30.0, max_windows=5, n_control_windows=0)
    assert len(wins) <= 5
    assert any("tracked_mask_fill" in w["hypotheses"] for w in wins)  # 低分类型不被挤光


def test_plan_windows_caps_chain_merge_duration():
    """试点实测教训：0.3s 间隔密集候选会链式合并滚出巨窗（0~8s），必须有上限。"""
    cands = [_cand(1.0 + i * 0.3, conf=0.6) for i in range(20)]     # 1.0~6.7s 每 0.3s 一个
    wins = plan_windows(cands, 25.0, n_control_windows=0)
    assert len(wins) >= 2                                   # 不再是一个巨窗
    assert all(w["end"] - w["start"] <= 2.5 + 0.65 for w in wins)  # ≤上限+边界余量


def test_plan_windows_control_windows_at_signal_valleys():
    series = {"t": [round(i * 0.1, 3) for i in range(30)],
              "diff_global": [5.0] * 10 + [0.1] * 10 + [5.0] * 9,   # 1.0~2.0s 谷底
              "flow_mag": [1.0] * 10 + [0.05] * 10 + [1.0] * 9}
    wins = plan_windows([_cand(5.0)], 25.0, series=series, n_control_windows=1)
    ctrl = [w for w in wins if w["control"]]
    assert len(ctrl) == 1 and 1.0 <= (ctrl[0]["start"] + ctrl[0]["end"]) / 2 <= 2.0
    assert all(w["idx"] == i for i, w in enumerate(wins))           # 重排后 idx 连续


def test_window_prompt_carries_signature_and_enum():
    w = {"idx": 3, "start": 2.4, "end": 3.6, "hypotheses": ["tracked_mask_fill"],
         "control": False, "signature": {"cells_active": 8}}
    p = build_window_prompt(w, 25.3)
    assert "tracked_mask_fill" in p and "cells_active=8" in p
    assert "hard_cut" in p and "uncertain" in p and "25.3" in p


def test_parse_window_answer_normalizes_and_rejects():
    w = {"start": 2.4, "end": 3.6}
    ok = parse_window_answer(json.dumps({
        "what_changed": "主体内部", "op_type": "蒙版擦除", "subject": "人像",
        "flash": False, "motion": "zoom_in", "direction": None,
        "evidence_quote": "人像内部纹理逐帧变化", "event_time_original_s": 3.05,
        "confidence": 0.8}), w)
    assert ok["op_type"] == "uncertain" and ok["what_changed"] == "global"  # 越界归一
    assert ok["event_time_original_s"] == 3.05
    out_of_win = dict(ok, event_time_original_s=9.9)
    bad = parse_window_answer(json.dumps(out_of_win), w)
    assert bad["event_time_original_s"] is None                          # 越窗自报时刻弃用
    assert parse_window_answer("这不是JSON", w) is None
    assert parse_window_answer('{"no_op_type": 1}', w) is None


def test_run_decompose_resumes_and_writes_aggregate(tmp_path, monkeypatch):
    """FakeRunner：首跑 2 窗 → 改答案后 force=False 只补缺失窗（信封幂等）。"""
    calls = []

    class FakeAns:
        def __init__(self, text):
            self.text, self.elapsed_s, self.output_tokens = text, 1.0, 50

    class FakeRunner:
        def watch(self, video, prompt, *, start_s, end_s, clip_dir, max_new_tokens,
                  duration_s):
            calls.append(start_s)
            return FakeAns(json.dumps({
                "what_changed": "global", "op_type": "hard_cut", "subject": "x",
                "flash": False, "motion": "none", "direction": None,
                "evidence_quote": "q", "event_time_original_s": start_s + 0.3,
                "confidence": 0.7}))

    from src.config import AppConfig, PathsCfg

    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, perception={}, template={},
        generation={}, logging_level="INFO", library={"windows": {}},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "perception",
                       generation_dir=tmp_path / "g",
                       library_dir=tmp_path / "library"))
    vid = "t1"
    (cfg.paths.videos_dir / vid).mkdir(parents=True)
    (cfg.paths.videos_dir / vid / "video.mp4").write_bytes(b"x")
    from src.perception import common
    (cfg.paths.perception_dir / vid / "inspect").mkdir(parents=True, exist_ok=True)
    common.write_result_json(cfg.paths.perception_dir / vid / "inspect",
                             tool="inspect", aweme_id=vid,
                             params={}, output={"duration_s": 25.0})
    (cfg.paths.library_dir / "editing" / vid / "signals").mkdir(parents=True,
                                                                exist_ok=True)
    common.write_result_json(cfg.paths.library_dir / "editing" / vid / "signals",
                             tool="signals", aweme_id=vid, params={},
                             output={"candidates": [_cand(3.0), _cand(12.0)],
                                     "series": None})

    run_decompose(cfg, vid, no_controls=True, runner=FakeRunner())
    assert len(calls) == 2
    agg = json.loads((cfg.paths.library_dir / "editing" / vid / "windows" /
                      "result.json").read_text(encoding="utf-8"))
    assert agg["n_json"] == 2 and agg["n_windows"] == 2
    # 二跑：产物齐 → 零新调用（断点续跑语义）
    run_decompose(cfg, vid, no_controls=True, runner=FakeRunner())
    assert len(calls) == 2
    # --only：只跑指定窗（force 重跑该窗）
    run_decompose(cfg, vid, only=[0], force=True, no_controls=True, runner=FakeRunner())
    assert len(calls) == 3
