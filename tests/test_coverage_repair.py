"""extract_template 覆盖修复闭环单测：首版 json 有段间空隙 → 自动 repair → 连续覆盖。

FakeRunner 脚本化两次 ask：第一版 timeline 0→5 + 6→16.5（空隙 1s），
第二版连续覆盖；断言最终产物是第二版且 parse_meta 记了两轮 attempt。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import AppConfig
from src.template.extract_template import synthesize


@dataclass
class _Answer:
    text: str
    input_tokens: int = 100
    output_tokens: int = 500
    elapsed_s: float = 1.0


def _tl(spans):
    return [{"start": s, "end": e, "role": "setup", "visual": "v", "speech": None, "text": None}
            for s, e in spans]


def _tpl(spans):
    return {"trend_summary": "s", "core_meme": "c", "timeline": _tl(spans),
            "audio": {"bgm": None, "beat_points": [], "speech": []},
            "fixed_elements": ["f"], "replaceable_elements": ["r"],
            "generation_plan": ["g"]}


class FakeRunner:
    def __init__(self, answers):
        self._answers = list(answers)

    def ask(self, prompt, max_new_tokens=None):
        return _Answer(self._answers.pop(0))


def _setup(tmp_path, spans_list):
    pid = tmp_path / "data" / "perception" / "vid1"
    (pid / "omni_full").mkdir(parents=True)
    (pid / "omni_full" / "result.json").write_text(json.dumps(
        {"schema_version": 1, "tool": "omni_full", "aweme_id": "vid1", "params": {},
         "created_at": "t", "host": {}, "output": {"sections": {}}}), encoding="utf-8")
    (pid / "inspect").mkdir()
    (pid / "inspect" / "result.json").write_text(json.dumps(
        {"schema_version": 1, "tool": "inspect", "aweme_id": "vid1", "params": {},
         "created_at": "t", "host": {}, "output": {"duration_s": 16.9, "width": 720,
                                                   "height": 1280, "fps": 30}}), encoding="utf-8")
    (pid / "shots").mkdir()
    (pid / "shots" / "result.json").write_text(json.dumps(
        {"schema_version": 1, "tool": "shots", "aweme_id": "vid1", "params": {},
         "created_at": "t", "host": {}, "output": {"shot_count": 3, "boundaries_s": [0.0, 5.0, 6.0],
                                                   "shots": []}}), encoding="utf-8")
    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, perception={}, template={"max_retries": 1},
        generation={}, logging_level="INFO", paths=_paths(tmp_path))
    runner = FakeRunner([json.dumps(_tpl(s), ensure_ascii=False) for s in spans_list])
    return cfg, runner, pid.parent / "vid1" / "template" / "result.json"


def _paths(tmp_path):
    from src.config import PathsCfg
    return PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p", videos_dir=tmp_path / "v",
                    logs_dir=tmp_path / "l", perception_dir=tmp_path / "data" / "perception",
                    generation_dir=tmp_path / "g")


def test_coverage_gap_triggers_repair(tmp_path):
    cfg, runner, out = _setup(tmp_path, [[(0.0, 5.0), (6.0, 16.5)], [(0.0, 16.5)]])
    path = synthesize(cfg, "vid1", force=True, runner=runner)
    assert path == out
    d = json.loads(path.read_text(encoding="utf-8"))
    tl = d["output"]["template"]["timeline"]
    assert len(tl) == 1 and tl[0]["end"] == 16.5          # 用了修复后的连续版本
    assert d["output"]["parse_meta"]["attempts"] == 2      # 两轮 attempt 落账
    assert not any("未覆盖" in w for w in d["output"]["parse_meta"]["warnings"])


def test_fallback_keeps_first_json_when_repair_worse(tmp_path):
    cfg, runner, out = _setup(tmp_path, [[(0.0, 5.0), (6.0, 16.5)], [(0.0, 3.0)]])
    # 第二版尾部缺失更严重 → 仍应保底用第一版
    path = synthesize(cfg, "vid1", force=True, runner=runner)
    d = json.loads(path.read_text(encoding="utf-8"))
    tl = d["output"]["template"]["timeline"]
    assert len(tl) == 2                                    # 第一版（保底）
