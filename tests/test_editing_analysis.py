"""analyze_editing 单测：证据装配 / 校验（防自造时刻）/ JSON 解析降级。"""
from __future__ import annotations

import json

from src.library.analyze_editing import EDITING_PROMPT_HEADER, build_evidence, validate_analysis


def test_validate_ok_and_beat_anchor():
    ev = {"duration_s": 30.0, "beats": [1.0, 2.0, 3.0], "bounds": []}
    ok = {"style": "卡点", "timeline": [{"t_s": 1.0, "technique": "hard_cut_on_beat", "note": "x"}],
          "audio": {"bgm": "电子"}, "ffmpeg_hints": ["切点取 beat_points"]}
    assert validate_analysis(ok, ev) == []
    bad_t = {"style": "卡点", "audio": {}, "ffmpeg_hints": [],
             "timeline": [{"t_s": 1.5, "technique": "flash", "note": "x"}]}
    assert any("未落在任何节拍" in e for e in validate_analysis(bad_t, ev))
    bad_k = {"style": "卡点", "audio": {}, "ffmpeg_hints": [],
             "timeline": [{"t_s": 1.0, "technique": "赛博切割", "note": "x"}]}
    assert any("不在枚举" in e for e in validate_analysis(bad_k, ev))


def test_validate_requires_keys():
    errs = validate_analysis({"timeline": []}, {"duration_s": 10, "beats": [], "bounds": []})
    assert any("缺键" in e for e in errs)


def test_evidence_blocks(tmp_path, monkeypatch):
    from src.library import analyze_editing as ae
    from src.config import PathsCfg

    p = tmp_path / "perception" / "v1"
    for tool, out in (("inspect", {"duration_s": 12.5}),
                      ("beats", {"beat_points_s": [0.5, 1.0, 1.5]}),
                      ("shots", {"boundaries_s": [0.5, 2.0]})):
        d = p / tool
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(
            {"schema_version": 1, "tool": tool, "aweme_id": "v1", "params": {},
             "created_at": "t", "host": {}, "output": out}), encoding="utf-8")

    class _C:
        paths = PathsCfg(raw_dir=tmp_path, processed_dir=tmp_path, videos_dir=tmp_path,
                         logs_dir=tmp_path, perception_dir=tmp_path / "perception",
                         generation_dir=tmp_path, library_dir=tmp_path / "library")
    ev, meta = build_evidence(_C, "v1")
    assert "12.5s" in ev and "0.5, 1, 1.5" in ev and "镜头边界" in ev
    assert meta["beats"] == [0.5, 1.0, 1.5]
    assert "节拍点" in EDITING_PROMPT_HEADER
