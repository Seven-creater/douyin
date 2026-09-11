"""perception common 单测：信封读写、幂等跳过、metrics 行格式、原子写、路径解析。"""
from __future__ import annotations

import json

import pytest

from src.perception import common


def test_tool_dir_and_clip_dir(tmp_path):
    d = common.tool_dir_for(tmp_path, "id1", "inspect")
    assert d == tmp_path / "id1" / "inspect" and d.exists()
    c = common.clip_dir_for(tmp_path, "id1", 3.0, 9.5)
    assert c == tmp_path / "id1" / "omni_clips" / "3-9.5" and c.exists()


def test_resolve_video_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phase 1"):
        common.resolve_video_path(tmp_path, "nope")


def test_list_aweme_ids_only_complete_dirs(tmp_path):
    (tmp_path / "a" / "video.mp4").parent.mkdir(parents=True)
    (tmp_path / "a" / "video.mp4").write_bytes(b"x")
    (tmp_path / "b").mkdir()          # 无 video.mp4
    (tmp_path / "manifest.json").write_text("{}")
    assert common.list_aweme_ids(tmp_path) == ["a"]


def test_write_and_read_result_json_roundtrip(tmp_path):
    p = common.write_result_json(tmp_path, tool="t", aweme_id="id1",
                                 params={"fps": 1.0}, output={"k": "值"})
    assert p.name == "result.json"
    assert not p.with_suffix(".json.tmp").exists()  # 无 .tmp 残留
    env = common.read_result_json(tmp_path)
    assert env["tool"] == "t" and env["aweme_id"] == "id1"
    assert env["params"] == {"fps": 1.0}
    assert env["output"]["k"] == "值"
    assert env["schema_version"] == 1
    assert "created_at" in env and "host" in env


def test_read_result_json_tolerates_corrupt(tmp_path):
    (tmp_path / "result.json").write_text("{broken", encoding="utf-8")
    assert common.read_result_json(tmp_path) is None


def test_done_or_skip_semantics(tmp_path):
    common.write_result_json(tmp_path, tool="t", aweme_id="id1", params={"a": 1}, output={})
    # 已有且同参 → 跳过（返回旧信封）
    assert common.done_or_skip(tmp_path, {"a": 1}, force=False) is not None
    # force → 重跑
    assert common.done_or_skip(tmp_path, {"a": 1}, force=True) is None
    # 参数漂移 → 仍跳过（默认沿用旧产物）
    assert common.done_or_skip(tmp_path, {"a": 2}, force=False) is not None
    # 无产物 → 跑
    assert common.done_or_skip(tmp_path / "empty", {"a": 1}, force=False) is None


def test_append_metric_jsonl_lines(tmp_path):
    mp = common.metrics_path_for(tmp_path)
    common.append_metric(mp, tool="t", aweme_id="x", status="ok", elapsed_s=1.5)
    common.append_metric(mp, tool="t", aweme_id="y", status="error", extra={"e": "中文"})
    lines = mp.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    r1, r2 = (json.loads(l) for l in lines)
    assert r1["tool"] == "t" and r1["elapsed_s"] == 1.5 and "ts" in r1
    assert r2["status"] == "error" and r2["extra"]["e"] == "中文"


def test_load_config_has_perception_dir():
    from src.config import load_config

    cfg = load_config()
    assert cfg.paths.perception_dir.is_absolute()
    assert isinstance(cfg.perception, dict)
    assert cfg.perception["omni"]["model_path"].endswith("qwen3-omni-30b-a3b-instruct")


def test_write_result_json_tolerates_concurrent_rename(tmp_path, monkeypatch):
    """2026-09-11 鬼灭 B∥C 同参考同 inspect 实锤：一方 os.replace 后另一方
    的 tmp 已不存在——目标已在即视为写入成功，不崩整跑。"""
    import os as os_mod

    import src.perception.common as common_mod

    calls = {"n": 0}
    real_replace = os_mod.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            from pathlib import Path

            Path(dst).touch()                      # 模拟并发方已写完目标
            raise FileNotFoundError
        return real_replace(src, dst)

    monkeypatch.setattr(common_mod.os, "replace", flaky_replace)
    path = common_mod.write_result_json(tmp_path, tool="t", aweme_id="a",
                                        params={}, output={"ok": 1})
    assert calls["n"] == 1 and path.exists()       # 未重试也未抛
