from __future__ import annotations

import json

from src.agentic_video.cli import build_parser, main
from src.agentic_video.manifest import RunManifest, json_hash


def test_cli_exposes_required_subcommands():
    parser = build_parser()
    for command in ("discover", "benchmark", "index", "decompose", "render", "run"):
        if command == "discover":
            args = parser.parse_args([command, "--output", "o"])
            assert args.command == command
            continue
        if command == "benchmark":
            args = parser.parse_args([command, "--no-render"])
        elif command == "index":
            args = parser.parse_args([command])
        elif command == "decompose":
            args = parser.parse_args([command, "--reference", "r.mp4", "--output", "o"])
        elif command == "render":
            args = parser.parse_args([command, "--recipe", "r.json", "--theme", "t",
                                      "--output", "o"])
        else:
            args = parser.parse_args([command, "--reference", "r.mp4", "--theme", "t",
                                      "--output", "o"])
        assert args.command == command


def test_run_manifest_is_resumable_and_keeps_prior_stages(tmp_path):
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"x")
    path = tmp_path / "run_manifest.json"
    first = RunManifest(path, command="test", input_path=input_path, config={"a": 1},
                        repo_root=tmp_path)
    first.stage("one", "complete", value=2)
    second = RunManifest(path, command="ignored", input_path=None, config={}, repo_root=tmp_path)
    assert second.data["stages"]["one"]["value"] == 2
    assert len(second.data["input"]["sha256"]) == 64
    assert json_hash({"b": 2, "a": 1}) == json_hash({"a": 1, "b": 2})


def test_benchmark_cli_can_generate_specs_without_rendering(tmp_path):
    code = main(["benchmark", "--suite-dir", str(tmp_path / "suite"), "--no-render"])
    assert code == 0
    rows = (tmp_path / "suite" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 144
    assert json.loads(rows[0])["kind"] == "single"


def test_gpu_spec_requires_two_or_four_distinct_cards():
    """M4：只允许 2/4 张不重复数字卡号——旧实现任意串直通 CUDA_VISIBLE_DEVICES。"""
    import pytest

    from src.agentic_video.cli import _validated_gpu_spec

    assert _validated_gpu_spec("0,1") == "0,1"
    assert _validated_gpu_spec("0,1,2,3") == "0,1,2,3"
    assert _validated_gpu_spec(" 0, 1 ") == "0,1"          # 容忍空白/尾逗号（归一）
    assert _validated_gpu_spec("0,1,") == "0,1"
    for bad in ("0", "0,1,2", "0,1,2,3,5", "0,0", "a,b"):
        with pytest.raises(SystemExit):
            _validated_gpu_spec(bad)


def test_cli_module_does_not_import_cv2():
    """cv2 必须晚于 torch 进程（2026-09-10 服务器段错误）：cli 模块自身不得
    在 import 时拉起 cv2，否则 main() 的 torch 预载失去排序控制权。"""
    import os
    import subprocess
    import sys

    code = "import sys; import src.agentic_video.cli; print('cv2' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=env, cwd=os.getcwd())
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
