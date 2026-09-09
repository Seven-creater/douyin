from __future__ import annotations

import json

from src.agentic_video.cli import build_parser, main
from src.agentic_video.manifest import RunManifest, json_hash


def test_cli_exposes_five_required_subcommands():
    parser = build_parser()
    for command in ("benchmark", "index", "decompose", "render", "run"):
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
