"""Run the isolated Qwen3-Omni v4 comparison; no production publication."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-json", type=Path)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    from src.agentic_video.reference_understanding_v4 import run_reference_understanding_v4
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    set_visible_gpus(args.gpu_pair)
    config = load_config(args.config)
    runner = OmniRunner(config.perception["omni"])
    try:
        value = run_reference_understanding_v4(
            args.reference, args.video, args.output, runner, asr_path=args.asr_json)
    except Exception as exc:
        # Keep all completed requests/responses untouched. A failed run is a
        # historical attempt; another attempt must use a new output directory.
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "run_failure.json").write_text(json.dumps({
            "schema_version": "reference_understanding_v4_failure",
            "error_type": type(exc).__name__, "error": str(exc),
            "reusable_as_completed_run": False}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise
    print(json.dumps({"status": value["status"],
                      "artifact_sha": value["artifact_sha"],
                      "model_call_count": value["model_call_count"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
