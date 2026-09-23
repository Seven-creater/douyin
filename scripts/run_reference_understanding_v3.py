"""Run the isolated v3 reference-understanding comparison."""
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
    parser.add_argument("--masked-video", type=Path, required=True)
    parser.add_argument("--baseline-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-json", type=Path)
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    from src.agentic_video.reference_understanding_v3 import (
        run_reference_understanding_v3)
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    set_visible_gpus(args.gpu_pair)
    config = load_config(args.config)
    runner = OmniRunner(config.perception["omni"])
    value = run_reference_understanding_v3(
        args.reference, args.video, args.masked_video, args.baseline_v2,
        args.output, runner, asr_path=args.asr_json,
        replay_from=args.replay_from)
    print(json.dumps({"status": value["status"],
                      "artifact_sha": value["artifact_sha"],
                      "local_media_call_count": value["local_media_call_count"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
