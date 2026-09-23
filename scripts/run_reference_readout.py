"""Prepare or run the isolated reference-understanding pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "model"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--masked-video", type=Path)
    parser.add_argument("--asr-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    from src.agentic_video.reference_readout import (
        prepare_readout, run_model_readout)

    if args.phase == "prepare":
        value = prepare_readout(args.reference, args.video, args.output,
                                asr_path=args.asr_json)
        print(json.dumps({"status": "prepared", "artifact_sha": value["artifact_sha"],
                          "shot_count": value["shot_count"],
                          "transition_count": value["transition_count"]}))
        return

    if args.masked_video is None:
        parser.error("--masked-video is required for model phase")
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    set_visible_gpus(args.gpu_pair)
    config = load_config(args.config)
    runner = OmniRunner(config.perception["omni"])
    value = run_model_readout(args.reference, args.video,
                              args.masked_video, args.output, runner,
                              asr_path=args.asr_json)
    print(json.dumps({"status": value["status"],
                      "artifact_sha": value["artifact_sha"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
