"""Run an isolated, text-only two-arm Story revision comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config")
    args = parser.parse_args()

    from src.agentic_video.creative_pipeline.critique_trial import (
        run_critique_trial,
    )
    from src.agentic_video.manifest import json_hash
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    root = Path(__file__).resolve().parents[1]
    baseline = Path(args.baseline_dir)
    structure = json.loads((root / "experiments/creative_structure_minimality/"
                            "r2_c5/creative_structure_spec_v1.json").read_text(
                                encoding="utf-8"))
    theme = json.loads((baseline / "01_theme/candidate.json").read_text(
        encoding="utf-8"))
    story = json.loads((baseline / "02_story/candidate.json").read_text(
        encoding="utf-8"))
    set_visible_gpus(args.gpu_pair)
    cfg = load_config(Path(args.config) if args.config else None)
    omni_cfg = cfg.perception.get("omni") or {}
    runner = OmniRunner(omni_cfg)
    result = run_critique_trial(
        runner, structure, theme, story, Path(args.output_dir),
        model_id=str(omni_cfg.get("model_path") or "omni"),
        model_config_sha=json_hash(omni_cfg))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
