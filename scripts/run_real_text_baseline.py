"""Run a one-chain creative baseline on a text-capable Omni worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config")
    parser.add_argument("--user-brief")
    args = parser.parse_args()

    from src.agentic_video.creative_pipeline.real_text_baseline import (
        run_real_text_baseline,
    )
    from src.agentic_video.manifest import json_hash
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    root = Path(__file__).resolve().parents[1]
    structure = json.loads((root / "experiments/creative_structure_minimality/"
                            "r2_c5/creative_structure_spec_v1.json").read_text(
                                encoding="utf-8"))
    brief = json.loads(Path(args.user_brief).read_text(encoding="utf-8")) \
        if args.user_brief else {}
    fmt = {"schema_version": "screenplay_format_constraints_v1",
           "target_duration_s": 30.0, "min_duration_s": 25.0,
           "max_duration_s": 35.0, "max_scenes": 3,
           "max_characters": 4, "max_props": 3}
    set_visible_gpus(args.gpu_pair)
    cfg = load_config(Path(args.config) if args.config else None)
    omni_cfg = cfg.perception.get("omni") or {}
    runner = OmniRunner(omni_cfg)
    result = run_real_text_baseline(
        runner, structure, Path(args.output_dir), format_constraints=fmt,
        user_brief=brief, model_id=str(omni_cfg.get("model_path") or "omni"),
        model_config_sha=json_hash(omni_cfg))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
