#!/usr/bin/env python3
"""Run the explicit 3-shot x 3-candidate R2-E.1 video acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agentic_video.creative_pipeline.fake_skills import build_fake_registry
from src.agentic_video.creative_pipeline.generation.real import RealVideoAdapter
from src.agentic_video.creative_pipeline.orchestrator import (
    CreativePipelineOrchestrator,
)
from src.agentic_video.generation_v9g import SGLangH3Client
from src.agentic_video.workspace import Workspace


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPEC = (
    ROOT / "experiments/creative_structure_minimality/r2_c5/"
    "creative_structure_spec_v1.json")
FORMAT_CONSTRAINTS = {
    "schema_version": "screenplay_format_constraints_v1",
    "target_duration_s": 30.0,
    "min_duration_s": 25.0,
    "max_duration_s": 35.0,
    "max_scenes": 3,
    "max_characters": 4,
    "max_props": 3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        raise SystemExit("real generation requires explicit --execute")
    if len(args.endpoint) != 3:
        raise SystemExit("exactly three --endpoint values are required")

    spec = json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    workspace = Workspace(args.workspace)
    orchestrator = CreativePipelineOrchestrator(
        workspace, build_fake_registry())
    if workspace.effective_status("creative:storyboard") != "committed":
        orchestrator.run_wave2(spec)
        orchestrator.run_wave3(
            spec, format_constraints=FORMAT_CONSTRAINTS)
        orchestrator.run_wave4(spec)

    clients = [SGLangH3Client(endpoint) for endpoint in args.endpoint]
    adapter = RealVideoAdapter(
        clients, model_id=args.model_id,
        model_version=args.model_version, model_sha=args.model_sha,
        aspect_ratio="9:16", inference_steps=args.inference_steps)
    result = orchestrator.run_r2e1_real_video_acceptance(
        adapter, output_dir=args.output_dir)
    result_path = args.workspace / "r2_e1_acceptance_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "shot_count": result["shot_count"],
        "candidate_count": result["candidate_count"],
        "result_path": str(result_path.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
