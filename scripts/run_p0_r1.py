"""Run the staged P0-R1 evidence repair and Creative DNA v2 candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["perceive", "validate", "candidate"],
                        required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-config", default="config/p0_r1_current.json")
    parser.add_argument("--source-video", default=None)
    parser.add_argument("--p04e-dir", default=None)
    parser.add_argument("--review-file")
    parser.add_argument("--holdout")
    parser.add_argument("--gpu-pair", default="0,1")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = _args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    source = Path(args.source_video or run_config["source_video"])
    p04e = Path(args.p04e_dir or run_config["legacy_p04e_dir"])
    from src.agentic_video.p0_r1 import (
        build_release_candidate, compile_validated_reference,
        run_visual_perception)

    if args.phase == "validate":
        if not args.review_file:
            raise SystemExit("--review-file is required for validate")
        validated, ledger, diff = compile_validated_reference(
            run_dir / "perception" / "visual_perception_draft.json", p04e,
            Path(args.review_file), run_dir / "validated")
        print(json.dumps({"status": "validated",
                          "validated_reference_sha": validated["artifact_sha"],
                          "claim_count": len(ledger["claims"]),
                          "diff": diff}, ensure_ascii=False))
        return

    from src.config import load_config
    from src.perception.omni_runner import OmniRunner, set_visible_gpus
    set_visible_gpus(args.gpu_pair)
    cfg = load_config(Path(args.config) if args.config else None)
    omni_config = cfg.perception.get("omni") or {}
    runner = OmniRunner(omni_config)
    if args.phase == "perceive":
        value = run_visual_perception(
            source, p04e, run_dir / "perception", runner=runner,
            mask_regions=list(run_config.get("mask_regions") or []),
            forbidden_markers=list(run_config.get("forbidden_markers") or []),
            mask_version=str(run_config.get("mask_version") or
                             "visual_mask_v1"))
        print(json.dumps({"status": "perception_complete",
                          "artifact_sha": value["artifact_sha"],
                          "output": str(run_dir / "perception")},
                         ensure_ascii=False))
        return

    result = build_release_candidate(
        run_dir / "validated" / "validated_reference.json",
        run_dir / "perception" / "visual_perception_draft.json",
        run_dir / "candidate", runner=runner, model_config=omni_config,
        repo_root=repo,
        holdout_path=Path(args.holdout) if args.holdout else None)
    print(json.dumps({"status": result["release_decision"]["status"],
                      "second_commit_allowed": result["release_decision"][
                          "second_commit_allowed"],
                      "output": str(run_dir / "candidate")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
