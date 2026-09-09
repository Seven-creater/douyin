"""Unified command-line interface for Agentic Video Program Induction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import ensure_utf8_stdio, load_config, repo_root, setup_logging
from src.perception import common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-video")
    parser.add_argument("--config", default=None)
    parser.add_argument("--gpus", default=None,
                        help="CUDA_VISIBLE_DEVICES for Qwen stages, e.g. 0,1")
    sub = parser.add_subparsers(dest="command", required=True)

    benchmark = sub.add_parser("benchmark", help="generate/evaluate controlled truth suite")
    benchmark.add_argument("--suite-dir", default="data/benchmarks/controlled_v1")
    benchmark.add_argument("--predictions", default=None)
    benchmark.add_argument("--no-render", action="store_true")
    benchmark.add_argument("--force", action="store_true")
    benchmark.add_argument("--fixed-metrics", default=None)
    benchmark.add_argument("--agent-metrics", default=None)

    index = sub.add_parser("index", help="build a high-action long-video index")
    index.add_argument("--source", default="guimie")
    index.add_argument("--video", default=None)
    index.add_argument("--limit", type=int, default=None,
                       help="caption only N new shots for a GPU smoke test")
    index.add_argument("--force", action="store_true")
    index.add_argument("--skip-captions", action="store_true")
    index.add_argument("--skip-embeddings", action="store_true")

    decompose = sub.add_parser("decompose", help="reference video to Recipe v2")
    decompose.add_argument("--reference", required=True)
    decompose.add_argument("--output", required=True)
    decompose.add_argument("--force", action="store_true")

    render = sub.add_parser("render", help="Recipe v2 plus theme to rendered video")
    render.add_argument("--recipe", required=True)
    render.add_argument("--theme", required=True)
    render.add_argument("--library", default="guimie")
    render.add_argument("--output", required=True)
    render.add_argument("--force", action="store_true")
    render.add_argument("--no-mask-backend", action="store_true")

    run = sub.add_parser("run", help="decompose, retrieve, render, and critique")
    run.add_argument("--reference", required=True)
    run.add_argument("--theme", required=True)
    run.add_argument("--library", default="guimie")
    run.add_argument("--output", required=True)
    run.add_argument("--force", action="store_true")
    run.add_argument("--no-mask-backend", action="store_true")
    return parser


def _benchmark(args) -> dict:
    from src.agentic_video.benchmark import (compare_ablation, evaluate_suite,
                                             generate_suite)

    suite_dir = repo_root() / args.suite_dir if not Path(args.suite_dir).is_absolute() \
        else Path(args.suite_dir)
    manifest = generate_suite(suite_dir, render=not args.no_render, overwrite=args.force)
    result = {"manifest": str(manifest), "cases": 144}
    if args.predictions:
        metrics = evaluate_suite(suite_dir, Path(args.predictions))
        metrics_path = suite_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        result["metrics"] = metrics
    if args.fixed_metrics and args.agent_metrics:
        fixed = json.loads(Path(args.fixed_metrics).read_text(encoding="utf-8"))
        agent = json.loads(Path(args.agent_metrics).read_text(encoding="utf-8"))
        ablation = compare_ablation(fixed, agent)
        (suite_dir / "ablation.json").write_text(
            json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8")
        result["ablation"] = ablation
    return result


def _index(args, cfg) -> dict:
    from src.agentic_video.long_video import run_high_action_index
    from src.library.build_index import build
    from src.library.caption_shots import Qwen2VLRunner, caption_video

    result_path = run_high_action_index(
        cfg, args.source, video=Path(args.video) if args.video else None, force=args.force)
    if not args.skip_captions:
        runner = Qwen2VLRunner(cfg.library.get("captions") or {})
        caption_video(cfg, result_path, force=args.force, limit=args.limit, runner=runner)
    index_path = None
    if not args.skip_embeddings:
        index_path = build(cfg)
    env = json.loads(result_path.read_text(encoding="utf-8"))
    return {"shots": str(result_path), "n_shots": env["output"]["n_shots"],
            "index": str(index_path) if index_path else None}


def _decompose(args, cfg) -> dict:
    from src.agentic_video.pipeline import run_decomposition

    recipe, vid = run_decomposition(cfg, Path(args.reference), Path(args.output),
                                    force=args.force)
    return {"reference_id": vid, "recipe": str(Path(args.output) / "reference.recipe.json"),
            "operations": len(recipe["operations"])}


def _render(args, cfg) -> dict:
    from src.agentic_video.manifest import RunManifest
    from src.agentic_video.pipeline import run_rendering

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    output = Path(args.output)
    manifest = RunManifest(output / "run_manifest.json", command="render",
                           input_path=Path(args.recipe),
                           config={"theme": args.theme, "library": args.library},
                           repo_root=repo_root())
    manifest.stage("render", "running")
    plan, retrieval, final = run_rendering(
        cfg, recipe, theme=args.theme, library=args.library, output_dir=output,
        force=args.force, use_mask_backend=not args.no_mask_backend)
    manifest.stage("render", "complete", output=str(final))
    return {"output": str(final), "slots": len(plan["slots"]),
            "missing": sum(bool(row.get("missing")) for row in retrieval)}


def _run(args, cfg) -> dict:
    from src.agentic_video.pipeline import run_full

    final = run_full(cfg, Path(args.reference), theme=args.theme, library=args.library,
                     output_dir=Path(args.output), force=args.force,
                     use_mask_backend=not args.no_mask_backend)
    return {"output": str(final), "report": str(Path(args.output) / "report.md")}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gpus:
        from src.perception.omni_runner import set_visible_gpus
        set_visible_gpus(args.gpus)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="agentic_video")
    handlers = {"benchmark": _benchmark, "index": _index, "decompose": _decompose,
                "render": _render, "run": _run}
    try:
        result = handlers[args.command](args, cfg) if args.command != "benchmark" \
            else handlers[args.command](args)
        common.emit_status_line("ok", command=args.command, **result)
        return 0
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception("agentic-video %s failed", args.command)
        common.emit_status_line("error", command=args.command, error=str(exc)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
