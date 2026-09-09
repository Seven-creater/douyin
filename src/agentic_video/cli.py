"""Unified command-line interface for Agentic Video Program Induction."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.config import ensure_utf8_stdio, load_config, repo_root, setup_logging
from src.perception import common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-video")
    parser.add_argument("--config", default=None)
    parser.add_argument("--gpus", default=None,
                        help="CUDA_VISIBLE_DEVICES for Qwen stages, e.g. 0,1")
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="collect and select meaningful popular videos")
    discover.add_argument("--date", default=None, help="end date YYYY-MM-DD; default today")
    discover.add_argument("--days", type=int, default=7)
    discover.add_argument("--per-category", type=int, default=4)
    discover.add_argument("--output", required=True)
    discover.add_argument("--refetch", action="store_true")
    discover.add_argument("--skip-collect", action="store_true")
    discover.add_argument("--skip-download", action="store_true")
    discover.add_argument("--skip-audition", action="store_true")
    discover.add_argument("--force", action="store_true")

    benchmark = sub.add_parser("benchmark", help="generate/evaluate controlled truth suite")
    benchmark.add_argument("--suite-dir", default="data/benchmarks/controlled_v1")
    benchmark.add_argument("--predictions", default=None)
    benchmark.add_argument("--no-render", action="store_true")
    benchmark.add_argument("--force", action="store_true")
    benchmark.add_argument("--fixed-metrics", default=None)
    benchmark.add_argument("--signal-metrics", default=None,
                            help="optional signal-guided (non-agent) metrics JSON")
    benchmark.add_argument("--agent-metrics", default=None)
    benchmark.add_argument("--fault-suite", default=None,
                            help="directory for the 60-case injected-fault suite")
    benchmark.add_argument("--fault-count", type=int, default=60)
    benchmark.add_argument("--fault-critiques", default=None,
                            help="directory containing <case_id>.critic.json outputs")
    benchmark.add_argument("--clean-critiques", default=None,
                            help="optional clean-case critic JSON directory for false positives")

    index = sub.add_parser("index", help="build a long-video narrative or high-action index")
    index.add_argument("--source", default="guimie")
    index.add_argument("--video", default=None)
    index.add_argument("--profile", choices=("high_action", "narrative"),
                       default="narrative")
    index.add_argument("--limit", type=int, default=None,
                       help="caption only N new shots for a GPU smoke test")
    index.add_argument("--limit-windows", type=int, default=None,
                       help="annotate only N narrative windows for a GPU smoke test")
    index.add_argument("--force", action="store_true")
    index.add_argument("--skip-captions", action="store_true")
    index.add_argument("--skip-embeddings", action="store_true")
    index.add_argument("--skip-annotations", action="store_true")

    decompose = sub.add_parser("decompose", help="reference video to Recipe v2")
    decompose.add_argument("--reference", required=True)
    decompose.add_argument("--output", required=True)
    decompose.add_argument("--mode", choices=("edit", "narrative", "both"), default="both")
    decompose.add_argument("--force", action="store_true")

    render = sub.add_parser("render", help="Recipe v2 plus theme to rendered video")
    render.add_argument("--recipe", required=True)
    render.add_argument("--theme", required=True)
    render.add_argument("--library", default="guimie")
    render.add_argument("--output", required=True)
    render.add_argument("--narrative-program", default=None)
    render.add_argument("--story-plan", default=None)
    render.add_argument("--target-duration", type=float, default=60.0)
    render.add_argument("--force", action="store_true")
    render.add_argument("--no-mask-backend", action="store_true")

    run = sub.add_parser("run", help="decompose, retrieve, render, and critique")
    run.add_argument("--reference", required=True)
    run.add_argument("--theme", required=True)
    run.add_argument("--library", default="guimie")
    run.add_argument("--output", required=True)
    run.add_argument("--target-duration", type=float, default=60.0)
    run.add_argument("--force", action="store_true")
    run.add_argument("--no-mask-backend", action="store_true")
    return parser


def _benchmark(args) -> dict:
    from src.agentic_video.benchmark import (compare_ablation,
                                             compare_ablation_variants,
                                             evaluate_suite, generate_suite)
    from src.agentic_video.critic_v2 import (build_fault_suite, score_fault_critiques,
                                              write_fault_suite)

    suite_dir = repo_root() / args.suite_dir if not Path(args.suite_dir).is_absolute() \
        else Path(args.suite_dir)
    manifest = generate_suite(suite_dir, render=not args.no_render, overwrite=args.force)
    result = {"manifest": str(manifest), "cases": 144}
    fault_dir = (Path(args.fault_suite) if args.fault_suite else suite_dir / "critic_faults")
    if not fault_dir.is_absolute():
        fault_dir = repo_root() / fault_dir
    faults = build_fault_suite(
        [json.loads(Path(row["truth"]).read_text(encoding="utf-8"))
         for row in (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line)],
        count=args.fault_count)
    fault_manifest = write_fault_suite(faults, fault_dir)
    result["fault_manifest"] = str(fault_manifest)
    result["fault_cases"] = len(faults)
    if args.fault_critiques:
        critiques_dir = Path(args.fault_critiques)
        critiques = [json.loads((critiques_dir / f"{row['case_id']}.critic.json")
                                .read_text(encoding="utf-8")) for row in faults]
        clean_rows = []
        if args.clean_critiques:
            clean_rows = [json.loads(path.read_text(encoding="utf-8"))
                          for path in sorted(Path(args.clean_critiques).glob("*.json"))]
        clean_false_positives = sum(bool(row.get("issues") or row.get("patches"))
                                    for row in clean_rows)
        fault_score = score_fault_critiques(
            faults, critiques, clean_false_positives=clean_false_positives,
            clean_count=len(clean_rows))
        (fault_dir / "score.json").write_text(
            json.dumps(fault_score, ensure_ascii=False, indent=2), encoding="utf-8")
        result["fault_score"] = fault_score
    if args.predictions:
        metrics = evaluate_suite(suite_dir, Path(args.predictions))
        metrics_path = suite_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        result["metrics"] = metrics
    if args.fixed_metrics and args.agent_metrics:
        fixed = json.loads(Path(args.fixed_metrics).read_text(encoding="utf-8"))
        agent = json.loads(Path(args.agent_metrics).read_text(encoding="utf-8"))
        signal = (json.loads(Path(args.signal_metrics).read_text(encoding="utf-8"))
                  if args.signal_metrics else None)
        ablation = (compare_ablation_variants(fixed, signal, agent)
                    if signal is not None else compare_ablation(fixed, agent))
        (suite_dir / "ablation.json").write_text(
            json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8")
        result["ablation"] = ablation
    return result


def _discover(args, cfg) -> dict:
    from src.agentic_video.discovery import run_discovery

    end_date = args.date or datetime.now().strftime("%Y-%m-%d")
    return run_discovery(
        cfg, end_date=end_date, days=args.days, per_category=args.per_category,
        output_dir=Path(args.output), refetch=args.refetch,
        collect=not args.skip_collect, download=not args.skip_download,
        audition=not args.skip_audition, force=args.force)


def _index(args, cfg) -> dict:
    from src.agentic_video.long_video import run_high_action_index
    from src.library.build_index import build
    from src.library.caption_shots import Qwen2VLRunner, caption_video

    if args.profile == "narrative":
        from src.agentic_video.narrative_index import run_narrative_index
        result_path = run_narrative_index(
            cfg, args.source, video=Path(args.video) if args.video else None,
            force=args.force)
    else:
        result_path = run_high_action_index(
            cfg, args.source, video=Path(args.video) if args.video else None,
            force=args.force)
    if not args.skip_captions:
        runner = Qwen2VLRunner(cfg.library.get("captions") or {})
        caption_video(cfg, result_path, force=args.force, limit=args.limit, runner=runner)
    annotations = None
    if args.profile == "narrative" and not args.skip_annotations:
        from src.agentic_video.narrative_index import run_narrative_annotations
        annotations = run_narrative_annotations(
            cfg, result_path, force=args.force, limit_windows=args.limit_windows)
    index_path = None
    if not args.skip_embeddings:
        index_path = build(cfg)
    env = json.loads(result_path.read_text(encoding="utf-8"))
    return {"shots": str(result_path), "n_shots": env["output"]["n_shots"],
            "profile": args.profile,
            "annotations": str(annotations) if annotations else None,
            "index": str(index_path) if index_path else None}


def _decompose(args, cfg) -> dict:
    from src.agentic_video.pipeline import (run_decomposition,
                                             run_narrative_decomposition)

    recipe = narrative = None
    vid = ""
    if args.mode in {"edit", "both"}:
        recipe, vid = run_decomposition(cfg, Path(args.reference), Path(args.output),
                                        force=args.force)
    if args.mode in {"narrative", "both"}:
        narrative, vid = run_narrative_decomposition(
            cfg, Path(args.reference), Path(args.output), force=args.force)
    return {
        "reference_id": vid, "mode": args.mode,
        "recipe": str(Path(args.output) / "reference.recipe.json") if recipe else None,
        "operations": len(recipe["operations"]) if recipe else None,
        "narrative": str(Path(args.output) / "reference.narrative.json") if narrative else None,
        "events": len(narrative["events"]) if narrative else None,
    }


def _render(args, cfg) -> dict:
    from src.agentic_video.manifest import RunManifest
    from src.agentic_video.pipeline import run_rendering

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    narrative_path = (Path(args.narrative_program) if args.narrative_program else
                      Path(args.recipe).with_name("reference.narrative.json"))
    narrative = (json.loads(narrative_path.read_text(encoding="utf-8"))
                 if narrative_path.exists() else None)
    story_plan = (json.loads(Path(args.story_plan).read_text(encoding="utf-8"))
                  if args.story_plan else None)
    output = Path(args.output)
    manifest = RunManifest(output / "run_manifest.json", command="render",
                           input_path=Path(args.recipe),
                           config={"theme": args.theme, "library": args.library,
                                   "narrative": bool(narrative),
                                   "target_duration": args.target_duration},
                           repo_root=repo_root())
    manifest.stage("render", "running")
    plan, retrieval, final = run_rendering(
        cfg, recipe, theme=args.theme, library=args.library, output_dir=output,
        force=args.force, use_mask_backend=not args.no_mask_backend,
        narrative=narrative, story_plan=story_plan,
        target_duration_s=args.target_duration)
    manifest.stage("render", "complete", output=str(final))
    return {"output": str(final), "slots": len(plan["slots"]),
            "missing": sum(bool(row.get("missing")) for row in retrieval)}


def _run(args, cfg) -> dict:
    from src.agentic_video.pipeline import run_full

    final = run_full(cfg, Path(args.reference), theme=args.theme, library=args.library,
                     output_dir=Path(args.output), force=args.force,
                     use_mask_backend=not args.no_mask_backend,
                     target_duration_s=args.target_duration)
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
    handlers = {"discover": _discover, "benchmark": _benchmark,
                "index": _index, "decompose": _decompose,
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
