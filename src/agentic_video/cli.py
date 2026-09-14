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
    index.add_argument("--windows", default=None,
                       help="comma-separated window indexes for targeted annotation "
                            "(3-window pack-on/off gate)")
    index.add_argument("--force", action="store_true")
    index.add_argument("--skip-captions", action="store_true")
    index.add_argument("--skip-embeddings", action="store_true")
    index.add_argument("--skip-annotations", action="store_true")

    facets = sub.add_parser(
        "facets", help="type-dimension facet annotation for a narrative library source")
    facets.add_argument("--source", default="guimie")
    facets.add_argument("--windows", default=None,
                        help="comma-separated window indexes for a pilot run, e.g. 0,1,2")
    facets.add_argument("--force", action="store_true")

    bootstrap = sub.add_parser(
        "bootstrap", help="P1.5 Film Knowledge Bootstrap: identify film, build knowledge pack")
    bootstrap.add_argument("--source", required=True)
    bootstrap.add_argument("--force", action="store_true")
    bootstrap.add_argument("--verify", action="store_true",
                           help="标注后跑绑定状态机 + source_entities 回填")
    bootstrap.add_argument("--gt", default=None,
                           help="人工 GT JSON（verified_entities 列表）——verified 只认独立证据")
    bootstrap.add_argument("--mask-metadata", action="store_true",
                           help="benchmark 模式：Identity Gate 不看文件名/元数据")

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

    roughcut = sub.add_parser(
        "roughcut", help="scene-scoped editorial control experiment (V5.1)")
    roughcut.add_argument("--spec", default="config/roughcuts/lxh1_w15.json",
                          help="RoughcutSpec JSON (default: lxh1_w15)")
    roughcut.add_argument("--source", default=None,
                          help="legacy override; must match spec source")
    roughcut.add_argument("--window", type=int, default=None,
                          help="legacy window label (spec base_window is authoritative)")
    roughcut.add_argument("--theme", default="")
    roughcut.add_argument("--output", required=True)
    roughcut.add_argument("--target-duration", type=float, default=None)
    roughcut.add_argument("--unverified", action="store_true",
                          help="offline control test only; never deliverable")
    roughcut.add_argument("--human-acceptance", default=None,
                          help="human acceptance JSON; normally use roughcut-accept after review")
    roughcut.add_argument("--force", action="store_true")

    roughcut_accept = sub.add_parser(
        "roughcut-accept", help="finalize an automated roughcut candidate after human review")
    roughcut_accept.add_argument("--output", required=True,
                                 help="existing roughcut output directory")
    roughcut_accept.add_argument("--human-acceptance", required=True,
                                 help="human acceptance JSON")

    pattern = sub.add_parser(
        "reference-pattern", help="extract V6 editing grammar from a reference video")
    pattern.add_argument("--reference", required=True)
    pattern.add_argument("--output", required=True)
    pattern.add_argument("--with-omni", action="store_true",
                         help="optionally enrich the deterministic pattern with Omni")

    evidence = sub.add_parser(
        "evidence-mine", help="mine fine-grained V6 visual evidence units")
    evidence.add_argument("--video", required=True)
    evidence.add_argument("--shots", required=True)
    evidence.add_argument("--scope-start", type=float, required=True)
    evidence.add_argument("--scope-end", type=float, required=True)
    evidence.add_argument("--output", required=True)
    evidence.add_argument("--unverified", action="store_true",
                          help="offline parser test only; never deliverable")

    evidence_plan = sub.add_parser(
        "evidence-plan", help="plan a micro montage from evidence_units.json")
    evidence_plan.add_argument("--evidence", required=True)
    evidence_plan.add_argument("--pattern", required=True)
    evidence_plan.add_argument("--output", required=True)
    evidence_plan.add_argument("--preferred-duration", type=float, default=12.0)
    evidence_plan.add_argument("--min-duration", type=float, default=4.0)
    evidence_plan.add_argument("--max-duration", type=float, default=20.0)

    evidence_render = sub.add_parser(
        "evidence-render", help="render a V6 evidence edit plan and audio variants")
    evidence_render.add_argument("--plan", required=True)
    evidence_render.add_argument("--video", default=None)
    evidence_render.add_argument("--output", required=True)
    evidence_render.add_argument("--bgm", default=None)
    evidence_render.add_argument("--force", action="store_true")

    evidence_v6 = sub.add_parser(
        "evidence-v6", help="run the complete V6 evidence-centric control loop")
    evidence_v6.add_argument(
        "--spec", default="config/roughcuts/lxh1_w15_evidence_v6.json")
    evidence_v6.add_argument("--video", default=None,
                             help="optional source movie override")
    evidence_v6.add_argument("--output", required=True)
    evidence_v6.add_argument(
        "--gpu-pairs", default=None,
        help="parallel Omni workers, e.g. '0,1;2,3;4,5;6,7'")
    evidence_v6.add_argument("--worker-timeout", type=float, default=3600.0)
    evidence_v6.add_argument("--human-acceptance", default=None)
    evidence_v6.add_argument("--force", action="store_true")

    evidence_v6_accept = sub.add_parser(
        "evidence-v6-accept", help="release V6 output after human review")
    evidence_v6_accept.add_argument("--output", required=True)
    evidence_v6_accept.add_argument("--human-acceptance", required=True)

    evidence_v61 = sub.add_parser(
        "evidence-v61-diagnostic",
        help="run V6.1 coarse/dense/oracle Evidence diagnostics")
    evidence_v61.add_argument(
        "--spec", default="config/experiments/lxh1_v61_diagnostic.json")
    evidence_v61.add_argument(
        "--oracle", default="config/oracles/lxh1_5385_5460.json")
    evidence_v61.add_argument("--video", default=None)
    evidence_v61.add_argument("--output", required=True)
    evidence_v61.add_argument(
        "--gpu-pairs", default=None,
        help="parallel Omni workers, e.g. '0,1;2,3;4,5;6,7'")
    evidence_v61.add_argument("--worker-timeout", type=float, default=3600.0)

    evidence_v7 = sub.add_parser(
        "evidence-v7-target",
        help="run one phase of the V7 target-centered no-detector experiment")
    evidence_v7.add_argument(
        "--phase", required=True, choices=("prepare", "browse", "verify"))
    evidence_v7.add_argument(
        "--spec", default="config/experiments/lxh1_v7_target.json")
    evidence_v7.add_argument("--video", default=None)
    evidence_v7.add_argument("--output", required=True)
    evidence_v7.add_argument(
        "--gpu-pairs", default=None,
        help="Omni pairs for prepare/verify, e.g. '0,1;2,3;4,5;6,7'")
    evidence_v7.add_argument("--worker-timeout", type=float, default=3600.0)
    evidence_v7.add_argument("--force", action="store_true")

    evidence_v7_accept = sub.add_parser(
        "evidence-v7-accept", help="release a V7 output after human review")
    evidence_v7_accept.add_argument("--output", required=True)
    evidence_v7_accept.add_argument("--human-acceptance", required=True)

    run = sub.add_parser("run", help="decompose, retrieve, render, and critique")
    run.add_argument("--reference", required=True)
    run.add_argument("--theme", required=True)
    run.add_argument("--library", default="guimie")
    run.add_argument("--output", required=True)
    run.add_argument("--target-duration", type=float, default=60.0)
    run.add_argument("--force", action="store_true")
    run.add_argument("--no-mask-backend", action="store_true")
    run.add_argument("--no-verification", action="store_true",
                     help="skip slot verification + layered re-search (P4 B-arm)")
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
        # 卸载 caption 模型再进 Omni 标注（2026-09-11 film1 实锤：Qwen2-VL 驻留
        # + Omni 34GB/卡 在同进程叠加 → 46.77GB OOM；guimie 时代 caption 与
        # 标注分进程跑没踩过，今天同进程串跑才爆）
        runner.unload()
    annotations = None
    if args.profile == "narrative" and not args.skip_annotations:
        from src.agentic_video.narrative_index import run_narrative_annotations
        annotations = run_narrative_annotations(
            cfg, result_path, force=args.force, limit_windows=args.limit_windows,
            windows=([int(w) for w in str(args.windows).split(",") if w.strip()]
                     if args.windows else None))
    index_path = None
    if not args.skip_embeddings:
        index_path = build(cfg)
    env = json.loads(result_path.read_text(encoding="utf-8"))
    return {"shots": str(result_path), "n_shots": env["output"]["n_shots"],
            "profile": args.profile,
            "annotations": str(annotations) if annotations else None,
            "index": str(index_path) if index_path else None}


def _facets(args, cfg) -> dict:
    from src.agentic_video.narrative_index import run_type_facets

    windows = None
    if args.windows:
        windows = [int(value) for value in str(args.windows).split(",") if value.strip()]
    path = run_type_facets(cfg, args.source, windows=windows, force=args.force)
    state = json.loads(path.read_text(encoding="utf-8"))
    n_facets = sum(len((row or {}).get("facets") or [])
                   for row in (state.get("windows") or {}).values())
    return {"facets": str(path),
            "windows_done": len(state.get("completed") or {}),
            "facets_total": n_facets}


def _bootstrap(args, cfg) -> dict:
    from src.library.film_bootstrap import (bootstrap_film_knowledge,
                                            update_pack_from_annotations)

    pack_path = bootstrap_film_knowledge(
        cfg, args.source, force=args.force, mask_metadata=args.mask_metadata)
    report = None
    if args.verify:
        gt = Path(args.gt) if args.gt else None
        report = update_pack_from_annotations(cfg, args.source, gt_path=gt)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    return {"pack": str(pack_path),
            "tier": (pack.get("work_identity") or {}).get("tier"),
            "title": (pack.get("work_identity") or {}).get("title"),
            "entities": len(pack.get("entities") or []),
            "verify": report}


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
            # V3 修正：missing 只数 unsupported 槽——旧口径把带 re_searched 溯源
            # 说明的 supported 槽也误计成缺素材（C2 wide 渲染 missing:2 假警报）
            "missing": sum(1 for row in retrieval if (row.get("picked") is None))}


def _roughcut(args, cfg) -> dict:
    from src.agentic_video.roughcut import run_roughcut
    runner = None
    if not args.unverified:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {},
                            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))

    final = run_roughcut(cfg, args.source, args.window, theme=args.theme,
                         output_dir=Path(args.output),
                         target_duration_s=args.target_duration, force=args.force,
                         spec=Path(args.spec), runner=runner,
                         allow_unverified=args.unverified,
                         human_acceptance=(Path(args.human_acceptance)
                                           if args.human_acceptance else None))
    return {"output": str(final)}


def _roughcut_accept(args, _cfg) -> dict:
    from src.agentic_video.roughcut import finalize_roughcut_delivery

    final = finalize_roughcut_delivery(Path(args.output),
                                       Path(args.human_acceptance))
    return {"output": str(final)}


def _reference_pattern(args, cfg) -> dict:
    from src.agentic_video.reference_pattern import extract_reference_pattern
    runner = None
    if args.with_omni:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {},
                            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    pattern = extract_reference_pattern(
        Path(args.reference), runner=runner, output=Path(args.output),
        ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
        ffprobe_bin=cfg.perception.get("ffprobe_bin", "ffprobe"))
    return {"output": str(args.output), "pattern_type": pattern["pattern_type"],
            "shot_count": pattern["shot_count"]}


def _evidence_mine(args, cfg) -> dict:
    from src.perception.evidence_miner import mine_evidence
    shots = json.loads(Path(args.shots).read_text(encoding="utf-8"))
    runner = None
    if not args.unverified:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {},
                            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    result = mine_evidence(
        Path(args.video), shots, runner=runner,
        scope_interval=(args.scope_start, args.scope_end), output=Path(args.output),
        clip_dir=Path(args.output).parent / "omni_clips",
        allow_unverified=args.unverified)
    return {"output": str(args.output), "units": len(result["units"]),
            "passed": result["passed"], "failure_class": result["failure_class"]}


def _evidence_plan(args, _cfg) -> dict:
    from src.agentic_video.evidence_planner import (build_evidence_edit_plan,
                                                    write_evidence_edit_plan)
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    pattern = json.loads(Path(args.pattern).read_text(encoding="utf-8"))
    plan = build_evidence_edit_plan(
        evidence, pattern, preferred_duration_s=args.preferred_duration,
        min_duration_s=args.min_duration, max_duration_s=args.max_duration)
    write_evidence_edit_plan(plan, Path(args.output))
    return {"output": str(args.output), "segments": len(plan["segments"]),
            "duration_s": plan["duration_s"], "passed": plan["passed"],
            "failure_class": plan["failure_class"]}


def _evidence_render(args, cfg) -> dict:
    from src.agentic_video.renderer import render_micro_montage
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    bgm = Path(args.bgm) if args.bgm else None
    if bgm is None:
        configured = str((cfg.library.get("narrative_render") or {}).get("bgm_path") or "")
        if configured:
            bgm = Path(configured) if Path(configured).is_absolute() else repo_root() / configured
    result = render_micro_montage(
        cfg, plan, Path(args.output), source_video=Path(args.video) if args.video else None,
        bgm_path=bgm, force=args.force)
    return {"content_master": str(result["content_master"]),
            "variants": {key: str(value) for key, value in (result["variants"] or {}).items()},
            "duration_s": result["duration_s"]}


def _evidence_v6(args, cfg) -> dict:
    from src.agentic_video.evidence_pipeline import (read_evidence_spec,
                                                      run_evidence_pipeline)
    from src.perception.omni_pool import OmniProcessPool

    spec_path = Path(args.spec)
    spec, _ = read_evidence_spec(spec_path)
    configured_pairs = (spec.get("analysis") or {}).get("gpu_pairs") or []
    pairs = args.gpu_pairs or ";".join(str(pair) for pair in configured_pairs)
    if not pairs:
        raise ValueError("evidence-v6 requires --gpu-pairs or analysis.gpu_pairs")
    with OmniProcessPool(
            pairs, cfg.perception.get("omni") or {},
            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            response_timeout_s=args.worker_timeout) as runner:
        result = run_evidence_pipeline(
            cfg, spec, Path(args.output), runner=runner,
            source_video=Path(args.video) if args.video else None,
            human_acceptance=(Path(args.human_acceptance)
                              if args.human_acceptance else None),
            force=args.force)
    acceptance = result["acceptance"]
    return {"output": str(result["output_dir"]),
            "automated_passed": acceptance.get("automated_passed", False),
            "passed": acceptance.get("passed", False),
            "failure_class": acceptance.get("failure_class"),
            "delivery": acceptance.get("delivery", "blocked")}


def _evidence_v6_accept(args, _cfg) -> dict:
    from src.agentic_video.evidence_pipeline import finalize_evidence_delivery

    final = finalize_evidence_delivery(Path(args.output), Path(args.human_acceptance))
    return {"output": str(final), "delivery": "passed"}


def _evidence_v61_diagnostic(args, cfg) -> dict:
    from src.agentic_video.evidence_diagnostic import (
        read_diagnostic_spec, run_evidence_diagnostic,
    )
    from src.perception.omni_pool import OmniProcessPool

    spec, _ = read_diagnostic_spec(Path(args.spec))
    configured = (spec.get("perception") or {}).get("gpu_pairs") or []
    pairs = args.gpu_pairs or ";".join(str(pair) for pair in configured)
    if not pairs:
        raise ValueError(
            "evidence-v61-diagnostic requires --gpu-pairs or perception.gpu_pairs")
    with OmniProcessPool(
            pairs, cfg.perception.get("omni") or {},
            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            response_timeout_s=args.worker_timeout) as runner:
        result = run_evidence_diagnostic(
            cfg, spec, Path(args.output), runner=runner,
            oracle_path=Path(args.oracle),
            source_video=Path(args.video) if args.video else None)
    acceptance = result["acceptance"]
    return {
        "output": str(result["output_dir"]),
        "diagnostic_completed": acceptance.get("diagnostic_completed", False),
        "failure_class": acceptance.get("failure_class"),
        "failure_stage": acceptance.get("failure_stage"),
        "delivery": acceptance.get("delivery", "blocked"),
    }


def _v7_clients(spec: dict) -> dict:
    from src.perception.flashvid_client import FlashVIDClient, FlashVIDEndpoint

    model = str(spec.get("flashvid_model") or "Qwen3.5-4B")
    return {
        arm: FlashVIDClient(FlashVIDEndpoint(
            arm=arm, base_url=str(row["base_url"]), model=model,
            fps=float(row["fps"]), retention_ratio=float(row["retention_ratio"]),
            backend=str(row["backend"])))
        for arm, row in (spec.get("browse_arms") or {}).items()
    }


def _v7_paths(spec: dict, args) -> tuple[Path, Path, Path, Path]:
    root = repo_root()
    source = Path(args.video or spec["source_video"])
    if not source.is_absolute():
        source = root / source
    reference = Path(spec["reference_video"])
    if not reference.is_absolute():
        reference = root / reference
    bgm = Path(spec["bgm_path"])
    if not bgm.is_absolute():
        bgm = root / bgm
    oracle = Path(spec["oracle"])
    if not oracle.is_absolute():
        oracle = root / oracle
    return source, reference, bgm, oracle


def _v7_omni_pairs(spec: dict, args) -> str:
    configured = (spec.get("perception") or {}).get("gpu_pairs") or []
    pairs = args.gpu_pairs or ";".join(str(pair) for pair in configured)
    if not pairs:
        raise ValueError("V7 prepare/verify requires --gpu-pairs")
    return pairs


def _evidence_v7_target_impl(args, cfg) -> dict:
    from src.agentic_video.recipe_v2 import sha256_file
    from src.agentic_video.target_v7 import (
        build_reference_driven_edit_plan, build_target_album,
        collect_flashvid_runtime, export_frame, finalize_target_microcut, oracle_evidence_bank,
        evaluate_target_album, prepare_reference_task, read_v7_spec, run_browse_matrix,
        verify_target_evidence,
    )

    spec_path = Path(args.spec)
    spec, spec_sha = read_v7_spec(spec_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, reference, bgm, oracle_path = _v7_paths(spec, args)
    manifest_path = output / "run_manifest.json"
    previous_manifest = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_manifest = {}
    manifest = {
        "schema_version": "v7_run_manifest_v1", "phase": args.phase,
        "spec": str(spec_path.resolve()), "spec_sha256": spec_sha,
        "source_video": str(source), "reference_video": str(reference),
        "custom_flashvid_port": True, "base_model": "Qwen3.5-4B",
        "flashvid_official_support_claimed": False,
        "flashvid_runtime": collect_flashvid_runtime(Path(spec["flashvid_runtime"])),
        "phases": dict(previous_manifest.get("phases") or {}),
    }
    if source.is_file():
        source_size = source.stat().st_size
        manifest["source_size_bytes"] = source_size
        cached_source_hash = (previous_manifest.get("source_sha256")
                              if previous_manifest.get("source_video") == str(source) and
                              previous_manifest.get("source_size_bytes") == source_size
                              else None)
        manifest["source_sha256"] = cached_source_hash or sha256_file(source)
    if reference.is_file():
        manifest["reference_sha256"] = sha256_file(reference)
    clients = _v7_clients(spec)

    if args.phase == "prepare":
        from src.perception.omni_pool import OmniProcessPool

        seed_path = output / "target_album" / "target_seed.jpg"
        export_frame(cfg.perception.get("ffmpeg_bin", "ffmpeg"), source,
                     float(spec["trusted_seed"]["source_time_s"]), seed_path)
        with OmniProcessPool(
                _v7_omni_pairs(spec, args), cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            reference_task = prepare_reference_task(
                cfg, spec, output, vanilla_client=clients["D"])
            album = build_target_album(
                cfg, spec, output / "target_album", trusted_seed=seed_path,
                source_video=source, vanilla_client=clients["D"], runner=runner)
        manifest.update({"reference_task_sha256": sha256_file(
            output / "reference_task.json"), "target_album_sha256": sha256_file(
                output / "target_album" / "target_album.json")})
        result = {"reference_sections": len(reference_task["edit_sections"]),
                  "album_positive": len(album["positive"]),
                  "album_hard_negative": len(album["hard_negative"])}
    elif args.phase == "browse":
        reference_task = json.loads((output / "reference_task.json").read_text(
            encoding="utf-8"))
        album = json.loads((output / "target_album" / "target_album.json").read_text(
            encoding="utf-8"))
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        browse = run_browse_matrix(
            cfg, spec, output / "browse", clients=clients, source_video=source,
            reference_task=reference_task, target_album=album, oracle=oracle,
            reuse_completed=not args.force)
        result = {"arms": {arm: len(row["candidates"])
                           for arm, row in browse["arms"].items()},
                  "diagnosis": browse["comparison"]["diagnostic_attribution"]}
    else:
        from src.perception.omni_pool import OmniProcessPool
        from src.agentic_video.target_v7 import V7Blocked

        reference_task = json.loads((output / "reference_task.json").read_text(
            encoding="utf-8"))
        album = json.loads((output / "target_album" / "target_album.json").read_text(
            encoding="utf-8"))
        arms = {arm: json.loads((output / "browse" / "arms" / arm /
                                 "browse_results.json").read_text(encoding="utf-8"))
                for arm in ("A", "B", "C", "D")}
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        automatic_result = None
        automatic_failure = None
        oracle_result = None
        oracle_failure = None
        with OmniProcessPool(
                _v7_omni_pairs(spec, args), cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            try:
                automatic_bank = verify_target_evidence(
                    cfg, spec, output / "agent", runner=runner,
                    source_video=source, browse_arms=arms, target_album=album)
                plan = build_reference_driven_edit_plan(reference_task, automatic_bank)
                (output / "edit_plan.json").write_text(
                    json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                if plan.get("passed"):
                    automatic_result = finalize_target_microcut(
                        cfg, plan, output, source_video=source, bgm_path=bgm,
                        runner=runner, force=args.force)
                else:
                    automatic_failure = {
                        "failure_class": plan.get("failure_class", "verification"),
                        "failure_stage": plan.get("failure_stage", "planning"),
                        "reason_code": plan.get("reason_code", "automatic_plan_blocked"),
                        "detail": str(plan.get("missing_goals") or ""),
                    }
            except V7Blocked as exc:
                automatic_failure = {
                    "failure_class": ("content" if exc.failure_stage == "browsing"
                                      else "verification"),
                    "failure_stage": exc.failure_stage,
                    "reason_code": exc.reason_code,
                    "detail": str(exc),
                }
                blocked_plan = {
                    "schema_version": "reference_driven_edit_plan_v1",
                    "passed": False, **automatic_failure, "segments": [],
                }
                (output / "edit_plan.json").write_text(
                    json.dumps(blocked_plan, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            except Exception as exc:  # automatic failure must not suppress Oracle
                automatic_failure = {
                    "failure_class": "infrastructure",
                    "failure_stage": "verification",
                    "reason_code": "automatic_verification_runtime_failure",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                blocked_plan = {
                    "schema_version": "reference_driven_edit_plan_v1",
                    "passed": False, **automatic_failure, "segments": [],
                }
                (output / "edit_plan.json").write_text(
                    json.dumps(blocked_plan, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            oracle_bank = oracle_evidence_bank(oracle, source)
            oracle_dir = output / "arms" / "oracle"
            oracle_dir.mkdir(parents=True, exist_ok=True)
            (oracle_dir / "evidence_bank.json").write_text(
                json.dumps(oracle_bank, ensure_ascii=False, indent=2), encoding="utf-8")
            oracle_plan = build_reference_driven_edit_plan(reference_task, oracle_bank)
            (oracle_dir / "edit_plan.json").write_text(
                json.dumps(oracle_plan, ensure_ascii=False, indent=2), encoding="utf-8")
            if oracle_plan.get("passed"):
                try:
                    oracle_result = finalize_target_microcut(
                        cfg, oracle_plan, oracle_dir, source_video=source, bgm_path=bgm,
                        runner=runner, force=args.force)
                except V7Blocked as exc:
                    oracle_failure = {
                        "failure_class": "verification",
                        "failure_stage": exc.failure_stage,
                        "reason_code": exc.reason_code,
                        "detail": str(exc),
                    }
                    (oracle_dir / "acceptance.json").write_text(json.dumps({
                        "schema_version": "v7_acceptance_v1",
                        "automated_passed": False, "human_passed": None,
                        "passed": False, "delivery": "blocked", **oracle_failure,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                oracle_failure = {
                    "failure_class": oracle_plan.get("failure_class", "verification"),
                    "failure_stage": oracle_plan.get("failure_stage", "planning"),
                    "reason_code": oracle_plan.get("reason_code", "oracle_plan_blocked"),
                    "detail": str(oracle_plan.get("missing_goals") or ""),
                }
                (oracle_dir / "acceptance.json").write_text(json.dumps({
                    "schema_version": "v7_acceptance_v1",
                    "automated_passed": False, "human_passed": None,
                    "passed": False, "delivery": "blocked", **oracle_failure,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        comparison = json.loads((output / "browse" / "arm_comparison.json").read_text(
            encoding="utf-8"))
        identity_audit = evaluate_target_album(album, oracle)
        (output / "target_album" / "heldout_identity_audit.json").write_text(
            json.dumps(identity_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        best_hits = max(int(row.get("candidate_temporal_hits") or 0)
                        for row in comparison["metrics"].values())
        browse_failures = sum(len(row.get("failures") or []) for row in arms.values())
        automatic_passed = bool(automatic_result and
                                automatic_result["acceptance"]["automated_passed"])
        oracle_passed = bool(oracle_result and
                             oracle_result["acceptance"]["automated_passed"])
        if identity_audit["hard_negative_false_merges"]:
            first_failure = ("identity", "heldout_hard_negative_false_merge")
        elif best_hits < 4:
            first_failure = ("browsing", "browse_gt_recall_below_threshold")
        elif automatic_failure:
            first_failure = (automatic_failure["failure_stage"],
                             automatic_failure["reason_code"])
        elif not oracle_passed:
            first_failure = ((oracle_failure or {}).get("failure_stage", "blind"),
                             (oracle_failure or {}).get("reason_code",
                                                        "oracle_control_failed"))
        else:
            first_failure = ("acceptance", "v7_automatic_acceptance_failed")
        module_diagnosis = {
            "schema_version": "v7_module_diagnosis_v1",
            "first_failure_stage": first_failure[0],
            "reason_code": first_failure[1],
            "automatic_failure": automatic_failure,
            "oracle_failure": oracle_failure,
            "identity_audit": identity_audit,
            "browse_gt_hits_best_arm": best_hits,
            "automatic_plan_blind_passed": automatic_passed,
            "oracle_plan_blind_passed": oracle_passed,
            "oracle_executed_despite_automatic_block": True,
        }
        metrics_dir = output / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "module_diagnosis.json").write_text(
            json.dumps(module_diagnosis, ensure_ascii=False, indent=2),
            encoding="utf-8")
        top = {
            "schema_version": "v7_acceptance_v1",
            "automated_passed": (automatic_passed and oracle_passed and best_hits >= 4 and
                                 identity_audit["hard_negative_false_merges"] == 0 and
                                 identity_audit["unlabeled_count"] == 0 and
                                 browse_failures == 0),
            "human_passed": None, "passed": False, "delivery": "blocked",
            "hard_negative_false_merges": identity_audit["hard_negative_false_merges"],
            "identity_gt_unlabeled_count": identity_audit["unlabeled_count"],
            "browse_gt_hits_best_arm": best_hits,
            "browse_request_failures": browse_failures,
            "automatic_plan_blind_passed": automatic_passed,
            "oracle_plan_blind_passed": oracle_passed,
        }
        if not top["automated_passed"]:
            top.update({"failure_class": "verification",
                        "failure_stage": first_failure[0],
                        "reason_code": first_failure[1],
                        "automatic_failure": automatic_failure,
                        "oracle_failure": oracle_failure})
        (output / "acceptance.json").write_text(
            json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "rendered.mp4").unlink(missing_ok=True)
        result = {"automatic_plan_blind_passed": automatic_passed,
                  "oracle_plan_blind_passed": oracle_passed,
                  "browse_gt_hits_best_arm": best_hits,
                  "delivery": "blocked_pending_human" if top["automated_passed"]
                  else "blocked"}
    manifest["phases"][args.phase] = result
    manifest.update(result)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return {"output": str(output), "phase": args.phase, **result}


def _evidence_v7_target(args, cfg) -> dict:
    from src.agentic_video.target_v7 import V7Blocked

    output = Path(args.output).resolve()
    try:
        return _evidence_v7_target_impl(args, cfg)
    except V7Blocked as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure_class = ("infrastructure" if exc.failure_stage == "model" else
                         "content" if exc.failure_stage == "browsing" else
                         "verification")
        acceptance = {
            "schema_version": "v7_acceptance_v1", "automated_passed": False,
            "human_passed": None, "passed": False, "delivery": "blocked",
            "failure_class": failure_class, "failure_stage": exc.failure_stage,
            "reason_code": exc.reason_code, "detail": str(exc),
        }
        (output / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output), "phase": args.phase, **acceptance}
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        acceptance = {
            "schema_version": "v7_acceptance_v1", "automated_passed": False,
            "human_passed": None, "passed": False, "delivery": "blocked",
            "failure_class": "infrastructure", "failure_stage": args.phase,
            "reason_code": "unhandled_runtime_failure",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        (output / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def _evidence_v7_accept(args, _cfg) -> dict:
    from src.agentic_video.target_v7 import accept_v7_output

    final = accept_v7_output(Path(args.output), Path(args.human_acceptance))
    return {"output": str(final), "delivery": "passed"}


def _run(args, cfg) -> dict:
    from src.agentic_video.pipeline import run_full

    if args.no_verification:
        cfg.library.setdefault("verification", {})["enabled"] = False
    final = run_full(cfg, Path(args.reference), theme=args.theme, library=args.library,
                     output_dir=Path(args.output), force=args.force,
                     use_mask_backend=not args.no_mask_backend,
                     target_duration_s=args.target_duration)
    return {"output": str(final), "report": str(Path(args.output) / "report.md")}


def _validated_gpu_spec(spec: str) -> str:
    """Omni 推理的卡数约束（M4）：只允许 2 或 4 张不重复数字卡号。

    旧实现对任意串（1 卡、3 卡、尾逗号 typo）都直通 CUDA_VISIBLE_DEVICES，
    静默以错卡数降级或加载崩溃——按用户约定在这里快速失败。"""
    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    if (len(parts) not in (2, 4) or not all(p.isdigit() for p in parts)
            or len(set(parts)) != len(parts)):
        raise SystemExit(f"--gpus 只允许 2 或 4 张不重复的数字卡号"
                         f"（如 0,1 或 0,1,2,3），得到：{spec!r}")
    return ",".join(parts)


def _preload_torch_before_cv2() -> None:
    """服务器实证（2026-09-10）：cv2 先于 torch 导入的进程里，Qwen3-Omni
    四卡加载必现 SIGSEGV（原生 OpenMP 运行时冲突，core dumped、无 Python 栈，
    cv2→加载 二分复现 2/2，torch→cv2→加载 2/2 通过）。本仓库 signals 等模块
    import cv2，而 run/分解又在同进程加载 VLM——进 handler 前先占住 torch
    的原生运行时即可化解；本地无 torch 时静默跳过。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gpus:
        from src.perception.omni_runner import set_visible_gpus
        set_visible_gpus(_validated_gpu_spec(args.gpus))
    _preload_torch_before_cv2()
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="agentic_video")
    handlers = {"discover": _discover, "benchmark": _benchmark,
                "index": _index, "facets": _facets, "bootstrap": _bootstrap,
                "decompose": _decompose,
                "render": _render, "roughcut": _roughcut,
                "roughcut-accept": _roughcut_accept,
                "reference-pattern": _reference_pattern,
                "evidence-mine": _evidence_mine,
                "evidence-plan": _evidence_plan,
                "evidence-render": _evidence_render,
                "evidence-v6": _evidence_v6,
                "evidence-v6-accept": _evidence_v6_accept,
                "evidence-v61-diagnostic": _evidence_v61_diagnostic,
                "evidence-v7-target": _evidence_v7_target,
                "evidence-v7-accept": _evidence_v7_accept,
                "run": _run}
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
