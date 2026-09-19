"""Unified command-line interface for Agentic Video Program Induction."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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

    character_batch = sub.add_parser(
        "character-batch",
        help="run one phase of the V8 occurrence-first shared character experiment")
    character_batch.add_argument(
        "--phase", required=True,
        choices=("bootstrap", "coverage", "coverage-audit", "pilot", "context",
                 "identity", "events", "plan", "render"))
    character_batch.add_argument(
        "--spec", default="config/experiments/lxh1_v8_character_batch.json")
    character_batch.add_argument("--video", default=None)
    character_batch.add_argument("--output", required=True)
    character_batch.add_argument("--seed-manifest", default=None,
                                 help="human-confirmed seed/form manifest for identity phase")
    character_batch.add_argument(
        "--gpu-pairs", default=None,
        help="Omni worker pairs, e.g. '0,1;2,3;4,5;6,7'")
    character_batch.add_argument("--worker-timeout", type=float, default=3600.0)
    character_batch.add_argument(
        "--context-scope", choices=("pilot", "full"), default="pilot",
        help="context phase input: diagnostic pilot union or full coverage leads")
    character_batch.add_argument(
        "--pilot-ground-truth", default=None,
        help="optional post-hoc GT JSON; never enters browse prompts")
    character_batch.add_argument("--force", action="store_true")

    character_batch_accept = sub.add_parser(
        "character-batch-accept", help="release individually approved V8 character cuts")
    character_batch_accept.add_argument("--output", required=True)
    character_batch_accept.add_argument("--human-acceptance", required=True)

    target_recall = sub.add_parser(
        "character-recall-v82",
        help="run one V8.2 target-centric subject-recall diagnostic phase")
    target_recall.add_argument(
        "--phase", required=True,
        choices=("bootstrap", "prepare", "preflight-input", "preflight", "browse",
                 "observe", "timeline", "audit-pack", "evaluate"))
    target_recall.add_argument(
        "--spec", default="config/experiments/lxh1_v82_target_recall.json")
    target_recall.add_argument("--video", default=None)
    target_recall.add_argument("--output", required=True)
    target_recall.add_argument("--seed-manifest", default=None)
    target_recall.add_argument("--preflight-ground-truth", default=None,
                               help="post-freeze human GT; never passed to a model")
    target_recall.add_argument("--audit-annotations", default=None,
                               help="completed blind 45-second-block audit JSON")
    target_recall.add_argument(
        "--append-audit", action="store_true",
        help="append the configured per-stratum blind sample without changing the run")
    target_recall.add_argument(
        "--gpu-pairs", default=None,
        help="Omni worker pairs, e.g. '0,1;2,3;4,5;6,7'")
    target_recall.add_argument("--worker-timeout", type=float, default=3600.0)
    target_recall.add_argument("--force", action="store_true")

    omni_edit = sub.add_parser(
        "omni-edit-trial",
        help="run a human-anchored, Omni-planned debug microcut")
    omni_edit.add_argument(
        "--spec", default="config/experiments/lxh1_omni_edit_trial.json")
    omni_edit.add_argument("--video", default=None)
    omni_edit.add_argument("--output", required=True)
    omni_edit.add_argument(
        "--gpu-pairs", default=None,
        help="Omni worker pairs, e.g. '0,1;2,3;4,5;6,7'")
    omni_edit.add_argument("--worker-timeout", type=float, default=3600.0)
    omni_edit.add_argument("--force", action="store_true")

    reference_v9 = sub.add_parser(
        "reference-program-v9",
        help="induce evidence-bound Content/Edit/Material programs from one reference")
    reference_v9.add_argument(
        "--reference",
        default="data/videos/7682719919410072847/video.mp4")
    reference_v9.add_argument("--output", required=True)
    reference_v9.add_argument(
        "--gpu-pairs", required=True,
        help="Omni worker pairs, e.g. '0,1;2,3;4,5;6,7'")
    reference_v9.add_argument("--worker-timeout", type=float, default=3600.0)
    reference_v9.add_argument("--force", action="store_true")

    reference_v9_accept = sub.add_parser(
        "reference-program-v9-accept",
        help="freeze V9 reference programs after section and transfer review")
    reference_v9_accept.add_argument("--output", required=True)
    reference_v9_accept.add_argument("--human-review", required=True)

    reference_v9g = sub.add_parser(
        "reference-generate-v9g",
        help="run one V9-G contract, capability, generation, selection or evaluation phase")
    reference_v9g.add_argument(
        "--phase", required=True,
        choices=("license", "capability", "contract", "pilot", "generate",
                 "select", "assemble", "evaluate"))
    reference_v9g.add_argument("--output", required=True)
    reference_v9g.add_argument("--v9-output", default=None)
    reference_v9g.add_argument("--target-setting", default=None)
    reference_v9g.add_argument(
        "--code-license-gate", default="license_gates/code_license_gate.json")
    reference_v9g.add_argument(
        "--model-license-gate", default="license_gates/model_license_gate.json")
    reference_v9g.add_argument("--fl2va-endpoint", default=None)
    reference_v9g.add_argument("--ref2va-endpoint", default=None)
    reference_v9g.add_argument("--fixture-dir", default=None)
    reference_v9g.add_argument(
        "--plan-only", action="store_true",
        help="write a CPU-only, unverified H3 capability smoke plan")
    reference_v9g.add_argument("--dry-run-real-backend", action="store_true",
                               help="serialize and audit official H3 JSON without HTTP")
    reference_v9g.add_argument("--execute", action="store_true",
                               help="run real H3 smoke only after three safety gates")
    reference_v9g.add_argument("--sync-check", action="store_true",
                               help="read-only server git comparison; never pull")
    reference_v9g.add_argument("--expected-code-sha", default=None)
    reference_v9g.add_argument("--ssh-target", default=None)
    reference_v9g.add_argument("--server-root", default=None)
    reference_v9g.add_argument("--gpu-set", default=None,
                               help="GPU indices needed by the selected H3 service")
    reference_v9g.add_argument("--owned-service-pids", default=None)
    reference_v9g.add_argument("--cases", default=None,
                               help="comma-separated S0-S7 cases for staged variants")
    reference_v9g.add_argument("--input", default=None)
    reference_v9g.add_argument(
        "--gpu-pairs", default=None,
        help="Omni worker pairs used only by pilot/generate")
    reference_v9g.add_argument("--worker-timeout", type=float, default=3600.0)

    reference_v9g_accept = sub.add_parser(
        "reference-generate-v9g-accept",
        help="finalize V9-G only after Section, blind and human review all pass")
    reference_v9g_accept.add_argument("--output", required=True)
    reference_v9g_accept.add_argument("--human-review", required=True)

    long_take = sub.add_parser(
        "long-take-lt0",
        help="H3-LT0: S2 long-take quality experiment (4 recipes x 2 seeds)")
    long_take.add_argument("--p04e-output", required=True,
                           help="green P0 run directory (p04e)")
    long_take.add_argument("--output", required=True)
    long_take.add_argument(
        "--reference", default="data/videos/7682719919410072847/video.mp4")
    long_take.add_argument("--ref2va-endpoint",
                           default="http://127.0.0.1:30011")
    long_take.add_argument("--fl2va-endpoint",
                           default="http://127.0.0.1:30011",
                           help="unified diffusers serve handles t2va too; "
                                "set differently only for sglang backends")
    long_take.add_argument("--h3-backend", default="diffusers",
                           choices=("diffusers", "sglang"),
                           help="diffusers unified serve (driver-safe) or "
                                "sglang (needs cu13 driver >=580)")
    long_take.add_argument("--h3-python-bin", default=None,
                           help="python for the diffusers serve (default: "
                                "current interpreter)")
    long_take.add_argument("--gpu-set", default="0,1,6,7")
    long_take.add_argument("--gpu-pairs", default="0,1;6,7",
                           help="Omni observation pairs (H3 stopped first)")
    long_take.add_argument("--seconds", type=float, default=12.0)
    long_take.add_argument("--seeds", default="1001,1002")
    long_take.add_argument("--recipes", default=None,
                           help="comma-filter of recipes for targeted reruns "
                                "(e.g. video_only,canonical_plus_video)")
    long_take.add_argument("--pack-picks", default=None,
                           help="JSON file mapping role -> chosen time_s")
    long_take.add_argument("--plan-only", action="store_true",
                           help="CPU plan + contact sheets for human review")
    long_take.add_argument("--execute", action="store_true",
                           help="run real H3 takes (LT0 capability experiment)")
    long_take.add_argument("--no-manage-server", action="store_true",
                           help="assume the ref2va server is already up")

    produce = sub.add_parser(
        "long-take-produce",
        help="production single-take workflow: constraints -> gates -> 1xH3 "
             "-> two-stage verification -> bounded repair")
    produce.add_argument("--p04e-output", required=True)
    produce.add_argument("--output", required=True)
    produce.add_argument(
        "--reference", default="data/videos/7682719919410072847/video.mp4")
    produce.add_argument("--constraints", default=None,
                         help="subject_constraints JSON (default: built-in "
                              "user-approved card)")
    produce.add_argument("--pack-picks", default=None,
                         help="JSON: {picks: {role: candidate_NN|time}, "
                              "evidence_quality: {role: {...human approval}}}")
    produce.add_argument("--repair-plan", default=None,
                         help="repair_plan JSON with condition_delta; "
                              "enables the single repair attempt")
    produce.add_argument("--endpoint", default="http://127.0.0.1:30011")
    produce.add_argument("--gpu-set", default="0,1")
    produce.add_argument("--gpu-pairs", default="0,1")
    produce.add_argument("--h3-backend", default="diffusers")
    produce.add_argument("--h3-python-bin", default=None)
    produce.add_argument("--no-manage-server", action="store_true")
    produce.add_argument("--plan-only", action="store_true")
    produce.add_argument("--execute", action="store_true")

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


def _v8_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def _v8_omni_pairs(spec: dict, args) -> str:
    configured = (spec.get("perception") or {}).get("gpu_pairs") or []
    pairs = args.gpu_pairs or ";".join(str(pair) for pair in configured)
    if not pairs:
        raise ValueError(f"V8 phase {args.phase} requires --gpu-pairs")
    return pairs


def _character_batch_impl(args, cfg) -> dict:
    from src.agentic_video.character_batch import (
        V8Blocked, _read_jsonl, bind_occurrence_identities,
        build_context_watch_bank, build_coverage_audit,
        build_character_creation_queue, build_character_profiles,
        build_external_character_prior, build_mention_index,
        build_investigation_leads, build_occurrence_bank_from_leads,
        build_seed_review_sheet, build_uniform_coverage_map,
        derive_character_evidence_views, read_v8_spec, run_character_batch,
        run_v81_diagnostic, select_v81_pilot_windows, verify_shared_event_facts,
    )
    from src.agentic_video.recipe_v2 import sha256_file
    from src.agentic_video.target_v7 import collect_flashvid_runtime

    spec_path = _v8_path(args.spec)
    spec, spec_sha = read_v8_spec(spec_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = _v8_path(args.video or spec["source_video"])
    bgm = _v8_path(spec["bgm_path"])
    transcript_path = _v8_path(spec["transcript_path"])
    reference = _v8_path(spec["reference_video"])
    manifest_path = output / "run_manifest.json"
    previous = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {})
    source_size = source.stat().st_size
    source_hash = (previous.get("source_sha256")
                   if previous.get("source_video") == str(source)
                   and previous.get("source_size_bytes") == source_size
                   else sha256_file(source))
    manifest = {
        "schema_version": "character_batch_run_manifest_v1",
        "spec": str(spec_path), "spec_sha256": spec_sha,
        "source_video": str(source), "source_size_bytes": source_size,
        "source_sha256": source_hash,
        "fixed_characters": [row["character_id"] for row in spec["characters"]],
        "automatic_character_selection_claimed": False,
        "no_yolo_tracker_reid": True,
        "custom_flashvid_port": True,
        "base_model": "Qwen3.5-4B",
        "flashvid_official_support_claimed": False,
        "flashvid_runtime": collect_flashvid_runtime(Path(spec["flashvid_runtime"])),
        "transcript_path": str(transcript_path),
        "transcript_sha256": sha256_file(transcript_path),
        "reference_video": str(reference),
        "reference_sha256": sha256_file(reference),
        "bgm_path": str(bgm),
        "bgm_sha256": sha256_file(bgm),
        "phases": dict(previous.get("phases") or {}),
    }
    # Persist the expensive source hash before phase work so a late failure can resume.
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    if args.phase == "bootstrap":
        prior = build_external_character_prior(spec, output / "external_prior")
        raw_transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript_rows = (raw_transcript.get("segments") or
                           raw_transcript.get("utterances") or []) \
            if isinstance(raw_transcript, dict) else raw_transcript
        mentions = build_mention_index(
            transcript_rows, spec["characters"], output / "mention_index.jsonl")
        seed_review = build_seed_review_sheet(
            cfg, spec, output / "character_profiles" / "seed_review",
            source_video=source)
        reference_reuse = spec.get("reference_task_reuse")
        if reference_reuse and _v8_path(reference_reuse).is_file():
            reference_source = _v8_path(reference_reuse)
            shutil.copy2(reference_source, output / "reference_task.json")
            reference_status = "reused_verified_artifact"
        else:
            from src.agentic_video.target_v7 import prepare_reference_task
            from src.perception.flashvid_client import FlashVIDClient, FlashVIDEndpoint
            row = spec["reference_backend"]
            client = FlashVIDClient(FlashVIDEndpoint(
                arm="D", base_url=str(row["base_url"]),
                model=str(row.get("model") or "Qwen3.5-4B"), fps=4.0,
                retention_ratio=1.0, backend="native_bypass"))
            prepare_reference_task(cfg, spec, output, vanilla_client=client)
            reference_status = "generated_native_bypass"
        result = {
            "prior_claim_count": len(prior["claims"]), "mention_count": len(mentions),
            "seed_proposal_count": len(seed_review["proposals"]),
            "reference_status": reference_status,
        }
    elif args.phase == "coverage":
        from src.perception.flashvid_client import FlashVIDClient, FlashVIDEndpoint
        row = spec["coverage"]["endpoint"]
        client = FlashVIDClient(FlashVIDEndpoint(
            arm="A", base_url=str(row["base_url"]),
            model=str(row.get("model") or "Qwen3.5-4B"),
            fps=float(spec["coverage"]["fps"]),
            retention_ratio=float(spec["coverage"]["retention_ratio"]),
            backend="flashvid"))
        coverage = build_uniform_coverage_map(
            cfg, spec, output / "coverage", client=client, source_video=source,
            source_sha256=source_hash, reuse_completed=not args.force)
        result = {
            "block_count": coverage["block_count"],
            "covered_count": coverage["covered_count"],
            "failed_count": coverage["failed_count"],
            "occurrence_count": coverage["occurrence_count"],
        }
    elif args.phase == "coverage-audit":
        coverage_path = output / "coverage" / "coverage_manifest.json"
        coverage_manifest = (json.loads(coverage_path.read_text(encoding="utf-8"))
                             if coverage_path.is_file() else {"blocks": []})
        global_intervals = [row["source_interval"]
                            for row in coverage_manifest.get("blocks") or []
                            if row.get("status") == "covered"]
        dense_intervals = []
        for context_path in (
                output / "diagnostic" / "context_watch" / "context_watch_manifest.json",
                output / "context_watch" / "context_watch_manifest.json"):
            if context_path.is_file():
                context_manifest = json.loads(context_path.read_text(encoding="utf-8"))
                dense_intervals.extend(
                    row["source_interval"] for row in context_manifest.get("results") or []
                    if row.get("status") in {"observed", "observed_empty"})
        legacy_intervals = []
        legacy_root = output / "occurrence_observation"
        if legacy_root.is_dir():
            for result_path in legacy_root.glob("*/result.json"):
                try:
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if row.get("source_interval"):
                    legacy_intervals.append(row["source_interval"])
        duration_s = common.video_duration_s(
            cfg.perception.get("ffprobe_bin", "ffprobe"), source)
        audit = build_coverage_audit(
            spec, duration_s=duration_s, global_intervals=global_intervals,
            dense_intervals=dense_intervals, legacy_intervals=legacy_intervals,
            output_path=output / "coverage_audit.json")
        result = {
            "global_coverage_ratio": audit["global_browse"]["coverage_ratio"],
            "dense_coverage_ratio": audit["dense_context_watch"]["coverage_ratio"],
            "legacy_coverage_ratio": audit["legacy_targeted_windows"]["coverage_ratio"],
        }
    elif args.phase == "pilot":
        from src.perception.flashvid_client import FlashVIDClient, FlashVIDEndpoint

        mentions = _read_jsonl(output / "mention_index.jsonl")
        coverage_candidates_path = output / "coverage" / "event_candidates.jsonl"
        coverage_candidates = (_read_jsonl(coverage_candidates_path)
                               if coverage_candidates_path.is_file() else [])
        leads = build_investigation_leads(mentions, coverage_candidates)
        duration_s = common.video_duration_s(
            cfg.perception.get("ffprobe_bin", "ffprobe"), source)
        pilot_windows = select_v81_pilot_windows(
            spec, duration_s=duration_s, high_value_leads=leads)
        clients = {}
        for row in (spec.get("diagnostic") or {}).get("browse_arms") or []:
            arm_id = str(row["id"])
            clients[arm_id] = FlashVIDClient(FlashVIDEndpoint(
                arm=arm_id, base_url=str(row["base_url"]), model=str(row["model"]),
                fps=float(row["fps"]), retention_ratio=float(row["retention_ratio"]),
                backend=str(row["backend"])))
        ground_truth = None
        if args.pilot_ground_truth:
            raw_gt = json.loads(Path(args.pilot_ground_truth).read_text(encoding="utf-8"))
            ground_truth = (raw_gt.get("ground_truth") or []
                            if isinstance(raw_gt, dict) else raw_gt)
        diagnostic = run_v81_diagnostic(
            cfg, spec, output / "diagnostic" / "pilot", clients=clients,
            source_video=source, pilot_windows=pilot_windows,
            source_sha256=source_hash, ground_truth=ground_truth)
        result = {
            "pilot_window_count": diagnostic["pilot_window_count"],
            "selected_candidate_count": diagnostic["selected_candidate_count"],
            "complete": diagnostic["complete"],
            "diagnosis": diagnostic["arm_comparison"]["diagnosis"],
        }
    elif args.phase in {"context", "identity", "events", "render"}:
        from src.perception.omni_pool import OmniProcessPool

        with OmniProcessPool(
                _v8_omni_pairs(spec, args), cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            if args.phase == "context":
                context_cfg = (spec.get("diagnostic") or {}).get("context_watch") or {}
                if args.context_scope == "pilot":
                    candidate_path = (output / "diagnostic" / "pilot" /
                                      "selected_candidates.jsonl")
                    context_output = output / "diagnostic" / "context_watch"
                    if not candidate_path.is_file():
                        raise V8Blocked("context", "context_candidates_missing")
                    candidates = _read_jsonl(candidate_path)
                else:
                    candidate_path = output / "coverage" / "event_candidates.jsonl"
                    context_output = output / "context_watch"
                    if not candidate_path.is_file():
                        raise V8Blocked("context", "context_candidates_missing")
                    coverage_candidates = _read_jsonl(candidate_path)
                    mentions = _read_jsonl(output / "mention_index.jsonl")
                    candidates = build_investigation_leads(
                        mentions, coverage_candidates)
                candidates.extend(context_cfg.get("forced_candidates") or [])
                context_manifest = build_context_watch_bank(
                    cfg, spec, candidates, context_output, source_video=source,
                    runner=runner, source_sha256=source_hash,
                    initial_context_s=float(context_cfg.get("initial_context_s", 10.0)),
                    max_context_s=float(context_cfg.get("max_context_s", 40.0)),
                    expansion_step_s=float(context_cfg.get("expansion_step_s", 4.0)),
                    context_fps=float(context_cfg.get("fps", 4.0)),
                    merge_gap_s=float(context_cfg.get("merge_gap_s", 1.0)),
                    reuse_completed=not args.force)
                result = {
                    "context_scope": args.context_scope,
                    "candidate_count": context_manifest["candidate_count"],
                    "observed_count": context_manifest["observed_count"],
                    "partial_count": context_manifest["partial_count"],
                    "unreliable_count": context_manifest["unreliable_count"],
                    "failed_count": context_manifest["failed_count"],
                    "complete": context_manifest["complete"],
                }
            elif args.phase == "identity":
                coverage_manifest = json.loads((output / "coverage" /
                    "coverage_manifest.json").read_text(encoding="utf-8"))
                if not coverage_manifest.get("complete"):
                    raise V8Blocked("coverage", "uniform_coverage_incomplete")
                context_manifest_path = (output / "context_watch" /
                                         "context_watch_manifest.json")
                if context_manifest_path.is_file():
                    occurrence_manifest = json.loads(
                        context_manifest_path.read_text(encoding="utf-8"))
                    if not occurrence_manifest.get("complete"):
                        raise V8Blocked("context", "full_context_watch_incomplete")
                    shutil.copy2(output / "context_watch" / "occurrence_bank.jsonl",
                                 output / "occurrence_bank.jsonl")
                    shutil.copy2(output / "context_watch" / "event_candidates.jsonl",
                                 output / "event_candidates.jsonl")
                else:
                    coverage_candidates_path = output / "coverage" / "event_candidates.jsonl"
                    if not coverage_candidates_path.is_file():
                        coverage_candidates_path = output / "event_candidates.jsonl"
                    mentions = _read_jsonl(output / "mention_index.jsonl")
                    coverage_candidates = _read_jsonl(coverage_candidates_path)
                    leads = build_investigation_leads(mentions, coverage_candidates)
                    occurrence_manifest = build_occurrence_bank_from_leads(
                        cfg, spec, leads, output / "occurrence_observation",
                        source_video=source, runner=runner,
                        source_sha256=source_hash, reuse_completed=not args.force)
                    if not occurrence_manifest.get("complete"):
                        raise V8Blocked(
                            "occurrence", "neutral_occurrence_observation_incomplete")
                if not args.seed_manifest:
                    raise V8Blocked("identity", "human_seed_manifest_required")
                seed_manifest = json.loads(
                    Path(args.seed_manifest).read_text(encoding="utf-8"))
                profiles = build_character_profiles(
                    cfg, spec, seed_manifest, output / "character_profiles",
                    source_video=source, runner=runner)
                occurrences = _read_jsonl(output / "occurrence_bank.jsonl")
                bindings = bind_occurrence_identities(
                    cfg, occurrences, profiles, output / "identity",
                    source_video=source, runner=runner)
                result = {
                    "usable_forms": profiles["usable_form_count"],
                    "neutral_observations": occurrence_manifest.get(
                        "observation_count", occurrence_manifest.get("candidate_count", 0)),
                    "neutral_observation_failures": occurrence_manifest.get(
                        "failed_observation_count", occurrence_manifest.get("failed_count", 0)),
                    **bindings["metrics"],
                }
            elif args.phase == "events":
                occurrences = _read_jsonl(output / "occurrence_bank.jsonl")
                event_candidates = _read_jsonl(output / "event_candidates.jsonl")
                bindings = _read_jsonl(output / "identity_bindings.jsonl")
                verified_occurrence_ids = {
                    str(row["occurrence_id"]) for row in bindings
                    if row.get("status") == "verified"
                }
                event_candidates = [
                    row for row in event_candidates
                    if row.get("actor_occurrence_id") in verified_occurrence_ids
                    or row.get("patient_occurrence_id") in verified_occurrence_ids
                ]
                verified = verify_shared_event_facts(
                    cfg, event_candidates, occurrences, output / "events",
                    source_video=source, runner=runner,
                    native_frame_max=int((spec.get("perception") or {}).get(
                        "native_frame_max", 60)))
                views = derive_character_evidence_views(
                    verified["event_facts"], bindings,
                    output / "character_evidence_views")
                result = {
                    "verified_events": verified["verified_count"],
                    "blocked_events": verified["blocked_count"],
                    "view_counts": views["counts"],
                }
            else:
                queue = json.loads((output / "creation_queue.json").read_text(
                    encoding="utf-8"))
                reference_task = json.loads((output / "reference_task.json").read_text(
                    encoding="utf-8"))
                view_manifest = json.loads((output / "character_evidence_views" /
                                            "views_manifest.json").read_text(encoding="utf-8"))
                batch = run_character_batch(
                    cfg, queue, reference_task, view_manifest["views"], output,
                    source_video=source, bgm_path=bgm, runner=runner, force=args.force)
                result = {
                    "selected_task_count": batch["selected_task_count"],
                    "preview_ready_count": batch["preview_ready_count"],
                    "delivery": batch["delivery"],
                }
    else:
        view_manifest = json.loads((output / "character_evidence_views" /
                                    "views_manifest.json").read_text(encoding="utf-8"))
        queue = build_character_creation_queue(
            view_manifest["views"], output / "creation_queue.json")
        result = {
            "candidate_task_count": len(queue["candidates"]),
            "selected_task_count": len(queue["selected_tasks"]),
            "ready_character_count": queue["ready_character_count"],
        }
    manifest["phases"][args.phase] = result
    manifest["last_phase"] = args.phase
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return {"output": str(output), "phase": args.phase, **result}


def _character_batch(args, cfg) -> dict:
    from src.agentic_video.character_batch import V8Blocked

    try:
        return _character_batch_impl(args, cfg)
    except V8Blocked as exc:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "character_batch_phase_failure_v1",
            "passed": False, "delivery": "blocked",
            "failure_class": "verification", "failure_stage": exc.stage,
            "reason_code": exc.reason_code, "detail": exc.detail or str(exc),
        }
        (output / f"{args.phase}_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output), "phase": args.phase, **failure}
    except Exception as exc:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "character_batch_phase_failure_v1",
            "passed": False, "delivery": "blocked",
            "failure_class": "infrastructure", "failure_stage": args.phase,
            "reason_code": "unhandled_runtime_failure",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        (output / f"{args.phase}_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def _character_batch_accept(args, _cfg) -> dict:
    from src.agentic_video.character_batch import accept_character_batch

    result = accept_character_batch(
        Path(args.output).resolve(), Path(args.human_acceptance).resolve())
    return {"output": str(Path(args.output).resolve()),
            "delivery": result["delivery"],
            "final_passed_count": result["final_passed_count"]}


def _v82_runtime(args, cfg) -> dict:
    from src.agentic_video.character_recall import (
        V82Blocked, _read_json, attach_audit_design, bind_target_and_form,
        bootstrap_movie_knowledge, build_candidate_scenes,
        build_neighbor_ranges, build_observation_tasks, build_stratified_audit_pack,
        build_target_search_card, build_target_timelines, evaluate_v82_recall,
        interval_coverage_ratio,
        prepare_multiform_profile, read_v82_spec,
        run_observe_queue, run_preflight_flashvid_comparison,
        run_target_recall_browse, run_v82_preflight,
        verify_target_event_facts,
    )
    from src.agentic_video.character_batch import (
        _read_jsonl, _write_jsonl, build_mention_index,
    )
    from src.agentic_video.recipe_v2 import sha256_file
    from src.agentic_video.target_v7 import collect_flashvid_runtime

    spec_path = _v8_path(args.spec)
    spec, spec_sha = read_v82_spec(spec_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = _v8_path(args.video or spec["source_video"])
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest_path = output / "run_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.is_file() else {}
    source_size = source.stat().st_size
    source_hash = (previous.get("source_sha256")
                   if previous.get("source_video") == str(source)
                   and previous.get("source_size_bytes") == source_size
                   else sha256_file(source))
    manifest = {
        "schema_version": "target_recall_run_manifest_v82",
        "spec": str(spec_path), "spec_sha256": spec_sha,
        "source_video": str(source), "source_size_bytes": source_size,
        "source_sha256": source_hash,
        "perception_objective": spec["perception_objective"],
        "movie_knowledge_prior": str(_v8_path(spec["movie_knowledge_prior"])),
        "implemented_harvest_policies": ["subject"],
        "future_policy_placeholders": ["scene", "object", "interaction", "semantic"],
        "future_policies_implemented": False,
        "editing_or_rendering_started": False,
        "custom_flashvid_port": True,
        "flashvid_runtime": collect_flashvid_runtime(Path(spec["flashvid_runtime"])),
        "phases": dict(previous.get("phases") or {}),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    profile_dir = output / "target_profile"
    profile_manifest_path = profile_dir / "profiles_manifest.json"
    knowledge_dir = output / "film_knowledge"
    knowledge_manifest_path = knowledge_dir / "knowledge_manifest.json"
    search_card_path = output / "target_search_card.json"

    def require_knowledge() -> dict:
        if not knowledge_manifest_path.is_file():
            raise V82Blocked("knowledge", "movie_knowledge_bootstrap_required")
        knowledge = _read_json(knowledge_manifest_path)
        if knowledge.get("knowledge_is_prior") is not True:
            raise V82Blocked("knowledge", "movie_knowledge_prior_contract_missing")
        return knowledge

    def require_profile() -> dict:
        if not profile_manifest_path.is_file():
            raise V82Blocked("prepare", "human_confirmed_profile_required")
        profile = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
        if profile.get("status") != "ready":
            raise V82Blocked("prepare", "three_form_profile_not_ready")
        return profile

    def album_paths(profile: dict) -> list[Path]:
        paths = [Path(row["path"]) for row in
                 (profile.get("browse_album") or {}).get("ordered_inputs") or []]
        if len(paths) != 4 or any(not path.is_file() for path in paths):
            raise V82Blocked("prepare", "four_profile_mosaics_required")
        return paths

    def require_search_card() -> dict:
        if not search_card_path.is_file():
            raise V82Blocked("prepare", "target_search_card_required")
        return _read_json(search_card_path)

    def flash_client(endpoint_key: str):
        from src.perception.flashvid_client import FlashVIDClient, FlashVIDEndpoint
        row = spec["browse"][endpoint_key]
        is_vanilla = endpoint_key == "vanilla_endpoint"
        return FlashVIDClient(FlashVIDEndpoint(
            arm="D" if is_vanilla else "C", base_url=str(row["base_url"]),
            model=str(row["model"]), fps=4.0,
            retention_ratio=1.0 if is_vanilla else .25,
            backend=str(row["backend"])))

    def omni_pairs() -> str:
        pairs = args.gpu_pairs or ";".join(spec["observe"].get("gpu_pairs") or [])
        if not pairs:
            raise V82Blocked(args.phase, "omni_gpu_pairs_required")
        return pairs

    if args.phase == "bootstrap":
        prior_path = _v8_path(spec["movie_knowledge_prior"])
        knowledge = bootstrap_movie_knowledge(
            prior_path, knowledge_dir,
            target_id=spec["perception_objective"]["target_id"])
        result = {
            "phase_status": "ready", "knowledge_is_prior": True,
            "knowledge_manifest_sha256": knowledge["knowledge_manifest_sha256"],
            "source_count": knowledge["source_count"],
            "claim_count": knowledge["claim_count"],
        }
    elif args.phase == "prepare":
        knowledge = require_knowledge()
        seed_manifest = (json.loads(Path(args.seed_manifest).read_text(encoding="utf-8"))
                         if args.seed_manifest else None)
        if seed_manifest is None:
            profile = prepare_multiform_profile(
                cfg, spec, profile_dir, source_video=source)
        else:
            from src.perception.omni_pool import OmniProcessPool
            with OmniProcessPool(
                    omni_pairs(), cfg.perception.get("omni") or {},
                    ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                    response_timeout_s=args.worker_timeout) as runner:
                profile = prepare_multiform_profile(
                    cfg, spec, profile_dir, source_video=source,
                    seed_manifest=seed_manifest, runner=runner)
        search_card = None
        if profile.get("status") == "ready":
            search_card = build_target_search_card(
                knowledge, profile, search_card_path)
        transcript = _v8_path(spec["transcript_path"])
        mention_count = 0
        if transcript.is_file():
            raw_transcript = json.loads(transcript.read_text(encoding="utf-8"))
            rows = ((raw_transcript.get("segments") or raw_transcript.get("utterances") or [])
                    if isinstance(raw_transcript, dict) else raw_transcript)
            target = dict((knowledge.get("target_character_prior") or {}))
            if not target:
                target = dict(spec["target_profile"])
            mentions = build_mention_index(rows, [{
                "character_id": spec["perception_objective"]["target_id"],
                "display_name": target["display_name"],
                "aliases": target.get("aliases") or [],
            }], output / "mention_index.jsonl")
            mention_count = len(mentions)
        result = {"phase_status": profile["status"], "mention_count": mention_count,
                  "identity_truth_claimed": profile["status"] == "ready",
                  "search_card_sha256": ((search_card or {}).get(
                      "search_card_sha256"))}
    elif args.phase == "preflight-input":
        profile = require_profile()
        search_card = require_search_card()
        comparison = run_preflight_flashvid_comparison(
            cfg, spec, output / "preflight" / "flashvid",
            source_video=source, album_images=album_paths(profile),
            r025_client=flash_client("r025_endpoint"),
            vanilla_client=flash_client("vanilla_endpoint"),
            search_card=search_card)
        if not comparison["passed"]:
            raise V82Blocked("preflight-input", "flashvid_input_audit_failed")
        result = {
            "decision": "PASS",
            "flashvid_input_audit_ready": True,
            "compared_case_count": comparison["case_count"],
            "semantic_agreement_is_not_accuracy_proof": True,
        }
    elif args.phase == "preflight":
        profile = require_profile()
        require_search_card()
        comparison_path = (output / "preflight" / "flashvid" /
                           "comparison_manifest.json")
        if not comparison_path.is_file():
            raise V82Blocked("preflight", "preflight_input_audit_required")
        comparison = _read_json(comparison_path)
        if not comparison.get("passed"):
            raise V82Blocked("preflight", "flashvid_input_audit_failed")
        gt = (json.loads(Path(args.preflight_ground_truth).read_text(encoding="utf-8"))
              if args.preflight_ground_truth else None)
        from src.perception.omni_pool import OmniProcessPool
        with OmniProcessPool(
                omni_pairs(), cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            preflight = run_v82_preflight(
                cfg, spec, output / "preflight", source_video=source,
                runner=runner, profiles_manifest=profile, ground_truth=gt)
        result = {"decision": preflight["decision"],
                  "passed_case_count": preflight["passed_case_count"],
                  "full_movie_observe_allowed": preflight["full_movie_observe_allowed"]}
    elif args.phase == "browse":
        profile = require_profile()
        search_card = require_search_card()
        browse = run_target_recall_browse(
            cfg, spec, output / "browse", source_video=source,
            album_images=album_paths(profile),
            r025_client=flash_client("r025_endpoint"),
            vanilla_client=flash_client("vanilla_endpoint"),
            source_sha256=source_hash, reuse_completed=not args.force,
            search_card=search_card)
        result = {key: browse[key] for key in (
            "block_count", "covered_count", "failed_count", "unreliable_count",
            "candidate_count", "complete")}
    elif args.phase == "observe":
        preflight_path = output / "preflight" / "preflight_manifest.json"
        if not preflight_path.is_file() or json.loads(
                preflight_path.read_text(encoding="utf-8")).get("decision") != "PASS":
            raise V82Blocked("observe", "preflight_pass_required")
        browse_manifest = _read_json(output / "browse" / "browse_manifest.json")
        if not browse_manifest.get("complete"):
            raise V82Blocked("observe", "full_browse_incomplete")
        profile = require_profile()
        visual = _read_jsonl(output / "browse" / "visual_candidates.jsonl")
        mentions = (_read_jsonl(output / "mention_index.jsonl")
                    if (output / "mention_index.jsonl").is_file() else [])
        duration = common.video_duration_s(
            cfg.perception.get("ffprobe_bin", "ffprobe"), source)
        shots = []
        if spec.get("shot_intervals_path"):
            shot_path = _v8_path(spec["shot_intervals_path"])
            if shot_path.is_file():
                shot_raw = json.loads(shot_path.read_text(encoding="utf-8"))
                shots = [row.get("source_interval") or row.get("interval")
                         for row in (shot_raw.get("shots") or shot_raw)]
        scenes = build_candidate_scenes(
            visual, mentions, spec.get("prior_ranges") or [], duration_s=duration,
            shot_intervals=shots, output_path=output / "candidate_scenes.jsonl")
        tasks = build_observation_tasks(
            scenes, fps=4.0, max_frames=80, overlap_s=4.0,
            shot_intervals=shots, output_path=output / "observation_tasks.jsonl")
        from src.perception.omni_pool import OmniProcessPool
        with OmniProcessPool(
                omni_pairs(), cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            all_tasks = list(tasks)

            def drain_queue() -> dict:
                prior_signature = None
                current = {}
                for _ in range(12):
                    current = run_observe_queue(
                        cfg, spec, output / "observe", source_video=source,
                        runner=runner, tasks=all_tasks, reuse_completed=not args.force,
                        source_duration_s=duration)
                    persisted = _read_jsonl(output / "observe" / "observe_queue.jsonl")
                    known = {row["task_id"] for row in all_tasks}
                    new_rows = [row for row in persisted if row["task_id"] not in known]
                    if new_rows:
                        all_tasks.extend(new_rows)
                    signature = (tuple((row["task_id"], row["state"])
                                       for row in persisted), len(all_tasks))
                    if current["queue_closed"] or signature == prior_signature:
                        return current
                    prior_signature = signature
                return current

            observe = drain_queue()
            occurrences = _read_jsonl(output / "observe" / "occurrence_bank.jsonl")
            bindings = bind_target_and_form(
                cfg, occurrences, profile, output / "identity",
                source_video=source, runner=runner,
                target_id=spec["perception_objective"]["target_id"],
                reuse_completed=not args.force)
            expanded_path = output / "observe" / "neighbor_expanded_occurrences.json"
            expanded = set(json.loads(expanded_path.read_text(encoding="utf-8"))
                           if expanded_path.is_file() and not args.force else [])
            max_rounds = int(spec["observe"].get("max_neighbor_rounds", 8))
            neighbor_scenes = []
            for _round in range(max_rounds):
                supported_rows = [row for row in bindings["bindings"]
                                  if row["character_status"] == "supported"]
                by_occurrence = {row["occurrence_id"]: row for row in occurrences}
                fresh = [by_occurrence[row["occurrence_id"]] for row in supported_rows
                         if row["occurrence_id"] not in expanded and
                         row["occurrence_id"] in by_occurrence]
                if not fresh:
                    break
                neighbor_ranges = build_neighbor_ranges(
                    fresh, source_duration_s=duration,
                    step_s=float(spec["observe"].get("neighbor_padding_s", 20.0)),
                    clear_intervals_to_stop=2)
                round_scenes = build_candidate_scenes(
                    [], neighbor_ranges=neighbor_ranges, duration_s=duration,
                    shot_intervals=shots)
                round_tasks = build_observation_tasks(
                    round_scenes, fps=4.0, max_frames=80, overlap_s=4.0,
                    shot_intervals=shots)
                existing_intervals = [row["source_interval"] for row in all_tasks]
                new_tasks = []
                for task in round_tasks:
                    start, end = task["source_interval"]
                    if interval_coverage_ratio(
                            [start, end], existing_intervals) < .8:
                        new_tasks.append(task)
                        existing_intervals.append(task["source_interval"])
                expanded.update(row["occurrence_id"] for row in fresh)
                expanded_path.write_text(json.dumps(sorted(expanded), indent=2),
                                         encoding="utf-8")
                if not new_tasks:
                    continue
                neighbor_scenes.extend(round_scenes)
                all_tasks.extend(new_tasks)
                observe = drain_queue()
                occurrences = _read_jsonl(
                    output / "observe" / "occurrence_bank.jsonl")
                bindings = bind_target_and_form(
                    cfg, occurrences, profile, output / "identity",
                    source_video=source, runner=runner,
                    target_id=spec["perception_objective"]["target_id"],
                    reuse_completed=not args.force)
            if neighbor_scenes:
                _write_jsonl(output / "neighbor_candidate_scenes.jsonl", neighbor_scenes)
            supported = {row["occurrence_id"] for row in bindings["bindings"]
                         if row["character_status"] == "supported"}
            event_candidates = _read_jsonl(
                output / "observe" / "neutral_event_candidates.jsonl")
            verified = verify_target_event_facts(
                cfg, event_candidates, occurrences, source_video=source,
                runner=runner, target_occurrence_ids=supported,
                output_dir=output / "events", source_duration_s=duration)
        _write_jsonl(output / "event_facts.jsonl", verified["event_facts"])
        result = {"scene_count": len(scenes) + len(neighbor_scenes),
                  "task_count": len(all_tasks),
                  "queue_closed": observe["queue_closed"],
                  "processing_complete": observe["processing_complete"],
                  "occurrence_count": len(occurrences),
                  "identity_supported_count": bindings["metrics"]["supported"],
                  "verified_event_count": verified["verified_count"],
                  "incomplete_ranges": observe["incomplete_ranges"]}
    elif args.phase == "timeline":
        required = [output / "browse" / "visual_candidates.jsonl",
                    output / "observe" / "occurrence_bank.jsonl",
                    output / "identity" / "identity_bindings.jsonl",
                    output / "event_facts.jsonl"]
        if any(not path.is_file() for path in required):
            raise V82Blocked("timeline", "observe_artifacts_missing")
        candidates = _read_jsonl(required[0])
        if (output / "mention_index.jsonl").is_file():
            candidates.extend(_read_jsonl(output / "mention_index.jsonl"))
        for path in (output / "candidate_scenes.jsonl",
                     output / "neighbor_candidate_scenes.jsonl",
                     output / "observe" / "observe_queue.jsonl"):
            if path.is_file():
                candidates.extend(_read_jsonl(path))
        timeline = build_target_timelines(
            candidates, _read_jsonl(required[1]), _read_jsonl(required[2]),
            _read_jsonl(required[3]), output / "timelines")
        result = timeline
    elif args.phase == "audit-pack":
        browse = _read_json(output / "browse" / "browse_manifest.json")
        final_hit_ids = set(browse.get("final_candidate_block_ids") or [])
        blocks = [{**row, "candidate_hit": str(row["block_id"]) in final_hit_ids}
                  for row in browse["blocks"]]
        eligible = [browse["blocks"][0]["source_interval"][0],
                    browse["blocks"][-1]["source_interval"][1]]
        audit = build_stratified_audit_pack(
            blocks, eligible_interval=eligible, output_dir=output / "audit",
            per_stratum=int(spec["audit"]["initial_per_stratum"]),
            random_seed=int(spec["audit"]["random_seed"]),
            source_video=source,
            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            append_per_stratum=(int(spec["audit"]["append_per_stratum"])
                                if args.append_audit else 0),
            max_total=int(spec["audit"]["max_blocks"]))
        result = {"sample_count": audit["sample_count"],
                  "appended_count": audit["appended_count"],
                  "frozen_sha256": audit["frozen_sha256"]}
    else:
        if not args.audit_annotations:
            raise V82Blocked("evaluate", "blind_audit_annotations_required")
        annotations = json.loads(Path(args.audit_annotations).read_text(encoding="utf-8"))
        rows = annotations.get("rows") or annotations
        if not isinstance(rows, list):
            raise ValueError("audit annotations must contain a rows array")
        audit_dir = output / "audit"
        rows = attach_audit_design(
            rows, private_key=_read_json(audit_dir / "private_sample_key.json"),
            sampling_design=_read_json(audit_dir / "sampling_design.json"))
        observe_manifest_path = output / "observe" / "observe_queue_manifest.json"
        operational = {}
        if observe_manifest_path.is_file():
            observe_manifest = _read_json(observe_manifest_path)
            incomplete = list(observe_manifest.get("incomplete_ranges") or [])
            operational = {
                "incomplete_ranges": incomplete,
                "incomplete_range_count": len(incomplete),
                "queue_closed": bool(observe_manifest.get("queue_closed")),
            }
        operational_gate = bool(operational.get("queue_closed")) and not bool(
            operational.get("incomplete_range_count"))
        evaluation = evaluate_v82_recall(
            rows, thresholds=spec["audit"]["thresholds"],
            regression_hard_gate_passed=bool(
                annotations.get("regression_hard_gate_passed", False)) and
                operational_gate,
            max_blocks_reached=len(rows) >= int(spec["audit"]["max_blocks"]),
            bootstrap_replicates=int(spec["audit"]["bootstrap_replicates"]),
            random_seed=int(spec["audit"]["random_seed"]),
            output_path=output / "evaluation.json", auxiliary=operational)
        result = {"decision": evaluation["decision"],
                  "next_action": evaluation["next_action"]}
    manifest["phases"][args.phase] = result
    manifest["last_phase"] = args.phase
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return {"output": str(output), "phase": args.phase, **result}


def _target_recall_v82(args, cfg) -> dict:
    from src.agentic_video.character_recall import V82Blocked
    try:
        return _v82_runtime(args, cfg)
    except V82Blocked as exc:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "target_recall_phase_failure_v82",
            "decision": "FAIL", "failure_class": (
                "infrastructure" if exc.stage == "infrastructure" else "verification"),
            "failure_stage": exc.stage, "reason_code": exc.reason_code,
            "detail": exc.detail or str(exc),
        }
        (output / f"{args.phase}_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output), "phase": args.phase, **failure}


def _omni_edit_trial(args, cfg) -> dict:
    from src.agentic_video.omni_edit_trial import (
        OmniEditTrialBlocked, read_omni_edit_trial_spec,
        run_omni_edit_trial, write_blocked_acceptance,
    )
    from src.perception.omni_pool import OmniProcessPool

    spec_path = _v8_path(args.spec)
    spec, spec_sha256 = read_omni_edit_trial_spec(spec_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = _v8_path(args.video or spec["source_video"])
    bgm = _v8_path(spec["bgm_path"])
    pairs = args.gpu_pairs or ";".join(spec.get("gpu_pairs") or [])
    if not pairs:
        raise ValueError("omni-edit-trial requires --gpu-pairs")
    try:
        with OmniProcessPool(
                pairs, cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            return run_omni_edit_trial(
                cfg, spec, output, source_video=source, bgm_path=bgm,
                runner=runner, spec_sha256=spec_sha256, force=args.force)
    except OmniEditTrialBlocked as exc:
        acceptance = write_blocked_acceptance(output, exc)
        return {"output": str(output), **acceptance}


def _reference_program_v9(args, cfg) -> dict:
    from src.agentic_video.reference_program_v9 import run_reference_program_v9
    from src.perception.omni_pool import OmniProcessPool

    reference = _v8_path(args.reference)
    output = Path(args.output).resolve()
    with OmniProcessPool(
            args.gpu_pairs, cfg.perception.get("omni") or {},
            ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            response_timeout_s=args.worker_timeout) as runner:
        return run_reference_program_v9(
            cfg, reference, output, runner=runner, force=args.force)


def _reference_program_v9_accept(args, _cfg) -> dict:
    from src.agentic_video.reference_program_v9 import accept_reference_programs

    return accept_reference_programs(Path(args.output), Path(args.human_review))


def _v9g_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def _reference_generate_v9g(args, cfg) -> dict:
    from src.agentic_video.generation_v9g import (
        SGLangH3Client, adapt_contract_to_filmdsl,
        build_asset_and_state_pack, build_generation_jobs,
        compile_generation_contract, evaluate_v9g_experiment,
        finalize_unit_snippets,
        render_and_review_sections, run_v9g_generation_jobs,
        validate_generation_contract,
        validate_upstream_lock,
        validate_license_gates,
        _write_json,
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    capability_modes = (args.plan_only, args.dry_run_real_backend,
                        args.execute, args.sync_check)
    if sum(bool(value) for value in capability_modes) > 1:
        raise ValueError("choose only one capability mode")
    if args.phase == "capability" and args.sync_check:
        from src.agentic_video.generation_p51 import query_server_sync_plan

        if not args.expected_code_sha or not args.ssh_target or not args.server_root:
            raise ValueError("sync-check requires expected SHA, SSH target and root")
        result = query_server_sync_plan(
            repo_root(), expected_sha=args.expected_code_sha,
            ssh_target=args.ssh_target, server_root=args.server_root)
        (output / "server_sync_plan.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output / "server_sync_plan.json"),
                "server_sync_plan": result}
    if args.phase == "capability" and args.dry_run_real_backend:
        from src.agentic_video.generation_p4 import build_h3_capability_smoke_plan
        from src.agentic_video.generation_p51 import audit_real_h3_payload

        if not args.fixture_dir or not args.fl2va_endpoint or \
                not args.ref2va_endpoint:
            raise ValueError("dry-run requires fixtures and both H3 endpoints")
        plan = build_h3_capability_smoke_plan(
            fixture_dir=_v9g_path(args.fixture_dir),
            backend_endpoints={"fl2va": args.fl2va_endpoint,
                               "ref2va": args.ref2va_endpoint})
        requests_dir = output / "capability" / "requests"
        for case in plan["cases"]:
            payload = SGLangH3Client.serialize_payload(case["request"])
            receipt = audit_real_h3_payload(case, payload)
            case_dir = requests_dir / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            for name, value in (("request.json", payload),
                                ("request_manifest.json", case["manifest"]),
                                ("condition_assets.json", case["assets"]),
                                ("client_payload_receipt.json", receipt)):
                (case_dir / name).write_text(
                    json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")
        capability_dir = output / "capability"
        (capability_dir / "capability_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        (capability_dir / "backend_registry.json").write_text(
            json.dumps(plan["backend_registry"], ensure_ascii=False, indent=2),
            encoding="utf-8")
        return {"output": str(capability_dir), "dry_run_real_backend": True,
                "case_count": len(plan["cases"]), "http_requests_sent": 0,
                "real_capability_verified": False}
    if args.plan_only:
        if args.phase != "capability":
            raise ValueError("--plan-only is supported only for capability phase")
        from src.agentic_video.generation_p4 import build_h3_capability_smoke_plan

        plan = build_h3_capability_smoke_plan(
            fixture_dir=_v9g_path(args.fixture_dir) if args.fixture_dir else None)
        capability_dir = output / "capability"
        requests_dir = capability_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        for case in plan["cases"]:
            if "request" not in case:
                continue
            case_dir = requests_dir / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            for filename, value in (("request.json", case["request"]),
                                    ("condition_manifest.json", case["manifest"]),
                                    ("assets.json", case["assets"])):
                (case_dir / filename).write_text(
                    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        for filename, value in (("capability_plan.json", plan),
                                ("backend_registry.json", plan["backend_registry"]),
                                ("manifest.json", {"plan_only": True,
                                                    "fixtures_ready": plan["fixtures_ready"],
                                                    "case_count": len(plan["cases"]),
                                                    "real_capability_verified": False})):
            (capability_dir / filename).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(capability_dir), "plan_only": True,
                "case_count": len(plan["cases"]),
                "fixtures_ready": plan["fixtures_ready"],
                "real_capability_verified": False}
    code_gate = _v9g_path(args.code_license_gate)
    model_gate = _v9g_path(args.model_license_gate)
    licenses = validate_license_gates(code_gate, model_gate)
    upstream = validate_upstream_lock(repo_root())
    if not upstream["passed"]:
        raise RuntimeError("upstream fixture drift: " + ";".join(upstream["errors"]))
    if args.phase == "license":
        result = {"schema_version": "v9g_license_validation_v1",
                  "licenses": licenses, "upstream": upstream, "passed": True}
        (output / "license_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    contract_path = output / "generation_contract.json"
    if args.phase == "contract":
        if not args.v9_output or not args.target_setting:
            raise ValueError("contract phase requires --v9-output and --target-setting")
        contract = compile_generation_contract(
            _v9g_path(args.v9_output), _v9g_path(args.target_setting), contract_path)
        state = build_asset_and_state_pack(contract)
        (output / "asset_state_pack.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        filmdsl = adapt_contract_to_filmdsl(contract, state)
        (output / "filmdsl.json").write_text(
            json.dumps(filmdsl, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output), "contract_hash": contract["contract_hash"],
                "section_count": len(contract["sections"])}

    if args.phase == "capability":
        from src.agentic_video.generation_p4 import build_h3_capability_smoke_plan
        from src.agentic_video.generation_p51 import (
            audit_real_h3_payload, finalize_capability_registry,
            require_execute_gates, run_h3_capability_case,
            verify_capability_run_identity,
        )

        if not args.execute:
            raise ValueError("capability requires --plan-only, --dry-run-real-backend, "
                             "--sync-check or --execute")
        if not all((args.v9_output, args.fixture_dir, args.fl2va_endpoint,
                    args.ref2va_endpoint, args.expected_code_sha, args.gpu_set)):
            raise ValueError("execute requires frozen V9, fixtures, endpoints, "
                             "expected code SHA and GPU set")
        selected_gpus = [int(value) for value in args.gpu_set.split(",")]
        owned_pids = {int(value) for value in
                      (args.owned_service_pids or "").split(",") if value}
        gates = require_execute_gates(
            _v9g_path(args.v9_output), expected_code_sha=args.expected_code_sha,
            repo=repo_root(), required_gpus=selected_gpus,
            owned_service_pids=owned_pids)
        selected = set((args.cases or "S0,S1,S2,S3,S4,S5,S6,S7").split(","))
        if not selected or not selected <= {f"S{index}" for index in range(8)}:
            raise ValueError("--cases must contain S0-S7 IDs")
        plan = build_h3_capability_smoke_plan(
            fixture_dir=_v9g_path(args.fixture_dir),
            backend_endpoints={"fl2va": args.fl2va_endpoint,
                               "ref2va": args.ref2va_endpoint})
        capability_dir = output / "capability_real"
        capability_dir.mkdir(parents=True, exist_ok=True)
        verify_capability_run_identity(
            capability_dir / "capability_run_identity.json",
            expected_code_sha=args.expected_code_sha, plan=plan)
        registry_path = capability_dir / "backend_registry.json"
        registry = (json.loads(registry_path.read_text(encoding="utf-8")) if
                    registry_path.is_file() else plan["backend_registry"])
        for case in plan["cases"]:
            if case["case_id"] in selected:
                audit_real_h3_payload(
                    case, SGLangH3Client.serialize_payload(case["request"]),
                    registry=registry)
        (capability_dir / "execution_gates.json").write_text(
            json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
        (capability_dir / "registry_before.json").write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        clients = {"fl2va": SGLangH3Client(args.fl2va_endpoint),
                   "ref2va": SGLangH3Client(args.ref2va_endpoint)}
        results = []
        for case in plan["cases"]:
            if case["case_id"] not in selected:
                continue
            result = run_h3_capability_case(
                case, clients[case["model_variant"]],
                capability_dir / case["case_id"])
            results.append(result)
        existing = []
        results_path = capability_dir / "capability_results.json"
        if results_path.is_file():
            existing = json.loads(results_path.read_text(encoding="utf-8"))
        by_id = {row["case_id"]: row for row in existing}
        by_id.update({row["case_id"]: row for row in results})
        all_results = [by_id[key] for key in sorted(by_id)]
        _write_json(results_path, all_results)
        registry_result = finalize_capability_registry(
            registry_path, registry, all_results, real_backend=True)
        return {"output": str(capability_dir), "case_count": len(results),
                "registry_committed": registry_result["committed"],
                "all_cases_observed": len(all_results) == 8}

    if not contract_path.is_file():
        raise FileNotFoundError(f"generation contract missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_validation = validate_generation_contract(contract)
    if not contract_validation["passed"]:
        raise RuntimeError("generation contract invalid: " +
                           ";".join(contract_validation["errors"]))
    capability_path = output / "capability" / "capability_manifest.json"
    if args.phase in {"pilot", "generate"}:
        from src.agentic_video.generation_p4 import (
            P4Blocked, validate_formal_h3_job, verify_frozen_program_snapshot,
        )

        registry_path = output / "capability" / "backend_registry.json"
        if not args.v9_output or not registry_path.is_file():
            raise P4Blocked("p4_registry_or_v9_snapshot_missing")
        p0_frozen = verify_frozen_program_snapshot(_v9g_path(args.v9_output), contract)
        if not p0_frozen:
            raise P4Blocked("p0_programs_not_frozen")
        if not capability_path.is_file():
            raise FileNotFoundError(f"capability manifest missing: {capability_path}")
        if not args.fl2va_endpoint or not args.ref2va_endpoint or not args.gpu_pairs:
            raise ValueError("pilot/generate requires both H3 endpoints and --gpu-pairs")
        capabilities = json.loads(capability_path.read_text(encoding="utf-8"))
        jobs = build_generation_jobs(
            contract, capabilities, output / "generation_jobs.json")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        selected_jobs = jobs[:2] if args.phase == "pilot" else jobs
        request_hashes = {}
        for job in selected_jobs:
            plan_path = output / "generation_conditions" / f"{job['unit_id']}.json"
            if not plan_path.is_file():
                raise P4Blocked("formal_condition_manifest_missing", job["unit_id"])
            condition_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            validate_formal_h3_job(
                job["request"], condition_plan["manifest"],
                condition_plan["assets"], registry, p0_frozen=p0_frozen,
                expected_endpoint=(args.ref2va_endpoint if
                                   job["request"]["task"] == "ref2va" else
                                   args.fl2va_endpoint),
                expected_contract_hash=contract["contract_hash"],
                expected_unit_id=job["unit_id"])
            request_hashes[job["unit_id"]] = condition_plan["manifest"]["request_sha256"]
        authorization = {
            "schema_version": "v9g_p4_formal_authorization_v1", "passed": True,
            "simulation_only": False, "contract_hash": contract["contract_hash"],
            "v9_output": str(_v9g_path(args.v9_output)),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "job_request_sha256": request_hashes,
        }
        (output / "formal_generation_authorization.json").write_text(
            json.dumps(authorization, ensure_ascii=False, indent=2), encoding="utf-8")
        from src.perception.omni_pool import OmniProcessPool
        with OmniProcessPool(
                args.gpu_pairs, cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            results = run_v9g_generation_jobs(
                contract, jobs, output, clients={
                    "fl2va": SGLangH3Client(args.fl2va_endpoint),
                    "ref2va": SGLangH3Client(args.ref2va_endpoint),
                }, runner=runner, capabilities=capabilities,
                max_units=2 if args.phase == "pilot" else None,
                formal_authorization=authorization)
        return {"output": str(output), "unit_count": len(results),
                "passed_count": sum(row.get("passed") is True for row in results)}

    if not args.input:
        raise ValueError(f"{args.phase} phase requires --input")
    value = json.loads(_v9g_path(args.input).read_text(encoding="utf-8"))
    if args.phase == "select":
        if not args.gpu_pairs:
            raise ValueError("select requires --gpu-pairs for post-crop Omni review")
        rows = value.get("results") if isinstance(value, dict) else value
        state_path = output / "asset_state_pack.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        from src.perception.omni_pool import OmniProcessPool
        with OmniProcessPool(
                args.gpu_pairs, cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            results = finalize_unit_snippets(
                cfg, contract, rows or [], state, output, runner=runner)
        return {"output": str(output), "passed_count": sum(
            row.get("snippet_review", {}).get("passed") is True for row in results)}
    if args.phase == "assemble":
        if not args.gpu_pairs:
            raise ValueError("assemble requires --gpu-pairs for Section/blind review")
        rows = value.get("results") if isinstance(value, dict) else value
        from src.perception.omni_pool import OmniProcessPool
        with OmniProcessPool(
                args.gpu_pairs, cfg.perception.get("omni") or {},
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
                response_timeout_s=args.worker_timeout) as runner:
            assembly = render_and_review_sections(
                cfg, contract, rows or [], output, runner=runner)
        return {"output": str(output), "passed": assembly["passed"]}
    rows = value.get("rows") if isinstance(value, dict) else value
    metrics = evaluate_v9g_experiment(
        rows or [], output_path=output / "experiment_metrics.json")
    return {"output": str(output), **metrics}


def _reference_generate_v9g_accept(args, _cfg) -> dict:
    from src.agentic_video.generation_v9g import accept_v9g

    return accept_v9g(Path(args.output), Path(args.human_review))


def _long_take_lt0(args, cfg) -> dict:
    import json as _json

    from src.agentic_video.generation_long_take import run_lt0_experiment

    picks = None
    if args.pack_picks:
        picks = _json.loads(Path(args.pack_picks).read_text(encoding="utf-8"))
    seeds = tuple(int(item) for item in str(args.seeds).split(",")
                  if item.strip())
    recipes = (tuple(item.strip() for item in str(args.recipes).split(",")
                     if item.strip()) if args.recipes else None)
    return run_lt0_experiment(
        cfg, Path(args.p04e_output), Path(args.output),
        reference=Path(args.reference), plan_only=args.plan_only,
        execute=args.execute, ref2va_endpoint=args.ref2va_endpoint,
        fl2va_endpoint=args.fl2va_endpoint,
        gpu_set=args.gpu_set, seconds=float(args.seconds), seeds=seeds,
        recipes=recipes,
        pack_picks=picks, gpu_pairs=args.gpu_pairs,
        manage_server=not args.no_manage_server,
        h3_backend=args.h3_backend, h3_python_bin=args.h3_python_bin,
        ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))


def _long_take_produce(args, cfg) -> dict:
    import json as _json

    from src.agentic_video.production_take import run_production_take

    constraints = None
    if args.constraints:
        constraints = _json.loads(
            Path(args.constraints).read_text(encoding="utf-8"))
    pack_picks = None
    if args.pack_picks:
        pack_picks = _json.loads(
            Path(args.pack_picks).read_text(encoding="utf-8"))
    repair_plan = None
    if args.repair_plan:
        repair_plan = _json.loads(
            Path(args.repair_plan).read_text(encoding="utf-8"))
    return run_production_take(
        cfg, Path(args.p04e_output), Path(args.output),
        reference=Path(args.reference), constraints=constraints,
        pack_picks=pack_picks, repair_plan=repair_plan,
        plan_only=args.plan_only, execute=args.execute,
        endpoint=args.endpoint, gpu_set=args.gpu_set,
        gpu_pairs=args.gpu_pairs, h3_backend=args.h3_backend,
        h3_python_bin=args.h3_python_bin,
        manage_server=not args.no_manage_server,
        ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))


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
                "character-batch": _character_batch,
                "character-batch-accept": _character_batch_accept,
                "character-recall-v82": _target_recall_v82,
                "omni-edit-trial": _omni_edit_trial,
                "reference-program-v9": _reference_program_v9,
                "reference-program-v9-accept": _reference_program_v9_accept,
                "reference-generate-v9g": _reference_generate_v9g,
                "reference-generate-v9g-accept": _reference_generate_v9g_accept,
                "long-take-lt0": _long_take_lt0,
                "long-take-produce": _long_take_produce,
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
