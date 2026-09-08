"""run_generation：Phase 4 全流程编排（plan → variants → rewrite → generate → assemble）。

CLI：python -m src.generation.run_generation --template-id <id> [--variants 3]
        [--instances 4] [--stage all|plan|variants|rewrite|generate|assemble] [--allow-missing] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.generation import assemble as gen_assemble
from src.generation import planner as gen_planner
from src.generation import rewrite as gen_rewrite
from src.generation import variant as gen_variant
from src.generation.minimax_client import ClipTask, GenerationManifest, GenerationScheduler, MiniMaxService
from src.perception import common

logger = logging.getLogger(__name__)


def run_generation(cfg: AppConfig, *, template_id: str, n_variants: int, instances: int,
                   stage: str = "all", variant_file: Path | None = None,
                   allow_missing: bool = False, force: bool = False) -> dict:
    t_all = time.time()
    g_cfg = cfg.generation or {}
    gdir = cfg.paths.generation_dir / template_id
    gdir.mkdir(parents=True, exist_ok=True)

    # 1) plan
    plan = gen_planner.run_for_template(cfg, template_id, force=force) \
        or gen_assemble.load_plan(cfg, template_id)
    logger.info("[run] plan: %d 单元 %.1fs", len(plan.units), plan.source_duration_s)
    if stage == "plan":
        return {"stage": "plan", "units": len(plan.units)}

    template = gen_planner.load_template(cfg, template_id)
    runner = None

    # 2) variants
    variants_path = gdir / "variants.json"
    if variant_file:
        specs = gen_variant.load_variant_file(variant_file)
        vmeta = {"mode": "file"}
    elif variants_path.exists() and not force:
        specs = [gen_variant.VariantSpec(**v) for v in json.loads(variants_path.read_text(encoding="utf-8"))["variants"]]
        vmeta = {"mode": "cached"}
    else:
        if runner is None:
            from src.perception.omni_runner import OmniRunner

            runner = OmniRunner(cfg.perception.get("omni") or {})
        r_cfg = g_cfg.get("rewrite") or {}
        specs, vmeta = gen_variant.propose_variants(
            runner, template, n_variants,
            max_new_tokens=int(r_cfg.get("max_new_tokens", 2048)),
            max_retries=int(r_cfg.get("max_retries", 1)))
    for w in gen_variant.validate_variants(specs, template.get("replaceable_elements") or []):
        logger.warning("[variants] %s", w)
    variants_path.write_text(json.dumps(
        {"mode": vmeta.get("mode"), "variants": [s.__dict__ for s in specs]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[run] variants(%s): %s", vmeta.get("mode"), [s.variant_id for s in specs])
    if stage == "variants":
        return {"stage": "variants", "variants": [s.variant_id for s in specs]}

    # 3) rewrite（每变体）
    for spec in specs:
        gen_rewrite.run_for_variant(cfg, plan, spec, force=force, runner=runner)
    if stage == "rewrite":
        return {"stage": "rewrite", "n": len(specs)}

    # 4) generate
    mm_cfg = dict(g_cfg.get("minimax") or {})
    mm_cfg.setdefault("serve_cwd", str(cfg.paths.logs_dir.parent))  # 仓库根：-m 启动 fl2va serve 用
    svc = MiniMaxService(mm_cfg)
    plan_cfg = g_cfg.get("plan") or {}
    manifest = GenerationManifest(gdir / "manifest.json", gdir / "variants")
    tasks = []
    for spec in specs:
        ups = {p["unit_id"]: p for p in gen_assemble.load_prompts(cfg, template_id, spec.variant_id)}
        for u in plan.units:
            if manifest.is_done(spec.variant_id, u.unit_id):
                continue
            p = ups[u.unit_id]
            tasks.append(ClipTask(
                variant_id=spec.variant_id, unit_id=u.unit_id, prompt=p["prompt"],
                seed=int(p["seed"]), num_frames=int(u.num_frames),
                output_name=f"{template_id}_{spec.variant_id}_u{u.unit_id:02d}.mp4"))
    if stage == "generate" or (stage == "all" and tasks):
        ready, warns = svc.ensure_instances(instances)
        for w in warns:
            logger.warning("[minimax] %s", w)
        if ready:
            scheduler = GenerationScheduler(ready, svc.http, manifest, mm_cfg,
                                            gdir / "variants",
                                            width=int(plan_cfg.get("width", 544)),
                                            height=int(plan_cfg.get("height", 960)))
            gen_stats = scheduler.run(svc, tasks)
            logger.info("[run] generate: %s", gen_stats)
        elif tasks:
            logger.error("[run] 无可用 MiniMax 实例，跳过生成（已有产物仍可装配）")
    if stage == "generate":
        return {"stage": "generate", "done": sum(1 for e in manifest.data["clips"].values()
                                                 if e.get("status") == "done")}

    # 5) assemble
    finals = []
    for spec in specs:
        try:
            finals.append(str(gen_assemble.assemble_variant(cfg, template_id, spec.variant_id,
                                                            allow_missing=allow_missing)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[assemble %s] 失败", spec.variant_id)
            finals.append(f"ERROR: {exc}")

    return {"stage": "all", "variants": [s.variant_id for s in specs], "finals": finals,
            "total_s": round(time.time() - t_all, 1)}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Phase 4 生成管线")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--stage", default="all",
                    choices=["all", "plan", "variants", "rewrite", "generate", "assemble"])
    ap.add_argument("--variant-file", type=Path, default=None)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="generation_run")

    try:
        result = run_generation(cfg, template_id=args.template_id, n_variants=args.variants,
                                instances=args.instances or int((cfg.generation or {}).get("minimax", {}).get("max_instances", 4)),
                                stage=args.stage, variant_file=args.variant_file,
                                allow_missing=args.allow_missing, force=args.force)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_generation 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1

    out = cfg.paths.processed_dir / f"generation_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("=" * 60)
    print(f"生成管线 Summary（stage={result['stage']}）")
    for k, v in result.items():
        if k != "stage":
            print(f"  {k}: {v}")
    print(f"产物: {out}")
    print("=" * 60)
    common.emit_status_line("ok", stage=result["stage"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
