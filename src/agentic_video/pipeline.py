"""End-to-end orchestration for the Agentic Video CLI."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from src.agentic_video.agent import AgentBudget, build_recipe_from_agent, run_bounded_agent
from src.agentic_video.critic_v2 import (apply_recipe_patches, run_structured_critic,
                                         should_stop)
from src.agentic_video.manifest import RunManifest
from src.agentic_video.planner import build_asset_plan, run_asset_retrieval
from src.agentic_video.recipe_v2 import recipe_hash, sha256_file, write_recipe
from src.agentic_video.renderer import GroundedSamSubprocessBackend, render_recipe
from src.agentic_video.report import write_run_report
from src.config import AppConfig, repo_root
from src.library.signals import run_signals
from src.perception.detect_beats import run_for_video as run_beats
from src.perception.detect_shots import run_for_video as run_shots
from src.perception.inspect_video import run_for_video as run_inspect


def register_reference(cfg: AppConfig, reference: Path) -> tuple[str, Path]:
    reference = Path(reference).resolve()
    if not reference.exists():
        raise FileNotFoundError(reference)
    digest = sha256_file(reference)
    if reference.name == "video.mp4" and reference.parent.parent == cfg.paths.videos_dir:
        return reference.parent.name, reference
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_"
                        for char in reference.stem)[:48]
    vid = f"{safe_stem}_{digest[:10]}"
    target = cfg.paths.videos_dir / vid / "video.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != reference.stat().st_size:
        shutil.copy2(reference, target)
    return vid, target


def run_decomposition(cfg: AppConfig, reference: Path, output_dir: Path, *,
                      force: bool = False, runner=None,
                      budget: AgentBudget | None = None) -> tuple[dict, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vid, registered = register_reference(cfg, reference)
    budget_cfg = cfg.library.get("agent") or {}
    actual_budget = budget or AgentBudget(
        max_initial_windows=int(budget_cfg.get("max_initial_windows", 48)),
        max_rounds=int(budget_cfg.get("max_rounds", 2)),
        max_refinement_windows=int(budget_cfg.get("max_refinement_windows", 16)),
        confidence_threshold=float(budget_cfg.get("confidence_threshold", 0.65)))
    manifest = RunManifest(output_dir / "run_manifest.json", command="decompose",
                           input_path=registered, config={"budget": asdict(actual_budget)},
                           repo_root=repo_root())
    manifest.stage("inspect", "running")
    run_inspect(cfg, vid, force=force)
    manifest.stage("inspect", "complete")
    manifest.stage("shots", "running")
    run_shots(cfg, vid, force=force)
    manifest.stage("shots", "complete")
    manifest.stage("beats", "running")
    run_beats(cfg, vid, force=force)
    manifest.stage("beats", "complete")
    manifest.stage("signals", "running")
    run_signals(cfg, vid, force=force)
    manifest.stage("signals", "complete")
    manifest.stage("agent", "running")
    agent_path = run_bounded_agent(cfg, vid, budget=actual_budget, force=force, runner=runner)
    agent_result = json.loads(agent_path.read_text(encoding="utf-8"))
    manifest.stage("agent", "complete", tool_calls=len(agent_result.get("tool_calls") or []),
                   stop_reason=agent_result.get("stop_reason"))
    recipe_path = output_dir / "reference.recipe.json"
    recipe = build_recipe_from_agent(cfg, vid, agent_result, output=recipe_path)
    manifest.stage("recipe", "complete", operations=len(recipe["operations"]),
                   recipe_hash=recipe_hash(recipe))
    return recipe, vid


def run_rendering(cfg: AppConfig, recipe: dict, *, theme: str, library: str,
                  output_dir: Path, force: bool = False,
                  use_mask_backend: bool = True) -> tuple[dict, list[dict], Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_plan = build_asset_plan(recipe, theme, library=library)
    (output_dir / "asset_plan.json").write_text(
        json.dumps(asset_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    retrieval_path = run_asset_retrieval(cfg, asset_plan, output_dir / "retrieval_results.json",
                                         force=force)
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    backend_cfg = cfg.library.get("mask_backend") or {}
    backend = (GroundedSamSubprocessBackend(backend_cfg)
               if use_mask_backend and backend_cfg else None)
    final = render_recipe(cfg, recipe, asset_plan, retrieval, output_dir,
                          mask_backend=backend, force=force)
    return asset_plan, retrieval, final


def run_full(cfg: AppConfig, reference: Path, *, theme: str, library: str,
             output_dir: Path, force: bool = False, runner=None,
             use_mask_backend: bool = True) -> Path:
    output_dir = Path(output_dir)
    recipe, _vid = run_decomposition(cfg, reference, output_dir, force=force, runner=runner)
    manifest = RunManifest(output_dir / "run_manifest.json", command="run",
                           input_path=Path(recipe["reference"]["uri"]),
                           config={"theme": theme, "library": library}, repo_root=repo_root())
    manifest.data["command"] = "run"
    manifest.stage("render", "running")
    asset_plan, retrieval, final = run_rendering(
        cfg, recipe, theme=theme, library=library, output_dir=output_dir,
        force=force, use_mask_backend=use_mask_backend)
    manifest.stage("render", "complete", output=str(final))

    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
    current_recipe = recipe
    current_video = final
    history = []
    critiques = []
    critic_cfg = cfg.library.get("critic_v2") or {}
    max_rounds = int(critic_cfg.get("max_rounds", 2))
    min_improvement = float(critic_cfg.get("min_improvement", 0.02))
    for round_idx in range(1, max_rounds + 1):
        critique = run_structured_critic(current_video, current_recipe, asset_plan,
                                         retrieval, runner=runner)
        critiques.append(critique)
        (output_dir / f"critic_round_{round_idx}.json").write_text(
            json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")
        patched, audit = apply_recipe_patches(current_recipe, critique.get("patches") or [])
        (output_dir / f"recipe_patch_round_{round_idx}.json").write_text(
            json.dumps({"patches": critique.get("patches") or [], "audit": audit},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        history.append({"round": round_idx, "score": critique.get("score"),
                        "recipe_hash": recipe_hash(current_recipe)})
        if not any(row["status"] == "applied" for row in audit):
            manifest.stage("critic", "complete", rounds=round_idx,
                           stop_reason="no_applicable_patch")
            break
        # The round budget counts applied revisions. Check repetition/improvement
        # before rendering, then stop after the final allowed revision.
        stop, reason = should_stop(history, patched, max_rounds=max_rounds + 1,
                                   min_improvement=min_improvement)
        if stop:
            manifest.stage("critic", "complete", rounds=round_idx, stop_reason=reason)
            break
        current_recipe = patched
        round_dir = output_dir / f"critic_render_{round_idx}"
        asset_plan, retrieval, current_video = run_rendering(
            cfg, current_recipe, theme=theme, library=library, output_dir=round_dir,
            force=True, use_mask_backend=use_mask_backend)
        shutil.copy2(current_video, output_dir / "rendered.mp4")
        for name in ("asset_plan.json", "retrieval_results.json", "render_manifest.json"):
            shutil.copy2(round_dir / name, output_dir / name)
        write_recipe(current_recipe, output_dir / "reference.recipe.json")
        if round_idx >= max_rounds:
            manifest.stage("critic", "complete", rounds=round_idx, stop_reason="max_rounds")
            break
    else:
        manifest.stage("critic", "complete", rounds=max_rounds, stop_reason="max_rounds")

    render_path = output_dir / "render_manifest.json"
    render_manifest = (json.loads(render_path.read_text(encoding="utf-8"))
                       if render_path.exists() else {})
    report = write_run_report(output_dir / "report.md", recipe=current_recipe,
                              asset_plan=asset_plan, retrieval=retrieval,
                              render_manifest=render_manifest, critiques=critiques)
    manifest.stage("report", "complete", output=str(report))
    return output_dir / "rendered.mp4"
