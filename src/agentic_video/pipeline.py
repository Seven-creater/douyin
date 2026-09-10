"""End-to-end orchestration for the Agentic Video CLI."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from src.agentic_video.agent import AgentBudget, build_recipe_from_agent, run_bounded_agent
from src.agentic_video.critic_v2 import (apply_recipe_patches, run_structured_critic,
                                         should_stop)
from src.agentic_video.manifest import RunManifest, json_hash
from src.agentic_video.narrative_agent import NarrativeBudget, run_narrative_agent
from src.agentic_video.narrative_critic import (apply_story_patches,
                                                 run_narrative_critic)
from src.agentic_video.planner import build_asset_plan, run_asset_retrieval
from src.agentic_video.recipe_v2 import recipe_hash, sha256_file, write_recipe
from src.agentic_video.renderer import GroundedSamSubprocessBackend, render_recipe
from src.agentic_video.report import write_run_report
from src.agentic_video.story_planner import (build_story_plan_from_index,
                                               story_plan_execution_inputs,
                                               write_story_plan)
from src.config import AppConfig, repo_root
from src.library.signals import run_signals
from src.perception.detect_beats import run_for_video as run_beats
from src.perception.detect_shots import run_for_video as run_shots
from src.perception.extract_frames import run_for_video as run_frames
from src.perception.inspect_video import run_for_video as run_inspect
from src.perception.ocr_frames import run_for_video as run_ocr
from src.perception.transcribe_audio import run_for_video as run_transcribe


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


def _write_decomposition_report(path: Path, narrative: dict) -> Path:
    supported = 0
    uncertain = 0
    evidence = 0
    for section in ("entities", "events", "causal_links", "utterances", "emotion_curve"):
        for row in narrative.get(section) or []:
            supported += row.get("status") == "supported"
            uncertain += row.get("status") == "uncertain"
            evidence += len(row.get("evidence") or [])
    lines = [f"# Narrative Decomposition · {narrative['reference']['id']}", "",
             f"- 状态：{narrative.get('status')}，顶层证据："
             f"{len(narrative.get('evidence') or [])}",
             f"- 主题：{narrative['intent'].get('topic')}",
             f"- 表达：{narrative['intent'].get('message')}",
             f"- 实体：{len(narrative.get('entities') or [])}",
             f"- 事件：{len(narrative.get('events') or [])}",
             f"- 因果边：{len(narrative.get('causal_links') or [])}",
             f"- 证据：{evidence}，supported={supported}，uncertain={uncertain}", "",
             "## 叙事弧", ""]
    for segment in narrative.get("arc") or []:
        lines.append(f"- **{segment['role']}**：{', '.join(segment.get('event_ids') or [])}")
    if narrative.get("uncertainties"):
        lines.extend(["", "## 未确定项", ""])
        lines.extend(f"- {item}" for item in narrative["uncertainties"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_narrative_decomposition(cfg: AppConfig, reference: Path, output_dir: Path, *,
                                force: bool = False, runner=None,
                                budget: NarrativeBudget | None = None) -> tuple[dict, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vid, registered = register_reference(cfg, reference)
    budget_cfg = cfg.library.get("narrative_agent") or {}
    actual_budget = budget or NarrativeBudget(
        max_initial_windows=int(budget_cfg.get("max_initial_windows", 24)),
        max_rounds=int(budget_cfg.get("max_rounds", 2)),
        max_refinement_windows=int(budget_cfg.get("max_refinement_windows", 12)),
        target_window_s=float(budget_cfg.get("target_window_s", 12.0)))
    manifest = RunManifest(output_dir / "run_manifest.json", command="decompose",
                           input_path=registered,
                           config={"narrative_budget": asdict(actual_budget)},
                           repo_root=repo_root())
    for name, fn in (("inspect", run_inspect), ("frames", run_frames),
                     ("shots", run_shots), ("ocr", run_ocr),
                     ("transcribe", run_transcribe), ("beats", run_beats)):
        manifest.stage(f"narrative_{name}", "running")
        fn(cfg, vid, force=force)
        manifest.stage(f"narrative_{name}", "complete")
    manifest.stage("narrative_agent", "running")
    narrative = run_narrative_agent(
        cfg, vid, output=output_dir / "reference.narrative.json",
        budget=actual_budget, force=force, runner=runner)
    agent_result = json.loads((cfg.paths.perception_dir / vid / "narrative_agent" /
                               "result.json").read_text(encoding="utf-8"))
    manifest.stage("narrative_agent", "complete",
                   initial_windows=agent_result.get("initial_windows"),
                   refinement_windows=agent_result.get("refinement_windows"),
                   stop_reason=agent_result.get("stop_reason"))
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name in ("inspect", "frames", "shots", "ocr", "transcribe", "beats"):
        source = cfg.paths.perception_dir / vid / name / "result.json"
        if source.exists():
            shutil.copy2(source, evidence_dir / f"{name}.json")
    report = _write_decomposition_report(output_dir / "decomposition_report.md", narrative)
    manifest.stage("narrative_program", "complete", events=len(narrative["events"]),
                   report=str(report))
    return narrative, vid


def run_rendering(cfg: AppConfig, recipe: dict, *, theme: str, library: str,
                  output_dir: Path, force: bool = False,
                  use_mask_backend: bool = True, narrative: dict | None = None,
                  story_plan: dict | None = None,
                  target_duration_s: float = 60.0,
                  runner=None) -> tuple[dict, list[dict], Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if narrative is not None:
        if story_plan is None:
            story_plan, candidate_groups = build_story_plan_from_index(
                cfg, narrative, theme=theme, library=library,
                target_duration_s=target_duration_s)
            (output_dir / "story_candidates.json").write_text(
                json.dumps(candidate_groups, ensure_ascii=False, indent=2), encoding="utf-8")
        # 文案轨（2026-09-10 MVP，7682 型模板）：钩子/成就卡/反转梗 + BGM 模式。
        # setdefault 语义：critic 轮传入的 current_story 已带 copy 则不重生成，
        # 文案跨轮稳定；人工改 story_plan.json 的 copy 后走 render 子命令可重渲。
        if "copy" not in story_plan:
            from src.agentic_video.copywriter import build_copy_cues

            story_plan["copy"] = build_copy_cues(story_plan, runner=runner)
        write_story_plan(story_plan, output_dir / "story_plan.json")
        asset_plan, retrieval = story_plan_execution_inputs(story_plan, recipe)
        retrieval_path = output_dir / "retrieval_results.json"
        retrieval_path.write_text(json.dumps(retrieval, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    else:
        asset_plan = build_asset_plan(recipe, theme, library=library)
        retrieval_path = run_asset_retrieval(
            cfg, asset_plan, output_dir / "retrieval_results.json", force=force)
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    (output_dir / "asset_plan.json").write_text(
        json.dumps(asset_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    backend_cfg = cfg.library.get("mask_backend") or {}
    backend = (GroundedSamSubprocessBackend(backend_cfg)
               if use_mask_backend and backend_cfg else None)
    final = render_recipe(cfg, recipe, asset_plan, retrieval, output_dir,
                          mask_backend=backend, force=force)
    return asset_plan, retrieval, final


def _release_gpu_cache() -> None:
    """critic 前释放检索/渲染阶段驻留的显存（2026-09-10 v4：E5 检索 + 渲染后
    GPU0 仅剩 10.9GiB，叙事 critic 的视觉编码要 11.45GiB 直接 OOM 崩掉整轮）。
    allocator 缓存不随局部变量释放，需要显式 empty_cache。"""
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - 无 torch/CPU 环境静默跳过
        pass


def run_full(cfg: AppConfig, reference: Path, *, theme: str, library: str,
             output_dir: Path, force: bool = False, runner=None,
             use_mask_backend: bool = True,
             target_duration_s: float = 60.0) -> Path:
    output_dir = Path(output_dir)
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
    recipe, _vid = run_decomposition(cfg, reference, output_dir, force=force, runner=runner)
    narrative, _ = run_narrative_decomposition(
        cfg, reference, output_dir, force=force, runner=runner)
    manifest = RunManifest(output_dir / "run_manifest.json", command="run",
                           input_path=Path(recipe["reference"]["uri"]),
                           config={"theme": theme, "library": library,
                                   "target_duration_s": target_duration_s},
                           repo_root=repo_root())
    manifest.data["command"] = "run"
    manifest.stage("render", "running")
    asset_plan, retrieval, final = run_rendering(
        cfg, recipe, theme=theme, library=library, output_dir=output_dir,
        force=force, use_mask_backend=use_mask_backend, narrative=narrative,
        target_duration_s=target_duration_s, runner=runner)
    manifest.stage("render", "complete", output=str(final))
    _release_gpu_cache()

    current_recipe = recipe
    current_story = json.loads((output_dir / "story_plan.json").read_text(encoding="utf-8"))
    current_video = final
    history = []
    edit_critiques = []
    narrative_critiques = []
    critic_cfg = cfg.library.get("critic_v2") or {}
    max_rounds = int(critic_cfg.get("max_rounds", 2))
    min_improvement = float(critic_cfg.get("min_improvement", 0.02))
    for round_idx in range(1, max_rounds + 1):
        narrative_critique = run_narrative_critic(
            current_video, narrative, current_story, runner=runner)
        edit_critique = run_structured_critic(
            current_video, current_recipe, asset_plan, retrieval, runner=runner)
        narrative_critiques.append(narrative_critique)
        edit_critiques.append(edit_critique)
        (output_dir / f"narrative_critic_round_{round_idx}.json").write_text(
            json.dumps(narrative_critique, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / f"edit_critic_round_{round_idx}.json").write_text(
            json.dumps(edit_critique, ensure_ascii=False, indent=2), encoding="utf-8")
        patched_recipe, recipe_audit = apply_recipe_patches(
            current_recipe, edit_critique.get("patches") or [])
        patched_story, story_audit = apply_story_patches(
            current_story, narrative_critique.get("patches") or [])
        (output_dir / f"recipe_patch_round_{round_idx}.json").write_text(
            json.dumps({"patches": edit_critique.get("patches") or [],
                        "audit": recipe_audit},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / f"story_patch_round_{round_idx}.json").write_text(
            json.dumps({"patches": narrative_critique.get("patches") or [],
                        "audit": story_audit}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        score = (float(edit_critique.get("score") or 0)
                 + float(narrative_critique.get("theme_relevance") or 0)
                 + float(narrative_critique.get("narrative_coherence") or 0)) / 3
        next_hash = json_hash({"recipe": patched_recipe, "story": patched_story})
        history.append({"round": round_idx, "score": score,
                        "state_hash": json_hash({"recipe": current_recipe,
                                                 "story": current_story})})
        any_applied = (any(row["status"] == "applied" for row in recipe_audit)
                       or any(row["status"] == "applied" for row in story_audit))
        if not any_applied:
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="no_applicable_patch")
            break
        if any(row["state_hash"] == next_hash for row in history):
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="state_repeated")
            break
        if len(history) >= 2 and history[-1]["score"] - history[-2]["score"] < min_improvement:
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="insufficient_improvement")
            break
        current_recipe, current_story = patched_recipe, patched_story
        round_dir = output_dir / f"critic_render_{round_idx}"
        asset_plan, retrieval, current_video = run_rendering(
            cfg, current_recipe, theme=theme, library=library, output_dir=round_dir,
            force=True, use_mask_backend=use_mask_backend, narrative=narrative,
            story_plan=current_story, target_duration_s=target_duration_s,
            runner=runner)
        shutil.copy2(current_video, output_dir / "rendered.mp4")
        for name in ("story_plan.json", "asset_plan.json", "retrieval_results.json",
                     "render_manifest.json", "subtitles.srt"):
            source = round_dir / name
            if source.exists():
                shutil.copy2(source, output_dir / name)
        write_recipe(current_recipe, output_dir / "reference.recipe.json")
        _release_gpu_cache()
        if round_idx >= max_rounds:
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="max_rounds")
            break
    else:
        manifest.stage("critics", "complete", rounds=max_rounds,
                       stop_reason="max_rounds")

    render_path = output_dir / "render_manifest.json"
    render_manifest = (json.loads(render_path.read_text(encoding="utf-8"))
                       if render_path.exists() else {})
    report = write_run_report(output_dir / "report.md", recipe=current_recipe,
                              asset_plan=asset_plan, retrieval=retrieval,
                              render_manifest=render_manifest, critiques=edit_critiques,
                              narrative=narrative, story_plan=current_story,
                              narrative_critiques=narrative_critiques)
    manifest.stage("report", "complete", output=str(report))
    return output_dir / "rendered.mp4"
