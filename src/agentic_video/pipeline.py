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
from src.agentic_video.verify_slots import (blind_video_check,
                                            deterministic_story_check)
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
    # P0 参考身份核对：任何验收前先核对 reference_id ↔ 文件 ↔ sha256 ↔ 标题
    identity = agent_result.get("reference_identity")
    if isinstance(identity, dict) and identity:
        (output_dir / "reference_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
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
            # V4 B4：coarse 槽（无锚且 >2×budget）localize-or-reject——在渲染
            # 前定位，绝不带盲切区间进片。无 runner（render 子命令手动路径）
            # 时跳过并在日志注明。
            if any((slot.get("source") or {}).get("anchor") == "coarse"
                   for slot in story_plan.get("slots") or []):
                if runner is not None:
                    from src.agentic_video.verify_slots import localize_coarse_slots

                    localize_coarse_slots(cfg, story_plan, runner=runner,
                                          output_dir=output_dir)
                else:
                    logging.getLogger(__name__).warning(
                        "[render] coarse slots present but no runner——localize "
                        "跳过（手动 render 路径），区间将按槽预算截断")
        # V3 P4：文案不再在渲染前生成——critic 轮看的是无文案的干净素材，
        # copy 在验证收敛后最后生成并逐句绑定证据（run_full 尾部）。
        # 传入的 story_plan 自带 copy（人工改写/render 子命令）则照用。
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


def _overall_verdict(det: dict | None, blind: dict | None, grounding: dict | None,
                     story: dict, *, blind_required: bool) -> dict:
    """V4 E3 总门控（外审三轮必改③：blind 开启时 None/unparsed/inconsistent
    一律 blocked——原「blind is None → 放行」会让没跑成的盲看比失败的盲看
    分数还高）。

    passed = det.passed
          AND plan_complete（必选槽无 unsupported）
          AND 文案轨道达标（assertion 型：hook 与 punchline 均存活——替换
              可以，消失不行；空文案轨=copy_track_empty）
          AND blind_required ? (parsed AND consistent_protagonist) : True
    失败 reasons 逐项命名，供 blocked 交付的判读。"""
    reasons: list[str] = []
    if not (det or {}).get("passed"):
        reasons.append("det_violations:" + ";".join((det or {}).get("violations") or [])
                       or "det_failed")
    incomplete = [int(slot["slot_idx"]) for slot in (story.get("slots") or [])
                  if slot.get("status") == "unsupported"
                  and (slot.get("need_spec") or {}).get("required")]
    if incomplete:
        reasons.append(f"plan_incomplete:{incomplete}")
    cues = list(((story.get("copy") or {}).get("cues")) or [])
    kinds = {str(cue.get("kind")) for cue in cues}
    if not cues:
        reasons.append("copy_track_empty")
    elif "hook_line" not in kinds or "punchline" not in kinds:
        missing = [kind for kind in ("hook_line", "punchline") if kind not in kinds]
        reasons.append(f"copy_required_kinds_missing:{missing}")
    if blind_required:
        if blind is None:
            reasons.append("blind_missing")
        elif not blind.get("parsed"):
            reasons.append("blind_unparsed")
        elif blind.get("consistent_protagonist") is not True:
            reasons.append("blind_inconsistent")
    return {"passed": not reasons, "reasons": reasons,
            "blind_required": blind_required}


def _release_gpu_cache() -> None:
    """critic 前释放检索/渲染阶段驻留的显存（2026-09-10 v4：E5 检索 + 渲染后
    GPU0 仅剩 10.9GiB，叙事 critic 的视觉编码要 11.45GiB 直接 OOM 崩掉整轮）。
    allocator 缓存不随局部变量释放，需要显式 empty_cache；2026-09-11 罗小黑
    P4 实锤第二层：E5 模型对象在 retrieval 后虽无引用但带循环引用，不 gc.collect
    就不回收——15GB 驻留把 B/C 的 critic 阶段双双顶死。"""
    try:
        import gc

        import torch

        gc.collect()
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
    re_search_log = []
    critic_cfg = cfg.library.get("critic_v2") or {}
    max_rounds = int(critic_cfg.get("max_rounds", 2))
    min_improvement = float(critic_cfg.get("min_improvement", 0.02))
    # P3 分层重搜的候选池（初次规划的完整候选组）与已拒绝清单
    candidate_pool: dict[int, list[dict]] = {}
    pool_path = output_dir / "story_candidates.json"
    if pool_path.exists():
        try:
            candidate_pool = {idx: group for idx, group
                              in enumerate(json.loads(pool_path.read_text(encoding="utf-8")))}
        except (ValueError, OSError):
            candidate_pool = {}
    rejected_sources: list[dict] = []

    def _re_render(round_dir_name: str):
        nonlocal asset_plan, retrieval, current_video
        round_dir = output_dir / round_dir_name
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

    last_round_modified = False
    verification_enabled = bool(((cfg.library.get("verification") or {})
                                  .get("enabled", True)))
    for round_idx in range(1, max_rounds + 1):
        # ① 槽级验证：看实际进片区间，must_have 逐条核对（每轮一次验证 pass）。
        # 失败槽进分层重搜（层1候选池→层2改写查询全库），换掉的旧素材进已拒绝
        # 清单——重搜不是换一轮旧候选。verification.enabled=false 时整体跳过
        # （P4 的 B/C 对照开关：B=关，C=开）。
        round_reports = []
        if verification_enabled:
            # 注意：deterministic_story_check/blind_video_check 用模块级导入——
            # 此处再局部 import 会让整个函数作用域里的同名绑定变成局部变量，
            # verification 关闭时（B4 实锤）尾部调用直接 UnboundLocalError
            from src.agentic_video.verify_slots import verify_slots
            from src.agentic_video.story_planner import re_search_slot

            verification = verify_slots(cfg, current_story, runner=runner)
            (output_dir / f"verification_round_{round_idx}.json").write_text(
                json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
            # V3 P3 全局确定性检查（零模型）：主角锁定槽缺人/未解释主体切换
            # → 与槽级失败槽一起进分层重搜。Slot0=小黑、Slot1=无限在这里就被拦。
            det_check = deterministic_story_check(current_story)
            (output_dir / f"deterministic_check_round_{round_idx}.json").write_text(
                json.dumps(det_check, ensure_ascii=False, indent=2), encoding="utf-8")
            failed_idxs = list(verification["failed_slots"])
            for det_idx in det_check["re_search_slots"]:
                if det_idx not in failed_idxs:
                    failed_idxs.append(det_idx)
            for failed_idx in failed_idxs:
                old_source = current_story["slots"][failed_idx].get("source") or {}
                if old_source.get("video"):
                    rejected_sources.append({
                        "video": old_source.get("video"),
                        "start_s": old_source.get("start_s"),
                        "end_s": old_source.get("end_s"),
                        "reason": "verification_failed"})
                report = re_search_slot(
                    cfg, current_story, failed_idx, theme=theme,
                    candidate_pool=candidate_pool.get(failed_idx),
                    rejected=rejected_sources)
                # V4 E2：验证通道直接改 current_story 且当场重渲 → 恒采用
                report["channel"] = "verification"
                report["adopted"] = True
                round_reports.append(report)
            if any(row.get("status") == "replaced" for row in round_reports):
                _re_render(f"verify_render_{round_idx}")      # 重搜生效先重渲再给 critic 看
                det_check = deterministic_story_check(current_story)
                (output_dir / f"deterministic_check_round_{round_idx}.json").write_text(
                    json.dumps(det_check, ensure_ascii=False, indent=2), encoding="utf-8")

        narrative_critique = run_narrative_critic(
            current_video, narrative, current_story, runner=runner)
        edit_critique = run_structured_critic(
            current_video, current_recipe, asset_plan, retrieval, runner=runner,
            narrative=narrative)
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
        # ② critic 重搜指令通道：断点描述 + 改写需求 → 分层重搜（critic 不自己编素材）。
        # 与 ① 同属 P4 的 C 档能力，B 档（纯 need/facet/路径约束）跳过。
        if verification_enabled and narrative_critique.get("re_search"):
            from src.agentic_video.story_planner import re_search_slot

            for directive in narrative_critique["re_search"]:
                slot_idx = int(directive["slot_idx"])
                if not 0 <= slot_idx < len(patched_story["slots"]):
                    continue
                old_source = patched_story["slots"][slot_idx].get("source") or {}
                if old_source.get("video"):
                    rejected_sources.append({
                        "video": old_source.get("video"),
                        "start_s": old_source.get("start_s"),
                        "end_s": old_source.get("end_s"),
                        "reason": f"critic: {directive.get('reason') or ''}"[:80]})
                report = re_search_slot(
                    cfg, patched_story, slot_idx, theme=theme,
                    candidate_pool=candidate_pool.get(slot_idx),
                    need_hint=directive.get("need_hint"), rejected=rejected_sources)
                report["channel"] = "critic"    # patched_story 是 deepcopy 副本——
                round_reports.append(report)    # 采用与否在循环收敛判定之后回填
        re_search_log.extend(round_reports)   # 引用 dict：adopted 可事后回填
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
                       or any(row["status"] == "applied" for row in story_audit)
                       or any(row.get("status") == "replaced" for row in round_reports))

        def _mark_critic_adopted(value: bool) -> None:
            # V4 E2：critic 通道的重搜发生在 deepcopy 副本上——循环在采用前
            # break 时这些"replaced"从未进入 current_story。日志必须区分
            # "提出"与"采用"（V3_C3 实锤：日志记 slot2→5100s，终稿仍 2775s）。
            for row in round_reports:
                if row.get("channel") == "critic":
                    row["adopted"] = value

        if not any_applied:
            _mark_critic_adopted(False)
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="no_applicable_patch")
            last_round_modified = False
            break
        if any(row["state_hash"] == next_hash for row in history):
            _mark_critic_adopted(False)
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="state_repeated")
            last_round_modified = False
            break
        if len(history) >= 2 and history[-1]["score"] - history[-2]["score"] < min_improvement:
            _mark_critic_adopted(False)
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="insufficient_improvement")
            last_round_modified = False
            break
        current_recipe, current_story = patched_recipe, patched_story
        _mark_critic_adopted(True)
        _re_render(f"critic_render_{round_idx}")
        last_round_modified = True
        if round_idx >= max_rounds:
            manifest.stage("critics", "complete", rounds=round_idx,
                           stop_reason="max_rounds")
            break
    else:
        manifest.stage("critics", "complete", rounds=max_rounds,
                       stop_reason="max_rounds")
        last_round_modified = max_rounds > 0

    # ③ V4 终版流水线（外审三轮必改③：审谁就交谁）：det → 文案（烧录前
    # grounding 丢弃/替换）→ 真终渲（文案+mix/bgm 终混）→ 盲看看**终版** →
    # overall 门控。失败仍产出（调试预览）但 delivery=blocked——判读以
    # final_review.overall 为准，不再"机器指标全绿"式自欺。
    final_review: dict = {}
    det_final = deterministic_story_check(current_story)   # 零模型，恒跑
    final_review["deterministic"] = det_final
    from src.agentic_video.copywriter import build_copy_cues, validate_copy_grounding

    current_story["copy"] = build_copy_cues(current_story, runner=runner)
    grounding = validate_copy_grounding(current_story["copy"], current_story)
    (output_dir / "copy_grounding.json").write_text(
        json.dumps(grounding, ensure_ascii=False, indent=2), encoding="utf-8")
    final_review["copy_grounding"] = grounding
    _re_render("final_render")                            # 带文案终渲=真交付物
    blind = None
    if verification_enabled:                                # B 档关（对照开关）
        # V3 晨修（B3/C3 实锤）：critic 两轮 + copy ask 后进程驻留 46.85GB，
        # 盲看 watch 直接 OOM——盲看前必须清显存
        _release_gpu_cache()
        blind = blind_video_check(current_video, runner=runner)
        (output_dir / "blind_review.json").write_text(
            json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
        final_review["blind"] = {
            "consistent_protagonist": blind.get("consistent_protagonist"),
            "main_character": blind.get("main_character"),
            "story_in_one_sentence": blind.get("story_in_one_sentence"),
        }
    overall = _overall_verdict(det_final, blind, grounding, current_story,
                               blind_required=verification_enabled)
    final_review["overall"] = overall
    manifest.stage("final_review", "complete",
                   det_passed=det_final.get("passed"),
                   blind_consistent=(blind or {}).get("consistent_protagonist"),
                   copy_dropped=len(grounding.get("dropped") or []),
                   overall_passed=overall["passed"])
    manifest.stage("delivery", "complete" if overall["passed"] else "blocked",
                  reasons=overall["reasons"])
    (output_dir / "final_review.json").write_text(
        json.dumps(final_review, ensure_ascii=False, indent=2), encoding="utf-8")
    if re_search_log:
        (output_dir / "re_search_log.json").write_text(
            json.dumps(re_search_log, ensure_ascii=False, indent=2), encoding="utf-8")

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
