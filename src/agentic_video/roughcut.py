"""Roughcut（V5 P5，外审六轮阶段二）：单窗真实粗剪控制实验。

素材侧最小闭环：一个真实源窗口 → 合成 RoughcutSpec 适配 planner →
规划/localize/verify/det → 真实源片段出片（**绝不导黑片**）。回答的
问题：已存在的对白能否被找到、精确切出、保留原声、让观众看懂——
连一段完整对话都无法保真搬运时，不配挑战跨电影多事件重构。

provenance 纪律：program_source="roughcut_test_fixture"（素材侧控制实验，
绝不与 reference-derived narrative 混淆——外审六轮第二十三节）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import AppConfig, repo_root

logger = logging.getLogger(__name__)

ROUGHCAST_PROGRAM_SOURCE = "roughcut_test_fixture"


def build_roughcut_narrative(rows: list[dict], window_idx: int,
                             reference_uri: str) -> dict:
    """从单窗索引行确定性合成最小叙事程序（无模型调用）。

    events=窗内 event 行；arc=按 story_role 聚合的 2-4 段；entities=窗内
    参与者。只消费已核实的索引文本（P0 修复后的 search_text 语义）。"""
    events, entities = [], []
    seen_entities: set[str] = set()
    role_events: dict[str, list[str]] = {}
    for row in rows:
        event_id = str(row.get("event_id") or f"w{window_idx}_e{len(events)}")
        participants = list(row.get("entity_ids") or [])[:2]
        if event_id not in {e["id"] for e in events}:
            events.append({
                "id": event_id,
                "interval": [float(row.get("start_s") or 0),
                             float(row.get("end_s") or 0)],
                "participants": participants,
                "action": str(row.get("caption") or row.get("event_summary")
                              or "")[:60],
                "state_before": "uncertain", "state_after": "uncertain",
                "importance": "medium", "evidence": [
                    {"source": f"window_{window_idx}"}],
                "confidence": float(row.get("annotation_confidence") or 0.5),
                "status": "supported" if row.get("event_summary") else "uncertain",
            })
        role = str(row.get("story_role") or "context")
        role_events.setdefault(role, []).append(event_id)
        for name in row.get("entity_names") or []:
            key = str(name)
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append({"id": f"ent_{len(entities)}",
                                 "name_or_role": key,
                                 "type": "person_or_creature",
                                 "evidence": [{"source": f"window_{window_idx}"}],
                                 "confidence": 0.6, "status": "uncertain"})
    # arc：按固定角色序聚合（hook→context→conflict→…），最多 4 段
    order = ["hook", "context", "conflict", "choice", "climax",
             "consequence", "resolution"]
    arc = []
    for role in order:
        if role in role_events and len(arc) < 4:
            arc.append({"role": role, "event_ids": role_events[role][:2]})
    if not arc:
        arc = [{"role": "context", "event_ids": [e["id"] for e in events[:2]]}]
    return {
        "program_version": "1.0",
        "reference": {"id": f"roughcut_w{window_idx}",
                      "uri": reference_uri, "sha256": "",
                      "duration_s": 20.0, "fps": 24.0},
        "provenance": {"model": ROUGHCAST_PROGRAM_SOURCE},
        "status": "supported",
        "evidence": [{"source": f"window_{window_idx}_annotations"}],
        "intent": {"topic": "素材侧粗剪控制实验（单窗保真搬运）",
                   "message": "验证索引→检索→定位→成片链在真实源窗口上的可用性",
                   "content_type": "roughcut",
                   "protagonist": "uncertain", "goal": "uncertain",
                   "problem": "uncertain", "motivation": "uncertain",
                   "change": "uncertain", "outcome": "uncertain",
                   "evidence": [{"source": "roughcut"}], "confidence": 0.5,
                   "status": "uncertain"},
        "entities": entities, "events": events, "arc": arc,
        "causal_links": [], "utterances": [], "emotion_curve": [],
        "uncertainties": ["roughcut：合成的最小叙事程序，仅供素材链路控制实验"],
    }


def run_roughcut(cfg: AppConfig, source: str, window_idx: int, *, theme: str,
                 output_dir: Path, target_duration_s: float = 22.0,
                 runner=None, force: bool = False) -> Path:
    """单窗粗剪编排。必选槽 unsupported → fail loudly（不渲黑片）。"""
    from src.agentic_video.narrative import validate_narrative_program
    from src.agentic_video.recipe_v2 import new_recipe
    from src.agentic_video.story_planner import (build_story_plan_from_index,
                                                 fit_slot_intervals)
    from src.agentic_video.verify_slots import (deterministic_story_check,
                                                localize_coarse_slots,
                                                verify_slots)
    from src.library.build_index import load_index

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "rendered.mp4").exists() and not force:
        logger.info("[roughcut] 已存在，跳过（--force 重跑）")
        return output_dir / "rendered.mp4"

    rows, _emb = load_index(cfg)
    window_rows = [row for row in rows
                   if str(row.get("video_stem") or "").startswith(f"{source}__")
                   and int(row.get("window_idx", -1)) == int(window_idx)]
    if not window_rows:
        raise FileNotFoundError(f"索引无 {source} 窗口 {window_idx} 的行"
                                "（先跑 index / 检查窗口号）")
    video = str(window_rows[0].get("video") or "")

    narrative = build_roughcut_narrative(window_rows, window_idx, video)
    narrative["template_choice"] = {"form": "assertion_visual_payoff",
                                    "source": "manual"}
    errors = validate_narrative_program(narrative)
    if errors:
        raise ValueError(f"roughcut 合成程序不合法: {errors[:3]}")
    (output_dir / "reference.narrative.json").write_text(
        json.dumps(narrative, ensure_ascii=False, indent=2), encoding="utf-8")

    plan, _candidates = build_story_plan_from_index(
        cfg, narrative, theme=theme, library=source,
        target_duration_s=target_duration_s)
    (output_dir / "story_candidates.json").write_text(
        json.dumps(_candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    if runner is not None:
        localize_coarse_slots(cfg, plan, runner=runner, output_dir=output_dir)

    required_missing = [int(slot["slot_idx"]) for slot in plan["slots"]
                        if slot.get("status") == "unsupported"
                        and (slot.get("need_spec") or {}).get("required")]
    if required_missing:
        (output_dir / "gap_report.json").write_text(json.dumps({
            "failed": "required_slots_missing",
            "slots": required_missing,
            "message": "必选槽 unsupported——fail loudly，不导黑片（外审："
                       "真实源窗口不导黑片占位）"}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise RuntimeError(f"roughcut {source} w{window_idx}: 必选槽缺失 "
                           f"{required_missing}（见 gap_report.json）")

    fit_slot_intervals(plan, mode="content_preserving")

    det = deterministic_story_check(plan)
    (output_dir / "deterministic_check.json").write_text(
        json.dumps(det, ensure_ascii=False, indent=2), encoding="utf-8")

    # 最小 Recipe（纯硬切）+ 渲染
    from src.agentic_video.pipeline import run_rendering
    recipe = new_recipe(reference_id=f"roughcut_w{window_idx}",
                        reference_uri=video, sha256="",
                        duration_s=float(plan["target_duration_s"]), fps=24.0)
    _plan, _retrieval, final = run_rendering(
        cfg, recipe, theme=theme, library=source, output_dir=output_dir,
        force=True, use_mask_backend=False, narrative=narrative,
        story_plan=plan, target_duration_s=float(plan["target_duration_s"]),
        runner=runner)
    return Path(final)
