"""Bounded active-perception agent for multi-operation edit decomposition."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agentic_video.recipe_v2 import (OP_TYPES, POINT_OP_TYPES,
                                         SUPPORTED_RENDER_OPS, new_recipe,
                                         sha256_file, write_recipe)
from src.config import AppConfig
from src.library.decompose import parse_window_answer, run_decompose
from src.perception import common

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentBudget:
    max_initial_windows: int = 48
    max_rounds: int = 2
    max_refinement_windows: int = 16
    confidence_threshold: float = 0.65


@dataclass(frozen=True)
class ProbeTask:
    round_idx: int
    source_window_idx: int
    start: float
    end: float
    probe: str
    reason: str
    hypotheses: tuple[str, ...]


def answer_operations(answer: Any) -> list[dict]:
    if not isinstance(answer, dict):
        return []
    operations = answer.get("operations")
    if isinstance(operations, list):
        return [op for op in operations if isinstance(op, dict)]
    return [answer] if "op_type" in answer else []


def choose_probe(hypotheses: list[str], operations: list[dict]) -> str:
    changed = {op.get("what_changed") for op in operations}
    types = set(hypotheses)
    if "text" in changed or types & {"text_layer_animation"}:
        return "ocr_probe"
    if types & {"tracked_mask_fill", "subject_cutout_composite", "mask_wipe",
                "foreground_occlusion_transition"}:
        return "region_segmentation_probe"
    if types & {"speed_ramp", "zoom_punch", "whip_pan", "beat_freeze"}:
        return "motion_probe"
    return "focused_multiframe_probe"


def propose_refinement_tasks(windows_plan: list[dict], results: list[dict], *,
                              round_idx: int, remaining: int,
                              confidence_threshold: float = 0.65,
                              seen: set[tuple] | None = None) -> list[ProbeTask]:
    """Prioritize uncertain and layer-mismatch windows without exceeding remaining."""
    seen = seen or set()
    by_idx = {int(r.get("idx", -1)): r for r in results}
    ranked = []
    layer_types = {"tracked_mask_fill", "subject_cutout_composite", "mask_wipe",
                   "foreground_occlusion_transition", "text_layer_animation"}
    for window in windows_plan:
        operations = answer_operations((by_idx.get(window["idx"]) or {}).get("answer"))
        reported = {op.get("op_type") for op in operations}
        low_confidence = (not operations or any(op.get("op_type") == "uncertain"
                          or float(op.get("confidence", 0)) < confidence_threshold
                          for op in operations))
        missed_layer = bool((set(window.get("hypotheses") or []) & layer_types) - reported)
        if not (low_confidence or missed_layer):
            continue
        priority = (2 if missed_layer else 0) + (1 if low_confidence else 0)
        center = (float(window["start"]) + float(window["end"])) / 2
        valid_times = [float(op["event_time_original_s"])
                       for op in operations
                       if isinstance(op.get("event_time_original_s"), (int, float))]
        if valid_times:
            center = valid_times[0]
        start = round(max(float(window["start"]), center - 0.45), 3)
        end = round(min(float(window["end"]), center + 0.45), 3)
        if end - start < 0.25:
            start, end = float(window["start"]), float(window["end"])
        probe = choose_probe(list(window.get("hypotheses") or []), operations)
        key = (round(start, 2), round(end, 2), probe)
        if key in seen:
            continue
        reason = "layer_hypothesis_not_explained" if missed_layer else "low_confidence"
        ranked.append((priority, float(window.get("confidence") or 0), ProbeTask(
            round_idx=round_idx, source_window_idx=int(window["idx"]), start=start,
            end=end, probe=probe, reason=reason,
            hypotheses=tuple(window.get("hypotheses") or []))))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2].start))
    return [item[2] for item in ranked[:max(0, remaining)]]


PROBE_INSTRUCTIONS = {
    "ocr_probe": "逐帧检查文字是否出现、消失、位移、缩放或改变透明度。",
    "region_segmentation_probe": "逐帧比较主体内部、轮廓和背景，区分遮罩、抠图叠加与普通切换。",
    "motion_probe": "跟踪固定视觉点，区分素材自身运动、整体相机运动、变速和定格。",
    "focused_multiframe_probe": "按时间顺序逐帧比较变化，可同时报告多个独立剪辑操作。",
}


def build_probe_prompt(task: ProbeTask, duration_s: float,
                       probe_evidence: dict | None = None) -> str:
    from src.library.decompose import WINDOW_PROMPT

    signature = (f"主动探针={task.probe}；原因={task.reason}；"
                 f"候选={'/'.join(task.hypotheses) or 'unknown'}；"
                 f"专项要求={PROBE_INSTRUCTIONS[task.probe]}；"
                 f"工具证据={json.dumps(probe_evidence or {}, ensure_ascii=False)}")
    return WINDOW_PROMPT.format(signature=signature, start=task.start, end=task.end,
                                duration=duration_s,
                                ops="/".join((*OP_TYPES, "uncertain")))


def _evidence_keys(results: list[dict]) -> set[tuple]:
    keys = set()
    for row in results:
        for op in answer_operations(row.get("answer")):
            if op.get("op_type") == "uncertain":
                continue
            t = op.get("event_time_original_s")
            keys.add((op.get("op_type"), round(float(t), 2) if isinstance(t, (int, float)) else None,
                      op.get("what_changed")))
    return keys


def run_bounded_agent(cfg: AppConfig, vid: str, *, budget: AgentBudget | None = None,
                      force: bool = False, runner=None) -> Path:
    """Run initial D2 windows and at most two active refinement rounds."""
    budget = budget or AgentBudget()
    d_cfg = dict(cfg.library.get("windows") or {})
    d_cfg["max_windows"] = min(int(d_cfg.get("max_windows", 48)), budget.max_initial_windows)
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        omni_cfg = {**cfg.perception.get("omni", {}),
                    "fps": float(d_cfg.get("fps", 8.0))}
        runner = OmniRunner(omni_cfg)

    aggregate_path = run_decompose(cfg, vid, force=force, runner=runner)
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    initial_results = list(aggregate.get("results") or [])
    all_results = list(initial_results)
    tool_calls = [{"tool": "omni_window", "window_idx": row.get("idx"),
                   "phase": "initial"} for row in initial_results]
    seen: set[tuple] = set()
    refinements: list[dict] = []
    before = _evidence_keys(all_results)
    root = cfg.paths.library_dir / "editing" / vid / "agent"
    duration = float((common.read_result_json(cfg.paths.perception_dir / vid / "inspect")
                      or {}).get("output", {}).get("duration_s") or 0)
    video = cfg.paths.videos_dir / vid / "video.mp4"

    for round_idx in range(1, budget.max_rounds + 1):
        remaining = budget.max_refinement_windows - len(refinements)
        if remaining <= 0:
            break
        tasks = propose_refinement_tasks(
            aggregate.get("windows_plan") or [], all_results, round_idx=round_idx,
            remaining=remaining, confidence_threshold=budget.confidence_threshold, seen=seen)
        if not tasks:
            break
        round_results = []
        for task_idx, task in enumerate(tasks):
            seen.add((round(task.start, 2), round(task.end, 2), task.probe))
            task_dir = root / f"round_{round_idx}" / f"{task_idx:03d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            from src.agentic_video.probes import collect_probe_evidence
            probe_evidence = collect_probe_evidence(
                video, task, task_dir,
                ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))
            ans = runner.watch(video, build_probe_prompt(task, duration, probe_evidence),
                               start_s=task.start, end_s=task.end, clip_dir=task_dir,
                               max_new_tokens=int(d_cfg.get("max_new_tokens", 1024)),
                               duration_s=task.end - task.start)
            window = {"start": task.start, "end": task.end}
            parsed = parse_window_answer(ans.text, window)
            row = {"idx": task.source_window_idx, "phase": "refinement",
                   "round": round_idx, "probe": task.probe, "task": asdict(task),
                   "probe_evidence": probe_evidence,
                   "answer": parsed, "parse": "json" if parsed else "raw",
                   "raw_head": None if parsed else ans.text[:300]}
            common.write_result_json(task_dir, tool="agent_probe", aweme_id=vid,
                                     params=asdict(task), output=row)
            refinements.append(row)
            round_results.append(row)
            tool_calls.append({"tool": task.probe, "window_idx": task.source_window_idx,
                               "round": round_idx, "start": task.start, "end": task.end})
        all_results.extend(round_results)
        after = _evidence_keys(all_results)
        if after == before:
            break
        before = after

    result = {
        "aweme_id": vid, "budget": asdict(budget), "initial": initial_results,
        "refinements": refinements, "results": all_results,
        "tool_calls": tool_calls, "n_evidence": len(_evidence_keys(all_results)),
        "stop_reason": ("refinement_budget_exhausted"
                        if len(refinements) >= budget.max_refinement_windows
                        else "no_new_evidence_or_no_tasks"),
    }
    output = root / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _track_for_operation(op_type: str) -> tuple[str, str, int]:
    if op_type.startswith("text"):
        return "text_main", "text", 20
    if op_type in {"tracked_mask_fill", "mask_wipe"}:
        return "mask_main", "mask", 10
    if op_type in {"subject_cutout_composite", "foreground_occlusion_transition",
                   "opacity_blend"}:
        return "overlay_main", "overlay", 10
    return "video_main", "video", 0


def build_recipe_from_agent(cfg: AppConfig, vid: str, agent_result: dict,
                            *, output: Path | None = None) -> dict:
    """Convert grounded window answers into an executable Recipe v2."""
    inspect = (common.read_result_json(cfg.paths.perception_dir / vid / "inspect")
               or {}).get("output") or {}
    duration = float(inspect.get("duration_s") or 0)
    fps = float(inspect.get("fps") or 24)
    video = cfg.paths.videos_dir / vid / "video.mp4"
    recipe = new_recipe(reference_id=vid, reference_uri=str(video),
                        sha256=sha256_file(video), duration_s=duration, fps=fps,
                        model="bounded_active_perception",
                        prompt_version="window_v2", seed=0)
    recipe["provenance"]["tool_calls"] = list(agent_result.get("tool_calls") or [])
    tracks = {track["id"]: track for track in recipe["tracks"]}
    seen: set[tuple] = set()
    for row in agent_result.get("results") or []:
        task = row.get("task") or {}
        for raw in answer_operations(row.get("answer")):
            reported_type = raw.get("op_type")
            op_type = reported_type if reported_type in OP_TYPES else "unknown_edit"
            event_t = raw.get("event_time_original_s")
            if not isinstance(event_t, (int, float)):
                event_t = float(task.get("start", 0))
            event_t = min(duration, max(0.0, float(event_t)))
            interval = raw.get("interval_original_s")
            if not (isinstance(interval, list) and len(interval) == 2):
                if op_type in POINT_OP_TYPES:
                    interval = [event_t, event_t]
                else:
                    start = float(task.get("start", max(0.0, event_t - 0.2)))
                    end = float(task.get("end", min(duration, event_t + 0.2)))
                    interval = [max(0.0, start), min(duration, max(end, start + 0.001))]
            interval = [round(max(0.0, float(interval[0])), 6),
                        round(min(duration, float(interval[1])), 6)]
            # Omni occasionally reports a layer operation at a single frame.
            # Point operations may keep that representation; duration-based
            # operations need a small evidence-bounded interval to remain
            # executable instead of making the whole Recipe invalid.
            if interval[1] <= interval[0] and op_type not in POINT_OP_TYPES:
                task_start = max(0.0, float(task.get("start", event_t)))
                task_end = min(duration, float(task.get("end", event_t)))
                if task_end > task_start:
                    interval = [round(task_start, 6), round(task_end, 6)]
                else:
                    left = max(0.0, event_t - 0.2)
                    right = min(duration, max(event_t + 0.2, left + 0.001))
                    interval = [round(left, 6), round(right, 6)]
            key = (op_type, round(event_t, 2), raw.get("what_changed"))
            if key in seen:
                continue
            seen.add(key)
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0))))
            if op_type == "unknown_edit" or confidence < 0.5:
                status = "uncertain"
            elif op_type in SUPPORTED_RENDER_OPS:
                status = "supported"
            else:
                status = "unsupported"
            track_id, track_type, z = _track_for_operation(op_type)
            tracks.setdefault(track_id, {"id": track_id, "type": track_type, "z": z})
            recipe["operations"].append({
                "id": f"op_{len(recipe['operations']):04d}", "type": op_type,
                "interval": interval, "track_id": track_id, "inputs": [],
                "depends_on": [],
                "params": {k: raw[k] for k in ("subject", "what_changed", "motion",
                                                 "direction", "flash") if k in raw},
                "evidence": [{"source": row.get("probe") or "omni_window",
                              "window_idx": row.get("idx"), "timestamp_s": event_t,
                              "quote": str(raw.get("evidence_quote") or "")[:200]}],
                "confidence": confidence, "status": status,
            })
    recipe["tracks"] = sorted(tracks.values(), key=lambda item: (item["z"], item["id"]))
    recipe["operations"].sort(key=lambda item: (item["interval"][0], item["track_id"],
                                                 item["id"]))
    for idx, op in enumerate(recipe["operations"]):
        op["id"] = f"op_{idx:04d}"
    if output is not None:
        write_recipe(recipe, output)
    return recipe
