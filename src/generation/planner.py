"""planner：template.timeline → 生成单元（role 感知贪心合并 + 17n+5 帧数 + 裁剪映射）。

CLI：python -m src.generation.planner --template-id <id> [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.generation import models
from src.perception import common

logger = logging.getLogger(__name__)


def clip_frames_for(needed_s: float, *, fps: int = models.DEFAULT_FPS,
                    max_frames: int = 260, min_frames: int = models.MIN_FRAMES) -> int:
    """最小 17n+5 使 frames/fps ≥ needed_s；夹在 [min_frames, max_frames]。"""
    needed = max(math.ceil(needed_s * fps), min_frames)
    n = math.ceil((needed - models.FRAME_BASE) / models.FRAME_STEP)
    frames = models.FRAME_BASE + n * models.FRAME_STEP
    return max(min_frames, min(frames, max_frames))


def _dur(seg: dict) -> float:
    return float(seg.get("end", 0)) - float(seg.get("start", 0))


def _split_if_over(group: list[dict], max_unit_s: float) -> list[list[dict]]:
    """单 role 组超 max → 在段边界均衡拆分（每片尽量等长，段本身超 max 时单独成片）。"""
    if sum(_dur(s) for s in group) <= max_unit_s:
        return [group]
    parts: list[list[dict]] = []
    cur: list[dict] = []
    for seg in group:
        if cur and sum(_dur(s) for s in cur) + _dur(seg) > max_unit_s:
            parts.append(cur)
            cur = []
        cur.append(seg)
    if cur:
        parts.append(cur)
    return parts


def merge_segments(timeline: list[dict], *, min_unit_s: float, max_unit_s: float) -> list[list[dict]]:
    """role 感知贪心：A 不足 min 无条件吸收；B 同 role 继续吸收；C 超限拆分。"""
    groups: list[list[dict]] = []
    i = 0
    while i < len(timeline):
        group = [timeline[i]]
        # 规则 A
        while (sum(_dur(s) for s in group) < min_unit_s and i + 1 < len(timeline)
               and sum(_dur(s) for s in group) + _dur(timeline[i + 1]) <= max_unit_s):
            i += 1
            group.append(timeline[i])
        # 规则 B：与组内最后一段同 role 则继续吸收（叙事连贯）
        while (i + 1 < len(timeline)
               and timeline[i + 1].get("role") == group[-1].get("role")
               and sum(_dur(s) for s in group) + _dur(timeline[i + 1]) <= max_unit_s):
            i += 1
            group.append(timeline[i])
        groups.extend(_split_if_over(group, max_unit_s))
        i += 1
    return groups


def plan_units(template: dict, *, template_id: str, plan_cfg: dict) -> models.GenerationPlan:
    fps = int(plan_cfg.get("fps", models.DEFAULT_FPS))
    head_pad = float(plan_cfg.get("head_pad_s", 0.3))
    tail_pad = float(plan_cfg.get("tail_pad_s", 0.3))
    min_unit_s = float(plan_cfg.get("min_unit_s", 5.2))
    max_unit_s = float(plan_cfg.get("max_unit_s", 10.0))
    max_frames = int(plan_cfg.get("max_frames", 260))

    timeline = [s for s in template.get("timeline") or [] if _dur(s) > 0.01]
    groups = merge_segments(timeline, min_unit_s=min_unit_s, max_unit_s=max_unit_s)

    units: list[models.GenerationUnit] = []
    for unit_id, group in enumerate(groups):
        duration_s = sum(_dur(s) for s in group)
        needed = duration_s + head_pad + tail_pad
        num_frames = clip_frames_for(needed, fps=fps, max_frames=max_frames)
        segs: list[models.UnitSegment] = []
        offset = head_pad
        for seg in group:
            segs.append(models.UnitSegment(
                seg_index=int(seg.get("_idx", 0)),
                start_s=float(seg["start"]), end_s=float(seg["end"]),
                unit_offset_s=round(offset, 3),
            ))
            offset += _dur(seg)
        units.append(models.GenerationUnit(
            unit_id=unit_id,
            roles=[s.get("role", "other") for s in group],
            timeline_span=(float(group[0]["start"]), float(group[-1]["end"])),
            duration_s=round(duration_s, 3),
            num_frames=num_frames,
            seed_base=models.fnv1a(f"{template_id}|{unit_id}") % 2**31,
            visual_brief="；".join(s.get("visual") or "" for s in group if s.get("visual")),
            speech_lines=[s.get("speech") for s in group if s.get("speech")],
            segments=segs,
        ))
    audio = template.get("audio") or {}
    return models.GenerationPlan(
        template_id=template_id,
        source_duration_s=round(sum(_dur(s) for s in timeline), 3),
        audio_brief=audio.get("bgm"),
        units=units,
    )


def plan_to_output(plan: models.GenerationPlan) -> dict:
    return {
        "plan": {
            "template_id": plan.template_id,
            "source_duration_s": plan.source_duration_s,
            "audio_brief": plan.audio_brief,
            "units": [
                {
                    "unit_id": u.unit_id, "roles": u.roles,
                    "timeline_span": list(u.timeline_span),
                    "duration_s": u.duration_s, "num_frames": u.num_frames,
                    "seed_base": u.seed_base, "visual_brief": u.visual_brief,
                    "speech_lines": u.speech_lines,
                    "segments": [
                        {"seg_index": s.seg_index, "start_s": s.start_s, "end_s": s.end_s,
                         "unit_offset_s": s.unit_offset_s} for s in u.segments
                    ],
                }
                for u in plan.units
            ],
        }
    }


def load_template(cfg: AppConfig, template_id: str) -> dict:
    env = common.read_result_json(cfg.paths.perception_dir / template_id / "template")
    if env is None or env.get("tool") != "extract_template":
        raise FileNotFoundError(
            f"缺少模板产物：data/perception/{template_id}/template/result.json（先跑 Phase 3 run_templates）"
        )
    return env["output"]["template"]


def run_for_template(cfg: AppConfig, template_id: str, *, force: bool = False,
                     dry_run: bool = False) -> models.GenerationPlan | None:
    plan_cfg = (cfg.generation or {}).get("plan") or {}
    template = load_template(cfg, template_id)
    # 给 timeline 段带上原始下标（merge 后 seg_index 仍可追溯）
    for idx, seg in enumerate(template.get("timeline") or []):
        seg["_idx"] = idx

    gdir = cfg.paths.generation_dir / template_id
    pdir = gdir / "plan"
    pdir.mkdir(parents=True, exist_ok=True)
    params = {"prompt_version": "plan_v1", **{k: plan_cfg.get(k) for k in
               ("fps", "width", "height", "min_unit_s", "max_unit_s", "head_pad_s", "tail_pad_s")}}
    if not dry_run and common.done_or_skip(pdir, params, force=force):
        logger.info("[plan %s] 已有产物，跳过", template_id)
        return None

    plan = plan_units(template, template_id=template_id, plan_cfg=plan_cfg)
    if dry_run:
        return plan
    common.write_result_json(pdir, tool="plan_units", aweme_id=template_id,
                             params=params, output=plan_to_output(plan))
    logger.info("[plan %s] %d 个单元（总时长 %.1fs）→ %s", template_id,
                len(plan.units), plan.source_duration_s, pdir / "result.json")
    return plan


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="timeline → 生成单元规划")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="gen_plan")
    try:
        plan = run_for_template(cfg, args.template_id, force=args.force, dry_run=args.dry_run)
        if plan is None:
            common.emit_status_line("ok", skipped=True)
            return 0
        print(f"模板 {plan.template_id}（{plan.source_duration_s}s）→ {len(plan.units)} 个生成单元：")
        for u in plan.units:
            print(f"  u{u.unit_id:02d} [{u.timeline_span[0]:.1f}-{u.timeline_span[1]:.1f}s] "
                  f"{u.duration_s:5.1f}s → {u.num_frames}f ({u.num_frames/24:.1f}s) "
                  f"roles={'+'.join(u.roles)} segs={len(u.segments)}")
        common.emit_status_line("ok", units=len(plan.units))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("planner 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
