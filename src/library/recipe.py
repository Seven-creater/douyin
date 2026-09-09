"""recipe：D3 Editing Recipe JSON——把成片反编译成"可执行剪辑程序"。

Editing Branch V1（2026-09-09 计划）：signals 候选（物理证据）+ decompose 窗口答案
（Omni 判读）+ 感知产物（节拍/镜头边界/字幕）→ 规则底稿（候选直填，零编造）+
LLM 合成（rich 化：主体/图层/参数/证据引用）→ 三层防线（校验/repair/penalty 回退）。

结构：flat operations[] + segments[]（渲染阶段再 group-by）。
反编造：每个 op.t_s 必须落在已知证据时间戳 ±tolerance（信号候选∪镜头边界∪节拍∪
字幕事件∪窗口自报时刻）；change_anchor 必须 ±0.15s 命中节拍；
texture_sequence 只有窗口确认过 tracked_mask_fill 才允许。

CLI：python -m src.library.recipe --id <vid> [--force] [--dry-run]
产物：data/library/editing/<id>/recipe/result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common
from src.template.prompts import dedupe_json_repetition
from src.template.schema import extract_json_block

logger = logging.getLogger(__name__)

OP_TYPES = ("hard_cut", "match_cut", "crossfade", "mask_wipe",
            "foreground_occlusion_transition", "luma_flash", "speed_ramp",
            "tracked_mask_fill", "subject_cutout_composite", "text_layer_animation",
            "montage_burst", "zoom_punch", "whip_pan", "beat_freeze")

CAMERA_MOTION = ("static", "pan_left", "pan_right", "tilt_up", "tilt_down",
                 "zoom_in", "zoom_out", "follow", "forward", "uncertain")
SHOT_SCALE = ("closeup", "medium", "wide", "uncertain")

RECIPE_SKELETON = """{"video": {"aweme_id": "...", "duration_s": 0.0},
 "tempo_bpm_est": 0.0,
 "segments": [{"index": 0, "start": 0.0, "end": 3.02,
   "source": {"subject": "walking person", "shot_scale": "medium", "camera_motion": "forward"},
   "transition_out": {"type": "foreground_occlusion_transition", "duration": 0.18}}],
 "operations": [{"t_s": 3.02, "type": "tracked_mask_fill", "target": "person",
   "texture_sequence": ["stone", "fire"], "change_anchor": 3.05, "confidence": 0.8,
   "evidence": {"source": "window", "window_idx": 5, "quote": "人像内部纹理逐帧变化"}}],
 "global_notes": "整体风格一句话"}"""

RECIPE_PROMPT = """你是剪辑程序反编译器。下面给了一条爆款剪辑视频的全部确定性证据
（信号候选/窗口判读/节拍/镜头边界/字幕事件），把它们综合成一份 Editing Recipe JSON：
不仅回答"哪里切了"，还要回答"为什么切、主体是谁、有没有图层/参数/转场类型"。

【硬性规则】
1. 只输出一个 JSON 对象，无围栏无解释，结构照抄骨架。
2. segments[].start/end 只能取：0、视频时长、镜头边界、切点类候选时刻（我已给清单）；
   首段 start=0，末段 end≈时长，升序不重叠。
3. operations[].t_s 只能取「证据时刻清单」里的值（±0.25s 程序侧校验，直接抄）；
   type 只能取：{ops}。
4. change_anchor 只能取节拍点；texture_sequence 只有窗口判读确认 tracked_mask_fill/
   subject_interior 变化时才允许填，否则删掉该键。
5. 窗口判读为 uncertain 的不要写成操作；证据不足的操作宁可不要（宁缺毋滥）；
   operations 最多 30 条（挑最具代表性的，普通 hard_cut 只留标志性几处）。
6. evidence 引用：来自哪个 window_idx（给了编号）、信号签名一句话。
7. transition_out.type 只能取 {ops} 或删键（段尾无转场）。

【输出骨架】
{skeleton}

【证据】
"""


def bpm_from_beats(beats: list[float]) -> float | None:
    """节拍点 → BPM 估计（间隔中位数）。"""
    if len(beats) < 4:
        return None
    ivs = sorted(b - a for a, b in zip(beats[:-1], beats[1:]) if 0.2 < b - a < 2.0)
    if not ivs:
        return None
    return round(60.0 / ivs[len(ivs) // 2], 2)


def load_recipe_inputs(cfg: AppConfig, vid: str) -> dict:
    """汇聚 signals/windows/感知产物 → meta（证据时刻/窗口确认/边界）。"""
    pdir = cfg.paths.perception_dir / vid
    edir = cfg.paths.library_dir / "editing" / vid

    def _out(tool: str, base: Path) -> dict | None:
        env = common.read_result_json(base / tool)
        return env["output"] if env else None

    insp = _out("inspect", pdir) or {}
    duration = float(insp.get("duration_s") or 0)
    beats = list((_out("beats", pdir) or {}).get("beat_points_s") or [])
    bounds = list((_out("shots", pdir) or {}).get("boundaries_s") or [])
    ocr_events = ((_out("ocr", pdir) or {}).get("text_events") or [])
    sig = _out("signals", edir) or {}
    cands = list(sig.get("candidates") or [])
    win_raw = edir / "windows" / "result.json"        # 汇总文件是裸 JSON（无信封层）
    win_agg = {}
    if win_raw.exists():
        obj = json.loads(win_raw.read_text(encoding="utf-8"))
        win_agg = obj.get("output") or obj           # 兼容信封/裸两种形态
    answers = [r for r in (win_agg.get("results") or [])
               if r.get("parse") == "json" and r.get("answer")]

    window_confirms = []
    for r in answers:
        a = r["answer"]
        t = a.get("event_time_original_s")
        if a.get("op_type") not in (None, "uncertain") and isinstance(t, (int, float)):
            window_confirms.append({"t_s": round(float(t), 2),
                                    "op_type": a["op_type"],
                                    "idx": r.get("idx"),
                                    "quote": (a.get("evidence_quote") or "")[:80],
                                    "subject": a.get("subject") or ""})
    window_confirms.sort(key=lambda w: w["t_s"])

    known_ts = sorted({round(float(b), 2) for b in beats}
                      | {round(float(b), 2) for b in bounds}
                      | {round(c["t_s"], 2) for c in cands}
                      | {round(e.get("t_start_s", 0), 2) for e in ocr_events
                         if isinstance(e.get("t_start_s"), (int, float))}
                      | {w["t_s"] for w in window_confirms})
    # montage 成员时刻也入锚（组内每刀都是真事件）
    for c in cands:
        for m in c.get("members_s") or []:
            known_ts.append(round(float(m), 2))
    known_ts = sorted(set(known_ts))

    cut_ts = {round(c["t_s"], 2) for c in cands
              if set(c["type_hypotheses"]) & {"hard_cut", "match_cut", "montage_burst"}}
    for c in cands:
        for m in c.get("members_s") or []:
            cut_ts.add(round(float(m), 2))
    seg_bounds = sorted({0.0, round(duration, 2)} | cut_ts
                        | {round(float(b), 2) for b in bounds})

    return {"duration_s": duration, "beats": beats, "bounds": bounds,
            "ocr_events": ocr_events, "candidates": cands,
            "window_confirms": window_confirms, "known_ts": known_ts,
            "seg_bounds": seg_bounds, "bpm": bpm_from_beats(beats)}


def build_context(meta: dict, *, max_chars: int = 8000) -> str:
    """证据 → 紧凑上下文（预算截断：头 85% 尾 10%，同 template/context 惯例）。"""
    parts: list[str] = []
    d = meta["duration_s"]
    parts.append(f"[视频] 时长 {d:.2f}s，BPM 估计 {meta['bpm']}")
    beats = meta["beats"]
    if beats:
        parts.append("[节拍点·change_anchor 只能取这些] "
                     + ", ".join(f"{b:g}" for b in beats[:80])
                     + (" …" if len(beats) > 80 else ""))
    if meta["bounds"]:
        parts.append("[镜头边界] " + ", ".join(f"{b:g}" for b in meta["bounds"][:60]))
    if meta["candidates"]:
        # 候选行截断：模型会逐条转写全部候选 → 输出超长截断（GLM 版实测 12KB 仍砍断）。
        # top30 按置信度 + 类型保底；known_ts 清单仍全量（校验锚不变）
        by_conf = sorted(meta["candidates"], key=lambda c: -c["confidence"])
        shown, seen_types = [], set()
        for c in by_conf:
            new_type = any(h not in seen_types for h in c["type_hypotheses"])
            if len(shown) >= 30 and not new_type:          # 满额后只放行新类型
                continue
            shown.append(c)
            seen_types.update(c["type_hypotheses"])
        shown.sort(key=lambda c: c["t_s"])
        omitted = len(meta["candidates"]) - len(shown)
        lines = [f"- t={c['t_s']:g} [{'/'.join(c['type_hypotheses'])}] conf={c['confidence']} "
                 f"sig={{{', '.join(f'{k}={v}' for k, v in c['signature'].items())}}}"
                 + (f" members={c['members_s']}" if c.get("members_s") else "")
                 for c in shown]
        if omitted > 0:
            lines.append(f"（其余 {omitted} 个低置信候选已省略，不要为它们编操作）")
        parts.append("[信号候选·t_s 优先取这些]\n" + "\n".join(lines))
    if meta["window_confirms"]:
        lines = [f"- 窗{w['idx']} t={w['t_s']:g} {w['op_type']} 主体「{w['subject'][:20]}」"
                 f"证据「{w['quote']}」" for w in meta["window_confirms"]]
        parts.append("[窗口判读·window_idx 对应这里]\n" + "\n".join(lines))
    if meta["ocr_events"]:
        parts.append("[字幕事件] " + "; ".join(
            f"{e.get('t_start_s', 0):g}s「{str(e.get('text', ''))[:12]}」"
            for e in meta["ocr_events"][:12]))
    ts = meta["known_ts"]
    parts.append("[证据时刻清单·operations[].t_s 与 segments 边界只能取这些] "
                 + ", ".join(f"{t:g}" for t in ts[:200]) + (" …" if len(ts) > 200 else ""))
    text = "\n".join(parts)
    if len(text) > max_chars:
        head, tail = int(max_chars * 0.85), int(max_chars * 0.10)
        text = text[:head] + "\n…(截断)…\n" + text[-tail:]
    return text


def build_rule_draft(meta: dict, vid: str) -> dict:
    """规则底稿：候选直填（零编造），既是回退底牌也是反编造锚。"""
    cand_to_op = {"texture_replace": "tracked_mask_fill"}
    seg_bounds = meta["seg_bounds"]
    segments = [{"index": i, "start": round(a, 2), "end": round(b, 2)}
                for i, (a, b) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:]))
                if b > a]
    operations = []
    for c in meta["candidates"]:
        ops = [cand_to_op.get(h, h) for h in c["type_hypotheses"] if h in OP_TYPES
               or h in cand_to_op]
        if not ops:
            continue
        operations.append({"t_s": c["t_s"], "type": ops[0],
                           "confidence": round(min(c["confidence"], 0.4), 2),
                           "evidence": {"source": "signals",
                                        "sig": ", ".join(f"{k}={v}" for k, v
                                                         in c["signature"].items())}})
    return {"video": {"aweme_id": vid, "duration_s": round(meta["duration_s"], 2)},
            "tempo_bpm_est": meta["bpm"],
            "segments": segments, "operations": operations,
            "global_notes": "规则底稿（LLM 合成失败回退）"}


def validate_recipe(obj, meta: dict, *, tol_s: float = 0.25,
                    anchor_tol_s: float = 0.15) -> list[str]:
    """Recipe 硬校验：结构/枚举/证据锚定/节拍锚定/去重。"""
    if not isinstance(obj, dict):
        return ["顶层不是 JSON 对象"]
    errors: list[str] = []
    for k in ("video", "segments", "operations"):
        if k not in obj:
            errors.append(f"缺键 {k}")
    if errors:
        return errors
    d, dur = meta["duration_s"], meta["duration_s"]
    segs = obj.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return ["segments 为空或非列表"]
    prev_end = 0.0
    for i, sg in enumerate(segs):
        s, e = sg.get("start"), sg.get("end")
        if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
            errors.append(f"segments[{i}] start/end 非数")
            continue
        if s < -0.05 or e > dur + 0.5 or e <= s:
            errors.append(f"segments[{i}] 区间非法 [{s},{e}]")
        if s < prev_end - 1e-6:
            errors.append(f"segments[{i}] 与前段重叠")
        prev_end = e
        to = sg.get("transition_out")
        if isinstance(to, dict) and to.get("type") not in OP_TYPES:
            errors.append(f"segments[{i}].transition_out.type 越界")
        src = sg.get("source")
        if isinstance(src, dict):
            if src.get("camera_motion") not in (None,) + CAMERA_MOTION:
                src["camera_motion"] = "uncertain"           # 软归一
            if src.get("shot_scale") not in (None,) + SHOT_SCALE:
                src["shot_scale"] = "uncertain"
    if abs(prev_end - d) > 1.0:
        errors.append(f"segments 未覆盖到结尾（末段 end={prev_end:g} vs {d:g}）")

    anchors = meta["known_ts"]
    beats = meta["beats"]
    seen_t: list[float] = []
    confirmed_fill = [w for w in meta["window_confirms"]
                      if w["op_type"] in ("tracked_mask_fill", "subject_cutout_composite")]
    for i, op in enumerate(obj.get("operations") or []):
        if not isinstance(op, dict):
            errors.append(f"operations[{i}] 非对象")
            continue
        t, ty = op.get("t_s"), op.get("type")
        if ty not in OP_TYPES:
            errors.append(f"operations[{i}].type={ty!r} 不在枚举")
        if not isinstance(t, (int, float)) or not (0 <= t <= d + 0.5):
            errors.append(f"operations[{i}].t_s={t} 非法")
            continue
        if anchors and not any(abs(t - a) <= tol_s for a in anchors):
            errors.append(f"operations[{i}].t_s={t} 未落在证据时刻 ±{tol_s}s（禁止自造）")
        if any(abs(t - u) < 0.06 for u in seen_t):
            errors.append(f"operations[{i}].t_s={t} 与已有操作重复")
        seen_t.append(t)
        conf = op.get("confidence", 0.5)
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            errors.append(f"operations[{i}].confidence={conf} 越界")
        ca = op.get("change_anchor")
        if ca is not None:
            if not isinstance(ca, (int, float)) or (
                    beats and not any(abs(ca - b) <= anchor_tol_s for b in beats)):
                errors.append(f"operations[{i}].change_anchor={ca} 未命中节拍 ±{anchor_tol_s}s")
        if op.get("texture_sequence") is not None and not any(
                abs(t - w["t_s"]) <= 0.5 for w in confirmed_fill):
            errors.append(f"operations[{i}].texture_sequence 无窗口确认（禁止自造）")
    return errors


def _prune_invalid_ops(obj, meta: dict) -> dict | None:
    """op 级剪枝：逐个 op 单独校验，剔除带错的（编造时刻/越界枚举/无确认 texture）。
    剩余 ≥ 一半且整体校验干净才接受——否则宁可回退规则底稿（服务器首跑教训：
    GLM 版 63-op 底稿全是光流噪声，1 个残留错误整包回退不公平也不可用）。"""
    ops = [o for o in (obj.get("operations") or []) if isinstance(o, dict)]
    kept = []
    for o in ops:
        single = dict(obj)
        single["operations"] = [o]
        if not validate_recipe(single, meta):
            kept.append(o)
    if not ops or len(kept) < max(1, len(ops) // 2):
        return None
    pruned = dict(obj)
    pruned["operations"] = kept
    if not validate_recipe(pruned, meta):
        # 残余段级问题（覆盖/重叠）→ 段结构用确定性边界重建（ops 才是语义主体）
        draft = build_rule_draft(meta, (obj.get("video") or {}).get("aweme_id", ""))
        pruned["segments"] = draft["segments"]
    return pruned if not validate_recipe(pruned, meta) else None


def recipe_penalty(obj, meta: dict) -> float:
    """回退比较：未锚定操作数×10 + 覆盖缺口秒。"""
    if not isinstance(obj, dict):
        return 1e9
    anchors = meta["known_ts"]
    bad = sum(1 for op in (obj.get("operations") or [])
              if isinstance(op, dict) and isinstance(op.get("t_s"), (int, float))
              and anchors and not any(abs(op["t_s"] - a) <= 0.25 for a in anchors))
    segs = obj.get("segments") or []
    covered = sum(max(0.0, (s.get("end", 0) - s.get("start", 0)))
                  for s in segs if isinstance(s, dict))
    return bad * 10.0 + max(0.0, meta["duration_s"] - covered)


def run_recipe(cfg: AppConfig, vid: str, *, force: bool = False,
               runner=None, dry_run: bool = False) -> Path | None:
    r_cfg = cfg.library.get("recipe") or {}
    tdir = cfg.paths.library_dir / "editing" / vid / "recipe"
    params = {"prompt_version": r_cfg.get("prompt_version", "recipe_v1"),
              "evidence_tolerance_s": r_cfg.get("evidence_tolerance_s", 0.25)}
    if not force and common.done_or_skip(tdir, params, force=False) is not None:
        logger.info("[recipe %s] 已有产物，跳过", vid)
        return tdir / "result.json"

    meta = load_recipe_inputs(cfg, vid)
    if not meta["duration_s"]:
        raise FileNotFoundError(f"缺 inspect 产物：{vid}")
    if not meta["candidates"]:
        raise FileNotFoundError(f"缺 signals 产物（先跑 D1）：{vid}")

    rule_draft = build_rule_draft(meta, vid)
    ctx = build_context(meta, max_chars=int(r_cfg.get("max_context_chars", 8000)))
    if dry_run:
        logger.info("[recipe %s] dry-run：ctx %d 字符，证据时刻 %d，候选 %d，"
                    "窗口确认 %d，规则底稿 %d ops", vid, len(ctx), len(meta["known_ts"]),
                    len(meta["candidates"]), len(meta["window_confirms"]),
                    len(rule_draft["operations"]))
        return None

    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})

    prompt = RECIPE_PROMPT.format(ops="/".join(OP_TYPES), skeleton=RECIPE_SKELETON) + ctx
    max_new = int(r_cfg.get("max_new_tokens", 4096))
    max_retries = int(r_cfg.get("max_retries", 1))
    result, mode, attempts = None, "fail", []
    for attempt in range(1 + max_retries):
        p = prompt if attempt == 0 else (
            "上次输出未过校验：" + "; ".join(attempts[-1]["errors"][:8])
            + "。修正后只返回完整 JSON。\n\n" + prompt)
        ans = runner.ask(p, max_new_tokens=max_new)
        raw = dedupe_json_repetition(ans.text, first_key='"video"')
        if attempt == 0 or not result:
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"raw_answer_{attempt}.txt").write_text(ans.text, encoding="utf-8")
        block = extract_json_block(raw)
        obj = None
        try:
            obj = json.loads(block) if block else None
        except ValueError:
            pass
        errs = validate_recipe(obj, meta) if isinstance(obj, dict) else ["JSON 解析失败"]
        attempts.append({"attempt": attempt, "errors": errs[:8], "elapsed_s": ans.elapsed_s})
        logger.info("[recipe %s] 第 %d 次：%d 错误", vid, attempt + 1, len(errs))
        if not errs:
            result, mode = obj, "json"
            break
        if obj is not None and result is None:
            result, mode = obj, "partial"     # 保底存档（供剪枝/回退裁决）
    if mode == "partial":
        pruned = _prune_invalid_ops(result, meta)
        if pruned is not None:
            result, mode = pruned, "pruned"   # op 级剪枝接受（保留过半）
            logger.info("[recipe %s] 剪枝接受：%d ops", vid, len(pruned["operations"]))
    if result is None or (mode not in ("json", "pruned")
                          and recipe_penalty(result, meta)
                          >= recipe_penalty(rule_draft, meta)):
        if result is not None:
            logger.warning("[recipe %s] LLM 输出 penalty ≥ 规则底稿 → 回退底稿", vid)
        result, mode = rule_draft, "rule_fallback"

    tdir.mkdir(parents=True, exist_ok=True)   # write_result_json 不建父目录（前车之鉴）
    if mode == "rule_fallback" and result is not rule_draft:
        (tdir / "raw_partial.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    path = common.write_result_json(tdir, tool="recipe", aweme_id=vid, params=params,
                                    output={"mode": mode, "attempts": attempts,
                                            "n_operations": len(result["operations"]),
                                            "recipe": result})
    logger.info("[recipe %s] mode=%s %d ops → %s", vid, mode,
                len(result["operations"]), path)
    common.emit_status_line("ok", vid=vid, mode=mode, ops=len(result["operations"]))
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="D3 Editing Recipe JSON 合成")
    ap.add_argument("--id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="打印上下文统计，不调模型")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_recipe")
    try:
        run_recipe(cfg, args.id, force=args.force, dry_run=args.dry_run)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("recipe 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
