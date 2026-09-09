"""decompose：D2 窗口级 Omni 问答——对 signals 候选逐窗确认编辑操作。

Editing Branch V1（2026-09-09 计划）：信号轨只给"哪里有变化+物理签名"，
本模块让 Qwen3-Omni 看窗口片段（含音频）判定"是什么剪辑操作"。

窗口规划：候选 ±window_pad_s 合并（间隔 < merge_gap_s），上限 max_windows
（置信度排序 + 类型覆盖配额：每种假设类型保 ≥1 窗），另加 n_control_windows
个控制窗（信号谷底、远离候选——测信号轨漏检率，基准可信度关键）。

每窗独立信封 → 断点续跑；--only 0,1,2 试点（pilot 门：先给用户看答案质量再放全量）。
fps 8：1.2s 窗 ~10 帧；感知默认 2fps 只有 2-3 帧，分辨不了 0.2s 级编辑事件。

CLI：python -m src.library.decompose --id <vid> [--only 0,1,2] [--force]
     [--no-controls] [--gpus 0,1]
产物：data/library/editing/<id>/windows/<idx>/result.json + windows/result.json 汇总
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

# 与 D3 Recipe 同一枚举（窗口答案的 op_type 直接收敛于此）
OP_TYPES = ("hard_cut", "match_cut", "crossfade", "mask_wipe",
            "foreground_occlusion_transition", "luma_flash", "speed_ramp",
            "tracked_mask_fill", "subject_cutout_composite", "text_layer_animation",
            "montage_burst", "zoom_punch", "whip_pan", "beat_freeze", "uncertain")

WHAT_CHANGED = ("global", "subject_interior", "background", "local_region", "text")

WINDOW_PROMPT = """你是剪辑手法鉴定专家。下面是原视频的一个短窗口片段（含音频），
以及确定性信号检测轨在这个窗口捕获的物理特征。判断窗口内发生了什么剪辑操作。

【信号特征】{signature}
【原视频时间】窗口 {start:g}s ~ {end:g}s（全长 {duration:g}s；event_time_original_s
必须落在此窗口内，按原视频时间报）
【时间换算】你看到的片段是从原视频 {start:g}s 截取的——片段内第 n 秒 ≈ 原视频
{start:g}+n 秒，报时刻请先换算。

【只输出一个 JSON 对象，无围栏无解释】
{{"operations": [
  {{"what_changed": "global",            # global/subject_interior/background/local_region/text
    "op_type": "hard_cut",                # 只能取：{ops}
    "subject": "画面主体一句话",
    "flash": false,
    "motion": "none",                     # none/zoom_in/zoom_out/pan/tilt/follow/uncertain
    "direction": null,
    "evidence_quote": "具体可见证据，不能只复述操作名",
    "event_time_original_s": 0.0,
    "interval_original_s": [0.0, 0.0],
    "confidence": 0.0}}
 ],
 "window_summary": "这一窗口的图层关系和变化顺序"}}

同一时刻若同时发生切镜、文字动画、遮罩或主体内部变化，必须分别列为多个 operations；
不要把图层操作吞并成 hard_cut。看不准的操作取 op_type="uncertain" 且 confidence<0.5。"""


def plan_windows(candidates: list[dict], duration_s: float, *, series: dict | None = None,
                 window_pad_s: float = 0.6, merge_gap_s: float = 0.5, max_windows: int = 48,
                 n_control_windows: int = 3, max_window_s: float = 2.5) -> list[dict]:
    """信号候选 → 窗口列表（合并/时长上限/限额/类型覆盖配额/控制窗）。纯函数。

    max_window_s：密集候选链式合并会滚出巨窗（试点实测 0~8s/11~24s），
    单窗单答丢逐事件粒度 → 超上限即关窗另开。
    """
    wins: list[dict] = []
    for c in sorted(candidates, key=lambda c: c["t_s"]):
        s, e = c["t_s"] - window_pad_s, c["t_s"] + window_pad_s
        if wins and s - wins[-1]["end"] < merge_gap_s \
                and e - wins[-1]["start"] <= max_window_s:
            w = wins[-1]
            w["end"] = max(w["end"], e)
            for h in c["type_hypotheses"]:
                if h not in w["hypotheses"]:
                    w["hypotheses"].append(h)
            w["confidence"] = max(w["confidence"], c["confidence"])
            w["signature"].update(c.get("signature") or {})
            w["n_candidates"] += 1
        else:
            wins.append({"start": s, "end": e, "hypotheses": list(c["type_hypotheses"]),
                         "confidence": c["confidence"],
                         "signature": dict(c.get("signature") or {}), "n_candidates": 1})
    for w in wins:
        w["start"], w["end"] = max(0.0, round(w["start"], 2)), min(duration_s, round(w["end"], 2))
        w["control"] = False

    # 限额：贪心保类型覆盖（优先挑"带来最多未覆盖类型"的窗，平分按置信度——
    # 多类型合并窗也算数），再按分数补满
    if len(wins) > max_windows:
        by_conf = sorted(wins, key=lambda w: -w["confidence"])
        all_types = {h for w in wins for h in w["hypotheses"]}
        remaining, covered, keep = list(by_conf), set(), []
        while remaining and len(covered) < len(all_types) and len(keep) < max_windows:
            best = max(remaining, key=lambda w: (len(set(w["hypotheses"]) - covered),
                                                 w["confidence"]))
            keep.append(best)
            remaining.remove(best)
            covered.update(best["hypotheses"])
        for w in by_conf:
            if len(keep) >= max_windows:
                break
            if w not in keep:
                keep.append(w)
        wins = sorted(keep, key=lambda w: w["start"])

    # 控制窗：信号谷底（diff+flow 双低）、远离一切候选窗 ≥1s
    if series and n_control_windows > 0:
        import numpy as np

        act = np.asarray(series["diff_global"], float) + np.asarray(series["flow_mag"], float)
        t = np.asarray(series["t"], float)[1:]
        order = np.argsort(act)
        taken = 0
        for i in order:
            if taken >= n_control_windows:
                break
            tt = float(t[i])
            if any(w["start"] - 1.0 <= tt <= w["end"] + 1.0 for w in wins):
                continue
            if wins and tt - wins[-1]["end"] < 0 and any(
                    w["control"] and abs(tt - (w["start"] + w["end"]) / 2) < 2.0 for w in wins):
                continue
            wins.append({"start": max(0.0, round(tt - window_pad_s, 2)),
                         "end": min(duration_s, round(tt + window_pad_s, 2)),
                         "hypotheses": [], "confidence": 0.0, "signature": {},
                         "n_candidates": 0, "control": True})
            taken += 1
        wins.sort(key=lambda w: w["start"])
    for i, w in enumerate(wins):
        w["idx"] = i
    return wins


def build_window_prompt(window: dict, duration_s: float) -> str:
    sig = "; ".join(f"{k}={v}" for k, v in (window["signature"] or {}).items()) or "无"
    hy = "/".join(window["hypotheses"]) if window["hypotheses"] else \
        "控制窗（信号轨未报事件——验证是否漏检）"
    return WINDOW_PROMPT.format(
        signature=f"候选类型 [{hy}]，指标 {sig}",
        start=window["start"], end=window["end"], duration=duration_s,
        ops="/".join(OP_TYPES))


def parse_window_answer(raw: str, window: dict) -> dict | None:
    """窗口答案 → 归一化 JSON；兼容 v1 单操作与 v2 多操作输出。"""
    block = extract_json_block(dedupe_json_repetition(raw, first_key='"operations"'))
    if not block:
        block = extract_json_block(dedupe_json_repetition(raw, first_key='"what_changed"'))
    if not block:
        return None
    try:
        obj = json.loads(block)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None

    legacy = "op_type" in obj
    operations = [obj] if legacy else obj.get("operations")
    if not isinstance(operations, list):
        return None
    normalized = []
    for item in operations[:8]:
        if not isinstance(item, dict) or "op_type" not in item:
            continue
        item = dict(item)
        if item.get("op_type") not in OP_TYPES:
            item["op_type"] = "uncertain"          # 归一越界，不拒收（软任务）
        if item.get("what_changed") not in WHAT_CHANGED:
            item["what_changed"] = "global"
        t = item.get("event_time_original_s")
        if not isinstance(t, (int, float)) \
                or not (window["start"] - 0.3 <= t <= window["end"] + 0.3):
            item["event_time_original_s"] = None   # 越窗自报时刻 → 弃用（防编造）
        interval = item.get("interval_original_s")
        if not (isinstance(interval, list) and len(interval) == 2
                and all(isinstance(v, (int, float)) for v in interval)
                and window["start"] - 0.3 <= interval[0] <= interval[1]
                <= window["end"] + 0.3):
            item.pop("interval_original_s", None)
        confidence = item.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            item["confidence"] = 0.0
        else:
            item["confidence"] = min(1.0, max(0.0, float(confidence)))
        normalized.append(item)
    if not normalized:
        return None
    if legacy:
        return normalized[0]
    return {"operations": normalized,
            "window_summary": str(obj.get("window_summary") or "")[:200]}


def run_decompose(cfg: AppConfig, vid: str, *, only: list[int] | None = None,
                  force: bool = False, no_controls: bool = False,
                  runner=None) -> Path | None:
    d_cfg = cfg.library.get("windows") or {}
    root = cfg.paths.library_dir / "editing" / vid
    sig_env = common.read_result_json(root / "signals")
    if not sig_env:
        raise FileNotFoundError(f"缺 signals 产物（先跑 D1）：{vid}")
    sig_out = sig_env["output"]
    insp_env = common.read_result_json(cfg.paths.perception_dir / vid / "inspect")
    duration = float(((insp_env or {}).get("output") or {}).get("duration_s") or 0)
    if not duration:
        raise FileNotFoundError(f"缺 inspect 产物：{vid}")

    windows = plan_windows(
        sig_out["candidates"], duration, series=sig_out.get("series"),
        window_pad_s=float(d_cfg.get("window_pad_s", 0.6)),
        merge_gap_s=float(d_cfg.get("merge_gap_s", 0.5)),
        max_windows=int(d_cfg.get("max_windows", 48)),
        n_control_windows=0 if no_controls else int(d_cfg.get("n_control_windows", 3)),
        max_window_s=float(d_cfg.get("max_window_s", 2.5)))
    todo = [w for w in windows if only is None or w["idx"] in only]
    logger.info("[decompose %s] %d 窗（试点 %s）", vid, len(windows),
                only if only is not None else "全量")

    if runner is None:
        from src.perception.omni_runner import OmniRunner
        omni_cfg = {**cfg.perception.get("omni", {}),
                    "fps": float(d_cfg.get("fps", 8.0))}   # 窗口级 fps 覆盖
        runner = OmniRunner(omni_cfg)

    video = cfg.paths.videos_dir / vid / "video.mp4"
    max_new = int(d_cfg.get("max_new_tokens", 1024))
    params_base = {"prompt_version": d_cfg.get("prompt_version", "window_v1")}
    results = []
    for w in todo:
        wdir = root / "windows" / f"{w['idx']:03d}"
        params = {**params_base, "start": w["start"], "end": w["end"]}
        if not force and common.done_or_skip(wdir, params, force=False) is not None:
            env = common.read_result_json(wdir)
            results.append({"idx": w["idx"], "status": "skip",
                            "answer": env["output"]["answer"]})
            logger.info("[decompose %s] 窗 %d 已有产物，跳过", vid, w["idx"])
            continue
        logger.info("[decompose %s] 窗 %d/%d（%g~%gs %s）", vid, w["idx"], len(windows),
                    w["start"], w["end"], "控制" if w["control"] else "候选")
        wdir.mkdir(parents=True, exist_ok=True)   # cut_clip 要写 clip.mp4（先建目录）
        ans = runner.watch(video, build_window_prompt(w, duration),
                           start_s=w["start"], end_s=w["end"], clip_dir=wdir,
                           max_new_tokens=max_new, duration_s=w["end"] - w["start"])
        answer = parse_window_answer(ans.text, w)
        wdir.mkdir(parents=True, exist_ok=True)   # write_result_json 不建父目录（前车之鉴）
        common.write_result_json(wdir, tool="decompose", aweme_id=vid, params=params,
                                 output={"window": {k: w[k] for k in
                                         ("idx", "start", "end", "hypotheses", "control")},
                                         "answer": answer, "parse": "json" if answer else "raw",
                                         "raw_head": None if answer else ans.text[:300],
                                         "elapsed_s": ans.elapsed_s,
                                         "output_tokens": ans.output_tokens})
        results.append({"idx": w["idx"], "status": "ok",
                        "answer": answer, "parse": "json" if answer else "raw"})

    n_json = sum(1 for r in results if r.get("parse") == "json")
    agg = root / "windows" / "result.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps({
        "aweme_id": vid, "n_windows": len(windows), "n_run": len(results),
        "n_json": n_json, "windows_plan": [
            {k: w.get(k) for k in ("idx", "start", "end", "hypotheses", "control")}
            for w in windows],
        "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[decompose %s] %d/%d 窗 JSON 有效 → %s", vid, n_json, len(results), agg)
    common.emit_status_line("ok", vid=vid, windows=len(results), json_ok=n_json)
    return agg


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="D2 窗口级 Omni 编辑操作问答")
    ap.add_argument("--id", required=True)
    ap.add_argument("--only", default=None, help="试点窗口序号，如 0,1,2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.gpus:
        from src.perception.omni_runner import set_visible_gpus
        set_visible_gpus(args.gpus)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_decompose")
    try:
        only = [int(x) for x in args.only.split(",")] if args.only else None
        run_decompose(cfg, args.id, only=only, force=args.force,
                      no_controls=args.no_controls)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("decompose 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
