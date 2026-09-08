"""analyze_editing：让 Omni 看「热门卡点/剪辑模板」视频，反推剪辑手法与音频处理。

输出（data/editing/<id>/analysis/result.json）一份可执行的剪辑知识 JSON：
  style / bpm_feel / cut_density / on_beat_est
  timeline[]：每个手法事件 {t_s, technique, note}（时刻锚定到节拍/镜头边界证据）
  audio：BGM 类型 / 踩点方式 / 音效 / 人声处理 / 变速
  ffmpeg_hints：FFmpeg 直接可复刻的实现要点
  not_reproducible：当前工具链做不了的（如蒙版抠像 → SAM2）

证据打底（防编造）：节拍点（librosa）+ 镜头边界（scene）+ OCR 字幕事件先算好，
Omni 只做"看画面+听音频 → 手法判读"，timeline t_s 软校验须落在证据 ±0.3s。

CLI：python -m src.library.analyze_editing --id <vid> [--force]
前置：data/videos/<id>/video.mp4 + 感知产物（inspect/shots/beats/ocr，可部分缺失）
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

EDITING_PROMPT_HEADER = """你是专业剪辑师。看这条热门短视频（含音频），结合给出的
「节拍点/镜头边界/字幕事件」证据，反推它是怎么剪的。目的：用这套手法去剪其它素材。

【硬性规则】
1. 只输出一个 JSON 对象，无围栏无解释。
2. timeline[] 最多 20 个最具代表性的手法事件（普通 hard_cut_on_beat 只挑标志性几处，
   优先收录 zoom_punch/flash/speed_ramp/mask_wipe 等特殊手法），每个 t_s 必须取自
   证据时刻（节拍点或镜头边界，±0.3s 内），technique 只能取：hard_cut_on_beat /
   zoom_punch / flash / speed_ramp / match_cut / whip_pan / text_pop / freeze / shake /
   mask_wipe / beat_drop_pause / other。
3. audio 讲清：BGM 类型与强度感、踩点方式（每拍切/半拍切/重拍切）、音效（whoosh/riser/砰）、
   人声处理、变速位置。
4. ffmpeg_hints 写 FFmpeg 命令级要点（如"白帧 0.08s 叠加在切点上""切点取 beat_points"）。
5. 看不准的标 uncertain，不要编造。

【输出骨架】
{"style": "卡点类型一句话", "bpm_feel": 0, "cut_density": 0.0, "on_beat_est": 0.0,
 "timeline": [{"t_s": 0.0, "technique": "hard_cut_on_beat", "note": "切到鼓点"}],
 "audio": {"bgm": "", "beat_sync": "", "sfx": [], "voice": "", "speed_change": ""},
 "ffmpeg_hints": [], "not_reproducible": []}

【证据】
"""


def build_evidence(cfg: AppConfig, vid: str) -> tuple[str, dict]:
    pdir = cfg.paths.perception_dir / vid
    parts: list[str] = []

    def _out(tool: str) -> dict | None:
        env = common.read_result_json(pdir / tool)
        return env["output"] if env else None

    insp = _out("inspect")
    dur = float((insp or {}).get("duration_s") or 0)
    if dur:
        parts.append(f"[时长] {dur:.1f}s")
    beats = _out("beats")
    beat_pts = list((beats or {}).get("beat_points_s") or [])
    if beat_pts:
        pts = ", ".join(f"{p:g}" for p in beat_pts[:80])
        parts.append(f"[节拍点·timeline t_s 优先取这些] {pts}"
                     f"{' …' if len(beat_pts) > 80 else ''}")
    else:
        parts.append("[节拍点] 未检测")
    shots = _out("shots")
    bounds = list((shots or {}).get("boundaries_s") or [])
    if bounds:
        parts.append(f"[镜头边界] {', '.join(f'{b:g}' for b in bounds[:60])}"
                     f"{' …' if len(bounds) > 60 else ''}")
    ocr = _out("ocr")
    events = (ocr or {}).get("text_events") or []
    if events:
        parts.append("[字幕事件] " + "; ".join(
            f"{e.get('t_start_s', 0):g}s「{e.get('text', '')[:14]}」" for e in events[:12]))
    return "\n".join(parts), {"duration_s": dur, "beats": beat_pts, "bounds": bounds}


TECHNIQUES = ("hard_cut_on_beat", "zoom_punch", "flash", "speed_ramp", "match_cut",
              "whip_pan", "text_pop", "freeze", "shake", "mask_wipe",
              "beat_drop_pause", "other")


def validate_analysis(obj, evidence: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层不是 JSON 对象"]
    for k in ("style", "timeline", "audio", "ffmpeg_hints"):
        if k not in obj:
            errors.append(f"缺键 {k}")
    if errors:
        return errors
    anchors = sorted(set(evidence.get("beats") or []) | set(evidence.get("bounds") or []))
    dur = evidence.get("duration_s") or 1e9
    for i, ev in enumerate(obj.get("timeline") or []):
        if not isinstance(ev, dict):
            errors.append(f"timeline[{i}] 非对象")
            continue
        t = ev.get("t_s")
        if not isinstance(t, (int, float)) or not (0 <= t <= dur + 0.5):
            errors.append(f"timeline[{i}].t_s={t} 非法")
        if ev.get("technique") not in TECHNIQUES:
            errors.append(f"timeline[{i}].technique={ev.get('technique')!r} 不在枚举")
        elif anchors and isinstance(t, (int, float)) \
                and not any(abs(t - a) <= 0.35 for a in anchors):
            errors.append(f"timeline[{i}].t_s={t} 未落在任何节拍/镜头边界 ±0.35s（禁止自造）")
    return errors


def run_analysis(cfg: AppConfig, vid: str, *, force: bool = False, runner=None) -> Path | None:
    a_cfg = cfg.library.get("editing_analysis") or {}
    tdir = cfg.paths.library_dir / "editing" / vid / "analysis"
    params = {"prompt_version": a_cfg.get("prompt_version", "editing_v1")}
    if not force and common.done_or_skip(tdir, params, force=False) is not None:
        logger.info("[edit %s] 已有产物，跳过", vid)
        return tdir / "result.json"

    video = cfg.paths.videos_dir / vid / "video.mp4"
    if not video.exists():
        logger.error("[edit %s] 视频不存在: %s", vid, video)
        return None
    evidence, ev_meta = build_evidence(cfg, vid)
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})

    prompt = EDITING_PROMPT_HEADER + evidence
    max_new = int(a_cfg.get("max_new_tokens", 3072))
    max_retries = int(a_cfg.get("max_retries", 1))
    result, mode, attempts = None, "fail", []
    for attempt in range(1 + max_retries):
        p = prompt if attempt == 0 else (
            "上次输出未过校验：" + "; ".join(attempts[-1]["errors"][:8])
            + "。修正后只返回完整 JSON。\n\n" + prompt)
        ans = runner.watch(video, p, max_new_tokens=max_new)
        raw = dedupe_json_repetition(ans.text, first_key='"style"')
        block = extract_json_block(raw)
        obj = None
        try:
            obj = json.loads(block) if block else None
        except ValueError:
            pass
        errs = validate_analysis(obj, ev_meta) if isinstance(obj, dict) else ["JSON 解析失败"]
        attempts.append({"attempt": attempt, "errors": errs[:8], "elapsed_s": ans.elapsed_s})
        logger.info("[edit %s] 第 %d 次：%d 错误", vid, attempt + 1, len(errs))
        if not errs:
            result, mode = obj, "json"
            break
        if obj is not None and result is None:
            result, mode = obj, "partial"      # 保底存档（时间戳告警不拦）
    if result is None:
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "raw_answer.txt").write_text(str(attempts), encoding="utf-8")
        logger.error("[edit %s] 分析失败，raw 已存（重跑即重试）", vid)
        return None

    tdir.mkdir(parents=True, exist_ok=True)   # write_result_json 不建父目录（实测4连崩）
    path = common.write_result_json(tdir, tool="analyze_editing", aweme_id=vid,
                                    params=params, output={
                                        "mode": mode, "attempts": len(attempts),
                                        "analysis": result})
    logger.info("[edit %s] mode=%s %d 手法事件 → %s",
                vid, mode, len(result.get("timeline") or []), path)
    common.emit_status_line("ok", vid=vid, events=len(result.get("timeline") or []))
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Omni 反推剪辑手法")
    ap.add_argument("--id", required=True, help="data/videos/<id>")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_editing")
    try:
        run_analysis(cfg, args.id, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_editing 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
