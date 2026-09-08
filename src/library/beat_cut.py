"""beat_cut：用「热门卡点模板」的手法策略 + 模板节拍点，剪素材库出卡点片。

策略来源（analyze_editing 的六条共性配方，2026-09-08 实测）：
  128BPM 电子舞曲 / 每拍切（audio.beat_sync=半拍切则细分）/ 白帧 0.08s 叠加切点 /
  切点取模板自身 beat_points / 音频直接用模板原声（卡点灵魂是音乐）

流程：模板节拍 → 切点序列 → 每切点用轮换需求句检索素材镜头（去重）→
  逐段截 0.5s 级片段（fps24/544×960 统一）→ concat（重编码，帧率混杂教训）→
  切点处白帧（drawbox enable between）→ 模板原声 → final.mp4 + 审计

CLI：python -m src.library.beat_cut --template-id <tid> [--style 高能] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.library.build_index import E5Embedder, load_index
from src.perception import common

logger = logging.getLogger(__name__)

# 轮换需求句：卡点混剪无叙事线，按多样性轮换取材（可被 --style 前缀加权）
QUERY_POOL = [
    "主体：蜘蛛侠战衣；动作：高速摆荡飞行；场景：城市楼宇间",
    "主体：彼得帕克；动作：摘下面具；情绪：震惊",
    "主体：蜘蛛侠；动作：近身格斗出拳；情绪：紧张激烈",
    "主体：蜘蛛侠战衣；动作：发射蛛丝；场景：夜晚",
    "主体：反派；动作：逼近镜头；情绪：压迫感",
    "主体：蜘蛛侠；动作：高空坠落俯冲；场景：天空",
    "主体：彼得帕克与同伴；动作：对话；场景：室内",
    "主体：蜘蛛侠；动作：落地蓄力起身；情绪：爆发",
    "主体：城市爆炸火光；动作：冲击波扩散",
    "主体：蜘蛛侠特写；动作：眼神变化；情绪：决心",
]


def build_cut_plan(beats: list[float], duration_s: float, *, half_beat: bool = False,
                   min_len_s: float = 0.3, max_cuts: int = 64) -> list[tuple[float, float]]:
    """节拍 → 切段区间列表 [(start, end)]，覆盖到视频结尾。

    half_beat=True 时在相邻拍中点细分（audio.beat_sync=半拍切 的策略落地）。
    """
    pts = sorted(b for b in beats if 0 <= b <= duration_s + 0.5)
    if half_beat and len(pts) >= 2:
        subs = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            subs.extend([(a + b) / 2, b])
        pts = subs
    bounds = [0.0] + [p for p in pts if p > min_len_s] + [duration_s]
    bounds = sorted(dict.fromkeys(round(b, 3) for b in bounds))
    plan = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= min_len_s]
    if len(plan) > max_cuts:                     # 超量均匀抽稀（保持节奏感）
        step = len(plan) / max_cuts
        plan = [plan[int(i * step)] for i in range(max_cuts)]
    return plan


def pick_shots(rows, emb, text_embs, durs, *, used: set, min_len_s: float,
               top_scan: int = 40) -> list[int | None]:
    """每段选一个未用过的镜头（时长够），无合适则 None（黑场）。纯函数可测。"""
    import numpy as np

    picked: list[int | None] = []
    for q, dur in zip(text_embs, durs):
        cos = (emb @ q.reshape(-1)).ravel()
        choice = None
        for ri in np.argsort(-cos)[:top_scan]:
            if int(ri) in used:
                continue
            if rows[ri]["duration_s"] >= min(dur, min_len_s):
                choice = int(ri)
                break
        if choice is None:                        # 全用过/太短 → 放宽时长再试
            for ri in np.argsort(-cos)[:top_scan]:
                if int(ri) not in used:
                    choice = int(ri)
                    break
        if choice is not None:
            used.add(choice)
        picked.append(choice)
    return picked


def run_beat_cut(cfg: AppConfig, template_id: str, *, force: bool = False,
                 flash_s: float = 0.08) -> Path:
    out_dir = cfg.paths.library_dir / "stories" / f"{template_id}_beatcut"
    final = out_dir / "final.mp4"
    if final.exists() and not force:
        logger.info("[beatcut %s] 已有产物，跳过", template_id)
        return final

    tdir = cfg.paths.perception_dir / template_id
    insp = common.read_result_json(tdir / "inspect") or {}
    duration = float(insp.get("duration_s") or 0)
    beats_env = common.read_result_json(tdir / "beats") or {}
    beats = list(beats_env.get("beat_points_s") or [])
    if not duration or not beats:
        raise FileNotFoundError(f"缺 inspect/beats 产物（先跑感知基础件）：{template_id}")

    analysis = None
    a_path = cfg.paths.library_dir / "editing" / template_id / "analysis" / "result.json"
    if a_path.exists():
        analysis = json.loads(a_path.read_text(encoding="utf-8"))["output"]["analysis"]
    half = bool(analysis and "半拍" in str((analysis.get("audio") or {}).get("beat_sync", "")))

    plan = build_cut_plan(beats, duration, half_beat=half)
    durs = [b - a for a, b in plan]
    logger.info("[beatcut %s] %.1fs，%d 刀（half_beat=%s）", template_id, duration,
                len(plan), half)

    rows, emb = load_index(cfg)
    queries = [QUERY_POOL[i % len(QUERY_POOL)] for i in range(len(plan))]
    embedder = E5Embedder(cfg.library.get("embed") or {})
    text_embs = embedder.embed(queries)
    picked = pick_shots(rows, emb, text_embs, durs, used=set(),
                        min_len_s=0.45)

    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    a_cfg = cfg.library.get("assemble") or {}
    crf = int(a_cfg.get("crf", 20))
    seg_files, audit = [], []
    for i, ((start, end), ri) in enumerate(zip(plan, picked)):
        out = work / f"c{i:03d}.mp4"
        take = end - start
        if ri is None:
            common.run_ffmpeg("ffmpeg", ["-y", "-loglevel", "error", "-f", "lavfi",
                                         "-i", f"color=black:s=544x960:d={take:g}:r=24",
                                         "-c:v", "libx264", "-preset", "veryfast",
                                         "-pix_fmt", "yuv420p", str(out)])
            audit.append({"cut": i, "start": start, "end": end, "placeholder": True})
        else:
            row = rows[ri]
            common.run_ffmpeg("ffmpeg", [
                "-y", "-loglevel", "error", "-ss", f"{row['start_s']:g}",
                "-i", row["video"], "-t", f"{take:g}",
                "-vf", "scale=544:960:force_original_aspect_ratio=increase,"
                       "crop=544:960,fps=24",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", str(out)])
            audit.append({"cut": i, "start": start, "end": end,
                          "shot": {"video": row["video_stem"], "shot_idx": row["shot_idx"],
                                   "caption": row.get("caption", "")}})
        seg_files.append(out)

    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{f.as_posix()}'\n" for f in seg_files),
                           encoding="utf-8")
    concat_out = work / "concat.mp4"
    common.run_ffmpeg("ffmpeg", ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                                 "-i", str(concat_list), "-an", "-c:v", "libx264",
                                 "-preset", "veryfast", "-crf", str(crf),
                                 "-pix_fmt", "yuv420p", str(concat_out)])

    # 白帧：每个切点起点 flash_s 秒（drawbox enable between —— 单遍滤镜）
    cuts_t = [s for s, _ in plan if s > 0.05]
    vf_parts = [f"drawbox=x=0:y=0:w=iw:h=ih:color=white@1.0:t=fill:"
                f"enable='between(t,{s:g},{s + flash_s:g})'" for s in cuts_t]
    src_video = cfg.paths.videos_dir / template_id / "video.mp4"
    voiced = work / "voiced.mp4"
    common.run_ffmpeg("ffmpeg", [
        "-y", "-loglevel", "error", "-i", str(concat_out), "-i", str(src_video),
        "-vf", ",".join(vf_parts) if vf_parts else "null",
        "-map", "0:v", "-map", "1:a?", "-af", "apad",
        "-t", f"{duration + 0.5:g}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", str(int(a_cfg.get("audio_rate", 44100))), str(voiced)])

    dur = common.video_duration_s(cfg.perception.get("ffprobe_bin", "ffprobe"), voiced)
    final.write_bytes(voiced.read_bytes())
    (out_dir / "beatcut.json").write_text(json.dumps({
        "template_id": template_id, "final": str(final), "duration_s": dur,
        "n_cuts": len(plan), "half_beat": half, "flash_s": flash_s,
        "placeholder_cuts": sum(1 for a in audit if a.get("placeholder")),
        "cuts": audit}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[beatcut %s] final %.2fs，%d 刀（%d 占位）→ %s",
                template_id, dur or 0, len(plan),
                sum(1 for a in audit if a.get("placeholder")), final)
    common.emit_status_line("ok", output=str(final), cuts=len(plan), duration_s=dur)
    return final


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="节拍锁定素材库卡点剪辑")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_beatcut")
    try:
        run_beat_cut(cfg, args.template_id, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("beat_cut 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
