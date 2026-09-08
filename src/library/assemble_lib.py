"""assemble_lib：C3b——把选中的素材镜头按模板时间线装配成片。

策略：
  每段取 picked 镜头：镜头够长则从头截段长；不够长则截整条+末帧冻结补齐（tpad clone）
  → 统一参数重编码 → concat → 铺模板原视频的音频轨（时长天然对齐）→ drawtext 字幕
  → final.mp4 + assembly.json（每段选用镜头/分数/不足标记，可审计）

CLI：python -m src.library.assemble_lib --template-id <id>
产物：data/library/stories/<tid>/final.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.generation.assemble import build_drawtext_cmd, escape_drawtext
from src.library.build_index import load_index
from src.perception import common

logger = logging.getLogger(__name__)


def build_seg_cmd(src: Path, dst: Path, *, shot_start: float, shot_dur: float,
                  need_dur: float, crf: int, w: int = 544, h: int = 960) -> list[str]:
    """截取镜头片段；不足段长时末帧冻结补齐。统一 544×960 竖屏重编码。"""
    take = min(shot_dur, need_dur)
    pad = max(0.0, need_dur - shot_dur)
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    if pad > 0.05:
        vf += f",tpad=stop_mode=clone:stop_duration={pad:g}"
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{shot_start:g}",
            "-i", str(src), "-t", f"{take:g}", "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", str(dst)]


def run_assemble(cfg: AppConfig, template_id: str, *, force: bool = False) -> Path:
    a_cfg = cfg.library.get("assemble") or {}
    crf = int(a_cfg.get("crf", 20))
    story_dir = cfg.paths.library_dir / "stories" / template_id
    final = story_dir / "final.mp4"
    if final.exists() and not force:
        logger.info("[assemble %s] 已有产物，跳过", template_id)
        return final

    sb = json.loads((story_dir / "storyboard.json").read_text(encoding="utf-8"))["segments"]
    sel = json.loads((story_dir / "selection.json").read_text(encoding="utf-8"))
    rows, _emb = load_index(cfg)
    template = json.loads(
        (cfg.paths.perception_dir / template_id / "template" / "result.json")
        .read_text(encoding="utf-8"))["output"]["template"]

    work = story_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    seg_files, audit = [], []
    for seg, pick in zip(sb, sel):
        dur = float(seg["end"]) - float(seg["start"])
        out = work / f"seg{pick['seg_idx']:02d}.mp4"
        if pick.get("picked") is None:
            # 库内不足：黑场占位（保时间线节奏，人工可见缺口）
            common.run_ffmpeg("ffmpeg", [
                "-y", "-loglevel", "error", "-f", "lavfi",
                "-i", f"color=black:s=544x960:d={dur:g}:r=24", "-c:v", "libx264",
                "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)])
            audit.append({**pick, "placeholder": True})
            seg_files.append(out)
            continue
        row = rows[pick["picked"]["row_idx"]]
        cmd = build_seg_cmd(Path(row["video"]), out, shot_start=row["start_s"],
                            shot_dur=row["duration_s"], need_dur=dur, crf=crf)
        common.run_ffmpeg("ffmpeg", cmd[1:])
        # 实长校验：镜头真实素材可能早于标称结束（scene 边界在片尾/静止段的误差），
        # 不足段长则按实测补冻结帧（2026-09-08 首条成片 8.0s/12.9s 实测）
        ffprobe = cfg.perception.get("ffprobe_bin", "ffprobe")
        actual = common.video_duration_s(ffprobe, out)
        if actual is not None and actual + 0.1 < dur:
            pad = dur - actual + 0.2
            common.run_ffmpeg("ffmpeg", [
                "-y", "-loglevel", "error", "-i", str(out), "-vf",
                f"tpad=stop_mode=clone:stop_duration={pad:g}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-t", f"{dur:g}", str(out)])
            logger.info("[assemble %s] seg%02d 实长 %.2f 补齐至 %.2f",
                        template_id, pick["seg_idx"], actual, dur)
        audit.append({**pick, "used": {"video": row["video_stem"],
                                       "shot_idx": row["shot_idx"],
                                       "caption": row.get("caption", "")}})
        seg_files.append(out)

    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{f.as_posix()}'\n" for f in seg_files),
                           encoding="utf-8")
    concat_out = work / "concat.mp4"
    common.run_ffmpeg("ffmpeg", ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                                 "-i", str(concat_list), "-c", "copy", str(concat_out)])

    # 模板原视频的音频轨；音轨可能短于画面（猫模板实测 8s/12.9s）→ apad 垫齐而非 -shortest 截断
    src_video = cfg.paths.videos_dir / template_id / "video.mp4"
    voiced = work / "voiced.mp4" if src_video.exists() else None
    if voiced:
        concat_dur = common.video_duration_s(
            cfg.perception.get("ffprobe_bin", "ffprobe"), concat_out) or 0
        common.run_ffmpeg("ffmpeg", [
            "-y", "-loglevel", "error", "-i", str(concat_out), "-i", str(src_video),
            "-map", "0:v", "-map", "1:a?", "-af", "apad", "-t", f"{concat_dur + 0.5:g}",
            "-c:v", "copy", "-c:a", "aac", "-ar",
            str(int(a_cfg.get("audio_rate", 44100))), str(voiced)])
    base = voiced or concat_out

    # 字幕：模板 timeline 的 text 段（drawtext，复用 generation 的转义）
    font = Path(cfg.generation.get("assemble", {}).get(
        "font", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_size = int(cfg.generation.get("assemble", {}).get("font_size", 56))
    overlays = [{"start_s": s["start"], "end_s": s["end"], "text": s["text"]}
                for s in template.get("timeline") or [] if s.get("text")]
    if overlays:
        cmd = build_drawtext_cmd(base, final, overlays, font, font_size)
        common.run_ffmpeg("ffmpeg", cmd[1:])
    else:
        final.write_bytes(base.read_bytes())

    dur = common.video_duration_s(cfg.perception.get("ffprobe_bin", "ffprobe"), final)
    (story_dir / "assembly.json").write_text(json.dumps({
        "template_id": template_id, "final": str(final), "duration_s": dur,
        "n_segments": len(seg_files), "n_placeholder": sum(1 for a in audit if a.get("placeholder")),
        "segments": audit}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[assemble %s] final %.2fs（%d 段，%d 占位）→ %s",
                template_id, dur or 0, len(seg_files),
                sum(1 for a in audit if a.get("placeholder")), final)
    common.emit_status_line("ok", output=str(final), duration_s=dur)
    return final


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="素材库装配成片")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_assemble")
    try:
        run_assemble(cfg, args.template_id, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("assemble_lib 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
