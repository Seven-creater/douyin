"""assemble：裁剪回时间线 → concat → drawtext 字幕 → final.mp4。

CLI：python -m src.generation.assemble --template-id <id> --variant <vid> [--allow-missing]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.generation import models
from src.generation.planner import load_template
from src.perception import common

logger = logging.getLogger(__name__)


def escape_drawtext(text: str) -> str:
    """drawtext 双层转义：filter 层（' : \）+ text 字段层（% ,）。"""
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:").replace("'", "\\\\'")
    text = text.replace("%", "\\%").replace(",", "\\,")
    return text


def build_trim_cmd(src: Path, dst: Path, *, ss: float, t: float, audio_rate: int, crf: int) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:g}", "-i", str(src), "-t", f"{t:g}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", str(audio_rate), "-ac", "2", "-movflags", "+faststart", str(dst)]


def build_concat_cmd(list_path: Path, dst: Path) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-c", "copy", str(dst)]


def build_drawtext_cmd(src: Path, dst: Path, overlays: list[dict], font: Path, font_size: int) -> list[str]:
    filters = []
    for ov in overlays:
        filters.append(
            f"drawtext=fontfile='{font}':text='{escape_drawtext(ov['text'])}':"
            f"fontcolor=white:fontsize={font_size}:borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y=h*0.78:enable='between(t,{ov['start_s']:g},{ov['end_s']:g})'"
        )
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "copy", str(dst)]


def build_black_placeholder(dst: Path, *, duration_s: float, audio_rate: int) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", f"color=black:s=544x960:d={duration_s:g}:r=24",
            "-f", "lavfi", "-i", f"anullsrc=r={audio_rate}:cl=stereo",
            "-t", f"{duration_s:g}", "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(dst)]


def load_plan(cfg: AppConfig, template_id: str) -> models.GenerationPlan:
    env = common.read_result_json(cfg.paths.generation_dir / template_id / "plan")
    if env is None:
        raise FileNotFoundError(f"缺少 plan 产物（先跑 planner）：{cfg.paths.generation_dir / template_id / 'plan'}")
    p = env["output"]["plan"]
    units = [models.GenerationUnit(
        unit_id=u["unit_id"], roles=u["roles"], timeline_span=tuple(u["timeline_span"]),
        duration_s=u["duration_s"], num_frames=u["num_frames"], seed_base=u["seed_base"],
        visual_brief=u["visual_brief"], speech_lines=u["speech_lines"],
        segments=[models.UnitSegment(**s) for s in u["segments"]],
    ) for u in p["units"]]
    return models.GenerationPlan(template_id=p["template_id"], source_duration_s=p["source_duration_s"],
                                 audio_brief=p.get("audio_brief"), units=units,
                                 core_meme=p.get("core_meme") or "",
                                 fixed_elements=list(p.get("fixed_elements") or []))


def load_prompts(cfg: AppConfig, template_id: str, variant_id: str) -> list[dict]:
    env = common.read_result_json(cfg.paths.generation_dir / template_id / "variants" / variant_id / "prompts")
    if env is None:
        raise FileNotFoundError(f"缺少 prompts 产物（先跑 rewrite）：variant={variant_id}")
    return env["output"]["unit_prompts"]


def assemble_variant(cfg: AppConfig, template_id: str, variant_id: str, *,
                     allow_missing: bool = False) -> Path:
    a_cfg = (cfg.generation or {}).get("assemble") or {}
    audio_rate = int(a_cfg.get("audio_rate", 44100))
    crf = int(a_cfg.get("crf", 20))
    font = Path(a_cfg.get("font", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_size = int(a_cfg.get("font_size", 56))
    if not font.exists():
        raise FileNotFoundError(f"字幕字体不存在: {font}")

    plan = load_plan(cfg, template_id)
    template = load_template(cfg, template_id)
    gdir = cfg.paths.generation_dir / template_id
    vdir = gdir / "variants" / variant_id
    clips_dir = vdir / "clips"
    work = vdir / "work"
    work.mkdir(parents=True, exist_ok=True)

    ffprobe_bin = cfg.perception.get("ffprobe_bin", "ffprobe")
    steps = []
    seg_files: list[Path] = []
    for u in plan.units:
        clip = clips_dir / f"u{u.unit_id:02d}.mp4"
        if not clip.exists():
            if not allow_missing:
                raise FileNotFoundError(f"缺少生成片段 {clip}（先跑 generate 或用 --allow-missing）")
            ph = work / f"placeholder_u{u.unit_id:02d}.mp4"
            if not ph.exists():
                common.run_ffmpeg("ffmpeg", build_black_placeholder(ph, duration_s=u.duration_s + 0.6,
                                                                    audio_rate=audio_rate)[3:])
            # 占位片按段切分
            for seg in u.segments:
                t = seg.end_s - seg.start_s
                out = work / f"u{u.unit_id:02d}_seg{seg.seg_index}.mp4"
                cmd = build_trim_cmd(ph, out, ss=seg.unit_offset_s, t=t, audio_rate=audio_rate, crf=crf)
                common.run_ffmpeg("ffmpeg", cmd[1:])
                steps.append({"step": "trim", "unit": u.unit_id, "seg": seg.seg_index,
                              "src": str(ph), "out": str(out), "ss": seg.unit_offset_s, "t": t,
                              "placeholder": True, "cmd": cmd})
                seg_files.append(out)
            continue
        # 生成片段实际时长校验（不足则收缩窗口）
        actual = common.video_duration_s(ffprobe_bin, clip)
        for seg in u.segments:
            t = seg.end_s - seg.start_s
            ss = seg.unit_offset_s
            if actual and ss + t > actual:
                ss = max(0.0, actual - t)
                logger.warning("[assemble %s] u%02d 时长 %.2f 不足，窗口收缩 ss=%.2f",
                               variant_id, u.unit_id, actual, ss)
            out = work / f"u{u.unit_id:02d}_seg{seg.seg_index}.mp4"
            cmd = build_trim_cmd(clip, out, ss=ss, t=t, audio_rate=audio_rate, crf=crf)
            common.run_ffmpeg("ffmpeg", cmd[1:])
            steps.append({"step": "trim", "unit": u.unit_id, "seg": seg.seg_index,
                          "src": str(clip), "out": str(out), "ss": ss, "t": t, "cmd": cmd})
            seg_files.append(out)

    # concat（demuxer，参数已统一）
    concat_list = work / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{f.as_posix()}'\n" for f in seg_files), encoding="utf-8")
    concat_out = work / "concat.mp4"
    cmd = build_concat_cmd(concat_list, concat_out)
    common.run_ffmpeg("ffmpeg", cmd[1:])
    steps.append({"step": "concat", "n_inputs": len(seg_files), "out": str(concat_out), "cmd": cmd})

    # 字幕（timeline[].text 非 null 段）
    overlays = [{"start_s": s["start"], "end_s": s["end"], "text": s["text"]}
                for s in template.get("timeline") or [] if s.get("text")]
    final = vdir / "final.mp4"
    if overlays:
        cmd = build_drawtext_cmd(concat_out, final, overlays, font, font_size)
        common.run_ffmpeg("ffmpeg", cmd[1:])
        steps.append({"step": "drawtext", "events": len(overlays), "font": str(font),
                      "out": str(final), "cmd": cmd})
    else:
        final.write_bytes(concat_out.read_bytes())

    final_dur = common.video_duration_s(ffprobe_bin, final)
    (vdir / "assembly.json").write_text(json.dumps({
        "variant_id": variant_id, "final_path": str(final), "duration_s": final_dur,
        "expected_duration_s": plan.source_duration_s, "overlays": overlays, "steps": steps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[assemble %s] final %.2fs（期望 %.2fs）→ %s", variant_id, final_dur,
                plan.source_duration_s, final)
    return final


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="装配最终视频")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="gen_assemble")
    try:
        final = assemble_variant(cfg, args.template_id, args.variant, allow_missing=args.allow_missing)
        common.emit_status_line("ok", output=str(final))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("assemble 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
