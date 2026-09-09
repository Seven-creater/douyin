"""Deterministic FFmpeg/OpenCV recipe renderer with explicit capability reporting."""
from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from src.agentic_video.recipe_v2 import validate_recipe_v2
from src.config import AppConfig
from src.config import repo_root
from src.generation.assemble import escape_drawtext
from src.perception import common

MASK_OPS = {"tracked_mask_fill", "subject_cutout_composite", "mask_wipe",
            "foreground_occlusion_transition"}
FFMPEG_OPS = {"hard_cut", "trim", "speed_ramp", "crop_reframe", "zoom_punch",
              "luma_flash", "color_adjust", "opacity_blend", "text_overlay",
              "text_layer_animation", "beat_freeze"}


class MaskBackend(Protocol):
    def apply(self, video: Path, operations: list[dict], output: Path,
              work_dir: Path) -> list[list[str]]: ...


class GroundedSamSubprocessBackend:
    """Run local GroundingDINO+SAM2 in its dedicated server environment."""

    def __init__(self, config: dict):
        self.config = config

    def build_command(self, source: Path, operation: dict, output: Path,
                      work_dir: Path) -> list[str]:
        params = operation.get("params") or {}
        target = str(params.get("target") or params.get("subject") or "person")
        if not target.isascii():
            target = "person"
        if not target.endswith("."):
            target += "."
        return [
            str(self.config["python"]), "-m", "src.agentic_video.grounded_sam_backend",
            "--input", str(source), "--output", str(output), "--work-dir", str(work_dir),
            "--start", str(operation["interval"][0]), "--end", str(operation["interval"][1]),
            "--mode", operation["type"], "--target", target,
            "--grounding-model", str(self.config["grounding_model"]),
            "--sam2-code-root", str(self.config["sam2_code_root"]),
            "--sam2-checkpoint", str(self.config["sam2_checkpoint"]),
            "--sam2-model-config", str(self.config.get(
                "sam2_model_config", "configs/sam2.1/sam2.1_hiera_b+.yaml")),
        ]

    def apply(self, video: Path, operations: list[dict], output: Path,
              work_dir: Path) -> list[list[str]]:
        commands = []
        current = video
        for index, operation in enumerate(operations):
            destination = output if index == len(operations) - 1 else \
                work_dir / f"mask_stage_{index:02d}.mp4"
            stage_dir = work_dir / f"mask_stage_{index:02d}"
            command = self.build_command(current, operation, destination, stage_dir)
            proc = subprocess.run(command, cwd=repo_root(), capture_output=True, text=True,
                                  timeout=int(self.config.get("timeout_s", 1800)))
            if proc.returncode != 0:
                raise RuntimeError(f"Grounded-SAM2 failed: {(proc.stderr or '')[-500:]}")
            commands.append(command)
            current = destination
        return commands


def operations_for_interval(recipe: dict, start: float, end: float) -> list[dict]:
    return [op for op in recipe.get("operations") or []
            if op["interval"][0] <= end and op["interval"][1] >= start]


def segment_filter(operations: list[dict], *, width: int = 720, height: int = 960,
                   duration_s: float, focus_x: float = 0.5) -> str:
    focus_x = min(1.0, max(0.0, float(focus_x)))
    crop_x = (f"(iw-{width})*{focus_x:g}" if abs(focus_x - 0.5) > 1e-6
              else f"(iw-{width})/2")
    filters = [f"scale={width}:{height}:force_original_aspect_ratio=increase",
               f"crop={width}:{height}:{crop_x}:(ih-{height})/2",
               "setsar=1", "fps=24"]
    for op in operations:
        params = op.get("params") or {}
        if op["type"] == "speed_ramp":
            rate = min(4.0, max(0.25, float(params.get("rate", 1.0))))
            filters.append(f"setpts=PTS/{rate:g}")
        elif op["type"] in {"crop_reframe", "zoom_punch"}:
            scale = min(2.0, max(1.0, float(params.get("scale", 1.2))))
            filters.extend([f"scale=iw*{scale:g}:ih*{scale:g}",
                            f"crop={width}:{height}:{crop_x}:(ih-{height})/2"])
        elif op["type"] == "color_adjust":
            filters.append("eq=saturation=1.25:contrast=1.08")
        elif op["type"] == "beat_freeze":
            filters.append("tpad=stop_mode=clone:stop_duration=0.25")
    filters.extend([f"tpad=stop_mode=clone:stop_duration={duration_s:g}",
                    f"trim=duration={duration_s:g}", "setpts=PTS-STARTPTS"])
    return ",".join(filters)


def final_filter(recipe: dict, *, font: Path | None = None) -> str:
    filters = []
    for op in recipe.get("operations") or []:
        start, end = map(float, op["interval"])
        params = op.get("params") or {}
        if op["type"] == "luma_flash":
            if end <= start:
                end = start + max(1 / float(recipe["reference"]["fps"]), 0.08)
            filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=white@1:t=fill:"
                           f"enable='between(t,{start:g},{end:g})'")
        elif op["type"] == "opacity_blend":
            opacity = min(1.0, max(0.0, float(params.get("opacity", 0.5))))
            filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=purple@{opacity:g}:t=fill:"
                           f"enable='between(t,{start:g},{end:g})'")
        elif op["type"] in {"text_overlay", "text_layer_animation"}:
            text = escape_drawtext(str(params.get("text") or params.get("subject") or ""))
            if not text:
                continue
            y = "h*0.78"
            if op["type"] == "text_layer_animation":
                # Commas inside an FFmpeg expression must be escaped or the
                # filter parser treats them as additional filter separators.
                y = f"h-min(h*0.22\\,(t-{start:g})*h*0.8)"
            font_arg = f":fontfile='{font.as_posix()}'" if font and font.exists() else ""
            filters.append(f"drawtext=text='{text}'{font_arg}:fontsize=52:fontcolor=white:"
                           f"borderw=3:x=(w-text_w)/2:y={y}:enable='between(t,{start:g},{end:g})'")
    return ",".join(filters) if filters else "null"


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_story_subtitles(asset_plan: dict, retrieval: list[dict], output: Path) -> Path:
    """Map source-time dialogue translations into deterministic output-time SRT."""
    by_slot = {int(row["slot_idx"]): row for row in retrieval}
    cues = []
    for slot in asset_plan.get("slots") or []:
        picked = (by_slot.get(int(slot["slot_idx"])) or {}).get("picked") or {}
        source_start = float(picked.get("source_start_s") or 0)
        target_start = float(slot["start_s"])
        target_end = float(slot["end_s"])
        for line in picked.get("dialogue") or []:
            subtitle = str(line.get("translation_zh") or "").strip()
            if not subtitle or subtitle == "uncertain":
                continue
            start = target_start + max(0.0, float(line.get("start_s") or source_start)
                                       - source_start)
            end = target_start + max(0.0, float(line.get("end_s") or source_start)
                                     - source_start)
            start, end = max(target_start, start), min(target_end, end)
            if end - start >= 0.1:
                cues.append((start, end, subtitle.replace("\r", " ").replace("\n", " ")))
    output = Path(output)
    lines = []
    for idx, (start, end, subtitle) in enumerate(sorted(cues), 1):
        lines.extend([str(idx), f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}",
                      subtitle, ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _scaled_recipe(recipe: dict, duration_s: float) -> dict:
    reference_duration = float(recipe["reference"]["duration_s"])
    if reference_duration <= 0 or abs(reference_duration - duration_s) < 1e-6:
        return recipe
    scaled = deepcopy(recipe)
    ratio = duration_s / reference_duration
    scaled["reference"]["duration_s"] = duration_s
    for operation in scaled.get("operations") or []:
        operation["interval"] = [round(float(value) * ratio, 6)
                                 for value in operation["interval"]]
    return scaled


def _subtitle_filter(path: Path) -> str:
    escaped = path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
    return f"subtitles=filename='{escaped}':charenc=UTF-8"


def _has_audio_stream(ffprobe_bin: str, video: Path) -> bool:
    probe = common.run_ffprobe_json(ffprobe_bin, video)
    return any(stream.get("codec_type") == "audio"
               for stream in probe.get("streams") or [])


def render_recipe(cfg: AppConfig, recipe: dict, asset_plan: dict, retrieval: list[dict],
                  output_dir: Path, *, mask_backend: MaskBackend | None = None,
                  force: bool = False) -> Path:
    errors = validate_recipe_v2(recipe)
    if errors:
        raise ValueError("invalid Recipe v2: " + "; ".join(errors))
    # FFmpeg's concat demuxer resolves entries relative to concat.txt.  Always
    # materialize the run directory before writing segment paths so a relative
    # CLI output such as ``data/runs/demo`` cannot be prefixed twice.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / "rendered.mp4"
    if final.exists() and not force:
        return final
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    narrative_mode = bool(asset_plan.get("narrative_program_required"))
    canvas_cfg = (cfg.library.get("narrative_render") or {}
                  if narrative_mode else cfg.generation.get("assemble", {}))
    canvas_width = int(canvas_cfg.get("width", 1920 if narrative_mode else 720))
    canvas_height = int(canvas_cfg.get("height", 1080 if narrative_mode else 960))
    render_duration = (max(float(slot["end_s"]) for slot in asset_plan["slots"])
                       if narrative_mode else float(recipe["reference"]["duration_s"]))
    execution_recipe = _scaled_recipe(recipe, render_duration) if narrative_mode else recipe
    retrieval_by_slot = {int(row["slot_idx"]): row for row in retrieval}
    segment_paths = []
    commands: list[list[str]] = []
    runtime_status = []
    audio_streams: dict[str, bool] = {}
    for slot in asset_plan["slots"]:
        index = int(slot["slot_idx"])
        duration = float(slot["need_duration_s"])
        destination = work / f"segment_{index:04d}.mp4"
        selection = retrieval_by_slot.get(index) or {}
        picked = selection.get("picked")
        active = operations_for_interval(execution_recipe, float(slot["start_s"]),
                                         float(slot["end_s"]))
        if picked is None:
            args = ["-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=black:s={canvas_width}x{canvas_height}:d={duration:g}:r=24"]
            if narrative_mode:
                args.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                             "-map", "0:v:0", "-map", "1:a:0"])
            else:
                args.append("-an")
            args.extend(["-t", f"{duration:g}", "-c:v", "libx264", "-pix_fmt",
                         "yuv420p"])
            if narrative_mode:
                args.extend(["-c:a", "aac"])
            args.append(str(destination))
        else:
            source_video = Path(picked["video"])
            source_duration = max(
                0.1, float(picked.get("source_end_s",
                                      picked["source_start_s"] + duration))
                - float(picked["source_start_s"]))
            args = ["-y", "-loglevel", "error", "-ss", f"{picked['source_start_s']:g}",
                    "-t", f"{source_duration:g}", "-i", str(source_video)]
            source_has_audio = False
            if narrative_mode:
                cache_key = str(source_video.resolve())
                if cache_key not in audio_streams:
                    audio_streams[cache_key] = _has_audio_stream(
                        cfg.perception.get("ffprobe_bin", "ffprobe"), source_video)
                source_has_audio = audio_streams[cache_key]
                if not source_has_audio:
                    args.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"])
            args.extend(["-vf", segment_filter(active, width=canvas_width,
                                           height=canvas_height, duration_s=duration,
                                           focus_x=float(picked.get("focus_x", 0.5))),
                         "-t", f"{max(duration, 0.1):g}"])
            if narrative_mode:
                args.extend(["-map", "0:v:0", "-map",
                             "0:a:0" if source_has_audio else "1:a:0", "-af",
                             f"apad=pad_dur={duration:g},atrim=duration={duration:g},"
                             "asetpts=PTS-STARTPTS"])
            else:
                args.append("-an")
            args.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                         "-pix_fmt", "yuv420p"])
            if narrative_mode:
                args.extend(["-c:a", "aac", "-ar", "48000", "-ac", "2"])
            args.append(str(destination))
        common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args,
                          timeout_s=max(300, duration * 20))
        commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])
        segment_paths.append(destination)
    concat_file = work / "concat.txt"
    concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
                           encoding="utf-8")
    concat_video = work / "concat.mp4"
    args = ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
            str(concat_file)]
    if not narrative_mode:
        args.append("-an")
    args.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-pix_fmt", "yuv420p"])
    if narrative_mode:
        args.extend(["-c:a", "aac", "-ar", "48000", "-ac", "2"])
    args.append(str(concat_video))
    common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args)
    commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])

    mask_operations = [op for op in execution_recipe.get("operations") or []
                       if op["type"] in MASK_OPS]
    masked_video = concat_video
    if mask_operations and mask_backend is not None:
        masked_video = work / "masked.mp4"
        commands.extend(mask_backend.apply(concat_video, mask_operations, masked_video, work))
        runtime_status.extend({"operation_id": op["id"], "status": "executed",
                               "backend": "grounded_sam2"} for op in mask_operations)
    elif mask_operations:
        runtime_status.extend({"operation_id": op["id"], "status": "unsupported",
                               "reason": "mask backend unavailable"} for op in mask_operations)

    reference = Path(recipe["reference"]["uri"])
    filter_chain = final_filter(execution_recipe, font=Path(cfg.generation.get("assemble", {}).get(
        "font", "")))
    if narrative_mode:
        subtitles = write_story_subtitles(asset_plan, retrieval, output_dir / "subtitles.srt")
        if subtitles.stat().st_size:
            subtitle_filter = _subtitle_filter(subtitles)
            filter_chain = (subtitle_filter if filter_chain == "null"
                            else f"{filter_chain},{subtitle_filter}")
        args = ["-y", "-loglevel", "error", "-i", str(masked_video),
                "-vf", filter_chain, "-map", "0:v", "-map", "0:a?",
                "-t", f"{render_duration:g}", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", str(final)]
        common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args)
        commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])
    elif reference.exists():
        args = ["-y", "-loglevel", "error", "-i", str(masked_video), "-i", str(reference),
                "-vf", filter_chain, "-map", "0:v", "-map", "1:a?", "-af", "apad",
                "-t", f"{render_duration:g}", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a",
                "aac", str(final)]
        common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args)
        commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])
    elif filter_chain != "null":
        args = ["-y", "-loglevel", "error", "-i", str(masked_video), "-vf", filter_chain,
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(final)]
        common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args)
        commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])
    else:
        shutil.copy2(masked_video, final)

    executed_types = FFMPEG_OPS | (MASK_OPS if mask_backend is not None else set())
    for op in recipe.get("operations") or []:
        if op["type"] in MASK_OPS:
            continue
        status = "executed" if op["type"] in executed_types else "unsupported"
        row = {"operation_id": op["id"], "status": status}
        if status == "unsupported":
            row["reason"] = f"renderer has no {op['type']} implementation"
        runtime_status.append(row)
    (output_dir / "render_manifest.json").write_text(json.dumps({
        "output": str(final), "commands": commands, "operations": runtime_status,
        "deterministic": True, "seed": recipe["provenance"]["seed"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return final
