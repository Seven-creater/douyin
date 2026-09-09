"""Deterministic FFmpeg/OpenCV recipe renderer with explicit capability reporting."""
from __future__ import annotations

import json
import shutil
import subprocess
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


def segment_filter(operations: list[dict], *, width: int = 544, height: int = 960,
                   duration_s: float) -> str:
    filters = [f"scale={width}:{height}:force_original_aspect_ratio=decrease",
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black", "fps=24"]
    for op in operations:
        params = op.get("params") or {}
        if op["type"] == "speed_ramp":
            rate = min(4.0, max(0.25, float(params.get("rate", 1.0))))
            filters.append(f"setpts=PTS/{rate:g}")
        elif op["type"] in {"crop_reframe", "zoom_punch"}:
            scale = min(2.0, max(1.0, float(params.get("scale", 1.2))))
            filters.extend([f"scale=iw*{scale:g}:ih*{scale:g}",
                            f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"])
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
    retrieval_by_slot = {int(row["slot_idx"]): row for row in retrieval}
    segment_paths = []
    commands: list[list[str]] = []
    runtime_status = []
    for slot in asset_plan["slots"]:
        index = int(slot["slot_idx"])
        duration = float(slot["need_duration_s"])
        destination = work / f"segment_{index:04d}.mp4"
        selection = retrieval_by_slot.get(index) or {}
        picked = selection.get("picked")
        active = operations_for_interval(recipe, float(slot["start_s"]), float(slot["end_s"]))
        if picked is None:
            args = ["-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=black:s=544x960:d={duration:g}:r=24", "-an", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(destination)]
        else:
            args = ["-y", "-loglevel", "error", "-ss", f"{picked['source_start_s']:g}",
                    "-i", str(picked["video"]), "-t", f"{max(duration, 0.1):g}",
                    "-vf", segment_filter(active, duration_s=duration), "-an", "-c:v",
                    "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt",
                    "yuv420p", str(destination)]
        common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args,
                          timeout_s=max(300, duration * 20))
        commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])
        segment_paths.append(destination)
    concat_file = work / "concat.txt"
    concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
                           encoding="utf-8")
    concat_video = work / "concat.mp4"
    args = ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
            str(concat_file), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf",
            "20", "-pix_fmt", "yuv420p", str(concat_video)]
    common.run_ffmpeg(cfg.perception.get("ffmpeg_bin", "ffmpeg"), args)
    commands.append([cfg.perception.get("ffmpeg_bin", "ffmpeg"), *args])

    mask_operations = [op for op in recipe.get("operations") or [] if op["type"] in MASK_OPS]
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
    filter_chain = final_filter(recipe, font=Path(cfg.generation.get("assemble", {}).get(
        "font", "")))
    if reference.exists():
        args = ["-y", "-loglevel", "error", "-i", str(masked_video), "-i", str(reference),
                "-vf", filter_chain, "-map", "0:v", "-map", "1:a?", "-af", "apad",
                "-t", f"{float(recipe['reference']['duration_s']):g}", "-c:v", "libx264",
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
