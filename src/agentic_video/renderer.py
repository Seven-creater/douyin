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
from src.perception import common

MASK_OPS = {"tracked_mask_fill", "subject_cutout_composite", "mask_wipe",
            "foreground_occlusion_transition"}
FFMPEG_OPS = {"hard_cut", "trim", "speed_ramp", "crop_reframe", "zoom_punch",
              "luma_flash", "color_adjust", "opacity_blend", "text_overlay",
              "text_layer_animation", "beat_freeze"}

# 成片画布规则（2026-09-11 用户拍板）：所有剪辑成片一律 1920×1080 横屏。
# 叙事/编辑两模式共用此默认；要出竖屏必须显式改配置，不得靠代码回退值复活。
RENDER_CANVAS_DEFAULT = (1920, 1080)


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


def drawtext_text_value(text: str) -> str:
    """drawtext text= 的安全值（H2）：整体单引号包裹，' 用 关-转-开 注入，
    配 expansion=none 免 %/{} 展开。旧 escape_drawtext 的 \\' 转义在
    text='...' 包装下实测（ffmpeg 6.1.1）引号配对错乱直接 Filter not found。"""
    return "'" + str(text).replace("\\", "\\\\").replace("'", "'\\''") + "'"


def operations_for_interval(recipe: dict, start: float, end: float) -> list[dict]:
    return [op for op in recipe.get("operations") or []
            if op["interval"][0] <= end and op["interval"][1] >= start]


def segment_filter(operations: list[dict], *,
                   width: int = RENDER_CANVAS_DEFAULT[0],
                   height: int = RENDER_CANVAS_DEFAULT[1],
                   duration_s: float, focus_x: float = 0.5,
                   slot_start: float = 0.0, slot_end: float | None = None,
                   ) -> tuple[str, list[dict]]:
    """slot 内特效滤镜（H4）：op 区间与 slot 求交，按 timeline 支持能力分级。

    eq 支持 enable=between → 局部窗口生效；setpts/scale/tpad 不支持 timeline
    （ffmpeg 6.1.1 实测报 "Timeline not supported"）→ 仅当 op 覆盖几乎整个
    slot 才整段应用，否则跳过并在返回的 skipped 里记录（render_manifest
    显式报告 partial_interval_skipped，不静默错位）。segment 输出时间轴从 0
    起算（结尾 setpts=PTS-STARTPTS），故 enable 窗口用 op∩slot 再减 slot_start。
    """
    slot_end = duration_s if slot_end is None else slot_end
    slot_len = max(1e-6, slot_end - slot_start)
    focus_x = min(1.0, max(0.0, float(focus_x)))
    crop_x = (f"(iw-{width})*{focus_x:g}" if abs(focus_x - 0.5) > 1e-6
              else f"(iw-{width})/2")
    filters = [f"scale={width}:{height}:force_original_aspect_ratio=increase",
               f"crop={width}:{height}:{crop_x}:(ih-{height})/2",
               "setsar=1", "fps=24"]
    skipped: list[dict] = []

    def _local_window(op: dict) -> tuple[float, float, float]:
        left = max(float(op["interval"][0]), slot_start) - slot_start
        right = min(float(op["interval"][1]), slot_end) - slot_start
        coverage = max(0.0, right - left) / slot_len
        return max(0.0, left), max(0.0, right), coverage

    def _skip(op: dict, left: float, right: float):
        skipped.append({"operation_id": op.get("id"), "type": op["type"],
                        "status": "partial_interval_skipped",
                        "reason": "filter has no timeline support and op interval "
                                  f"covers only part of the slot [{left:g},{right:g}]s"})

    for op in operations:
        params = op.get("params") or {}
        left, right, coverage = _local_window(op)
        if coverage <= 0:
            continue
        whole = coverage >= 0.98
        if op["type"] == "speed_ramp":
            if not whole:
                _skip(op, left, right)
                continue
            rate = min(4.0, max(0.25, float(params.get("rate", 1.0))))
            filters.append(f"setpts=PTS/{rate:g}")
        elif op["type"] in {"crop_reframe", "zoom_punch"}:
            if not whole:
                _skip(op, left, right)
                continue
            scale = min(2.0, max(1.0, float(params.get("scale", 1.2))))
            filters.extend([f"scale=iw*{scale:g}:ih*{scale:g}",
                            f"crop={width}:{height}:{crop_x}:(ih-{height})/2"])
        elif op["type"] == "color_adjust":
            if whole:
                filters.append("eq=saturation=1.25:contrast=1.08")
            else:
                filters.append(f"eq=saturation=1.25:contrast=1.08:"
                               f"enable='between(t,{left:g},{right:g})'")
        elif op["type"] == "beat_freeze":
            if not whole:
                _skip(op, left, right)
                continue
            filters.append("tpad=stop_mode=clone:stop_duration=0.25")
    filters.extend([f"tpad=stop_mode=clone:stop_duration={duration_s:g}",
                    f"trim=duration={duration_s:g}", "setpts=PTS-STARTPTS"])
    return ",".join(filters), skipped


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
            text = drawtext_text_value(
                str(params.get("text") or params.get("subject") or ""))
            if text == "''":
                continue
            y = "h*0.78"
            if op["type"] == "text_layer_animation":
                # Commas inside an FFmpeg expression must be escaped or the
                # filter parser treats them as additional filter separators.
                y = f"h-min(h*0.22\\,(t-{start:g})*h*0.8)"
            font_arg = f":fontfile='{font.as_posix()}'" if font and font.exists() else ""
            filters.append(f"drawtext=text={text}{font_arg}:fontsize=52:fontcolor=white:"
                           f"expansion=none:borderw=3:x=(w-text_w)/2:y={y}:"
                           f"enable='between(t,{start:g},{end:g})'")
    return ",".join(filters) if filters else "null"


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_story_subtitles(asset_plan: dict, retrieval: list[dict], output: Path) -> Path:
    """Map source-time dialogue translations into deterministic output-time SRT.

    文案轨优先（2026-09-10 MVP）：asset_plan 带 copy_cues（7682 型模板的
    钩子/成就卡/反转梗，已在成片时间轴）时直接写文案 SRT 并跳过对白映射——
    该型片子的表达公式是"纯文案 + 纯 BGM"，电影对白翻译字幕会互相打架。
    """
    by_slot = {int(row["slot_idx"]): row for row in retrieval}
    cues = []
    total = max((float(slot.get("end_s") or 0)
                 for slot in asset_plan.get("slots") or []), default=0.0)
    copy_cues = asset_plan.get("copy_cues") or []
    if copy_cues:
        for cue in copy_cues:
            start = max(0.0, float(cue.get("start_s") or 0))
            end = min(total, float(cue.get("end_s") or start))
            subtitle = str(cue.get("text") or "").strip()
            if subtitle and end - start >= 0.1:
                cues.append((start, end, subtitle.replace("\r", " ").replace("\n", " ")))
    else:
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
    """叙事重剪的执行视图：区间缩放到目标时长，并剔除参考片的文字层操作。

    text_overlay/text_layer_animation 的 text 描述的是参考视频自己的字幕/贴纸
    （如"黄色的 NANCHANG 文字叠加"），叙事重剪烧到新素材上是张冠李戴
    （2026-09-10 v5 帧验实锤）。编辑模式不经过本函数，模仿参考剪辑程序时保留。
    """
    scaled = deepcopy(recipe)
    reference_duration = float(scaled["reference"]["duration_s"])
    if reference_duration > 0 and abs(reference_duration - duration_s) >= 1e-6:
        ratio = duration_s / reference_duration
        scaled["reference"]["duration_s"] = duration_s
        for operation in scaled.get("operations") or []:
            operation["interval"] = [round(float(value) * ratio, 6)
                                     for value in operation["interval"]]
    scaled["operations"] = [
        op for op in scaled.get("operations") or []
        if op.get("type") not in {"text_overlay", "text_layer_animation"}]
    return scaled


def _subtitle_filter(path: Path) -> str:
    escaped = path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
    return f"subtitles=filename='{escaped}':charenc=UTF-8"


def _has_audio_stream(ffprobe_bin: str, video: Path) -> bool:
    probe = common.run_ffprobe_json(ffprobe_bin, video)
    return any(stream.get("codec_type") == "audio"
               for stream in probe.get("streams") or [])


def source_subtitle_treatment(cfg: AppConfig, video_stem: str) -> str:
    """按源配置的内嵌字幕带裁切（V3 P4）：罗小黑 WEB-DL 源自带烧录中文字幕，
    与成片文案轨同屏打架（lxh_p4_C2 盲看实锤「你是要」抢戏）。渲染侧裁掉
    底部字幕带；Omni 索引侧不裁（字幕是标注证据）。返回前置 crop 滤镜或空串。

    crop_top（2026-09-12 二轮）：film1 是 16:9 容器镶 2.39:1 画面（顶黑边
    12.4%），只裁底部会把顶部黑边整段保留 → 成片单边黑边；film2 原生
    3840×1616 无黑边。两源几何不同，crop_top/crop_bottom 各自按实测配。"""
    name = str(video_stem or "").split("__")[0]
    band = ((cfg.library.get("sources") or {}).get(name) or {}).get("subtitle_band") or {}
    top = float(band.get("crop_top") or 0)
    bottom = float(band.get("crop_bottom") or 0)
    if top < 0 or bottom < 0 or not 0 < top + bottom < 0.4:
        return ""
    y_expr = f"ih*{top:g}" if top else "0"
    return f"crop=iw:ih*{1 - top - bottom:g}:0:{y_expr}"


def render_cache_key(recipe: dict, asset_plan: dict, retrieval: list[dict], *,
                     canvas_width: int, canvas_height: int,
                     narrative_mode: bool,
                     subtitle_crops: list | None = None) -> dict:
    """产物缓存键（H3）：recipe/theme/槽位/选材/画布/叙事模式/源字幕裁切全量
    参与——旧逻辑只看 rendered.mp4 是否存在，改主题重跑会静默返回旧主题成片。
    subtitle_crops=[[slot_idx, crop_filter]]：裁切是 vf 链一部分，配置改了
    裁切比例而键不变 → 带字幕的旧段被静默复用（C3 看片实锤：0.12 裁不掉
    距底 16.7~21.1% 的字幕带，改 0.23 后旧缓存差点直接命中）。"""
    import hashlib

    payload = json.dumps({
        "reference": recipe.get("reference"),
        "operations": recipe.get("operations"),
        # 语义版本标记：执行策略变更（v2=叙事模式剔除参考文字层；
        # v3=文案轨 + BGM 换轨；v4=amix 终混 + 盲看输入改终版）后旧缓存失效
        "exec_policy": 4,
        "copy_cues": asset_plan.get("copy_cues"),
        "audio_mode": asset_plan.get("audio_mode"),
        "theme": asset_plan.get("theme"),
        "slots": asset_plan.get("slots"),
        "picked": [[row.get("slot_idx"),
                    (row.get("picked") or {}).get("video"),
                    (row.get("picked") or {}).get("source_start_s")]
                   for row in retrieval],
        "canvas": [canvas_width, canvas_height], "narrative": narrative_mode,
        "subtitle_crops": subtitle_crops or [],
    }, ensure_ascii=False, sort_keys=True)
    return {"sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def _final_audio_args(audio_mode: str, render_duration: float, *, bgm_volume: float,
                      mix_volume: float, has_bgm: bool
                      ) -> tuple[list[str], list[str], list[str]]:
    """终混音频参数（V4 D2 纯函数，便于单测）：→ (pre_inputs, filter_complex,
    map_args)。

    mix = 原声为主 + BGM 低混（外审三轮十节：验证靠对白成立的素材，交付
    必须保留对白——原声 volume 1.0，BGM 按 mix_volume≈0.25 垫底）。
    **amix normalize=0 必须显式**：默认 1/inputs 归一会把原声压半。
    bgm = 7682 纯文案公式（BGM 整轨替换）；bgm 缺失回退素材原声（调用方
    记 audio_fallback）。输入布局：[0]=视频（masked）[1]=原声（concat）
    [2]=BGM(loop)——原声取 concat 而非 masked，规避 mask 后端丢音轨。"""
    if audio_mode == "mix" and has_bgm:
        fade_out = max(0.0, render_duration - 1.0)
        filter_complex = (
            f"[1:a]volume=1.0[ra];"
            f"[2:a]volume={mix_volume:g},afade=t=in:st=0:d=0.5,"
            f"afade=t=out:st={fade_out:g}:d=1.0[bg];"
            f"[ra][bg]amix=inputs=2:duration=first:normalize=0[au]")
        return (["-stream_loop", "-1", "-i", "__BGM__"],
                filter_complex,
                ["-map", "0:v", "-map", "[au]"])
    if audio_mode == "bgm" and has_bgm:
        fade_out = max(0.0, render_duration - 1.0)
        return (["-stream_loop", "-1", "-i", "__BGM__"],
                "",
                ["-map", "0:v", "-map", "1:a:0", "-af",
                 f"atrim=duration={render_duration:g},volume={bgm_volume:g},"
                 f"afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out:g}:d=1.0"])
    return [], "", ["-map", "0:v", "-map", "0:a?"]


def render_canvas(cfg: AppConfig, narrative_mode: bool) -> tuple[int, int]:
    """成片画布解析：叙事模式读 library.narrative_render，编辑模式读
    generation.assemble（均可用 width/height 覆盖）；缺省一律落
    RENDER_CANVAS_DEFAULT（1920×1080 横屏，2026-09-11 规则）。"""
    section = (cfg.library.get("narrative_render") if narrative_mode
               else cfg.generation.get("assemble")) or {}
    width = int(section.get("width") or RENDER_CANVAS_DEFAULT[0])
    height = int(section.get("height") or RENDER_CANVAS_DEFAULT[1])
    return width, height


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
    narrative_mode = bool(asset_plan.get("narrative_program_required"))
    canvas_width, canvas_height = render_canvas(cfg, narrative_mode)
    cache = render_cache_key(
        recipe, asset_plan, retrieval,
        canvas_width=canvas_width, canvas_height=canvas_height,
        narrative_mode=narrative_mode,
        subtitle_crops=[[int(row["slot_idx"]),
                         source_subtitle_treatment(
                             cfg, str((row.get("picked") or {}).get("video_stem") or ""))]
                        for row in retrieval])
    cache_path = output_dir / "render_cache.json"
    if final.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            cached = {}
        if cached.get("sha256") == cache["sha256"]:
            return final
        # 键不匹配 → 旧产物属于另一 recipe/theme/画布：继续重渲覆盖（H3）
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
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
            seg_filter, skipped_ops = segment_filter(
                active, width=canvas_width, height=canvas_height, duration_s=duration,
                focus_x=float(picked.get("focus_x", 0.5)),
                slot_start=float(slot["start_s"]), slot_end=float(slot["end_s"]))
            runtime_status.extend(skipped_ops)
            subtitle_crop = source_subtitle_treatment(
                cfg, str(picked.get("video_stem") or ""))
            vf_chain = f"{subtitle_crop},{seg_filter}" if subtitle_crop else seg_filter
            args.extend(["-vf", vf_chain,
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
    use_bgm = False
    audio_mode = "source"              # 叙事分支按 asset_plan.audio_mode 覆盖
    applied_audio_mode = "source"      # V4：manifest 记实际生效的终混模式
    if narrative_mode:
        subtitles = write_story_subtitles(asset_plan, retrieval, output_dir / "subtitles.srt")
        if subtitles.stat().st_size:
            subtitle_filter = _subtitle_filter(subtitles)
            filter_chain = (subtitle_filter if filter_chain == "null"
                            else f"{filter_chain},{subtitle_filter}")
        # 终混（V4 D2）：mix = 原声为主 + BGM 低混（对白承载内容的槽交付
        # 必须保留对白）；bgm = 7682 纯文案公式；bgm 缺失回退素材音轨并在
        # manifest 记 audio_fallback（不崩整跑）。
        render_cfg = cfg.library.get("narrative_render") or {}
        bgm_setting = str(render_cfg.get("bgm_path") or "")
        bgm_path = (repo_root() / bgm_setting if bgm_setting and not Path(bgm_setting).is_absolute()
                    else Path(bgm_setting) if bgm_setting else None)
        has_bgm = bgm_path is not None and bgm_path.exists()
        audio_mode = str(asset_plan.get("audio_mode") or "source")
        applied_audio_mode = (audio_mode if audio_mode in {"mix", "bgm"} and has_bgm
                              else "source")
        pre_inputs, filter_complex, map_args = _final_audio_args(
            audio_mode, render_duration,
            bgm_volume=float(render_cfg.get("bgm_volume", 0.9)),
            mix_volume=float(render_cfg.get("bgm_mix_volume", 0.25)),
            has_bgm=has_bgm)
        if audio_mode in {"mix", "bgm"} and not has_bgm:
            runtime_status.append({"operation_id": "audio", "status": "audio_fallback",
                                   "reason": "bgm missing"})
        if audio_mode == "mix" and has_bgm:
            # mix 三输入：[0]=masked(视频) [1]=concat(原声) [2]=bgm(loop)
            # -vf（视频滤镜链/字幕烧录）与 -filter_complex（音频终混）并存
            args = (["-y", "-loglevel", "error", "-i", str(masked_video),
                     "-i", str(concat_video), *pre_inputs,
                     "-vf", filter_chain,
                     "-filter_complex", filter_complex, *map_args,
                     "-t", f"{render_duration:g}", "-c:v", "libx264",
                     "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", str(final)])
        else:
            pre_inputs = [str(a) if a != "__BGM__" else str(bgm_path)
                          for a in pre_inputs] if has_bgm else []
            args = (["-y", "-loglevel", "error", "-i", str(masked_video), *pre_inputs,
                     "-vf", filter_chain, *map_args,
                     "-t", f"{render_duration:g}", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", str(final)])
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
        "audio": {"mode": applied_audio_mode,
                  "fallback": bool(audio_mode in {"mix", "bgm"} and applied_audio_mode == "source")},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return final
