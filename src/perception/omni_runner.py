"""omni_runner：Qwen3-Omni thinker-only 加载一次、进程内复用、token/显存记账、clip 切片。

所有 torch/transformers import 都在 load()/watch() 内（延迟），
本地无 GPU 栈也能 import 本模块做单测。

输入构造两条路径（S6 闸门定默认值）：
- 原生（默认）：processor.apply_chat_template(load_audio_from_video=True, ..., use_audio_in_video=True)
- 备选（use_qwen_omni_utils=true）：qwen_omni_utils.process_mm_info 三步
  （process_mm_info / processor / generate 三处 use_audio_in_video 必须一致）
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.perception import common

logger = logging.getLogger(__name__)


@dataclass
class OmniAnswer:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_s: float                      # 仅 generate 计时
    vram_peak_gb: list[float] = field(default_factory=list)
    vram_reserved_gb: list[float] = field(default_factory=list)
    frames_estimate: int | None = None    # round(duration × fps)，参考量
    clip_path: Path | None = None
    input_build_s: float = 0.0            # 预处理耗时（切片/抽帧/编码）


def set_visible_gpus(spec: str) -> None:
    """必须在 import torch 之前调用（CLI 入口最先执行）。"""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = spec


def cut_clip(ffmpeg_bin: str, video_path: Path, clip_dir: Path, *,
             start_s: float, end_s: float, max_width: int = 1920) -> Path:
    """切片段（重编码保证帧精确）；已存在且时长匹配则复用。

    分辨率策略（用户拍板 2026-09-11 晚：1K/1080p）：2160p 原生切片视觉 token
    会把 2 卡 Omni 顶到 46GB OOM（当日实锤）；1080p 在 gc.collect + 窗间
    empty_cache 修复后可稳定容纳，角色识别/画面小字远比 720p 清晰。
    1080p 及以下源不受影响（scale=min 不放大）。5.1 声源必须下混立体声
    （同日实锤：杜比/DTS 6 声道让 librosa audioread reshape 直接 ValueError）。
    """
    if clip_dir is None:
        raise ValueError("clip_dir is required when start_s/end_s are provided")
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip = clip_dir / "clip.mp4"
    if clip.exists():
        try:
            dur = common.video_duration_s("ffprobe", clip)
            if abs(dur - (end_s - start_s)) <= 0.3:
                return clip
        except common.FFmpegError:
            pass
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-ss", f"{start_s}", "-to", f"{end_s}", "-i", str(video_path),
        "-vf", f"scale='min({max_width},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-ac", "2", "-c:a", "aac", "-movflags", "+faststart",
        str(clip),
    ], timeout_s=300)
    return clip


class OmniRunner:
    """thinker-only 加载一次、进程内复用。"""

    def __init__(self, omni_cfg: dict, *, ffmpeg_bin: str = "ffmpeg"):
        self.cfg = omni_cfg
        self.ffmpeg_bin = ffmpeg_bin
        self._model = None
        self._processor = None
        self._load_elapsed: float | None = None

    # ---------- 加载 ----------
    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        model_path = self.cfg.get("model_path")
        t0 = time.time()
        logger.info("[omni] 加载 thinker-only: %s", model_path)
        self._model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=self.cfg.get("dtype", "auto"), device_map=self.cfg.get("device_map", "auto"),
        )
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        max_pixels = self.cfg.get("max_pixels")
        if max_pixels:
            self._processor.max_pixels = int(max_pixels)
        self._load_elapsed = time.time() - t0
        logger.info("[omni] 加载完成 %.1fs", self._load_elapsed)

    @property
    def load_elapsed_s(self) -> float | None:
        return self._load_elapsed

    # ---------- 输入构造 ----------
    @staticmethod
    def _sampling_audit(video, metadata: dict, *, requested_fps: float,
                        source_origin_s: float) -> dict:
        """Convert qwen-omni-utils reader metadata into an auditable frame list."""
        indices = metadata.get("frames_indices")
        if indices is None:
            indices = []
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        indices = [int(value) for value in indices]
        try:
            frame_count = int(video.shape[0])
            raw_fps = float(metadata.get("fps"))
            total_frames = int(metadata.get("total_num_frames"))
        except (AttributeError, TypeError, ValueError):
            frame_count, raw_fps, total_frames = 0, 0.0, 0
        relative = [round(value / raw_fps, 6) for value in indices] if raw_fps > 0 else []
        absolute = [round(float(source_origin_s) + value, 6) for value in relative]
        duration_s = total_frames / raw_fps if raw_fps > 0 else 0.0
        effective_fps = frame_count / duration_s if duration_s > 0 else 0.0
        verified = (
            frame_count > 0 and frame_count == len(indices) == len(relative) and
            raw_fps > 0 and total_frames >= frame_count and
            all(0 <= value < total_frames for value in indices) and
            all(math.isfinite(value) and 0 <= value <= duration_s + 1e-6
                for value in relative)
        )
        return {
            "requested_fps": float(requested_fps),
            "effective_fps": round(effective_fps, 6),
            "raw_fps": round(raw_fps, 6),
            "total_input_frames": total_frames,
            "actual_frame_count": frame_count,
            "actual_frame_indices": indices,
            "actual_frame_timestamps_relative_s": relative,
            "actual_frame_timestamps_absolute_s": absolute,
            "source_time_origin_s": float(source_origin_s),
            "video_backend": metadata.get("video_backend"),
            "sampling_verified": verified,
        }

    def _build_inputs(self, video_path: Path, prompt: str, *, fps: float | None = None,
                      source_origin_s: float = 0.0):
        requested_fps = float(fps if fps is not None else self.cfg.get("fps", 2.0))
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path),
                 "fps": requested_fps},   # 文本放多模态之后
                {"type": "text", "text": prompt},
            ],
        }]
        if self.cfg.get("use_qwen_omni_utils"):
            # Read video once with metadata.  ``fps`` belongs to the video
            # element, not process_mm_info's function signature.  Passing the
            # already sampled tensor with do_sample_frames=False prevents a
            # second, invisible sampling pass in the Transformers processor.
            try:
                from qwen_omni_utils import process_audio_info, process_vision_info

                audios = process_audio_info(conversation, use_audio_in_video=True)
                images, video_records = process_vision_info(
                    conversation, return_video_metadata=True)
                videos = []
                audits = []
                for record in video_records or []:
                    if not isinstance(record, tuple) or len(record) != 2:
                        raise ValueError("qwen video metadata was not returned")
                    video, metadata = record
                    videos.append(video)
                    audits.append(self._sampling_audit(
                        video, metadata, requested_fps=requested_fps,
                        source_origin_s=source_origin_s))
                if len(videos) != 1 or len(audits) != 1:
                    raise ValueError("OmniRunner expects exactly one audited video")
                self._last_sampling = audits[0]
                processor_fps = float(audits[0]["effective_fps"] or requested_fps)
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                from qwen_omni_utils import process_mm_info

                audios, images, videos = process_mm_info(
                    conversation, use_audio_in_video=True)
                processor_fps = requested_fps
                self._last_sampling = {
                    "requested_fps": requested_fps,
                    "sampling_verified": False,
                    "sampling_error": f"{type(exc).__name__}: {exc}",
                }
            text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self._processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True,
                do_sample_frames=False, fps=processor_fps,
            )
        else:
            inputs = self._processor.apply_chat_template(
                conversation,
                load_audio_from_video=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                fps=requested_fps,
                padding=True,
                use_audio_in_video=True,
            )
            self._last_sampling = {
                "requested_fps": requested_fps,
                "sampling_verified": False,
                "sampling_error": "native_processor_path_has_no_frame_metadata",
            }
        # 官方示例原样：device + dtype 双 cast（processor 输出 float32，模型 bf16，缺 dtype 会炸 conv）
        return inputs.to(self._model.device).to(self._model.dtype)

    def _build_media_inputs(self, image_paths: list[Path], prompt: str, *,
                            video_path: Path | None = None,
                            fps: float | None = None,
                            source_origin_s: float = 0.0):
        """Build an audited image-only or images-plus-video Omni request.

        V7 identity checks need full frames, deterministic ROI crops and album
        anchors in the same request.  This path intentionally requires
        qwen-omni-utils: the native processor path does not expose the sampled
        video tensor/metadata needed by the audit contract.
        """
        if not self.cfg.get("use_qwen_omni_utils"):
            raise ValueError("audited multi-image input requires use_qwen_omni_utils=true")
        if not image_paths and video_path is None:
            raise ValueError("at least one image or one video is required")
        requested_fps = float(fps if fps is not None else self.cfg.get("fps", 2.0))
        content = [{"type": "image", "image": str(Path(path))}
                   for path in image_paths]
        if video_path is not None:
            content.append({"type": "video", "video": str(Path(video_path)),
                            "fps": requested_fps})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]

        from qwen_omni_utils import process_audio_info, process_vision_info

        use_audio = video_path is not None
        audios = (process_audio_info(conversation, use_audio_in_video=True)
                  if use_audio else None)
        images, video_records = process_vision_info(
            conversation, return_video_metadata=True)
        videos = []
        processor_fps = requested_fps
        if video_path is not None:
            audits = []
            for record in video_records or []:
                if not isinstance(record, tuple) or len(record) != 2:
                    raise ValueError("qwen video metadata was not returned")
                video, metadata = record
                videos.append(video)
                audits.append(self._sampling_audit(
                    video, metadata, requested_fps=requested_fps,
                    source_origin_s=source_origin_s))
            if len(videos) != 1 or len(audits) != 1:
                raise ValueError("OmniRunner expects exactly one audited video")
            if not audits[0].get("sampling_verified"):
                raise ValueError("sampling_audit_failed")
            self._last_sampling = audits[0]
            processor_fps = float(audits[0]["effective_fps"] or requested_fps)
        else:
            self._last_sampling = None

        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        kwargs = {
            "text": text,
            "audio": audios,
            "images": images,
            "videos": videos or None,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_audio,
            "do_sample_frames": False,
        }
        if video_path is not None:
            kwargs["fps"] = processor_fps
        inputs = self._processor(**kwargs)
        return inputs.to(self._model.device).to(self._model.dtype), use_audio

    # ---------- 观看 ----------
    def watch(self, video_path: Path, prompt: str, *,
              start_s: float | None = None, end_s: float | None = None,
              clip_dir: Path | None = None, max_new_tokens: int | None = None,
              duration_s: float | None = None,
              fps: float | None = None) -> OmniAnswer:
        self.load()
        clip_path = None
        t_pre0 = time.time()
        if start_s is not None and end_s is not None:
            clip_path = cut_clip(self.ffmpeg_bin, video_path, clip_dir, start_s=start_s, end_s=end_s)
            actual = clip_path
        else:
            actual = video_path
        requested_fps = float(fps if fps is not None else self.cfg.get("fps", 2.0))
        inputs = self._build_inputs(
            actual, prompt, fps=requested_fps,
            source_origin_s=float(start_s or 0.0))
        input_build_s = time.time() - t_pre0

        frames_est = None
        if duration_s is not None:
            frames_est = round(duration_s * requested_fps)
        self._last_sampling = {**(getattr(self, "_last_sampling", {}) or {}),
                               "frames_estimate": frames_est,
                               "actual_sampled_frames": (
                                   getattr(self, "_last_sampling", {}) or {}
                               ).get("actual_frame_count"),
                               "estimated": False}

        return self._generate(inputs, max_new_tokens=max_new_tokens,
                              use_audio_in_video=True, frames_estimate=frames_est,
                              clip_path=clip_path, input_build_s=input_build_s)

    def inspect_media(self, image_paths: list[Path], prompt: str, *,
                      video_path: Path | None = None,
                      fps: float | None = None,
                      source_origin_s: float = 0.0,
                      max_new_tokens: int | None = None) -> OmniAnswer:
        """Inspect images, optionally together with one already-cut video."""
        self.load()
        t_pre0 = time.time()
        inputs, use_audio = self._build_media_inputs(
            [Path(path) for path in image_paths], prompt,
            video_path=Path(video_path) if video_path is not None else None,
            fps=fps, source_origin_s=source_origin_s)
        input_build_s = time.time() - t_pre0
        return self._generate(
            inputs, max_new_tokens=max_new_tokens,
            use_audio_in_video=use_audio, input_build_s=input_build_s)

    # ---------- 纯文本推理（Phase 3 模板抽取用） ----------
    def ask(self, prompt: str, *, max_new_tokens: int | None = None) -> OmniAnswer:
        """纯文本问答。注意：chat template 在 processor 上（tokenizer.chat_template 未设，
        2026-09-07 冒烟实测）；纯文本消息无多模态占位符，不经过 audio 占位符替换的坑路径。"""
        self.load()
        t_pre0 = time.time()
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        # BatchEncoding.to(dtype) 只转浮点张量（int 的 input_ids 不动）——S6 已实证安全
        inputs = inputs.to(self._model.device).to(self._model.dtype)
        input_build_s = time.time() - t_pre0
        try:
            return self._generate(inputs, max_new_tokens=max_new_tokens,
                                  input_build_s=input_build_s)
        except TypeError:
            # 5.8.0 若 generate 强制要求 use_audio_in_video，补传 False 重试一次
            return self._generate(inputs, max_new_tokens=max_new_tokens,
                                   use_audio_in_video=False, input_build_s=input_build_s)

    # ---------- 共享生成尾部 ----------
    def _generate(self, inputs, *, max_new_tokens: int | None = None,
                  use_audio_in_video: bool | None = None, frames_estimate: int | None = None,
                  clip_path: Path | None = None, input_build_s: float = 0.0) -> OmniAnswer:
        import torch

        input_len = inputs["input_ids"].shape[1]
        torch.cuda.reset_peak_memory_stats()
        max_new = int(max_new_tokens or self.cfg.get("max_new_tokens", 2048))
        gen_kwargs = dict(max_new_tokens=max_new,
                          repetition_penalty=float(self.cfg.get("repetition_penalty", 1.05)))
        if use_audio_in_video is not None:
            gen_kwargs["use_audio_in_video"] = use_audio_in_video
        t0 = time.time()
        generated = self._model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        text = self._processor.batch_decode(
            generated[:, input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        vram_peak, vram_reserved = [], []
        for i in range(torch.cuda.device_count()):
            vram_peak.append(round(torch.cuda.max_memory_allocated(i) / 2**30, 2))
            vram_reserved.append(round(torch.cuda.memory_reserved(i) / 2**30, 2))

        return OmniAnswer(
            text=text,
            input_tokens=int(input_len),
            output_tokens=int(generated.shape[1] - input_len),
            elapsed_s=round(elapsed, 2),
            vram_peak_gb=vram_peak,
            vram_reserved_gb=vram_reserved,
            frames_estimate=frames_estimate,
            clip_path=clip_path,
            input_build_s=round(input_build_s, 2),
        )
