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
             start_s: float, end_s: float) -> Path:
    """切片段（重编码保证帧精确）；已存在且时长匹配则复用。"""
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
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
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
    def _build_inputs(self, video_path: Path, prompt: str):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},   # 文本放多模态之后
                {"type": "text", "text": prompt},
            ],
        }]
        fps = float(self.cfg.get("fps", 2.0))
        if self.cfg.get("use_qwen_omni_utils"):
            # 官方 README 路径（逐字对齐；三处 use_audio_in_video=True 必须一致）
            from qwen_omni_utils import process_mm_info

            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self._processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True,
            )
        else:
            inputs = self._processor.apply_chat_template(
                conversation,
                load_audio_from_video=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                fps=fps,
                padding=True,
                use_audio_in_video=True,
            )
        # 官方示例原样：device + dtype 双 cast（processor 输出 float32，模型 bf16，缺 dtype 会炸 conv）
        return inputs.to(self._model.device).to(self._model.dtype)

    # ---------- 观看 ----------
    def watch(self, video_path: Path, prompt: str, *,
              start_s: float | None = None, end_s: float | None = None,
              clip_dir: Path | None = None, max_new_tokens: int | None = None,
              duration_s: float | None = None) -> OmniAnswer:
        self.load()
        clip_path = None
        t_pre0 = time.time()
        if start_s is not None and end_s is not None:
            clip_path = cut_clip(self.ffmpeg_bin, video_path, clip_dir, start_s=start_s, end_s=end_s)
            actual = clip_path
        else:
            actual = video_path
        inputs = self._build_inputs(actual, prompt)
        input_build_s = time.time() - t_pre0

        frames_est = None
        if duration_s is not None:
            frames_est = round(duration_s * float(self.cfg.get("fps", 2.0)))

        return self._generate(inputs, max_new_tokens=max_new_tokens,
                              use_audio_in_video=True, frames_estimate=frames_est,
                              clip_path=clip_path, input_build_s=input_build_s)

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
