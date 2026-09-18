"""MiniMax-H3 统一图常驻推理服务——官方 SGLang /v1/videos 契约的 diffusers 实现。

背景（2026-09-18 LT0 execute 实测）：sglang main 钉死 cu13 栈而服务器驱动
575.57.08 上限 CUDA 12.9 → SGLang 路线驱动级阻断；diffusers 0.40（h3 env，
Phase-4 已实证 FL2VA 双卡 int8 配方）的模块化图原生支持 ref2va（references
触发 transformer_ref 分区）。本服务把官方 wire 契约（POST /v1/videos →
{id}；GET /v1/videos/{id} → status；GET /v1/videos/{id}/content → 视频字节）
翻译到 diffusers 统一图，使 SGLangH3Client / LT0 实验零改动。

复用：minimax_serve.py 的统一图装载/分卡/int8/队列骨架与官方 wire 契约
（sgl-project/sglang MiniMax-H3 cookbook）。

数据流（diffusers 0.40 modular blocks 实测文档）：
    before_encode(references, height, width) → normalized_references(+h/w/frames)
    text_encoder(prompt, references)         → state(prompt_embeds, text_token_tags)
    _rest(state, normalized_references, ...) → denoise(transformer_ref) + decode
卡分配沿用实证配方：prep+text_encoder→cuda:1，transformer(+ref)/VAE→cuda:0，
int8 + auto_cpu_offload，组件各一份（transformer_ref 仅 ref2va 请求换页使用）。

启动：/data02/usr/wangqihao/miniconda3/envs/h3/bin/python \
        -m src.generation.minimax_h3_ref2va_serve --host 0.0.0.0 --port 30011
（仓库根目录；卡对由 CUDA_VISIBLE_DEVICES 指定）
"""
from __future__ import annotations

import argparse
import math
import queue
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
import uvicorn
from diffusers import (
    ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline,
    TorchAoConfig)
from diffusers.utils.export_utils import encode_video
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torchao.quantization import Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig

REPO = "/data02/usr/wangqihao/Demo/checkpoints/MiniMax-H3-diffusers"
OUTDIR = "/data02/usr/wangqihao/Demo/test/minimax_h3/results_ref2va"
FPS = 24
MIN_SECONDS, MAX_SECONDS = 5.0, 15.0
SHORT_EDGE = 768
ASPECTS = {"9:16": 9 / 16, "3:4": 3 / 4, "1:1": 1.0, "4:3": 4 / 3,
           "16:9": 16 / 9, "21:9": 21 / 9}


def int8(subfolder: str) -> MiniMaxH3Transformer3DModel:
    return MiniMaxH3Transformer3DModel.from_pretrained(
        REPO, subfolder=subfolder, dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(),
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                "time_proj", "token_refiner", "norm_out", "proj_out",
                "audio_proj_out",
            ],
        ),
    )


def int8_text_encoder():
    return Qwen3VLForConditionalGeneration.from_pretrained(
        REPO, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(),
            modules_to_not_convert=[
                "model.visual", "model.language_model.embed_tokens",
                "model.language_model.norm", "lm_head",
            ],
        ),
    )


def _field(result, name):
    if isinstance(result, dict):
        return result[name]
    return getattr(result, name)


class Engine:
    """统一图：单份组件，references 有无自动路由 t2va / ref2va。"""

    def __init__(self):
        self.ready = threading.Event()
        self.error = None
        self.load_s = None
        self._prep = None
        self._cond = None
        self._rest = None

    def load(self):
        t0 = time.time()
        try:
            blocks = ModularPipeline.from_pretrained(REPO).blocks

            cond_manager = ComponentsManager()
            cond_manager.enable_auto_cpu_offload(device="cuda:1")
            self._prep = blocks.sub_blocks.pop("before_encode").init_pipeline(
                REPO, components_manager=cond_manager)
            self._cond = blocks.sub_blocks.pop("text_encoder").init_pipeline(
                REPO, components_manager=cond_manager)
            self._cond.update_components(text_encoder=int8_text_encoder())
            self._cond.load_components(dtype=torch.bfloat16)

            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(device="cuda:0")
            self._rest = blocks.init_pipeline(REPO, components_manager=manager)
            self._rest.update_components(transformer=int8("transformer"))
            # ref2va 分区：仅 references 请求换页使用（62G int8 后常驻 CPU）
            self._rest.update_components(transformer_ref=int8("transformer_ref"))
            self._rest.load_components(dtype=torch.bfloat16)
            self._rest.transformer.requires_grad_(False)

            self.load_s = time.time() - t0
            print(f"[serve] unified-graph (t2va/ref2va) ready in "
                  f"{self.load_s:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
            print(f"[serve] LOAD FAILED: {self.error}", flush=True)
            raise
        finally:
            self.ready.set()

    def generate(self, *, prompt, references, height, width, num_frames,
                 num_inference_steps, seed):
        t0 = time.time()
        generator = torch.Generator().manual_seed(int(seed))
        prep_out = self._prep(
            references=references, height=height, width=width)
        normalized = _field(prep_out, "normalized_references")
        out_h = _field(prep_out, "height") or height
        out_w = _field(prep_out, "width") or width
        out_frames = _field(prep_out, "num_frames") or num_frames
        state = self._cond(prompt=prompt, references=references)
        t_enc = time.time()
        results = self._rest(
            state=state, normalized_references=normalized,
            height=out_h, width=out_w, num_frames=out_frames,
            generator=generator, num_inference_steps=num_inference_steps,
            output=["videos", "audio", "sampling_rate"],
        )
        return results, {"encode_s": round(t_enc - t0, 1),
                         "denoise_decode_s": round(time.time() - t_enc, 1)}


engine = Engine()
job_queue: "queue.Queue[str]" = queue.Queue()
JOBS: dict = {}
jobs_lock = threading.Lock()


class Condition(BaseModel):
    type: str
    uri: str
    role: str = "reference"
    frame_index: int | None = None


class Target(BaseModel):
    short_edge: int = SHORT_EDGE
    aspect_ratio: str = "16:9"
    duration_seconds: float | None = None


class VideoRequest(BaseModel):
    model: str = "MiniMaxAI/MiniMax-H3"
    prompt: str
    seconds: float
    task: str = "t2va"
    conditions: list[Condition] = []
    target: Target = Target()
    num_outputs_per_prompt: int = 1
    num_inference_steps: int = 50
    flow_shift: float = 12.0
    audio_flow_shift: float = 3.0
    seed: int | None = None


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme in ("", "http", "https"):
        # 官方允许 http(s)——本机离线部署只接受服务器本地路径/file URI
        raise HTTPException(422, f"仅支持 file:// URI（离线部署），得到：{uri}")
    return Path(uri)


def _snap_frames(seconds: float) -> int:
    frames = max(1, round(seconds * FPS))
    n = max(1, math.ceil((frames - 5) / 17))
    return int(17 * n + 5)


def _canvas(target: Target) -> tuple[int, int]:
    ratio = ASPECTS.get(target.aspect_ratio)
    if ratio is None:
        raise HTTPException(422, f"未知 aspect_ratio：{target.aspect_ratio}")
    short = int(target.short_edge or SHORT_EDGE)
    if ratio >= 1.0:
        width = int(math.ceil(short * ratio / 32) * 32)
        height = int(math.ceil(short / 32) * 32)
    else:
        height = int(math.ceil(short / ratio / 32) * 32)
        width = int(math.ceil(short / 32) * 32)
    return height, width


def _load_references(conditions: list[Condition]):
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3ImageReference, MiniMaxH3VideoReference)
    references = []
    first_image = None
    last_image = None
    for condition in conditions:
        path = _uri_to_path(condition.uri)
        if not path.is_file():
            raise HTTPException(422, f"条件文件不存在：{condition.uri}")
        if condition.role == "keyframe":
            from PIL import Image
            image = Image.open(path).convert("RGB")
            if condition.frame_index == -1:
                last_image = image
            else:
                first_image = first_image or image
            continue
        if condition.type == "image":
            references.append(MiniMaxH3ImageReference.from_file(str(path)))
        elif condition.type in ("video", "video_audio"):
            references.append(MiniMaxH3VideoReference.from_file(str(path)))
        else:
            raise HTTPException(422, f"不支持的条件类型：{condition.type}")
    return references, first_image, last_image


def validate(req: VideoRequest):
    if not req.prompt.strip():
        raise HTTPException(422, "prompt 不能为空")
    if not MIN_SECONDS <= req.seconds <= MAX_SECONDS:
        raise HTTPException(422, f"seconds 须在 {MIN_SECONDS}-{MAX_SECONDS}")
    if req.task == "ref2va" and not req.conditions:
        raise HTTPException(422, "ref2va 至少一个 reference 条件")
    _load_references(req.conditions)  # 前置校验文件存在性与类型


def worker():
    engine.ready.wait()
    while True:
        job_id = job_queue.get()
        if engine.error:
            _finish(job_id, status="failed",
                    error=f"model load failed: {engine.error}")
            continue
        with jobs_lock:
            job = JOBS.get(job_id)
        if job is None:
            continue
        with jobs_lock:
            JOBS[job_id].update(status="running")
        try:
            params = job["params"]
            references, first_image, last_image = _load_references(
                [Condition(**row) for row in params["conditions"]])
            height, width = params["height"], params["width"]
            num_frames = _snap_frames(params["seconds"])
            if references:
                results, timings = engine.generate(
                    prompt=params["prompt"], references=references,
                    height=height, width=width, num_frames=num_frames,
                    num_inference_steps=params["num_inference_steps"],
                    seed=params["seed"])
            else:
                # t2va / fl2va（keyframe）走原统一图路径
                prep = engine._prep(image=first_image, last_image=last_image,
                                    height=height, width=width)
                keyframes = _field(prep, "keyframes")
                anchors = _field(prep, "keyframe_anchors")
                state = (engine._cond(prompt=params["prompt"], keyframes=keyframes)
                         if keyframes else
                         engine._cond(prompt=params["prompt"]))
                t0 = time.time()
                results = engine._rest(
                    state=state, keyframes=keyframes or None,
                    keyframe_anchors=anchors or (),
                    height=height, width=width, num_frames=num_frames,
                    generator=torch.Generator().manual_seed(params["seed"]),
                    num_inference_steps=params["num_inference_steps"],
                    output=["videos", "audio", "sampling_rate"])
                timings = {"encode_s": round(t0 % 1, 1),
                           "denoise_decode_s": 0.0}
            out_dir = Path(OUTDIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{job_id}.mp4"
            encode_video(
                results["videos"][0], fps=FPS, output_path=str(out_path),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"])
            torch.cuda.empty_cache()
            _finish(job_id, status="completed", video=str(out_path),
                    frames=num_frames, height=height, width=width,
                    duration_s=round(num_frames / FPS, 2), timings=timings)
        except Exception as exc:  # noqa: BLE001
            torch.cuda.empty_cache()
            _finish(job_id, status="failed", error=repr(exc)[:500])


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _finish(job_id, **fields):
    with jobs_lock:
        JOBS[job_id].update(fields, finished_at=_now())
    print(f"[serve] job {job_id} {fields.get('status')}: "
          f"{fields.get('video') or fields.get('error')}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=engine.load, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(
    title="MiniMax-H3 unified t2va/ref2va (/v1/videos diffusers backend)",
    lifespan=lifespan,
)


@app.get("/v1/models")
def models():
    return {"data": [{"id": "MiniMaxAI/MiniMax-H3"}]}


@app.get("/health")
def health():
    return {"status": "loading" if not engine.ready.is_set()
            else ("failed" if engine.error else "ready"),
            "workflow": "t2va+ref2va", "model_load_s": engine.load_s}


@app.post("/v1/videos", status_code=202)
def create_video(req: VideoRequest):
    validate(req)
    if engine.error:
        raise HTTPException(503, f"模型加载失败：{engine.error}")
    seed = req.seed if req.seed is not None else secrets.randbelow(2 ** 31)
    height, width = _canvas(req.target)
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id, "id": job_id, "status": "queued",
            "submitted_at": _now(),
            "params": {"prompt": req.prompt,
                       "conditions": [c.model_dump() for c in req.conditions],
                       "seconds": req.seconds, "height": height,
                       "width": width, "seed": seed,
                       "num_inference_steps": req.num_inference_steps},
            "video": None, "frames": None, "duration_s": None,
            "timings": None, "error": None,
        }
    job_queue.put(job_id)
    return {"id": job_id, "status": "queued"}


@app.get("/v1/videos/{job_id}")
def get_video(job_id: str):
    with jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    return job


@app.get("/v1/videos/{job_id}/content")
def get_video_content(job_id: str):
    with jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    if job.get("status") != "completed" or not job.get("video"):
        raise HTTPException(409, f"job 未完成：{job.get('status')}")
    return FileResponse(job["video"], media_type="video/mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MiniMax-H3 /v1/videos diffusers 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=30011)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
