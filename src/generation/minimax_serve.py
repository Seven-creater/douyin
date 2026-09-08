"""MiniMax-H3 统一图常驻推理服务（t2va 纯文本 + fl2va 首帧锚定，vendored 自 test/minimax_h3/serve.py）。

与原 t2va 服务的差异：/generate 额外接受 image（服务器本地图片路径）作为生成视频的
**首帧关键帧**——用于跨片段主体一致性（前一片段末帧 → 后一片段首帧）。统一图
（MiniMaxH3Blocks 本体）按输入自动路由：image=None 走 t2va，带 image 走 fl2va。

双卡配方与原版一致：text_encoder(+before_encode)→cuda:1，transformer/VAE→cuda:0，
int8 量化 + auto_cpu_offload，全部组件只加载一份。

启动：/data02/usr/wangqihao/miniconda3/envs/h3/bin/python -m src.generation.minimax_serve \
        --host 0.0.0.0 --port 8304  （服务器仓库根目录下；卡对由 CUDA_VISIBLE_DEVICES 指定）

数据流（2026-09-08 diffusers 0.40 实测窥探）：
    before_encode(image,last_image,h,w) → keyframes, keyframe_anchors
    text_encoder(prompt, keyframes?)    → prompt_embeds, text_token_tags
    vae_encoder(keyframes)              → condition_latents   ┐
    denoise + decode                    → video + audio       ┴ _rest（cuda:0）
"""
from __future__ import annotations

import argparse
import queue
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import uvicorn
from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.utils.export_utils import encode_video
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchao.quantization import Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig

REPO = "/data02/usr/wangqihao/Demo/checkpoints/MiniMax-H3-diffusers"
OUTDIR = "/data02/usr/wangqihao/Demo/test/minimax_h3/results"
FPS = 24
MIN_FRAMES = 124
MAX_SIDE = 1344


def int8_transformer():
    return MiniMaxH3Transformer3DModel.from_pretrained(
        REPO, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(),
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                "token_refiner", "norm_out", "proj_out", "audio_proj_out",
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
    """子管线返回值按名取字段（兼容 dict / 属性两种形态）。"""
    if isinstance(result, dict):
        return result[name]
    return getattr(result, name)


class Engine:
    """统一图（MiniMaxH3Blocks 本体）：单份组件，按输入自动路由 t2va（纯文本）/ fl2va（首帧锚定）。

    前身双图版在 cuda:1 放了两个 text_encoder，auto_cpu_offload 换页不彻底，
    第一个锚定任务后 47.4GB 全满 → 后续全 OOM（2026-09-08 实测）。统一图只加载
    一份 text_encoder/transformer/VAE，内存形态等价于已验证可行的单 fl2va serve。
    """

    def __init__(self):
        self.ready = threading.Event()
        self.error = None
        self.load_s = None
        self._prep = None      # before_encode（image→keyframes，轻量预处理）
        self._cond = None      # 唯一 text_encoder（keyframes 有无决定 t2va/fl2va 编码路径）
        self._rest = None      # 统一去噪/解码图（内含 t2va/fl2va 路由）

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
            self._rest.update_components(transformer=int8_transformer())
            self._rest.load_components(dtype=torch.bfloat16)
            self._rest.transformer.requires_grad_(False)

            self.load_s = time.time() - t0
            print(f"[serve] unified-graph (t2va/fl2va) ready in {self.load_s:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
            print(f"[serve] LOAD FAILED: {self.error}", flush=True)
            raise
        finally:
            self.ready.set()

    def generate(self, prompt, num_frames, height, width, seed, image):
        """返回 (results, 分段计时)。image 为 PIL.Image 或 None。只允许 worker 线程调用。"""
        t0 = time.time()
        if image is not None:
            prep = self._prep(image=image, last_image=None, height=height, width=width)
            keyframes = _field(prep, "keyframes")
            anchors = _field(prep, "keyframe_anchors")
            out_h = _field(prep, "height") or height
            out_w = _field(prep, "width") or width
            state = self._cond(prompt=prompt, keyframes=keyframes)
            t_enc = time.time()
            results = self._rest(
                state=state, keyframes=keyframes, keyframe_anchors=anchors,
                height=out_h, width=out_w,
                num_frames=num_frames, generator=torch.Generator().manual_seed(seed),
                output=["videos", "audio", "sampling_rate"],
            )
        else:
            state = self._cond(prompt=prompt)
            t_enc = time.time()
            results = self._rest(
                state=state, num_frames=num_frames, height=height, width=width,
                generator=torch.Generator().manual_seed(seed),
                output=["videos", "audio", "sampling_rate"],
            )
        return results, {"encode_s": round(t_enc - t0, 1),
                         "denoise_decode_s": round(time.time() - t_enc, 1)}


def _has(result, name):
    if isinstance(result, dict):
        return name in result
    return hasattr(result, name)


engine = Engine()
job_queue: "queue.Queue[str]" = queue.Queue()
JOBS: dict = {}
jobs_lock = threading.Lock()


class GenRequest(BaseModel):
    prompt: str
    image: str | None = None      # 服务器本地图片路径（首帧锚定；不传=纯文本）
    num_frames: int = MIN_FRAMES
    height: int = 544
    width: int = 960
    seed: int | None = None
    output: str | None = None


def validate(req: GenRequest):
    if not req.prompt.strip():
        raise HTTPException(422, "prompt 不能为空")
    if (req.num_frames - 5) % 17 != 0 or req.num_frames < MIN_FRAMES:
        raise HTTPException(422, f"num_frames 必须满足 17n+5 且 ≥{MIN_FRAMES}")
    for name, v in (("height", req.height), ("width", req.width)):
        if v % 32 != 0 or not 256 <= v <= MAX_SIDE:
            raise HTTPException(422, f"{name} 必须是 32 的倍数且在 256~{MAX_SIDE}")
    if req.image is not None:
        from pathlib import Path
        p = Path(req.image)
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            raise HTTPException(422, f"image 不是存在的图片文件: {req.image}")
    if req.output is not None:
        name = re.sub(r"[^\w.-]", "_", req.output.strip())
        if not name or name.startswith("."):
            raise HTTPException(422, "output 不是合法文件名")
        req.output = name if name.endswith(".mp4") else f"{name}.mp4"


def worker():
    engine.ready.wait()
    while True:
        job_id = job_queue.get()
        if engine.error:
            _finish(job_id, status="failed", error=f"model load failed: {engine.error}")
            continue
        with jobs_lock:
            job = JOBS.get(job_id)
            job.update(status="running", started_at=_now())
        try:
            image = None
            if job["params"]["image"]:
                from PIL import Image
                image = Image.open(job["params"]["image"]).convert("RGB")
            results, timings = engine.generate(
                job["params"]["prompt"], job["params"]["num_frames"],
                job["params"]["height"], job["params"]["width"],
                job["params"]["seed"], image,
            )
            out_path = f"{OUTDIR}/{job['params']['output']}"
            encode_video(
                results["videos"][0], fps=FPS, output_path=out_path,
                audio=results["audio"][0], audio_sample_rate=results["sampling_rate"],
            )
            video = results["videos"][0]
            if hasattr(video, "shape"):
                n_frames, h, w = (int(v) for v in video.shape[:3])
            else:
                n_frames = len(video)
                first = video[0]
                h, w = (first.shape[0], first.shape[1]) if hasattr(first, "shape") else (first.height, first.width)
            torch.cuda.empty_cache()
            _finish(
                job_id, status="done", video=out_path,
                frames=n_frames, height=int(h), width=int(w),
                duration_s=round(n_frames / FPS, 2), timings=timings,
                anchored=bool(job["params"]["image"]),
            )
        except Exception as exc:  # noqa: BLE001
            torch.cuda.empty_cache()
            _finish(job_id, status="failed", error=repr(exc))


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
    title="MiniMax-H3 fl2va",
    description="文生视频+原生音频，支持首帧关键帧锚定（跨片段一致性），双卡 int8 常驻服务",
    lifespan=lifespan,
)


@app.post("/generate", status_code=202)
def generate(req: GenRequest):
    validate(req)
    if not engine.ready.is_set():
        raise HTTPException(503, "模型还在加载中，稍后再试（看 /health）")
    if engine.error:
        raise HTTPException(503, f"模型加载失败：{engine.error}")
    seed = req.seed if req.seed is not None else secrets.randbelow(2**31)
    output = req.output or f"fl2va_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4"
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id, "status": "queued",
            "submitted_at": _now(), "started_at": None, "finished_at": None,
            "params": {"prompt": req.prompt, "image": req.image, "num_frames": req.num_frames,
                       "height": req.height, "width": req.width, "seed": seed, "output": output},
            "video": None, "frames": None, "duration_s": None, "timings": None,
            "anchored": None, "error": None,
        }
    job_queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "poll": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    return job


@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        return list(JOBS.values())


@app.get("/health")
def health():
    queued = job_queue.qsize()
    running = any(j["status"] == "running" for j in JOBS.values()) if engine.ready.is_set() else False
    return {
        "status": "loading" if not engine.ready.is_set() else ("failed" if engine.error else "ready"),
        "workflow": "fl2va", "model_load_s": engine.load_s,
        "queued": queued, "busy": running,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MiniMax-H3 fl2va serve（首帧锚定）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8304)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
