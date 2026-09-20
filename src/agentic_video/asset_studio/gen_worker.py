# -*- coding: utf-8 -*-
"""M2-A 图像生成 worker：每后端一个常驻子进程独占一卡。

协议：stdin 每行一个 JSON 任务 → stdout 每行一个 JSON 结果。
role=t2i  → QwenImagePipeline（Hero Master 文生图）
role=edit → QwenImageEditPlusPipeline（Edit-2511 身份保持编辑）
显存：全量加载 ~59GB > A6000 49GB → enable_model_cpu_offload
（文本编码器/VAE 用时上卡，denoise 时只有 transformer 常驻）。

用法：python -m src.agentic_video.asset_studio.gen_worker <role> <gpu> [model_id]
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "t2i"
    gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
    model_id = (sys.argv[3] if len(sys.argv) > 3 else
                "Qwen/Qwen-Image" if role == "t2i"
                else "Qwen/Qwen-Image-Edit-2511")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    import torch  # noqa: F401 （须在 diffusers 前）

    print(json.dumps({"event": "loading", "role": role, "gpu": gpu,
                      "model": model_id}), flush=True)
    pipe = _load(role, model_id)
    print(json.dumps({"event": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        if task.get("cmd") == "shutdown":
            break
        try:
            out = _run(pipe, role, task)
            print(json.dumps({"id": task.get("id"), "ok": True,
                              **out}), flush=True)
        except Exception as exc:  # noqa: BLE001 — worker 必须活过单任务失败
            print(json.dumps({"id": task.get("id"), "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}),
                  flush=True)


def _load(role: str, model_id: str):
    import torch
    from diffusers import QwenImagePipeline

    dtype = torch.bfloat16
    if role == "t2i":
        pipe = QwenImagePipeline.from_pretrained(
            model_id, torch_dtype=dtype)
    else:
        from diffusers import QwenImageEditPlusPipeline
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    return pipe


def _run(pipe, role: str, task: dict) -> dict:
    from src.agentic_video.asset_studio.image_io import save_image

    prompt = str(task["prompt"])
    width = int(task.get("width") or 1152)
    height = int(task.get("height") or 2048)
    seed = int(task.get("seed") or 0)
    out_path = str(task["out_path"])

    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = {"prompt": prompt, "width": width, "height": height,
              "generator": generator, "num_inference_steps":
                  int(task.get("steps") or 30)}
    if role != "t2i":
        kwargs["image1"] = task["reference_path"]

    image = pipe(**kwargs).images[0]
    save_image(image, out_path)
    return {"path": out_path, "width": image.width, "height": image.height,
            "seed": seed}


if __name__ == "__main__":
    main()
