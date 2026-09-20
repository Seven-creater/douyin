# -*- coding: utf-8 -*-
"""Asset Runtime 契约 preflight（P2-2：防 Mock 全绿、真机 API 漂移）。

不加载权重、不占 GPU——只 import pipeline 类、验 __call__ 签名、
记录版本 manifest。夜链启动 GPU 生产前必跑。

用法：python -m src.agentic_video.asset_studio.preflight [--out FILE]
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path


def check_contracts() -> tuple[list[str], dict]:
    """返回 (failures, runtime_manifest)。"""
    failures: list[str] = []
    manifest: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torch import failed: {exc}")

    try:
        import diffusers
        manifest["diffusers"] = diffusers.__version__
        from diffusers import QwenImagePipeline
        manifest["qwen_image_pipeline"] = "ok"
        t2i_sig = inspect.signature(QwenImagePipeline.__call__)
        for param in ("prompt", "negative_prompt", "true_cfg_scale"):
            if param not in t2i_sig.parameters:
                failures.append(
                    f"QwenImagePipeline.__call__ missing {param!r} "
                    f"— diffusers {diffusers.__version__} too old?")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"QwenImagePipeline contract failed: {exc}")

    try:
        from diffusers import QwenImageEditPlusPipeline
        manifest["qwen_edit_pipeline"] = "ok"
        edit_sig = inspect.signature(QwenImageEditPlusPipeline.__call__)
        # P0-1 回归：Edit 的参考图参数必须是 image（不是 image1）
        if "image" not in edit_sig.parameters:
            failures.append(
                "QwenImageEditPlusPipeline.__call__ missing 'image' "
                "parameter — worker adapter contract broken")
        if "image1" in edit_sig.parameters:
            failures.append(
                "QwenImageEditPlusPipeline.__call__ has 'image1' — "
                "unexpected API, verify worker adapter")
        for param in ("prompt", "true_cfg_scale", "guidance_scale"):
            if param not in edit_sig.parameters:
                failures.append(
                    f"QwenImageEditPlusPipeline.__call__ missing "
                    f"{param!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"QwenImageEditPlusPipeline contract failed: {exc}")

    try:
        import transformers
        manifest["transformers"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        failures.append(f"transformers import failed: {exc}")

    manifest["models"] = {"t2i": "Qwen/Qwen-Image",
                          "edit": "Qwen/Qwen-Image-Edit-2511"}
    return failures, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runtime_manifest.json")
    args = parser.parse_args()
    failures, manifest = check_contracts()
    manifest["contract_failures"] = failures
    Path(args.out).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1),
        encoding="utf-8")
    if failures:
        print("PREFLIGHT FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("PREFLIGHT PASS")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
