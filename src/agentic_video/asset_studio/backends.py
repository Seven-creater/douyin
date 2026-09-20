# -*- coding: utf-8 -*-
"""M2-A 后端抽象：Skill 固定接口，模型可替换工具。

AssetImageT2IBackend     Hero Master 文生图（默认 Qwen-Image）
AssetImageEditBackend    身份保持编辑（默认 Qwen-Image-Edit-2511）
MultiViewBackend         多视图（默认 Edit-2511 迭代；MV-Adapter 可换）
UpscaleBackend           4K 超分（v1 PIL Lanczos；RealESRGAN 可换）

重后端走常驻 worker 子进程（gen_worker.py，每卡一进程，
cpu_offload 解决 59GB 权重 vs 49GB A6000）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from src.agentic_video.asset_studio.image_io import save_image

DEFAULT_T2I_MODEL = "Qwen/Qwen-Image"
DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"


class _GenWorkerClient:
    """gen_worker.py 常驻子进程客户端（行 JSON 协议）。"""

    def __init__(self, role: str, gpu: str,
                 model_id: str, python_bin: str | None = None) -> None:
        self.role = role
        self.proc = subprocess.Popen(
            [python_bin or sys.executable, "-m",
             "src.agentic_video.asset_studio.gen_worker",
             role, gpu, model_id],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(Path(__file__).resolve().parents[3]))
        self._lock = threading.Lock()
        self._counter = 0
        self._drain_stderr()

    def _drain_stderr(self) -> None:
        def _pump() -> None:
            try:
                for line in self.proc.stderr:  # type: ignore[union-attr]
                    pass  # 防塞；详细错误走 stdout 的 error 字段
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=_pump, daemon=True).start()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._counter += 1
            task_id = f"{self.role}-{self._counter}"
            payload = {"id": task_id, **payload}
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            for line in self.proc.stdout:
                answer = json.loads(line)
                if answer.get("event"):
                    continue  # loading/ready 心跳
                if answer.get("id") == task_id:
                    return answer
            raise RuntimeError(
                f"gen_worker {self.role} exited before answering "
                f"(rc={self.proc.poll()})")

    def close(self) -> None:
        with self._lock:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write('{"cmd": "shutdown"}\n')
                    self.proc.stdin.flush()
                self.proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                self.proc.kill()


class AssetImageT2IBackend:
    """Hero Master 文生图后端（默认 Qwen-Image）。"""

    def __init__(self, gpu: str = "4",
                 model_id: str = DEFAULT_T2I_MODEL,
                 python_bin: str | None = None) -> None:
        self._client = _GenWorkerClient("t2i", gpu, model_id, python_bin)

    def generate(self, prompt: str, out_path: str | Path, *,
                 width: int = 1152, height: int = 2048,
                 seed: int = 0, steps: int = 30) -> dict[str, Any]:
        answer = self._client.request({
            "prompt": prompt, "width": width, "height": height,
            "seed": seed, "steps": steps, "out_path": str(out_path)})
        if not answer.get("ok"):
            raise RuntimeError(f"t2i failed: {answer.get('error')}")
        return answer

    def close(self) -> None:
        self._client.close()


class AssetImageEditBackend:
    """身份保持编辑后端（默认 Qwen-Image-Edit-2511）。"""

    def __init__(self, gpu: str = "5",
                 model_id: str = DEFAULT_EDIT_MODEL,
                 python_bin: str | None = None) -> None:
        self._client = _GenWorkerClient("edit", gpu, model_id, python_bin)

    def edit(self, reference_path: str | Path, prompt: str,
             out_path: str | Path, *, width: int = 1152,
             height: int = 2048, seed: int = 0,
             steps: int = 30) -> dict[str, Any]:
        answer = self._client.request({
            "prompt": prompt, "reference_path": str(reference_path),
            "width": width, "height": height, "seed": seed,
            "steps": steps, "out_path": str(out_path)})
        if not answer.get("ok"):
            raise RuntimeError(f"edit failed: {answer.get('error')}")
        return answer

    def close(self) -> None:
        self._client.close()


class MultiViewBackend:
    """多视图后端（接口固定；实现可换）。

    v1 默认 QwenEditIterativeMultiView（单 env 部署 + 身份保持最优，
    face 视图本就需要 Edit）；MVAdapterMultiView 预留为可替换实现。
    """

    def generate_views(self, master_path: str | Path,
                       view_ids: list[str], out_dir: Path,
                       identity_description: str,
                       seed_base: int) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def edit_single(self, reference_path: str | Path, view: str,
                    out_path: str | Path, identity_description: str,
                    seed: int) -> dict[str, Any]:
        """局部修复：从指定参考图重生成单个视图。"""
        raise NotImplementedError


class QwenEditIterativeMultiView(MultiViewBackend):
    """以 master 为身份参考逐视图 Edit 生成。"""

    def __init__(self, edit_backend: AssetImageEditBackend) -> None:
        self.edit_backend = edit_backend

    def generate_views(self, master_path: str | Path,
                       view_ids: list[str], out_dir: Path,
                       identity_description: str,
                       seed_base: int, name_prefix: str = "") -> dict[str, dict[str, Any]]:
        from src.agentic_video.asset_studio.prompts import build_view_prompt
        results: dict[str, dict[str, Any]] = {}
        for offset, view in enumerate(view_ids):
            out_path = out_dir / f"{name_prefix}{view}_v1.png"
            answer = self.edit_backend.edit(
                master_path, build_view_prompt(view, identity_description),
                out_path, seed=seed_base + offset)
            results[view] = answer
        return results

    def edit_single(self, reference_path: str | Path, view: str,
                    out_path: str | Path, identity_description: str,
                    seed: int) -> dict[str, Any]:
        from src.agentic_video.asset_studio.prompts import build_view_prompt
        return self.edit_backend.edit(
            reference_path, build_view_prompt(view, identity_description),
            out_path, seed=seed)


class MVAdapterMultiView(MultiViewBackend):
    """MV-Adapter 多视图（预留：需独立 env + SDXL 底座，配置切换用）。"""

    def generate_views(self, master_path, view_ids, out_dir,
                       identity_description, seed_base):
        raise NotImplementedError(
            "MV-Adapter backend not wired yet — switch config to "
            "qwen_edit_iterative")


class PILUpscaleBackend:
    """确定性 4K 超分（cheap_cpu；RealESRGAN 可后续替换）。"""

    def upscale(self, src_path: str | Path, out_path: str | Path
                ) -> dict[str, Any]:
        from src.agentic_video.asset_studio.image_io import upscale_lanczos
        path = upscale_lanczos(src_path, out_path)
        return {"path": str(path)}
