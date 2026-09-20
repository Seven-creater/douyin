# -*- coding: utf-8 -*-
"""M2-A 后端抽象：Skill 固定接口，模型可替换工具。

AssetImageT2IBackend     Hero Master 文生图（默认 Qwen-Image）
AssetImageEditBackend    身份保持编辑（默认 Qwen-Image-Edit-2511）
MultiViewBackend         多视图（默认 Edit-2511 迭代；MV-Adapter 可换）
UpscaleBackend           4K 派生（v1 PIL Lanczos；RealESRGAN 可换）

重后端走常驻 worker 子进程（gen_worker.py，每卡一进程，
cpu_offload 解决 59GB 权重 vs 49GB A6000）。

P0-9：client 有 load/request 双超时 + stderr 环形缓冲（卡死可诊断，
traceback 不再被吞）。
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

DEFAULT_T2I_MODEL = "Qwen/Qwen-Image"
DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"


class WorkerError(RuntimeError):
    """worker 加载/执行失败（携带 stderr 尾部供诊断）。"""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(f"{message}\n[stderr tail] {stderr_tail[-800:]}")
        self.stderr_tail = stderr_tail


class _GenWorkerClient:
    """gen_worker.py 常驻子进程客户端（行 JSON 协议 + 双超时）。"""

    def __init__(self, role: str, gpu: str, model_id: str,
                 python_bin: str | None = None, *,
                 load_timeout_s: float = 3600.0,
                 request_timeout_s: float = 3600.0,
                 stderr_log_path: str | Path | None = None,
                 ) -> None:
        self.role = role
        self._load_timeout = load_timeout_s
        self._request_timeout = request_timeout_s
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stderr_log = open(stderr_log_path, "a", encoding="utf-8",
                                errors="replace") if stderr_log_path \
            else None
        self._out_q: queue.Queue[dict] = queue.Queue()
        self.proc = subprocess.Popen(
            [python_bin or sys.executable, "-m",
             "src.agentic_video.asset_studio.gen_worker",
             role, gpu, model_id],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(Path(__file__).resolve().parents[3]))
        self._lock = threading.Lock()
        self._counter = 0
        self._closed = False
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._wait_ready()

    # ---- pump 线程 ----

    def _pump_stdout(self) -> None:
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    self._out_q.put(json.loads(line))
                except json.JSONDecodeError:
                    self._stderr_tail.append(f"stdout-nonjson: {line}")
        except Exception:  # noqa: BLE001
            pass

    def _pump_stderr(self) -> None:
        try:
            for line in self.proc.stderr:  # type: ignore[union-attr]
                line = line.rstrip()
                self._stderr_tail.append(line)
                if self._stderr_log:
                    self._stderr_log.write(line + "\n")
                    self._stderr_log.flush()
        except Exception:  # noqa: BLE001
            pass

    def _tail(self) -> str:
        return "\n".join(self._stderr_tail)

    # ---- 生命周期 ----

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._load_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise WorkerError(
                    f"gen_worker {self.role} exited during load "
                    f"(rc={self.proc.returncode})", self._tail())
            remaining = deadline - time.monotonic()
            try:
                item = self._out_q.get(timeout=min(5.0, max(
                    0.05, remaining)))
            except queue.Empty:
                continue
            event = item.get("event")
            if event == "ready":
                return
            # loading 心跳继续等
        self.kill()
        raise WorkerError(
            f"gen_worker {self.role} load timeout "
            f"({self._load_timeout}s)", self._tail())

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise WorkerError(f"gen_worker {self.role} closed")
            self._counter += 1
            task_id = f"{self.role}-{self._counter}"
            payload = {"id": task_id, **payload}
            assert self.proc.stdin
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise WorkerError(
                    f"gen_worker {self.role} pipe broken "
                    f"({exc})", self._tail()) from exc
            deadline = time.monotonic() + self._request_timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise WorkerError(
                        f"gen_worker {self.role} exited before "
                        f"answering (rc={self.proc.returncode})",
                        self._tail())
                remaining = deadline - time.monotonic()
                try:
                    answer = self._out_q.get(
                        timeout=min(5.0, max(0.05, remaining)))
                except queue.Empty:
                    continue
                if answer.get("event"):
                    continue
                if answer.get("id") == task_id:
                    return answer
            self.kill()
            raise WorkerError(
                f"gen_worker {self.role} request timeout "
                f"({self._request_timeout}s) task={task_id}",
                self._tail())

    def kill(self) -> None:
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.proc.stdin:
                    self.proc.stdin.write('{"cmd": "shutdown"}\n')
                    self.proc.stdin.flush()
                self.proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            finally:
                if self._stderr_log:
                    self._stderr_log.close()


class AssetImageT2IBackend:
    """Hero Master 文生图后端（默认 Qwen-Image）。"""

    def __init__(self, gpu: str = "4",
                 model_id: str = DEFAULT_T2I_MODEL,
                 python_bin: str | None = None, **worker_kw) -> None:
        self._client = _GenWorkerClient("t2i", gpu, model_id, python_bin,
                                        **worker_kw)

    def generate(self, prompt: str, out_path: str | Path, *,
                 width: int = 1152, height: int = 2048,
                 seed: int = 0, steps: int = 50,
                 negative_prompt: str | None = None,
                 true_cfg_scale: float = 4.0) -> dict[str, Any]:
        answer = self._client.request({
            "prompt": prompt, "width": width, "height": height,
            "seed": seed, "steps": steps,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "out_path": str(out_path)})
        if not answer.get("ok"):
            raise WorkerError(
                f"t2i failed: {answer.get('error')}", self._client._tail())
        return answer

    def close(self) -> None:
        self._client.close()


class AssetImageEditBackend:
    """身份保持编辑后端（默认 Qwen-Image-Edit-2511）。"""

    def __init__(self, gpu: str = "5",
                 model_id: str = DEFAULT_EDIT_MODEL,
                 python_bin: str | None = None, **worker_kw) -> None:
        self._client = _GenWorkerClient("edit", gpu, model_id, python_bin,
                                        **worker_kw)

    def edit(self, reference_path: str | Path, prompt: str,
             out_path: str | Path, *, width: int = 1152,
             height: int = 2048, seed: int = 0, steps: int = 40,
             negative_prompt: str | None = None,
             true_cfg_scale: float = 4.0,
             guidance_scale: float = 1.0) -> dict[str, Any]:
        answer = self._client.request({
            "prompt": prompt, "reference_path": str(reference_path),
            "width": width, "height": height, "seed": seed,
            "steps": steps, "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "out_path": str(out_path)})
        if not answer.get("ok"):
            raise WorkerError(
                f"edit failed: {answer.get('error')}", self._client._tail())
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
                       seed_base: int, name_prefix: str = "",
                       ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def edit_single(self, reference_path: str | Path, view: str,
                    out_path: str | Path, identity_description: str,
                    seed: int,
                    failures: list[dict[str, Any]] | None = None,
                    attempt: int = 1) -> dict[str, Any]:
        """局部修复：从指定参考图重生成单个视图（P1-2 诊断式）。"""
        raise NotImplementedError


class QwenEditIterativeMultiView(MultiViewBackend):
    """以 master 为身份参考逐视图 Edit 生成。"""

    def __init__(self, edit_backend: AssetImageEditBackend) -> None:
        self.edit_backend = edit_backend

    def generate_views(self, master_path: str | Path,
                       view_ids: list[str], out_dir: Path,
                       identity_description: str,
                       seed_base: int, name_prefix: str = ""
                       ) -> dict[str, dict[str, Any]]:
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
                    seed: int,
                    failures: list[dict[str, Any]] | None = None,
                    attempt: int = 1) -> dict[str, Any]:
        from src.agentic_video.asset_studio.prompts import (
            build_repair_view_prompt, build_view_prompt)
        prompt = build_repair_view_prompt(
            view, identity_description, failures or [], attempt) \
            if failures else build_view_prompt(view, identity_description)
        return self.edit_backend.edit(
            reference_path, prompt, out_path, seed=seed)


class MVAdapterMultiView(MultiViewBackend):
    """MV-Adapter 多视图（预留：需独立 env + SDXL 底座，配置切换用）。"""

    def generate_views(self, master_path, view_ids, out_dir,
                       identity_description, seed_base, name_prefix=""):
        raise NotImplementedError(
            "MV-Adapter backend not wired yet — switch config to "
            "qwen_edit_iterative")


class PILUpscaleBackend:
    """确定性 4K 派生（cheap_cpu；RealESRGAN 可后续替换）。

    命名诚实：这是尺寸 4K（Lanczos），不是细节增强 4K。
    """

    def upscale(self, src_path: str | Path, out_path: str | Path
                ) -> dict[str, Any]:
        from src.agentic_video.asset_studio.image_io import upscale_lanczos
        path = upscale_lanczos(src_path, out_path)
        return {"path": str(path)}
