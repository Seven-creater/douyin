"""minimax_client：MiniMax 服务生命周期 + 多实例调度 + 生成台账。

CLI：python -m src.generation.minimax_client {status|start|stop} [--instances N]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging

logger = logging.getLogger(__name__)


class MiniMaxHTTP:
    """瘦 HTTP 客户端（可 mock）。"""

    def __init__(self, timeout: float = 10.0, session=None):
        self._session = session or requests.Session()
        self._timeout = timeout

    def health(self, base_url: str) -> dict | None:
        try:
            r = self._session.get(f"{base_url}/health", timeout=self._timeout)
            return r.json() if r.status_code == 200 else None
        except requests.RequestException:
            return None

    def submit(self, base_url: str, *, prompt: str, num_frames: int,
               width: int, height: int, seed: int, output: str) -> str:
        r = self._session.post(f"{base_url}/generate", json={
            "prompt": prompt, "num_frames": num_frames, "height": height,
            "width": width, "seed": seed, "output": output,
        }, timeout=self._timeout)
        r.raise_for_status()
        return r.json()["job_id"]

    def job(self, base_url: str, job_id: str) -> dict:
        r = self._session.get(f"{base_url}/jobs/{job_id}", timeout=self._timeout)
        r.raise_for_status()
        return r.json()


class MiniMaxService:
    def __init__(self, mm_cfg: dict):
        self.cfg = mm_cfg
        self.http = MiniMaxHTTP()
        self.serve_sh = mm_cfg.get("serve_sh")

    def base_url(self, port: int) -> str:
        return f"http://{self.cfg.get('host', '127.0.0.1')}:{port}"

    def probe(self, port: int) -> dict | None:
        h = self.http.health(self.base_url(port))
        return h if h and h.get("status") in ("ready", "loading") else None

    def start_instance(self, port: int, gpu_pair: str) -> bool:
        """直接运行 serve.py 多实例（serve.sh 是单实例设计：共享 serve.pid/log，
        第二实例 start 会因 alive() 误判跳过——2026-09-07 实测）。运行非改码。"""
        import os

        serve_dir = Path(self.serve_sh).parent
        python = self.cfg.get("python", "/data02/usr/wangqihao/miniconda3/envs/h3/bin/python")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_pair}
        log_path = Path(self.cfg.get("instance_log_dir", "/tmp")) / f"mm_instance_{port}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab")
        proc = subprocess.Popen(
            [python, "serve.py", "--host", "0.0.0.0", "--port", str(port)],
            cwd=str(serve_dir), env=env,
            stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("[minimax] serve.py pid=%d port=%d → %s", proc.pid, port, log_path)
        return True

    def preflight_gpus(self) -> tuple[list[str], list[str]]:
        """只读探测；返回 (可用卡对, 告警)。占用 >10GB 的对视为忙。"""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            used = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in out.splitlines() if "," in line}
        except (subprocess.SubprocessError, ValueError):
            return list(self.cfg.get("gpu_pairs") or []), ["nvidia-smi 探测失败，按配置顺序使用"]
        ok, warns = [], []
        for pair in self.cfg.get("gpu_pairs") or []:
            gpus = [g.strip() for g in pair.split(",")]
            busy = [g for g in gpus if used.get(g, 0) > 10_000]
            if busy:
                warns.append(f"卡对 {pair} 忙（{','.join(busy)} 占用>10GB），跳过")
            else:
                ok.append(pair)
        return ok, warns

    def ensure_instances(self, n: int) -> tuple[list[int], list[str]]:
        """探测→补启动→等就绪。返回 (就绪端口列表, 告警)。"""
        pairs_all = self.cfg.get("gpu_pairs") or []
        free_pairs, warns = self.preflight_gpus()
        # 实例数不得超过空闲卡对数：free_pairs ⊆ pairs_all，原兜底按 pairs_all[i]
        # 取值会给多余实例重复分配已占用/已分配的卡对（2026-09-07 单命令模式下
        # Omni 常驻 0,1 时实测会触发：第 4 实例与第 3 实例同抢 "6,7" → OOM）
        n_capped = min(n, int(self.cfg.get("max_instances", 4)), len(pairs_all), len(free_pairs))
        if n_capped < n:
            warns.append(f"空闲卡对不足（{len(free_pairs)}/{len(pairs_all)}），实例数 {n} → {n_capped}")
        n = n_capped
        ready: list[int] = []
        base = int(self.cfg.get("base_port", 8300))
        timeout = float(self.cfg.get("health_timeout_s", 1500))
        for i in range(n):
            port = base + i
            pair = free_pairs[i]
            h = self.probe(port)
            if h and h.get("status") == "ready":
                ready.append(port)
                continue
            if h:  # loading 中
                logger.info("[minimax] 端口 %d 已在加载中（gpus=%s）", port, pair)
            else:
                logger.info("[minimax] 启动实例 port=%d gpus=%s", port, pair)
                self.start_instance(port, pair)
            # 等就绪：只信 HTTP（auto_cpu_offload 模式下空闲时 GPU 显存≈0 是正常形态，
            # 2026-09-07 实测：用"GPU 空=假就绪"判断会连环误杀健康实例）
            t0 = time.time()
            while time.time() - t0 < timeout:
                h = self.http.health(self.base_url(port))
                if h and h.get("status") == "ready":
                    ready.append(port)
                    break
                if h and h.get("status") == "failed":
                    warns.append(f"端口 {port} 服务 failed（看 serve.log）")
                    break
                time.sleep(10)
            else:
                warns.append(f"端口 {port} 等待就绪超时（{timeout:.0f}s）")
        return ready, warns

    def stop_instance(self, port: int) -> None:
        """按 cmdline 匹配停我们自己拉起的 serve.py（也顺带停 serve.sh 拉起的单实例）。"""
        subprocess.run(["pkill", "-f", f"serve.py --host 0.0.0.0 --port {port}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)


# ---------- 台账 ----------

class GenerationManifest:
    """clips 台账：is_done 只信本地 clips/uXX.mp4 存在。"""

    def __init__(self, path: Path, clips_root: Path):
        self.path = Path(path)
        self.clips_root = Path(clips_root)
        self.data: dict[str, Any] = {"version": 1, "clips": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except ValueError:
                pass

    def key(self, variant_id: str, unit_id: int) -> str:
        return f"{variant_id}/u{unit_id:02d}"

    def clip_path(self, variant_id: str, unit_id: int) -> Path:
        # 与 assemble 的 clips_dir 对齐：variants/<vid>/clips/uXX.mp4
        return self.clips_root / variant_id / "clips" / f"u{unit_id:02d}.mp4"

    def get(self, variant_id: str, unit_id: int) -> dict | None:
        return self.data["clips"].get(self.key(variant_id, unit_id))

    def is_done(self, variant_id: str, unit_id: int) -> bool:
        e = self.get(variant_id, unit_id)
        return bool(e and e.get("status") == "done" and self.clip_path(variant_id, unit_id).exists())

    def record(self, variant_id: str, unit_id: int, **row: Any) -> None:
        k = self.key(variant_id, unit_id)
        prev = self.data["clips"].get(k) or {}
        prev.update(row)
        self.data["clips"][k] = prev

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 调度 ----------

@dataclass
class ClipTask:
    variant_id: str
    unit_id: int
    prompt: str
    seed: int
    num_frames: int
    output_name: str
    # 运行态
    job_id: str | None = None
    port: int | None = None
    running_since: float | None = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


def expected_job_seconds(num_frames: int, factor: float = 2.0) -> float:
    """124f≈690s 实测线性外推 + 宽松窗。"""
    return num_frames / 124 * 690 * factor + 300


class GenerationScheduler:
    def __init__(self, ports: list[int], http: MiniMaxHTTP, manifest: GenerationManifest,
                 mm_cfg: dict, clips_root: Path, width: int, height: int):
        self.ports = list(ports)
        self.http = http
        self.manifest = manifest
        self.cfg = mm_cfg
        self.clips_root = Path(clips_root)
        self.width = int(width)
        self.height = int(height)

    def _fetch_and_copy(self, svc: MiniMaxService, task: ClipTask, job: dict) -> bool:
        src = job.get("video")
        if not src or not Path(src).exists() or Path(src).stat().st_size < 10_240:
            return False
        dst = self.manifest.clip_path(task.variant_id, task.unit_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst.exists() and dst.stat().st_size > 10_240

    def run(self, svc: MiniMaxService, tasks: list[ClipTask]) -> dict:
        poll_s = float(self.cfg.get("poll_interval_s", 20))
        max_retries = int(self.cfg.get("max_retries", 2))
        pending = [t for t in tasks if not self.manifest.is_done(t.variant_id, t.unit_id)]
        logger.info("[scheduler] %d 任务（跳过 %d 已完成）", len(pending), len(tasks) - len(pending))
        busy: dict[int, ClipTask] = {}
        metrics_path = self.clips_root.parent / "metrics.jsonl"
        t_all = time.time()

        while pending or busy:
            # 分发
            for port in list(self.ports):
                if port in busy or not pending:
                    continue
                task = pending.pop(0)
                task.attempts += 1
                try:
                    base = svc.base_url(port)
                    # seed 随尝试递增（换名防覆盖）
                    seed = task.seed + task.attempts - 1
                    output = task.output_name.replace(".mp4", f"_a{task.attempts}.mp4")
                    task.job_id = self.http.submit(base, prompt=task.prompt, num_frames=task.num_frames,
                                                   width=self.width, height=self.height,
                                                   seed=seed, output=output)
                    task.port = port
                    busy[port] = task
                    self.manifest.record(task.variant_id, task.unit_id, status="submitted",
                                         job_id=task.job_id, port=port, seed=seed,
                                         attempts=task.attempts,
                                         submitted_at=datetime.now().isoformat(timespec="seconds"))
                    self.manifest.save()
                    logger.info("[scheduler] %s/u%02d → port %d (job %s, attempt %d)",
                                task.variant_id, task.unit_id, port, task.job_id, task.attempts)
                except Exception as exc:  # noqa: BLE001
                    task.errors.append(f"submit: {exc}")
                    self._fail(task, exc, metrics_path, pending, max_retries)
            # 轮询
            for port, task in list(busy.items()):
                base = svc.base_url(port)
                try:
                    job = self.http.job(base, task.job_id)
                except requests.RequestException as exc:
                    task.errors.append(f"poll: {exc}")
                    self._fail(task, exc, metrics_path, pending, max_retries)
                    del busy[port]
                    continue
                status = job.get("status")
                if status == "running" and task.running_since is None:
                    task.running_since = time.time()
                    self.manifest.record(task.variant_id, task.unit_id,
                                         running_since=datetime.now().isoformat(timespec="seconds"))
                    self.manifest.save()
                if status == "done":
                    ok = self._fetch_and_copy(svc, task, job)
                    elapsed = round(time.time() - (task.running_since or time.time()), 1)
                    if ok:
                        self.manifest.record(task.variant_id, task.unit_id, status="done",
                                             source_path=job.get("video"),
                                             finished_at=datetime.now().isoformat(timespec="seconds"),
                                             elapsed_s=elapsed, errors=task.errors)
                        self.manifest.save()
                        logger.info("[scheduler] %s/u%02d 完成 %.0fs", task.variant_id, task.unit_id, elapsed)
                        self._metric(metrics_path, task, "ok", elapsed)
                    else:
                        self._fail(task, RuntimeError("产物缺失或过小"), metrics_path, pending, max_retries)
                    del busy[port]
                elif status == "failed":
                    self._fail(task, job.get("error") or "failed", metrics_path, pending, max_retries)
                    del busy[port]
                else:
                    # 超时只计 running 时长
                    if task.running_since and time.time() - task.running_since > expected_job_seconds(
                            task.num_frames, float(self.cfg.get("job_timeout_factor", 2.0))):
                        self._fail(task, "timeout", metrics_path, pending, max_retries)
                        del busy[port]
            if busy or pending:
                time.sleep(poll_s)

        return {"n_total": len(tasks), "n_pending_skipped": len(tasks) - len([t for t in tasks]),
                "elapsed_s": round(time.time() - t_all, 1),
                "done": sum(1 for k, e in self.manifest.data["clips"].items() if e.get("status") == "done")}

    def _fail(self, task: ClipTask, exc: Any, metrics_path: Path,
              pending: list[ClipTask], max_retries: int) -> None:
        err = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
        task.errors.append(err[:300])
        logger.warning("[scheduler] %s/u%02d 失败（attempt %d）: %s",
                       task.variant_id, task.unit_id, task.attempts, err[:200])
        self._metric(metrics_path, task, "error", None, error=err[:200])
        if task.attempts <= max_retries:
            task.job_id = None
            task.running_since = None
            pending.append(task)  # 重提（seed 随 attempts 递增）
        else:
            self.manifest.record(task.variant_id, task.unit_id, status="failed",
                                 finished_at=datetime.now().isoformat(timespec="seconds"),
                                 errors=task.errors)
            self.manifest.save()

    def _metric(self, metrics_path: Path, task: ClipTask, status: str,
                elapsed: float | None, error: str | None = None) -> None:
        from src.perception.common import append_metric

        append_metric(metrics_path, tool="generate_clip", aweme_id="", status=status,
                      elapsed_s=elapsed, extra={"variant": task.variant_id,
                                                "unit": task.unit_id, "frames": task.num_frames,
                                                "attempts": task.attempts,
                                                **({"error": error} if error else {})})


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="MiniMax 服务管理")
    ap.add_argument("action", choices=["status", "start", "stop"])
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg: AppConfig = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="gen_minimax")
    svc = MiniMaxService((cfg.generation or {}).get("minimax") or {})
    n = args.instances or int((cfg.generation or {}).get("minimax", {}).get("max_instances", 4))

    if args.action == "status":
        base = int(svc.cfg.get("base_port", 8300))
        for i in range(4):
            port = base + i
            h = svc.http.health(svc.base_url(port))
            print(f"port {port}: {h or '未启动'}")
        return 0
    if args.action == "start":
        ready, warns = svc.ensure_instances(n)
        print(f"就绪端口: {ready}")
        for w in warns:
            print(f"警告: {w}")
        return 0 if ready else 1
    if args.action == "stop":
        base = int(svc.cfg.get("base_port", 8300))
        for i in range(4):
            if svc.probe(base + i):
                svc.stop_instance(base + i)
                print(f"port {base + i}: 已停止")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
