"""Persistent multi-process Qwen3-Omni pool for one model replica per GPU pair.

The 30B thinker uses roughly 69 GB, so the eight A6000 server is used as four
independent two-card workers rather than one eight-card model.  Workers stay
alive across evidence mining and blind review, avoiding a model reload for
every six-second tile.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


def parse_gpu_pairs(value: str | Iterable[str]) -> list[str]:
    """Validate semicolon-separated, disjoint two-GPU worker assignments."""
    raw = value.split(";") if isinstance(value, str) else list(value)
    pairs: list[str] = []
    seen: set[str] = set()
    for item in raw:
        cards = [card.strip() for card in str(item).split(",") if card.strip()]
        if len(cards) != 2 or not all(card.isdigit() for card in cards):
            raise ValueError(f"each Omni worker requires exactly two GPUs: {item!r}")
        if len(set(cards)) != 2 or seen.intersection(cards):
            raise ValueError(f"GPU pairs must be distinct and disjoint: {value!r}")
        seen.update(cards)
        pairs.append(",".join(cards))
    if not pairs:
        raise ValueError("at least one GPU pair is required")
    return pairs


@dataclass
class PooledOmniAnswer:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    input_build_s: float = 0.0
    frames_estimate: int | None = None
    clip_path: Path | None = None
    sampling: dict[str, Any] | None = None
    gpu_pair: str | None = None


class OmniPoolError(RuntimeError):
    pass


def _worker_main(gpu_pair: str, omni_cfg: dict, ffmpeg_bin: str,
                 tasks, results, load_lock) -> None:
    from src.perception.omni_runner import OmniRunner, set_visible_gpus

    set_visible_gpus(gpu_pair)
    # Import torch before any optional cv2 import; this ordering is required on
    # the production host to avoid the observed native OpenMP crash.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    runner = OmniRunner(omni_cfg, ffmpeg_bin=ffmpeg_bin)
    loaded = False
    while True:
        item = tasks.get()
        if item is None:
            return
        task_id = str(item["task_id"])
        try:
            # Four simultaneous 30B safetensor loads can saturate the shared
            # model filesystem and have been observed to terminate every
            # worker mid-load.  Serialize only the cold load; inference stays
            # fully parallel after all four replicas are resident.
            if not loaded:
                with load_lock:
                    runner.load()
                loaded = True
            mode = str(item.get("mode") or "watch")
            kwargs = dict(item.get("kwargs") or {})
            if mode == "ask":
                answer = runner.ask(item["prompt"], **kwargs)
            else:
                if kwargs.get("clip_dir") is not None:
                    kwargs["clip_dir"] = Path(kwargs["clip_dir"])
                answer = runner.watch(Path(item["video_path"]), item["prompt"], **kwargs)
            payload = asdict(answer)
            if payload.get("clip_path") is not None:
                payload["clip_path"] = str(payload["clip_path"])
            results.put({"task_id": task_id, "ok": True, "answer": payload,
                         "sampling": (getattr(runner, "_last_sampling", None)
                                      if mode == "watch" else None),
                         "gpu_pair": gpu_pair})
        except BaseException as exc:  # worker must report OOM/parse errors to parent
            results.put({"task_id": task_id, "ok": False, "gpu_pair": gpu_pair,
                         "error": f"{type(exc).__name__}: {exc}"})


class OmniProcessPool:
    """Small persistent process pool with one Omni model per two-card pair."""

    def __init__(self, gpu_pairs: str | Iterable[str], omni_cfg: dict, *,
                 ffmpeg_bin: str = "ffmpeg", response_timeout_s: float = 3600.0):
        self.gpu_pairs = parse_gpu_pairs(gpu_pairs)
        self.omni_cfg = dict(omni_cfg)
        self.ffmpeg_bin = ffmpeg_bin
        self.response_timeout_s = float(response_timeout_s)
        self._ctx = mp.get_context("spawn")
        self._tasks = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._load_lock = self._ctx.Lock()
        self._processes = [self._ctx.Process(
            target=_worker_main,
            args=(pair, self.omni_cfg, self.ffmpeg_bin, self._tasks,
                  self._results, self._load_lock),
            name=f"omni-{pair.replace(',', '-')}") for pair in self.gpu_pairs]
        self._started = False
        self._sequence = 0

    def start(self) -> "OmniProcessPool":
        if not self._started:
            for process in self._processes:
                process.start()
            self._started = True
        return self

    def watch_many(self, requests: Iterable[dict[str, Any]]) -> list[PooledOmniAnswer]:
        self.start()
        queued: list[tuple[str, dict[str, Any]]] = []
        for request in requests:
            task_id = f"task_{self._sequence:06d}"
            self._sequence += 1
            row = dict(request)
            row["task_id"] = task_id
            if row.get("video_path") is None or row.get("prompt") is None:
                raise ValueError("watch request requires video_path and prompt")
            kwargs = dict(row.get("kwargs") or {})
            if kwargs.get("clip_dir") is not None:
                kwargs["clip_dir"] = str(kwargs["clip_dir"])
            row["video_path"] = str(row["video_path"])
            row["mode"] = "watch"
            row["kwargs"] = kwargs
            self._tasks.put(row)
            queued.append((task_id, row))
        return self._collect(queued)

    def _collect(self, queued: list[tuple[str, dict[str, Any]]]) \
            -> list[PooledOmniAnswer]:
        if not queued:
            return []
        pending = {task_id for task_id, _ in queued}
        received: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + self.response_timeout_s * max(
            1, (len(queued) + len(self._processes) - 1) // len(self._processes))
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OmniPoolError(f"Omni pool timed out with {len(pending)} tasks pending")
            try:
                result = self._results.get(timeout=min(5.0, remaining))
            except queue.Empty:
                if self._processes and not any(process.is_alive()
                                               for process in self._processes):
                    raise OmniPoolError("all Omni workers exited before returning results")
                continue
            task_id = str(result.get("task_id"))
            if task_id in pending:
                received[task_id] = result
                pending.remove(task_id)
        answers = []
        for task_id, _ in queued:
            result = received[task_id]
            if not result.get("ok"):
                raise OmniPoolError(
                    f"Omni worker {result.get('gpu_pair')} failed: {result.get('error')}")
            payload = dict(result.get("answer") or {})
            if payload.get("clip_path"):
                payload["clip_path"] = Path(payload["clip_path"])
            answers.append(PooledOmniAnswer(
                **{key: payload[key] for key in PooledOmniAnswer.__dataclass_fields__
                   if key in payload},
                sampling=result.get("sampling"), gpu_pair=result.get("gpu_pair")))
        return answers

    def watch(self, video_path: Path, prompt: str, **kwargs) -> PooledOmniAnswer:
        return self.watch_many([{"video_path": video_path, "prompt": prompt,
                                 "kwargs": kwargs}])[0]

    def ask_many(self, requests: Iterable[dict[str, Any]]) -> list[PooledOmniAnswer]:
        """Run text-only requests on the same persistent model workers."""
        self.start()
        queued = []
        for request in requests:
            task_id = f"task_{self._sequence:06d}"
            self._sequence += 1
            if request.get("prompt") is None:
                raise ValueError("ask request requires prompt")
            row = {
                "task_id": task_id, "mode": "ask", "prompt": request["prompt"],
                "kwargs": dict(request.get("kwargs") or {}), "video_path": "",
            }
            self._tasks.put(row)
            queued.append((task_id, row))
        return self._collect(queued)

    def ask(self, prompt: str, **kwargs) -> PooledOmniAnswer:
        return self.ask_many([{"prompt": prompt, "kwargs": kwargs}])[0]

    def close(self) -> None:
        if not self._started:
            return
        for _ in self._processes:
            self._tasks.put(None)
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self._started = False

    def __enter__(self) -> "OmniProcessPool":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
