"""Small client adapter for the external FlashVID OpenAI-compatible services.

The FlashVID algorithm and model plugin remain in Seven-creater/flashvid.  This
module only owns the request/audit contract required by the V7 experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _local_media_paths(value: Any) -> Any:
    if isinstance(value, list):
        return [_local_media_paths(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _local_media_paths(item) for key, item in value.items()}
    url = result.get("url")
    if isinstance(url, str) and url.startswith("file://"):
        parsed = urllib.parse.urlparse(url)
        path = urllib.request.url2pathname(parsed.path)
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        result["url"] = path
    return result


@dataclass(frozen=True)
class FlashVIDEndpoint:
    arm: str
    base_url: str
    model: str
    fps: float
    retention_ratio: float
    backend: str

    def __post_init__(self) -> None:
        if self.arm not in {"A", "B", "C", "D"}:
            raise ValueError(f"unsupported V7 browse arm: {self.arm}")
        if self.fps <= 0 or not 0 < self.retention_ratio <= 1:
            raise ValueError("fps and retention_ratio must be positive")
        if self.arm == "D" and (self.backend != "native_bypass" or
                                abs(self.retention_ratio - 1.0) > 1e-6):
            raise ValueError("arm D must be native_bypass at r1.00")


@dataclass(frozen=True)
class FlashVIDAnswer:
    text: str
    usage: dict[str, Any]
    latency_s: float
    raw: dict[str, Any]
    request_audit: dict[str, Any]


class OpenAICompatibleClient:
    """Subset of Seven-creater/flashvid's client contract used by V7."""

    def __init__(self, base_url: str, *, api_key: str = "no",
                 timeout_s: float = 900.0,
                 local_file_urls_as_paths: bool = False):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.local_file_urls_as_paths = local_file_urls_as_paths

    def chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float, str]:
        wire_payload = copy.deepcopy(payload)
        if self.local_file_urls_as_paths:
            wire_payload["messages"] = _local_media_paths(wire_payload["messages"])
        body = json.dumps(wire_payload, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FlashVID HTTP {exc.code}: {detail[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"FlashVID request failed: {exc.reason}") from exc
        return raw, time.perf_counter() - started, _sha256_bytes(body)


class FlashVIDClient:
    def __init__(self, endpoint: FlashVIDEndpoint, *, api_key: str = "no",
                 timeout_s: float = 900.0):
        self.endpoint = endpoint
        self.transport = OpenAICompatibleClient(
            endpoint.base_url, api_key=api_key, timeout_s=timeout_s)

    @staticmethod
    def requested_frames(duration_s: float, fps: float) -> int:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("video duration must be positive")
        count = max(4, int(math.floor(duration_s * fps)))
        return count - count % 2

    def watch(self, video_path: Path, prompt: str, *, duration_s: float,
              image_paths: list[Path] | None = None,
              max_tokens: int = 768) -> FlashVIDAnswer:
        images = [Path(path).resolve() for path in (image_paths or [])]
        if len(images) > 4:
            raise ValueError("V7 services accept at most four images")
        video = Path(video_path).resolve()
        frames = self.requested_frames(duration_s, self.endpoint.fps)
        content = [
            {"type": "image_url", "image_url": {"url": path.as_uri()}}
            for path in images
        ]
        content.extend([
            {"type": "video_url", "video_url": {"url": video.as_uri()}},
            {"type": "text", "text": prompt},
        ])
        payload = {
            "model": self.endpoint.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "mm_processor_kwargs": {"do_sample_frames": False},
            "media_io_kwargs": {"video": {"num_frames": frames, "fps": -1}},
        }
        raw, latency_s, request_sha = self.transport.chat(payload)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("FlashVID response has no choices")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if isinstance(text, list):
            text = "".join(str(row.get("text") or "") for row in text
                           if isinstance(row, dict))
        response_bytes = json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
        audit = {
            "arm": self.endpoint.arm,
            "custom_flashvid_port": self.endpoint.arm != "D",
            "base_model": "Qwen3.5-4B",
            "backend": self.endpoint.backend,
            "retention_ratio": self.endpoint.retention_ratio,
            "requested_fps": self.endpoint.fps,
            "requested_frames": frames,
            "do_sample_frames": False,
            "image_count": len(images),
            "video_count": 1,
            "transport_clip": str(video),
            "transport_clip_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "request_sha256": request_sha,
            "response_sha256": _sha256_bytes(response_bytes),
            "usage": raw.get("usage") or {},
            "latency_s": round(latency_s, 6),
        }
        return FlashVIDAnswer(
            text=str(text or ""), usage=dict(raw.get("usage") or {}),
            latency_s=latency_s, raw=raw, request_audit=audit)
