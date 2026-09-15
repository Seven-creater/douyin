#!/usr/bin/env python3
"""Serialize V8.2 FlashVID calls and attach exact input/token audit facts.

The proxy does not implement or modify FlashVID. It forwards OpenAI-compatible
requests to a vLLM service, inspects the actual presampled transport media, and
joins exact Qwen prompt pad-token counts with the optional worker-side FlashVID
token record. Serialization makes the sidecar record attributable to one
request instead of guessing from interleaved server logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence


QWEN_VISION_START_TOKEN_ID = 248053
QWEN_VISION_END_TOKEN_ID = 248054
QWEN_IMAGE_PAD_TOKEN_ID = 248056
QWEN_VIDEO_PAD_TOKEN_ID = 248057


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_facts(directory: Path, core_files: Sequence[str]) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=directory, capture_output=True, text=True,
            timeout=30, check=True)
        return result.stdout.strip()

    hashes = {}
    for raw in core_files:
        path = (directory / raw).resolve()
        hashes[raw] = _sha256(path) if path.is_file() else None
    return {
        "service_git_head": git("rev-parse", "HEAD"),
        "service_git_dirty": bool(git("status", "--porcelain")),
        "service_core_hashes": hashes,
    }


def _local_path(url: str) -> Path:
    if url.startswith("file://"):
        parsed = urllib.parse.urlparse(url)
        value = urllib.request.url2pathname(parsed.path)
        if parsed.netloc:
            value = f"//{parsed.netloc}{value}"
        return Path(value)
    return Path(url)


def _media_paths(payload: Mapping[str, Any]) -> tuple[Path, list[Path]]:
    video: Path | None = None
    images: list[Path] = []
    for message in payload.get("messages") or []:
        for item in message.get("content") or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "video_url":
                raw = (item.get("video_url") or {}).get("url")
                if raw:
                    if video is not None:
                        raise ValueError("V8.2 audit accepts exactly one video")
                    video = _local_path(str(raw)).resolve()
            elif item.get("type") == "image_url":
                raw = (item.get("image_url") or {}).get("url")
                if raw:
                    images.append(_local_path(str(raw)).resolve())
    if video is None or not video.is_file():
        raise ValueError("source transport video is missing")
    if any(not path.is_file() for path in images):
        raise ValueError("one or more identity-album images are missing")
    return video, images


def _video_facts(path: Path) -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise RuntimeError("PyAV is required by the V8.2 audit proxy") from exc
    relative: list[float] = []
    width = height = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = int(stream.width), int(stream.height)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("transport frame has no PTS")
            relative.append(round(float(frame.pts * frame.time_base), 6))
    if not relative or any(left >= right for left, right in zip(relative, relative[1:])):
        raise ValueError("transport frame PTS are empty or non-monotonic")
    return {"width": width, "height": height, "relative_pts": relative}


def _image_sizes(paths: Sequence[Path]) -> list[dict[str, int]]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise RuntimeError("Pillow is required by the V8.2 audit proxy") from exc
    result = []
    for path in paths:
        with Image.open(path) as image:
            result.append({"width": int(image.width), "height": int(image.height)})
    return result


def _vision_segments(token_ids: Sequence[int]) -> list[dict[str, int]]:
    segments = []
    current: list[int] | None = None
    for token_id in token_ids:
        if token_id == QWEN_VISION_START_TOKEN_ID:
            if current is not None:
                raise ValueError("nested Qwen vision segments")
            current = []
        elif token_id == QWEN_VISION_END_TOKEN_ID:
            if current is None:
                raise ValueError("unbalanced Qwen vision end token")
            segments.append({
                "image_tokens": current.count(QWEN_IMAGE_PAD_TOKEN_ID),
                "video_tokens": current.count(QWEN_VIDEO_PAD_TOKEN_ID),
            })
            current = None
        elif current is not None:
            current.append(int(token_id))
    if current is not None:
        raise ValueError("unbalanced Qwen vision start token")
    return segments


def _factor_grid(tokens: int, width: int, height: int, *, temporal: int = 1,
                 merge_size: int = 2) -> list[int]:
    """Recover the unique merged-token grid closest to the input aspect ratio."""
    if tokens <= 0 or temporal <= 0 or tokens % temporal:
        raise ValueError("visual token count is incompatible with temporal grid")
    spatial = tokens // temporal
    target_ratio = width / max(1, height)
    candidates = []
    for merged_h in range(1, int(math.sqrt(spatial)) + 1):
        if spatial % merged_h:
            continue
        merged_w = spatial // merged_h
        for h, w in ((merged_h, merged_w), (merged_w, merged_h)):
            error = abs(math.log((w / h) / target_ratio))
            candidates.append((error, h, w))
    if not candidates:
        raise ValueError("cannot factor visual token grid")
    _, merged_h, merged_w = min(candidates)
    return [temporal, merged_h * merge_size, merged_w * merge_size]


def _read_sidecar(path: Path, offset: int, *, timeout_s: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > offset:
            with path.open("rb") as handle:
                handle.seek(offset)
                rows = [json.loads(line) for line in handle.read().decode().splitlines()
                        if line.strip()]
            if len(rows) == 1:
                return rows[0]
            if len(rows) > 1:
                raise RuntimeError("multiple model audit rows for one serialized request")
        time.sleep(0.05)
    raise TimeoutError("FlashVID worker token audit row did not arrive")


class AuditProxy:
    def __init__(self, args: argparse.Namespace):
        self.upstream = args.upstream.rstrip("/")
        self.retention_ratio = float(args.retention_ratio)
        self.backend = str(args.backend)
        self.sidecar = Path(args.model_audit_jsonl).resolve() \
            if args.model_audit_jsonl else None
        self.merge_size = int(args.spatial_merge_size)
        self.temporal_patch_size = int(args.temporal_patch_size)
        self.lock = threading.Lock()
        self.runtime = {
            **_git_facts(Path(args.service_git_dir).resolve(), args.core_file),
            "service_startup_args": json.loads(args.service_startup_args_json),
            "audit_proxy_pid": os.getpid(),
            "audit_proxy_backend": self.backend,
        }

    def forward(self, body: bytes, headers: Mapping[str, str]) -> tuple[int, bytes]:
        with self.lock:
            payload = json.loads(body)
            payload["return_token_ids"] = True
            wire = json.dumps(payload, ensure_ascii=False,
                              separators=(",", ":")).encode()
            sidecar_offset = (self.sidecar.stat().st_size
                              if self.sidecar and self.sidecar.is_file() else 0)
            request = urllib.request.Request(
                self.upstream + "/v1/chat/completions", data=wire,
                headers={"Content-Type": "application/json",
                         "Authorization": headers.get("Authorization", "Bearer no")})
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    raw = json.load(response)
                    status = int(response.status)
            except urllib.error.HTTPError as exc:
                return int(exc.code), exc.read()
            video, images = _media_paths(payload)
            video_facts = _video_facts(video)
            image_sizes = _image_sizes(images)
            prompt_ids = raw.get("prompt_token_ids")
            if not isinstance(prompt_ids, list) or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in prompt_ids):
                raise ValueError("upstream response lacks exact prompt_token_ids")
            segments = _vision_segments(prompt_ids)
            video_segments = [row for row in segments if row["video_tokens"]]
            image_segments = [row for row in segments if row["image_tokens"]]
            if len(video_segments) != 1 or len(image_segments) != len(images):
                raise ValueError("processed media segments do not match request media")
            image_grids = [
                _factor_grid(segment["image_tokens"], size["width"], size["height"],
                             merge_size=self.merge_size)
                for segment, size in zip(image_segments, image_sizes)
            ]
            video_after = int(video_segments[0]["video_tokens"])
            if self.backend == "flashvid":
                if self.sidecar is None:
                    raise ValueError("compressed backend requires model audit sidecar")
                worker = _read_sidecar(self.sidecar, sidecar_offset)
                video_grid = list(map(int, worker["video_grid_thw"]))
                video_before = int(worker["video_tokens_before"])
                retained = int(worker["video_tokens_after"])
                if retained != video_after:
                    raise ValueError(
                        f"worker/prompt retained tokens disagree: {retained} != {video_after}")
            elif self.backend == "native_bypass":
                video_before = video_after
                temporal = math.ceil(
                    len(video_facts["relative_pts"]) / self.temporal_patch_size)
                video_grid = _factor_grid(
                    video_before, video_facts["width"], video_facts["height"],
                    temporal=temporal, merge_size=self.merge_size)
            else:
                raise ValueError(f"unsupported audit backend: {self.backend}")
            actual_ratio = video_after / video_before
            origin = float(headers.get("X-FlashVID-Source-Origin-S", 0.0))
            relative = video_facts["relative_pts"]
            audit = {
                "schema_version": "flashvid_server_input_audit_v82",
                "actual_sampled_frame_count": len(relative),
                "actual_frame_indices": list(range(len(relative))),
                "actual_frame_timestamps_relative_s": relative,
                "actual_frame_timestamps_absolute_s": [
                    round(origin + value, 6) for value in relative],
                "input_image_sizes": image_sizes,
                "processed_image_grid_thw": image_grids,
                "processed_video_grid_thw": [video_grid],
                "video_tokens_before": video_before,
                "video_tokens_after": video_after,
                "actual_retention_ratio": actual_ratio,
                "video_transport_sha256": _sha256(video),
                "image_sha256": [_sha256(path) for path in images],
                **self.runtime,
            }
            raw["flashvid_audit"] = audit
            return status, json.dumps(raw, ensure_ascii=False).encode()


def _handler(proxy: AuditProxy):
    class Handler(BaseHTTPRequestHandler):
        server_version = "V82FlashVIDAuditProxy/1.0"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = json.dumps({"status": "ok", **proxy.runtime}).encode()
                self._send(HTTPStatus.OK, body, "application/json")
                return
            try:
                with urllib.request.urlopen(proxy.upstream + self.path, timeout=30) as response:
                    self._send(int(response.status), response.read(),
                               response.headers.get_content_type())
            except urllib.error.HTTPError as exc:
                self._send(int(exc.code), exc.read(), "application/json")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}',
                           "application/json")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                status, body = proxy.forward(self.rfile.read(size), self.headers)
                self._send(status, body, "application/json")
            except Exception as exc:  # keep a machine-readable infrastructure failure
                body = json.dumps({"error": {
                    "type": "v82_audit_proxy_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }}).encode()
                self._send(HTTPStatus.BAD_GATEWAY, body, "application/json")

        def log_message(self, message: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {message % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--retention-ratio", required=True, type=float)
    parser.add_argument("--backend", choices=("flashvid", "native_bypass"), required=True)
    parser.add_argument("--model-audit-jsonl", default=None)
    parser.add_argument("--service-git-dir", required=True)
    parser.add_argument("--core-file", action="append",
                        default=["src/flashvid_vllm/model.py"])
    parser.add_argument("--service-startup-args-json", default="[]")
    parser.add_argument("--spatial-merge-size", type=int, default=2)
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    args = parser.parse_args()
    proxy = AuditProxy(args)
    server = ThreadingHTTPServer((args.host, args.port), _handler(proxy))
    print(json.dumps({"listening": [args.host, args.port],
                      "upstream": args.upstream, **proxy.runtime}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
