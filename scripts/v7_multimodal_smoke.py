"""Smoke-test V7 services with the exact four-images-plus-one-video shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agentic_video.target_v7 import export_frame  # noqa: E402
from src.perception.flashvid_client import (  # noqa: E402
    FlashVIDClient,
    FlashVIDEndpoint,
)
from src.perception.omni_runner import cut_clip  # noqa: E402

JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    candidate = text.strip()
    fenced = JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("smoke response must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=5385.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    video, output = Path(args.video).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clip = cut_clip("ffmpeg", video, output / "clip", start_s=args.start,
                    end_s=args.start + 6.0)
    images = [export_frame("ffmpeg", video, args.start + offset,
                           output / f"image_{index}.jpg")
              for index, offset in enumerate((0.5, 1.5, 2.5, 3.5))]
    rows = []
    for arm, port, ratio, backend in (
        ("A", 8101, .10, "flashvid"),
        ("C", 8102, .25, "flashvid"),
        ("D", 8104, 1.0, "native_bypass"),
    ):
        client = FlashVIDClient(FlashVIDEndpoint(
            arm, f"http://127.0.0.1:{port}/v1", "Qwen3.5-4B",
            2.0 if arm == "A" else 4.0, ratio, backend))
        answer = client.watch(
            clip,
            "Confirm you received both the ordered still images and the video. "
            "Return JSON only: {\"images_visible\":true,\"video_visible\":true}.",
            duration_s=6.0, image_paths=images, max_tokens=64)
        parsed = parse_json_object(answer.text)
        if parsed.get("images_visible") is not True or parsed.get("video_visible") is not True:
            raise RuntimeError(f"arm {arm} silently lost an input modality: {parsed}")
        if answer.request_audit["image_count"] != 4 or answer.request_audit["video_count"] != 1:
            raise RuntimeError(f"arm {arm} request shape was not 4 images + 1 video")
        rows.append({"arm": arm, "response": parsed, "audit": answer.request_audit})
    (output / "multimodal_smoke.json").write_text(
        json.dumps({"passed": True, "services": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
