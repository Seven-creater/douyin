from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "v82_flashvid_audit_proxy.py"
SPEC = importlib.util.spec_from_file_location("v82_flashvid_audit_proxy", SCRIPT)
assert SPEC and SPEC.loader
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def test_qwen_vision_segments_keep_video_and_images_separate() -> None:
    ids = [
        proxy.QWEN_VISION_START_TOKEN_ID,
        proxy.QWEN_VIDEO_PAD_TOKEN_ID,
        proxy.QWEN_VIDEO_PAD_TOKEN_ID,
        proxy.QWEN_VISION_END_TOKEN_ID,
        proxy.QWEN_VISION_START_TOKEN_ID,
        proxy.QWEN_IMAGE_PAD_TOKEN_ID,
        proxy.QWEN_VISION_END_TOKEN_ID,
    ]
    assert proxy._vision_segments(ids) == [
        {"image_tokens": 0, "video_tokens": 2},
        {"image_tokens": 1, "video_tokens": 0},
    ]


def test_factor_grid_preserves_token_product_and_aspect() -> None:
    grid = proxy._factor_grid(12, 1600, 900, merge_size=2)
    assert grid[0] == 1
    assert grid[1] * grid[2] // 4 == 12
    assert grid[2] > grid[1]


def test_video_factor_grid_includes_native_temporal_patches() -> None:
    grid = proxy._factor_grid(48, 1920, 1080, temporal=4, merge_size=2)
    assert grid[0] == 4
    assert grid[0] * grid[1] * grid[2] // 4 == 48
