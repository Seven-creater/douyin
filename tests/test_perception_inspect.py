"""inspect_video 单测（fake ffprobe JSON，不打真实进程）。"""
from __future__ import annotations

import pytest

from src.perception.inspect_video import _parse_fps, inspect_video


def sample_ffprobe(duration="22.012000", with_audio=True, rotation=None):
    vstream = {
        "codec_type": "video", "codec_name": "h264", "width": 576, "height": 1024,
        "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
    }
    if rotation is not None:
        vstream["side_data_list"] = [{"rotation": rotation}]
    streams = [vstream]
    if with_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac",
                        "sample_rate": "44100", "channels": 2})
    return {"streams": streams, "format": {"duration": duration, "bit_rate": "4254067", "size": "11705066"}}


def test_full_parse():
    out = inspect_video(sample_ffprobe())
    assert out["duration_s"] == 22.012
    assert (out["width"], out["height"]) == (576, 1024)
    assert out["fps"] == 30.0
    assert out["video_codec"] == "h264"
    assert out["has_audio"] and out["audio_codec"] == "aac"
    assert out["audio_sample_rate"] == 44100 and out["audio_channels"] == 2
    assert out["bit_rate_bps"] == 4254067 and out["size_bytes"] == 11705066
    assert out["rotation"] == 0


def test_no_audio_stream():
    out = inspect_video(sample_ffprobe(with_audio=False))
    assert out["has_audio"] is False
    assert out["audio_codec"] is None and out["audio_sample_rate"] is None


def test_rotation_normalized():
    # 语义：归一化到 [0,360) 的绝对角度（-90 display-matrix → 270）
    assert inspect_video(sample_ffprobe(rotation=-90))["rotation"] == 270
    assert inspect_video(sample_ffprobe(rotation=180))["rotation"] == 180
    assert inspect_video(sample_ffprobe(rotation=90))["rotation"] == 90


def test_missing_video_stream_raises():
    with pytest.raises(ValueError, match="无视频流"):
        inspect_video({"streams": [{"codec_type": "audio"}], "format": {}})


def test_parse_fps_edge():
    assert _parse_fps("30/1") == 30.0
    assert _parse_fps("0/0") is None        # 有的容器占位值
    assert _parse_fps(None) is None
    assert _parse_fps("25") == 25.0
