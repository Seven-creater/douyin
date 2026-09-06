"""ocr_frames 单测（fake OCR 引擎，不装 rapidocr）。"""
from __future__ import annotations

from src.perception.ocr_frames import merge_text_events, ocr_frames


class FakeOCR:
    def __init__(self, mapping):  # {文件名: [(bbox, text, conf), ...]}
        self.mapping = mapping

    def __call__(self, path_str):
        name = path_str.replace("\\", "/").split("/")[-1]
        items = self.mapping.get(name, [])
        return [[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, conf] for x1, y1, x2, y2, text, conf in items], None


FRAMES_PAYLOAD = {
    "frames": [
        {"index": 0, "t_s": 0.0, "file": "frame_000000.jpg"},
        {"index": 1, "t_s": 1.0, "file": "frame_000001.jpg"},
        {"index": 2, "t_s": 2.0, "file": "frame_000002.jpg"},
        {"index": 3, "t_s": 3.0, "file": "frame_000003.jpg"},
    ]
}


def _touch_frames(tmp_path):
    for fr in FRAMES_PAYLOAD["frames"]:
        (tmp_path / fr["file"]).write_bytes(b"jpg")


def test_ocr_frames_collect_and_filter(tmp_path):
    _touch_frames(tmp_path)
    ocr = FakeOCR({
        "frame_000000.jpg": [(10, 900, 560, 980, "水上飞人", 0.93), (0, 0, 10, 10, "噪点", 0.2)],
        "frame_000001.jpg": [(10, 900, 560, 980, "水上飞人", 0.91)],
        "frame_000002.jpg": [],
        "frame_000003.jpg": [(10, 100, 200, 150, "关注我", 0.88)],
    })
    out = ocr_frames(FRAMES_PAYLOAD, tmp_path, ocr=ocr, stride=1, min_confidence=0.5)
    assert out["frames_ocr_count"] == 3          # 3 帧有字（低置信被滤、空帧不算）
    assert out["frames_total"] == 4
    # bbox 归一为 [x1,y1,x2,y2]
    assert out["frames_ocr"][0]["lines"][0]["bbox"] == [10, 900, 560, 980]
    assert "水上飞人" in out["full_text"] and "噪点" not in out["full_text"]
    # 相邻同文本合并为一个事件
    ev = [e for e in out["text_events"] if e["text"] == "水上飞人"]
    assert len(ev) == 1 and ev[0]["occurrences"] == 2
    assert ev[0]["t_start_s"] == 0.0 and ev[0]["t_end_s"] == 1.0


def test_ocr_frames_stride(tmp_path):
    _touch_frames(tmp_path)
    ocr = FakeOCR({f"frame_00000{i}.jpg": [(0, 0, 1, 1, f"t{i}", 0.9)] for i in range(4)})
    out = ocr_frames(FRAMES_PAYLOAD, tmp_path, ocr=ocr, stride=2, min_confidence=0.5)
    assert out["stride"] == 2
    assert [fr["t_s"] for fr in out["frames_ocr"]] == [0.0, 2.0]


def test_merge_text_events_dedup_normalized():
    frames_ocr = [
        {"t_s": 0.0, "file": "a", "lines": [{"text": "你好 世界", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]},
        {"t_s": 1.0, "file": "b", "lines": [{"text": "你好世界", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]},
        {"t_s": 2.0, "file": "c", "lines": [{"text": "再见", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]},
    ]
    ev = merge_text_events(frames_ocr)
    assert len(ev) == 2
    assert ev[0]["occurrences"] == 2          # 空白差异不算新事件
    assert ev[1] == {"text": "再见", "t_start_s": 2.0, "t_end_s": 2.0, "occurrences": 1}
