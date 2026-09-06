"""ocr_frames：RapidOCR 帧内文字提取（CPU，onnx 模型内置包内、离线可用）。

依赖 extract_frames 的产物（frames/result.json）。
CLI：python -m src.perception.ocr_frames --aweme-id <id> [--stride 1] [--force]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def load_ocr():
    """延迟 import（本地无 GPU 栈也能跑单测）。"""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def ocr_image(ocr, image_path: Path, *, min_confidence: float) -> list[dict]:
    result, _ = ocr(str(image_path))
    lines = []
    for item in result or []:
        # RapidOCR 返回 [bbox(4点), text, confidence]
        bbox, text, conf = item[0], item[1], float(item[2])
        if text and conf >= min_confidence:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            lines.append({"text": text, "confidence": round(conf, 3),
                          "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
    return lines


def merge_text_events(frames_ocr: list[dict]) -> list[dict]:
    """相邻帧归一化文本相同 → 合并为时间跨度（字幕常驻检测）。"""
    events: list[dict] = []
    for fr in frames_ocr:
        for line in fr["lines"]:
            key = _normalize(line["text"])
            if not key:
                continue
            if events and _normalize(events[-1]["text"]) == key:
                events[-1]["t_end_s"] = fr["t_s"]
                events[-1]["occurrences"] += 1
                continue
            events.append({"text": line["text"], "t_start_s": fr["t_s"],
                           "t_end_s": fr["t_s"], "occurrences": 1})
    return [e for e in events if e["occurrences"] >= 1]


def ocr_frames(frames_payload: dict, frames_dir: Path, *, ocr, stride: int = 1,
               min_confidence: float = 0.5) -> dict:
    frames = frames_payload.get("frames") or []
    picked = frames[::stride]
    frames_ocr = []
    for fr in picked:
        path = frames_dir / fr["file"]
        if not path.exists():
            continue
        lines = ocr_image(ocr, path, min_confidence=min_confidence)
        if lines:
            frames_ocr.append({"t_s": fr["t_s"], "file": fr["file"], "lines": lines})
    full_text = " ".join(
        line["text"] for fr in frames_ocr for line in fr["lines"] if line["text"].strip()
    )
    return {
        "engine": "rapidocr",
        "stride": stride,
        "frames_ocr_count": len(frames_ocr),
        "frames_total": len(frames),
        "frames_ocr": frames_ocr,
        "text_events": merge_text_events(frames_ocr),
        "full_text": full_text[:2000],
    }


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False,
                  stride: int | None = None, ocr=None):
    p_cfg = common.perception_cfg(cfg)
    o_cfg = p_cfg.get("ocr") or {}
    stride = int(stride if stride is not None else o_cfg.get("stride", 1))
    min_confidence = float(o_cfg.get("min_confidence", 0.5))

    frames_result_path = cfg.paths.perception_dir / aweme_id / "frames" / "result.json"
    frames_env = common.read_result_json(frames_result_path.parent)
    if frames_env is None:
        raise FileNotFoundError(
            f"缺少抽帧产物 {frames_result_path}；先跑 python -m src.perception.extract_frames --aweme-id {aweme_id}"
        )

    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "ocr")
    params = {"engine": o_cfg.get("engine", "rapidocr"), "stride": stride,
              "min_confidence": min_confidence}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[ocr %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    if ocr is None:
        ocr = load_ocr()

    t0 = time.time()
    output = ocr_frames(frames_env["output"], frames_result_path.parent, ocr=ocr,
                        stride=stride, min_confidence=min_confidence)
    path = common.write_result_json(tdir, tool="ocr_frames", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="ocr_frames", aweme_id=aweme_id, status="ok",
        elapsed_s=round(time.time() - t0, 2),
        extra={"frames_ocr": output["frames_ocr_count"], "events": len(output["text_events"])},
    )
    logger.info("[ocr %s] %d/%d 帧有字 %d 个文本事件 → %s", aweme_id,
                output["frames_ocr_count"], output["frames_total"],
                len(output["text_events"]), path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="帧内 OCR（RapidOCR）")
    common.add_common_cli_args(ap)
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_ocr")
    try:
        aweme_id, _ = common.resolve_target(cfg, args)
        path = run_for_video(cfg, aweme_id, force=args.force, stride=args.stride)
        common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("ocr_frames 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
