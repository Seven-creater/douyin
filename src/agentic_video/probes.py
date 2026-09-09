"""Deterministic evidence probes selected by the bounded perception agent."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.agentic_video.agent import ProbeTask


def _extract_frames(video: Path, task: ProbeTask, output: Path, *,
                    ffmpeg_bin: str = "ffmpeg", fps: float = 12.0) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_bin, "-y", "-loglevel", "error", "-ss", f"{task.start:g}",
               "-i", str(video), "-t", f"{task.end - task.start:g}", "-vf",
               f"fps={fps:g},scale=320:-2", "-q:v", "3", str(output / "%04d.jpg")]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"probe frame extraction failed: {(proc.stderr or '')[-300:]}")
    return sorted(output.glob("*.jpg"))


def _visual_evidence(frames: list[Path]) -> dict:
    import cv2
    import numpy as np

    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in frames]
    images = [image for image in images if image is not None]
    if len(images) < 2:
        return {"n_frames": len(images), "global_diff": [], "flow_magnitude": [],
                "active_grid_cells": []}
    global_diff, flow_magnitude, active_cells = [], [], []
    for previous, current in zip(images, images[1:]):
        delta = cv2.absdiff(previous, current)
        global_diff.append(round(float(delta.mean()), 3))
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 2, 15, 2, 5, 1.1, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitude.append(round(float(np.median(magnitude)), 3))
        height, width = delta.shape
        cell_values = []
        for row in range(4):
            for col in range(4):
                cell = delta[row * height // 4:(row + 1) * height // 4,
                             col * width // 4:(col + 1) * width // 4]
                cell_values.append(float(cell.mean()))
        active_cells.append(sum(value >= 18.0 for value in cell_values))
    return {"n_frames": len(images), "global_diff": global_diff,
            "flow_magnitude": flow_magnitude, "active_grid_cells": active_cells}


def _ocr_evidence(frames: list[Path]) -> dict:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return {"available": False, "events": []}
    engine = RapidOCR()
    events = []
    for index, frame in enumerate(frames):
        result, _ = engine(str(frame))
        texts = [{"text": str(row[1]), "confidence": round(float(row[2]), 3)}
                 for row in (result or []) if len(row) >= 3 and float(row[2]) >= 0.5]
        if texts:
            events.append({"frame_idx": index, "texts": texts})
    return {"available": True, "events": events}


def collect_probe_evidence(video: Path, task: ProbeTask, output: Path, *,
                           ffmpeg_bin: str = "ffmpeg") -> dict:
    frames = _extract_frames(video, task, output / "frames", ffmpeg_bin=ffmpeg_bin)
    evidence = {"probe": task.probe, "interval": [task.start, task.end],
                "visual": _visual_evidence(frames)}
    if task.probe == "ocr_probe":
        evidence["ocr"] = _ocr_evidence(frames)
    path = output / "probe_evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence
