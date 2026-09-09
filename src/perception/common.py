"""perception 工具层统一约定：路径、result.json 信封、幂等跳过、metrics 流水、ffmpeg 封装。

所有工具遵循同一模式（给未来 coding agent 调用）：
    python -m src.perception.<tool> --aweme-id <id> [--force]
    产物 → data/perception/<aweme_id>/<tool>/result.json（存在即跳过）
    成本 → data/perception/metrics.jsonl 追加一行（error 也记）
    stdout 最后一行 emit_status_line 的单行 JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)

_HOST_INFO = {"platform": sys.platform.lower(), "node": socket.gethostname()}


# ---------- 路径 ----------

def tool_dir_for(perception_dir: Path, aweme_id: str, tool: str) -> Path:
    d = Path(perception_dir) / aweme_id / tool
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip_dir_for(perception_dir: Path, aweme_id: str, start_s: float, end_s: float) -> Path:
    d = Path(perception_dir) / aweme_id / "omni_clips" / f"{start_s:g}-{end_s:g}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_video_path(videos_dir: Path, aweme_id: str) -> Path:
    p = Path(videos_dir) / aweme_id / "video.mp4"
    if not p.exists():
        raise FileNotFoundError(
            f"视频不存在: {p}（Phase 1 未下载或未同步；先在本地跑 python -m src.pipeline.collect_trends）"
        )
    return p


def list_aweme_ids(videos_dir: Path) -> list[str]:
    vd = Path(videos_dir)
    if not vd.exists():
        return []
    return sorted(d.name for d in vd.iterdir() if (d / "video.mp4").exists())


def metrics_path_for(perception_dir: Path) -> Path:
    p = Path(perception_dir) / "metrics.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------- result.json 信封 + 幂等 ----------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_result_json(tool_dir: Path, *, tool: str, aweme_id: str, params: dict, output: dict) -> Path:
    """原子写（.tmp → os.replace）。"""
    envelope = {
        "schema_version": 1,
        "tool": tool,
        "aweme_id": aweme_id,
        "params": params,
        "created_at": _now_iso(),
        # platform.system()/platform.node() can enter a slow WMI query on Windows.
        # This metadata is process-constant, so resolve it once without WMI.
        "host": dict(_HOST_INFO),
        "output": output,
    }
    path = Path(tool_dir) / "result.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_result_json(tool_dir: Path) -> dict | None:
    path = Path(tool_dir) / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def done_or_skip(tool_dir: Path, params: dict, *, force: bool) -> dict | None:
    """result.json 已存在且非 force → 返回旧信封（参数漂移时告警）；否则 None 表示要跑。"""
    old = read_result_json(tool_dir)
    if old is None or force:
        return None
    if old.get("params") != params:
        logger.warning(
            "[skip] %s 已有产物但参数不同（旧=%s 新=%s），默认沿用旧产物；重跑请 --force",
            tool_dir, old.get("params"), params,
        )
    return old


# ---------- metrics ----------

def append_metric(metrics_path: Path, **row: Any) -> None:
    """追加单行 JSON（单进程串行设计）。"""
    row.setdefault("ts", _now_iso())
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- ffmpeg / ffprobe ----------

class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(ffmpeg_bin: str, args: list[str], *, timeout_s: float = 300) -> None:
    cmd = [ffmpeg_bin, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg 退出码 {proc.returncode}: {(proc.stderr or '')[-500:]}")


def run_ffprobe_json(ffprobe_bin: str, video_path: Path) -> dict:
    proc = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe 失败: {(proc.stderr or '')[-300:]}")
    return json.loads(proc.stdout)


def video_duration_s(ffprobe_bin: str, video_path: Path) -> float:
    data = run_ffprobe_json(ffprobe_bin, video_path)
    return float(data.get("format", {}).get("duration", 0.0))


# ---------- CLI 公共 ----------

def add_common_cli_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--aweme-id", default=None, help="视频主键（与 --video 二选一）")
    ap.add_argument("--video", default=None, help="直接给 video.mp4 路径（调试用）")
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 config/default.yaml）")


def resolve_target(cfg: AppConfig, args: argparse.Namespace) -> tuple[str, Path]:
    """--aweme-id 或 --video → (aweme_id, video_path)。"""
    if bool(args.aweme_id) == bool(args.video):
        raise SystemExit("必须且只能提供 --aweme-id 或 --video 之一")
    if args.video:
        p = Path(args.video)
        if not p.exists():
            raise SystemExit(f"视频不存在: {p}")
        return p.parent.name, p
    return args.aweme_id, resolve_video_path(cfg.paths.videos_dir, args.aweme_id)


def load_cfg(args: argparse.Namespace) -> AppConfig:
    return load_config(Path(args.config) if args.config else None)


def perception_cfg(cfg: AppConfig) -> dict:
    return cfg.perception or {}


def emit_status_line(status: str, **kw: Any) -> None:
    """stdout 最后一行单行 JSON（agent 可解析）。"""
    print(json.dumps({"status": status, **kw}, ensure_ascii=False))
