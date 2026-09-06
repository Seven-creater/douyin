"""本地 ↔ 服务器同步（scp，免密已配；失败不致命只告警）。

- push_videos：把新下载成功的 data/videos/<id>/ 目录 + manifest + processed 产物推到服务器
  （Phase 2 的 Qwen3-Omni 感知在服务器跑，视频必须在那边）
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def push_videos(
    ssh_target: str,
    remote_root: str,
    videos_dir: Path,
    processed_dir: Path,
    *,
    new_ids: list[str],
    manifest_path: Path,
) -> dict:
    """推送新下载内容到服务器。返回 {"pushed": n, "errors": [...]}。"""
    result = {"pushed": 0, "errors": []}
    if not new_ids:
        return result
    for aweme_id in new_ids:
        local_dir = videos_dir / aweme_id
        if not local_dir.exists():
            continue
        code, err = _run([
            "scp", "-q", *_SSH_OPTS, "-r",
            str(local_dir), f"{ssh_target}:{remote_root}/data/videos/",
        ])
        if code == 0:
            result["pushed"] += 1
        else:
            result["errors"].append({"id": aweme_id, "error": err[:300]})
            logger.warning("[sync %s] 推送失败: %s", aweme_id, err[:200])
    # 台账与榜单产物（幂等，覆盖远端）
    for src, dst in (
        (manifest_path, f"{ssh_target}:{remote_root}/data/videos/manifest.json"),
        (processed_dir / "trending_videos.jsonl", f"{ssh_target}:{remote_root}/data/processed/"),
        (processed_dir / "trending_videos.csv", f"{ssh_target}:{remote_root}/data/processed/"),
    ):
        if not src.exists():
            continue
        code, err = _run(["scp", "-q", *_SSH_OPTS, str(src), dst])
        if code != 0:
            result["errors"].append({"id": str(src.name), "error": err[:300]})
    if result["errors"]:
        logger.warning("[sync] %d 项同步失败（不致命）", len(result["errors"]))
    else:
        logger.info("[sync] 已推送 %d 个视频目录 + manifest + 榜单产物 → %s", result["pushed"], ssh_target)
    return result
