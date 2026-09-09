"""本地 → 服务器视频同步（scp；失败不致命只记录）。

同步策略（对账式，幂等）：
    远端已有文件 = ssh find 一次，记录 video.mp4 字节数
    需推送 = manifest 中 success 且本地文件存在，但远端缺失或大小不同
    逐个推送：单文件超时 1800s、最多 2 次尝试、单文件异常只记录不中断
这覆盖了"上轮崩溃没推成 / 半传文件 / 手工修复的孤儿记录"，
而不只是本轮新下载——服务器最终与本地 manifest 的 success 集合一致。
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
_PER_FILE_TIMEOUT = 1800  # 秒；覆盖慢速/卡顿连接下 ~20MB 级文件
_ATTEMPTS = 2


def _run(cmd: list[str], timeout: float) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def compute_missing(local_success_ids: list[str], remote_ids: list[str]) -> list[str]:
    """需推送 = 本地成功集合 − 远端已有（纯函数，供单测）。"""
    remote = set(remote_ids)
    seen: set[str] = set()
    missing: list[str] = []
    for i in local_success_ids:
        if i not in seen and i not in remote:
            seen.add(i)
            missing.append(i)
    return missing


def compute_size_mismatches(local_sizes: dict[str, int],
                            remote_sizes: dict[str, int]) -> list[str]:
    """Return local ids whose remote video is absent or not byte-identical in size."""
    return [aweme_id for aweme_id, size in local_sizes.items()
            if remote_sizes.get(aweme_id) != size]


def list_remote_video_dirs(ssh_target: str, remote_videos_dir: str) -> list[str]:
    """远端 data/videos/ 下已有哪些 <aweme_id> 目录。失败返回 []（触发全量补推）。"""
    try:
        code, out = _run(
            ["ssh", *_SSH_OPTS, ssh_target, f"ls -1 {remote_videos_dir} 2>/dev/null || true"],
            timeout=30,
        )
        if code != 0:
            logger.warning("[sync] 列远端目录失败（code=%s）：%s", code, out[:200])
            return []
        return [line.strip() for line in out.splitlines() if line.strip() and line.strip() != "manifest.json"]
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[sync] 列远端目录异常: %s", exc)
        return []


def list_remote_video_sizes(ssh_target: str, remote_videos_dir: str) -> dict[str, int]:
    """List completed-looking remote video files and their byte sizes."""
    try:
        command = (f"find {remote_videos_dir} -mindepth 2 -maxdepth 2 "
                   "-name video.mp4 -printf '%h\\t%s\\n'")
        code, out = _run(["ssh", *_SSH_OPTS, ssh_target, command], timeout=30)
        if code != 0:
            logger.warning("[sync] 列远端文件大小失败（code=%s）：%s", code, out[:200])
            return {}
        rows = {}
        for line in out.splitlines():
            try:
                parent, raw_size = line.rsplit("\t", 1)
                rows[parent.rstrip("/").rsplit("/", 1)[-1]] = int(raw_size)
            except (ValueError, TypeError):
                logger.warning("[sync] 无法解析远端文件记录：%s", line[:200])
        return rows
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[sync] 列远端文件大小异常: %s", exc)
        return {}


def _push_one_dir(ssh_target: str, remote_videos_dir: str, local_dir: Path) -> str | None:
    """推一个视频目录，重试 _ATTEMPTS 次。成功返回 None，失败返回错误串。"""
    last_err = "unknown"
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            code, err = _run(
                ["scp", "-q", *_SSH_OPTS, "-r", str(local_dir), f"{ssh_target}:{remote_videos_dir}/"],
                timeout=_PER_FILE_TIMEOUT,
            )
            if code == 0:
                return None
            last_err = f"scp code={code}: {err[:200]}"
        except subprocess.TimeoutExpired:
            last_err = f"scp 超时（>{_PER_FILE_TIMEOUT}s，第 {attempt} 次）"
        except OSError as exc:
            last_err = f"scp 启动失败: {exc}"
        logger.warning("[sync %s] 第 %d 次推送失败: %s", local_dir.name, attempt, last_err)
    return last_err


def push_videos(
    ssh_target: str,
    remote_root: str,
    videos_dir: Path,
    processed_dir: Path,
    *,
    manifest_success_ids: list[str],
    manifest_path: Path,
) -> dict:
    """对账式推送：缺什么推什么。返回 {"pushed": n, "errors": [...], "checked": n}。"""
    remote_videos = f"{remote_root}/data/videos"
    local_ids = [i for i in manifest_success_ids if (videos_dir / i / "video.mp4").exists()]
    local_sizes = {aweme_id: (videos_dir / aweme_id / "video.mp4").stat().st_size
                   for aweme_id in local_ids}
    remote_sizes = list_remote_video_sizes(ssh_target, remote_videos)
    missing = compute_size_mismatches(local_sizes, remote_sizes)

    result: dict = {"pushed": 0, "errors": [], "checked": len(local_ids)}
    logger.info("[sync] 本地成功 %d 条 / 远端完整文件 %d 条 / 待推或修复 %d 条",
                len(local_ids), len(remote_sizes), len(missing))

    for aweme_id in missing:
        err = _push_one_dir(ssh_target, remote_videos, videos_dir / aweme_id)
        if err is None:
            result["pushed"] += 1
        else:
            result["errors"].append({"id": aweme_id, "error": err})

    # 台账与榜单产物（幂等覆盖）
    for src, dst in (
        (manifest_path, f"{ssh_target}:{remote_videos}/manifest.json"),
        (processed_dir / "trending_videos.jsonl", f"{ssh_target}:{remote_root}/data/processed/"),
        (processed_dir / "trending_videos.csv", f"{ssh_target}:{remote_root}/data/processed/"),
    ):
        if not src.exists():
            continue
        try:
            code, err = _run(["scp", "-q", *_SSH_OPTS, str(src), dst], timeout=120)
            if code != 0:
                result["errors"].append({"id": src.name, "error": err[:200]})
        except (subprocess.TimeoutExpired, OSError) as exc:
            result["errors"].append({"id": src.name, "error": str(exc)[:200]})

    if result["errors"]:
        logger.warning("[sync] %d 项同步失败（不致命，重跑流水线会自动补推）", len(result["errors"]))
    else:
        logger.info("[sync] 完成：推送 %d 个视频目录 + manifest + 榜单产物", result["pushed"])
    return result
