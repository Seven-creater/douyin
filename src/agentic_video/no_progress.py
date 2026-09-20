# -*- coding: utf-8 -*-
"""No-Progress Detection（p0626 修正 5/6：世界状态增量监控 + 动作空间收缩）。

监控四类增量：artifact/validator/dependency/goal。连续 N 步全部为零 →
NO_PROGRESS → 通知 Controller 动作空间收缩（只能选 repair/rewrite/
ask_for_human/stop，不能继续 inspect）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from src.agentic_video.workspace import Workspace

DEFAULT_STAGNATION_LIMIT = 3


class NoProgressDetector:
    """世界状态增量检测器（非"同一个 skill 连续 N 次"）。"""

    def __init__(self, limit: int = DEFAULT_STAGNATION_LIMIT) -> None:
        self.limit = limit
        self._consecutive_zero = 0

    def compute_state_hash(self, workspace: Workspace) -> str:
        """全量 artifact 状态 + blocker 的指纹。"""
        fingerprint = {
            "artifacts": {
                name: {
                    "active": info.get("active_version"),
                    "status": (
                        (info.get("versions") or {}).get(
                            info.get("active_version") or "", {}
                        ).get("status")
                    ),
                    "sha": (
                        (info.get("versions") or {}).get(
                            info.get("active_version") or "", {}
                        ).get("sha")
                    ),
                }
                for name, info in workspace.state.get("artifacts", {}).items()
            },
            "blockers": workspace.compute_blockers(),
        }
        return hashlib.sha256(json.dumps(
            fingerprint, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:16]

    def compute_failure_hash(self, failures: list[dict[str, Any]]) -> str:
        return hashlib.sha256(json.dumps(
            failures, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:16]

    def step(self, workspace: Workspace,
             last_failure: dict[str, Any] | None) -> dict[str, Any]:
        """每步调用；返回 progress 报告。连续零增量达阈值 → stalled=True。"""
        state_hash = self.compute_state_hash(workspace)
        failure_hash = self.compute_failure_hash(
            (last_failure or {}).get("failures") or [])

        progress = (
            state_hash != getattr(self, "_last_state_hash", None)
            or failure_hash != getattr(self, "_last_failure_hash", None))

        if progress:
            self._consecutive_zero = 0
        else:
            self._consecutive_zero += 1

        self._last_state_hash = state_hash
        self._last_failure_hash = failure_hash

        stalled = self._consecutive_zero >= self.limit
        return {
            "state_hash": state_hash,
            "failure_hash": failure_hash,
            "progress": progress,
            "consecutive_stagnant_steps": self._consecutive_zero,
            "stalled": stalled,
            "event": "NO_PROGRESS" if stalled else None,
        }

    def restricted_actions(self) -> list[str]:
        """NO_PROGRESS 后动作空间收缩：只能 repair/rewrite/stop。"""
        return ["repair_screenplay", "write_screenplay",
                "repair_asset_graph", "stop"]
