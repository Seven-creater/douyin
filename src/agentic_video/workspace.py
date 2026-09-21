# -*- coding: utf-8 -*-
"""v4 Workspace：版本化不可变 artifact store + 依赖图 + blocker 动态推导。

p0626 八项设计原则：
- 只存事实（status/SHA/依赖），blockers 动态 compute_blockers() 推导
- committed artifact 不可变；修复产生新版本（Git 语义）
- 显式 Goal（target_artifact + required_status），非 all_done
- 依赖链：改上游 → 下游自动 STALE
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash

WORKSPACE_VERSION = "workspace_v2"

STATUSES = ("not_started", "in_progress", "draft", "tested", "committed",
            "stale", "revoked")


class WorkspaceBlocked(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


class Workspace:
    """Production workspace：五个 artifact 阶段的版本化状态管理。

    目录 = Agent 的 git repo（run_v4/00_reference..05_edit）。
    workspace.json 只存事实；blockers/goal_satisfied 动态推导。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "workspace.json"
        self.trace_path = self.root / "agent_trace.jsonl"
        if self.state_path.is_file():
            self.state = json.loads(
                self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "schema_version": WORKSPACE_VERSION,
                "goal": {"target_artifact": "asset_graph",
                         "required_status": "committed"},
                "artifacts": {},   # {name: {versions: {vN: {sha, status}}, active_version}}
                "dependencies": {},  # {name: {upstream_name: required_sha_or_any}}
            }
            self._save()

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=1),
            encoding="utf-8")

    # ---- Artifact 状态查询 ----

    def get_status(self, name: str) -> str:
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            return "not_started"
        return str((artifact.get("versions") or {}).get(
            version, {}).get("status") or "not_started")

    def get_sha(self, name: str) -> str | None:
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            return None
        return (artifact.get("versions") or {}).get(version, {}).get("sha")

    def get_dependencies(self, name: str) -> dict[str, str]:
        return dict(self.state["dependencies"].get(name) or {})

    # ---- 版本化写入 ----

    def write_draft(self, name: str, data: Any, *,
                    metadata: dict[str, Any] | None = None) -> str:
        """写入新 draft 版本（不覆盖 committed 版本）。返回版本号。"""
        artifact = self.state["artifacts"].setdefault(
            name, {"versions": {}, "active_version": None})
        versions = artifact["versions"]
        next_num = len(versions) + 1
        version_id = f"v{next_num}"
        sha = json_hash(data)
        provenance = dict(metadata or {})
        provenance.setdefault("artifact_id", f"{name}:{version_id}")
        provenance.setdefault("artifact_type", name.split(":")[0])
        provenance.setdefault("version", version_id)
        provenance.setdefault("sha", sha)
        provenance.setdefault("producer_run", str(self.root))
        provenance.setdefault("derived_from", [])
        provenance.setdefault("created_by", "workspace")
        provenance.setdefault("schema_version", "artifact_provenance_v1")
        if provenance["sha"] != sha or provenance["version"] != version_id:
            raise WorkspaceBlocked("artifact_metadata_mismatch", name)
        versions[version_id] = {
            "sha": sha, "status": "draft", "provenance": provenance}
        artifact["active_version"] = version_id
        # 写入文件（JSON）
        stage_dir = self._stage_dir(name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / f"{version_id}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        self._save()
        return version_id

    def mark_tested(self, name: str) -> None:
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if version:
            artifact["versions"][version]["status"] = "tested"
            self._save()

    def commit(self, name: str) -> str:
        """Commit 当前 active 版本（不可变；旧 committed 版本变 stale）。

        commit 自带依赖快照冻结（P0-6：不依赖调用方记得先 record）。
        """
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            raise WorkspaceBlocked("commit_no_active_version", name)
        self.record_dependency_snapshot(name)
        # 旧 committed 版本变 stale
        for vid, info in (artifact.get("versions") or {}).items():
            if info.get("status") == "committed" and vid != version:
                info["status"] = "stale"
        artifact["versions"][version]["status"] = "committed"
        self._save()
        self._invalidate_downstream(name)
        return str(version)

    def set_status(self, name: str, status: str) -> None:
        if status not in STATUSES:
            raise WorkspaceBlocked("invalid_status", status)
        artifact = self.state["artifacts"].setdefault(
            name, {"versions": {}, "active_version": None})
        version = artifact.get("active_version")
        if version:
            artifact["versions"][version]["status"] = status
        self._save()

    def revoke(self, name: str, reason: str, *, reviewer: str = "human") -> None:
        """Revoke the active version without deleting its immutable record."""
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            raise WorkspaceBlocked("revoke_no_active_version", name)
        info = artifact["versions"][version]
        info["status"] = "revoked"
        info["revocation"] = {"reason": str(reason), "reviewer": str(reviewer)}
        self._save()
        self._invalidate_downstream(name)

    def get_provenance(self, name: str) -> dict[str, Any]:
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            return {}
        return dict((artifact.get("versions") or {}).get(
            version, {}).get("provenance") or {})

    # ---- 依赖管理 ----

    def set_dependency(self, name: str, upstream: str,
                       required: str = "committed") -> None:
        self.state["dependencies"].setdefault(name, {})[upstream] = required
        self._save()

    def _invalidate_downstream(self, changed: str) -> None:
        """上游 commit/变更 → 下游递归 stale（P0-6：BFS 全链传播）。

        eager 标记 + effective_status 懒推导双保险：即使这里漏标，
        goal_satisfied/can_run 用的 effective_status 也不会误判。
        """
        reverse: dict[str, list[str]] = {}
        for child, deps in self.state["dependencies"].items():
            for upstream in deps:
                reverse.setdefault(upstream, []).append(child)
        queue = [changed]
        visited = {changed}
        while queue:
            upstream = queue.pop(0)
            for child in reverse.get(upstream, []):
                if child in visited:
                    continue
                visited.add(child)
                # Always continue through graph-only or draft intermediates.
                # The previous implementation only enqueued a child after it
                # marked a committed version stale, so revocation stopped at
                # missing/draft nodes and never reached cross-run consumers.
                queue.append(child)
                artifact = self.state["artifacts"].get(child) or {}
                version = artifact.get("active_version")
                if version:
                    info = artifact["versions"][version]
                    if (info.get("status") == "committed"
                            and self.effective_status(child)
                            != "committed"):
                        info["status"] = "stale"
                        self._save()

    def effective_status(self, name: str,
                         _seen: set[str] | None = None) -> str:
        """有效状态：自身 committed 且全部上游仍有效且 SHA 快照
        匹配才算 committed，否则 stale（P0-6 懒推导层）。"""
        raw = self.get_status(name)
        if raw != "committed":
            return raw
        seen = _seen or set()
        if name in seen:  # 环保护
            return raw
        seen = set(seen)
        seen.add(name)
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        recorded = (artifact.get("versions") or {}).get(
            version, {}).get("depends_on_shas") or {}
        for upstream, required in (
                self.state["dependencies"].get(name) or {}).items():
            if required != "committed":
                continue
            if self.effective_status(upstream, seen) != "committed":
                return "stale"
            if (upstream in recorded
                    and recorded[upstream] != self.get_sha(upstream)):
                return "stale"
        return "committed"

    def record_dependency_snapshot(self, name: str) -> None:
        """commit 时记录当前依赖 SHA 快照（用于后续失效检测）。"""
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            return
        shas = {}
        for upstream in (self.state["dependencies"].get(name) or {}):
            shas[upstream] = self.get_sha(upstream)
        artifact["versions"][version]["depends_on_shas"] = shas
        self._save()

    # ---- Blocker 推导（动态计算，不存储）----

    def compute_blockers(self) -> list[dict[str, str]]:
        """从事实推导当前阻塞项。不存储——每次调用重新计算。

        用 effective_status（P0-6）：上游 stale 同样构成下游阻塞。
        """
        blockers = []
        for name, deps in self.state["dependencies"].items():
            status = self.effective_status(name)
            if status in ("committed",):
                continue
            for upstream, required in deps.items():
                upstream_status = self.effective_status(upstream)
                if required == "committed" and upstream_status != "committed":
                    blockers.append({
                        "artifact": name,
                        "reason": f"depends on {upstream} "
                                  f"(status={upstream_status}, "
                                  f"needs {required})"})
        return blockers

    def goal_satisfied(self) -> bool:
        goal = self.state.get("goal") or {}
        target = str(goal.get("target_artifact") or "")
        required = str(goal.get("required_status") or "committed")
        return self.effective_status(target) == required

    # ---- Production Map（Aider repo map 同思想）----

    def build_map(self) -> str:
        """精简状态摘要——Agent 看了就知道下一步。"""
        lines = [f"GOAL: {self.state['goal']['target_artifact']} → "
                 f"{self.state['goal']['required_status']}", ""]
        for name in sorted(self.state["artifacts"]):
            status = self.effective_status(name)
            sha = (self.get_sha(name) or "")[:8]
            version = (self.state["artifacts"][name].get(
                "active_version") or "—")
            lines.append(f"  {name:30s} {version:5s} {status:12s} {sha}")
        blockers = self.compute_blockers()
        if blockers:
            lines.append("")
            lines.append("BLOCKERS:")
            for b in blockers:
                lines.append(f"  - {b['artifact']}: {b['reason']}")
        lines.append("")
        return "\n".join(lines)

    # ---- Agent Trace ----

    def record_trace(self, step: int, action: dict[str, Any],
                     result: dict[str, Any],
                     test_report: dict[str, Any]) -> None:
        entry = {"step": step, "action": action, "result": result,
                 "tests": test_report}
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recent_trace(self, count: int = 3) -> list[dict[str, Any]]:
        if not self.trace_path.is_file():
            return []
        lines = self.trace_path.read_text(
            encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-count:]]

    # ---- Artifact 文件读取 ----

    def read_artifact(self, name: str) -> Any:
        artifact = self.state["artifacts"].get(name) or {}
        version = artifact.get("active_version")
        if not version:
            return None
        path = self._stage_dir(name) / f"{version}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _stage_dir(self, name: str) -> Path:
        # 映射 artifact 名到目录（如 screenplay → 01_screenplay）。
        # M2 资产链 artifact 名含冒号（asset:C0_master 等）→
        # 03_asset_studio/<name>/（每 artifact 独立子目录防版本文件撞名）
        if name.startswith("asset:"):
            return self.root / "03_asset_studio" / name[len("asset:"):]
        if name in ("shot_plan", "storyboard_frames"):
            return self.root / "04_storyboard" / name
        mapping = {
            "creative_dna": "00_reference",
            "screenplay": "01_screenplay",
            "asset_graph": "02_assets",
        }
        stage = mapping.get(name.split(":")[0], "99_misc")
        return self.root / stage
