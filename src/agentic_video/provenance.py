"""Artifact lineage, approval binding and production-source preflight."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.manifest import json_hash
from src.agentic_video.workspace import Workspace, WorkspaceBlocked

PROVENANCE_VERSION = "artifact_provenance_v1"
APPROVAL_POLICY_VERSION = "approval_policy_v1"


def parent_sha_rows(workspace: Workspace, names: Iterable[str]) -> list[dict[str, str]]:
    rows = []
    for name in names:
        sha = workspace.get_sha(name)
        if not sha:
            raise WorkspaceBlocked("approval_parent_missing", name)
        rows.append({"artifact_id": name, "sha": sha})
    return rows


def validate_approval_binding(approval: dict[str, Any], *, candidate_id: str,
                              candidate_sha: str,
                              parent_shas: list[dict[str, str]],
                              policy_version: str = APPROVAL_POLICY_VERSION
                              ) -> list[str]:
    problems = []
    if approval.get("candidate_id") != candidate_id:
        problems.append("candidate_id_mismatch")
    if approval.get("candidate_sha") != candidate_sha:
        problems.append("candidate_sha_mismatch")
    if approval.get("approval_policy_version") != policy_version:
        problems.append("approval_policy_version_mismatch")
    expected = {(str(row.get("artifact_id")), str(row.get("sha")))
                for row in parent_shas}
    actual = {(str(row.get("artifact_id")), str(row.get("sha")))
              for row in approval.get("parent_shas") or []}
    if actual != expected:
        problems.append("parent_shas_mismatch")
    if approval.get("decision") not in {"approve", "reject"}:
        problems.append("decision_invalid")
    if not str(approval.get("reviewer") or ""):
        problems.append("reviewer_missing")
    return problems


def load_revocations(path: Path | None) -> set[str]:
    if path is None or not Path(path).is_file():
        return set()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row.get("sha")) for row in value.get("revocations") or []
            if row.get("sha")}


def assert_production_sources_current(
        workspace: Workspace, required_names: Iterable[str], *,
        revocation_path: Path | None = None) -> None:
    """Fail closed for legacy copied artifacts with no exact source identity."""
    revoked = load_revocations(revocation_path)
    for name in required_names:
        if workspace.effective_status(name) != "committed":
            raise WorkspaceBlocked("production_source_not_committed", name)
        provenance = workspace.get_provenance(name)
        if provenance.get("schema_version") != PROVENANCE_VERSION:
            raise WorkspaceBlocked("production_source_unprovenanced", name)
        current_sha = workspace.get_sha(name)
        sources = [current_sha] + [str(row.get("sha")) for row in
                                   provenance.get("derived_from") or []]
        if any(value in revoked for value in sources if value):
            raise WorkspaceBlocked("production_source_revoked", name)


def build_lineage_report(workspace: Workspace) -> dict[str, Any]:
    artifacts = []
    for name, artifact in sorted(workspace.state.get("artifacts", {}).items()):
        for version, info in sorted((artifact.get("versions") or {}).items()):
            artifacts.append({
                "name": name, "version": version, "sha": info.get("sha"),
                "status": info.get("status"),
                "effective_status": (workspace.effective_status(name)
                                     if artifact.get("active_version") == version
                                     else info.get("status")),
                "provenance": info.get("provenance") or {},
                "depends_on_shas": info.get("depends_on_shas") or {},
                "revocation": info.get("revocation"),
            })
    report = {"schema_version": "lineage_report_v1",
              "workspace": str(workspace.root), "artifacts": artifacts,
              "dependencies": workspace.state.get("dependencies") or {}}
    report["report_sha"] = json_hash(report)
    return report


def propagate_revocation(run_roots: Iterable[Path], source_shas: Iterable[str]
                         ) -> list[dict[str, str]]:
    """Mark cross-run copies stale by exact source SHA, then recurse locally."""
    revoked = set(map(str, source_shas))
    changed: list[dict[str, str]] = []
    for root in map(Path, run_roots):
        if not (root / "workspace.json").is_file():
            continue
        workspace = Workspace(root)
        for name, artifact in workspace.state.get("artifacts", {}).items():
            version = artifact.get("active_version")
            info = (artifact.get("versions") or {}).get(version) if version else None
            if not info or info.get("status") in {"stale", "revoked"}:
                continue
            provenance = info.get("provenance") or {}
            inputs = {str(row.get("sha")) for row in
                      provenance.get("derived_from") or [] if row.get("sha")}
            if not inputs.intersection(revoked):
                continue
            info["status"] = "stale"
            workspace._save()
            workspace._invalidate_downstream(name)
            changed.append({"workspace": str(root), "artifact": name,
                            "version": str(version), "status": "stale"})
    return changed
