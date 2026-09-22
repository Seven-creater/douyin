"""Exact-parent provenance helpers compatible with :class:`Workspace`."""
from __future__ import annotations

import re
from typing import Any

from src.agentic_video.creative_dna_v3.schema import PROVENANCE_SCHEMA_VERSION
from src.agentic_video.creative_dna_v3.validators import DNAV3Error

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_parent_rows(rows: list[dict[str, str]]) -> None:
    if not isinstance(rows, list) or not rows:
        raise DNAV3Error("lineage_parents_missing")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"artifact_id", "sha"}:
            raise DNAV3Error("lineage_parent_invalid")
        artifact_id = row.get("artifact_id")
        sha = row.get("sha")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise DNAV3Error("lineage_parent_id_invalid")
        if artifact_id in seen:
            raise DNAV3Error("lineage_parent_duplicate", artifact_id)
        seen.add(artifact_id)
        if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
            raise DNAV3Error("lineage_parent_sha_invalid", artifact_id)


def _metadata(*, artifact_id: str, artifact_type: str, sha: str,
              parents: list[dict[str, str]], producer_run: str,
              created_by: str) -> dict[str, Any]:
    validate_parent_rows(parents)
    if not _SHA_RE.fullmatch(sha):
        raise DNAV3Error("lineage_artifact_sha_invalid", artifact_id)
    if not producer_run or not created_by:
        raise DNAV3Error("lineage_producer_invalid", artifact_id)
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "sha": sha,
        "producer_run": producer_run,
        "derived_from": [dict(row) for row in parents],
        "created_by": created_by,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
    }


def build_audit_provenance(audit: dict[str, Any], *, producer_run: str,
                           created_by: str) -> dict[str, Any]:
    parents = [
        {"artifact_id": row["artifact_id"], "sha": row["sha"]}
        for row in audit.get("input_artifacts") or []
    ]
    return _metadata(
        artifact_id="creative_dna_audit_v3",
        artifact_type="creative_dna_audit",
        sha=str(audit.get("artifact_sha") or ""),
        parents=parents,
        producer_run=producer_run,
        created_by=created_by,
    )


def build_spec_provenance(*, spec_sha: str, audit_artifact_id: str,
                          audit_sha: str, producer_run: str,
                          created_by: str) -> dict[str, Any]:
    return _metadata(
        artifact_id="creative_spec_v1",
        artifact_type="creative_spec",
        sha=spec_sha,
        parents=[{"artifact_id": audit_artifact_id, "sha": audit_sha}],
        producer_run=producer_run,
        created_by=created_by,
    )
