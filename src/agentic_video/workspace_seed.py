# -*- coding: utf-8 -*-
"""从既往 run 的 committed artifact 种子新 workspace（M2 链复用 M1 产物）。"""
from __future__ import annotations

import json
from pathlib import Path

from src.agentic_video.workspace import Workspace


def seed_from_previous_run(ws: Workspace, prev_root: Path,
                           artifact_names: tuple[str, ...]) -> list[str]:
    """读取 prev run 中各 artifact 的 committed（或 active）版本，
    write_draft + commit 到 ws。返回成功种子的 artifact 列表。"""
    seeded: list[str] = []
    prev_state = json.loads(
        (Path(prev_root) / "workspace.json").read_text(encoding="utf-8"))
    for name in artifact_names:
        info = (prev_state.get("artifacts") or {}).get(name) or {}
        active = info.get("active_version")
        if not active:
            continue
        # 优先 committed 版本，否则 active
        version = active
        for vid, vinfo in (info.get("versions") or {}).items():
            if vinfo.get("status") == "committed":
                version = vid
                break
        mapping = {"creative_dna": "00_reference",
                   "screenplay": "01_screenplay",
                   "asset_graph": "02_assets"}
        stage = mapping.get(name, "99_misc")
        src = Path(prev_root) / stage / f"{version}.json"
        if not src.is_file():
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        source_version = (info.get("versions") or {}).get(version) or {}
        source_sha = str(source_version.get("sha") or "")
        if source_version.get("status") == "revoked":
            continue
        ws.write_draft(name, data, metadata={
            "producer_run": str(ws.root),
            "created_by": "workspace_seed",
            "derived_from": [{
                "artifact_id": ((source_version.get("provenance") or {}).get(
                    "artifact_id") or f"{name}:{version}"),
                "artifact_type": name.split(":")[0],
                "version": version,
                "sha": source_sha,
                "producer_run": str(Path(prev_root)),
            }],
        })
        ws.record_dependency_snapshot(name)
        ws.commit(name)
        seeded.append(name)
    return seeded
