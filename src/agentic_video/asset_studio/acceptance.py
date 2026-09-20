# -*- coding: utf-8 -*-
"""M2-A 资产验收报告生成器（晨检消费；answer to "哪张图适合什么任务"）。

不只报告"有四张图"，报告每张视图的 pass/repair 史 + sha +
recommended_reference_roles（M3/M4 的 Reference Router 直接吃）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.workspace import Workspace

# 角色 → 候选视图（按优先序；报告按实际可用性裁剪）
ROLE_VIEW_CANDIDATES = {
    "face_identity": ("face", "front"),
    "frontal_body": ("front",),
    "side_body": ("profile", "front"),
    "back_body": ("back",),
    "full_turnaround": ("front", "profile", "back", "face"),
}


def _repair_counts(workspace: Workspace) -> dict[str, int]:
    """从 agent_trace 统计 per-view 修复次数。"""
    counts: dict[str, int] = {}
    path = workspace.trace_path
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            entry = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        action = entry.get("action") or {}
        if action.get("skill") == "repair_character_view":
            view = str(action.get("target")
                       or entry.get("result", {}).get("target") or "?")
            counts[view] = counts.get(view, 0) + 1
    return counts


def build_acceptance_report(workspace: Workspace,
                            asset_id: str = "C0") -> dict[str, Any]:
    """生成资产验收报告（不落盘；调用方决定写哪）。"""
    final = workspace.read_artifact(f"asset:{asset_id}") or {}
    views_manifest = workspace.read_artifact(f"asset:{asset_id}_views") or {}
    master = workspace.read_artifact(f"asset:{asset_id}_master") or {}
    repairs = _repair_counts(workspace)
    committed = workspace.get_status(f"asset:{asset_id}") == "committed"

    views_report: dict[str, Any] = {}
    for view, entry in (views_manifest.get("views") or {}).items():
        sha4k = entry.get("sha4k") or entry.get("sha")
        views_report[view] = {
            "status": "pass" if committed else "pending",
            "sha": sha4k,
            "file": entry.get("file4k") or entry.get("file"),
            "repair_count": repairs.get(view, 0)}

    # 推荐参考角色：按可用性裁剪候选序；修复过的视图标注
    available = set(views_report)
    repaired = {v for v, c in repairs.items() if c > 0}
    recommended: dict[str, list[str]] = {}
    for role, candidates in ROLE_VIEW_CANDIDATES.items():
        usable = [v for v in candidates if v in available]
        if usable:
            recommended[role] = usable
    # 背面视图漂移风险 → 明示降级（back 修过 → 标记 fallback 到 front）
    if "back" in repaired and "back_body" in recommended:
        recommended["back_body"] = ["back(repaired,verify)", "front"]

    return {
        "asset_id": asset_id,
        "overall": "committed" if committed else "not_committed",
        "identity_description": final.get("identity_description")
        or master.get("identity_description"),
        "master": {
            "status": "pass" if master else "missing",
            "sha": (master.get("master") or {}).get("sha"),
            "generator": (master.get("master") or {}).get("generator")},
        "views": views_report,
        "identity_consistency": "pass" if committed else "pending",
        "human_review": "pending",
        "repair_history": repairs,
        "recommended_reference_roles": recommended,
        "identity_sheet": (final.get("sheet") or {}).get("file"),
    }


def write_acceptance_report(workspace: Workspace, asset_id: str = "C0"
                            ) -> Path:
    report = build_acceptance_report(workspace, asset_id)
    out = workspace.root / f"acceptance_{asset_id}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return out
