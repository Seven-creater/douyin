# -*- coding: utf-8 -*-
"""M2-A workspace 生产初始化（P0-7：依赖 DAG 注册不依赖夜链脚本）。"""
from __future__ import annotations

from src.agentic_video.workspace import Workspace


def prepare_m2a_workspace(ws: Workspace, asset_id: str = "C0",
                          *, night_goal: bool = True) -> None:
    """注册 M2-A 完整依赖链 + 设定目标。

    链：asset_graph → asset:C0_master → asset:C0_views →
        asset:C0_candidate →（human_approval:C0 + candidate）→ asset:C0

    night_goal=True：夜链无人值守目标 = candidate COMMITTED（机器验收
    完成即停，最终 COMMIT 必须等人审）。
    """
    master = f"asset:{asset_id}_master"
    views = f"asset:{asset_id}_views"
    candidate = f"asset:{asset_id}_candidate"
    final = f"asset:{asset_id}"

    ws.set_dependency(master, "asset_graph", "committed")
    ws.set_dependency(views, master, "committed")
    ws.set_dependency(candidate, views, "committed")
    ws.set_dependency(final, candidate, "committed")
    ws.set_dependency(final, f"human_approval:{asset_id}", "committed")

    if night_goal:
        ws.state["goal"] = {"target_artifact": candidate,
                            "required_status": "committed"}
        ws._save()
