# -*- coding: utf-8 -*-
"""Shot Plan 共享 Schema Contract（Skill 与 Validator 同源）。

M3-A：纯结构化"这一镜该拍什么"，零图片生成。
M3-B 才基于已 COMMIT 的资产生成 start/end frame pair。
"""
from __future__ import annotations

from typing import Any

SHOT_PLAN_SCHEMA_VERSION = "shot_plan_v1"

SHOT_SIZES = ("extreme_closeup", "closeup", "medium", "medium_full",
              "full", "wide")
CAMERA_MOVEMENTS = ("static", "push_in", "pull_back", "pan_left",
                    "pan_right", "tilt_up", "tilt_down", "tracking",
                    "handheld")


def _screenplay_beats(screenplay: dict[str, Any]) -> dict[str, str]:
    """beat_id → purpose（跨 scene 展平）。"""
    beats: dict[str, str] = {}
    for scene in screenplay.get("scenes") or []:
        for beat in scene.get("beats") or []:
            beats[str(beat.get("beat_id"))] = str(
                beat.get("purpose") or "")
    return beats


def _graph_asset_ids(asset_graph: dict[str, Any]) -> set[str]:
    return {str(a.get("asset_id"))
            for a in (asset_graph.get("assets") or [])}


def validate_shot_plan(plan: dict[str, Any],
                       screenplay: dict[str, Any],
                       asset_graph: dict[str, Any],
                       ) -> list[dict[str, str]]:
    """确定性校验（唯一权威）。返回 failures（空=过）。

    检查：schema_version / shot 唯一 / beat 覆盖 / 资产引用存在 /
    start→end 因果推进 / 时长预算合理 / camera 词表 / 单镜事件密度
    （visual_events ≤ 3）。
    """
    failures: list[dict[str, str]] = []
    if not isinstance(plan, dict):
        return [{"shot_id": "_root", "check": "schema",
                 "detail": "not a JSON object"}]
    if plan.get("schema_version") != SHOT_PLAN_SCHEMA_VERSION:
        failures.append({"shot_id": "_root", "check": "schema_version",
                         "detail": f"expected {SHOT_PLAN_SCHEMA_VERSION}, "
                                   f"got {plan.get('schema_version')}"})
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots:
        failures.append({"shot_id": "_root", "check": "shots",
                         "detail": "shots list missing or empty"})
        return failures

    beats = _screenplay_beats(screenplay)
    asset_ids = _graph_asset_ids(asset_graph)
    covered: set[str] = set()
    seen: set[str] = set()
    total_min = total_max = 0.0

    for shot in shots:
        shot_id = str(shot.get("shot_id") or "")
        if not shot_id:
            failures.append({"shot_id": "?", "check": "shot_id",
                             "detail": "missing"})
            continue
        if shot_id in seen:
            failures.append({"shot_id": shot_id, "check": "shot_id",
                             "detail": "duplicate"})
        seen.add(shot_id)

        beat_id = str(shot.get("beat_id") or "")
        if beat_id not in beats:
            failures.append({"shot_id": shot_id, "check": "beat_ref",
                             "detail": f"unknown beat {beat_id!r}"})
        else:
            covered.add(beat_id)

        # 资产引用存在性（characters/location/props/wardrobe/motion_refs）
        refs = (shot.get("characters") or []) \
            + ([shot["location"]] if shot.get("location") else []) \
            + (shot.get("props") or []) \
            + (shot.get("wardrobe") or []) \
            + (shot.get("motion_refs") or [])
        for ref in refs:
            if str(ref) not in asset_ids:
                failures.append({"shot_id": shot_id, "check": "asset_ref",
                                 "detail": f"unknown asset {ref!r}"})

        # 因果推进：start≠end 且都非空
        start = str(shot.get("start_state") or "")
        end = str(shot.get("end_state") or "")
        if not start or not end:
            failures.append({"shot_id": shot_id, "check": "causality",
                             "detail": "start_state/end_state empty"})
        elif start.strip() == end.strip():
            failures.append({"shot_id": shot_id, "check": "causality",
                             "detail": "end_state equals start_state "
                                       "(no narrative movement)"})

        # 时长预算
        budget = shot.get("duration_budget_s")
        if (not isinstance(budget, list) or len(budget) != 2
                or not all(isinstance(x, (int, float)) for x in budget)
                or budget[0] > budget[1] or budget[0] <= 0):
            failures.append({"shot_id": shot_id, "check": "duration",
                             "detail": f"invalid budget {budget!r}"})
        else:
            total_min += float(budget[0])
            total_max += float(budget[1])

        # camera 词表
        camera = shot.get("camera") or {}
        if camera.get("shot_size") not in SHOT_SIZES:
            failures.append({"shot_id": shot_id, "check": "shot_size",
                             "detail": f"invalid: {camera.get('shot_size')}"})
        if camera.get("movement") not in CAMERA_MOVEMENTS:
            failures.append({"shot_id": shot_id, "check": "movement",
                             "detail": f"invalid: {camera.get('movement')}"})

        # 单镜事件密度（一镜塞过多事件=剪辑失能）
        events = shot.get("visual_events") or []
        if len(events) > 3:
            failures.append({"shot_id": shot_id, "check": "event_density",
                             "detail": f"{len(events)} events in one shot "
                                       f"(max 3)"})

    # beat 覆盖：每个 beat 至少一个 shot 承载
    for beat_id in beats:
        if beat_id not in covered:
            failures.append({"shot_id": "_root", "check": "beat_coverage",
                             "detail": f"beat {beat_id} has no shot"})

    # 总时长 vs 剧本 target_duration（±30%）
    target = float(screenplay.get("target_duration") or 0)
    if target > 0 and (total_min > target * 1.3 or total_max < target * 0.7):
        failures.append({"shot_id": "_root", "check": "total_duration",
                         "detail": f"shots sum [{total_min:.1f},"
                                   f"{total_max:.1f}]s vs target "
                                   f"{target:.1f}s"})
    return failures


SHOT_PLAN_FORMAT_SPEC = """{
  "schema_version": "shot_plan_v1",
  "shots": [
    {"shot_id": "SH01", "beat_id": "B1",
     "narrative_role": "situation_setup",
     "duration_budget_s": [2.0, 3.5],
     "characters": ["C0"],
     "location": "L01",
     "props": ["P01"],
     "wardrobe": ["W01"],
     "motion_refs": ["A01"],
     "camera": {"shot_size": "medium", "movement": "push_in"},
     "start_state": "镜头开始时的可观察状态",
     "end_state": "镜头结束时的可观察状态（必须≠start）",
     "visual_events": ["至多3个可拍事件"]}
  ]
}
必填：schema_version="shot_plan_v1"；shot_id/beat_id 唯一且 beat 存在
于剧本；所有资产 id 必须存在于 asset_graph；duration_budget_s=[min,max]
且 min≤max；shot_size ∈ extreme_closeup|closeup|medium|medium_full|full|
wide；movement ∈ static|push_in|pull_back|pan_left|pan_right|tilt_up|
tilt_down|tracking|handheld；visual_events ≤ 3。"""
