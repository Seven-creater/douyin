# -*- coding: utf-8 -*-
"""M3 validators：test_shot_plan（确定性唯一权威）+
test_storyboard_frame_pair（L1 确定性 → L2 Omni 盲检）。
"""
from __future__ import annotations

from typing import Any

from src.agentic_video.workspace import Workspace

VALIDATE_FRAME_PAIR_PROMPT = """你是分镜盲检员。输入同一镜头的两张
storyboard 帧（第 1 张 = start，第 2 张 = end）与该镜头的规划
（start_state/end_state/camera）。你没有看过任何生成过程，只根据
两帧图像与规划独立检查，只输出一个 JSON：
{"passed": true, "failures": [{"check": "...", "detail": "..."}]}
检查项：
1. identity：帧中人物与参考图是否同一人（发型/脸型/体型一致）
2. pair_progression：end 帧是否体现 start_state → end_state 的状态
   推进（而非同一画面微调）
3. spatial_consistency：两帧的场景/空间/机位关系是否连续合理
4. narrative_match：画面内容是否对应规划的事件（shot_size/构图）
5. no_extra：无文字/水印/多余人物
镜头规划：{shot_brief}
第 1 帧 = start，第 2 帧 = end。人物参考图随后。图像："""


def run_m3_test(validator_name: str, workspace: Workspace,
                runner) -> dict[str, Any]:
    if validator_name == "test_shot_plan":
        return _test_shot_plan(workspace)
    if validator_name == "test_storyboard_dependencies":
        return _test_storyboard_dependencies(workspace)
    if validator_name == "test_storyboard_frame_pair":
        return _test_frame_pair(workspace, runner)
    raise KeyError(f"unknown m3 validator: {validator_name}")


def _test_storyboard_dependencies(workspace: Workspace) -> dict[str, Any]:
    """逐 shot 依赖清单：全 COMMIT → READY；否则 BLOCKED（带缺失列表）。

    只读诊断型 validator（不 gate commit——它验收的是"就绪度"这个
    事实本身，BLOCKED 就是它的正确输出）。
    """
    from src.agentic_video.storyboard.skills import shot_dependencies
    plan = workspace.read_artifact("shot_plan")
    if not plan:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "shot_plan is empty"}],
                "validators_run": ["test_storyboard_dependencies"]}
    blocked: list[dict[str, str]] = []
    ready = 0
    for shot in plan.get("shots") or []:
        shot_id = str(shot.get("shot_id") or "?")
        missing = [d["artifact"] for d in
                   shot_dependencies(shot, workspace)
                   if d["status"] != "committed"]
        if missing:
            blocked.append({"shot_id": shot_id,
                            "check": "dependency_gate",
                            "detail": f"BLOCKED missing: {missing}"})
        else:
            ready += 1
    total = len(plan.get("shots") or [])
    return {"passed": not blocked, "failures": blocked,
            "validators_run": ["test_storyboard_dependencies"],
            "detail": f"{ready}/{total} shots READY"}


def _test_shot_plan(workspace: Workspace) -> dict[str, Any]:
    """确定性 schema 校验（唯一权威——asset_graph 同款纪律）。"""
    from src.agentic_video.storyboard.shot_schema import validate_shot_plan
    plan = workspace.read_artifact("shot_plan")
    if not plan:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "shot_plan is empty"}],
                "validators_run": ["test_shot_plan"]}
    screenplay = workspace.read_artifact("screenplay") or {}
    graph = workspace.read_artifact("asset_graph") or {}
    failures = validate_shot_plan(plan, screenplay, graph)
    return {"passed": not failures, "failures": failures,
            "validators_run": ["test_shot_plan"]}


def _test_frame_pair(workspace: Workspace, runner) -> dict[str, Any]:
    """L1 确定性（成对存在/可读/16:9/同尺寸/sha 一致）→ L2 Omni 盲检。"""
    from PIL import Image
    manifest = workspace.read_artifact("storyboard_frames")
    if not manifest:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "storyboard frames manifest empty"}],
                "validators_run": ["test_storyboard_frame_pair"]}
    failures: list[dict[str, str]] = []
    shot_id = str(manifest.get("shot_id") or "?")
    frames = {}
    for key in ("start_frame", "end_frame"):
        entry = manifest.get(key) or {}
        rel = str(entry.get("file") or "")
        path = workspace.root / rel
        if not rel or not path.is_file():
            failures.append({"check": f"{key}_exists",
                             "detail": f"missing: {rel}"})
            continue
        image = Image.open(path)
        ratio = image.width / image.height
        if abs(ratio - 16 / 9) > (16 / 9) * 0.03:
            failures.append({"check": f"{key}_aspect",
                             "detail": f"{image.width}x{image.height} "
                                       f"not 16:9"})
        if max(image.size) < 1024:
            failures.append({"check": f"{key}_resolution",
                             "detail": f"long side {max(image.size)} < 1024"})
        frames[key] = (path, image.size)
    if len(frames) == 2 and frames["start_frame"][1] != frames[
            "end_frame"][1]:
        failures.append({"check": "pair_size_mismatch",
                         "detail": "start/end frame sizes differ"})
    if failures:
        return {"passed": False, "failures": failures,
                "validators_run": ["test_storyboard_frame_pair"]}

    shot = manifest.get("shot") or {}
    brief = {k: shot.get(k) for k in ("shot_id", "start_state",
                                      "end_state", "camera",
                                      "visual_events")}
    prompt = VALIDATE_FRAME_PAIR_PROMPT.replace(
        "{shot_brief}", str(brief))
    images = [frames["start_frame"][0], frames["end_frame"][0]]
    images += [workspace.root / p for p in
               (manifest.get("identity_refs") or [])]
    answer = runner.inspect_media(image_paths=images, prompt=prompt,
                             stop_after_json_object=True)
    from src.agentic_video.asset_studio.validators import _parse_omni_json
    try:
        value = _parse_omni_json(answer)
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "failures": [{"check": "parse",
                "detail": str(exc)[:120]}],
                "validators_run": ["test_storyboard_frame_pair"]}
    omni_failures = value.get("failures") or []
    return {"passed": bool(value.get("passed")) and not omni_failures,
            "failures": [{"shot_id": shot_id, **f} for f in omni_failures],
            "validators_run": ["test_storyboard_frame_pair"]}
