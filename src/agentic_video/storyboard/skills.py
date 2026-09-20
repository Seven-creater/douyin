# -*- coding: utf-8 -*-
"""M3 Skills：plan_storyboard / inspect_shot_plan / repair_shot_plan /
generate_storyboard_frame（动态依赖门 + Reference Router）。

依赖纪律：plan 可在资产未完成时运行（纯文本），frame 生成必须
全部所需资产 COMMIT——少一个 → STORYBOARD:SHxx BLOCKED。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.skills.registry import (
    SkillBlocked, SkillRegistry, SkillSpec)
from src.agentic_video.storyboard.shot_schema import SHOT_PLAN_FORMAT_SPEC
from src.agentic_video.workspace import Workspace

PLAN_STORYBOARD_PROMPT = """你是分镜规划师。输入：锁定剧本（scenes/
beats，每个 beat 有 purpose/visual_action/editing_role）+ Creative DNA
（节奏先验与剪辑语法）+ Asset Graph（可用资产 id 清单）。任务：把剧本
编译成 Shot Plan（22 秒总量约 6-10 镜）。要求：
1. 每个 beat 至少一个 shot 承载；counter_evidence/decisive_action
   beat 优先给决定性镜头（closeup 或 push_in）
2. start_state/end_state 必须可观察、可拍，且构成明确状态推进：
   推荐用 per-entity dict（键名 "<asset_id>.<属性>"，如
   "C0.pose"/"P01.state"），至少一个 entity 态变化
3. 所有资产 id 必须来自 asset_graph；特写镜头（closeup/
   extreme_closeup 且不依赖场景）location 可为 null
4. duration_budget_s 各镜总和落在剧本 target_duration ±20%
5. visual_events 每镜至多 3 个
输出必须逐字段遵循此 schema（顶层就是对象本身，不要外层包装）：
""" + SHOT_PLAN_FORMAT_SPEC + """
输入："""

REPAIR_SHOT_PLAN_PROMPT = """你是分镜医生。输入：当前 Shot Plan +
schema 校验失败明细。只修复失败的 shot（局部 patch），已 PASS 的
shot 原样保留。输出修复后的完整 Shot Plan（结构同输入）。
输入："""


def select_references(shot: dict[str, Any], workspace: Workspace
                      ) -> tuple[list[str], list[str]]:
    """Reference Router v1：按 shot 需求选已 COMMIT 的资产视图文件。

    特写用 face 视图，其余用 front；location 用其 master。
    返回 (ref_rel_paths, missing_artifacts)——missing 非空即 BLOCKED。
    """
    refs: list[str] = []
    missing: list[str] = []
    size = (shot.get("camera") or {}).get("shot_size") or "medium"
    want_face = size in ("closeup", "extreme_closeup")
    for cid in shot.get("characters") or []:
        art = f"asset:{cid}_views"
        if workspace.get_status(art) != "committed":
            missing.append(f"{art}({workspace.get_status(art)})")
            continue
        views = (workspace.read_artifact(art) or {}).get("views") or {}
        key = "face" if want_face and "face" in views else "front"
        entry = views.get(key) or {}
        rel = entry.get("file4k") or entry.get("file")
        if rel:
            refs.append(str(rel))
        else:
            missing.append(f"{art}(no views)")
    loc = shot.get("location")
    if loc:
        art = f"asset:{loc}"
        if workspace.get_status(art) != "committed":
            missing.append(f"{art}({workspace.get_status(art)})")
        else:
            master = (workspace.read_artifact(art) or {}
                      ).get("master") or {}
            if master.get("file"):
                refs.append(str(master["file"]))
    return refs, missing


def shot_dependencies(shot: dict[str, Any], workspace: Workspace
                      ) -> list[dict[str, str]]:
    """单 shot 的依赖清单（人物视图 + location master）。"""
    deps: list[dict[str, str]] = []
    for cid in shot.get("characters") or []:
        art = f"asset:{cid}_views"
        deps.append({"artifact": art,
                     "status": workspace.get_status(art)})
    loc = shot.get("location")
    if loc:
        art = f"asset:{loc}"
        deps.append({"artifact": art,
                     "status": workspace.get_status(art)})
    return deps


def pick_pilot_shot(plan: dict[str, Any], workspace: Workspace
                    ) -> dict[str, Any]:
    """M3-B 单 Shot Pilot 选镜：优先 counter_evidence/decisive_action，
    排序键 = (缺资产数, 复杂度[人物+道具数], 角色权重)——简单且关键。"""
    rank = {"counter_evidence": 0, "decisive_action": 1}
    best: dict[str, Any] | None = None
    best_key = (99, 99, 99)
    for shot in plan.get("shots") or []:
        role = str(shot.get("narrative_role") or "")
        if role not in rank:
            continue
        _, missing = select_references(shot, workspace)
        complexity = len(shot.get("characters") or []) + len(
            shot.get("props") or [])
        key = (len(missing), complexity, rank[role])
        if best is None or key < best_key:
            best, best_key = shot, key
    if best is None:
        shots = plan.get("shots") or []
        if not shots:
            raise ValueError("shot_plan has no shots")
        best = min(shots, key=lambda s: (
            len(select_references(s, workspace)[1]),
            len(s.get("characters") or []) + len(s.get("props") or [])))
    return best


def build_m3_registry(*, runner=None, storyboard_backend=None
                      ) -> SkillRegistry:
    """构建 M3 四个 skills。runner=Omni 池（plan/repair 文本）。"""
    registry = SkillRegistry()

    def _ask(prompt: str, max_tokens: int = 4096) -> dict[str, Any]:
        from src.agentic_video.reference_program_v9 import _parse_one_object
        answer = runner.ask(prompt, max_new_tokens=max_tokens,
                            stop_after_json_object=True)
        raw = getattr(answer, "text", str(answer))
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        value = _parse_one_object(text, stage="m3_skill")
        if isinstance(value, dict) and "shots" not in value:
            # 剥镜像包装（Omni 照抄 payload 外层键——Run A-v2 第 N 课）
            for key in ("shot_plan", "current_shot_plan",
                        "repaired_shot_plan"):
                inner = value.get(key)
                if isinstance(inner, dict):
                    return inner
        return value

    def run_plan_storyboard(workspace: Workspace, **kw) -> dict:
        screenplay = workspace.read_artifact("screenplay")
        dna = workspace.read_artifact("creative_dna") or {}
        graph = workspace.read_artifact("asset_graph")
        if not screenplay or not graph:
            raise ValueError("screenplay/asset_graph not in workspace")
        asset_ids = [str(a.get("asset_id"))
                     for a in graph.get("assets") or []]
        payload = json.dumps(
            {"screenplay": screenplay, "creative_dna": dna,
             "available_asset_ids": asset_ids},
            ensure_ascii=False, separators=(",", ":"))
        plan = _ask(PLAN_STORYBOARD_PROMPT + payload, max_tokens=6144)
        version = workspace.write_draft("shot_plan", plan)
        return {"artifact": "shot_plan", "version": version,
                "action": "planned"}

    def run_repair_shot_plan(workspace: Workspace, **kw) -> dict:
        plan = workspace.read_artifact("shot_plan")
        if not plan:
            raise ValueError("shot_plan not in workspace")
        failures = kw.get("test_failures") or []
        payload = json.dumps(
            {"current_shot_plan": plan, "failed_tests": failures},
            ensure_ascii=False, separators=(",", ":"))
        repaired = _ask(REPAIR_SHOT_PLAN_PROMPT + payload, max_tokens=6144)
        version = workspace.write_draft("shot_plan", repaired)
        return {"artifact": "shot_plan", "version": version,
                "action": "repaired"}

    def run_inspect_shot_plan(workspace: Workspace, **kw) -> dict:
        return {"artifact": "shot_plan", "action": "inspected",
                "data": workspace.read_artifact("shot_plan")}

    def run_generate_frames(workspace: Workspace, **kw) -> dict:
        plan = workspace.read_artifact("shot_plan")
        if not plan:
            raise ValueError("shot_plan not in workspace")
        target = str(kw.get("target") or "")
        shot = next((s for s in plan.get("shots") or []
                     if str(s.get("shot_id")) == target), None)
        if shot is None:
            shot = pick_pilot_shot(plan, workspace)
        shot_id = str(shot.get("shot_id"))
        refs, missing = select_references(shot, workspace)
        if missing:
            # 动态依赖门：少一个 COMMIT 资产都不许烧 GPU
            raise SkillBlocked(
                f"STORYBOARD:{shot_id} BLOCKED",
                f"missing committed assets: {missing}")
        if storyboard_backend is None:
            raise SkillBlocked(
                f"STORYBOARD:{shot_id} BLOCKED",
                "no storyboard backend wired (M3-B)")
        files_dir = workspace.root / "04_storyboard" / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        def _next(base: str) -> Path:
            n = 1
            while (files_dir / f"{base}_v{n}.png").is_file():
                n += 1
            return files_dir / f"{base}_v{n}.png"

        start_path = _next(f"{shot_id}_start")
        storyboard_backend.generate_start_frame(
            shot, [workspace.root / r for r in refs], start_path)
        end_path = _next(f"{shot_id}_end")
        storyboard_backend.generate_end_frame(
            shot, start_path, [workspace.root / r for r in refs], end_path)

        from src.agentic_video.asset_studio.image_io import file_sha256
        manifest = {
            "shot_id": shot_id, "shot": shot,
            "identity_refs": refs,
            "start_frame": {"file": str(
                start_path.relative_to(workspace.root)).replace("\\", "/"),
                "sha": file_sha256(start_path)},
            "end_frame": {"file": str(
                end_path.relative_to(workspace.root)).replace("\\", "/"),
                "sha": file_sha256(end_path)}}
        version = workspace.write_draft("storyboard_frames", manifest)
        return {"artifact": "storyboard_frames", "version": version,
                "action": "frames_generated", "target": shot_id}

    registry.register(SkillSpec(
        name="plan_storyboard",
        description="Compile screenplay+DNA+asset_graph into Shot Plan",
        inputs=["screenplay", "creative_dna", "asset_graph"],
        outputs=["shot_plan"],
        preconditions=["screenplay:committed", "asset_graph:committed"],
        validators=["test_shot_plan"],
        cost_class="cheap_text", idempotent=False,
        execute_fn=run_plan_storyboard))

    registry.register(SkillSpec(
        name="repair_shot_plan",
        description="Locally patch failing shots (not full replan)",
        inputs=["shot_plan"], outputs=["shot_plan"],
        preconditions=["shot_plan:exists"],
        validators=["test_shot_plan"],
        cost_class="cheap_text", idempotent=False,
        execute_fn=run_repair_shot_plan))

    registry.register(SkillSpec(
        name="inspect_shot_plan",
        description="Read-only diagnosis of current Shot Plan",
        inputs=["shot_plan"], outputs=[],
        preconditions=["shot_plan:exists"],
        validators=[], cost_class="cheap_text", idempotent=True,
        execute_fn=run_inspect_shot_plan))

    registry.register(SkillSpec(
        name="generate_storyboard_frame",
        description="Generate start/end frame pair for ONE shot "
                    "(requires all referenced assets COMMITTED)",
        inputs=["shot_plan"], outputs=["storyboard_frames"],
        preconditions=["shot_plan:committed"],
        validators=["test_storyboard_frame_pair"],
        cost_class="expensive_gpu", idempotent=False,
        execute_fn=run_generate_frames))

    return registry
