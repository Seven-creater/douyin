# -*- coding: utf-8 -*-
"""M2-A Character Asset Studio 的六个 Skills。

链路：generate_character_master → generate_character_multiview →
（局部 repair_character_view）→ upscale_character_view →
compose_identity_sheet → COMMIT asset:C0。
视图单图是真资产；identity_sheet 只是 overview/人审界面。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.asset_studio import prompts as studio_prompts
from src.agentic_video.asset_studio.image_io import (
    GEN_HEIGHT, GEN_WIDTH, VIEW_IDS, compose_identity_sheet, file_sha256,
    load_image)
from src.agentic_video.skills.registry import (
    SkillBlocked, SkillRegistry, SkillSpec)
from src.agentic_video.workspace import Workspace

SEED_BASE = 20260920


def _files_dir(workspace: Workspace) -> Path:
    d = workspace.root / "03_asset_studio" / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_version_file(directory: Path, base: str, ext: str = "png") -> Path:
    n = 1
    while (directory / f"{base}_v{n}.{ext}").is_file():
        n += 1
    return directory / f"{base}_v{n}.{ext}"


def _asset_entry(graph: dict[str, Any], asset_id: str) -> dict[str, Any]:
    for asset in graph.get("assets", []):
        if str(asset.get("asset_id")) == asset_id:
            return asset
    raise ValueError(f"{asset_id} not in asset_graph")


def _identity_description(asset: dict[str, Any]) -> str:
    parts: list[str] = []
    canonical = asset.get("canonical") or {}
    desc = canonical.get("description")
    if desc:
        parts.append(str(desc))
    immutable = asset.get("immutable") or {}
    for key in ("identity", "face", "body_build", "hair"):
        if immutable.get(key):
            parts.append(f"{key}: {immutable[key]}")
    # P1-6：asset_id 从 asset dict 取（原实现引用未定义变量会 NameError）
    return "; ".join(str(p) for p in parts if p) or \
        str(asset.get("asset_id") or "character")


def _repair_attempt_count(workspace: Workspace, target: str,
                          manifest_entry: dict[str, Any] | None = None,
                          ) -> int:
    """该 target 已有修复次数（含 master）。

    事实源优先：当前 manifest entry 的 repair_attempt（agent_loop 外
    直调 skill 时不写 trace，只看 trace 会数丢）；trace 作为增量。
    """
    count = 0
    if manifest_entry and isinstance(
            manifest_entry.get("repair_attempt"), int):
        count = max(count, manifest_entry["repair_attempt"])
    if workspace.trace_path.is_file():
        for line in workspace.trace_path.read_text(
                encoding="utf-8").strip().splitlines():
            try:
                entry = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            action = entry.get("action") or {}
            if action.get("skill") in ("repair_character_view",
                                       "repair_character_master") and \
                    str(action.get("target") or "") == target:
                count += 1
    return count


def _image_entry(rel_path: Path, workspace: Workspace, **extra: Any
                 ) -> dict[str, Any]:
    image = load_image(workspace.root / rel_path)
    return {"file": str(rel_path).replace("\\", "/"),
            "sha": file_sha256(workspace.root / rel_path),
            "width": image.width, "height": image.height, **extra}


def build_m2a_registry(*, t2i, multiview, upscale,
                       asset_id: str = "C0") -> SkillRegistry:
    """构建 M2-A 六个 skills。t2i/multiview/upscale 为后端实例。"""
    registry = SkillRegistry()
    master_art = f"asset:{asset_id}_master"
    views_art = f"asset:{asset_id}_views"
    candidate_art = f"asset:{asset_id}_candidate"
    approval_art = f"human_approval:{asset_id}"
    final_art = f"asset:{asset_id}"

    def run_generate_master(workspace: Workspace, **kw) -> dict:
        graph = workspace.read_artifact("asset_graph")
        if not graph:
            raise ValueError("asset_graph not in workspace")
        asset = _asset_entry(graph, asset_id)
        identity = _identity_description(asset)
        spec = {k: asset[k] for k in ("pose_profile", "anatomy_constraints",
                                      "mobility_aids",
                                      "identity_accessories") if k in asset}
        out = _next_version_file(_files_dir(workspace),
                                 f"{asset_id}_master_front")
        attempt = _repair_attempt_count(
            workspace, "front",
            ((workspace.read_artifact(master_art) or {}
              ).get("master") or {})) + 1
        seed = (kw.get("seed") or SEED_BASE) + (attempt - 1) * 1000
        t2i.generate(
            studio_prompts.build_master_prompt(identity, spec or None),
            out, width=GEN_WIDTH, height=GEN_HEIGHT, seed=seed)
        manifest = {
            "asset_id": asset_id, "asset_type": asset.get("type"),
            "identity_description": identity,
            "master": _image_entry(out.relative_to(workspace.root),
                                   workspace, view="front",
                                   generator="qwen-image", seed=seed,
                                   attempt=attempt)}
        version = workspace.write_draft(master_art, manifest)
        return {"artifact": master_art, "version": version,
                "action": "master_generated"}

    def run_generate_multiview(workspace: Workspace, **kw) -> dict:
        master = workspace.read_artifact(master_art)
        if not master:
            raise ValueError(f"{master_art} not in workspace")
        identity = master.get("identity_description") or asset_id
        master_rel = master["master"]["file"]
        todo = [v for v in ("profile", "back", "face")]
        results = multiview.generate_views(
            workspace.root / master_rel, todo, _files_dir(workspace),
            identity, seed_base=SEED_BASE + 100,
            name_prefix=f"{asset_id}_")
        views = {"front": dict(master["master"], source="master")}
        for view, answer in results.items():
            rel = Path(answer["path"]).relative_to(workspace.root)
            views[view] = _image_entry(rel, workspace,
                                       source="edit", seed=answer["seed"])
        manifest = {"asset_id": asset_id,
                    "master_artifact": master_art,
                    "identity_description": identity,
                    "views": views}
        version = workspace.write_draft(views_art, manifest)
        return {"artifact": views_art, "version": version,
                "action": "views_generated"}

    def run_repair_view(workspace: Workspace, **kw) -> dict:
        current = workspace.read_artifact(views_art)
        if not current:
            raise ValueError(f"{views_art} not in workspace")
        target = str(kw.get("target") or "")
        if target not in VIEW_IDS:
            target = _first_failing_view(kw.get("test_failures") or [])
        if target not in VIEW_IDS:
            raise ValueError(
                f"repair_character_view needs a view target "
                f"(front/profile/back/face), got {target!r}")
        if target == "front":
            # P1-3：front = canonical master 本体，不允许局部修——
            # master 失效必须走 repair_character_master（下游全 stale）
            raise SkillBlocked(
                "master_invalid",
                "front view IS the canonical master — regenerate via "
                "repair_character_master; derived views will stale")
        failures = kw.get("test_failures") or []
        attempt = _repair_attempt_count(
            workspace, target, current["views"].get(target)) + 1
        # P1-2：seed 随 attempt 递进（同 seed 重抽=浪费 GPU），
        # 失败原因注入修复 prompt（诊断式修复）
        seed = (SEED_BASE + 500 + VIEW_IDS.index(target)
                + attempt * 1000)
        views = json.loads(json.dumps(current["views"]))  # deep copy
        master = workspace.read_artifact(master_art)
        master_rel = master["master"]["file"]
        out = _next_version_file(_files_dir(workspace),
                                 f"{asset_id}_{target}")
        answer = multiview.edit_single(
            workspace.root / master_rel, target, out,
            current["identity_description"], seed=seed,
            failures=failures, attempt=attempt)
        rel = Path(answer["path"]).relative_to(workspace.root)
        old_sha = views[target].get("sha")
        views[target] = _image_entry(rel, workspace, source="edit",
                                     seed=answer["seed"],
                                     repair_attempt=attempt,
                                     repair_reason=[
                                         str(f.get("detail") or f.get("check"))
                                         for f in failures][:3],
                                     parent_sha=old_sha)
        manifest = {**current, "views": views,
                    "repaired_view": target}
        version = workspace.write_draft(views_art, manifest)
        return {"artifact": views_art, "version": version,
                "action": "view_repaired", "target": target}

    def run_repair_master(workspace: Workspace, **kw) -> dict:
        """P1-3：master 失效的正规修复路径（重生成 master，下游全 stale）。"""
        graph = workspace.read_artifact("asset_graph")
        if not graph:
            raise ValueError("asset_graph not in workspace")
        asset = _asset_entry(graph, asset_id)
        identity = _identity_description(asset)
        failures = kw.get("test_failures") or []
        attempt = _repair_attempt_count(
            workspace, "front",
            ((workspace.read_artifact(master_art) or {}
              ).get("master") or {})) + 1
        seed = SEED_BASE + attempt * 1000
        prompt = studio_prompts.build_master_prompt(identity)
        if failures:
            notes = "\n".join(
                f"- ({f.get('check')}) {f.get('detail')}" for f in failures)
            prompt = (f"Repair regeneration (attempt {attempt}). "
                      f"Previous validation failures:\n{notes}\n{prompt}")
        out = _next_version_file(_files_dir(workspace),
                                 f"{asset_id}_master_front")
        t2i.generate(prompt, out, width=GEN_WIDTH, height=GEN_HEIGHT,
                     seed=seed)
        manifest = {
            "asset_id": asset_id, "asset_type": asset.get("type"),
            "identity_description": identity,
            "master": _image_entry(out.relative_to(workspace.root),
                                   workspace, view="front",
                                   generator="qwen-image", seed=seed,
                                   repair_attempt=attempt)}
        version = workspace.write_draft(master_art, manifest)
        return {"artifact": master_art, "version": version,
                "action": "master_repaired", "target": "front"}

    def run_upscale(workspace: Workspace, **kw) -> dict:
        current = workspace.read_artifact(views_art)
        if not current:
            raise ValueError(f"{views_art} not in workspace")
        views = json.loads(json.dumps(current["views"]))
        for view, entry in views.items():
            src = workspace.root / entry["file"]
            out = _next_version_file(_files_dir(workspace),
                                     f"{asset_id}_{view}_4k")
            upscale.upscale(src, out)
            rel = out.relative_to(workspace.root)
            entry["file4k"] = str(rel).replace("\\", "/")
            entry["sha4k"] = file_sha256(out)
            # P2-1 命名诚实：尺寸 4K ≠ 细节 4K
            entry["upscale_method"] = "lanczos"
            entry["detail_enhanced"] = False
        manifest = {**current, "views": views, "upscaled": True,
                    "upscale_method": "lanczos", "detail_enhanced": False}
        version = workspace.write_draft(views_art, manifest)
        return {"artifact": views_art, "version": version,
                "action": "upscaled_4k_derived"}

    def run_compose_sheet(workspace: Workspace, **kw) -> dict:
        """P0-5：机器链终点 = candidate（L1+L2 验收后 COMMIT 到此为止）。

        最终 asset:C0 COMMIT 必须等 human_approval（approve skill）。
        """
        current = workspace.read_artifact(views_art)
        if not current:
            raise ValueError(f"{views_art} not in workspace")
        views = current["views"]
        sheet = _next_version_file(_files_dir(workspace),
                                   f"{asset_id}_identity_sheet")
        compose_identity_sheet(
            {v: workspace.root / views[v]["file4k"] for v in VIEW_IDS},
            sheet)
        rel = sheet.relative_to(workspace.root)
        manifest = {
            "asset_id": asset_id, "asset_type": "character",
            "master_artifact": master_art, "views_artifact": views_art,
            "identity_description": current["identity_description"],
            "views_4k": {v: {"file": views[v]["file4k"],
                             "sha": views[v]["sha4k"]} for v in VIEW_IDS},
            "sheet": {"file": str(rel).replace("\\", "/"),
                      "sha": file_sha256(sheet)},
            "human_review": "pending"}
        version = workspace.write_draft(candidate_art, manifest)
        return {"artifact": candidate_art, "version": version,
                "action": "sheet_composed"}

    def run_approve(workspace: Workspace, **kw) -> dict:
        """P0-5：人审批准（确定性 SHA 绑定，不调模型）。

        要求 human_approval:{id} 已 COMMIT 且 decision=approve，且其
        绑定的 candidate_sha / view shas 与当前 candidate 完全一致。
        """
        candidate = workspace.read_artifact(candidate_art)
        approval = workspace.read_artifact(approval_art)
        if not candidate or not approval:
            raise SkillBlocked(
                "approval_missing",
                f"need committed {candidate_art} and {approval_art}")
        if str(approval.get("decision") or "") != "approve":
            raise SkillBlocked(
                "approval_rejected",
                f"decision={approval.get('decision')!r}")
        if str(approval.get("candidate_sha") or "") != \
                str(workspace.get_sha(candidate_art) or ""):
            raise SkillBlocked(
                "approval_stale",
                "approval is bound to a different candidate sha — "
                "re-review required")
        for view, sha in (approval.get("views") or {}).items():
            bound = str(((candidate.get("views_4k") or {})
                         .get(view) or {}).get("sha") or "")
            if str(sha) != bound:
                raise SkillBlocked(
                    "approval_stale",
                    f"view {view} sha mismatch: approval={sha} "
                    f"candidate={bound}")
        final_manifest = {**candidate,
                          "approval": {k: approval[k] for k in
                                       ("decision", "reviewer", "note")
                                       if k in approval},
                          "human_review": "approved"}
        version = workspace.write_draft(final_art, final_manifest)
        return {"artifact": final_art, "version": version,
                "action": "human_approved"}

    def run_inspect(workspace: Workspace, **kw) -> dict:
        return {"artifact": final_art, "action": "inspected",
                "master": workspace.read_artifact(master_art),
                "views": workspace.read_artifact(views_art),
                "final": workspace.read_artifact(final_art)}

    registry.register(SkillSpec(
        name="generate_character_master",
        description="Generate Hero Master (front full body) via T2I",
        inputs=["asset_graph"], outputs=[master_art],
        preconditions=["asset_graph:committed"],
        validators=["test_character_master"],
        cost_class="expensive_gpu", idempotent=False,
        execute_fn=run_generate_master))

    registry.register(SkillSpec(
        name="generate_character_multiview",
        description="Generate profile/back/face views from master",
        inputs=[master_art], outputs=[views_art],
        preconditions=[f"{master_art}:committed"],
        validators=["test_character_multiview"],
        cost_class="expensive_gpu", idempotent=False,
        execute_fn=run_generate_multiview))

    registry.register(SkillSpec(
        name="repair_character_view",
        description="Regenerate ONLY the failing view (diagnostic local "
                    "repair; front is forbidden — use "
                    "repair_character_master)",
        inputs=[views_art], outputs=[views_art],
        preconditions=[f"{views_art}:exists"],
        validators=["test_character_multiview"],
        cost_class="expensive_gpu", idempotent=False,
        execute_fn=run_repair_view))

    registry.register(SkillSpec(
        name="repair_character_master",
        description="Regenerate canonical master (front) — derived views "
                    "go STALE via dependency DAG",
        inputs=["asset_graph"], outputs=[master_art],
        preconditions=["asset_graph:committed"],
        validators=["test_character_master"],
        cost_class="expensive_gpu", idempotent=False,
        execute_fn=run_repair_master))

    registry.register(SkillSpec(
        name="upscale_character_view",
        description="Derive 4K-size views (deterministic Lanczos; not "
                    "detail-enhanced)",
        inputs=[views_art], outputs=[views_art],
        preconditions=[f"{views_art}:committed"],
        validators=["test_views_4k"],
        cost_class="cheap_cpu", idempotent=False,
        execute_fn=run_upscale))

    registry.register(SkillSpec(
        name="compose_identity_sheet",
        description="Compose 2x2 sheet → machine-validated CANDIDATE "
                    "(final COMMIT requires human approval)",
        inputs=[views_art], outputs=[candidate_art],
        preconditions=[f"{views_art}:committed"],
        validators=["test_character_asset"],
        cost_class="cheap_cpu", idempotent=True,
        execute_fn=run_compose_sheet))

    registry.register(SkillSpec(
        name="approve_character_asset",
        description="Human-approval SHA binding → final asset COMMIT "
                    "(deterministic, no model calls)",
        inputs=[candidate_art], outputs=[final_art],
        preconditions=[f"{candidate_art}:committed",
                       f"{approval_art}:committed"],
        validators=["test_asset_approved"],
        cost_class="cheap_text", idempotent=False,
        execute_fn=run_approve))

    registry.register(SkillSpec(
        name="inspect_character_asset",
        description="Read-only diagnosis of current asset state",
        inputs=[master_art], outputs=[],
        preconditions=[f"{master_art}:exists"],
        validators=[], cost_class="cheap_text", idempotent=True,
        execute_fn=run_inspect))

    return registry


def _first_failing_view(failures: list[dict[str, Any]]) -> str:
    for failure in failures:
        view = str(failure.get("view") or "")
        if view in VIEW_IDS:
            return view
    return ""
