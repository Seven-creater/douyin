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
from src.agentic_video.skills.registry import SkillRegistry, SkillSpec
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
    return "; ".join(str(p) for p in parts if p) or str(asset_id)


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
    final_art = f"asset:{asset_id}"

    def run_generate_master(workspace: Workspace, **kw) -> dict:
        graph = workspace.read_artifact("asset_graph")
        if not graph:
            raise ValueError("asset_graph not in workspace")
        asset = _asset_entry(graph, asset_id)
        identity = _identity_description(asset)
        out = _next_version_file(_files_dir(workspace),
                                 f"{asset_id}_master_front")
        seed = kw.get("seed") or SEED_BASE
        t2i.generate(studio_prompts.build_master_prompt(identity), out,
                     width=GEN_WIDTH, height=GEN_HEIGHT, seed=seed)
        manifest = {
            "asset_id": asset_id, "asset_type": asset.get("type"),
            "identity_description": identity,
            "master": _image_entry(out.relative_to(workspace.root),
                                   workspace, view="front",
                                   generator="qwen-image", seed=seed)}
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
        # 局部修复：只重生成 target，其余视图文件与 sha 原样保留
        views = json.loads(json.dumps(current["views"]))  # deep copy
        if target == "front":
            # front 挂 → 以 face（身份锚）为参考重生成正面
            ref_rel = views["face"]["file"]
            out = _next_version_file(_files_dir(workspace),
                                     f"{asset_id}_front")
            answer = multiview.edit_single(
                workspace.root / ref_rel, "front", out,
                current["identity_description"],
                seed=SEED_BASE + 500)
        else:
            master = workspace.read_artifact(master_art)
            master_rel = master["master"]["file"]
            out = _next_version_file(_files_dir(workspace),
                                     f"{asset_id}_{target}")
            answer = multiview.edit_single(
                workspace.root / master_rel, target, out,
                current["identity_description"],
                seed=SEED_BASE + 500 + VIEW_IDS.index(target))
        rel = Path(answer["path"]).relative_to(workspace.root)
        views[target] = _image_entry(rel, workspace, source="edit",
                                     seed=answer["seed"])
        manifest = {**current, "views": views,
                    "repaired_view": target}
        version = workspace.write_draft(views_art, manifest)
        return {"artifact": views_art, "version": version,
                "action": "view_repaired", "target": target}

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
        manifest = {**current, "views": views, "upscaled": True}
        version = workspace.write_draft(views_art, manifest)
        return {"artifact": views_art, "version": version,
                "action": "upscaled_4k"}

    def run_compose_sheet(workspace: Workspace, **kw) -> dict:
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
                      "sha": file_sha256(sheet)}}
        version = workspace.write_draft(final_art, manifest)
        return {"artifact": final_art, "version": version,
                "action": "sheet_composed"}

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
        description="Regenerate ONLY the failing view (local repair)",
        inputs=[views_art], outputs=[views_art],
        preconditions=[f"{views_art}:exists"],
        validators=["test_character_multiview"],
        cost_class="expensive_gpu", idempotent=False,
        execute_fn=run_repair_view))

    registry.register(SkillSpec(
        name="upscale_character_view",
        description="Upscale all views to 4K (deterministic Lanczos)",
        inputs=[views_art], outputs=[views_art],
        preconditions=[f"{views_art}:committed"],
        validators=["test_views_4k"],
        cost_class="cheap_cpu", idempotent=False,
        execute_fn=run_upscale))

    registry.register(SkillSpec(
        name="compose_identity_sheet",
        description="Deterministic 2x2 identity sheet compose",
        inputs=[views_art], outputs=[final_art],
        preconditions=[f"{views_art}:committed"],
        validators=["test_character_asset"],
        cost_class="cheap_cpu", idempotent=True,
        execute_fn=run_compose_sheet))

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
