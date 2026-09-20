# -*- coding: utf-8 -*-
"""M2-A Character 资产验收（Actor ≠ Test）。

三层：L1 程序化（尺寸/比例/可读/视图齐/背景干净）→
L2 Omni 视觉（跨视图同一人/视角正确/无多余人物）→ L3 人工（sheet）。
L1 失败直接 FAIL——程序能查的不浪费 Omni（expensive 资产省着验）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.workspace import Workspace

VALIDATE_MASTER_PROMPT = """你是角色资产验收员（不是生成者）。输入一张
Hero Master 全身照。只根据图像本身独立检查，只输出一个 JSON：
{"passed": true, "failures": [{"check": "...", "detail": "..."}]}
检查项：
1. single_person：画面中只有一个人物，无额外人/镜像
2. full_body：头到脚全身可见（含鞋）
3. neutral_bg：中性灰无缝棚拍背景，无场景元素
4. no_text：无文字/水印/logo
5. neutral_look：中性表情闭嘴、自然站姿、中性深灰基础服装
你没有看过任何生成过程——只根据当前图像判断。图像："""

VALIDATE_VIEWS_PROMPT = """你是角色一致性验收员。输入同一角色的四张
视图照片，顺序：1=front 正面全身、2=profile 90°侧面全身、3=back 背面
全身、4=face 面部特写。只输出一个 JSON：
{"passed": true,
 "views": [{"view": "front", "same_person": true, "view_correct": true,
             "issues": []},
            {"view": "profile", "same_person": true,
             "view_correct": true, "issues": ["..."]}],
 "failures": [{"view": "...", "check": "...", "detail": "..."}]}
检查项（以第 1 张 front 为身份锚）：
1. same_person：每张是否与 front 是同一人（年龄/脸型/发型/体型一致）
2. view_correct：视角是否正确（profile 必须严格 90° 侧面、back 必须
   正背面、face 必须是同一个人脸的特写）
3. extra_person：每张是否只有一个人物
4. wardrobe：四张服装一致（深灰基础款）
你没有看过任何生成过程——只根据当前四张图像判断。图像："""

VALIDATE_ASSET_PROMPT = """你是最终资产验收员。输入同一角色提交版
四视图（顺序 front/profile/back/face）。这是最终 COMMIT 前的独立
复核，只输出一个 JSON：
{"passed": true, "failures": [{"view": "...", "check": "...",
  "detail": "..."}]}
复核项：
1. identity：四张是否同一人（年龄/脸型/发型/身材比例零漂移）
2. expression：中性表情、闭嘴
3. lighting：棚拍均匀光、中性灰背景
任何一张不符合 → passed=false 且 failures 标明 view。
你没有看过任何生成过程。图像："""


def run_m2_test(validator_name: str, workspace: Workspace,
                runner) -> dict[str, Any]:
    """M2 validator 分发。"""
    if validator_name == "test_character_master":
        return _test_character_master(workspace, runner)
    if validator_name == "test_character_multiview":
        return _test_character_multiview(workspace, runner)
    if validator_name == "test_views_4k":
        return _test_views_4k(workspace)
    if validator_name == "test_character_asset":
        return _test_character_asset(workspace, runner)
    raise KeyError(f"unknown m2 validator: {validator_name}")


# ---- L1 工具 ----

def _resolve(workspace: Workspace, rel: str) -> Path:
    return workspace.root / rel


def _l1_image(rel: str, workspace: Workspace,
              min_long_side: int = 1024) -> list[dict[str, str]]:
    """单图 L1：存在/可读/9:16/长边下限/背景干净。"""
    from src.agentic_video.asset_studio.image_io import (
        check_aspect_9_16, check_background_clean, load_image)
    path = _resolve(workspace, rel)
    failures: list[dict[str, str]] = []
    if not path.is_file():
        return [{"view": _view_of(rel), "check": "exists",
                 "detail": f"missing file: {rel}"}]
    try:
        image = load_image(path)
    except Exception as exc:  # noqa: BLE001
        return [{"view": _view_of(rel), "check": "readable",
                 "detail": f"{type(exc).__name__}: {exc}"}]
    if not check_aspect_9_16(image):
        failures.append({"view": _view_of(rel), "check": "aspect",
                         "detail": f"{image.width}x{image.height} not 9:16"})
    if max(image.size) < min_long_side:
        failures.append({"view": _view_of(rel), "check": "resolution",
                         "detail": f"long side {max(image.size)} "
                                   f"< {min_long_side}"})
    if not check_background_clean(image):
        failures.append({"view": _view_of(rel), "check": "background",
                         "detail": "border strips not neutral/clean"})
    return failures


def _view_of(rel: str) -> str:
    name = Path(rel).stem
    for view in ("front", "profile", "back", "face"):
        if name.startswith(view):
            return view
    return name


def _parse_omni_json(answer) -> dict[str, Any]:
    from src.agentic_video.reference_program_v9 import _parse_one_object
    text = str(getattr(answer, "text", "") or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return _parse_one_object(text, stage="m2_validator")


# ---- Validators ----

def _test_character_master(workspace: Workspace, runner) -> dict[str, Any]:
    manifest = workspace.read_artifact("asset:C0_master")
    if not manifest:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "asset:C0_master is empty"}],
                "validators_run": ["test_character_master"]}
    rel = str((manifest.get("master") or {}).get("file") or "")
    if not rel:
        return {"passed": False, "failures": [{"check": "manifest",
                "detail": "master.file missing"}],
                "validators_run": ["test_character_master"]}
    failures = _l1_image(rel, workspace)
    if failures:  # L1 挂不烧 Omni
        return {"passed": False, "failures": failures,
                "validators_run": ["test_character_master"]}
    answer = runner.inspect_media(
        image_paths=[_resolve(workspace, rel)], prompt=VALIDATE_MASTER_PROMPT)
    try:
        value = _parse_omni_json(answer)
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "failures": [{"check": "parse",
                "detail": str(exc)[:120]}],
                "validators_run": ["test_character_master"]}
    omni_failures = value.get("failures") or []
    return {"passed": bool(value.get("passed")) and not omni_failures,
            "failures": omni_failures,
            "validators_run": ["test_character_master"]}


def _test_character_multiview(workspace: Workspace, runner) -> dict[str, Any]:
    from src.agentic_video.asset_studio.image_io import VIEW_IDS
    manifest = workspace.read_artifact("asset:C0_views")
    if not manifest:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "asset:C0_views is empty"}],
                "validators_run": ["test_character_multiview"]}
    views = manifest.get("views") or {}
    failures: list[dict[str, str]] = []
    paths: dict[str, str] = {}
    for view in VIEW_IDS:
        rel = str((views.get(view) or {}).get("file") or "")
        if not rel:
            failures.append({"view": view, "check": "views",
                             "detail": f"{view} missing in manifest"})
            continue
        paths[view] = rel
        failures.extend(_l1_image(rel, workspace))
    if failures:
        return {"passed": False, "failures": failures,
                "validators_run": ["test_character_multiview"]}
    answer = runner.inspect_media(
        image_paths=[_resolve(workspace, paths[v]) for v in VIEW_IDS],
        prompt=VALIDATE_VIEWS_PROMPT)
    try:
        value = _parse_omni_json(answer)
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "failures": [{"check": "parse",
                "detail": str(exc)[:120]}],
                "validators_run": ["test_character_multiview"]}
    omni_failures = value.get("failures") or []
    # view 级 verdict 兜底映射（Omni 有时只填 views 不填 failures）
    for verdict in value.get("views") or []:
        view = str(verdict.get("view") or "")
        issues = verdict.get("issues") or []
        if verdict.get("same_person") is False:
            omni_failures.append({
                "view": view, "check": "identity",
                "detail": "not the same person as front"})
        if verdict.get("view_correct") is False:
            omni_failures.append({
                "view": view, "check": "view_angle",
                "detail": f"viewpoint incorrect: {issues}"})
    return {"passed": bool(value.get("passed")) and not omni_failures,
            "failures": omni_failures,
            "validators_run": ["test_character_multiview"]}


def _test_views_4k(workspace: Workspace) -> dict[str, Any]:
    """纯确定性：全部视图长边 ≥ 3840 且文件 sha 与 manifest 一致。"""
    from src.agentic_video.asset_studio.image_io import (
        TARGET_4K_LONG_SIDE, VIEW_IDS, file_sha256, load_image)
    manifest = workspace.read_artifact("asset:C0_views")
    if not manifest:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "asset:C0_views is empty"}],
                "validators_run": ["test_views_4k"]}
    failures: list[dict[str, str]] = []
    views = manifest.get("views") or {}
    for view in VIEW_IDS:
        entry = views.get(view) or {}
        rel = str(entry.get("file4k") or "")
        if not rel:
            failures.append({"view": view, "check": "file4k",
                             "detail": "4k file missing"})
            continue
        path = _resolve(workspace, rel)
        if not path.is_file():
            failures.append({"view": view, "check": "file4k",
                             "detail": f"missing: {rel}"})
            continue
        if max(load_image(path).size) < TARGET_4K_LONG_SIDE:
            failures.append({"view": view, "check": "resolution",
                             "detail": "long side < 3840"})
        recorded = str(entry.get("sha4k") or "")
        if recorded and recorded != file_sha256(path):
            failures.append({"view": view, "check": "sha",
                             "detail": "4k file sha mismatch"})
    return {"passed": not failures, "failures": failures,
            "validators_run": ["test_views_4k"]}


def _test_character_asset(workspace: Workspace, runner) -> dict[str, Any]:
    """最终验收：L1（sheet 存在/视图 4k/sha 链）→ L2 Omni 终审。"""
    from src.agentic_video.asset_studio.image_io import (
        TARGET_4K_LONG_SIDE, VIEW_IDS, file_sha256, load_image)
    manifest = workspace.read_artifact("asset:C0")
    if not manifest:
        return {"passed": False, "failures": [{"check": "exists",
                "detail": "asset:C0 is empty"}],
                "validators_run": ["test_character_asset"]}
    failures: list[dict[str, str]] = []
    sheet_rel = str((manifest.get("sheet") or {}).get("file") or "")
    sheet_path = _resolve(workspace, sheet_rel)
    if not sheet_rel or not sheet_path.is_file():
        failures.append({"check": "sheet", "detail": "identity sheet missing"})
    views_4k = manifest.get("views_4k") or {}
    for view in VIEW_IDS:
        entry = views_4k.get(view) or {}
        rel = str(entry.get("file") or "")
        path = _resolve(workspace, rel)
        if not rel or not path.is_file():
            failures.append({"view": view, "check": "views_4k",
                             "detail": f"missing: {rel}"})
            continue
        if max(load_image(path).size) < TARGET_4K_LONG_SIDE:
            failures.append({"view": view, "check": "resolution",
                             "detail": "long side < 3840"})
        recorded = str(entry.get("sha") or "")
        if recorded and recorded != file_sha256(path):
            failures.append({"view": view, "check": "sha",
                             "detail": "sha mismatch"})
    if failures:
        return {"passed": False, "failures": failures,
                "validators_run": ["test_character_asset"]}
    answer = runner.inspect_media(
        image_paths=[_resolve(workspace, views_4k[v]["file"])
                     for v in VIEW_IDS if v in views_4k],
        prompt=VALIDATE_ASSET_PROMPT)
    try:
        value = _parse_omni_json(answer)
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "failures": [{"check": "parse",
                "detail": str(exc)[:120]}],
                "validators_run": ["test_character_asset"]}
    omni_failures = value.get("failures") or []
    return {"passed": bool(value.get("passed")) and not omni_failures,
            "failures": omni_failures,
            "validators_run": ["test_character_asset"]}
