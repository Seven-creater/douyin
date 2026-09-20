# -*- coding: utf-8 -*-
"""Asset Graph 共享 Schema Contract（Skill 与 Validator 同源）。

p0626 修正：extract_assets 和 test_asset_graph 不再各自在 prompt 里
描述格式——双方共用此模块的 schema 定义 + deterministic 校验。
"""
from __future__ import annotations

from typing import Any

ASSET_GRAPH_SCHEMA_VERSION = "asset_graph_v1"

ASSET_TYPES = ("character", "location", "prop", "wardrobe", "motion",
               "style", "audio", "graphics", "creature")
TIERS = ("A", "B", "C")


def validate_asset_graph(graph: dict[str, Any]) -> list[dict[str, str]]:
    """Deterministic schema 校验（唯一权威——无 Omni 语义二次判定）。

    返回 failures 列表（空=过）。检查：必填字段/asset_id 唯一/tier 合法/
    character immutable 必含 identity/face/body_build/tier_A 覆盖检查。
    Run A-v2 教训：确定性 schema 通过后不得再让 Omni 重新猜"immutable
    应含什么"——两个 verifier 会互相矛盾导致死循环。
    """
    failures: list[dict[str, str]] = []

    if not isinstance(graph, dict):
        return [{"asset_id": "_root", "check": "schema",
                 "detail": "not a JSON object"}]

    if graph.get("schema_version") != ASSET_GRAPH_SCHEMA_VERSION:
        failures.append({"asset_id": "_root", "check": "schema_version",
                         "detail": f"expected {ASSET_GRAPH_SCHEMA_VERSION}, "
                                   f"got {graph.get('schema_version')}"})

    assets = graph.get("assets")
    if not isinstance(assets, list) or not assets:
        failures.append({"asset_id": "_root", "check": "assets",
                         "detail": "assets list missing or empty"})
        return failures

    seen_ids: set[str] = set()
    has_tier_a = False
    for asset in assets:
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id:
            failures.append({"asset_id": "?", "check": "asset_id",
                             "detail": "missing"})
            continue
        if asset_id in seen_ids:
            failures.append({"asset_id": asset_id, "check": "asset_id",
                             "detail": "duplicate"})
        seen_ids.add(asset_id)

        if asset.get("type") not in ASSET_TYPES:
            failures.append({"asset_id": asset_id, "check": "type",
                             "detail": f"invalid: {asset.get('type')}"})

        if asset.get("tier") not in TIERS:
            failures.append({"asset_id": asset_id, "check": "tier",
                             "detail": f"invalid: {asset.get('tier')}"})

        if asset.get("tier") == "A":
            has_tier_a = True

        if asset.get("type") == "character":
            immutable = asset.get("immutable")
            if not isinstance(immutable, dict) or not immutable:
                failures.append({
                    "asset_id": asset_id, "check": "immutable",
                    "detail": "character must have immutable dict "
                              "with identity fields (face/body_build)"})
            elif not any(key in immutable for key in
                         ("identity", "face", "body_build")):
                failures.append({
                    "asset_id": asset_id, "check": "immutable",
                    "detail": "character immutable must include at least "
                              "one of identity/face/body_build"})

        status = str(asset.get("status") or "active")
        if status not in ("active", "deferred"):
            failures.append({"asset_id": asset_id, "check": "status",
                             "detail": f"invalid: {status}"})

    if not has_tier_a:
        failures.append({"asset_id": "_root", "check": "tier_coverage",
                         "detail": "at least one Tier A asset required"})

    return failures


# Skill 和 Validator 共用的 prompt 片段（从 schema 生成，不手写两份）
ASSET_GRAPH_FORMAT_SPEC = """{
  "schema_version": "asset_graph_v1",
  "assets": [
    {"asset_id": "C0", "type": "character", "tier": "A",
     "immutable": {"identity": "...", "face": "...",
                    "body_build": "...", "hair": "..."},
     "mutable": {"wardrobe": true, "expression": true},
     "usage": ["B1", "B2"],
     "status": "active"},
    {"asset_id": "L01", "type": "location", "tier": "A", ...},
    {"asset_id": "A01", "type": "motion", "tier": "A",
     "status": "deferred"}
  ]
}
必填：schema_version="asset_graph_v1"；每个 asset 有 asset_id/type/tier；
character 必须有 immutable dict（至少含 identity/face/body_build 之一）；
status ∈ active|deferred。"""
