"""Canonical Entity Registry（V3 P2）：source-local entity → canonical id 归一。

罗小黑两部电影各自标注出独立的 entity_id（film1/entity_005 与 film2/entity_017
都可能是小黑）——只看源内 ID 会误判"换人了"。注册表把别名与源内 ID 双路映射
到 char:xxx，跨片同人保持同一身份键，Entity Continuity Contract 才能跨电影
锁主角。缺注册表时退回源内 entity_id/名字键（同片内约束仍成立，不崩）。
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from src.config import repo_root


def _norm(name: str) -> str:
    """别名归一：NFKC（全半角）、去标点空白、小写。"""
    out = []
    for char in unicodedata.normalize("NFKC", str(name or "")):
        if char.isalnum():
            out.append(char.lower())
    return "".join(out)


def load_entity_registry(cfg=None) -> dict:
    """加载注册表：cfg.library.entities.registry_path 或仓库
    config/entity_registry.json；缺文件/解析失败返回空表（显式降级）。"""
    path = None
    if cfg is not None:
        entities_cfg = (cfg.library.get("entities") or {}) if hasattr(cfg, "library") else {}
        path = entities_cfg.get("registry_path")
    if not path:
        path = repo_root() / "config" / "entity_registry.json"
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def build_alias_maps(registry: dict) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """→ (norm 别名 → canonical, (源名, entity_id) → canonical)。"""
    aliases: dict[str, str] = {}
    source_map: dict[tuple[str, str], str] = {}
    for canon, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases") or []:
            aliases[_norm(alias)] = canon
        for reference in entry.get("source_entities") or []:
            source, _, entity_id = str(reference).partition("/")
            source_map[(source.split("__")[0], entity_id)] = canon
    return aliases, source_map


def canonical_entities(row: dict, registry: dict) -> set[str]:
    """行实体 → canonical id 集合（别名 + 源内 ID 双路）。"""
    aliases, source_map = build_alias_maps(registry)
    found = set()
    source = str(row.get("video_stem") or row.get("source") or "").split("__")[0]
    for entity_id in row.get("entity_ids") or []:
        canon = source_map.get((source, str(entity_id)))
        if canon:
            found.add(canon)
    for name in row.get("entity_names") or []:
        canon = aliases.get(_norm(name))
        if canon:
            found.add(canon)
    return found


def row_identity_keys(row: dict, registry: dict) -> set[str]:
    """身份键（Entity Continuity Contract 的判定基础）：
    canonical > 源内 entity_id > 归一名字。未注册实体保留源内键——
    同片约束仍生效，跨片归一依赖注册表补条目。"""
    keys = canonical_entities(row, registry)
    source = str(row.get("video_stem") or row.get("source") or "").split("__")[0]
    for entity_id in row.get("entity_ids") or []:
        keys.add(f"id:{source}/{entity_id}")
    if not keys:
        for name in row.get("entity_names") or []:
            norm = _norm(name)
            if norm:
                keys.add(f"name:{norm}")
    return keys
