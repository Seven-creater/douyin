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
    """加载注册表（P1.5 起三层合并）：

    1. 手写 config/entity_registry.json（或 cfg.library.entities.registry_path）；
    2. auto 回填 library_dir/entity_registry.auto.json（bootstrap --verify 从
       标注绑定生成）——同 canonical 只补 source_entities，手写别名优先；
    3. 缺文件/解析失败逐层显式降级为空表。"""
    path = None
    auto_path = None
    if cfg is not None:
        entities_cfg = (cfg.library.get("entities") or {}) if hasattr(cfg, "library") else {}
        path = entities_cfg.get("registry_path")
        if hasattr(cfg, "paths"):
            auto_path = cfg.paths.library_dir / "entity_registry.auto.json"
    if not path:
        path = repo_root() / "config" / "entity_registry.json"
    registry: dict = {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        registry = {}
    if auto_path is not None and auto_path.exists():
        try:
            auto = json.loads(auto_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            auto = {}
        for canonical, entry in (auto or {}).items():
            if not isinstance(entry, dict):
                continue
            if canonical not in registry:
                registry[canonical] = {"aliases": list(entry.get("aliases") or []),
                                        "source_entities": list(
                                            entry.get("source_entities") or [])}
                continue
            refs = set(registry[canonical].get("source_entities") or [])
            refs |= set(entry.get("source_entities") or [])
            registry[canonical]["source_entities"] = sorted(refs)
    return registry


def build_alias_maps(registry: dict) -> tuple[dict[str, str],
                                              dict[tuple[str, str], str],
                                              dict[tuple[str, str, str], str]]:
    """→ (norm 别名 → canonical, (源名, entity_id) → canonical,
    (源名, wN, entity_id) → canonical)。

    V4（外审三轮必改精神）：只有 `aliases`（身份名：专名/罗马音）进等价
    map。`appearance_aliases`（黑发持刀——只是长什么样）与 `role_labels`
    （执行人——组织头衔，film2 有多名执行人）完全移出——外观/头衔撞型的
    不同角色被合成同一 canonical 是 V3_C3 假绿的直接根因（外审反例：删
    "黑发持刀"别名后必选槽主角存在率 100%→0%）。

    source_entities 支持 `source/w{N}/{id}` 窗口作用域引用——库侧 entity_id
    是窗口级编号（各窗各自 e001 起），不带窗口的引用会把别的窗口同号 id
    也归到同一 canonical（V4 实锤：测试夹具里 w7/e001 撞上 w1/e001）。
    无窗口引用按旧语义（源内任意窗口）匹配，向后兼容。"""
    aliases: dict[str, str] = {}
    source_map: dict[tuple[str, str], str] = {}
    windowed_map: dict[tuple[str, str, str], str] = {}
    for canon, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases") or []:
            aliases[_norm(alias)] = canon
        for reference in entry.get("source_entities") or []:
            parts = [part for part in str(reference).split("/") if part]
            if not parts:
                continue
            source = parts[0].split("__")[0]
            if len(parts) == 3 and parts[1].startswith("w"):
                windowed_map[(source, parts[1], parts[2])] = canon
            else:
                source_map[(source, parts[-1])] = canon
    return aliases, source_map, windowed_map


def canonical_entities(row: dict, registry: dict) -> set[str]:
    """行实体 → canonical id 集合（身份别名 + 源内 ID 双路）。

    别名匹配含子串（alias≥3 字）——V4 起 alias 表只含身份名，「黑发持刀
    少年」这类外观串匹配不上任何键（外观/头衔在 build_alias_maps 就没进表）。"""
    aliases, source_map, windowed_map = build_alias_maps(registry)
    found = set()
    source = str(row.get("video_stem") or row.get("source") or "").split("__")[0]
    window = row.get("window_idx")
    window_key = f"w{int(window)}" if isinstance(window, int) else None
    for entity_id in row.get("entity_ids") or []:
        canon = ((windowed_map.get((source, window_key, str(entity_id)))
                  if window_key else None)
                 or source_map.get((source, str(entity_id))))
        if canon:
            found.add(canon)
    for name in row.get("entity_names") or []:
        norm = _norm(name)
        canon = aliases.get(norm)
        if canon:
            found.add(canon)
            continue
        for alias, candidate in aliases.items():
            if len(alias) >= 3 and (alias in norm or norm in alias):
                found.add(candidate)
                break
    return found


def row_identity_keys(row: dict, registry: dict) -> set[str]:
    """身份键（Entity Continuity Contract 的判定基础）：
    binding canonical > 别名 canonical > 窗口内 entity_id > 归一名字。

    窗口作用域（V3 晨跑实锤的假等价 bug）：库侧 entity_id 是**窗口级编号**
    ——每个 45s 窗各自从 e001 起，窗1 的 e001（黑发少年）≠ 窗7 的 e001
    （小女孩）。id 键必须带 window_idx，否则三槽三主角被算成同人，
    deterministic check 假绿（盲看抓到真相，det 没抓到）。

    P1.5 两层 Identity：标注行旁挂 bindings（local vis_ id → canonical，
    binding_status/confidence/evidence）——supported 绑定直接贡献 canonical
    键（跨窗同人成立的正路）；conflict 绑定不贡献（先验与视觉矛盾时以视觉为准）。"""
    keys: set[str] = set()
    for binding in row.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        if str(binding.get("binding_status") or "") in {"supported", "verified"}:
            canonical = str(binding.get("canonical_entity_id") or "")
            if canonical:
                keys.add(canonical)
    keys |= canonical_entities(row, registry)
    source = str(row.get("video_stem") or row.get("source") or "").split("__")[0]
    window = row.get("window_idx")
    scope = f"/w{int(window)}" if isinstance(window, int) else ""
    for entity_id in row.get("entity_ids") or []:
        keys.add(f"id:{source}{scope}/{entity_id}")
    if not keys:
        for name in row.get("entity_names") or []:
            norm = _norm(name)
            if norm:
                keys.add(f"name:{norm}")
    return keys


def shared_identity_keys(left: dict, right: dict, registry: dict) -> set[str]:
    """两行的共享身份键（V4 单一事实源：story_planner 五处连续性判定共用）。

    硬身份不变式（外审三轮）：只有 身份别名 exact/子串、manual
    source_entities、supported/verified bindings 参与判等——proposed/
    uncertain/conflict binding 与一切 canonical guess 不进硬约束。"""
    return row_identity_keys(left, registry) & row_identity_keys(right, registry)
