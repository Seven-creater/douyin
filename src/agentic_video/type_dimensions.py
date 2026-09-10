"""类型→关注维度路由（V1 计划 P1a，2026-09-10）。

content_type 只提供**默认**维度；本次任务实际维度 = 固定模板的具体素材需求 ∪
类型默认。目标不是"补全所有类型字段"，而是让固定模板需要的证据进入可检索、
可核验的字段（模板需要"被低估—行动—结果反驳"时，关键维度可能是主体身份/
被质疑的能力/行动过程/行动结果，地点年龄未必关键）。

库侧 facet 提取按全类型并集跑一遍（电影库要服务所有模板类型），且每条 facet
绑定到实体/实体对与时间区间——窗口级印象不得复制给整窗镜头。
"""
from __future__ import annotations

# 各内容类型的默认关注维度（P4 起由 config library.type_dimensions 可覆盖）
DIMENSION_SETS: dict[str, tuple[str, ...]] = {
    "real_story": ("location", "relationship", "effort_evidence"),
    "growth_story": ("age_appearance", "time_span", "setback"),
    "travel_story": ("location", "transition_sequence", "season"),
    "screen_story": ("dialogue_theme", "relationship"),
    "uncertain": ("location", "relationship"),
}

# 库侧 facet 提取实际下发给模型的维度（全类型并集的可达子集 + 交互边）。
# era/time_span 这类跨窗维度首版不提取（单窗看不出来）；emotion 已在叙事
# 标注里逐镜头产出，P1b 索引合并时补进可过滤字段，不重复提取。
LIBRARY_FACET_DIMENSIONS: tuple[str, ...] = (
    "age_appearance", "location", "era", "interaction")

INTERACTION_RELATIONS = ("保护", "对抗", "师徒", "亲情", "同伴", "救助", "陌生")


def resolve_dimensions(content_type: str,
                       template_needs: tuple[str, ...] | list[str] = ()) -> tuple[str, ...]:
    """默认维度 ∪ 模板需求，保持出现顺序（默认在前，模板补充在后）。"""
    merged = list(DIMENSION_SETS.get(content_type, DIMENSION_SETS["uncertain"]))
    for dimension in template_needs:
        if dimension not in merged:
            merged.append(str(dimension))
    return tuple(merged)


def dimension_config(cfg) -> dict[str, tuple[str, ...]]:
    """config library.type_dimensions 覆盖代码默认（键=类型，值=维度列表）。"""
    overrides = (getattr(cfg, "library", None) or {}).get("type_dimensions") or {}
    sets = {key: tuple(values) for key, values in DIMENSION_SETS.items()}
    for key, values in overrides.items():
        if isinstance(values, list) and values:
            sets[str(key)] = tuple(str(value) for value in values)
    return sets
