"""Human-readable, evidence-first run report."""
from __future__ import annotations

from pathlib import Path


def write_run_report(output: Path, *, recipe: dict, asset_plan: dict,
                     retrieval: list[dict], render_manifest: dict,
                     critiques: list[dict], narrative: dict | None = None,
                     story_plan: dict | None = None,
                     narrative_critiques: list[dict] | None = None) -> Path:
    operations = recipe.get("operations") or []
    supported = sum(op.get("status") == "supported" for op in operations)
    uncertain = sum(op.get("status") == "uncertain" for op in operations)
    runtime = render_manifest.get("operations") or []
    unsupported = [row for row in runtime if row.get("status") == "unsupported"]
    lines = [
        f"# Agentic Video Run · {recipe['reference']['id']}", "",
        f"- Recipe：v{recipe['recipe_version']}，{len(operations)} 个操作，"
        f"{supported} supported，{uncertain} uncertain",
        f"- 主题：{asset_plan['theme']}",
        f"- 素材库：{asset_plan['library']}，{len(asset_plan['slots'])} 个时间槽",
        f"- 检索缺口：{sum(bool(row.get('missing')) for row in retrieval)}",
        f"- 运行时不支持操作：{len(unsupported)}", "",
    ]
    if narrative:
        lines.extend([
            "## Narrative Program", "",
            f"- 状态：{narrative.get('status')}，顶层证据 "
            f"{len(narrative.get('evidence') or [])} 条",
            f"- 内容主题：{narrative.get('intent', {}).get('topic')}",
            f"- 核心表达：{narrative.get('intent', {}).get('message')}",
            f"- 实体/事件/因果：{len(narrative.get('entities') or [])}/"
            f"{len(narrative.get('events') or [])}/"
            f"{len(narrative.get('causal_links') or [])}", "",
        ])
        for segment in narrative.get("arc") or []:
            lines.append(f"- **{segment.get('role')}**："
                         f"{', '.join(segment.get('event_ids') or [])}")
        lines.append("")
    if story_plan:
        lines.extend(["## Story Plan", ""])
        for slot in story_plan.get("slots") or []:
            source = slot.get("source") or {}
            lines.append(f"- 槽 {slot.get('slot_idx')} **{slot.get('role')}**："
                         f"{source.get('event_id') or '无事件'} · "
                         f"{source.get('caption') or slot.get('reason') or ''}")
        lines.append("")
    lines.extend(["## Editing Recipe 时间轴", ""])
    for op in operations:
        evidence = op.get("evidence") or []
        quote = str(evidence[0].get("quote") or evidence[0].get("source") or "") \
            if evidence else "无证据"
        lines.append(f"- `{op['interval'][0]:.3f}~{op['interval'][1]:.3f}s` "
                     f"**{op['type']}** [{op['status']}/{op['confidence']:.2f}] — {quote}")
    lines.extend(["", "## 素材检索", ""])
    for slot, result in zip(asset_plan["slots"], retrieval):
        picked = result.get("picked")
        chosen = (f"{picked['video_stem']}#{picked['shot_idx']} · {picked.get('caption', '')}"
                  if picked else "无素材")
        suffix = f" ({result['missing']})" if result.get("missing") else ""
        lines.append(f"- 槽 {slot['slot_idx']} `{slot['start_s']:.2f}~{slot['end_s']:.2f}s`："
                     f"{chosen}{suffix}")
    if unsupported:
        lines.extend(["", "## 未执行操作", ""])
        for row in unsupported:
            lines.append(f"- `{row['operation_id']}`：{row.get('reason', 'unsupported')}")
    if critiques:
        lines.extend(["", "## Edit Critic", ""])
        for index, critique in enumerate(critiques, 1):
            lines.append(f"- Round {index}：score={critique.get('score')}，"
                         f"{critique.get('verdict', '')}")
    if narrative_critiques:
        lines.extend(["", "## Narrative Critic（模型裁判，仅作 Demo 指标）", ""])
        for index, critique in enumerate(narrative_critiques, 1):
            comprehension = critique.get("comprehension") or {}
            lines.append(f"- Round {index}：理解题 "
                         f"{comprehension.get('correct', 0)}/"
                         f"{comprehension.get('total', 5)}；"
                         f"主题={critique.get('theme_relevance')}；"
                         f"连贯={critique.get('narrative_coherence')}；"
                         f"{critique.get('verdict', '')}")
    output = Path(output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
