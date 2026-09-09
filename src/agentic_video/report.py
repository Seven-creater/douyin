"""Human-readable, evidence-first run report."""
from __future__ import annotations

from pathlib import Path


def write_run_report(output: Path, *, recipe: dict, asset_plan: dict,
                     retrieval: list[dict], render_manifest: dict,
                     critiques: list[dict]) -> Path:
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
        "## Editing Recipe 时间轴", "",
    ]
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
        lines.extend(["", "## Critic", ""])
        for index, critique in enumerate(critiques, 1):
            lines.append(f"- Round {index}：score={critique.get('score')}，"
                         f"{critique.get('verdict', '')}")
    output = Path(output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
