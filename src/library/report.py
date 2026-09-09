"""report：D4 Decomposer 基准对比报告——参考片 vs 重剪版的 Recipe 指标 + 人工验收清单。

校准目标（分析文档实测，2026-09-07）：参考片 ~35 编辑事件 vs GLM 版 ~16；
节拍对齐 ~85.7% vs ~100%；两片同 BPM（~112.35）。报告若复现这个形状，
说明 Decomposer 抓到了"卡点已解决、编辑操作理解缺失"的差距。

CLI：python -m src.library.report --id-a 7668161826782078931 --id-b reedit_glm_001
产物：data/library/editing/report/<a>_vs_<b>/{report.md, result.json}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)

# 人工验收必查现象（分析文档点名；每条带"怎么看"）
VERIFY_GUIDES = {
    "tracked_mask_fill": "0.25 倍速拖到 t±0.3s：人物轮廓内部纹理是否连续替换（外部不变）",
    "foreground_occlusion_transition": "慢放转场：是否由前景物体扫过画面完成 A→B 切换",
    "luma_flash": "逐帧看：是否整屏闪白/闪黑一帧后回落",
    "montage_burst": "t 附近 1.2s 内是否 ≥3 连切",
    "text_layer_animation": "文字层是否单独出现/消失/位移（不在素材画面内）",
    "zoom_punch": "画面是否急推/急拉一拍",
    "whip_pan": "是否横向甩镜衔接两个场景",
    "beat_freeze": "是否有一拍画面定格",
    "match_cut": "切点两侧构图/运动是否刻意对位",
    "speed_ramp": "同镜头内是否明显加速/减速",
    "mask_wipe": "是否有遮罩扫过完成画面替换",
    "subject_cutout_composite": "主体是否被抠出叠加到别的背景",
    "crossfade": "切点是否叠化过渡",
    "hard_cut": "确认这里是普通硬切（对照项）",
}
GUARANTEED_ABSENT_NOTE = "（Decomposer 未在任一片中主张此现象——可复核是否漏检）"


def recipe_metrics(recipe: dict, beats: list[float], *, beat_tol_s: float = 0.15) -> dict:
    """Recipe + 节拍 → 指标块。纯函数。"""
    ops = [o for o in (recipe.get("operations") or []) if isinstance(o, dict)]
    d = float((recipe.get("video") or {}).get("duration_s") or 0)
    hist: dict[str, int] = {}
    for o in ops:
        hist[o.get("type", "?")] = hist.get(o.get("type", "?"), 0) + 1
    aligned = sum(1 for o in ops if isinstance(o.get("t_s"), (int, float))
                  and any(abs(o["t_s"] - b) <= beat_tol_s for b in beats))
    segs = [s for s in (recipe.get("segments") or []) if isinstance(s, dict)]
    seg_lens = [max(0.0, s.get("end", 0) - s.get("start", 0)) for s in segs]
    return {"n_ops": len(ops),
            "ops_per_min": round(len(ops) * 60 / d, 1) if d else None,
            "op_hist": hist,
            "beat_aligned_pct": round(100 * aligned / len(ops), 1) if ops else None,
            "n_montage": hist.get("montage_burst", 0),
            "mean_seg_s": round(sum(seg_lens) / len(seg_lens), 2) if seg_lens else None,
            "tempo_bpm_est": recipe.get("tempo_bpm_est"),
            "n_segments": len(segs)}


def build_checklist(recipes: dict[str, dict]) -> list[dict]:
    """两片 Recipe → 人工验收清单（每主张一条 + 对照项 + 2 个自由栏）。"""
    items: list[dict] = []
    seen: set[tuple] = set()
    for vid, rec in recipes.items():
        for o in rec.get("operations") or []:
            ty = o.get("type")
            if ty not in VERIFY_GUIDES or (vid, ty) in seen:
                continue
            seen.add((vid, ty))
            items.append({"video": vid, "t_s": o.get("t_s"), "phenomenon": ty,
                          "claim": f"Decomposer 主张 {ty}（conf={o.get('confidence')}）",
                          "how": VERIFY_GUIDES[ty]})
    # 基准可信度对照项：Decomposer 没主张过的必查现象（测漏检）
    claimed = {ty for _, ty in seen}
    for ty, guide in VERIFY_GUIDES.items():
        if ty not in claimed and ty != "hard_cut":
            items.append({"video": "任一片", "t_s": None, "phenomenon": ty,
                          "claim": GUARANTEED_ABSENT_NOTE, "how": guide})
    items.append({"video": "重剪版", "t_s": None, "phenomenon": "整体观感",
                  "claim": "预期：只有到点换画面（PPT 感），无图层/转场手法",
                  "how": "整片看一遍：除切换外是否有任何其它剪辑手法"})
    for i in range(1, 3):
        items.append({"video": "自由", "t_s": None, "phenomenon": f"意外发现 #{i}",
                      "claim": "（人工填写）", "how": "看片中发现的其他手法/问题"})
    return items


def render_markdown(a: str, b: str, ma: dict, mb: dict, ra: dict, rb: dict,
                    checklist: list[dict]) -> str:
    lines = [f"# Decomposer 基准对比：{a}（参考） vs {b}（重剪）", "",
             "校准目标（分析文档实测）：参考 ~35 事件 vs 重剪 ~16；节拍对齐 ~85.7% vs ~100%。", "",
             "## 指标对照", "", "| 指标 | 参考 | 重剪 |", "|---|---|---|"]
    for k in ("n_ops", "ops_per_min", "beat_aligned_pct", "n_montage",
              "mean_seg_s", "tempo_bpm_est", "n_segments"):
        lines.append(f"| {k} | {ma.get(k)} | {mb.get(k)} |")
    lines += ["", "## 操作类型直方图差", "", "| 类型 | 参考 | 重剪 |", "|---|---|---|"]
    for ty in sorted(set(ma["op_hist"]) | set(mb["op_hist"])):
        lines.append(f"| {ty} | {ma['op_hist'].get(ty, 0)} | {mb['op_hist'].get(ty, 0)} |")
    for vid, rec in ((a, ra), (b, rb)):
        lines += ["", f"## {vid} 事件表", "", "| t_s | type | conf | evidence |",
                  "|---|---|---|---|"]
        for o in rec.get("operations") or []:
            ev = o.get("evidence") or {}
            quote = ev.get("quote") or ev.get("sig") or ""
            lines.append(f"| {o.get('t_s')} | {o.get('type')} | {o.get('confidence')}"
                         f" | {str(quote)[:40]} |")
    lines += ["", "## 人工验收清单（≤30min，逐条打勾/叉）", "",
              "| 状态 | 片 | t_s | 现象 | Decomposer 主张 | 怎么看 |",
              "|---|---|---|---|---|---|"]
    for it in checklist:
        lines.append(f"| [ ] | {it['video']} | {it['t_s']} | {it['phenomenon']}"
                     f" | {it['claim']} | {it['how']} |")
    lines.append("")
    return "\n".join(lines)


def run_report(cfg: AppConfig, id_a: str, id_b: str, *, force: bool = False) -> Path:
    rep_cfg = cfg.library.get("report") or {}
    beat_tol = float(rep_cfg.get("beat_tolerance_s", 0.15))
    out_dir = cfg.paths.library_dir / "editing" / "report" / f"{id_a}_vs_{id_b}"
    md = out_dir / "report.md"
    if md.exists() and not force:
        logger.info("[report] 已有产物，跳过")
        return md

    recipes = {}
    for vid in (id_a, id_b):
        env = common.read_result_json(cfg.paths.library_dir / "editing" / vid / "recipe")
        if not env:
            raise FileNotFoundError(f"缺 recipe 产物（先跑 D3）：{vid}")
        recipes[vid] = env["output"]["recipe"]
        beats_env = common.read_result_json(cfg.paths.perception_dir / vid / "beats")
        recipes[f"{vid}__beats"] = list(((beats_env or {}).get("output") or {})
                                        .get("beat_points_s") or [])
    ma = recipe_metrics(recipes[id_a], recipes[f"{id_a}__beats"], beat_tol_s=beat_tol)
    mb = recipe_metrics(recipes[id_b], recipes[f"{id_b}__beats"], beat_tol_s=beat_tol)
    checklist = build_checklist({id_a: recipes[id_a], id_b: recipes[id_b]})
    text = render_markdown(id_a, id_b, ma, mb, recipes[id_a], recipes[id_b], checklist)

    out_dir.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps({
        "id_a": id_a, "id_b": id_b, "metrics_a": ma, "metrics_b": mb,
        "checklist": checklist}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[report] %d ops vs %d ops，清单 %d 条 → %s",
                ma["n_ops"], mb["n_ops"], len(checklist), md)
    common.emit_status_line("ok", a=ma["n_ops"], b=mb["n_ops"], items=len(checklist))
    return md


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="D4 Decomposer 基准对比报告")
    ap.add_argument("--id-a", required=True)
    ap.add_argument("--id-b", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_report")
    try:
        run_report(cfg, args.id_a, args.id_b, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("report 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
