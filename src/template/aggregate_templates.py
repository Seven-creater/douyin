"""aggregate_templates：跨视频模板聚合报告（批量翻拍适合度底表）。

CLI：python -m src.template.aggregate_templates
产物：data/processed/template_aggregate_<ts>.json + 控制台摘要
适合度为透明加权启发式（各分量原值可见，供人工筛选，非 AI 评分）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.config import ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def load_templates(perception_dir: Path) -> list[dict]:
    out = []
    for d in sorted(Path(perception_dir).iterdir()):
        env = common.read_result_json(d / "template")
        if env and env.get("tool") == "extract_template":
            out.append(env)
    return out


def aggregate(envelopes: list[dict]) -> dict:
    n = len(envelopes)
    fixed_counter: Counter = Counter()
    replace_counter: Counter = Counter()
    fixed_where: dict[str, list[str]] = defaultdict(list)
    replace_where: dict[str, list[str]] = defaultdict(list)
    role_seqs: Counter = Counter()
    role_dist: Counter = Counter()
    twist_positions = []
    durations = []
    seg_counts = []
    mode_stats = Counter()
    warnings_counter: Counter = Counter()
    suitability_rows = []

    for env in envelopes:
        aweme_id = env["aweme_id"]
        tpl = env["output"]["template"]
        meta = env["output"].get("parse_meta") or {}
        mode_stats[meta.get("mode", "?")] += 1
        for w in meta.get("warnings") or []:
            warnings_counter[w.split("，")[0][:40]] += 1

        for e in tpl.get("fixed_elements") or []:
            k = e.strip().lower()
            fixed_counter[k] += 1
            fixed_where[k].append(aweme_id)
        for e in tpl.get("replaceable_elements") or []:
            k = e.strip().lower()
            replace_counter[k] += 1
            replace_where[k].append(aweme_id)

        tl = tpl.get("timeline") or []
        seq = "→".join(s.get("role", "other") for s in tl)
        role_seqs[seq] += 1
        for s in tl:
            role_dist[s.get("role", "other")] += 1
        seg_counts.append(len(tl))
        dur = env["output"].get("video", {}).get("duration_s") or 0
        durations.append(dur)
        twist_idx = [i for i, s in enumerate(tl) if s.get("role") == "twist"]
        if twist_idx and tl:
            twist_positions.append(round(twist_idx[0] / len(tl), 2))

        # 适合度（透明加权，分量保留）
        rep = len(tpl.get("replaceable_elements") or [])
        fix = len(tpl.get("fixed_elements") or [])
        seq_freq = role_seqs[seq] / max(n, 1)
        has_early_twist = 1.0 if twist_idx and dur and (sum(s["end"] - s["start"] for i, s in enumerate(tl) if i <= twist_idx[0]) / dur) <= 0.6 else 0.0
        dur_fit = 1.0 if 8 <= dur <= 35 else 0.0
        rep_ratio = rep / (rep + fix) if (rep + fix) else 0.0
        parse_ok = 1.0 if meta.get("mode") == "json" else 0.0
        score = round(0.3 * seq_freq + 0.2 * has_early_twist + 0.2 * dur_fit + 0.2 * rep_ratio + 0.1 * parse_ok, 3)
        suitability_rows.append({
            "aweme_id": aweme_id, "score": score,
            "components": {"seq_freq": round(seq_freq, 3), "early_twist": has_early_twist,
                           "dur_fit": dur_fit, "replace_ratio": round(rep_ratio, 3),
                           "parse_json": parse_ok},
            "role_seq": seq, "duration_s": dur, "n_segments": len(tl),
        })

    def _top(counter: Counter, where: dict, k: int = 15):
        return [{"element": e, "count": c, "videos": where.get(e, [])}
                for e, c in counter.most_common(k)]

    return {
        "n_templates": n,
        "parse_modes": dict(mode_stats),
        "fixed_top": _top(fixed_counter, fixed_where),
        "replaceable_top": _top(replace_counter, replace_where),
        "role_sequences": [{"seq": s, "count": c} for s, c in role_seqs.most_common(10)],
        "role_distribution": dict(role_dist),
        "twist_position_ratio_of_segments": twist_positions,
        "segment_count": {"mean": round(sum(seg_counts) / n, 1) if n else 0,
                          "min": min(seg_counts) if seg_counts else 0,
                          "max": max(seg_counts) if seg_counts else 0},
        "duration_buckets": _buckets(durations),
        "suitability": sorted(suitability_rows, key=lambda r: -r["score"]),
        "warnings_top": [{"warning": w, "count": c} for w, c in warnings_counter.most_common(10)],
    }


def _buckets(durations: list[float]) -> dict:
    buckets = {"<10s": 0, "10-20s": 0, "20-40s": 0, ">40s": 0}
    for d in durations:
        if d < 10:
            buckets["<10s"] += 1
        elif d < 20:
            buckets["10-20s"] += 1
        elif d < 40:
            buckets["20-40s"] += 1
        else:
            buckets[">40s"] += 1
    return buckets


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="模板聚合报告")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="template_aggregate")

    envelopes = load_templates(cfg.paths.perception_dir)
    if not envelopes:
        print("没有 template 产物（先跑 run_templates）")
        return 1
    report = aggregate(envelopes)
    out = cfg.paths.processed_dir / f"template_aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"模板聚合报告（{report['n_templates']} 条） · parse_modes={report['parse_modes']}")
    print("=" * 70)
    print("高频固定元素 top5:")
    for r in report["fixed_top"][:5]:
        print(f"  {r['count']:2d}x {r['element'][:50]}")
    print("高频可替换元素 top5:")
    for r in report["replaceable_top"][:5]:
        print(f"  {r['count']:2d}x {r['element'][:50]}")
    print(f"角色序列 top5: {[(r['seq'], r['count']) for r in report['role_sequences'][:5]]}")
    print(f"时长分布: {report['duration_buckets']} · 平均段数 {report['segment_count']['mean']}")
    print("批量翻拍适合度 top5:")
    for r in report["suitability"][:5]:
        print(f"  {r['score']:.3f} {r['aweme_id']} {r['role_seq'][:40]} {r['duration_s']}s")
    print(f"\n产物: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
