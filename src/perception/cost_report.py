"""cost_report：聚合 metrics.jsonl → 感知成本报表（Phase 3 对比实验底表）。

CLI：python -m src.perception.cost_report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)


def load_rows(metrics_path: Path) -> list[dict]:
    rows = []
    if not metrics_path.exists():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def aggregate(rows: list[dict]) -> dict:
    per_tool: dict[str, dict] = defaultdict(lambda: {
        "n_ok": 0, "n_error": 0, "elapsed_total_s": 0.0,
        "input_tokens_total": 0, "output_tokens_total": 0, "vram_peak_gb_max": 0.0,
    })
    per_video: dict[tuple[str, str], dict] = {}
    for r in rows:
        tool, aid, status = r.get("tool", "?"), r.get("aweme_id", "?"), r.get("status", "?")
        agg = per_tool[tool]
        if status == "ok":
            agg["n_ok"] += 1
            agg["elapsed_total_s"] += float(r.get("elapsed_s") or 0)
            agg["input_tokens_total"] += int(r.get("input_tokens") or 0)
            agg["output_tokens_total"] += int(r.get("output_tokens") or 0)
            for v in r.get("vram_peak_gb") or []:
                agg["vram_peak_gb_max"] = max(agg["vram_peak_gb_max"], float(v))
        else:
            agg["n_error"] += 1
        cell = per_video.setdefault((tool, aid), {"status": status, "elapsed_s": r.get("elapsed_s"),
                                                  "input_tokens": r.get("input_tokens")})
        if status == "ok":
            cell.update({"elapsed_s": r.get("elapsed_s"), "input_tokens": r.get("input_tokens"),
                         "output_tokens": r.get("output_tokens")})
    return {"per_tool": dict(per_tool), "per_video": {f"{t}|{a}": v for (t, a), v in sorted(per_video.items())}}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="感知成本报表")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg: AppConfig = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_cost")

    rows = load_rows(common.metrics_path_for(cfg.paths.perception_dir))
    if not rows:
        print("metrics.jsonl 为空")
        return 1
    report = aggregate(rows)
    out = cfg.paths.processed_dir / "perception_cost_summary.json"
    out.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"), **report},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("感知工具成本报表（来自 metrics.jsonl，error 调用不计耗时）")
    print("=" * 78)
    print(f"{'tool':22s} {'ok':>3s} {'err':>3s} {'elapsed_s':>10s} {'mean_s':>8s} "
          f"{'in_tok':>8s} {'out_tok':>8s} {'vram_max':>8s}")
    for tool, a in sorted(report["per_tool"].items()):
        mean = a["elapsed_total_s"] / a["n_ok"] if a["n_ok"] else 0
        print(f"{tool:22s} {a['n_ok']:3d} {a['n_error']:3d} {a['elapsed_total_s']:10.1f} "
              f"{mean:8.1f} {a['input_tokens_total']:8d} {a['output_tokens_total']:8d} "
              f"{a['vram_peak_gb_max']:8.1f}")
    print(f"\n产物: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
