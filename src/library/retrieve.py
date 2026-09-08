"""retrieve：C3a——故事板段需求 → 素材镜头语义检索（E5 文本嵌入，VLM caption 侧）。

纯函数 rank_segments 可单测；GPU 只在 query 嵌入一步（秒级）。
CLI：python -m src.library.retrieve --template-id <id>
产物：data/library/stories/<tid>/selection.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.library.build_index import E5Embedder, load_index
from src.perception import common

logger = logging.getLogger(__name__)


def rank_segments(rows: list[dict], emb, text_embs, seg_durations: list[float],
                  *, queries: list[str] | None = None, top_k: int = 5,
                  min_cosine: float = 0.18, dedupe_window: int = 2,
                  keyword_boost: float = 0.05,
                  captions: list[str] | None = None) -> list[dict]:
    """每段返回 {candidates: [{row_idx, cosine, score}], picked, missing}。

    - cosine：CLIP 图文相似度；score = cosine + 关键词命中加成（查询词根命中 caption）
    - 时长约束：镜头时长 ≥ 段时长×0.8 才可作 picked（短镜头可进 candidates 供人工看）
    - 去重：相邻 dedupe_window 段内不复用同一镜头
    - missing：最高分 < min_cosine 或无满足时长者 → 标记库内不足
    """
    import re

    import numpy as np

    queries = queries or [""] * len(text_embs)
    captions = captions or [r.get("caption") or "" for r in rows]
    used_recent: list[int] = []
    results = []
    for si, (q_emb, q_text, need_dur) in enumerate(zip(text_embs, queries, seg_durations)):
        cos = (emb @ q_emb.reshape(-1)).ravel()
        words = [w for w in re.split(r"\W+", q_text) if len(w) > 3][:6]
        order = np.argsort(-cos)[:200]
        cands = []
        for ri in order:
            score = float(cos[ri])
            if keyword_boost and words \
                    and any(w.lower() in captions[ri].lower() for w in words):
                score += keyword_boost
            cands.append({"row_idx": int(ri), "cosine": round(float(cos[ri]), 4),
                          "score": round(score, 4),
                          "duration_s": rows[ri]["duration_s"]})
        picked, missing = None, ""
        best = cands[0] if cands else None
        for c in cands[:top_k * 4]:
            if c["row_idx"] in used_recent:
                continue
            if c["score"] < min_cosine:
                break                      # cands 按分降序，后面只会更低
            if c["duration_s"] >= max(need_dur * 0.8, 0.4):
                picked = c
                break
        if picked is None:
            if best is None or best["score"] < min_cosine:
                missing = (f"库内不足（最高分 {best['score'] if best else 0:.3f}"
                           f" < {min_cosine}）")
            else:
                picked = best
                missing = (f"时长不足（最优 {best['duration_s']}s < 需 {need_dur:.1f}s），"
                           f"取最优镜头截取")
        if picked:
            used_recent.append(picked["row_idx"])
            used_recent = used_recent[-dedupe_window:]
        results.append({"seg_idx": si, "need_duration_s": round(need_dur, 2),
                        "picked": picked, "missing": missing,
                        "candidates": cands[:top_k]})
    return results


def run_retrieve(cfg: AppConfig, template_id: str, *, force: bool = False) -> Path:
    r_cfg = cfg.library.get("retrieve") or {}
    out_path = cfg.paths.library_dir / "stories" / template_id / "selection.json"
    if out_path.exists() and not force:
        logger.info("[retrieve %s] 已有产物，跳过", template_id)
        return out_path

    sb = json.loads((cfg.paths.library_dir / "stories" / template_id / "storyboard.json")
                    .read_text(encoding="utf-8"))["segments"]
    rows, emb = load_index(cfg)
    embedder = E5Embedder(cfg.library.get("embed") or {})
    # query 用故事板的中文需求句（与 caption 同语言同结构，E5 语义匹配更稳）
    queries = [s.get("need_zh") or f"主体：{s['subject']}；动作：{s['action']}" for s in sb]
    text_embs = embedder.embed(queries)
    durs = [float(s["end"]) - float(s["start"]) for s in sb]
    ranked = rank_segments(rows, emb, text_embs, durs, queries=queries,
                           top_k=int(r_cfg.get("top_k", 5)),
                           min_cosine=float(r_cfg.get("min_cosine", 0.18)),
                           dedupe_window=int(r_cfg.get("dedupe_window", 2)))
    n_missing = sum(1 for r in ranked if r["missing"])
    out_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[retrieve %s] %d 段，%d 段标记不足 → %s",
                template_id, len(ranked), n_missing, out_path)
    common.emit_status_line("ok", segments=len(ranked), missing=n_missing)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="故事板→素材镜头检索")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_retrieve")
    try:
        run_retrieve(cfg, args.template_id, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("retrieve 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
