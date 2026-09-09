"""build_index：素材库 B3——VLM 镜头描述 → E5 文本嵌入 → 可检索索引。

架构（用户拍板 2026-09-08）：不用 CLIP 图文检索，走「VLM 看画面转文本 → 文本语义检索」
——检索可解释（命中理由就是描述文字），且描述带主体/动作/场景/情绪结构。
嵌入模型：E5-Omni-7B（服务器自有，sentence-transformers 5.4 原生支持）。

产物：
  data/library/index/shots.jsonl   每行一个镜头（含 caption）
  data/library/index/cap_emb.npy   [N, D] 归一化 caption 嵌入（行号= jsonl 行号）
CLI：python -m src.library.build_index
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

FALLBACK_CAPTION = "电影画面"

# 访谈/花絮类镜头黑名单（critic 首跑实测：预告合集混入对镜头说话的低运动画面，
# 卡点混剪需要高运动镜头；在 pick/swap 侧跳过，不动索引本体）
JUNK_KEYWORDS = ("访谈", "采访", "花絮", "幕后", "制作特辑", "对镜头说", "面向镜头",
                 "记者", "发布会", "首映礼")


def is_junk_caption(text: str) -> bool:
    return any(k in (text or "") for k in JUNK_KEYWORDS)


class E5Embedder:
    """E5-Omni-7B 文本嵌入（caption 与 query 同一侧，cosine 可比）。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None

    def load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        path = self.cfg.get("model_path")
        dev = self.cfg.get("device", "cuda:0")
        logger.info("[e5] 加载: %s → %s", path, dev)
        self._model = SentenceTransformer(path, device=dev)

    def embed(self, texts: list[str], batch: int = 16):
        self.load()
        vecs = self._model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                  show_progress_bar=False)
        return vecs.astype("float32")


def collect_rows(cfg: AppConfig) -> list[dict]:
    """shots/*/result.json + captions.json → 镜头行（无 caption 的镜头跳过并告警）。"""
    rows, skipped = [], 0
    for r in sorted((cfg.paths.library_dir / "shots").glob("*/result.json")):
        env = json.loads(r.read_text(encoding="utf-8"))
        caps = {}
        cap_path = r.parent / "captions.json"
        if cap_path.exists():
            caps = json.loads(cap_path.read_text(encoding="utf-8"))
        for s in env["output"]["shots"]:
            cap = caps.get(str(s["shot_idx"]), "").strip()
            if not cap:
                skipped += 1
                cap = FALLBACK_CAPTION
            rows.append({**s, "caption": cap})
    if skipped:
        logger.warning("[index] %d 个镜头无描述（用兜底文案，建议补跑 caption_shots）", skipped)
    return rows


def build(cfg: AppConfig) -> Path:
    rows = collect_rows(cfg)
    if not rows:
        raise FileNotFoundError("素材库无镜头（先跑 index_shots）")
    embedder = E5Embedder(cfg.library.get("embed") or {})

    emb = embedder.embed([r["caption"] for r in rows])
    out_dir = cfg.paths.library_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "shots.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import numpy as np

    np.save(out_dir / "cap_emb.npy", emb)
    logger.info("[index] %d 镜头，emb %s → %s", len(rows), emb.shape, jsonl)
    common.emit_status_line("ok", shots=len(rows), dim=list(emb.shape))
    return jsonl


def load_index(cfg: AppConfig):
    """C3 检索入口：返回 (rows, emb)。"""
    import numpy as np

    out_dir = cfg.paths.library_dir / "index"
    rows = [json.loads(l) for l in (out_dir / "shots.jsonl")
            .read_text(encoding="utf-8").splitlines() if l]
    emb = np.load(out_dir / "cap_emb.npy")
    if len(rows) != emb.shape[0]:
        raise RuntimeError(f"索引不同步：rows={len(rows)} emb={emb.shape[0]}（重跑 build_index）")
    return rows, emb


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="构建素材库文本语义索引（E5）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_index")
    try:
        build(cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("build_index 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
