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


def merge_dialogue(asr: list[dict], model: list[dict]) -> list[dict]:
    """ASR/模型对白合并（H1）：SenseVoice 时间戳（电影坐标）为准；
    模型对白只用来补中文翻译/兜底——互相覆盖会把坐标错位或空数组抹掉真值。"""
    if not asr:
        return [dict(m) for m in model if isinstance(m, dict)]
    candidates = [dict(m) for m in model
                  if isinstance(m, dict)
                  and isinstance(m.get("start_s"), (int, float))
                  and isinstance(m.get("end_s"), (int, float))]
    merged = []
    for line in asr:
        out = dict(line)
        span = max(1e-6, float(out["end_s"]) - float(out["start_s"]))
        best = None
        for m in candidates:
            overlap = (min(float(out["end_s"]), float(m["end_s"]))
                       - max(float(out["start_s"]), float(m["start_s"])))
            if overlap <= 0:
                continue
            ratio = overlap / span
            if best is None or ratio > best[0]:
                best = (ratio, m)
        if best and best[0] >= 0.5 and str(best[1].get("translation_zh") or "") \
                not in ("", "uncertain"):
            out["translation_zh"] = best[1]["translation_zh"]
            try:
                out["confidence"] = round(max(float(out.get("confidence", 0)),
                                              float(best[1].get("confidence", 0))), 3)
            except (TypeError, ValueError):
                pass
        merged.append(out)
    return merged


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


def _shot_facets(shot: dict, facets: list[dict]) -> dict:
    """把窗口 facet 映射到镜头：按 apply_interval 与镜头区间的实质重叠（≥50%
    镜头时长）——窗口级印象不复制给整窗镜头；仅收 supported 状态的 facet。"""
    start, end = float(shot["start_s"]), float(shot["end_s"])
    duration = max(1e-6, end - start)
    out = {"locations": [], "interactions": [], "age_appearances": [], "era": None}
    for facet in facets or []:
        if facet.get("status") != "supported":
            continue
        interval = facet.get("apply_interval") or [0, 0]
        span = max(1e-6, float(interval[1]) - float(interval[0]))
        overlap = min(end, float(interval[1])) - max(start, float(interval[0]))
        # 双向实质重叠：镜头大半在 facet 适用区内，或 facet 大半落在本镜头内
        # （保护动作这类点状交互证据常短于半镜头，但确实发生在本镜头）
        if overlap < 0.5 * duration and overlap < 0.8 * span:
            continue
        dimension = facet.get("dimension")
        if dimension == "location":
            out["locations"].append(str(facet.get("value") or ""))
        elif dimension == "era":
            out["era"] = str(facet.get("value") or "") or None
        elif dimension == "age_appearance":
            out["age_appearances"].append({
                "entity_id": str(facet.get("entity_id") or ""),
                "age_range": str(facet.get("value") or "")})
        elif dimension == "interaction":
            out["interactions"].append({
                "a_entity_id": str(facet.get("a_entity_id") or ""),
                "b_entity_id": str(facet.get("b_entity_id") or ""),
                "relation": str(facet.get("relation") or "")})
    out["locations"] = [v for v in dict.fromkeys(out["locations"]) if v]
    out["age_appearances"] = [row for row in out["age_appearances"] if row["age_range"]]
    out["interactions"] = [row for row in out["interactions"]
                           if row["a_entity_id"] and row["b_entity_id"]]
    return out


def _facet_search_terms(facets: dict) -> list[str]:
    terms = []
    if facets.get("locations"):
        terms.append("地点：" + "、".join(facets["locations"][:3]))
    if facets.get("era"):
        terms.append("时代：" + facets["era"])
    if facets.get("age_appearances"):
        terms.append("年龄：" + "、".join(sorted({row["age_range"]
                        for row in facets["age_appearances"]}))[:24])
    if facets.get("interactions"):
        terms.append("关系：" + "、".join(sorted({row["relation"]
                        for row in facets["interactions"]}))[:24])
    return terms


def collect_rows(cfg: AppConfig) -> list[dict]:
    """shots/*/result.json + captions.json → 镜头行（无 caption 的镜头跳过并告警）。"""
    rows, skipped = [], 0
    for r in sorted((cfg.paths.library_dir / "shots").glob("*/result.json")):
        env = json.loads(r.read_text(encoding="utf-8"))
        caps = {}
        cap_path = r.parent / "captions.json"
        if cap_path.exists():
            caps = json.loads(cap_path.read_text(encoding="utf-8"))
        annotations = {}
        causal_predecessors: dict[str, list[str]] = {}
        annotation_path = r.parent / "narrative_annotations.json"
        if annotation_path.exists():
            annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotations = annotation_payload.get("shots") or {}
            for link in annotation_payload.get("causal_links") or []:
                if isinstance(link, dict) and link.get("from_event") and link.get("to_event"):
                    causal_predecessors.setdefault(str(link["to_event"]), []).append(
                        str(link["from_event"]))
        # 类型维度 facet（P1b，加性）：无 type_facets.json 时零影响
        window_facets: dict[int, list[dict]] = {}
        facets_path = r.parent / "type_facets.json"
        if facets_path.exists():
            try:
                facets_payload = json.loads(facets_path.read_text(encoding="utf-8"))
                window_facets = {int(key): (row or {}).get("facets") or []
                                 for key, row in (facets_payload.get("windows") or {}).items()}
            except ValueError:
                window_facets = {}
        for s in env["output"]["shots"]:
            cap = caps.get(str(s["shot_idx"]), "").strip()
            if not cap:
                skipped += 1
                cap = FALLBACK_CAPTION
            annotation = annotations.get(str(s["shot_idx"]), {})
            row = {**s, **annotation, "caption": cap}
            row["dialogue"] = merge_dialogue(s.get("dialogue") or [],
                                             annotation.get("dialogue") or [])
            row["causal_predecessors"] = causal_predecessors.get(
                str(row.get("event_id") or ""), [])
            # 老格式镜头行（预告片时代）没有 window_idx 键——取不到就不做 facet 映射
            facets = _shot_facets(
                s, window_facets.get(int(s["window_idx"])) or []
                if s.get("window_idx") is not None else [])
            if window_facets:
                row["facets"] = facets
            emotion = str(row.get("emotion") or "")
            dialogue_text = " ".join(
                str(line.get("translation_zh") or line.get("original") or "")
                for line in row.get("dialogue") or [] if isinstance(line, dict))
            row["search_text"] = "；".join(value for value in (
                cap, str(row.get("event_summary") or ""),
                "人物：" + "、".join(row.get("entity_names") or []),
                "叙事角色：" + str(row.get("story_role") or ""),
                # emotion 落进可检索文本（P1：不再只标注不消费；uncertain 不入）
                ("情绪：" + emotion) if emotion and emotion != "uncertain" else "",
                *_facet_search_terms(facets), dialogue_text,
            ) if value)
            rows.append(row)
    if skipped:
        logger.warning("[index] %d 个镜头无描述（用兜底文案，建议补跑 caption_shots）", skipped)
    return rows


def build(cfg: AppConfig) -> Path:
    rows = collect_rows(cfg)
    if not rows:
        raise FileNotFoundError("素材库无镜头（先跑 index_shots）")
    embedder = E5Embedder(cfg.library.get("embed") or {})

    emb = embedder.embed([r.get("search_text") or r["caption"] for r in rows])
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
