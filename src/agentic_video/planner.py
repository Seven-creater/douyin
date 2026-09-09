"""Theme-driven asset requirement planning and explainable retrieval."""
from __future__ import annotations

import json
from pathlib import Path

from src.agentic_video.recipe_v2 import validate_recipe_v2
from src.config import AppConfig
from src.library.build_index import E5Embedder, load_index


def timeline_slots(recipe: dict) -> list[tuple[float, float]]:
    duration = float(recipe["reference"]["duration_s"])
    cuts = {0.0, duration}
    for op in recipe.get("operations") or []:
        if op.get("type") in {"hard_cut", "match_cut", "montage_burst"}:
            cuts.add(float(op["interval"][0]))
    bounds = sorted(value for value in cuts if 0 <= value <= duration)
    return [(start, end) for start, end in zip(bounds, bounds[1:]) if end - start >= 0.05]


def build_asset_plan(recipe: dict, theme: str, *, library: str) -> dict:
    errors = validate_recipe_v2(recipe)
    if errors:
        raise ValueError("invalid Recipe v2: " + "; ".join(errors))
    if not theme.strip():
        raise ValueError("theme must not be empty")
    slots = []
    for index, (start, end) in enumerate(timeline_slots(recipe)):
        active = [op for op in recipe.get("operations") or []
                  if op["interval"][0] <= end and op["interval"][1] >= start]
        subjects = [str((op.get("params") or {}).get("subject") or "")
                    for op in active if (op.get("params") or {}).get("subject")]
        motion = [str((op.get("params") or {}).get("motion") or "")
                  for op in active if (op.get("params") or {}).get("motion")]
        operation_types = sorted({op["type"] for op in active})
        intensity = min(1.0, 0.35 + 0.12 * len(active)
                        + (0.2 if any(t in {"speed_ramp", "whip_pan", "montage_burst"}
                                          for t in operation_types) else 0.0))
        scale = "closeup" if any(t in {"zoom_punch", "crop_reframe"}
                                 for t in operation_types) else "uncertain"
        descriptors = [theme.strip()]
        if subjects:
            descriptors.append("主体：" + "、".join(dict.fromkeys(subjects)))
        if motion:
            descriptors.append("动作/运动：" + "、".join(dict.fromkeys(motion)))
        descriptors.append(f"视觉强度：{intensity:.2f}")
        if scale != "uncertain":
            descriptors.append("镜头尺度：特写")
        slots.append({
            "slot_idx": index, "start_s": round(start, 6), "end_s": round(end, 6),
            "need_duration_s": round(end - start, 6), "theme": theme.strip(),
            "subject": subjects[0] if subjects else "", "motion": motion[0] if motion else "",
            "shot_scale": scale, "visual_intensity": round(intensity, 3),
            "operation_types": operation_types, "query": "；".join(descriptors),
        })
    return {"plan_version": "1.0", "theme": theme.strip(), "library": library,
            "reference_id": recipe["reference"]["id"], "slots": slots}


def rank_asset_slots(rows: list[dict], embeddings, query_embeddings, slots: list[dict], *,
                     source: str | None = None, top_k: int = 5, min_score: float = 0.18,
                     dedupe_window: int = 2) -> list[dict]:
    allowed = [idx for idx, row in enumerate(rows)
               if source is None or row.get("source") == source
               or str(row.get("video_stem", "")).startswith(f"{source}__")]
    if not allowed:
        raise ValueError(f"index has no rows for source {source!r}")
    used_recent: list[int] = []
    results = []
    for slot, query in zip(slots, query_embeddings):
        cosine = (embeddings @ query.reshape(-1)).ravel()
        candidates = []
        for row_idx in allowed:
            row = rows[row_idx]
            action = float(row.get("action_score", 0.5))
            desired = float(slot.get("visual_intensity", 0.5))
            action_fit = 1.0 - abs(action - desired)
            score = float(cosine[row_idx]) + 0.08 * action_fit
            candidates.append({
                "row_idx": row_idx, "cosine": round(float(cosine[row_idx]), 4),
                "score": round(score, 4), "action_fit": round(action_fit, 4),
                "duration_s": float(row["duration_s"]), "video": row["video"],
                "video_stem": row["video_stem"], "shot_idx": row["shot_idx"],
                "source_start_s": float(row["start_s"]),
                "source_end_s": float(row["end_s"]), "caption": row.get("caption", ""),
            })
        candidates.sort(key=lambda item: (-item["score"], item["row_idx"]))
        picked = None
        for candidate in candidates:
            if candidate["row_idx"] in used_recent or candidate["score"] < min_score:
                continue
            if candidate["duration_s"] >= min(float(slot["need_duration_s"]), 0.45):
                picked = candidate
                break
        missing = ""
        if picked is None:
            viable = [candidate for candidate in candidates if candidate["score"] >= min_score]
            if viable:
                picked = viable[0]
                missing = "duration_or_dedupe_constraint_relaxed"
            else:
                missing = "library_insufficient"
        if picked:
            used_recent.append(picked["row_idx"])
            used_recent = used_recent[-max(1, dedupe_window):]
        results.append({"slot_idx": slot["slot_idx"], "query": slot["query"],
                        "picked": picked, "missing": missing,
                        "candidates": candidates[:top_k]})
    return results


def run_asset_retrieval(cfg: AppConfig, asset_plan: dict, output: Path, *,
                        force: bool = False) -> Path:
    output = Path(output)
    if output.exists() and not force:
        return output
    rows, embeddings = load_index(cfg)
    slots = asset_plan["slots"]
    embedder = E5Embedder(cfg.library.get("embed") or {})
    query_embeddings = embedder.embed([slot["query"] for slot in slots])
    retrieve_cfg = cfg.library.get("retrieve") or {}
    ranked = rank_asset_slots(
        rows, embeddings, query_embeddings, slots, source=asset_plan.get("library"),
        top_k=int(retrieve_cfg.get("top_k", 5)),
        min_score=float(retrieve_cfg.get("min_cosine", 0.18)),
        dedupe_window=int(retrieve_cfg.get("dedupe_window", 2)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
