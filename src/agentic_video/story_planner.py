"""Traceable, continuity-aware story planning over narrative index rows."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from src.agentic_video.narrative import ARC_ROLES, validate_narrative_program
from src.agentic_video import zones

STORY_PLAN_VERSION = "1.0"
MIN_STORY_SLOTS = 3
_EXPAND_ROLES = ("conflict", "climax", "resolution")


def _transition_score(left: dict, right: dict) -> float:
    score = 0.0
    if set(left.get("entity_ids") or []) & set(right.get("entity_ids") or []):
        score += 0.25
    if left.get("video_stem") == right.get("video_stem"):
        score += 0.05
    if float(left.get("source_start_s", 0)) <= float(right.get("source_start_s", 0)):
        score += 0.08
    if left.get("event_id") in set(right.get("causal_predecessors") or []):
        score += 0.25
    if right.get("event_id") in set(left.get("causal_predecessors") or []):
        score -= 0.35
    if left.get("row_idx") == right.get("row_idx"):
        score -= 0.30
    return score


def rank_story_path(candidate_groups: list[list[dict]]) -> list[dict]:
    """Viterbi-like path ranking with explicit entity/time continuity bonuses."""
    if not candidate_groups or any(not group for group in candidate_groups):
        return []
    scores = [{idx: float(row.get("semantic_score", row.get("score", 0)))
               for idx, row in enumerate(candidate_groups[0])}]
    parents: list[dict[int, int | None]] = [{idx: None for idx in scores[0]}]
    for group_idx in range(1, len(candidate_groups)):
        group, previous = candidate_groups[group_idx], candidate_groups[group_idx - 1]
        current_scores: dict[int, float] = {}
        current_parents: dict[int, int] = {}
        for idx, row in enumerate(group):
            options = [(scores[-1][prev_idx] + _transition_score(prev, row), prev_idx)
                       for prev_idx, prev in enumerate(previous)]
            best_score, best_parent = max(options, key=lambda item: (item[0], -item[1]))
            current_scores[idx] = best_score + float(row.get("semantic_score",
                                                             row.get("score", 0)))
            current_parents[idx] = best_parent
        scores.append(current_scores)
        parents.append(current_parents)
    cursor = max(scores[-1], key=lambda idx: (scores[-1][idx], -idx))
    indexes = [cursor]
    for group_idx in range(len(candidate_groups) - 1, 0, -1):
        cursor = parents[group_idx][cursor]
        indexes.append(cursor)
    indexes.reverse()
    return [deepcopy(candidate_groups[idx][row_idx])
            for idx, row_idx in enumerate(indexes)]


def _candidates_for_arc(segment: dict, source_rows: list[dict]) -> list[dict]:
    event_ids = set(segment.get("event_ids") or [])
    role = segment.get("role")
    scored = []
    for row in source_rows:
        score = float(row.get("semantic_score", row.get("score", 0.5)))
        if row.get("event_id") in event_ids:
            score += 0.35
        if row.get("story_role") == role:
            score += 0.20
        candidate = {**row, "semantic_score": score}
        if row.get("event_id") in event_ids or row.get("story_role") == role:
            scored.append(candidate)
    return sorted(scored, key=lambda row: (-row["semantic_score"],
                                           float(row.get("source_start_s", 0))))[:12]


def _expand_thin_arc(narrative: dict, *,
                     min_slots: int = MIN_STORY_SLOTS) -> tuple[dict, list[str]]:
    """参考片叙事弧过薄时补足槽位，禁止单槽撑满整条成片。

    2026-09-10 首跑：参考程序 arc=[hook]（旧缓存、仅 2 事件），60 秒目标
    全灌进 1 个槽，成片从头到尾一个窗口、零剪辑。补位段按冲突→高潮→结局
    取已有时序事件的分段引用；检索侧的兼容角色池会自然把素材分流到不同
    镜头，段上带 synthesized 标记供报告溯源。
    """
    arc = list(narrative.get("arc") or [])
    if len(arc) >= min_slots:
        return narrative, []
    events = sorted(narrative.get("events") or [],
                    key=lambda row: float((row.get("interval") or [0.0, 0.0])[0]))
    if not events:
        return narrative, []
    roles = [role for role in _EXPAND_ROLES
             if role not in {segment["role"] for segment in arc}][:min_slots - len(arc)]
    if not roles:
        return narrative, []
    event_ids = [row["id"] for row in events]
    expanded = deepcopy(narrative)
    for idx, role in enumerate(roles):
        lo = idx * len(event_ids) // len(roles)
        hi = ((idx + 1) * len(event_ids) // len(roles)
              if idx < len(roles) - 1 else len(event_ids))
        expanded["arc"].append({"role": role, "event_ids": event_ids[lo:hi],
                                "synthesized": True})
    order = {role: idx for idx, role in enumerate(ARC_ROLES)}
    expanded["arc"].sort(key=lambda segment: order[segment["role"]])
    expanded["arc_expanded"] = {"reason": "thin_reference_arc", "added_roles": roles}
    return expanded, roles


def build_story_plan(narrative: dict, source_rows: list[dict], *, theme: str,
                     library: str, target_duration_s: float = 60.0) -> dict:
    errors = validate_narrative_program(narrative)
    if errors:
        raise ValueError("invalid Narrative Program: " + "; ".join(errors))
    if not 45 <= target_duration_s <= 75:
        raise ValueError("target_duration_s must be within 45..75")
    narrative, added_roles = _expand_thin_arc(narrative)
    arc = narrative.get("arc") or []
    candidate_groups = [_candidates_for_arc(segment, source_rows) for segment in arc]
    plan = _assemble_story_plan(narrative, candidate_groups, theme=theme, library=library,
                                target_duration_s=target_duration_s)
    if added_roles:
        plan["arc_expanded"] = {"reason": "thin_reference_arc", "added_roles": added_roles}
    return plan


def _assemble_story_plan(narrative: dict, candidate_groups: list[list[dict]], *,
                         theme: str, library: str, target_duration_s: float) -> dict:
    arc = narrative.get("arc") or []
    path: list[dict | None] = [None] * len(candidate_groups)
    run_start = 0
    while run_start < len(candidate_groups):
        if not candidate_groups[run_start]:
            run_start += 1
            continue
        run_end = run_start
        while run_end < len(candidate_groups) and candidate_groups[run_end]:
            run_end += 1
        path[run_start:run_end] = rank_story_path(candidate_groups[run_start:run_end])
        run_start = run_end
    slot_duration = target_duration_s / max(1, len(arc))
    slots = []
    for idx, segment in enumerate(arc):
        start = round(idx * slot_duration, 6)
        end = round(target_duration_s if idx == len(arc) - 1
                    else (idx + 1) * slot_duration, 6)
        picked = path[idx]
        dialogue = list((picked or {}).get("dialogue") or [])
        source_start = float((picked or {}).get("source_start_s", 0))
        source_end = float((picked or {}).get("source_end_s", 0))
        if dialogue:
            source_start = min([source_start] +
                               [float(line.get("start_s", source_start)) for line in dialogue])
            source_end = max([source_end] +
                             [float(line.get("end_s", source_end)) for line in dialogue])
        # 槽预算装不下整个事件段时裁剪而非弃槽（2026-09-10 v4 黑屏事故）：
        # 旧逻辑 dialogue_would_be_cut 一票否决把三个槽全判 uncertain → picked=None
        # → 渲染器 0 segment 纯黑 60s。合并段以事件起点为锚截到预算，对白只保留
        # 完整落在窗内的行（不截半句），裁剪信息记 source_interval_trimmed 供溯源。
        slot_budget = end - start
        source_trimmed = False
        if picked and source_end - source_start > slot_budget + 1e-6:
            source_end = source_start + slot_budget
            source_trimmed = True
        if picked:
            dialogue = [line for line in dialogue
                        if float(line.get("start_s", source_start)) >= source_start - 1e-6
                        and float(line.get("end_s", source_end)) <= source_end + 1e-6]
        source = {
            "video": str((picked or {}).get("video") or ""),
            "video_stem": str((picked or {}).get("video_stem") or ""),
            "shot_idx": (picked or {}).get("shot_idx"),
            "shot_indices": list((picked or {}).get("shot_indices") or []),
            "start_s": source_start, "end_s": source_end,
            "event_id": str((picked or {}).get("event_id") or ""),
            "causal_predecessors": list((picked or {}).get("causal_predecessors") or []),
            "caption": str((picked or {}).get("caption") or ""),
            "entity_ids": list((picked or {}).get("entity_ids") or []),
            "focus_x": min(1.0, max(0.0, float((picked or {}).get("focus_x", 0.5)))),
            "dialogue": dialogue,
        }
        transition_reason = "opening"
        if idx and picked and path[idx - 1]:
            previous = path[idx - 1]
            if set(previous.get("entity_ids") or []) & set(picked.get("entity_ids") or []):
                transition_reason = "entity_continuity"
            elif previous.get("event_id") in set(picked.get("causal_predecessors") or []):
                transition_reason = "causal_transition"
            else:
                transition_reason = "unexplained"
        slot_status = ("supported" if picked and transition_reason != "unexplained"
                       else "uncertain" if picked else "unsupported")
        reason = ("unexplained_entity_switch" if transition_reason == "unexplained"
                  else "" if picked else "library_insufficient")
        slots.append({
            "slot_idx": idx, "role": segment["role"],
            "target_interval": [start, end],
            "reference_event_ids": list(segment.get("event_ids") or []),
            "source": source,
            "status": slot_status,
            "reason": reason,
            "transition_reason": transition_reason,
            "source_interval_trimmed": source_trimmed,
        })
    return {
        "story_plan_version": STORY_PLAN_VERSION, "theme": theme.strip(),
        "library": library, "target_duration_s": round(target_duration_s, 6),
        "reference_id": narrative["reference"]["id"], "slots": slots,
    }


def _arc_query(narrative: dict, segment: dict, theme: str) -> str:
    event_by_id = {event["id"]: event for event in narrative.get("events") or []}
    parts = [theme, f"叙事角色：{segment['role']}"]
    for event_id in segment.get("event_ids") or []:
        event = event_by_id.get(event_id) or {}
        parts.extend([str(event.get("action") or ""), str(event.get("state_before") or ""),
                      str(event.get("state_after") or "")])
    return "；".join(part for part in parts if part and part != "uncertain")


def merge_event_candidates(rows: list[dict]) -> list[dict]:
    """Merge adjacent shot rows that the narrative index assigned to one event.

    Rendering a single midpoint shot for an entire story role creates long held
    frames.  Event ranges preserve the original multi-shot action/dialogue and
    remain traceable to every constituent shot.
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        key = (str(row.get("video") or ""), str(row.get("video_stem") or ""),
               row.get("window_idx"), event_id, row.get("story_role"))
        grouped.setdefault(key, []).append(row)
    merged = []
    for (_video, _stem, _window, event_id, role), members in grouped.items():
        ordered = sorted(members, key=lambda row: float(row.get("source_start_s", 0)))
        first = deepcopy(ordered[0])
        start = min(float(row.get("source_start_s", 0)) for row in ordered)
        end = max(float(row.get("source_end_s", 0)) for row in ordered)
        dialogue = []
        seen_dialogue = set()
        for row in ordered:
            for line in row.get("dialogue") or []:
                key = (line.get("start_s"), line.get("end_s"), line.get("original"),
                       line.get("translation_zh"))
                if key not in seen_dialogue:
                    seen_dialogue.add(key)
                    dialogue.append(deepcopy(line))
        captions = list(dict.fromkeys(str(row.get("event_summary")
                                            or row.get("caption") or "").strip()
                                      for row in ordered
                                      if str(row.get("event_summary")
                                             or row.get("caption") or "").strip()))
        first.update({
            "event_id": event_id, "story_role": role,
            "source_start_s": start, "source_end_s": end,
            "duration_s": end - start,
            "shot_idx": ordered[0].get("shot_idx"),
            "shot_indices": [row.get("shot_idx") for row in ordered],
            "caption": "；".join(captions),
            "entity_ids": sorted({str(value) for row in ordered
                                  for value in row.get("entity_ids") or []}),
            "causal_predecessors": sorted({str(value) for row in ordered
                                           for value in row.get("causal_predecessors") or []}),
            "dialogue": sorted(dialogue, key=lambda line: float(line.get("start_s", 0))),
            "focus_x": sum(float(row.get("focus_x", 0.5)) for row in ordered)
            / len(ordered),
            "semantic_score": max(float(row.get("semantic_score", 0)) for row in ordered),
        })
        merged.append(first)
    return merged


def build_story_plan_from_index(cfg, narrative: dict, *, theme: str, library: str,
                                target_duration_s: float = 60.0,
                                top_k: int = 12) -> tuple[dict, list[list[dict]]]:
    """Semantic shortlist per arc slot, followed by a globally coherent path."""
    errors = validate_narrative_program(narrative)
    if errors:
        raise ValueError("invalid Narrative Program: " + "; ".join(errors))
    from src.library.build_index import E5Embedder, load_index

    narrative, added_roles = _expand_thin_arc(narrative)
    rows, embeddings = load_index(cfg)
    scoped = [(idx, row) for idx, row in enumerate(rows)
              if row.get("source") == library
              or str(row.get("video_stem") or "").startswith(f"{library}__")]
    if not scoped:
        raise ValueError(f"index has no rows for source {library!r}")
    # 片头/片尾职员表区不进叙事检索池（2026-09-10 首跑选中 ED 段的教训）
    excluded = zones.excluded_row_indices([row for _idx, row in scoped], cfg)
    allowed = [pair for pair_idx, pair in enumerate(scoped) if pair_idx not in excluded]
    if not allowed:
        raise ValueError(f"source {library!r} has no rows outside the credits zone")
    arc = narrative.get("arc") or []
    queries = [_arc_query(narrative, segment, theme) for segment in arc]
    query_embeddings = E5Embedder(cfg.library.get("embed") or {}).embed(queries)
    candidate_groups = []
    compatible_roles = {
        "hook": {"hook", "context", "conflict"},
        "context": {"context", "hook"},
        "conflict": {"conflict"},
        "choice": {"choice", "conflict", "climax"},
        "climax": {"climax", "conflict"},
        "consequence": {"consequence", "resolution", "context"},
        "resolution": {"resolution", "consequence"},
    }
    min_score = float((cfg.library.get("retrieve") or {}).get("min_cosine", 0.18))
    for segment, query, query_embedding in zip(arc, queries, query_embeddings):
        cosine = (embeddings @ query_embedding.reshape(-1)).ravel()
        group = []
        for row_idx, row in allowed:
            if row.get("story_role") not in compatible_roles[segment["role"]]:
                continue
            role_bonus = 0.15 if row.get("story_role") == segment["role"] else 0.0
            dialogue_bonus = min(0.08, 0.02 * len(row.get("dialogue") or []))
            score = float(cosine[row_idx]) + role_bonus + dialogue_bonus
            group.append({
                **row, "row_idx": row_idx, "semantic_score": round(score, 6),
                "source_start_s": float(row.get("start_s") or row.get("source_start_s") or 0),
                "source_end_s": float(row.get("end_s") or row.get("source_end_s") or 0),
                "query": query,
            })
        group = merge_event_candidates(group)
        for row in group:
            duration_fit = min(1.0, float(row.get("duration_s", 0))
                               / max(1.0, target_duration_s / max(1, len(arc))))
            row["semantic_score"] = round(float(row["semantic_score"])
                                           + 0.08 * duration_fit, 6)
        group.sort(key=lambda row: (-row["semantic_score"], row["row_idx"]))
        candidate_groups.append([row for row in group
                                 if row["semantic_score"] >= min_score][:top_k])
    plan = _assemble_story_plan(narrative, candidate_groups, theme=theme, library=library,
                                target_duration_s=target_duration_s)
    if excluded:
        plan["zone_filter"] = {"excluded_rows": len(excluded),
                               "config": zones.zone_config(cfg)}
    if added_roles:
        plan["arc_expanded"] = {"reason": "thin_reference_arc", "added_roles": added_roles}
    return plan, candidate_groups


def story_plan_execution_inputs(story_plan: dict, recipe: dict) -> tuple[dict, list[dict]]:
    """Adapt a traceable story path to the existing deterministic renderer contract."""
    errors = validate_story_plan(story_plan)
    if errors:
        raise ValueError("invalid Story Plan: " + "; ".join(errors))
    # Recipe intervals live in the reference video's timebase.  A narrative
    # render normally targets 45--75 seconds, so map those intervals before
    # deciding which edit operations are active in each output slot.  Keep the
    # caller's Recipe untouched; the scaled copy is only an execution view.
    execution_recipe = deepcopy(recipe)
    reference_duration = float((recipe.get("reference") or {}).get("duration_s") or 0)
    target_duration = float(story_plan.get("target_duration_s") or 0)
    if reference_duration > 0 and target_duration > 0 \
            and abs(reference_duration - target_duration) > 1e-6:
        ratio = target_duration / reference_duration
        execution_recipe.setdefault("reference", {})["duration_s"] = target_duration
        for operation in execution_recipe.get("operations") or []:
            interval = operation.get("interval")
            if isinstance(interval, list) and len(interval) == 2:
                operation["interval"] = [round(float(value) * ratio, 6)
                                           for value in interval]
    # 叙事重剪不复用参考片的文字层操作：text 描述的是参考视频自己的字幕/贴纸
    # （如"黄色的 NANCHANG 文字"），烧到新素材上是张冠李戴（2026-09-10 v5 帧验：
    # 该行中文出现在鬼灭画面上方）。编辑模式模仿参考剪辑程序时保留，此处剔除。
    execution_recipe["operations"] = [
        op for op in execution_recipe.get("operations") or []
        if op.get("type") not in {"text_overlay", "text_layer_animation"}]

    slots, retrieval = [], []
    for item in story_plan["slots"]:
        start, end = map(float, item["target_interval"])
        active = [op for op in execution_recipe.get("operations") or []
                  if float(op["interval"][0]) <= end and float(op["interval"][1]) >= start]
        source = item["source"]
        slots.append({
            "slot_idx": item["slot_idx"], "start_s": start, "end_s": end,
            "need_duration_s": end - start, "theme": story_plan["theme"],
            "story_role": item["role"], "event_id": source.get("event_id"),
            "operation_types": sorted({op["type"] for op in active}),
            "query": source.get("caption") or item["role"],
        })
        picked = None
        if item["status"] == "supported":
            picked = {
                "row_idx": item["slot_idx"], "video": source["video"],
                "video_stem": source.get("video_stem") or Path(source["video"]).stem,
                "shot_idx": source.get("shot_idx"),
                "shot_indices": source.get("shot_indices") or [],
                "source_start_s": float(source["start_s"]),
                "source_end_s": float(source["end_s"]),
                "duration_s": float(source["end_s"]) - float(source["start_s"]),
                "caption": source.get("caption") or "",
                "entity_ids": source.get("entity_ids") or [],
                "event_id": source.get("event_id"),
                "causal_predecessors": source.get("causal_predecessors") or [],
                "dialogue": source.get("dialogue") or [],
                "focus_x": float(source.get("focus_x", 0.5)),
            }
        retrieval.append({
            "slot_idx": item["slot_idx"], "query": slots[-1]["query"],
            "picked": picked, "missing": item.get("reason") or "", "candidates": [],
        })
    asset_plan = {
        "plan_version": "narrative-1.0", "theme": story_plan["theme"],
        "library": story_plan["library"], "reference_id": story_plan["reference_id"],
        "narrative_program_required": True, "slots": slots,
    }
    return asset_plan, retrieval


def write_story_plan(plan: dict, path: Path) -> Path:
    errors = validate_story_plan(plan)
    if errors:
        raise ValueError("invalid Story Plan: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_story_plan(plan: dict) -> list[str]:
    if not isinstance(plan, dict):
        return ["story plan must be an object"]
    errors = []
    if plan.get("story_plan_version") != STORY_PLAN_VERSION:
        errors.append(f"story_plan_version must be {STORY_PLAN_VERSION}")
    if not str(plan.get("theme") or "").strip():
        errors.append("theme missing")
    if not str(plan.get("library") or "").strip():
        errors.append("library missing")
    duration = float(plan.get("target_duration_s") or 0)
    if not 45 <= duration <= 75:
        errors.append("target_duration_s outside 45..75")
    slots = plan.get("slots")
    if not isinstance(slots, list) or not slots:
        return errors + ["slots must be non-empty"]
    last_end = 0.0
    role_order = {role: idx for idx, role in enumerate(ARC_ROLES)}
    last_role = -1
    for idx, slot in enumerate(slots):
        prefix = f"slots[{idx}]"
        if not isinstance(slot, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if slot.get("slot_idx") != idx:
            errors.append(f"{prefix}.slot_idx invalid")
        if slot.get("role") not in ARC_ROLES:
            errors.append(f"{prefix}.role invalid")
        else:
            current_role = role_order[slot["role"]]
            if current_role < last_role:
                errors.append(f"{prefix}.role is out of narrative order")
            last_role = current_role
        interval = slot.get("target_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            errors.append(f"{prefix}.target_interval invalid")
        else:
            start, end = map(float, interval)
            if abs(start - last_end) > 1e-5 or end <= start or end > duration + 1e-6:
                errors.append(f"{prefix}.target_interval not continuous")
            last_end = end
        if slot.get("status") not in {"supported", "uncertain", "unsupported"}:
            errors.append(f"{prefix}.status invalid")
        source = slot.get("source")
        if not isinstance(source, dict):
            errors.append(f"{prefix}.source must be an object")
            continue
        if slot.get("status") == "supported":
            if not str(source.get("video") or "").strip():
                errors.append(f"{prefix}.source.video missing")
            if not str(source.get("event_id") or "").strip():
                errors.append(f"{prefix}.source.event_id missing")
            if float(source.get("end_s") or 0) <= float(source.get("start_s") or 0):
                errors.append(f"{prefix}.source interval invalid")
            if slot.get("transition_reason") == "unexplained":
                errors.append(f"{prefix} has unexplained entity switch")
    if abs(last_end - duration) > 1e-5:
        errors.append("slots do not cover target duration")
    return errors
