"""Traceable, continuity-aware story planning over narrative index rows."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from src.agentic_video.narrative import ARC_ROLES, validate_narrative_program
from src.agentic_video.narrative_form import (ROLE_ENTITY_REQUIREMENTS,
                                              compile_form_need,
                                              infer_narrative_form,
                                              protagonist_required,
                                              resolve_slot_sequence)
from src.agentic_video import zones
from src.library.entity_registry import load_entity_registry, row_identity_keys

STORY_PLAN_VERSION = "1.0"   # 结构兼容旧计划（新字段 entity_contract 等可选）
# 目标时长界（2026-09-10 放宽下界 45→20）：7682 型情感叙事模板天然 20-35s，
# 旧 45..75 把整类模板挡在门外。上界 75 保持不动（全部现有测试时长都在区间内）。
STORY_MIN_TARGET_DURATION_S = 20.0
STORY_MAX_TARGET_DURATION_S = 75.0
# 必选弧角色（厚参考弧路径用；薄弧路径的 required 由 Narrative Form 模板提供）
REQUIRED_ROLES = frozenset({"hook", "conflict", "climax", "resolution"})

# 角色级证据规格模板：need 是观众必须获得的信息，must_have/must_not 是
# 可核验的证据约束——检索和 P3 验证器都以这里为准，角色标签本身不是需求。
_ROLE_NEED_TEMPLATES = {
    "hook": {
        "need": "开场呈现主角的处境或未被解决的表达张力，让观众想看后续",
        "must_have": ["主角在场的可见处境"],
        "must_not": ["与主角无关的纯环境空镜"],
        "evidence_mode": "visual",
    },
    "context": {
        "need": "让观众理解主角与关键他人/环境的关系背景",
        "must_have": ["主角与至少一个关键对象同框或明确指向"],
        "must_not": ["完全不含主角的镜头"],
        "evidence_mode": "visual",
    },
    "conflict": {
        "need": "呈现主角面对的问题具象化为可见的冲突或危险",
        "must_have": ["冲突/危险可见", "主角在场"],
        "must_not": ["主角不在场的纯环境镜头"],
        "evidence_mode": "visual",
    },
    "choice": {
        "need": "呈现主角做出关键选择或行动决断的可见过程",
        "must_have": ["主角的行动可辨认"],
        "must_not": ["只有他人的行动"],
        "evidence_mode": "visual",
    },
    "climax": {
        "need": "呈现主角的关键行动与情绪峰值",
        "must_have": ["高潮动作或情绪峰值画面", "主角在场"],
        "must_not": ["平淡过场"],
        "evidence_mode": "visual",
    },
    "consequence": {
        "need": "呈现行动带来的直接后果状态",
        "must_have": ["结果状态可辨认（伤势/局势/他人反应等）"],
        "must_not": ["与前面行动无关的新场景"],
        "evidence_mode": "both",
    },
    "resolution": {
        "need": "呈现情绪收束与最终状态",
        "must_have": ["收束性画面（和解/启程/定格等）"],
        "must_not": ["悬而未决的新冲突"],
        "evidence_mode": "both",
    },
}


def _usable(value: str) -> str | None:
    """六问字段可用性：unknown/not_applicable/空 都不是可引用的内容。"""
    text = str(value or "").strip()
    return text if text and text not in {"unknown", "not_applicable", "uncertain"} else None


def slot_need_spec(narrative: dict, segment: dict, idx: int) -> dict:
    """把槽位（参考弧段或 Form 模板槽）编译成证据规格（确定性）。

    V3 P1：need 优先按 form_function 编译（narrative_form.compile_form_need，
    零参考事实）；无 form_function 的厚弧段回退角色模板。红线不变：六问的
    protagonist/problem 等是参考片事实，绝不灌进 need（C2 验证器实锤的
    「找穿跆拳道服的角色」跨库荒谬）。entity_bindings 仍从事件参与实体推导，
    库侧匹配靠 entity_names 语义；entity_requirements 声明主角在场要求——
    Entity Continuity Contract 的硬约束来源。
    """
    role = segment["role"]
    form_need = compile_form_need(str(segment.get("form_function") or ""))
    template = dict(_ROLE_NEED_TEMPLATES[role])
    need = form_need.get("need") or template["need"]
    must_have = form_need.get("must_have") or template["must_have"]
    must_not = form_need.get("must_not") or template["must_not"]
    evidence_mode = form_need.get("evidence_mode") or template["evidence_mode"]
    entity_by_id = {row["id"]: row for row in narrative.get("entities") or []}
    bindings = {}
    for event_id in segment.get("event_ids") or []:
        event = next((row for row in narrative.get("events") or []
                      if row["id"] == event_id), None) or {}
        for entity_id in event.get("participants") or []:
            entity = entity_by_id.get(entity_id) or {}
            if entity_id in {v["reference_entity_id"] for v in bindings.values()}:
                continue
            key = "A" if "A" not in bindings else "B" if "B" not in bindings else None
            if key is None:                                    # 首版只绑 A/B 双主角
                break
            bindings[key] = {
                "reference_entity_id": entity_id,
                "reference_name": str(entity.get("name_or_role") or entity_id),
                "library_hint": str(entity.get("name_or_role") or "") or None,
            }
    spec = {
        "function": str(segment.get("function")
                        or segment.get("form_function")
                        or f"{role} 段在整条表达中的作用"),
        "need": need,
        "entity_bindings": bindings,
        "must_have": must_have,
        "must_not": must_not,
        "evidence_mode": evidence_mode,
        "required": bool(segment.get("required", role in REQUIRED_ROLES)),
        "entity_requirements": dict(segment.get("entity_requirements")
                                    or ROLE_ENTITY_REQUIREMENTS.get(role) or {}),
        "form_function": str(segment.get("form_function") or ""),
        "depends_on": None,
    }
    return spec


def _link_depends_on(specs: list[dict]) -> None:
    """相邻槽共享实体绑定代号 → 后槽 depends_on 前槽（路径硬连续约束）。"""
    for idx in range(1, len(specs)):
        previous_keys = set(specs[idx - 1].get("entity_bindings") or {})
        current_keys = set(specs[idx].get("entity_bindings") or {})
        if previous_keys & current_keys:
            specs[idx]["depends_on"] = idx - 1


def _hard_continuity(specs: list[dict]) -> list[bool]:
    """edge[i] = 组 i 与组 i-1 之间是否要求实体连续（可行性约束，非加分）。"""
    return [idx > 0 and specs[idx].get("depends_on") == idx - 1
            for idx in range(len(specs))]


def _build_specs(narrative: dict) -> list[dict]:
    """槽序列经 resolve_slot_sequence 解析（厚弧=参考弧段；薄弧=Form 模板），
    再逐槽编译证据规格。V3 P1 删除了 _expand_thin_arc 自动补弧——薄参考弧
    （hook/consequence）被强补 conflict 是 lxh_p4_C2 选错故事语法的直接原因；
    Form 模板按参考片实际表达结构给声明式槽序列，绝不合成不存在的冲突。"""
    slots, _form_name = resolve_slot_sequence(narrative)
    specs = [slot_need_spec(narrative, slot, idx)
             for idx, slot in enumerate(slots)]
    _link_depends_on(specs)
    return specs


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
    elif left.get("event_id") in set(right.get("preceding_event_ids") or []):
        # V3 P2：库内因果多为空（lxh_p4_C2 36 候选 0 非空）——同窗时序前驱是
        # 不造假的事件序列信号，替代虚构的因果图加分。
        score += 0.10
    if right.get("event_id") in set(left.get("causal_predecessors") or []):
        score -= 0.35
    if left.get("row_idx") == right.get("row_idx"):
        score -= 0.30
    return score


def rank_story_path(candidate_groups: list[list[dict]],
                    hard_continuity: list[bool] | None = None,
                    allowed: list[set[int]] | None = None) -> list[dict | None]:
    """Viterbi-like path ranking with explicit entity/time continuity bonuses.

    hard_continuity[i] 为真时，组 i-1 → i 的转移要求两组实体集有交集——这是
    可行性约束（非法转移不进图），不是加分项；某组全部候选都不可达时该槽
    返回 None（由上层按必选/可选决定 unsupported 或跳过），链条在该槽重启：
    重启组的候选以链头身份入图（parent=None），不再受与前组的连续约束。

    allowed[i]（V3 P3 Entity Continuity Contract）：主角必需槽只允许含锁定
    主角的候选——**身份约束先于相似度**（违反 = 候选不存在，不是 -0.25 加权，
    embedding 再高也赢不过换主角）。None = 无契约约束。
    """
    if not candidate_groups or any(not group for group in candidate_groups):
        return []
    hard = list(hard_continuity or [False] * len(candidate_groups))

    def _self_score(row: dict) -> float:
        return float(row.get("semantic_score", row.get("score", 0)))

    def _permitted(group_idx: int, row_idx: int) -> bool:
        return allowed is None or row_idx in allowed[group_idx]

    scores: list[dict[int, float]] = [{idx: _self_score(row)
                                       for idx, row in enumerate(candidate_groups[0])
                                       if _permitted(0, idx)}]
    # parents[g][idx] = 前驱下标；None = 链头（首组或断链重启组）
    parents: list[dict[int, int | None]] = [{idx: None for idx in scores[0]}]
    for group_idx in range(1, len(candidate_groups)):
        group, previous = candidate_groups[group_idx], candidate_groups[group_idx - 1]
        previous_alive = bool(scores[-1])                     # 前组是否断链
        current_scores: dict[int, float] = {}
        current_parents: dict[int, int | None] = {}
        for idx, row in enumerate(group):
            if not _permitted(group_idx, idx):
                continue                                       # 契约禁入：边不存在
            if previous_alive:
                options: list[tuple[float, int]] = []
                for prev_idx, prev_score in scores[-1].items():
                    prev = previous[prev_idx]
                    if hard[group_idx] and not (set(prev.get("entity_ids") or [])
                                                & set(row.get("entity_ids") or [])):
                        continue                               # 非法转移：不进图
                    options.append((prev_score + _transition_score(prev, row), prev_idx))
                if not options:
                    continue                                   # 该候选不可达
            else:
                options = [(0.0, -1)]                          # 重启：链头身份
            best_score, best_parent = max(options, key=lambda item: (item[0], -item[1]))
            current_scores[idx] = best_score + _self_score(row)
            current_parents[idx] = best_parent if best_parent >= 0 else None
        scores.append(current_scores)
        parents.append(current_parents)
    path: list[dict | None] = [None] * len(candidate_groups)
    cursor_group = len(candidate_groups) - 1
    while cursor_group >= 0 and not scores[cursor_group]:
        cursor_group -= 1                                     # 断链组：保持 None
    if cursor_group < 0:
        return path
    cursor = max(scores[cursor_group], key=lambda idx: (scores[cursor_group][idx], -idx))
    while cursor_group >= 0:
        path[cursor_group] = deepcopy(candidate_groups[cursor_group][cursor])
        parent = parents[cursor_group].get(cursor)
        if parent is None:
            # 链头：跳过紧邻的断链组，向前找上一段可达链
            cursor_group -= 1
            while cursor_group >= 0 and not scores[cursor_group]:
                cursor_group -= 1
            if cursor_group < 0:
                break
            cursor = max(scores[cursor_group],
                         key=lambda idx: (scores[cursor_group][idx], -idx))
            continue
        cursor = parent
        cursor_group -= 1
    return path


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


def build_story_plan(narrative: dict, source_rows: list[dict], *, theme: str,
                     library: str, target_duration_s: float = 60.0) -> dict:
    errors = validate_narrative_program(narrative)
    if errors:
        raise ValueError("invalid Narrative Program: " + "; ".join(errors))
    if not STORY_MIN_TARGET_DURATION_S <= target_duration_s <= STORY_MAX_TARGET_DURATION_S:
        raise ValueError(f"target_duration_s must be within "
                         f"{STORY_MIN_TARGET_DURATION_S:g}..{STORY_MAX_TARGET_DURATION_S:g}")
    slots, _form_name = resolve_slot_sequence(narrative)
    candidate_groups = [_candidates_for_arc(slot, source_rows) for slot in slots]
    return _assemble_story_plan(narrative, candidate_groups, theme=theme,
                                library=library, target_duration_s=target_duration_s,
                                cfg=None)


def _rank_under_contract(specs: list[dict], candidate_groups: list[list[dict]],
                         hard: list[bool], registry: dict
                         ) -> tuple[list[dict | None], dict]:
    """Entity Continuity Contract 实例化（V3 P1/P3，外审二轮核心）：

    首个 protagonist-required 槽的每个候选实体身份各成一个"主角假设"；对每个
    假设跑带 allowed 硬过滤的路径排序（主角槽候选不含假设实体 = 边不存在），
    取（覆盖槽数, 总分）最优。主角身份由**第一次选材**锁定，之后每个主角槽
    只在该实体的候选内排序——semantic similarity 永远赢不过 identity
    constraint（lxh_p4_C2：三槽三个主角都各自 PASS 的病灶）。主角持续性
    relaxed=False：锁不住就少槽/unsupported，绝不换人顶班。
    """
    contract = {
        "protagonist": None, "protagonist_name": None, "persistence": "global",
        "locked_slots": [idx for idx, spec in enumerate(specs)
                         if protagonist_required(spec)],
        "free_slots": [idx for idx, spec in enumerate(specs)
                       if not protagonist_required(spec)],
        "relaxed": False,
    }
    identities = [[row_identity_keys(row, registry) for row in group]
                  for group in candidate_groups]
    first_req = next((idx for idx, spec in enumerate(specs)
                      if protagonist_required(spec) and candidate_groups[idx]), None)
    if first_req is None:
        path = _rank_runs(candidate_groups, hard)
        return path, contract
    hypotheses: list[tuple[str, str]] = []      # (身份键, 展示名)
    for row, keys in zip(candidate_groups[first_req], identities[first_req]):
        for key in sorted(keys):
            if key not in {h[0] for h in hypotheses}:
                name = _identity_display_name(key, row)
                hypotheses.append((key, name))
    if not hypotheses:
        path = _rank_runs(candidate_groups, hard)      # 无实体信息：如实不锁
        return path, contract
    best_key, best = None, None
    for hypothesis, display in hypotheses:
        allowed = []
        for idx, group in enumerate(candidate_groups):
            if protagonist_required(specs[idx]):
                allowed.append({j for j in range(len(group))
                                if hypothesis in identities[idx][j]})
            else:
                allowed.append(set(range(len(group))))
        path = _rank_runs(candidate_groups, hard, allowed=allowed)
        covered = sum(1 for row in path if row is not None)
        score = sum(float((row or {}).get("semantic_score", 0)) for row in path)
        ranking = (covered, round(score, 6))
        if best_key is None or ranking > best_key:
            best_key, best = ranking, (path, hypothesis, display)
    path, hypothesis, display = best
    contract["protagonist"] = hypothesis
    contract["protagonist_name"] = display
    return path, contract


def _identity_display_name(key: str, row: dict) -> str:
    names = [str(name) for name in (row.get("entity_names") or []) if name]
    return names[0] if names else key


def _rank_runs(candidate_groups: list[list[dict]], hard: list[bool],
               allowed: list[set[int]] | None = None) -> list[dict | None]:
    """非空候选段各自排序，空组保持 None（断链槽由上层定 unsupported/跳过）。"""
    path: list[dict | None] = [None] * len(candidate_groups)
    run_start = 0
    while run_start < len(candidate_groups):
        if not candidate_groups[run_start]:
            run_start += 1
            continue
        run_end = run_start
        while run_end < len(candidate_groups) and candidate_groups[run_end]:
            run_end += 1
        path[run_start:run_end] = rank_story_path(
            candidate_groups[run_start:run_end],
            hard_continuity=hard[run_start:run_end],
            allowed=(allowed[run_start:run_end] if allowed is not None else None))
        run_start = run_end
    return path


def _assemble_story_plan(narrative: dict, candidate_groups: list[list[dict]], *,
                         theme: str, library: str, target_duration_s: float,
                         cfg=None) -> dict:
    resolved, form_name = resolve_slot_sequence(narrative)
    specs = _build_specs(narrative)
    hard = _hard_continuity(specs)
    registry = load_entity_registry(cfg)
    path, contract = _rank_under_contract(specs, candidate_groups, hard, registry)
    # 同源去重二 pass（2026-09-10 C3 核验：Viterbi 的 -0.30 同行惩罚只作用于相邻
    # 槽，7682 弧的 hook/conflict 引用同一旁白事件时隔槽撞段拦不住——短成片里
    # 重复素材会直接复发"零剪辑"感）。重复时换组内次优未用候选，无替代保留并
    # 标记 dedup_conflict 供报告溯源。
    used_rows: set = set()
    for idx, picked in enumerate(path):
        if not picked or picked.get("row_idx") is None:
            continue
        if picked.get("row_idx") in used_rows:
            def _contract_ok(row: dict) -> bool:
                # 去重换件同样受主角契约约束（V3：换候选不得顺手换主角）
                if not (contract.get("protagonist")
                        and protagonist_required(specs[idx])):
                    return True
                return contract["protagonist"] in row_identity_keys(row, registry)
            replacement = next(
                (row for row in candidate_groups[idx]
                 if row.get("row_idx") is not None
                 and row.get("row_idx") not in used_rows
                 and _contract_ok(row)), None)
            if replacement is not None:
                path[idx] = replacement
        used_rows.add(path[idx].get("row_idx"))
    row_use_counts: dict = {}
    for picked in path:
        key = (picked or {}).get("row_idx")
        if key is not None:
            row_use_counts[key] = row_use_counts.get(key, 0) + 1
    slot_duration = target_duration_s / max(1, len(resolved))
    slots = []
    for idx, segment in enumerate(resolved):
        start = round(idx * slot_duration, 6)
        end = round(target_duration_s if idx == len(resolved) - 1
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
        evidence_caption = None
        evidence_shots = None
        if picked:
            evidence = _select_evidence_window(
                picked.get("member_shots") or [], slot_budget,
                "；".join(filter(None, [
                    specs[idx].get("need") or "",
                    *(specs[idx].get("must_have") or []),
                    str(picked.get("query") or ""), str(picked.get("caption") or "")])))
            if evidence is not None:
                if (abs(evidence["start_s"] - source_start) > 1e-6
                        or evidence["end_s"] < source_end - 1e-6):
                    source_trimmed = True
                source_start, source_end = evidence["start_s"], evidence["end_s"]
                if source_end - source_start > slot_budget + 1e-6:
                    source_end = source_start + slot_budget   # 单镜头超预算兜底截断
                evidence_caption = evidence["caption"] or None
                evidence_shots = evidence.get("shot_indices")
            elif source_end - source_start > slot_budget + 1e-6:
                source_end = source_start + slot_budget
                source_trimmed = True
        if picked:
            dialogue = [line for line in dialogue
                        if float(line.get("start_s", source_start)) >= source_start - 1e-6
                        and float(line.get("end_s", source_end)) <= source_end + 1e-6]
        source = {
            "video": str((picked or {}).get("video") or ""),
            "video_stem": str((picked or {}).get("video_stem") or ""),
            "window_idx": (picked or {}).get("window_idx"),
            "shot_idx": (picked or {}).get("shot_idx"),
            "shot_indices": evidence_shots or list((picked or {}).get("shot_indices") or []),
            "start_s": source_start, "end_s": source_end,
            "event_id": str((picked or {}).get("event_id") or ""),
            "causal_predecessors": list((picked or {}).get("causal_predecessors") or []),
            "caption": str(evidence_caption or (picked or {}).get("caption") or ""),
            "entity_ids": list((picked or {}).get("entity_ids") or []),
            "entity_names": list((picked or {}).get("entity_names") or []),
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
        if not picked and not specs[idx]["required"]:
            reason = "optional_slot_no_evidence"           # 可选槽：记录后跳过
        slots.append({
            "slot_idx": idx, "role": segment["role"],
            "target_interval": [start, end],
            "reference_event_ids": list(segment.get("event_ids") or []),
            "source": source,
            "status": slot_status,
            "reason": reason,
            "transition_reason": transition_reason,
            "source_interval_trimmed": source_trimmed,
            "dedup_conflict": bool(picked and row_use_counts.get(picked.get("row_idx"), 0) > 1),
            "need_spec": specs[idx],
        })
    # V1 P2 红线：必选槽 unsupported = 任务未完成，不作为成功成片交付；
    # 缺口如实上报（P3 重搜会尝试补，补不上就以未达标收尾）。
    required_unsupported = [slot["slot_idx"] for slot in slots
                            if slot["status"] == "unsupported"
                            and slot["need_spec"]["required"]]
    return {
        "story_plan_version": STORY_PLAN_VERSION, "theme": theme.strip(),
        "library": library, "target_duration_s": round(target_duration_s, 6),
        "reference_id": narrative["reference"]["id"], "slots": slots,
        "required_unsupported": required_unsupported,
        "plan_complete": not required_unsupported,
        # V3 P1：叙事形态（表达结构模板）与人物连续性契约进计划——
        # 谁在演主角、哪些槽锁谁，是成片叙事成立的第一约束。
        "narrative_form": {"name": form_name,
                           "slot_functions": [slot.get("form_function")
                                              for slot in resolved]},
        "entity_contract": contract,
        # 情绪峰值提示（参考弧 climax 段在参考时长中的比例）：P2 起透传给
        # copywriter 对齐卡点连发，emotion_curve 首次有了消费者。
        "emotion_peak_hint": _emotion_peak_hint(narrative),
    }


def _emotion_peak_hint(narrative: dict) -> dict | None:
    curve = narrative.get("emotion_curve") or []
    climax = next((row for row in narrative.get("arc") or []
                   if row.get("role") == "climax"), None)
    if climax:
        events = [row for row in narrative.get("events") or []
                  if row["id"] in set(climax.get("event_ids") or [])]
        if events:
            reference = narrative.get("reference") or {}
            duration = float(reference.get("duration_s") or 0)
            mid = sum(float(row["interval"][0]) for row in events) / len(events)
            if duration > 0:
                return {"source": "climax", "peak_ratio": round(mid / duration, 3)}
    if curve:
        peak = max(curve, key=lambda row: float(row.get("intensity", 0)))
        reference = narrative.get("reference") or {}
        duration = float(reference.get("duration_s") or 0)
        if duration > 0:
            mid = (float(peak["interval"][0]) + float(peak["interval"][1])) / 2
            return {"source": "emotion_curve", "peak_ratio": round(mid / duration, 3)}
    return None


def _arc_query(narrative: dict, segment: dict, theme: str,
               spec: dict | None = None) -> str:
    """检索 query v3：叙事需求（观众要什么信息）优先于角色标签——裸角色匹配
    只会召回"最燃镜头"而不是支持故事的证据。must_have/must_not 文本同入查询，
    P3 验证器再逐条核证据（检索召回 + 事后验证双层把关）。"""
    event_by_id = {event["id"]: event for event in narrative.get("events") or []}
    parts = [theme]
    if spec:
        parts.append(f"叙事需求：{spec.get('need') or ''}")
        parts.extend(f"需要：{item}" for item in spec.get("must_have") or [])
        parts.extend(f"排除：{item}" for item in spec.get("must_not") or [])
        hints = [binding.get("library_hint") for binding
                 in (spec.get("entity_bindings") or {}).values()
                 if binding.get("library_hint")]
        if hints:
            parts.append("人物：" + "、".join(hints[:3]))
    parts.append(f"叙事角色：{segment['role']}")
    for event_id in segment.get("event_ids") or []:
        event = event_by_id.get(event_id) or {}
        parts.extend([str(event.get("action") or ""), str(event.get("state_before") or ""),
                      str(event.get("state_after") or "")])
    # 过滤噪声 token：英文 uncertain + 中文「不确定」（2026-09-10 C4 核验：
    # 7682 程序事件的 state 文本是中文「不确定」，直接拼进 query 污染检索语义）
    return "；".join(part for part in parts
                     if part and part not in {"uncertain", "不确定"})


def _text_shingles(text: str) -> set[str]:
    """查询重合用的粗粒度 token：中文二元组 + 英文词。"""
    value = str(text or "")
    tokens: set[str] = set()
    ascii_run: list[str] = []
    cjk = [char for char in value if "一" <= char <= "鿿"]
    for a, b in zip(cjk, cjk[1:]):
        tokens.add(a + b)
    for char in value:
        if char.isascii() and char.isalnum():
            ascii_run.append(char.lower())
        elif ascii_run:
            tokens.add("".join(ascii_run))
            ascii_run = []
    if ascii_run:
        tokens.add("".join(ascii_run))
    return tokens


def _select_evidence_window(member_shots: list[dict], budget_s: float,
                            query_text: str) -> dict | None:
    """预算内覆盖 required evidence 的连续镜头子序列（V3 P2）。

    评分 = 窗口内镜头 caption 与槽需求（need/must_have/查询）的平均重合度，
    并列取覆盖更长者；证据在后段时窗口跟着证据走——不再"从事件起点盲切
    7.3s"（lxh_p4_C2：caption 说奔跑、切出的是开头的施法静止段）。
    单镜头超预算仍保底返回，由上层截断。
    """
    shots = [shot for shot in (member_shots or [])
             if float(shot.get("end_s") or 0) > float(shot.get("start_s") or 0)]
    if not shots:
        return None
    query = _text_shingles(query_text)
    if not query:
        return None
    best: dict | None = None
    for start_idx in range(len(shots)):
        picked_shots, total = [], 0.0
        for end_idx in range(start_idx, len(shots)):
            shot = shots[end_idx]
            duration = float(shot["end_s"]) - float(shot["start_s"])
            if picked_shots and total + duration > budget_s + 1e-6:
                break
            picked_shots.append(shot)
            total += duration
            overlap = sum(len(_text_shingles(item.get("caption") or "") & query)
                          for item in picked_shots) / len(picked_shots)
            ranking = (round(overlap, 6), round(min(total, budget_s), 3))
            if best is None or ranking > best["ranking"]:
                best = {"ranking": ranking, "shots": list(picked_shots)}
    if best is None:
        return None
    chosen = best["shots"]
    return {
        "start_s": float(chosen[0]["start_s"]),
        "end_s": float(chosen[-1]["end_s"]),
        "caption": "；".join(str(shot.get("caption") or "") for shot in chosen
                            if str(shot.get("caption") or "").strip()),
        "shot_indices": [shot.get("shot_idx") for shot in chosen],
    }


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
            "entity_names": sorted({str(value) for row in ordered
                                    for value in row.get("entity_names") or []}),
            "causal_predecessors": sorted({str(value) for row in ordered
                                           for value in row.get("causal_predecessors") or []}),
            "dialogue": sorted(dialogue, key=lambda line: float(line.get("start_s", 0))),
            "focus_x": sum(float(row.get("focus_x", 0.5)) for row in ordered)
            / len(ordered),
            "semantic_score": max(float(row.get("semantic_score", 0)) for row in ordered),
            # V3 P2：成员镜头明细——事件聚合 caption 描述整个事件，但槽预算只装
            # 得下其中一段；裁剪按"预算内覆盖 required evidence 的连续镜头子序列"
            # 选，caption 只保留被裁入镜头（lxh_p4_C2 断言：caption 说奔跑、
            # 画面切出施法的"caption 对素材错"病灶）。
            "member_shots": [
                {"shot_idx": row.get("shot_idx"),
                 "start_s": float(row.get("source_start_s", 0) or 0),
                 "end_s": float(row.get("source_end_s", 0) or 0),
                 "caption": str(row.get("caption") or ""),
                 "entity_ids": list(row.get("entity_ids") or []),
                 "story_role": row.get("story_role")}
                for row in ordered],
            "preceding_event_ids": [],
        })
        merged.append(first)
    # 同窗事件时序前驱（V3 P2）：库侧 causal_links 普遍为空，"事件图"实际退化
    # 成 embedding+role——前驱关系如实记录窗内顺序，不造假因果。
    by_window: dict[tuple, list[dict]] = {}
    for event in merged:
        by_window.setdefault((str(event.get("video") or ""),
                              event.get("window_idx")), []).append(event)
    for _key, events in by_window.items():
        events.sort(key=lambda row: float(row.get("source_start_s", 0)))
        for previous, current in zip(events, events[1:]):
            if float(previous.get("source_end_s", 0)) <= float(current.get("source_start_s", 0)) + 1e-6:
                current["preceding_event_ids"] = [str(previous.get("event_id") or "")]
    return merged


COMPATIBLE_ROLES = {
    "hook": {"hook", "context", "conflict"},
    "context": {"context", "hook"},
    "conflict": {"conflict"},
    "choice": {"choice", "conflict", "climax"},
    "climax": {"climax", "conflict"},
    "consequence": {"consequence", "resolution", "context"},
    "resolution": {"resolution", "consequence"},
}


def _library_scope(library: str) -> tuple[set[str], set[str]]:
    """素材池解析：支持逗号分隔多源（罗小黑 = luoxiaohei1,luoxiaohei2 两部电影
    共用一个池）。返回 (源名集合, video_stem 前缀集合)。"""
    names = {part.strip() for part in str(library).split(",") if part.strip()}
    return names, {f"{name}__" for name in names}


def _row_in_library(row: dict, library: str) -> bool:
    names, prefixes = _library_scope(library)
    return str(row.get("source") or "") in names \
        or any(str(row.get("video_stem") or "").startswith(prefix)
               for prefix in prefixes)


def score_slot_candidates(cfg, rows, embeddings, *, query, query_embedding, role,
                          library, slot_budget_s, top_k=12,
                          used_rows=None) -> list[dict]:
    """单槽候选打分（从 build_story_plan_from_index 抽出，初次规划与 P3 重搜共用）。

    返回按 semantic_score 降序的合并事件候选；used_rows 中的行（其它槽已选）
    被排除，重搜不会换汤不换药地撞回同一段素材。
    """
    scoped = [(idx, row) for idx, row in enumerate(rows) if _row_in_library(row, library)]
    excluded = zones.excluded_row_indices([row for _idx, row in scoped], cfg)
    allowed = [pair for pair_idx, pair in enumerate(scoped) if pair_idx not in excluded]
    cosine = (embeddings @ query_embedding.reshape(-1)).ravel()
    compatible = COMPATIBLE_ROLES.get(role, {role})
    min_score = float((cfg.library.get("retrieve") or {}).get("min_cosine", 0.18))
    used = set(used_rows or ())
    group = []
    for row_idx, row in allowed:
        if row.get("story_role") not in compatible or row_idx in used:
            continue
        role_bonus = 0.15 if row.get("story_role") == role else 0.0
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
        duration_fit = min(1.0, float(row.get("duration_s", 0)) / max(1.0, slot_budget_s))
        row["semantic_score"] = round(float(row["semantic_score"]) + 0.08 * duration_fit, 6)
    group.sort(key=lambda row: (-row["semantic_score"], row.get("row_idx", 0)))
    return [row for row in group if row["semantic_score"] >= min_score][:top_k]


def build_story_plan_from_index(cfg, narrative: dict, *, theme: str, library: str,
                                target_duration_s: float = 60.0,
                                top_k: int = 12) -> tuple[dict, list[list[dict]]]:
    """Semantic shortlist per arc slot, followed by a globally coherent path."""
    errors = validate_narrative_program(narrative)
    if errors:
        raise ValueError("invalid Narrative Program: " + "; ".join(errors))
    from src.library.build_index import E5Embedder, load_index

    rows, embeddings = load_index(cfg)
    resolved, _form_name = resolve_slot_sequence(narrative)
    specs = _build_specs(narrative)
    queries = [_arc_query(narrative, segment, theme, spec=spec)
               for segment, spec in zip(resolved, specs)]
    query_embeddings = E5Embedder(cfg.library.get("embed") or {}).embed(queries)
    slot_budget = target_duration_s / max(1, len(resolved))
    candidate_groups = [
        score_slot_candidates(cfg, rows, embeddings, query=query,
                              query_embedding=query_embedding, role=segment["role"],
                              library=library, slot_budget_s=slot_budget, top_k=top_k)
        for segment, query, query_embedding in zip(resolved, queries, query_embeddings)]
    plan = _assemble_story_plan(narrative, candidate_groups, theme=theme, library=library,
                                target_duration_s=target_duration_s, cfg=cfg)
    plan["retrieval_meta"] = {"rows_total": len(rows), "top_k": top_k,
                              "slot_budget_s": round(slot_budget, 3)}
    return plan, candidate_groups


def _overlaps(left: dict, right: dict, *, pad_s: float = 0.0) -> bool:
    """同视频 + 时间区间重叠。兼容两种键名：索引/候选行用 source_start_s，
    槽 source 与已拒绝清单用 start_s。"""
    def _bounds(row: dict) -> tuple[float, float]:
        return (float(row.get("source_start_s", row.get("start_s", 0)) or 0),
                float(row.get("source_end_s", row.get("end_s", 0)) or 0))
    left_start, left_end = _bounds(left)
    right_start, right_end = _bounds(right)
    return (str(left.get("video") or "") == str(right.get("video") or "")
            and min(left_end, right_end) + pad_s > max(left_start, right_start) - pad_s)


def re_search_slot(cfg, story_plan: dict, slot_idx: int, *, theme: str,
                   narrative: dict | None = None, candidate_pool: list[dict] | None = None,
                   need_hint: str | None = None, rejected: list[dict] | None = None,
                   top_k: int = 12) -> dict:
    """分层重搜（V1 P3）：换旧候选不叫重搜。

    层1 = 候选池里未检查且未被其它槽占用的项；层2 = 按 need_hint 改写查询，
    重新编码并对完整现有索引重搜（「复用索引」不等于「零 E5 调用」）。
    已拒绝候选按 源视频+区间重叠 去重，不再换回同一段失败素材。替换后整计划
    必须仍过 validate_story_plan，否则回退并报 failed——绝不放宽约束凑满。
    """
    slot = story_plan["slots"][slot_idx]
    spec = slot.get("need_spec") or {}
    role = slot["role"]
    reason = str(need_hint or spec.get("need") or "slot failed verification")
    others = [other["source"] for idx, other in enumerate(story_plan["slots"])
              if idx != slot_idx and other.get("status") in {"supported", "uncertain"}
              and other.get("source", {}).get("video")]
    rejected = list(rejected or [])
    # 硬连续约束在重搜里同样生效：depends_on 槽的替换件必须与被依赖槽实体相交
    depends_on = spec.get("depends_on")
    anchor_entities = set()
    if depends_on is not None and 0 <= int(depends_on) < len(story_plan["slots"]):
        anchor_entities = set(story_plan["slots"][int(depends_on)]
                              .get("source", {}).get("entity_ids") or [])
    # V3 P3 主角契约在重搜同样生效：protagonist 槽的替换件必须还是锁定的主角
    # （换件不换人）；主角素材枯竭 → 分层搜完仍无 → failed/unsupported，绝不
    # 拿别的角色顶班。
    contract = story_plan.get("entity_contract") or {}
    protagonist = contract.get("protagonist")
    registry = load_entity_registry(cfg)

    def _eligible(row: dict) -> bool:
        if not str(row.get("video") or ""):
            return False
        if anchor_entities and not (anchor_entities & set(row.get("entity_ids") or [])):
            return False
        if protagonist and protagonist_required(spec) \
                and protagonist not in row_identity_keys(row, registry):
            return False
        blocked = others + rejected
        return not any(_overlaps(row, item, pad_s=1.0) for item in blocked)

    attempts = []
    # 层1：候选池（story_candidates.json）里未检查的有效选项
    best_pool = next((row for row in sorted(candidate_pool or [],
                                            key=lambda row: -float(row.get("semantic_score", 0)))
                      if _eligible(row)), None)
    if best_pool is not None:
        attempts.append({"level": 1, "picked": best_pool})
    else:
        # 层2：改写查询全库重搜（查询文本变了要重新编码；「复用索引」≠「零 E5 调用」）。
        # 索引不可用（缺失/损坏）时显式失败——绝不因此放宽约束或崩掉整轮。
        from src.library.build_index import E5Embedder, load_index
        try:
            rows, embeddings = load_index(cfg)
        except Exception:                                     # noqa: BLE001 - 显式失败优于崩溃
            return {"slot_idx": slot_idx, "status": "failed",
                    "reason": "level2 index unavailable",
                    "levels_tried": [1]}
        query = "；".join(part for part in [theme, f"叙事需求：{reason}",
                                            f"叙事角色：{role}"] if part)
        query_embedding = E5Embedder(cfg.library.get("embed") or {}).embed([query])[0]
        budget = float(slot["target_interval"][1]) - float(slot["target_interval"][0])
        fresh = score_slot_candidates(cfg, rows, embeddings, query=query,
                                      query_embedding=query_embedding, role=role,
                                      library=story_plan["library"],
                                      slot_budget_s=budget, top_k=top_k)
        pick = next((row for row in fresh if _eligible(row)), None)
        if pick is not None:
            attempts.append({"level": 2, "picked": pick})
    if not attempts:
        return {"slot_idx": slot_idx, "status": "failed",
                "reason": "no eligible candidate in pool or index", "levels_tried": [1, 2]}

    picked = attempts[0]["picked"]
    trial = deepcopy(story_plan)
    trial_slot = trial["slots"][slot_idx]
    source_start = float(picked.get("source_start_s", 0))
    source_end = float(picked.get("source_end_s", 0))
    budget = float(trial_slot["target_interval"][1]) - float(trial_slot["target_interval"][0])
    trimmed = False
    caption = str(picked.get("caption") or "")
    shot_indices = list(picked.get("shot_indices") or [])
    evidence = _select_evidence_window(
        picked.get("member_shots") or [], budget,
        "；".join(filter(None, [reason, str(picked.get("query") or ""), caption])))
    if evidence is not None:
        source_start, source_end = evidence["start_s"], evidence["end_s"]
        if evidence["end_s"] - evidence["start_s"] > budget + 1e-6:
            source_end = source_start + budget          # 单镜头超预算兜底
        trimmed = (abs(evidence["start_s"] - float(picked.get("source_start_s", 0))) > 1e-6
                   or evidence["end_s"] < float(picked.get("source_end_s", 0)) - 1e-6)
        caption = evidence["caption"] or caption
        shot_indices = evidence.get("shot_indices") or shot_indices
    elif source_end - source_start > budget + 1e-6:
        source_end = source_start + budget
        trimmed = True
    trial_slot["source"] = {
        "video": str(picked.get("video") or ""), "video_stem": str(picked.get("video_stem") or ""),
        "window_idx": picked.get("window_idx"),
        "shot_idx": picked.get("shot_idx"), "shot_indices": shot_indices,
        "start_s": source_start, "end_s": source_end,
        "event_id": str(picked.get("event_id") or ""), "caption": caption,
        "causal_predecessors": list(picked.get("causal_predecessors") or []),
        "entity_ids": list(picked.get("entity_ids") or []),
        "focus_x": min(1.0, max(0.0, float(picked.get("focus_x", 0.5)))),
        "dialogue": [line for line in (picked.get("dialogue") or [])
                     if float(line.get("start_s", source_start)) >= source_start - 1e-6
                     and float(line.get("end_s", source_end)) <= source_end + 1e-6],
    }
    errors = validate_story_plan(trial)
    if errors:
        return {"slot_idx": slot_idx, "status": "failed",
                "reason": "candidate breaks plan: " + "; ".join(errors[:3]),
                "levels_tried": [attempts[0]["level"]]}
    previous = trial["slots"][slot_idx - 1] if slot_idx else None
    if previous and previous.get("source", {}).get("entity_ids") \
            and set(previous["source"]["entity_ids"]) & set(trial_slot["source"]["entity_ids"]):
        trial_slot["transition_reason"] = "entity_continuity"
        trial_slot["status"] = "supported"
    else:
        trial_slot["transition_reason"] = "unexplained" if slot_idx else "opening"
        trial_slot["status"] = "uncertain" if slot_idx else "supported"
    trial_slot["reason"] = f"re_searched:{reason[:60]}"
    trial_slot["source_interval_trimmed"] = trimmed
    story_plan.clear()
    story_plan.update(trial)
    return {"slot_idx": slot_idx, "status": "replaced", "level": attempts[0]["level"],
            "video": trial_slot["source"]["video"],
            "start_s": trial_slot["source"]["start_s"]}


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
        # 文案轨（7682 型模板）：cues 已在成片时间轴，渲染时压制对白翻译字幕
        "copy_cues": (story_plan.get("copy") or {}).get("cues"),
        "audio_mode": (story_plan.get("copy") or {}).get("audio_mode"),
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
    if not STORY_MIN_TARGET_DURATION_S <= duration <= STORY_MAX_TARGET_DURATION_S:
        errors.append("target_duration_s outside "
                      f"{STORY_MIN_TARGET_DURATION_S:g}..{STORY_MAX_TARGET_DURATION_S:g}")
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
