"""V5.1 scene-scoped editorial planning.

Evidence localization owns source truth.  This module may only select verified
candidate ids, omit material, and preserve chronological order; it never moves
evidence boundaries or invents new source intervals.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.template.schema import extract_json_block


PLAN_IDS = ("viewpoint", "question_answer", "core_close")
EDITORIAL_FUNCTIONS = {"setup", "question", "core_statement", "response", "ending"}
CUT_IN_REASONS = {
    "question_starts", "speaker_change", "new_information",
    "core_statement_starts", "reaction_starts", "visual_subject_change",
}
CUT_OUT_REASONS = {
    "question_complete", "semantic_unit_complete", "answer_complete",
    "reaction_complete", "redundant_content_starts", "next_function_begins",
}


OBSERVATION_PROMPT = """你是素材观察员。观看给定的候选片段，只记录片段内实际可见、
可听的事实，不解释它在剪辑中应承担什么功能，不猜官方角色名。只输出 JSON：
{"observed_dialogue":"实际可听对白；没有则为空", "observed_action":"实际可见动作",
"speaker":"说话人的局部外观描述", "addressee":"说话对象的局部外观描述",
"observation_confidence":0.0}
禁止输出 question/core/ending 等编辑标签。"""


PLAN_PROMPT = """你是短视频 Editorial Planner。目标是最大化 communicative density：
每一秒都必须帮助陌生观众理解“{goal}”。候选素材已经由独立观察层核验；你只能引用
下列 candidate_id，不能修改其时间、拆分候选、重复候选或发明新素材。保持源片时间顺序，
音画锁定。为三个固定假设各给一个方案；确实没有所需事实就 supported=false，不能硬解释。

所有 supported=true 的方案都必须引用所有 required_evidence_ids 非空的候选。
question_answer 只有在核心观点之前确有可听问句时才可 supported=true；
core_close 必须以核心观点候选开场。不要为了凑段数加入纯氛围画面。

每个枚举字段只能填写一个值，严禁把多个值用“|”连接，严禁原样抄写候选列表或说明文字。
proposed_function 只能选：setup, question, core_statement, response, ending。
cut_in_reason 只能选：question_starts, speaker_change, new_information,
core_statement_starts, reaction_starts, visual_subject_change。
cut_out_reason 只能选：question_complete, semantic_unit_complete,
answer_complete, reaction_complete, redundant_content_starts, next_function_begins。
reason_detail 必须结合该候选的真实事实解释入点和出点。
unsupported_reason 必须具体说明候选中缺少的事实，不得写“缺少什么事实”。

候选事实：
{candidates}

只输出以下形状的 JSON。示例中的枚举值只是单值格式示意，必须按真实候选改写：
{{"variants":[
 {{"plan_id":"viewpoint","supported":true,"unsupported_reason":"",
   "segments":[{{"candidate_id":"clip_001","proposed_function":"setup",
   "cut_in_reason":"new_information","cut_out_reason":"next_function_begins",
   "reason_detail":"结合真实内容说明入点与出点"}}]}},
 {{"plan_id":"question_answer","supported":false,
   "unsupported_reason":"核心观点前没有可听的真实问句","segments":[]}},
 {{"plan_id":"core_close","supported":false,
   "unsupported_reason":"核心观点后没有能形成收束的真实反应","segments":[]}}
]}}

结构定义：viewpoint=必要语境→核心观点→可选收束；question_answer=真实问题→核心回答
→可选反应；core_close=前三秒直接出现核心观点→可选收束。"""


EDITORIAL_REVIEW_PROMPT = """你是第一次看到这条短视频的观众。只观看实际 mp4，
不知道 Story Plan、Edit Plan、候选标签或目标答案。判断真实剪辑是否自洽，只输出 JSON：
{"core_statement":"完整复述视频主要观点；无法复述则为空",
"first_three_seconds_summary":"前三秒给出的观看理由",
"core_meaning_preserved":true,
"opening_reason_clear":true,
"functionless_segment_present":false,
"all_cuts_purposeful":true,
"ending_intentional":true,
"transition_coherent":true,
"scores":{"core":0.0,"opening":0.0,"function":0.0,"ending":0.0,"continuity":0.0}}
scores 每项为 0~1；不要因为视频有切点就判为好剪辑，明显跳话、切半句、姿势突变或无功能
镜头必须反映在布尔字段与分数中。"""


def _interval(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _contained(interval: tuple[float, float], scope: tuple[float, float]) -> bool:
    return scope[0] - 1e-6 <= interval[0] < interval[1] <= scope[1] + 1e-6


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[1] > right[0] and left[0] < right[1]


def _json_object(raw: str) -> dict | None:
    block = extract_json_block(str(raw or ""))
    try:
        value = json.loads(block) if block else None
    except ValueError:
        value = None
    return value if isinstance(value, dict) else None


def _story_inputs(story_plan: dict) -> tuple[tuple[float, float], list[tuple[float, float]], str]:
    scope_dict = story_plan.get("source_scope") or {}
    scope = (float(scope_dict.get("start_s", 0)), float(scope_dict.get("end_s", 0)))
    required = []
    video = str(story_plan.get("source_video") or "")
    for slot in story_plan.get("slots") or []:
        source = slot.get("source") or {}
        if not video and source.get("video"):
            video = str(source["video"])
        interval = _interval(source.get("required_evidence_interval"))
        if slot.get("required") and interval:
            required.append(interval)
    return scope, required, video


def _dialogue_rows(scoped_rows: list[dict], scope: tuple[float, float]) -> list[dict]:
    seen, rows = set(), []
    for source in scoped_rows:
        for line in source.get("dialogue") or []:
            interval = _interval(line.get("utterance_interval") or
                                 [line.get("start_s"), line.get("end_s")])
            if not interval or not _overlap(interval, scope):
                continue
            clipped = (max(interval[0], scope[0]), min(interval[1], scope[1]))
            key = (str(line.get("utterance_id") or ""), clipped)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "kind": "dialogue", "source_interval": list(clipped),
                "utterance_ids": [str(line.get("utterance_id") or "")],
                "utterance_intervals": [list(interval)],
                "transcript_text": str(line.get("original") or
                                       line.get("translation_zh") or "").strip(),
                "dialogue": [deepcopy(line)],
            })
    return sorted(rows, key=lambda row: tuple(row["source_interval"]))


def _shot_rows(shots: list[dict], scope: tuple[float, float]) -> list[dict]:
    rows = []
    for shot in shots:
        interval = _interval([shot.get("start_s"), shot.get("end_s")])
        if not interval or not _overlap(interval, scope):
            continue
        clipped = (max(interval[0], scope[0]), min(interval[1], scope[1]))
        if clipped[1] - clipped[0] < 0.5:
            continue
        rows.append({
            "kind": "visual", "source_interval": list(clipped),
            "shot_ids": [shot.get("shot_idx", shot.get("index"))],
            "utterance_ids": [], "utterance_intervals": [],
            "transcript_text": "", "dialogue": [],
        })
    return sorted(rows, key=lambda row: tuple(row["source_interval"]))


def _candidate_pool(story_plan: dict, scoped_rows: list[dict], shots: list[dict],
                    *, candidate_max: int) -> list[dict]:
    scope, required, _video = _story_inputs(story_plan)
    dialogue = _dialogue_rows(scoped_rows, scope)
    visuals = _shot_rows(shots, scope)
    if not required:
        return []
    core_start, core_end = min(item[0] for item in required), max(item[1] for item in required)
    core_lines = [row for row in dialogue
                  if any(_overlap(tuple(row["source_interval"]), evidence)
                         for evidence in required)]
    if not core_lines:
        return []
    core = deepcopy(core_lines[0])
    if len(core_lines) > 1:
        core["source_interval"] = [core_lines[0]["source_interval"][0],
                                   core_lines[-1]["source_interval"][1]]
        core["utterance_ids"] = [value for row in core_lines
                                  for value in row["utterance_ids"]]
        core["utterance_intervals"] = [value for row in core_lines
                                        for value in row["utterance_intervals"]]
        core["transcript_text"] = " ".join(row["transcript_text"] for row in core_lines)
        core["dialogue"] = [value for row in core_lines for value in row["dialogue"]]
    core["required_evidence_ids"] = [f"required_{index:02d}"
                                      for index in range(len(required))]
    core["required_evidence_intervals"] = [list(item) for item in required]

    others = [row for row in dialogue if row not in core_lines]
    before_dialogue = [row for row in others if row["source_interval"][1] <= core_start]
    after_dialogue = [row for row in others if row["source_interval"][0] >= core_end]
    before_visual = [row for row in visuals if row["source_interval"][1] <= core_start - 0.25]
    after_visual = [row for row in visuals if row["source_interval"][0] >= core_end + 0.25]
    ordered = [core]
    for group, reverse in ((before_dialogue, True), (after_dialogue, False),
                           (before_visual, True), (after_visual, False)):
        if group:
            ordered.append(sorted(group, key=lambda row: tuple(row["source_interval"]),
                                  reverse=reverse)[0])
    remaining = [row for row in others + visuals if row not in ordered]
    remaining.sort(key=lambda row: (min(abs(row["source_interval"][0] - core_end),
                                            abs(row["source_interval"][1] - core_start)),
                                    row["kind"], tuple(row["source_interval"])))
    ordered.extend(remaining)

    unique = []
    for row in ordered:
        interval = tuple(row["source_interval"])
        if any(tuple(item["source_interval"]) == interval for item in unique):
            continue
        row.setdefault("required_evidence_ids", [])
        row.setdefault("required_evidence_intervals", [])
        unique.append(row)
        if len(unique) >= candidate_max:
            break
    unique.sort(key=lambda row: tuple(row["source_interval"]))
    return unique


def build_editorial_candidates(story_plan: dict, scoped_rows: list[dict], shots: list[dict],
                               *, runner, output_dir: Path,
                               editorial: dict | None = None) -> list[dict]:
    """Build 2--5 observed clips without assigning editorial functions."""
    settings = editorial or {}
    minimum = int(settings.get("candidate_min", 2))
    maximum = int(settings.get("candidate_max", 5))
    pool = _candidate_pool(story_plan, scoped_rows, shots, candidate_max=maximum)
    if len(pool) < minimum:
        raise RuntimeError(f"insufficient_editorial_candidates:{len(pool)}<{minimum}")
    _scope, _required, video = _story_inputs(story_plan)
    if not video or not Path(video).is_file():
        raise FileNotFoundError(video or "editorial source video missing")
    root = Path(output_dir) / "candidate_observations"
    root.mkdir(parents=True, exist_ok=True)
    source_meta = next(((slot.get("source") or {})
                        for slot in story_plan.get("slots") or []
                        if (slot.get("source") or {}).get("video")), {})
    candidates = []
    for index, row in enumerate(pool, 1):
        candidate = deepcopy(row)
        candidate_id = f"clip_{index:03d}"
        start, end = map(float, candidate["source_interval"])
        clip_dir = root / candidate_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        answer = runner.watch(Path(video), OBSERVATION_PROMPT, start_s=start, end_s=end,
                              duration_s=end - start, clip_dir=clip_dir,
                              max_new_tokens=512)
        observed = _json_object(answer.text)
        if not observed:
            raise ValueError(f"candidate_observation_parse_failed:{candidate_id}")
        candidate.update({
            "candidate_id": candidate_id,
            "video": video,
            "video_stem": str(source_meta.get("video_stem") or Path(video).stem),
            "focus_x": float(source_meta.get("focus_x", 0.5) or 0.5),
            "observed_dialogue": str(observed.get("observed_dialogue") or ""),
            "observed_action": str(observed.get("observed_action") or ""),
            "speaker": str(observed.get("speaker") or ""),
            "addressee": str(observed.get("addressee") or ""),
            "observation_confidence": max(0.0, min(1.0, float(
                observed.get("observation_confidence") or 0.0))),
            "verified": True,
        })
        candidates.append(candidate)
    return candidates


def _materialize_variant(raw: dict, candidates: list[dict], story_plan: dict) -> dict:
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    plan_id = str(raw.get("plan_id") or "")
    supported = raw.get("supported") is True
    segments, cursor = [], 0.0
    if supported:
        for index, item in enumerate(raw.get("segments") or []):
            candidate_id = str((item or {}).get("candidate_id") or "")
            candidate = candidate_by_id.get(candidate_id)
            if not candidate:
                segments.append({"segment_idx": index, "candidate_id": candidate_id,
                                 "invalid_candidate": True})
                continue
            start, end = map(float, candidate["source_interval"])
            duration = end - start
            segments.append({
                "segment_idx": index, "candidate_id": candidate_id,
                "source_interval": [start, end],
                "output_interval": [round(cursor, 3), round(cursor + duration, 3)],
                "proposed_function": str(item.get("proposed_function") or ""),
                "cut_in_reason": str(item.get("cut_in_reason") or ""),
                "cut_out_reason": str(item.get("cut_out_reason") or ""),
                "reason_detail": str(item.get("reason_detail") or ""),
                "required_evidence_ids": list(candidate.get("required_evidence_ids") or []),
            })
            cursor += duration
    return {
        "edit_plan_version": "edit_plan_v1", "plan_id": plan_id,
        "supported": supported,
        "unsupported_reason": str(raw.get("unsupported_reason") or ""),
        "editing_goal": str(story_plan.get("theme") or ""),
        "source_scope": deepcopy(story_plan.get("source_scope")),
        "segments": segments, "duration_s": round(cursor, 3),
    }


def plan_edit_variants(story_plan: dict, candidates: list[dict], *, runner,
                       output_dir: Path) -> dict:
    """Ask for the three fixed hypotheses; model output may only reference ids."""
    scope, _required, video = _story_inputs(story_plan)
    facts = [{key: row.get(key) for key in (
        "candidate_id", "source_interval", "transcript_text", "observed_dialogue",
        "observed_action", "speaker", "addressee", "required_evidence_ids")}
             for row in candidates]
    prompt = PLAN_PROMPT.format(goal=story_plan.get("theme") or "清楚表达核心观点",
                                candidates=json.dumps(facts, ensure_ascii=False, indent=2))
    clip_dir = Path(output_dir) / "editorial_planner_watch"
    clip_dir.mkdir(parents=True, exist_ok=True)
    answer = runner.watch(Path(video), prompt, start_s=scope[0], end_s=scope[1],
                          duration_s=scope[1] - scope[0], clip_dir=clip_dir,
                          max_new_tokens=2048)
    payload = _json_object(answer.text)
    if not payload or not isinstance(payload.get("variants"), list):
        raise ValueError("edit_plan_parse_failed")
    by_id = {str(row.get("plan_id") or ""): row for row in payload["variants"]
             if isinstance(row, dict)}
    variants = []
    for plan_id in PLAN_IDS:
        raw = deepcopy(by_id.get(plan_id) or {
            "plan_id": plan_id, "supported": False,
            "unsupported_reason": "planner_missing_variant", "segments": []})
        raw["plan_id"] = plan_id
        variants.append(_materialize_variant(raw, candidates, story_plan))
    return {"edit_plan_candidates_version": "edit_plan_candidates_v1",
            "variants": variants, "raw_head": str(answer.text)[:500]}


def evaluate_edit_plan(edit_plan: dict, story_plan: dict, candidates: list[dict],
                       *, editorial: dict | None = None) -> dict:
    """Deterministic schema/evidence checks plus a separate transform metric."""
    settings = editorial or {}
    scope, required, _video = _story_inputs(story_plan)
    min_duration = float((story_plan.get("duration_policy") or {}).get("min_s", 12.0))
    max_duration = float((story_plan.get("duration_policy") or {}).get("max_s", 26.4))
    min_gap = float(settings.get("min_source_gap_s", 0.25))
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    errors = []
    segments = edit_plan.get("segments") or []
    if not edit_plan.get("supported"):
        errors.append(f"unsupported:{edit_plan.get('unsupported_reason') or 'unspecified'}")
    if not segments:
        errors.append("segments_empty")
    prior_end = None
    target_cursor = 0.0
    seen_candidates = set()
    covered_ids = set()
    purpose_durations = {}
    gaps = []
    for index, segment in enumerate(segments):
        candidate = candidate_by_id.get(str(segment.get("candidate_id") or ""))
        interval = _interval(segment.get("source_interval"))
        if not candidate or segment.get("invalid_candidate"):
            errors.append(f"segment[{index}]:candidate_unknown")
            continue
        candidate_id = str(segment.get("candidate_id"))
        if candidate_id in seen_candidates:
            errors.append(f"segment[{index}]:candidate_repeated")
        seen_candidates.add(candidate_id)
        expected = _interval(candidate.get("source_interval"))
        if not interval or interval != expected:
            errors.append(f"segment[{index}]:candidate_interval_changed")
            continue
        if not _contained(interval, scope):
            errors.append(f"segment[{index}]:scope_violation")
        output_interval = _interval(segment.get("output_interval"))
        if not output_interval \
                or abs(output_interval[0] - target_cursor) > 1e-3 \
                or abs((output_interval[1] - output_interval[0])
                       - (interval[1] - interval[0])) > 1e-3:
            errors.append(f"segment[{index}]:output_timeline_invalid")
        else:
            target_cursor = output_interval[1]
        if prior_end is not None:
            if interval[0] < prior_end - 1e-6:
                errors.append(f"segment[{index}]:source_not_chronological")
            gaps.append(max(0.0, interval[0] - prior_end))
        prior_end = interval[1]
        function = str(segment.get("proposed_function") or "")
        if function not in EDITORIAL_FUNCTIONS:
            errors.append(f"segment[{index}]:invalid_function")
        if segment.get("cut_in_reason") not in CUT_IN_REASONS:
            errors.append(f"segment[{index}]:invalid_cut_in_reason")
        if segment.get("cut_out_reason") not in CUT_OUT_REASONS:
            errors.append(f"segment[{index}]:invalid_cut_out_reason")
        if not str(segment.get("reason_detail") or "").strip():
            errors.append(f"segment[{index}]:reason_detail_missing")
        covered_ids.update(candidate.get("required_evidence_ids") or [])
        purpose_durations[function] = purpose_durations.get(function, 0.0) + \
            (interval[1] - interval[0])
    duration = float(edit_plan.get("duration_s") or 0)
    if not min_duration <= duration <= max_duration:
        errors.append(f"duration_outside_budget:{duration:g}")
    required_ids = {f"required_{index:02d}" for index in range(len(required))}
    if not required_ids.issubset(covered_ids):
        errors.append("required_evidence_not_covered")
    functions = [str(row.get("proposed_function") or "") for row in segments]
    if "core_statement" not in functions:
        errors.append("core_statement_missing")
    plan_id = edit_plan.get("plan_id")
    if plan_id == "viewpoint":
        allowed = {"setup", "core_statement", "ending"}
        if any(value not in allowed for value in functions) \
                or functions.count("core_statement") != 1:
            errors.append("viewpoint_functions_invalid")
    required_utterances = []
    for candidate in candidates:
        if candidate.get("required_evidence_ids"):
            required_utterances.extend(
                interval for value in candidate.get("utterance_intervals") or []
                if (interval := _interval(value)))
    selected_intervals = [_interval(row.get("source_interval")) for row in segments]
    for utterance in required_utterances:
        pieces = sorted((max(interval[0], utterance[0]), min(interval[1], utterance[1]))
                        for interval in selected_intervals
                        if interval and _overlap(interval, utterance))
        if any(right[0] > left[1] + 1e-6 for left, right in zip(pieces, pieces[1:])):
            errors.append("required_dialogue_internal_gap")
    if plan_id == "question_answer":
        if not {"question", "core_statement"}.issubset(functions) \
                or functions.index("question") > functions.index("core_statement"):
            errors.append("question_answer_functions_missing")
    if plan_id == "core_close":
        core = next((row for row in segments
                     if row.get("proposed_function") == "core_statement"), None)
        if not core or float(core["output_interval"][0]) > 3.0 + 1e-6:
            errors.append("core_not_in_first_three_seconds")
    transform = len(segments) >= int(settings.get("min_segments", 2)) \
        and any(gap >= min_gap - 1e-6 for gap in gaps)
    total = sum(purpose_durations.values())
    return {
        "plan_id": edit_plan.get("plan_id"), "passed": not errors,
        "errors": errors,
        "metrics": {
            "segment_count": len(segments), "cut_count": max(0, len(segments) - 1),
            "source_gap_count": sum(gap >= min_gap - 1e-6 for gap in gaps),
            "omitted_source_duration_s": round(sum(gaps), 3),
            "editorial_transform_present": transform,
            "editorial_density": round(total / duration, 4) if duration > 0 else 0.0,
            "purposeful_duration_s": {key: round(value, 3)
                                      for key, value in purpose_durations.items()},
        },
    }


def edit_plan_execution_inputs(edit_plan: dict, story_plan: dict,
                               candidates: list[dict], *,
                               audio_policy: dict | None = None) -> tuple[dict, list[dict]]:
    """Adapt an immutable Edit Plan to the existing multi-segment renderer."""
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    slots, retrieval = [], []
    for index, segment in enumerate(edit_plan.get("segments") or []):
        candidate = candidate_by_id[segment["candidate_id"]]
        source_start, source_end = map(float, candidate["source_interval"])
        target_start, target_end = map(float, segment["output_interval"])
        slots.append({
            "slot_idx": index, "start_s": target_start, "end_s": target_end,
            "need_duration_s": target_end - target_start,
            "theme": story_plan.get("theme") or "",
            "story_role": segment.get("proposed_function"),
            "event_id": candidate.get("candidate_id"),
            "operation_types": ["hard_cut"],
            "query": segment.get("reason_detail") or segment.get("proposed_function"),
        })
        retrieval.append({
            "slot_idx": index, "query": slots[-1]["query"], "missing": "",
            "candidates": [], "picked": {
                "row_idx": index, "video": candidate["video"],
                "video_stem": candidate.get("video_stem") or Path(candidate["video"]).stem,
                "shot_idx": (candidate.get("shot_ids") or [None])[0],
                "shot_indices": candidate.get("shot_ids") or [],
                "source_start_s": source_start, "source_end_s": source_end,
                "duration_s": source_end - source_start,
                "caption": candidate.get("observed_action") or
                           candidate.get("observed_dialogue") or "",
                "entity_ids": [], "event_id": candidate["candidate_id"],
                "causal_predecessors": [], "dialogue": deepcopy(candidate.get("dialogue") or []),
                "bindings": [], "required_evidence_interval": (
                    candidate.get("required_evidence_intervals") or [None])[0],
                "actual_rendered_source_interval": [source_start, source_end],
                "utterance_interval": (candidate.get("utterance_intervals") or [None])[0],
                "focus_x": float(candidate.get("focus_x", 0.5)),
            },
        })
    return ({
        "plan_version": "editorial-1.0", "theme": story_plan.get("theme") or "",
        "library": story_plan.get("library"), "reference_id": story_plan.get("reference_id"),
        "narrative_program_required": True, "slots": slots, "copy_cues": None,
        "audio_mode": (audio_policy or {}).get("audio_mode"),
        "audio_policy": deepcopy(audio_policy),
        "source_scope": deepcopy(story_plan.get("source_scope")),
        "story_plan_sha256": story_plan.get("story_plan_sha256"),
    }, retrieval)


def parse_editorial_review(raw: str) -> dict:
    payload = _json_object(raw)
    if not payload:
        return {"parsed": False, "raw_head": str(raw or "")[:400]}
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    normalized = {}
    for key in ("core", "opening", "function", "ending", "continuity"):
        try:
            normalized[key] = max(0.0, min(1.0, float(scores.get(key, 0))))
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return {
        "parsed": True,
        "core_statement": str(payload.get("core_statement") or ""),
        "first_three_seconds_summary": str(payload.get("first_three_seconds_summary") or ""),
        "core_meaning_preserved": payload.get("core_meaning_preserved") is True,
        "opening_reason_clear": payload.get("opening_reason_clear") is True,
        "functionless_segment_present": payload.get("functionless_segment_present") is True,
        "all_cuts_purposeful": payload.get("all_cuts_purposeful") is True,
        "ending_intentional": payload.get("ending_intentional") is True,
        "transition_coherent": payload.get("transition_coherent") is True,
        "scores": normalized,
    }


def review_editorial_preview(preview: Path, *, runner) -> dict:
    """Blind review a real low-cost edit; no plan or target context is supplied."""
    answer = runner.watch(Path(preview), EDITORIAL_REVIEW_PROMPT, max_new_tokens=1024)
    return parse_editorial_review(answer.text)


def select_edit_plan(edit_plans: list[dict], deterministic: dict[str, dict],
                     reviews: dict[str, dict]) -> tuple[dict | None, dict]:
    """Hard booleans decide eligibility; scores rank only eligible previews."""
    rankings = []
    for plan in edit_plans:
        plan_id = str(plan.get("plan_id") or "")
        check = deterministic.get(plan_id) or {}
        review = reviews.get(plan_id) or {}
        reasons = list(check.get("errors") or [])
        metrics = check.get("metrics") or {}
        if check.get("passed") and metrics.get("editorial_transform_present") is not True:
            reasons.append("extraction_only")
        if check.get("passed"):
            if review.get("parsed") is not True:
                reasons.append("editorial_preview_parse_failed")
            if review.get("core_meaning_preserved") is not True:
                reasons.append("core_meaning_not_preserved")
            if review.get("opening_reason_clear") is not True:
                reasons.append("opening_reason_unclear")
            if review.get("functionless_segment_present") is not False:
                reasons.append("functionless_segment_present")
            if review.get("all_cuts_purposeful") is not True:
                reasons.append("cuts_not_purposeful")
        score_values = review.get("scores") or {}
        score = (30 * float(score_values.get("core", 0))
                 + 20 * float(score_values.get("opening", 0))
                 + 20 * float(score_values.get("function", 0))
                 + 15 * float(score_values.get("ending", 0))
                 + 15 * float(score_values.get("continuity", 0)))
        rankings.append({
            "plan_id": plan_id, "eligible": not reasons,
            "score": round(score, 3), "reasons": reasons,
            "duration_s": plan.get("duration_s"), "metrics": metrics,
        })
    order = {value: index for index, value in enumerate(PLAN_IDS)}
    eligible = [row for row in rankings if row["eligible"]]
    eligible.sort(key=lambda row: (-row["score"], float(row["duration_s"] or 0),
                                  int((row["metrics"] or {}).get("segment_count") or 0),
                                  order.get(row["plan_id"], 99)))
    selected_id = eligible[0]["plan_id"] if eligible else None
    selected = next((plan for plan in edit_plans if plan.get("plan_id") == selected_id), None)
    return selected, {"editorial_gate_version": "editorial_gate_v1",
                      "passed": selected is not None, "selected_plan_id": selected_id,
                      "rankings": rankings}
