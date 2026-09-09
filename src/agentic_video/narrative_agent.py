"""Bounded active-perception agent for evidence-grounded narrative induction."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agentic_video.narrative import (ARC_ROLES, CONTENT_TYPES, ENTITY_KINDS,
                                          NARRATIVE_VERSION, STATUSES,
                                          new_narrative_program,
                                          validate_narrative_program,
                                          write_narrative_program)
from src.agentic_video.recipe_v2 import sha256_file
from src.perception import common
from src.template.schema import extract_json_block


@dataclass(frozen=True)
class NarrativeBudget:
    max_initial_windows: int = 24
    max_rounds: int = 2
    max_refinement_windows: int = 12
    target_window_s: float = 12.0


@dataclass(frozen=True)
class NarrativeWindow:
    idx: int
    start: float
    end: float
    phase: str = "initial"
    probe: str = "event_observation"
    reason: str = "timeline_coverage"


def plan_narrative_windows(duration_s: float, *, max_windows: int = 24,
                           target_window_s: float = 12.0) -> list[NarrativeWindow]:
    if duration_s <= 0 or max_windows <= 0:
        return []
    target = max(1.0, float(target_window_s))
    count = min(max_windows, max(1, math.ceil(duration_s / target)))
    if duration_s <= count * target:
        width = duration_s / count
        starts = [idx * width for idx in range(count)]
        ends = [duration_s if idx == count - 1 else (idx + 1) * width
                for idx in range(count)]
    else:
        # A long film cannot be covered under the bounded model-call budget.
        # Keep each observation small enough for Omni and distribute those
        # observations across the full source, including both endpoints.
        width = min(target, duration_s)
        step = (duration_s - width) / max(1, count - 1)
        starts = [idx * step for idx in range(count)]
        ends = [min(duration_s, start + width) for start in starts]
    return [NarrativeWindow(idx=idx, start=round(start, 3), end=round(end, 3))
            for idx, (start, end) in enumerate(zip(starts, ends))]


WINDOW_PROMPT = """你是视频内容取证 Agent。请观看原视频 {start:g}~{end:g} 秒，只报告这段中
真正可见或可听的内容，不推测画面外剧情。只输出 JSON：
{{"observations":[{{"start_s":{start:g},"end_s":{end:g},"entities":["人物或角色"],
"action":"发生的动作/状态变化","speech":"可听对白概括或null",
"emotion":"可见/可听情绪或uncertain","story_role":"hook|context|conflict|choice|climax|consequence|resolution|uncertain",
"evidence":[{{"source":"frame|asr|ocr|audio","interval":[{start:g},{end:g}],"quote":"短证据","confidence":0.0}}],
"confidence":0.0}}],"uncertainties":[]}}
探针：{probe}；原因：{reason}。时间必须使用原视频时间，不确定就写 uncertain。
【同时间段确定性材料】{material}
"""


SYNTHESIS_PROMPT = """你是 Narrative Program 归纳 Agent。根据分窗观察和确定性材料生成一份
可验证的叙事程序。禁止根据标题或常识补写没有证据的剧情。只输出一个 JSON 对象，不要围栏。

顶层字段固定为 intent/entities/events/causal_links/arc/utterances/emotion_curve/evidence/status/uncertainties。
顶层 evidence/status 由程序根据各项证据重新汇总，模型不得用它掩盖无证据断言。
intent={{"topic":"...","message":"...","content_type":"real_story|screen_story|growth_story|uncertain",
"evidence":[],"confidence":0.0,"status":"supported|uncertain|unsupported"}}
entity={{"id":"entity_000","kind":"person|animal|object|place|group|unknown","name_or_role":"...",
"aliases":[],"evidence":[],"confidence":0.0,"status":"..."}}
event={{"id":"event_000","interval":[0.0,1.0],"participants":["entity_000"],"action":"...",
"state_before":"...或uncertain","state_after":"...或uncertain","evidence":[],"confidence":0.0,"status":"..."}}
causal_link={{"from_event":"event_000","to_event":"event_001",
"relation":"causes|motivates|enables|prevents|reveals","evidence":[],"confidence":0.0,"status":"..."}}
arc={{"role":"hook|context|conflict|choice|climax|consequence|resolution","event_ids":["event_000"]}}
utterance={{"id":"utterance_000","interval":[0.0,1.0],"speaker_id":"entity_000或null",
"original":"原文或可靠概括","translation_zh":"已有中文则同原文；否则可靠翻译或uncertain",
"evidence":[],"confidence":0.0,"status":"..."}}
emotion={{"interval":[0.0,1.0],"emotion":"...","intensity":0.0,
"evidence":[],"confidence":0.0,"status":"..."}}

所有 supported 项必须带 evidence；evidence 必须含 source、interval、confidence，可含 frame/bbox/quote。
因果只能从较早事件指向较晚事件。无法分辨时降为 uncertain，不要强行补齐七种 arc。

【视频时长】{duration:g}s
【确定性材料】{material}
【分窗观察】{observations}
【上一版程序与补充探针】{previous}
"""


def _read_output(cfg, vid: str, tool: str) -> dict:
    env = common.read_result_json(cfg.paths.perception_dir / vid / tool)
    return (env or {}).get("output") or {}


def build_narrative_material(cfg, vid: str, *, start: float | None = None,
                             end: float | None = None) -> str:
    metadata = {}
    # Manually curated references may carry the richer context.json instead of
    # the downloader's metadata.json.  Treat both as hypotheses for where to
    # look; the prompts still forbid using them as visual/narrative evidence.
    for filename in ("context.json", "metadata.json"):
        metadata_path = cfg.paths.videos_dir / vid / filename
        if not metadata_path.exists():
            continue
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                metadata.update(value)
        except ValueError:
            continue
    rows = [f"外部标题（仅作待验证假设，不是证据）：{metadata.get('title') or '无'}"]
    if metadata.get("author"):
        rows.append(f"外部作者：{metadata['author']}")
    if metadata.get("hashtags"):
        rows.append("外部话题：" + ",".join(str(value) for value in metadata["hashtags"][:20]))
    shots = _read_output(cfg, vid, "shots")
    boundaries = [float(value) for value in shots.get("boundaries_s") or []]
    if start is not None and end is not None:
        boundaries = [value for value in boundaries if start <= value <= end]
    rows.append("镜头边界：" + ",".join(f"{value:g}" for value in boundaries[:80]))
    ocr = _read_output(cfg, vid, "ocr")
    for event in ocr.get("text_events") or []:
        left = float(event.get("t_start_s") or 0)
        right = float(event.get("t_end_s") or left)
        if start is not None and (right < start or left > float(end)):
            continue
        rows.append(f"OCR {left:g}~{right:g}s：{event.get('text') or ''}")
    asr = _read_output(cfg, vid, "transcribe")
    segments = asr.get("segments") or []
    if segments:
        for segment in segments:
            left = float(segment.get("start_ms") or 0) / 1000
            right = float(segment.get("end_ms") or 0) / 1000
            if start is not None and (right < start or left > float(end)):
                continue
            rows.append(f"ASR {left:g}~{right:g}s：{segment.get('text') or ''}")
    elif asr.get("full_text"):
        rows.append("ASR 无可靠分段：" + str(asr["full_text"])[:1200])
    beats = _read_output(cfg, vid, "beats")
    points = [float(value) for value in beats.get("beat_points_s") or []]
    if start is not None and end is not None:
        points = [value for value in points if start <= value <= end]
    rows.append("节拍：" + ",".join(f"{value:g}" for value in points[:80]))
    return "\n".join(rows)[:10000]


def parse_observation(raw: str) -> dict:
    block = extract_json_block(raw)
    if not block:
        return {"observations": [], "uncertainties": ["parse_failed"]}
    try:
        value = json.loads(block)
    except ValueError:
        return {"observations": [], "uncertainties": ["parse_failed"]}
    if not isinstance(value, dict):
        return {"observations": [], "uncertainties": ["parse_failed"]}
    return {
        "observations": [item for item in value.get("observations") or []
                         if isinstance(item, dict)],
        "uncertainties": [str(item) for item in value.get("uncertainties") or []],
    }


def _normalize_evidence(items: Any, duration: float) -> list[dict]:
    evidence = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("source") or "").strip():
            continue
        interval = item.get("interval")
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        try:
            left = min(duration, max(0.0, float(interval[0])))
            right = min(duration, max(left, float(interval[1])))
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            continue
        evidence.append({**item, "interval": [round(left, 3), round(right, 3)],
                         "confidence": confidence})
    return evidence


def _claim(row: dict, duration: float) -> dict:
    confidence = row.get("confidence", 0.0)
    try:
        confidence = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = _normalize_evidence(row.get("evidence"), duration)
    status = row.get("status") if row.get("status") in STATUSES else "uncertain"
    if status == "supported" and not evidence:
        status = "uncertain"
    return {**row, "evidence": evidence, "confidence": confidence, "status": status}


def _summarize_program(program: dict) -> None:
    """Derive the envelope evidence and status from normalized claims."""
    aggregated = []
    seen = set()
    sections = [("intent", [program.get("intent")])]
    sections.extend((name, program.get(name) or []) for name in
                    ("entities", "events", "causal_links", "utterances",
                     "emotion_curve"))
    for section, rows in sections:
        for row in rows:
            if not isinstance(row, dict):
                continue
            claim_id = (row.get("id") or row.get("action") or row.get("emotion")
                        or row.get("topic") or section)
            for item in row.get("evidence") or []:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                aggregated.append({**item, "claim_type": section,
                                   "claim_id": str(claim_id)})
    program["evidence"] = aggregated
    events = program.get("events") or []
    supported_events = [row for row in events if row.get("status") == "supported"]
    roles = {row.get("role") for row in program.get("arc") or []}
    if not aggregated or not events:
        program["status"] = "unsupported"
    elif (program.get("intent", {}).get("status") == "supported"
          and len(supported_events) >= 3
          and {"conflict", "climax", "resolution"}.issubset(roles)):
        program["status"] = "supported"
    else:
        program["status"] = "uncertain"


def parse_narrative_program(raw: str, *, reference_id: str, reference_uri: str,
                            sha256: str, duration_s: float, fps: float,
                            model: str, tool_calls: list[dict]) -> dict:
    program = new_narrative_program(
        reference_id=reference_id, reference_uri=reference_uri, sha256=sha256,
        duration_s=duration_s, fps=fps, model=model, prompt_version="narrative_agent_v1")
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else {}
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    intent = _claim(intent, duration_s)
    intent["topic"] = str(intent.get("topic") or "uncertain")
    intent["message"] = str(intent.get("message") or "uncertain")
    if intent.get("content_type") not in CONTENT_TYPES:
        intent["content_type"] = "uncertain"
    program["intent"] = intent

    entities = []
    for idx, raw_entity in enumerate(payload.get("entities") or []):
        if not isinstance(raw_entity, dict):
            continue
        entity = _claim(raw_entity, duration_s)
        entity["id"] = str(entity.get("id") or f"entity_{idx:03d}")
        entity["kind"] = entity.get("kind") if entity.get("kind") in ENTITY_KINDS else "unknown"
        entity["name_or_role"] = str(entity.get("name_or_role") or "未知实体")
        entity["aliases"] = [str(value) for value in entity.get("aliases") or []]
        entities.append(entity)
    program["entities"] = entities
    entity_ids = {row["id"] for row in entities}

    events = []
    for idx, raw_event in enumerate(payload.get("events") or []):
        if not isinstance(raw_event, dict):
            continue
        event = _claim(raw_event, duration_s)
        interval = event.get("interval")
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        try:
            left = min(duration_s, max(0.0, float(interval[0])))
            right = min(duration_s, max(left + 0.001, float(interval[1])))
        except (TypeError, ValueError):
            continue
        if right > duration_s:
            right = duration_s
        if right <= left:
            continue
        event["id"] = str(event.get("id") or f"event_{idx:03d}")
        event["interval"] = [round(left, 3), round(right, 3)]
        event["participants"] = [value for value in event.get("participants") or []
                                 if value in entity_ids]
        event["action"] = str(event.get("action") or "uncertain")
        event["state_before"] = str(event.get("state_before") or "uncertain")
        event["state_after"] = str(event.get("state_after") or "uncertain")
        events.append(event)
    program["events"] = events
    event_ids = {row["id"] for row in events}

    links = []
    event_by_id = {row["id"]: row for row in events}
    for raw_link in payload.get("causal_links") or []:
        if not isinstance(raw_link, dict):
            continue
        source, target = raw_link.get("from_event"), raw_link.get("to_event")
        if source not in event_ids or target not in event_ids:
            continue
        if event_by_id[source]["interval"][0] > event_by_id[target]["interval"][0]:
            continue
        link = _claim(raw_link, duration_s)
        if link.get("relation") not in {"causes", "motivates", "enables", "prevents", "reveals"}:
            link["relation"] = "reveals"
        links.append(link)
    program["causal_links"] = links

    arc = []
    for segment in payload.get("arc") or []:
        if not isinstance(segment, dict) or segment.get("role") not in ARC_ROLES:
            continue
        ids = [value for value in segment.get("event_ids") or [] if value in event_ids]
        if ids:
            arc.append({"role": segment["role"], "event_ids": ids})
    program["arc"] = arc

    utterances = []
    for idx, raw_utterance in enumerate(payload.get("utterances") or []):
        if not isinstance(raw_utterance, dict):
            continue
        utterance = _claim(raw_utterance, duration_s)
        interval = utterance.get("interval")
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        try:
            left, right = float(interval[0]), float(interval[1])
        except (TypeError, ValueError):
            continue
        if left < 0 or right <= left or right > duration_s:
            continue
        utterance["id"] = str(utterance.get("id") or f"utterance_{idx:03d}")
        utterance["interval"] = [round(left, 3), round(right, 3)]
        if utterance.get("speaker_id") not in entity_ids:
            utterance["speaker_id"] = None
        utterance["original"] = str(utterance.get("original") or "uncertain")
        utterance["translation_zh"] = str(utterance.get("translation_zh") or "uncertain")
        utterances.append(utterance)
    program["utterances"] = utterances

    curve = []
    for raw_emotion in payload.get("emotion_curve") or []:
        if not isinstance(raw_emotion, dict):
            continue
        emotion = _claim(raw_emotion, duration_s)
        interval = emotion.get("interval")
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        try:
            left, right = float(interval[0]), float(interval[1])
            intensity = min(1.0, max(0.0, float(emotion.get("intensity", 0))))
        except (TypeError, ValueError):
            continue
        if left < 0 or right <= left or right > duration_s:
            continue
        emotion["interval"] = [round(left, 3), round(right, 3)]
        emotion["emotion"] = str(emotion.get("emotion") or "uncertain")
        emotion["intensity"] = intensity
        curve.append(emotion)
    program["emotion_curve"] = curve
    program["uncertainties"] = [str(value) for value in payload.get("uncertainties") or []]
    if not block:
        program["uncertainties"].append("narrative_program_parse_failed")
    program["provenance"]["tool_calls"] = list(tool_calls)
    _summarize_program(program)
    return program


def _evidence_signature(program: dict) -> set[tuple]:
    signature = set()
    for section in ("entities", "events", "causal_links", "utterances", "emotion_curve"):
        for row in program.get(section) or []:
            for evidence in row.get("evidence") or []:
                interval = evidence.get("interval") or [None, None]
                signature.add((section, row.get("id") or row.get("action") or row.get("emotion"),
                               evidence.get("source"), tuple(interval)))
    return signature


def _needs_refinement(program: dict) -> bool:
    if len(program.get("events") or []) < 3:
        return True
    roles = {row.get("role") for row in program.get("arc") or []}
    if not {"conflict", "climax", "resolution"}.issubset(roles):
        return True
    for section in ("intent", "entities", "events", "causal_links", "utterances"):
        rows = [program.get(section)] if section == "intent" else program.get(section) or []
        if any(isinstance(row, dict) and row.get("status") == "uncertain" for row in rows):
            return True
    return False


def _refinement_windows(program: dict, initial: list[NarrativeWindow], *, round_idx: int,
                        remaining: int) -> list[NarrativeWindow]:
    candidates: list[tuple[float, float, str, str]] = []
    for event in program.get("events") or []:
        if event.get("status") == "uncertain" or not event.get("evidence"):
            left, right = event.get("interval") or [0.0, 0.0]
            candidates.append((left, right, "causal_probe", "uncertain_event"))
    roles = {row.get("role") for row in program.get("arc") or []}
    if not {"conflict", "climax", "resolution"}.issubset(roles):
        for window in initial:
            candidates.append((window.start, window.end, "context_probe", "missing_arc_role"))
    if any(row.get("speaker_id") is None for row in program.get("utterances") or []):
        for row in program.get("utterances") or []:
            if row.get("speaker_id") is None:
                candidates.append((*row["interval"], "dialogue_alignment", "unknown_speaker"))
    tasks, seen = [], set()
    for left, right, probe, reason in candidates:
        key = (round(float(left), 2), round(float(right), 2), probe)
        if key in seen:
            continue
        seen.add(key)
        tasks.append(NarrativeWindow(
            idx=len(tasks), start=round(float(left), 3), end=round(float(right), 3),
            phase=f"refinement_{round_idx}", probe=probe, reason=reason))
        if len(tasks) >= remaining:
            break
    return tasks


def run_narrative_agent(cfg, vid: str, *, output: Path | None = None,
                        budget: NarrativeBudget | None = None, force: bool = False,
                        runner=None) -> dict:
    budget = budget or NarrativeBudget()
    root = cfg.paths.perception_dir / vid / "narrative_agent"
    result_path = root / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))["program"]
    inspect = _read_output(cfg, vid, "inspect")
    duration = float(inspect.get("duration_s") or 0)
    fps = float(inspect.get("fps") or 24)
    if duration <= 0:
        raise FileNotFoundError(f"缺 inspect 产物：{vid}")
    video = cfg.paths.videos_dir / vid / "video.mp4"
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})

    windows = plan_narrative_windows(
        duration, max_windows=budget.max_initial_windows,
        target_window_s=budget.target_window_s)
    observations = []
    tool_calls = []
    for window in windows:
        window_dir = root / "initial" / f"{window.idx:03d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        prompt = WINDOW_PROMPT.format(
            start=window.start, end=window.end, probe=window.probe, reason=window.reason,
            material=build_narrative_material(cfg, vid, start=window.start, end=window.end))
        answer = runner.watch(video, prompt, start_s=window.start, end_s=window.end,
                              clip_dir=window_dir, max_new_tokens=1024,
                              duration_s=window.end - window.start)
        parsed = parse_observation(answer.text)
        observations.extend(parsed["observations"])
        tool_calls.append({"tool": "omni_narrative_window", **asdict(window)})
        common.write_result_json(window_dir, tool="narrative_window", aweme_id=vid,
                                 params=asdict(window), output=parsed)

    full_material = build_narrative_material(cfg, vid)
    prompt = SYNTHESIS_PROMPT.format(
        duration=duration, material=full_material,
        observations=json.dumps(observations, ensure_ascii=False)[:24000], previous="无")
    answer = runner.ask(prompt, max_new_tokens=4096)
    program = parse_narrative_program(
        answer.text, reference_id=vid, reference_uri=str(video), sha256=sha256_file(video),
        duration_s=duration, fps=fps, model="bounded_active_perception",
        tool_calls=tool_calls)
    before = _evidence_signature(program)
    refinement_count = 0
    stop_reason = "no_refinement_needed"
    for round_idx in range(1, budget.max_rounds + 1):
        if not _needs_refinement(program):
            stop_reason = "narrative_complete"
            break
        remaining = budget.max_refinement_windows - refinement_count
        if remaining <= 0:
            stop_reason = "refinement_budget_exhausted"
            break
        tasks = _refinement_windows(program, windows, round_idx=round_idx, remaining=remaining)
        if not tasks:
            stop_reason = "no_refinement_tasks"
            break
        new_observations = []
        for task in tasks:
            task_dir = root / f"refinement_{round_idx}" / f"{task.idx:03d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            prompt = WINDOW_PROMPT.format(
                start=task.start, end=task.end, probe=task.probe, reason=task.reason,
                material=build_narrative_material(cfg, vid, start=task.start, end=task.end))
            answer = runner.watch(video, prompt, start_s=task.start, end_s=task.end,
                                  clip_dir=task_dir, max_new_tokens=1024,
                                  duration_s=task.end - task.start)
            parsed = parse_observation(answer.text)
            new_observations.extend(parsed["observations"])
            tool_calls.append({"tool": "omni_narrative_probe", **asdict(task)})
            common.write_result_json(task_dir, tool="narrative_probe", aweme_id=vid,
                                     params=asdict(task), output=parsed)
        refinement_count += len(tasks)
        observations.extend(new_observations)
        repair_prompt = SYNTHESIS_PROMPT.format(
            duration=duration, material=full_material,
            observations=json.dumps(observations, ensure_ascii=False)[:30000],
            previous=json.dumps(program, ensure_ascii=False)[:16000])
        answer = runner.ask(repair_prompt, max_new_tokens=4096)
        candidate = parse_narrative_program(
            answer.text, reference_id=vid, reference_uri=str(video), sha256=sha256_file(video),
            duration_s=duration, fps=fps, model="bounded_active_perception",
            tool_calls=tool_calls)
        errors = validate_narrative_program(candidate)
        if errors:
            program["uncertainties"].append("repair_validation_failed: " + "; ".join(errors[:5]))
            stop_reason = "repair_validation_failed"
            break
        after = _evidence_signature(candidate)
        program = candidate
        if after == before:
            stop_reason = "no_new_evidence"
            break
        before = after
        stop_reason = "max_rounds"

    errors = validate_narrative_program(program)
    if errors:
        raise ValueError("invalid Narrative Program: " + "; ".join(errors))
    root.mkdir(parents=True, exist_ok=True)
    envelope = {
        "budget": asdict(budget), "initial_windows": len(windows),
        "refinement_windows": refinement_count, "stop_reason": stop_reason,
        "observations": observations, "program": program,
    }
    result_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    destination = Path(output) if output else root / "reference.narrative.json"
    write_narrative_program(program, destination)
    return program
