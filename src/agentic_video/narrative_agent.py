"""Bounded active-perception agent for evidence-grounded narrative induction."""
from __future__ import annotations

import hashlib
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

# 两层依赖缓存（2026-09-10 V1 计划 P0）：命中依据是输入依赖而非文件存在性——
# 观察键 = 源哈希+区间+实际提示词哈希+采样参数+模型版本；归纳键 = 观察结果+
# 外部上下文+归纳提示词+模型配置。prompt_version 只是人读版本号。
CACHE_SCHEMA_VERSION = "narrative_cache_v1"
WINDOW_PROMPT_VERSION = "v1"
SYNTHESIS_PROMPT_VERSION = "v2"


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                     sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _omni_signature(cfg) -> dict:
    """参与两层缓存键的模型/采样配置指纹。"""
    perception = getattr(cfg, "perception", None) or {}
    omni = perception.get("omni") or {}
    try:
        fps = float(omni.get("fps") or 2.0)
    except (TypeError, ValueError):
        fps = 2.0
    return {"model": str(omni.get("model_path") or "omni"), "fps": fps,
            "max_new_tokens": 1024}


def _observation_key(video_sha: str, window: "NarrativeWindow", material: str,
                     omni_sig: dict) -> str:
    return _stable_hash({
        "schema": CACHE_SCHEMA_VERSION, "kind": "observation",
        "video_sha": video_sha, "interval": [window.start, window.end],
        "probe": window.probe, "reason": window.reason,
        "prompt_version": WINDOW_PROMPT_VERSION,
        "prompt_sha": _stable_hash(WINDOW_PROMPT),
        "material_sha": _stable_hash(material), **omni_sig})


def _load_window_cache(window_dir: Path, key: str) -> dict | None:
    """窗口观察缓存：params.cache_key 匹配才复用（缺失=旧产物，视为 miss 重看）。"""
    path = window_dir / "result.json"
    if not path.exists():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(envelope, dict):
        return None
    if (envelope.get("params") or {}).get("cache_key") != key:
        return None
    output = envelope.get("output")
    if isinstance(output, dict) and isinstance(output.get("observations"), list):
        return output
    return None


def _synthesis_key(video_sha: str, material: str, observations: list[dict],
                   omni_sig: dict) -> str:
    return _stable_hash({
        "schema": CACHE_SCHEMA_VERSION, "kind": "synthesis",
        "video_sha": video_sha, "material_sha": _stable_hash(material),
        "observations_sha": _stable_hash(observations),
        "prompt_version": SYNTHESIS_PROMPT_VERSION,
        "prompt_sha": _stable_hash(SYNTHESIS_PROMPT), **omni_sig})


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
可验证的叙事程序。只输出一个 JSON 对象，不要围栏。

证据分级（必须遵守）：
- 画面、可辨认对白：可支持可见行动、说话内容；不得支持未呈现的动机或关系。
- 视频内说明文字（OCR）：创作者对事件的陈述，不是已独立核实的事实。
- 外部标题/评论/相关推荐：只是语境、解读、待核验线索——引用时 evidence.source 用
  external_title/external_comment/external_related 且必须 quote 原文；不能当作本视频中必然发生的事件。
- 禁止根据常识补写没有证据的剧情；无法分辨就写 uncertain，不要编。

主角对账：开场/钩子文字陈述的主角处境，必须与画面主体的可见特征（性别、外观、伤情等）
逐项核对；冲突时以画面+OCR 为准，并把对应字段降为 uncertain 请求复核；不得预设人物属性。

顶层字段固定为 intent/entities/events/causal_links/arc/utterances/emotion_curve/evidence/status/uncertainties。
顶层 evidence/status 由程序根据各项证据重新汇总，模型不得用它掩盖无证据断言。
intent={{"topic":"...","message":"...","content_type":"real_story|screen_story|growth_story|uncertain",
"protagonist":"主角是谁（按画面可见特征描述）或unknown",
"goal":"主角的目标或not_applicable","problem":"面临什么问题或unknown",
"motivation":"为什么采取关键行动；画面未呈现就写unknown，不补写",
"change":"发生了什么变化或not_applicable","outcome":"最后的结果或unknown",
"evidence":[],"confidence":0.0,"status":"supported|uncertain|unsupported"}}
entity={{"id":"entity_000","kind":"person|animal|object|place|group|unknown","name_or_role":"...",
"aliases":[],"evidence":[],"confidence":0.0,"status":"..."}}
event={{"id":"event_000","interval":[0.0,1.0],"participants":["entity_000"],"action":"...",
"state_before":"...或uncertain","state_after":"...或uncertain",
"importance":"high|medium|low","evidence":[],"confidence":0.0,"status":"..."}}
causal_link={{"from_event":"event_000","to_event":"event_001",
"relation":"causes|motivates|enables|prevents|reveals","evidence":[],"confidence":0.0,"status":"..."}}
arc={{"role":"hook|context|conflict|choice|climax|consequence|resolution",
"event_ids":["event_000"],"function":"该段在整条表达中的作用（一句话）"}}
utterance={{"id":"utterance_000","interval":[0.0,1.0],"speaker_id":"entity_000或null",
"original":"原文或可靠概括","translation_zh":"已有中文则同原文；否则可靠翻译或uncertain",
"evidence":[],"confidence":0.0,"status":"..."}}
emotion={{"interval":[0.0,1.0],"emotion":"...","intensity":0.0,
"evidence":[],"confidence":0.0,"status":"..."}}

同一事件可以被多个弧段引用，但各段必须写明不同的 function、使用区间或新增信息；
禁止把同一事件的相同描述复制进多个弧段凑齐故事弧。

所有 supported 项必须带 evidence；evidence 必须含 source、interval、confidence，可含 frame/bbox/quote。
因果只能从较早事件指向较晚事件。无法分辨时降为 uncertain，不要强行补齐七种 arc。

【视频时长】{duration:g}s
【确定性材料（含外部语境线索）】{material}
【分窗观察】{observations}
【上一版程序与补充探针】{previous}
"""


def _read_output(cfg, vid: str, tool: str) -> dict:
    env = common.read_result_json(cfg.paths.perception_dir / vid / tool)
    return (env or {}).get("output") or {}


def _load_reference_metadata(cfg, vid: str) -> dict:
    """Manually curated references may carry the richer context.json instead of
    the downloader's metadata.json; merge both (context wins)."""
    metadata: dict = {}
    for filename in ("metadata.json", "context.json"):
        metadata_path = cfg.paths.videos_dir / vid / filename
        if not metadata_path.exists():
            continue
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                metadata.update(value)
        except ValueError:
            continue
    return metadata


def build_reference_identity(cfg, vid: str, *, video: Path, video_sha: str) -> dict:
    """P0 参考身份核对：任何验收前先确认 reference_id ↔ 文件 ↔ sha256 ↔ 标题
    对得上，并给出关键证据位置。防止拿计划里的描述当核对标准。"""
    metadata = _load_reference_metadata(cfg, vid)
    shots = _read_output(cfg, vid, "shots")
    boundaries = [float(value) for value in shots.get("boundaries_s") or []][:3]
    ocr = _read_output(cfg, vid, "ocr")
    key_evidence = []
    for event in (ocr.get("text_events") or [])[:5]:
        key_evidence.append({
            "type": "ocr",
            "interval": [float(event.get("t_start_s") or 0),
                         float(event.get("t_end_s") or 0)],
            "text": str(event.get("text") or "")[:60],
        })
    if boundaries:
        key_evidence.append({"type": "shot_boundary", "interval": boundaries})
    return {
        "reference_id": vid, "video": str(video), "source_sha256": video_sha,
        "metadata_title": str(metadata.get("title") or ""),
        "metadata_author": str(metadata.get("author") or ""),
        "key_evidence_locations": key_evidence,
    }


def build_narrative_material(cfg, vid: str, *, start: float | None = None,
                             end: float | None = None,
                             external: bool = True) -> str:
    """确定性材料。external=False 供窗口观察用（只有片内信号，杜绝外部语境
    污染观察层，同时观察缓存不随评论更新而失效）；external=True 供归纳层，
    外部线索按证据分级表标注引用方式。"""
    rows = []
    if external:
        metadata = _load_reference_metadata(cfg, vid)
        rows.append("外部标题（语境线索；引用须 source=external_title 并 quote 原文，"
                    "不作为本片事实）：" + str(metadata.get("title") or "无"))
        if metadata.get("author"):
            rows.append("外部作者：" + str(metadata["author"]))
        if metadata.get("hashtags"):
            rows.append("外部话题：" + ",".join(str(v) for v in metadata["hashtags"][:20]))
        comments = metadata.get("comments") or []
        for comment in comments[:20]:
            text = str(comment if isinstance(comment, str)
                       else comment.get("text") or "").strip()
            if text:
                rows.append(f"外部评论（语境线索，source=external_comment）：{text[:80]}")
        related = metadata.get("related_videos") or []
        titles = [str(row.get("title") or "").strip()[:40]
                  for row in related[:5] if isinstance(row, dict)]
        if titles:
            rows.append("相关推荐标题（待核验线索，source=external_related）："
                        + "；".join(titles))
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
        duration_s=duration_s, fps=fps, model=model, prompt_version="narrative_agent_v2")
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
    # 必答六问（P0 v2）：允许 unknown / not_applicable——未呈现就承认，不补写
    for field in ("protagonist", "goal", "problem", "motivation", "change", "outcome"):
        intent[field] = str(intent.get(field) or "").strip() or "unknown"
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
        if event.get("importance") not in ("high", "medium", "low"):
            event["importance"] = "medium"
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
        if not ids:
            continue
        entry = {"role": segment["role"], "event_ids": ids}
        # 同一事件可被多槽引用，但必须写明各段的不同功能（v2 红线：禁止复制描述凑弧）
        function = str(segment.get("function") or "").strip()
        if function:
            entry["function"] = function
        arc.append(entry)
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


def _ensure_runner(cfg, runner):
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
    return runner


def _observe_window(cfg, vid: str, video: Path, window: NarrativeWindow,
                    window_dir: Path, runner, omni_sig: dict, video_sha: str,
                    *, force: bool) -> tuple[dict, bool, object]:
    """单窗观察，依赖键命中即复用（miss 才看片）。返回 (parsed, cached, runner)。"""
    material = build_narrative_material(cfg, vid, start=window.start, end=window.end,
                                        external=False)
    key = _observation_key(video_sha, window, material, omni_sig)
    if not force:
        cached = _load_window_cache(window_dir, key)
        if cached is not None:
            return cached, True, runner
    runner = _ensure_runner(cfg, runner)
    prompt = WINDOW_PROMPT.format(
        start=window.start, end=window.end, probe=window.probe, reason=window.reason,
        material=material)
    answer = runner.watch(video, prompt, start_s=window.start, end_s=window.end,
                          clip_dir=window_dir, max_new_tokens=1024,
                          duration_s=window.end - window.start)
    parsed = parse_observation(answer.text)
    params = asdict(window)
    params["cache_key"] = key
    common.write_result_json(window_dir, tool="narrative_window", aweme_id=vid,
                             params=params, output=parsed)
    return parsed, False, runner


def run_narrative_agent(cfg, vid: str, *, output: Path | None = None,
                        budget: NarrativeBudget | None = None, force: bool = False,
                        runner=None) -> dict:
    budget = budget or NarrativeBudget()
    root = cfg.paths.perception_dir / vid / "narrative_agent"
    result_path = root / "result.json"
    inspect = _read_output(cfg, vid, "inspect")
    duration = float(inspect.get("duration_s") or 0)
    fps = float(inspect.get("fps") or 24)
    if duration <= 0:
        raise FileNotFoundError(f"缺 inspect 产物：{vid}")
    video = cfg.paths.videos_dir / vid / "video.mp4"
    video_sha = sha256_file(video)
    omni_sig = _omni_signature(cfg)
    full_material = build_narrative_material(cfg, vid)

    # 归纳缓存快路径：信封的依赖指纹（源哈希+外部材料+归纳提示词+模型配置）与
    # 当前输入完全一致才复用。旧信封（无 cache 字段）视为陈旧——改 prompt 后
    # 不再被存在性判断骗过。观察缓存照常兜底，miss 的窗口才重新看片。
    if result_path.exists() and not force:
        try:
            envelope = json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            envelope = None
        cache = (envelope or {}).get("cache") if isinstance(envelope, dict) else {}
        if (isinstance(envelope, dict)
                and (cache or {}).get("schema") == CACHE_SCHEMA_VERSION
                and ((cache or {}).get("prompt_versions") or {}).get("synthesis")
                == SYNTHESIS_PROMPT_VERSION
                and (cache or {}).get("synthesis_key") == _synthesis_key(
                    video_sha, full_material,
                    envelope.get("observations") or [], omni_sig)):
            program = envelope["program"]
            if output is not None:
                write_narrative_program(program, Path(output))
            return program

    windows = plan_narrative_windows(
        duration, max_windows=budget.max_initial_windows,
        target_window_s=budget.target_window_s)
    observations = []
    tool_calls = []
    for window in windows:
        window_dir = root / "initial" / f"{window.idx:03d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        parsed, cached, runner = _observe_window(
            cfg, vid, video, window, window_dir, runner, omni_sig, video_sha,
            force=force)
        observations.extend(parsed["observations"])
        tool_calls.append({"tool": "omni_narrative_window", "cached": cached,
                           **asdict(window)})

    runner = _ensure_runner(cfg, runner)
    prompt = SYNTHESIS_PROMPT.format(
        duration=duration, material=full_material,
        observations=json.dumps(observations, ensure_ascii=False)[:24000], previous="无")
    answer = runner.ask(prompt, max_new_tokens=4096)
    program = parse_narrative_program(
        answer.text, reference_id=vid, reference_uri=str(video), sha256=video_sha,
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
        for task in tasks:
            task_dir = root / f"refinement_{round_idx}" / f"{task.idx:03d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            parsed, cached, runner = _observe_window(
                cfg, vid, video, task, task_dir, runner, omni_sig, video_sha,
                force=force)
            observations.extend(parsed["observations"])
            tool_calls.append({"tool": "omni_narrative_probe", "cached": cached,
                               **asdict(task)})
        refinement_count += len(tasks)
        repair_prompt = SYNTHESIS_PROMPT.format(
            duration=duration, material=full_material,
            observations=json.dumps(observations, ensure_ascii=False)[:30000],
            previous=json.dumps(program, ensure_ascii=False)[:16000])
        answer = runner.ask(repair_prompt, max_new_tokens=4096)
        candidate = parse_narrative_program(
            answer.text, reference_id=vid, reference_uri=str(video), sha256=video_sha,
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
        "prompt_versions": {"window": WINDOW_PROMPT_VERSION,
                            "synthesis": SYNTHESIS_PROMPT_VERSION},
        "reference_identity": build_reference_identity(
            cfg, vid, video=video, video_sha=video_sha),
        "cache": {
            "schema": CACHE_SCHEMA_VERSION,
            "prompt_versions": {"window": WINDOW_PROMPT_VERSION,
                                "synthesis": SYNTHESIS_PROMPT_VERSION},
            "video_sha": video_sha,
            "synthesis_key": _synthesis_key(video_sha, full_material,
                                            observations, omni_sig),
        },
        "observations": observations, "program": program,
    }
    result_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    destination = Path(output) if output else root / "reference.narrative.json"
    write_narrative_program(program, destination)
    return program
