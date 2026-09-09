"""Narrative Program v1 contract and evidence-grounded validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

NARRATIVE_VERSION = "1.0"
STATUSES = ("supported", "uncertain", "unsupported")
ENTITY_KINDS = ("person", "animal", "object", "place", "group", "unknown")
ARC_ROLES = ("hook", "context", "conflict", "choice", "climax",
             "consequence", "resolution")
CAUSAL_RELATIONS = ("causes", "motivates", "enables", "prevents", "reveals")
CONTENT_TYPES = ("real_story", "screen_story", "growth_story", "uncertain")


def new_narrative_program(*, reference_id: str, reference_uri: str, sha256: str,
                          duration_s: float, fps: float,
                          model: str = "deterministic",
                          prompt_version: str = "narrative_v1",
                          seed: int = 0) -> dict:
    return {
        "program_version": NARRATIVE_VERSION,
        "reference": {
            "id": reference_id, "uri": reference_uri, "sha256": sha256,
            "duration_s": round(float(duration_s), 6), "fps": round(float(fps), 6),
        },
        "intent": {
            "topic": "uncertain", "message": "uncertain",
            "content_type": "uncertain", "evidence": [],
            "confidence": 0.0, "status": "uncertain",
        },
        "entities": [],
        "events": [],
        "causal_links": [],
        "arc": [],
        "utterances": [],
        "emotion_curve": [],
        "uncertainties": [],
        "provenance": {
            "model": model, "prompt_version": prompt_version,
            "tool_calls": [], "seed": int(seed),
        },
    }


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_interval(value: Any, duration: float, prefix: str,
                       errors: list[str], *, point_ok: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 2 or not all(_number(v) for v in value):
        errors.append(f"{prefix} invalid")
        return
    start, end = map(float, value)
    if start < 0 or end < start or end > duration + 1e-6 or (end == start and not point_ok):
        errors.append(f"{prefix} outside reference")


def _validate_claim(row: dict, duration: float, prefix: str,
                    errors: list[str]) -> None:
    status = row.get("status")
    if status not in STATUSES:
        errors.append(f"{prefix}.status invalid")
    confidence = row.get("confidence")
    if not _number(confidence) or not 0 <= confidence <= 1:
        errors.append(f"{prefix}.confidence outside [0,1]")
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        errors.append(f"{prefix}.evidence must be a list")
        return
    if status == "supported" and not evidence:
        errors.append(f"{prefix} supported claim requires evidence")
    for idx, item in enumerate(evidence):
        ep = f"{prefix}.evidence[{idx}]"
        if not isinstance(item, dict) or not str(item.get("source") or "").strip():
            errors.append(f"{ep}.source missing")
            continue
        if "interval" in item:
            _validate_interval(item["interval"], duration, f"{ep}.interval", errors,
                               point_ok=True)
        value = item.get("confidence")
        if value is not None and (not _number(value) or not 0 <= value <= 1):
            errors.append(f"{ep}.confidence outside [0,1]")


def validate_narrative_program(program: Any) -> list[str]:
    if not isinstance(program, dict):
        return ["narrative program must be an object"]
    errors: list[str] = []
    if program.get("program_version") != NARRATIVE_VERSION:
        errors.append(f"program_version must be {NARRATIVE_VERSION}")
    reference = program.get("reference")
    if not isinstance(reference, dict):
        errors.append("reference must be an object")
        duration = 0.0
    else:
        duration = float(reference.get("duration_s") or 0)
        for key in ("id", "uri", "sha256", "duration_s", "fps"):
            if key not in reference:
                errors.append(f"reference missing {key}")
        if duration <= 0:
            errors.append("reference.duration_s must be > 0")
        if not _number(reference.get("fps")) or reference.get("fps", 0) <= 0:
            errors.append("reference.fps must be > 0")
        digest = reference.get("sha256")
        if not isinstance(digest, str) or (digest and len(digest) != 64):
            errors.append("reference.sha256 must be empty or a 64-char digest")

    intent = program.get("intent")
    if not isinstance(intent, dict):
        errors.append("intent must be an object")
    else:
        if not str(intent.get("topic") or "").strip():
            errors.append("intent.topic missing")
        if not str(intent.get("message") or "").strip():
            errors.append("intent.message missing")
        if intent.get("content_type") not in CONTENT_TYPES:
            errors.append("intent.content_type invalid")
        _validate_claim(intent, duration, "intent", errors)

    entities = program.get("entities")
    if not isinstance(entities, list):
        errors.append("entities must be a list")
        entities = []
    entity_ids: set[str] = set()
    for idx, entity in enumerate(entities):
        prefix = f"entities[{idx}]"
        if not isinstance(entity, dict):
            errors.append(f"{prefix} must be an object")
            continue
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            errors.append(f"{prefix}.id missing")
        elif entity_id in entity_ids:
            errors.append(f"duplicate entity id {entity_id}")
        else:
            entity_ids.add(entity_id)
        if entity.get("kind") not in ENTITY_KINDS:
            errors.append(f"{prefix}.kind invalid")
        if not str(entity.get("name_or_role") or "").strip():
            errors.append(f"{prefix}.name_or_role missing")
        if not isinstance(entity.get("aliases", []), list):
            errors.append(f"{prefix}.aliases must be a list")
        _validate_claim(entity, duration, prefix, errors)

    events = program.get("events")
    if not isinstance(events, list):
        errors.append("events must be a list")
        events = []
    event_ids: set[str] = set()
    event_times: dict[str, tuple[float, float]] = {}
    for idx, event in enumerate(events):
        prefix = f"events[{idx}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            continue
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"{prefix}.id missing")
        elif event_id in event_ids:
            errors.append(f"duplicate event id {event_id}")
        else:
            event_ids.add(event_id)
        _validate_interval(event.get("interval"), duration, f"{prefix}.interval", errors)
        if isinstance(event.get("interval"), list) and len(event["interval"]) == 2 \
                and all(_number(v) for v in event["interval"]):
            event_times[str(event_id)] = tuple(map(float, event["interval"]))
        participants = event.get("participants")
        if not isinstance(participants, list):
            errors.append(f"{prefix}.participants must be a list")
        else:
            for entity_id in participants:
                if entity_id not in entity_ids:
                    errors.append(f"{prefix}.participants unknown entity {entity_id}")
        if not str(event.get("action") or "").strip():
            errors.append(f"{prefix}.action missing")
        _validate_claim(event, duration, prefix, errors)

    links = program.get("causal_links")
    if not isinstance(links, list):
        errors.append("causal_links must be a list")
        links = []
    graph: dict[str, list[str]] = {event_id: [] for event_id in event_ids}
    for idx, link in enumerate(links):
        prefix = f"causal_links[{idx}]"
        if not isinstance(link, dict):
            errors.append(f"{prefix} must be an object")
            continue
        source, target = link.get("from_event"), link.get("to_event")
        if source not in event_ids:
            errors.append(f"{prefix}.from_event unknown event {source}")
        if target not in event_ids:
            errors.append(f"{prefix}.to_event unknown event {target}")
        if link.get("relation") not in CAUSAL_RELATIONS:
            errors.append(f"{prefix}.relation invalid")
        if source in graph and target in event_ids:
            graph[source].append(target)
            if event_times.get(source) and event_times.get(target) \
                    and event_times[source][0] > event_times[target][0]:
                errors.append(f"{prefix} backward causal link")
        _validate_claim(link, duration, prefix, errors)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            errors.append(f"causal graph cycle at {node}")
            return
        if node in visited:
            return
        visiting.add(node)
        for target in graph.get(node, []):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for event_id in graph:
        visit(event_id)

    arc = program.get("arc")
    if not isinstance(arc, list):
        errors.append("arc must be a list")
        arc = []
    for idx, segment in enumerate(arc):
        prefix = f"arc[{idx}]"
        if not isinstance(segment, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if segment.get("role") not in ARC_ROLES:
            errors.append(f"{prefix}.role invalid")
        ids = segment.get("event_ids")
        if not isinstance(ids, list) or not ids:
            errors.append(f"{prefix}.event_ids must be non-empty")
        elif any(event_id not in event_ids for event_id in ids):
            errors.append(f"{prefix}.event_ids references unknown event")

    utterances = program.get("utterances")
    if not isinstance(utterances, list):
        errors.append("utterances must be a list")
        utterances = []
    utterance_ids: set[str] = set()
    for idx, utterance in enumerate(utterances):
        prefix = f"utterances[{idx}]"
        if not isinstance(utterance, dict):
            errors.append(f"{prefix} must be an object")
            continue
        uid = utterance.get("id")
        if not isinstance(uid, str) or not uid or uid in utterance_ids:
            errors.append(f"{prefix}.id missing or duplicate")
        else:
            utterance_ids.add(uid)
        _validate_interval(utterance.get("interval"), duration, f"{prefix}.interval", errors)
        speaker = utterance.get("speaker_id")
        if speaker is not None and speaker not in entity_ids:
            errors.append(f"{prefix}.speaker_id unknown entity {speaker}")
        if not str(utterance.get("original") or "").strip():
            errors.append(f"{prefix}.original missing")
        _validate_claim(utterance, duration, prefix, errors)

    curve = program.get("emotion_curve")
    if not isinstance(curve, list):
        errors.append("emotion_curve must be a list")
        curve = []
    for idx, emotion in enumerate(curve):
        prefix = f"emotion_curve[{idx}]"
        if not isinstance(emotion, dict):
            errors.append(f"{prefix} must be an object")
            continue
        _validate_interval(emotion.get("interval"), duration, f"{prefix}.interval", errors)
        if not str(emotion.get("emotion") or "").strip():
            errors.append(f"{prefix}.emotion missing")
        intensity = emotion.get("intensity")
        if not _number(intensity) or not 0 <= intensity <= 1:
            errors.append(f"{prefix}.intensity outside [0,1]")
        _validate_claim(emotion, duration, prefix, errors)

    uncertainties = program.get("uncertainties")
    if not isinstance(uncertainties, list):
        errors.append("uncertainties must be a list")
    provenance = program.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for key in ("model", "prompt_version", "tool_calls", "seed"):
            if key not in provenance:
                errors.append(f"provenance missing {key}")
        if not isinstance(provenance.get("tool_calls"), list):
            errors.append("provenance.tool_calls must be a list")
    return errors


def narrative_hash(program: dict) -> str:
    payload = json.dumps(program, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_narrative_program(program: dict, path: Path) -> Path:
    errors = validate_narrative_program(program)
    if errors:
        raise ValueError("invalid Narrative Program: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
