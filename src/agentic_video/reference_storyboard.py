"""Evidence-only shot storyboard for the P0-R2 reference track."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.agentic_video.manifest import json_hash

SHOT_STORYBOARD_VERSION = "shot_storyboard_v2"
AGENT_REFERENCE_VERSION = "agent_reference_v2"
IDENTITY_PREDICATES = {"same_entity_as", "different_entity_from"}


class ReferenceStoryboardError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass
class TransitionRef:
    transition_id: str
    transition_type: str | None
    start_s: float
    end_s: float
    duration_s: float
    from_shot_id: str | None
    to_shot_id: str | None


@dataclass
class ShotCard:
    shot_id: str
    section_id: str
    start_s: float
    end_s: float
    duration_s: float
    transition_in: TransitionRef | None = None
    transition_out: TransitionRef | None = None
    visual_claim_ids: list[str] = field(default_factory=list)
    text_claim_ids: list[str] = field(default_factory=list)
    audio_claim_ids: list[str] = field(default_factory=list)
    identity_claim_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    shot_size: str | None = None
    camera_angle: str | None = None
    camera_motion: str | None = None
    motion_speed: str | None = None
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interval(value: Any, *, reason_code: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ReferenceStoryboardError(reason_code)
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ReferenceStoryboardError(reason_code) from exc
    if start < 0 or end <= start:
        raise ReferenceStoryboardError(reason_code)
    return start, end


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _entity_root(value: Any) -> str | None:
    text = str(value or "")
    if not text.startswith("E"):
        return None
    return text.split(".", 1)[0]


def build_shot_storyboard(validated_reference: dict[str, Any]) -> dict[str, Any]:
    """Join accepted evidence to deterministic content-shot intervals.

    The function never derives narrative roles or editing functions. Missing
    cinematography observations remain explicit unresolved fields.
    """
    source_sha = str(validated_reference.get("source_sha") or "")
    if not source_sha:
        raise ReferenceStoryboardError("source_sha_missing")
    timeline = validated_reference.get("deterministic_timeline")
    if not isinstance(timeline, dict):
        raise ReferenceStoryboardError("timeline_missing")
    raw_segments = timeline.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ReferenceStoryboardError("timeline_segments_missing")

    ordered = sorted(
        raw_segments,
        key=lambda row: _interval(row.get("interval"), reason_code="segment_interval_invalid"),
    )
    transition_refs: dict[str, TransitionRef] = {}
    for index, segment in enumerate(ordered):
        if segment.get("segment_kind") != "transition":
            continue
        transition_id = str(segment.get("segment_id") or "")
        if not transition_id or transition_id in transition_refs:
            raise ReferenceStoryboardError("transition_id_invalid", transition_id)
        start, end = _interval(
            segment.get("interval"), reason_code="transition_interval_invalid")
        before = next((
            row for row in reversed(ordered[:index])
            if row.get("segment_kind") == "content"), None)
        after = next((
            row for row in ordered[index + 1:]
            if row.get("segment_kind") == "content"), None)
        transition_refs[transition_id] = TransitionRef(
            transition_id=transition_id,
            transition_type=(str(segment["transition_type"])
                             if segment.get("transition_type") else None),
            start_s=start,
            end_s=end,
            duration_s=round(end - start, 6),
            from_shot_id=(str(before.get("segment_id")) if before else None),
            to_shot_id=(str(after.get("segment_id")) if after else None),
        )
    claim_ids: set[str] = set()
    claims: list[tuple[dict[str, Any], tuple[float, float]]] = []
    for claim in validated_reference.get("accepted_claims") or []:
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id or claim_id in claim_ids:
            raise ReferenceStoryboardError("claim_id_invalid", claim_id)
        claim_ids.add(claim_id)
        claims.append((claim, _interval(
            claim.get("interval"), reason_code="claim_interval_invalid")))

    event_ids: set[str] = set()
    events: list[tuple[dict[str, Any], tuple[float, float]]] = []
    for event in validated_reference.get("accepted_events") or []:
        event_id = str(event.get("event_id") or "")
        if not event_id or event_id in event_ids:
            raise ReferenceStoryboardError("event_id_invalid", event_id)
        event_ids.add(event_id)
        events.append((event, _interval(
            event.get("interval"), reason_code="event_interval_invalid")))

    cards: list[dict[str, Any]] = []
    seen_shots: set[str] = set()
    for index, segment in enumerate(ordered):
        if segment.get("segment_kind") != "content":
            continue
        shot_id = str(segment.get("segment_id") or "")
        if not shot_id or shot_id in seen_shots:
            raise ReferenceStoryboardError("shot_id_invalid", shot_id)
        seen_shots.add(shot_id)
        start, end = _interval(
            segment.get("interval"), reason_code="shot_interval_invalid")
        shot_interval = (start, end)
        transition_in = None
        transition_out = None
        if index > 0 and ordered[index - 1].get("segment_kind") == "transition":
            transition_in = transition_refs[str(ordered[index - 1]["segment_id"])]
        if index + 1 < len(ordered) and ordered[index + 1].get("segment_kind") == "transition":
            transition_out = transition_refs[str(ordered[index + 1]["segment_id"])]

        by_modality: dict[str, list[str]] = {"V": [], "T": [], "A": []}
        identity_claim_ids: list[str] = []
        entities: set[str] = set()
        unresolved: list[str] = []
        for claim, interval in claims:
            if not _overlaps(shot_interval, interval):
                continue
            claim_id = str(claim["claim_id"])
            modality = str(claim.get("modality") or "")
            if claim.get("predicate") in IDENTITY_PREDICATES:
                identity_claim_ids.append(claim_id)
            elif modality in by_modality:
                by_modality[modality].append(claim_id)
            else:
                unresolved.append(f"unrouted_modality:{claim_id}:{modality}")
            for value in (claim.get("subject"), claim.get("object")):
                entity = _entity_root(value)
                if entity:
                    entities.add(entity)

        overlapping_events: list[str] = []
        for event, interval in events:
            if not _overlaps(shot_interval, interval):
                continue
            overlapping_events.append(str(event["event_id"]))
            for value in list(event.get("participants") or []) + list(
                    event.get("object_ids") or []):
                entity = _entity_root(value)
                if entity:
                    entities.add(entity)

        observed_fields = {
            key: segment.get(key)
            for key in ("shot_size", "camera_angle", "camera_motion", "motion_speed")
        }
        unresolved.extend(
            f"{key}_unobserved" for key, value in observed_fields.items()
            if value in (None, "")
        )
        card = ShotCard(
            shot_id=shot_id,
            section_id=str(segment.get("section_id") or ""),
            start_s=start,
            end_s=end,
            duration_s=round(end - start, 6),
            transition_in=transition_in,
            transition_out=transition_out,
            visual_claim_ids=sorted(by_modality["V"]),
            text_claim_ids=sorted(by_modality["T"]),
            audio_claim_ids=sorted(by_modality["A"]),
            identity_claim_ids=sorted(identity_claim_ids),
            event_ids=sorted(overlapping_events),
            entity_ids=sorted(entities),
            shot_size=observed_fields["shot_size"],
            camera_angle=observed_fields["camera_angle"],
            camera_motion=observed_fields["camera_motion"],
            motion_speed=observed_fields["motion_speed"],
            unresolved=sorted(unresolved),
        )
        cards.append(card.to_dict())

    if not cards:
        raise ReferenceStoryboardError("content_shots_missing")
    result = {
        "schema_version": SHOT_STORYBOARD_VERSION,
        "source_sha": source_sha,
        "timeline_schema_version": timeline.get("schema_version"),
        "shot_cards": cards,
    }
    result["artifact_sha"] = json_hash(result)
    return result


def build_agent_reference(validated_reference: dict[str, Any]) -> dict[str, Any]:
    """Publish the evidence-only R2 reference backbone."""
    storyboard = build_shot_storyboard(validated_reference)
    result = {
        "schema_version": AGENT_REFERENCE_VERSION,
        "source_sha": str(validated_reference.get("source_sha") or ""),
        "validated_reference_sha": str(validated_reference.get("artifact_sha") or ""),
        "shot_storyboard": storyboard,
        "claims": list(validated_reference.get("accepted_claims") or []),
        "events": list(validated_reference.get("accepted_events") or []),
        "deterministic_timeline": dict(
            validated_reference.get("deterministic_timeline") or {}),
        "coverage": dict(validated_reference.get("coverage") or {}),
        "unresolved_limitations": list(
            validated_reference.get("unresolved_limitations") or []),
    }
    result["artifact_sha"] = json_hash(result)
    return result
