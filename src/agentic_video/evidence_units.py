"""Fine-grained visual evidence units used by the V6 editing path.

The V5 planner reasons about complete events and utterances.  V6 keeps those
records intact but adds a smaller, auditable unit that can be cut into a
montage.  This module is deliberately dependency-free so parsers and planners
can be tested without a video model or FFmpeg.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


UNIT_TYPES = {"micro_clip", "keyframe_hold", "dialogue_excerpt"}
SOURCE_FORMS = {
    "visual_instant", "dynamic_action", "reaction", "dialogue_span",
    "establishing_visual",
}
EDITING_ROLES = {
    "hook", "conflict", "proof", "reversal", "peak", "payoff", "transition",
    "reaction", "context",
}


@dataclass(frozen=True)
class EvidenceUnit:
    """A minimal source interval with a visual/editorial meaning."""

    id: str
    interval: tuple[float, float]
    unit_type: str
    description: str
    meaning: str = ""
    editing_role: str = "proof"
    editorial_value: float = 0.0
    source_shot_id: str = ""
    keyframe_path: str | None = None
    confidence: float = 0.0
    source_video: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start, end = self.interval
        if end <= start:
            raise ValueError("evidence interval must have positive duration")
        if self.unit_type not in UNIT_TYPES:
            raise ValueError(f"unsupported evidence unit_type: {self.unit_type}")
        if self.editing_role not in EDITING_ROLES:
            raise ValueError(f"unsupported evidence editing_role: {self.editing_role}")
        if not 0.0 <= float(self.editorial_value) <= 1.0:
            raise ValueError("editorial_value must be in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def start_s(self) -> float:
        return float(self.interval[0])

    @property
    def end_s(self) -> float:
        return float(self.interval[1])

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["interval"] = [round(self.start_s, 3), round(self.end_s, 3)]
        payload["duration_s"] = round(self.duration_s, 3)
        return payload


@dataclass(frozen=True)
class EvidenceUnitV2:
    """Pattern-independent observation used by the V6.1 diagnostic path.

    ``core_interval`` says where the observable fact occurs.  It deliberately
    does not decide which editorial slot the fact should fill or how it should
    be rendered; those decisions belong to the matcher and planner.
    """

    id: str
    core_interval: tuple[float, float]
    container_interval: tuple[float, float]
    observation: dict[str, Any]
    source_form: str
    attributes: tuple[str, ...] = ()
    visual_strength: float = 0.0
    confidence: float = 0.0
    subject_track_id: str | None = None
    source_video: str | None = None
    sampling_provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        core_start, core_end = self.core_interval
        container_start, container_end = self.container_interval
        if core_end <= core_start:
            raise ValueError("evidence core_interval must have positive duration")
        if container_end <= container_start:
            raise ValueError("evidence container_interval must have positive duration")
        if core_start < container_start - 1e-6 or core_end > container_end + 1e-6:
            raise ValueError("evidence core_interval must be contained in its container")
        if self.source_form not in SOURCE_FORMS:
            raise ValueError(f"unsupported evidence source_form: {self.source_form}")
        if not 0.0 <= float(self.visual_strength) <= 1.0:
            raise ValueError("visual_strength must be in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.observation, dict):
            raise ValueError("observation must be an object")

    @property
    def start_s(self) -> float:
        return float(self.core_interval[0])

    @property
    def end_s(self) -> float:
        return float(self.core_interval[1])

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["core_interval"] = [round(self.start_s, 3), round(self.end_s, 3)]
        payload["container_interval"] = [
            round(float(self.container_interval[0]), 3),
            round(float(self.container_interval[1]), 3),
        ]
        payload["attributes"] = list(self.attributes)
        payload["core_duration_s"] = round(self.duration_s, 3)
        return payload


def evidence_v2_from_dict(row: dict[str, Any]) -> EvidenceUnitV2:
    """Read V6.1 observations and losslessly adapt historical V6 rows."""
    if row.get("core_interval") is not None:
        core = row["core_interval"]
        container = row.get("container_interval") or core
        return EvidenceUnitV2(
            id=str(row.get("id") or "evidence"),
            core_interval=(float(core[0]), float(core[1])),
            container_interval=(float(container[0]), float(container[1])),
            observation=dict(row.get("observation") or {}),
            source_form=str(row.get("source_form") or "dynamic_action"),
            attributes=tuple(str(value) for value in row.get("attributes") or []),
            visual_strength=min(1.0, max(0.0, float(row.get("visual_strength", 0.0)))),
            confidence=min(1.0, max(0.0, float(row.get("confidence", 0.0)))),
            subject_track_id=(str(row["subject_track_id"])
                              if row.get("subject_track_id") else None),
            source_video=row.get("source_video"),
            sampling_provenance=dict(row.get("sampling_provenance") or {}),
            metadata=dict(row.get("metadata") or {}),
        )
    interval = row.get("interval") or row.get("normalized_absolute_interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("historical evidence row has no valid interval")
    legacy_type = str(row.get("unit_type") or "micro_clip")
    form = {
        "keyframe_hold": "visual_instant",
        "dialogue_excerpt": "dialogue_span",
        "micro_clip": "dynamic_action",
    }.get(legacy_type, "dynamic_action")
    description = str(row.get("description") or row.get("meaning") or "").strip()
    metadata = dict(row.get("metadata") or {})
    metadata["legacy_v6"] = {
        "unit_type": legacy_type,
        "editing_role": row.get("editing_role") or row.get("role"),
    }
    return EvidenceUnitV2(
        id=str(row.get("id") or "evidence"),
        core_interval=(float(interval[0]), float(interval[1])),
        container_interval=(float(interval[0]), float(interval[1])),
        observation={"action": description, "subject_call_id": "unknown"},
        source_form=form,
        attributes=(),
        visual_strength=min(1.0, max(0.0, float(row.get("editorial_value", 0.0)))),
        confidence=min(1.0, max(0.0, float(row.get("confidence", 0.0)))),
        subject_track_id=None,
        source_video=row.get("source_video"),
        sampling_provenance={},
        metadata=metadata,
    )


def interval_contained(interval: Iterable[float], scope: Iterable[float]) -> bool:
    values = list(interval)
    bounds = list(scope)
    return (len(values) == 2 and len(bounds) == 2 and
            float(bounds[0]) - 1e-6 <= float(values[0]) < float(values[1]) <=
            float(bounds[1]) + 1e-6)


def sort_units(units: Iterable[EvidenceUnit]) -> list[EvidenceUnit]:
    """Stable source order, with editorial value as a tie breaker only."""
    return sorted(units, key=lambda unit: (unit.start_s, unit.end_s, unit.id))
