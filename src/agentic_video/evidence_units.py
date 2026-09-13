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


def interval_contained(interval: Iterable[float], scope: Iterable[float]) -> bool:
    values = list(interval)
    bounds = list(scope)
    return (len(values) == 2 and len(bounds) == 2 and
            float(bounds[0]) - 1e-6 <= float(values[0]) < float(values[1]) <=
            float(bounds[1]) + 1e-6)


def sort_units(units: Iterable[EvidenceUnit]) -> list[EvidenceUnit]:
    """Stable source order, with editorial value as a tie breaker only."""
    return sorted(units, key=lambda unit: (unit.start_s, unit.end_s, unit.id))

