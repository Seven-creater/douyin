"""Evidence-centric montage planning for V6.

The planner consumes independently verified EvidenceUnit records.  It never
expands an interval or pads the result with an unclassified shot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.evidence_units import EvidenceUnit, sort_units


ROLE_ORDER = ("hook", "conflict", "proof", "reversal", "peak", "payoff")


def _as_units(value: Any) -> list[EvidenceUnit]:
    rows = value.get("units") if isinstance(value, dict) else value
    result: list[EvidenceUnit] = []
    for row in rows or []:
        if isinstance(row, EvidenceUnit):
            result.append(row)
            continue
        if not isinstance(row, dict):
            continue
        interval = row.get("interval") or row.get("normalized_absolute_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        try:
            result.append(EvidenceUnit(
                id=str(row.get("id") or "unit"),
                interval=(float(interval[0]), float(interval[1])),
                unit_type=str(row.get("unit_type") or "micro_clip"),
                description=str(row.get("description") or row.get("moment") or ""),
                meaning=str(row.get("meaning") or ""),
                editing_role=str(row.get("editing_role") or row.get("role") or "proof"),
                editorial_value=min(1.0, max(0.0, float(row.get("editorial_value", 0.0)))),
                source_shot_id=str(row.get("source_shot_id") or ""),
                keyframe_path=row.get("keyframe_path"),
                confidence=min(1.0, max(0.0, float(row.get("confidence", 0.0)))),
                source_video=row.get("source_video"),
                metadata=dict(row.get("metadata") or {}),
            ))
        except (TypeError, ValueError):
            continue
    return sort_units(result)


def _role_candidates(units: list[EvidenceUnit], role: str, used: set[str]) -> list[EvidenceUnit]:
    return sorted((unit for unit in units
                   if unit.id not in used and unit.editing_role == role),
                  key=lambda unit: (-unit.editorial_value, -unit.confidence,
                                    unit.start_s, unit.id))


def build_evidence_edit_plan(units: Iterable[EvidenceUnit] | dict, pattern: dict | None = None,
                            *, preferred_duration_s: float = 12.0,
                            min_duration_s: float = 4.0,
                            max_duration_s: float = 20.0) -> dict[str, Any]:
    """Select short evidence units and return a deterministic montage plan."""
    all_units = _as_units(units)
    pattern = pattern or {}
    allowed = set(pattern.get("allowed_unit_types") or
                  {"micro_clip", "keyframe_hold", "dialogue_excerpt"})
    candidates = [unit for unit in all_units if unit.unit_type in allowed]
    selected: list[EvidenceUnit] = []
    used: set[str] = set()
    missing_roles: list[str] = []
    requested_roles = [str(row.get("role")) for row in pattern.get("beats") or []
                       if isinstance(row, dict) and row.get("role") in ROLE_ORDER]
    requested_roles = list(dict.fromkeys(requested_roles or ["hook", "proof", "payoff"]))
    for role in requested_roles:
        options = _role_candidates(candidates, role, used)
        if options:
            selected.append(options[0])
            used.add(options[0].id)
        elif role in {"hook", "proof", "payoff"}:
            missing_roles.append(role)
    # Fill a sparse role prior with the strongest remaining evidence.  This is
    # still a montage unit, never a full-shot fallback.
    remaining = sorted((unit for unit in candidates if unit.id not in used),
                       key=lambda unit: (-unit.editorial_value, -unit.confidence,
                                         unit.start_s, unit.id))
    for unit in remaining:
        if len(selected) >= 8:
            break
        current = sum(max(0.15, item.duration_s) for item in selected)
        projected = current + unit.duration_s
        if projected > max_duration_s + 1e-6:
            continue
        # Preferred duration is soft: do not pad a complete, role-diverse
        # montage merely to approach the number.  Continue only while minimum
        # duration or evidence-role coverage is still missing.
        role_count = len({item.editing_role for item in selected})
        if current >= min_duration_s and role_count >= 3 \
                and projected > preferred_duration_s + 1e-6:
            continue
        selected.append(unit)
        used.add(unit.id)
    selected = sorted(selected, key=lambda unit: (unit.start_s, unit.end_s, unit.id))
    segments = []
    cursor = 0.0
    for index, unit in enumerate(selected):
        duration = unit.duration_s
        if unit.unit_type == "keyframe_hold":
            duration = max(0.3, min(1.5, float(unit.metadata.get("hold_duration_s", duration))))
        segments.append({
            "index": index, "unit_id": unit.id, "editing_role": unit.editing_role,
            "unit_type": unit.unit_type, "source_video": unit.source_video,
            "source_interval": [round(unit.start_s, 3), round(unit.end_s, 3)],
            "target_interval": [round(cursor, 3), round(cursor + duration, 3)],
            "duration_s": round(duration, 3), "description": unit.description,
            "meaning": unit.meaning, "keyframe_path": unit.keyframe_path,
            "editorial_value": unit.editorial_value,
        })
        cursor += duration
    failure_class = None
    reasons: list[str] = []
    roles = {segment["editing_role"] for segment in segments}
    if len(segments) < 3:
        failure_class, reasons = "evidence_missing", ["fewer_than_three_evidence_units"]
    elif "hook" not in roles:
        failure_class, reasons = "evidence_missing", ["hook_missing"]
    elif len(roles - {"hook"}) < 2:
        failure_class, reasons = "evidence_missing", ["fewer_than_two_non_hook_roles"]
    elif cursor < min_duration_s:
        failure_class, reasons = "evidence_missing", ["montage_below_min_duration"]
    elif cursor > max_duration_s + 1e-6:
        failure_class, reasons = "budget", [f"montage_over_max_duration:{cursor:.3f}"]
    return {
        "schema_version": "evidence_edit_plan_v1", "pattern_type": pattern.get("pattern_type"),
        "preferred_duration_s": float(preferred_duration_s),
        "min_duration_s": float(min_duration_s), "max_duration_s": float(max_duration_s),
        "duration_s": round(cursor, 3), "segments": segments,
        "selected_unit_ids": [segment["unit_id"] for segment in segments],
        "missing_roles": missing_roles, "passed": failure_class is None,
        "failure_class": failure_class, "reasons": reasons,
    }


def write_evidence_edit_plan(plan: dict, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
