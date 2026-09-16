"""Stable dictionary boundary around the pinned NEWTON internal memory types."""
from __future__ import annotations

from typing import Any

from .memory import Attempt, Memory


class NewtonTraceAdapter:
    """Hide upstream class names from first-party V9-G code."""

    def __init__(self, objective: str) -> None:
        self._memory = Memory(question=objective)

    def add_generation(self, *, turn: int, prompt: str, video_path: str | None = None,
                       ref_images: list[str] | None = None,
                       issues: list[str] | None = None, summary: str | None = None,
                       passed: bool | None = None, error: str | None = None) -> None:
        self._memory.add(Attempt(
            turn=turn, kind="generate", video_prompt=prompt,
            ref_images=list(ref_images or []), video_path=video_path,
            rel_verdict={"issues": list(issues or []), "summary": summary},
            stop=passed, stop_reason="contract_passed" if passed else None,
            error=error))

    def history_text(self) -> str:
        return self._memory.for_condition_judge()

    def to_dict(self) -> dict[str, Any]:
        return self._memory.to_trace()
