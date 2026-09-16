"""Loop memory adapted from NEWTON at commit 072bda240a5991bc4da6ff8b6cd5f6f33fcfc058.

Upstream: https://github.com/CUTEPKQ/NEWTON/blob/072bda240a5991bc4da6ff8b6cd5f6f33fcfc058/loop/memory.py
License: MIT.  The field names remain compatible with the pinned JSON trace,
while V9-G accesses them only through its adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Attempt:
    """One planner/executor/verifier turn."""

    turn: int
    kind: str
    video_prompt: str = ""
    ref_video: Optional[str] = None
    ref_images: List[str] = field(default_factory=list)
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None
    ref_video_desc: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    condition_judge: Optional[Dict[str, Any]] = None
    condition_source_turn: Optional[int] = None
    video_path: Optional[str] = None
    archived_ref_video: Optional[str] = None
    archived_ref_images: List[str] = field(default_factory=list)
    archived_first_frame: Optional[str] = None
    archived_last_frame: Optional[str] = None
    rel_verdict: Optional[Dict[str, Any]] = None
    rel_score: Optional[int] = None
    stop: Optional[bool] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None


class Memory:
    """Single source of truth for a run: baseline plus ordered attempts."""

    def __init__(self, question: str) -> None:
        self.question = question
        self.baseline_video: Optional[str] = None
        self.attempts: List[Attempt] = []

    def set_baseline(self, video: Optional[str]) -> None:
        self.baseline_video = video

    def add(self, attempt: Attempt) -> Attempt:
        self.attempts.append(attempt)
        return attempt

    def for_condition_judge(self) -> str:
        lines: List[str] = [
            "BASELINE: a text-only video of the scenario is the bar to beat. "
            "A new condition is only worth generating if it is likely to improve it."
        ]
        for attempt in self.attempts:
            if attempt.kind == "condition":
                judge = attempt.condition_judge or {}
                if judge.get("reasonable") is False:
                    suggestions = ", ".join(judge.get("suggestions") or [])
                    lines.append(
                        f"- turn {attempt.turn}: condition REJECTED. "
                        f"Prompt: {(attempt.video_prompt or '')[:160]}. "
                        f"Reason: {judge.get('reason', '')}."
                        + (f" Suggestions: {suggestions}." if suggestions else ""))
            elif attempt.kind == "generate":
                if attempt.error:
                    lines.append(
                        f"- turn {attempt.turn}: generation FAILED ({attempt.error}).")
                    continue
                verdict = attempt.rel_verdict or {}
                problems = "; ".join(filter(None, [
                    verdict.get("sa_note", ""), verdict.get("pc_note", ""),
                    ", ".join(verdict.get("issues") or []),
                ]))
                lines.append(
                    f"- turn {attempt.turn}: GENERATED. "
                    f"Prompt: {(attempt.video_prompt or '')[:160]}. "
                    + (f"Remaining problems: {problems}." if problems else ""))
        return (
            "History of this run so far (avoid repeating failed conditions):\n"
            + "\n".join(lines))

    @staticmethod
    def _attempt_trace(attempt: Attempt) -> Dict[str, Any]:
        record: Dict[str, Any] = {"turn": attempt.turn, "kind": attempt.kind}
        if attempt.error:
            record["error"] = attempt.error
        record["planner"] = {
            "tools_called": [call.get("tool") for call in attempt.tool_calls],
            "tool_calls": attempt.tool_calls,
            "video_prompt": attempt.video_prompt,
        }
        record["condition"] = {
            "ref_video": attempt.ref_video,
            "ref_video_desc": attempt.ref_video_desc,
            "ref_images": attempt.ref_images,
            "first_frame": attempt.first_frame,
            "last_frame": attempt.last_frame,
            "archived_ref_video": attempt.archived_ref_video,
            "archived_ref_images": attempt.archived_ref_images,
            "archived_first_frame": attempt.archived_first_frame,
            "archived_last_frame": attempt.archived_last_frame,
        }
        if attempt.kind == "condition":
            record["verifier"] = {"pre_check": attempt.condition_judge}
        elif attempt.kind == "generate":
            record["generation"] = {
                "condition_source_turn": attempt.condition_source_turn,
                "video_path": attempt.video_path,
            }
            verdict = attempt.rel_verdict or {}
            record["verifier"] = {
                "gemini_rel": {
                    "score": attempt.rel_score,
                    "sa_note": verdict.get("sa_note"),
                    "pc_note": verdict.get("pc_note"),
                    "issues": verdict.get("issues"),
                    "summary": verdict.get("summary"),
                    "order": verdict.get("_order"),
                },
                "stop": attempt.stop,
                "stop_reason": attempt.stop_reason,
            }
        return record

    def to_trace(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "baseline_video": self.baseline_video,
            "attempts": [self._attempt_trace(attempt) for attempt in self.attempts],
        }
