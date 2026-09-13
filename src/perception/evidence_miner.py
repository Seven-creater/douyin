"""Omni-backed fine-grained visual evidence mining for V6.

Each call is scoped to one real shot. Raw responses and sampling metadata are
persisted alongside normalized absolute intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from src.agentic_video.evidence_units import EDITING_ROLES, EvidenceUnit, interval_contained
from src.template.schema import extract_json_block


EVIDENCE_MINING_PROMPT = """你是短视频剪辑素材分析员。只分析给定镜头中可见、可听的最小视觉瞬间，
不要总结整段剧情，也不要把整个镜头当成一个事件。请找出 0.15–2 秒内能够独立表达含义的证据单元，
例如：人物受挫、攻击、反击、胜利、反应、身份特征、能力展示、情绪峰值。
输出严格 JSON：
{"timebase":"relative","evidence_units":[
 {"interval":[0.0,0.8],"unit_type":"micro_clip|keyframe_hold|dialogue_excerpt",
  "description":"画面中发生的可验证瞬间","meaning":"它证明了什么",
  "editing_role":"hook|conflict|proof|reversal|peak|payoff|transition|reaction|context",
  "editorial_value":0.0,"confidence":0.0}
]}
时间必须是输入镜头内的相对秒数；关键帧可以把 interval 写成单个时间点或极短区间。
不要猜官方角色名，不要编造画面外信息。"""


def build_evidence_prompt(*, scope_start_s: float, scope_end_s: float,
                          shot_id: str, transcript: str = "", caption: str = "") -> str:
    context = []
    if transcript.strip():
        context.append(f"可听 ASR（只作辅助，不得替代画面）: {transcript.strip()}")
    if caption.strip():
        context.append(f"已有画面描述（只作辅助）: {caption.strip()}")
    details = "\n".join(context) if context else "无辅助文本。"
    return (f"{EVIDENCE_MINING_PROMPT}\n输入镜头 id={shot_id}，源片绝对范围 "
            f"{scope_start_s:.3f}–{scope_end_s:.3f}s.\n"
            f"{details}")


def _raw_interval(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _normalize_interval(raw: tuple[float, float], *, timebase: str,
                        input_clip: tuple[float, float],
                        scope: tuple[float, float]) -> tuple[float, float] | None:
    clip_start, clip_end = input_clip
    left, right = raw
    if right == left:
        right = left + 0.15
    if right <= left:
        return None
    if timebase == "relative":
        if left < -1e-6 or right > clip_end - clip_start + 1e-6:
            return None
        interval = (clip_start + left, clip_start + right)
    elif timebase == "absolute":
        interval = (left, right)
    else:
        return None
    return interval if interval_contained(interval, scope) else None


def parse_evidence_response(raw: str, *, input_clip_interval: Iterable[float],
                            scope_interval: Iterable[float], source_shot_id: str = "",
                            source_video: str | None = None,
                            min_unit_duration_s: float = 0.15,
                            max_unit_duration_s: float | None = None) \
        -> tuple[list[EvidenceUnit], dict[str, Any]]:
    """Parse and audit one Omni response; invalid rows are rejected."""
    clip = tuple(float(value) for value in input_clip_interval)
    scope = tuple(float(value) for value in scope_interval)
    block = extract_json_block(str(raw or ""))
    audit: dict[str, Any] = {"raw_timebase": None, "accepted": 0, "rejected": []}
    if not block:
        audit["error"] = "json_missing"
        return [], audit
    try:
        payload = json.loads(block)
    except (TypeError, ValueError):
        audit["error"] = "json_invalid"
        return [], audit
    if not isinstance(payload, dict):
        audit["error"] = "payload_not_object"
        return [], audit
    default_timebase = str(payload.get("timebase") or "relative").strip().lower()
    audit["raw_timebase"] = default_timebase
    rows = payload.get("evidence_units") or payload.get("visual_evidence") or []
    units: list[EvidenceUnit] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            audit["rejected"].append({"index": index, "reason": "row_not_object"})
            continue
        raw_value = row.get("interval") or row.get("time") or row.get("raw_interval")
        raw_value = raw_value if raw_value is not None else row.get("frame_time")
        raw_interval = _raw_interval(raw_value)
        if raw_interval is None:
            audit["rejected"].append({"index": index, "reason": "interval_invalid"})
            continue
        timebase = str(row.get("timebase") or default_timebase).strip().lower()
        normalized = _normalize_interval(raw_interval, timebase=timebase,
                                          input_clip=clip, scope=scope)
        if normalized is None:
            audit["rejected"].append({
                "index": index, "reason": "interval_outside_scope",
                "raw_interval": list(raw_interval), "timebase": timebase})
            continue
        unit_type = str(row.get("unit_type") or row.get("type") or "micro_clip").strip()
        if unit_type not in {"micro_clip", "keyframe_hold", "dialogue_excerpt"}:
            audit["rejected"].append({"index": index, "reason": "unit_type_invalid"})
            continue
        duration = normalized[1] - normalized[0]
        if (duration < min_unit_duration_s - 1e-6 or
                (max_unit_duration_s is not None and
                 duration > max_unit_duration_s + 1e-6)):
            audit["rejected"].append({
                "index": index, "reason": "evidence_duration_invalid",
                "duration_s": round(duration, 6)})
            continue
        role = str(row.get("editing_role") or row.get("role") or "proof").strip()
        if role not in EDITING_ROLES:
            role = "proof"
        try:
            value = min(1.0, max(0.0, float(
                row.get("editorial_value", row.get("strength", 0.0)))))
            confidence = min(1.0, max(0.0, float(row.get("confidence", 0.0))))
        except (TypeError, ValueError):
            value, confidence = 0.0, 0.0
        local_id = str(row.get("id") or f"ev_{index:03d}").strip()
        unit_id = f"{source_shot_id or 'shot'}:{local_id}"
        units.append(EvidenceUnit(
            id=unit_id,
            interval=normalized, unit_type=unit_type,
            description=str(row.get("description") or row.get("moment") or "").strip(),
            meaning=str(row.get("meaning") or "").strip(), editing_role=role,
            editorial_value=value, source_shot_id=source_shot_id,
            keyframe_path=str(row.get("keyframe_path") or "") or None,
            confidence=confidence, source_video=source_video,
            metadata={
                "raw_interval": list(raw_interval), "raw_timebase": timebase,
                "input_clip_interval": list(clip),
                "timebase_origin_s": clip[0] if timebase == "relative" else None,
                "normalized_absolute_interval": list(normalized),
            }))
    audit["accepted"] = len(units)
    return units, audit


def _shot_rows(shots: Any) -> list[dict[str, Any]]:
    if isinstance(shots, dict):
        shots = shots.get("shots") or (shots.get("output") or {}).get("shots") or []
    return [row for row in shots or [] if isinstance(row, dict)]


def split_analysis_intervals(start_s: float, end_s: float, *,
                             max_watch_s: float = 6.0,
                             overlap_s: float = 0.5) -> list[tuple[float, float]]:
    """Split a long continuous shot into small perception calls.

    A real 40-second oner must not become one coarse evidence request.  A small
    overlap keeps actions crossing a tile boundary visible to one complete
    call; duplicate evidence is removed after normalization.
    """
    start_s, end_s = float(start_s), float(end_s)
    if end_s <= start_s or max_watch_s <= 0 or overlap_s < 0 or overlap_s >= max_watch_s:
        raise ValueError("invalid analysis interval policy")
    if end_s - start_s <= max_watch_s:
        return [(start_s, end_s)]
    intervals = []
    cursor = start_s
    step = max_watch_s - overlap_s
    while cursor < end_s - 1e-6:
        right = min(end_s, cursor + max_watch_s)
        intervals.append((round(cursor, 6), round(right, 6)))
        if right >= end_s - 1e-6:
            break
        cursor += step
    return intervals


def _dedupe_units(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    kept: list[EvidenceUnit] = []
    for unit in sorted(units, key=lambda item: (-item.editorial_value,
                                                -item.confidence, item.start_s)):
        duplicate = any(
            unit.editing_role == old.editing_role and
            min(unit.end_s, old.end_s) - max(unit.start_s, old.start_s) >=
            0.5 * min(unit.duration_s, old.duration_s)
            for old in kept)
        if not duplicate:
            kept.append(unit)
    return sorted(kept, key=lambda item: (item.start_s, item.end_s, item.id))


def mine_evidence(video: Path, shots: Any, *, runner, scope_interval: Iterable[float],
                  output: Path | None = None, clip_dir: Path | None = None,
                  transcript_by_shot: dict[str, str] | None = None,
                  caption_by_shot: dict[str, str] | None = None,
                  allow_unverified: bool = False,
                  max_watch_s: float = 6.0,
                  overlap_s: float = 0.5,
                  min_unit_duration_s: float = 0.15,
                  max_unit_duration_s: float = 2.0) -> dict[str, Any]:
    """Run one Omni watch per scoped shot and persist raw responses plus units."""
    video = Path(video).resolve()
    scope = tuple(float(value) for value in scope_interval)
    if not video.exists():
        raise FileNotFoundError(video)
    scoped = []
    for index, row in enumerate(_shot_rows(shots)):
        start = float(row.get("start_s", row.get("start", 0.0)))
        end = float(row.get("end_s", row.get("end", 0.0)))
        if end > scope[0] and start < scope[1]:
            scoped.append((index, max(start, scope[0]), min(end, scope[1]), row))
    if not scoped:
        raise ValueError("no shots overlap the approved scope")
    if runner is None and not allow_unverified:
        raise RuntimeError("evidence mining requires an Omni runner")
    output = Path(output) if output is not None else None
    raw_dir = output.parent / f"{output.stem}_raw" if output else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
    transcript_by_shot = transcript_by_shot or {}
    caption_by_shot = caption_by_shot or {}
    units: list[EvidenceUnit] = []
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    analysis_rows = []
    for index, start, end, row in scoped:
        parent_id = str(row.get("id") or row.get("shot_id") or f"shot_{index:04d}")
        parts = split_analysis_intervals(
            start, end, max_watch_s=max_watch_s, overlap_s=overlap_s)
        for part, (left, right) in enumerate(parts):
            shot_id = parent_id if len(parts) == 1 else f"{parent_id}_part_{part:02d}"
            analysis_rows.append((parent_id, shot_id, left, right, row))

    prepared = []
    for parent_id, shot_id, start, end, row in analysis_rows:
        prompt = build_evidence_prompt(
            scope_start_s=start, scope_end_s=end, shot_id=shot_id,
            transcript=(transcript_by_shot.get(shot_id)
                        or transcript_by_shot.get(parent_id, "")),
            caption=(caption_by_shot.get(shot_id)
                     or caption_by_shot.get(parent_id, "")))
        call = {"shot_id": shot_id, "input_video": str(video),
                "input_clip_interval": [start, end], "prompt_type": "evidence_mining_v1"}
        watch_clip_dir = (clip_dir / shot_id if clip_dir else
                          output.parent / "omni_clips" / shot_id if output else None)
        prepared.append((shot_id, start, end, prompt, call, watch_clip_dir))

    answers = None
    if runner is not None and hasattr(runner, "watch_many"):
        try:
            answers = runner.watch_many([{
                "video_path": video, "prompt": prompt,
                "kwargs": {"start_s": start, "end_s": end,
                           "clip_dir": watch_clip_dir, "duration_s": end - start}
            } for _, start, end, prompt, _, watch_clip_dir in prepared])
        except Exception as exc:
            error = {"failure_class": "infrastructure",
                     "error": f"{type(exc).__name__}: {exc}",
                     "affected_calls": len(prepared)}
            errors.append(error)
            calls.extend(call for _, _, _, _, call, _ in prepared)

    for position, (shot_id, start, end, prompt, call, watch_clip_dir) in enumerate(prepared):
        if runner is None:
            errors.append({**call, "failure_class": "infrastructure",
                           "reason": "runner_missing"})
            calls.append(call)
            continue
        if answers is None and hasattr(runner, "watch_many"):
            continue
        try:
            answer = (answers[position] if answers is not None else runner.watch(
                video, prompt, start_s=start, end_s=end,
                clip_dir=watch_clip_dir, duration_s=end - start))
            raw = str(getattr(answer, "text", answer) or "")
            parsed, audit = parse_evidence_response(
                raw, input_clip_interval=(start, end), scope_interval=scope,
                source_shot_id=shot_id, source_video=str(video),
                min_unit_duration_s=min_unit_duration_s,
                max_unit_duration_s=max_unit_duration_s)
            units.extend(parsed)
            call.update({
                "raw_response_file": str(raw_dir / f"{shot_id}.txt") if raw_dir else None,
                "sampling": (getattr(answer, "sampling", None)
                             or getattr(runner, "_last_sampling", None)),
                "gpu_pair": getattr(answer, "gpu_pair", None), "audit": audit})
            if raw_dir:
                (raw_dir / f"{shot_id}.txt").write_text(raw, encoding="utf-8")
        except Exception as exc:
            call.update({"failure_class": "infrastructure",
                         "error": f"{type(exc).__name__}: {exc}"})
            errors.append(call)
        calls.append(call)
    units = _dedupe_units(units)
    payload = {
        "schema_version": "evidence_units_v1", "source_video": str(video),
        "scope_interval": list(scope), "analysis_policy": {
            "max_watch_s": float(max_watch_s), "overlap_s": float(overlap_s),
            "min_unit_duration_s": float(min_unit_duration_s),
            "max_unit_duration_s": float(max_unit_duration_s),
            "analysis_call_count": len(analysis_rows),
            "parallel_workers": len(getattr(runner, "gpu_pairs", [])) or 1},
        "units": [unit.to_dict() for unit in units],
        "calls": calls, "errors": errors,
        "passed": bool(units) and not errors,
        "failure_class": None if units and not errors else
        "infrastructure" if errors else "content",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mine V6 visual evidence units")
    ap.add_argument("--video", required=True)
    ap.add_argument("--shots", required=True, help="shots JSON")
    ap.add_argument("--scope-start", type=float, required=True)
    ap.add_argument("--scope-end", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--unverified", action="store_true")
    args = ap.parse_args(argv)
    from src.config import load_config
    from src.perception.omni_runner import OmniRunner
    cfg = load_config()
    runner = None if args.unverified else OmniRunner(
        cfg.perception.get("omni") or {}, ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    result = mine_evidence(
        Path(args.video), json.loads(Path(args.shots).read_text(encoding="utf-8")),
        runner=runner, scope_interval=(args.scope_start, args.scope_end),
        output=Path(args.output), allow_unverified=args.unverified)
    print(json.dumps({"status": "ok", "output": args.output,
                      "units": len(result["units"]), "passed": result["passed"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
