"""Extract a compact editing pattern from a reference short video.

The first implementation is intentionally conservative: deterministic shot
statistics always exist, while an Omni response can add semantic roles.  The
result describes *editing grammar* (hook/proof/reversal/payoff), never the
reference video's characters or plot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.evidence_units import EDITING_ROLES
from src.perception import common
from src.perception.detect_shots import detect_shots
from src.template.schema import extract_json_block


REFERENCE_PATTERN_PROMPT = """你是短视频剪辑模式分析员。只分析剪辑语言，不复述人物故事。
请观看参考视频，输出一个 JSON 对象：
{
  \"pattern_type\": \"proof_montage|narrative_arc|contrast|explanation\",
  \"beats\": [{\"role\": \"hook|conflict|proof|reversal|peak|payoff|transition\",\"start_s\":0.0,\"end_s\":1.0,\"duration_max_s\":1.0}],
  \"transition\": \"hard_cut|dissolve|mixed\",
  \"allowed_unit_types\": [\"micro_clip\",\"keyframe_hold\"],
  \"notes\": \"只描述节奏、镜头长度、证据堆叠方式\"
}
时间必须是视频内相对秒数；无法确定的字段使用空数组或 \"unknown\"，不要编造。"""


def _default_beats(duration_s: float, shot_lengths: list[float]) -> list[dict[str, Any]]:
    """Return a useful pattern prior without a model call."""
    if duration_s <= 0:
        return []
    if len(shot_lengths) >= 3 and sum(shot_lengths) / len(shot_lengths) <= 3.0:
        return [
            {"role": "hook", "duration_max_s": min(2.0, duration_s)},
            {"role": "conflict", "duration_max_s": 1.5},
            {"role": "proof", "duration_max_s": 1.5},
            {"role": "reversal", "duration_max_s": 2.0},
            {"role": "payoff", "duration_max_s": 2.0},
        ]
    return [
        {"role": "hook", "duration_max_s": min(2.0, duration_s)},
        {"role": "context", "duration_max_s": 3.0},
        {"role": "peak", "duration_max_s": 3.0},
        {"role": "payoff", "duration_max_s": 2.0},
    ]


def _parse_model_pattern(raw: str, duration_s: float) -> dict[str, Any] | None:
    block = extract_json_block(str(raw or ""))
    if not block:
        return None
    try:
        value = json.loads(block)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    beats = []
    for row in value.get("beats") or []:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip()
        if role not in EDITING_ROLES:
            continue
        clean = {"role": role}
        for key in ("start_s", "end_s", "duration_max_s"):
            if row.get(key) is not None:
                try:
                    clean[key] = max(0.0, min(float(row[key]), duration_s))
                except (TypeError, ValueError):
                    pass
        beats.append(clean)
    value["pattern_type"] = str(value.get("pattern_type") or "proof_montage")
    value["beats"] = beats
    value["transition"] = str(value.get("transition") or "hard_cut")
    value["allowed_unit_types"] = [
        unit for unit in (value.get("allowed_unit_types") or ["micro_clip", "keyframe_hold"])
        if unit in {"micro_clip", "keyframe_hold", "dialogue_excerpt"}
    ]
    return value


def extract_reference_pattern(video: Path, *, runner=None, output: Path | None = None,
                              ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe",
                              threshold: float = 0.1,
                              min_shot_len_s: float = 0.4) -> dict[str, Any]:
    """Extract editing grammar and persist it when ``output`` is supplied."""
    video = Path(video).resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    duration_s = common.video_duration_s(ffprobe_bin, video)
    shots = detect_shots(video, ffmpeg_bin=ffmpeg_bin, threshold=threshold,
                         min_shot_len_s=min_shot_len_s, duration_s=duration_s)
    lengths = [float(row["duration_s"]) for row in shots.get("shots") or []]
    pattern = {
        "pattern_version": "v6_pattern_v1",
        "source_video": str(video),
        "source_duration_s": round(duration_s, 3),
        "shot_count": int(shots.get("shot_count") or 0),
        "shot_durations_s": [round(value, 3) for value in lengths],
        "pattern_type": "proof_montage" if len(lengths) >= 3 and
        (sum(lengths) / len(lengths) <= 3.0) else "narrative_arc",
        "beats": _default_beats(duration_s, lengths),
        "transition": "hard_cut",
        "allowed_unit_types": ["micro_clip", "keyframe_hold"],
        "preferred_cut_duration_s": [0.2, 2.0],
        "source_shots": shots.get("shots") or [],
    }
    if runner is not None:
        try:
            answer = runner.watch(video, REFERENCE_PATTERN_PROMPT,
                                  duration_s=duration_s, max_new_tokens=768)
            raw = str(getattr(answer, "text", answer) or "")
            parsed = _parse_model_pattern(raw, duration_s)
            if parsed:
                pattern.update(parsed)
                pattern["model_enriched"] = True
            pattern["model_sampling"] = getattr(answer, "sampling", None)
            pattern["model_gpu_pair"] = getattr(answer, "gpu_pair", None)
            if output is not None:
                raw_path = Path(output).with_name(f"{Path(output).stem}_raw.txt")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(raw, encoding="utf-8")
                pattern["model_raw_response_file"] = str(raw_path)
        except Exception as exc:  # model enrichment is optional; keep deterministic prior
            pattern["model_enrichment_error"] = f"{type(exc).__name__}: {exc}"
    else:
        pattern["model_enriched"] = False
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(pattern, ensure_ascii=False, indent=2), encoding="utf-8")
    return pattern


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract V6 editing pattern from a reference video")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    pattern = extract_reference_pattern(Path(args.reference), output=Path(args.output))
    print(json.dumps({"status": "ok", "output": args.output,
                      "pattern_type": pattern["pattern_type"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
