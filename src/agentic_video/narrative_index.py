"""Two-pass narrative indexing for dialogue, action, emotion, and context windows."""
from __future__ import annotations

import json
from pathlib import Path

from src.agentic_video.long_video import (index_selected_windows, resolve_source_video,
                                           robust_normalize, scan_sparse_features)
from src.agentic_video.narrative import ARC_ROLES
from src.template.schema import extract_json_block

DEFAULT_QUOTAS = {"dialogue": 12, "action": 8, "emotion": 8, "context": 8}


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / max(1, len(rows))


def _dialogue_density(segments: list[dict], start: float, end: float) -> float:
    """Estimate spoken-content density without trusting gapless ASR intervals.

    SenseVoice sentence timestamps can tile the full source timeline, including
    BGM/no-speech spans represented by punctuation-only text.  Weighting useful
    characters by temporal overlap keeps those spans from looking like dialogue.
    """
    weighted_characters = 0.0
    for segment in segments:
        left = float(segment.get("start_ms") or 0) / 1000
        right = float(segment.get("end_ms") or 0) / 1000
        overlap = max(0.0, min(end, right) - max(start, left))
        segment_duration = right - left
        if overlap <= 0 or segment_duration <= 0:
            continue
        useful_characters = sum(char.isalnum() for char in str(segment.get("text") or ""))
        weighted_characters += useful_characters * overlap / segment_duration
    return weighted_characters / max(1e-6, end - start)


def score_narrative_windows(samples: list[dict], transcript: dict, duration_s: float, *,
                            window_s: float = 45.0,
                            stride_s: float = 15.0) -> list[dict]:
    windows = []
    start = 0.0
    segments = transcript.get("segments") or []
    while start < duration_s:
        end = min(duration_s, start + window_s)
        rows = [row for row in samples if start <= float(row["t_s"]) < end]
        if rows:
            windows.append({
                "start_s": round(start, 3), "end_s": round(end, 3),
                "motion": _mean(rows, "motion"),
                "cut_density": _mean(rows, "cut_density") or _mean(rows, "cut"),
                "audio_energy": _mean(rows, "audio_energy"),
                "audio_onset": _mean(rows, "audio_onset"),
                "dialogue_raw": _dialogue_density(segments, start, end),
            })
        if end >= duration_s:
            break
        start += stride_s
    for key in ("motion", "cut_density", "audio_energy", "audio_onset", "dialogue_raw"):
        values = robust_normalize([float(row[key]) for row in windows])
        for row, value in zip(windows, values):
            row[f"{key}_norm"] = value
    for row in windows:
        row["action_score"] = round(
            0.35 * row["motion_norm"] + 0.25 * row["cut_density_norm"]
            + 0.20 * row["audio_energy_norm"] + 0.20 * row["audio_onset_norm"], 6)
        row["dialogue_score"] = round(
            0.75 * row["dialogue_raw_norm"]
            + 0.15 * row["audio_energy_norm"]
            + 0.10 * (1.0 - row["motion_norm"]), 6)
        mid_motion = 1.0 - abs(row["motion_norm"] - 0.35)
        row["emotion_score"] = round(
            0.35 * row["audio_energy_norm"] + 0.25 * row["audio_onset_norm"]
            + 0.25 * row["dialogue_raw_norm"] + 0.15 * mid_motion, 6)
    peaks = sorted(windows, key=lambda row: max(row["action_score"], row["emotion_score"]),
                   reverse=True)[:max(1, min(12, len(windows)))]
    for row in windows:
        center = (row["start_s"] + row["end_s"]) / 2
        distances = [abs(center - (peak["start_s"] + peak["end_s"]) / 2)
                     for peak in peaks if peak is not row]
        nearest = min(distances, default=duration_s)
        adjacency = max(0.0, 1.0 - nearest / max(window_s * 2, 1.0))
        row["context_score"] = round(
            0.55 * adjacency + 0.25 * (1.0 - row["motion_norm"])
            + 0.20 * row["dialogue_raw_norm"], 6)
    return windows


def _overlap_ratio(left: dict, right: dict) -> float:
    overlap = max(0.0, min(float(left["end_s"]), float(right["end_s"]))
                  - max(float(left["start_s"]), float(right["start_s"])))
    shorter = min(float(left["end_s"]) - float(left["start_s"]),
                  float(right["end_s"]) - float(right["start_s"]))
    return overlap / shorter if shorter > 0 else 0.0


def select_narrative_windows(windows: list[dict], duration_s: float, *,
                             quotas: dict[str, int] | None = None,
                             max_overlap: float = 0.35,
                             partitions: int = 6) -> list[dict]:
    quotas = dict(quotas or DEFAULT_QUOTAS)
    selected: list[dict] = []
    per_partition = max(1, (sum(quotas.values()) + partitions - 1) // partitions)
    partition_counts = [0] * partitions
    for selection_type, quota in quotas.items():
        score_key = f"{selection_type}_score"
        taken = 0
        for row in sorted(windows, key=lambda item: (-float(item.get(score_key, 0)),
                                                      float(item["start_s"]))):
            if taken >= quota:
                break
            midpoint = (float(row["start_s"]) + float(row["end_s"])) / 2
            partition = min(partitions - 1,
                            int(midpoint / max(duration_s, 1) * partitions))
            if partition_counts[partition] >= per_partition:
                continue
            if any(_overlap_ratio(row, prior) > max_overlap for prior in selected):
                continue
            chosen = dict(row)
            chosen["selection_type"] = selection_type
            chosen["selection_score"] = float(row.get(score_key, 0))
            chosen["partition"] = partition
            selected.append(chosen)
            partition_counts[partition] += 1
            taken += 1
    selected.sort(key=lambda row: float(row["start_s"]))
    for idx, row in enumerate(selected):
        row["window_idx"] = idx
    return selected


def run_source_transcript(cfg, source: str, video: Path, *, force: bool = False,
                          model=None) -> dict:
    output_dir = cfg.paths.library_dir / "sources" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "narrative_transcript.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    from src.perception.transcribe_audio import (load_transcriber,
                                                  transcribe_with_model)
    transcribe_cfg = cfg.perception.get("transcribe") or {}
    if model is None:
        model = load_transcriber(
            model=transcribe_cfg.get("model", "iic/SenseVoiceSmall"),
            vad_model=transcribe_cfg.get("vad_model", "fsmn-vad"),
            vad_max_segment_ms=int(transcribe_cfg.get("vad_max_single_segment_ms", 30000)),
            device=transcribe_cfg.get("device", "cuda:0"))
    result = transcribe_with_model(model, video, language="ja")
    result["language"] = "ja"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def attach_transcript_to_shots(result_path: Path, transcript: dict) -> Path:
    envelope = json.loads(Path(result_path).read_text(encoding="utf-8"))
    for shot in envelope["output"]["shots"]:
        dialogue = []
        for segment in transcript.get("segments") or []:
            start = float(segment.get("start_ms") or 0) / 1000
            end = float(segment.get("end_ms") or 0) / 1000
            if end <= float(shot["start_s"]) or start >= float(shot["end_s"]):
                continue
            dialogue.append({
                "start_s": round(max(start, float(shot["start_s"])), 3),
                "end_s": round(min(end, float(shot["end_s"])), 3),
                "original": str(segment.get("text") or ""),
                "translation_zh": "uncertain", "confidence": 0.7,
            })
        shot["dialogue"] = dialogue
        shot["dialogue_score"] = max(float(shot.get("dialogue_score", 0)),
                                       min(1.0, sum(d["end_s"] - d["start_s"]
                                                    for d in dialogue)
                                           / max(float(shot["duration_s"]), 0.001)))
    Path(result_path).write_text(json.dumps(envelope, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return Path(result_path)


WINDOW_ANNOTATION_PROMPT = """你是电影叙事素材标注 Agent。观看 {start:g}~{end:g} 秒片段，
根据给定镜头编号、时间、三帧描述和日语 ASR，为每个镜头标注可见内容。只输出 JSON：
{{"shots":[{{"shot_idx":0,"entity_ids":["稳定、简短的角色ID"],
"entity_names":["角色名或可见身份"],"event_id":"window事件ID",
"event_summary":"主体做了什么并造成什么变化","story_role":"hook|context|conflict|choice|climax|consequence|resolution",
"emotion":"情绪或uncertain","focus_x":0.5,"dialogue":[{{"start_s":0.0,"end_s":1.0,"original":"日语原文",
"translation_zh":"忠实中文字幕","confidence":0.0}}],"confidence":0.0}}],
"causal_links":[{{"from_event":"...","to_event":"...","relation":"causes|motivates|enables|prevents|reveals"}}]}}
不得根据 IP 常识补写镜头外剧情；同一人物跨镜头使用相同 entity_id；没有可靠对白就保留空数组。
只有当本窗画面特征支持时才能复用已有 entity_id；无法确认就新建可见身份 ID，不得强行合并。
本窗新事件 ID 必须以 {event_prefix} 开头。
【已有实体注册表】{registry}
【镜头材料】{material}
"""


def build_entity_registry(shots: dict) -> list[dict]:
    """Build a compact deterministic identity memory from completed windows."""
    registry: dict[str, dict] = {}
    for shot_idx, annotation in sorted(
            shots.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9):
        if not isinstance(annotation, dict):
            continue
        ids = [str(value) for value in annotation.get("entity_ids") or []]
        names = [str(value) for value in annotation.get("entity_names") or []]
        for idx, entity_id in enumerate(ids):
            if not entity_id:
                continue
            entry = registry.setdefault(entity_id, {
                "entity_id": entity_id, "visible_names": [], "shot_idxs": []})
            name = names[idx] if idx < len(names) else ""
            if name and name not in entry["visible_names"]:
                entry["visible_names"].append(name)
            entry["shot_idxs"].append(int(shot_idx) if str(shot_idx).isdigit() else shot_idx)
    return list(registry.values())


def namespace_window_events(parsed: dict, window_idx: int) -> dict:
    """Prevent unrelated windows from accidentally reusing generic event ids."""
    prefix = f"w{window_idx:03d}_"
    mapping = {}
    for annotation in (parsed.get("shots") or {}).values():
        event_id = str(annotation.get("event_id") or "event")
        namespaced = event_id if event_id.startswith(prefix) else prefix + event_id
        mapping[event_id] = namespaced
        annotation["event_id"] = namespaced
    for link in parsed.get("causal_links") or []:
        for key in ("from_event", "to_event"):
            if link.get(key) in mapping:
                link[key] = mapping[link[key]]
    return parsed


def parse_window_annotations(raw: str, *, valid_shot_ids: set[int]) -> dict:
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else {}
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    rows = {}
    for item in payload.get("shots") or []:
        if not isinstance(item, dict):
            continue
        try:
            shot_idx = int(item.get("shot_idx"))
        except (TypeError, ValueError):
            continue
        if shot_idx not in valid_shot_ids:
            continue
        role = item.get("story_role") if item.get("story_role") in ARC_ROLES else "context"
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        dialogue = []
        for line in item.get("dialogue") or []:
            if not isinstance(line, dict):
                continue
            try:
                start, end = float(line.get("start_s")), float(line.get("end_s"))
                line_confidence = min(1.0, max(0.0, float(line.get("confidence", 0.5))))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            dialogue.append({
                "start_s": round(start, 3), "end_s": round(end, 3),
                "original": str(line.get("original") or ""),
                "translation_zh": str(line.get("translation_zh") or "uncertain"),
                "confidence": line_confidence,
            })
        try:
            focus_x = min(1.0, max(0.0, float(item.get("focus_x", 0.5))))
        except (TypeError, ValueError):
            focus_x = 0.5
        rows[str(shot_idx)] = {
            "entity_ids": [str(value) for value in item.get("entity_ids") or []],
            "entity_names": [str(value) for value in item.get("entity_names") or []],
            "event_id": str(item.get("event_id") or f"event_shot_{shot_idx:05d}"),
            "event_summary": str(item.get("event_summary") or "uncertain"),
            "story_role": role, "emotion": str(item.get("emotion") or "uncertain"),
            "focus_x": focus_x, "dialogue": dialogue,
            "annotation_confidence": confidence,
        }
    links = [link for link in payload.get("causal_links") or [] if isinstance(link, dict)]
    return {"shots": rows, "causal_links": links,
            "parse_status": "json" if block else "failed"}


def run_narrative_annotations(cfg, result_path: Path, *, force: bool = False,
                              runner=None, limit_windows: int | None = None) -> Path:
    result_path = Path(result_path)
    output_path = result_path.parent / "narrative_annotations.json"
    saved = {"shots": {}, "causal_links": [], "completed_windows": []}
    if output_path.exists() and not force:
        saved = json.loads(output_path.read_text(encoding="utf-8"))
    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    shots = envelope["output"]["shots"]
    video = Path(envelope["output"]["video"])
    captions_path = result_path.parent / "captions.json"
    captions = (json.loads(captions_path.read_text(encoding="utf-8"))
                if captions_path.exists() else {})
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
    by_window: dict[int, list[dict]] = {}
    for shot in shots:
        by_window.setdefault(int(shot["window_idx"]), []).append(shot)
    processed = 0
    for window_idx, window_shots in sorted(by_window.items()):
        if window_idx in saved.get("completed_windows", []) and not force:
            continue
        if limit_windows is not None and processed >= limit_windows:
            break
        start = min(float(row["start_s"]) for row in window_shots)
        end = max(float(row["end_s"]) for row in window_shots)
        material = [{
            "shot_idx": row["shot_idx"], "start_s": row["start_s"],
            "end_s": row["end_s"], "caption": captions.get(str(row["shot_idx"]), ""),
            "dialogue_asr": row.get("dialogue") or [],
        } for row in window_shots]
        prompt = WINDOW_ANNOTATION_PROMPT.format(
            start=start, end=end, event_prefix=f"w{window_idx:03d}_",
            registry=json.dumps(build_entity_registry(saved.get("shots") or {}),
                                ensure_ascii=False)[:8000],
            material=json.dumps(material, ensure_ascii=False)[:18000])
        clip_dir = result_path.parent / "annotation_clips" / f"{window_idx:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        answer = runner.watch(video, prompt, start_s=start, end_s=end, clip_dir=clip_dir,
                              max_new_tokens=3072, duration_s=end - start)
        parsed = parse_window_annotations(
            answer.text, valid_shot_ids={int(row["shot_idx"]) for row in window_shots})
        parsed = namespace_window_events(parsed, window_idx)
        saved.setdefault("shots", {}).update(parsed["shots"])
        saved.setdefault("causal_links", []).extend(parsed["causal_links"])
        saved.setdefault("completed_windows", []).append(window_idx)
        saved["completed_windows"] = sorted(set(saved["completed_windows"]))
        output_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        processed += 1
    return output_path


def run_narrative_index(cfg, source: str, *, video: Path | None = None,
                        force: bool = False, transcriber=None) -> Path:
    video = resolve_source_video(cfg, source, video)
    index_cfg = cfg.library.get("narrative_index") or {}
    output_dir = cfg.paths.library_dir / "sources" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = output_dir / "narrative_windows.json"
    transcript = run_source_transcript(cfg, source, video, force=force, model=transcriber)
    if scan_path.exists() and not force:
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        selected = scan["selected_windows"]
    else:
        samples, duration = scan_sparse_features(
            video, ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"),
            ffprobe_bin=cfg.perception.get("ffprobe_bin", "ffprobe"),
            fps=float(index_cfg.get("scan_fps", 2.0)),
            width=int(index_cfg.get("scan_width", 160)),
            audio_sr=int(index_cfg.get("audio_sr", 8000)))
        candidates = score_narrative_windows(
            samples, transcript, duration,
            window_s=float(index_cfg.get("window_s", 45.0)),
            stride_s=float(index_cfg.get("stride_s", 15.0)))
        quotas = {key: int((index_cfg.get("quotas") or {}).get(key, value))
                  for key, value in DEFAULT_QUOTAS.items()}
        selected = select_narrative_windows(
            candidates, duration, quotas=quotas,
            max_overlap=float(index_cfg.get("max_overlap", 0.35)),
            partitions=int(index_cfg.get("partitions", 6)))
        scan_path.write_text(json.dumps({
            "source": source, "video": str(video), "duration_s": duration,
            "configuration": index_cfg, "quotas": quotas,
            "candidate_count": len(candidates), "sample_count": len(samples),
            "selected_windows": selected,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    result = index_selected_windows(cfg, source, video, selected, force=force,
                                    profile="narrative")
    return attach_transcript_to_shots(result, transcript)
