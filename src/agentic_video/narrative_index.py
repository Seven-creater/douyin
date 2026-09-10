"""Two-pass narrative indexing for dialogue, action, emotion, and context windows."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.agentic_video.long_video import (index_selected_windows, resolve_source_video,
                                           robust_normalize, scan_sparse_features)
from src.agentic_video.narrative import ARC_ROLES
from src.agentic_video.zones import zone_config
from src.template.schema import extract_json_block

DEFAULT_QUOTAS = {"dialogue": 12, "action": 8, "emotion": 8, "context": 8}


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp + os.replace 原子写：崩溃在半途不再截断既有状态（H5）。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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
                             partitions: int = 6,
                             exclude_head_s: float = 0.0,
                             exclude_tail_s: float = 0.0) -> list[dict]:
    quotas = dict(quotas or DEFAULT_QUOTAS)
    selected: list[dict] = []
    per_partition = max(1, (sum(quotas.values()) + partitions - 1) // partitions)
    partition_counts = [0] * partitions
    # 片头/片尾排除区：OP/ED 职员表窗口不配浪费标注预算，也不进检索池
    eligible = [row for row in windows
                if not (float(row["start_s"]) < exclude_head_s
                        or float(row["end_s"]) > duration_s - exclude_tail_s)]
    for selection_type, quota in quotas.items():
        score_key = f"{selection_type}_score"
        taken = 0
        for row in sorted(eligible, key=lambda item: (-float(item.get(score_key, 0)),
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
    _atomic_write_json(Path(result_path), envelope)
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


def parse_window_annotations(raw: str, *, valid_shot_ids: set[int],
                             time_offset_s: float = 0.0,
                             clip_duration_s: float | None = None) -> dict:
    """解析窗口标注答案。

    H1：模型看的是从 time_offset_s 截出的切片（片段内 0 起算），其对白 start_s/end_s
    必须 + time_offset_s 归一回电影时间轴，并夹到窗口范围内——否则窗口在 3700s 时，
    下游会把 12.5s 当电影坐标从片头切素材/排字幕。
    """
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
        window_end = (time_offset_s + clip_duration_s
                      if clip_duration_s is not None else float("inf"))
        dialogue = []
        for line in item.get("dialogue") or []:
            if not isinstance(line, dict):
                continue
            try:
                start = float(line.get("start_s")) + time_offset_s
                end = float(line.get("end_s")) + time_offset_s
                line_confidence = min(1.0, max(0.0, float(line.get("confidence", 0.5))))
            except (TypeError, ValueError):
                continue
            start = max(time_offset_s, min(window_end, start))
            end = max(time_offset_s, min(window_end, end))
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
        try:
            saved = json.loads(output_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(
                f"标注状态文件损坏：{output_path}（可能是历史非原子写遗留）。"
                "请从备份恢复；删除重跑会全量重标 36 窗。") from exc
    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    shots = envelope["output"]["shots"]
    video = Path(envelope["output"]["video"])
    by_window: dict[int, list[dict]] = {}
    for shot in shots:
        by_window.setdefault(int(shot["window_idx"]), []).append(shot)
    # 旧数据迁移（H1 存量）：修复前写入的对白是切片坐标（0 起算），
    # 无 coord_system 标记——按各窗口起点统一 +offset 归一回电影轴
    if saved.get("shots") and saved.get("coord_system") != "movie":
        window_start = {str(row["shot_idx"]): min(float(r["start_s"])
                                                  for r in window_shots)
                        for window_shots in by_window.values() for row in window_shots}
        for key, annotation in saved["shots"].items():
            offset = window_start.get(str(key))
            if offset is None:
                continue
            for line in annotation.get("dialogue") or []:
                try:
                    line["start_s"] = round(offset + float(line.get("start_s", 0)), 3)
                    line["end_s"] = round(offset + float(line.get("end_s", 0)), 3)
                except (TypeError, ValueError):
                    continue
        saved["coord_system"] = "movie"
        _atomic_write_json(output_path, saved)      # 迁移立即落盘（纯加载也持久化）
    captions_path = result_path.parent / "captions.json"
    captions = (json.loads(captions_path.read_text(encoding="utf-8"))
                if captions_path.exists() else {})
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
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
            answer.text, valid_shot_ids={int(row["shot_idx"]) for row in window_shots},
            time_offset_s=start, clip_duration_s=end - start)
        parsed = namespace_window_events(parsed, window_idx)
        saved.setdefault("shots", {}).update(parsed["shots"])
        saved.setdefault("causal_links", []).extend(parsed["causal_links"])
        saved.setdefault("completed_windows", []).append(window_idx)
        saved["completed_windows"] = sorted(set(saved["completed_windows"]))
        saved["coord_system"] = "movie"
        _atomic_write_json(output_path, saved)
        processed += 1
    return output_path


FACET_PROMPT = """你是电影素材类型维度标注 Agent。观看 {start:g}~{end:g} 秒片段，按维度
age_appearance / location / era / interaction 提取结构化 facet。只输出 JSON：
{{"facets":[
 {{"dimension":"age_appearance","entity_id":"实体ID","value":"幼儿/儿童/少年/青年/成年/老年/uncertain",
   "apply_interval":[0.0,1.0],"evidence_interval":[0.0,1.0],"evidence_source":"frame","confidence":0.0}},
 {{"dimension":"location","value":"具体地点或场景类型（雪野/室内/城镇/列车…）",
   "apply_interval":[0.0,1.0],"evidence_interval":[0.0,1.0],"evidence_source":"frame","confidence":0.0}},
 {{"dimension":"era","value":"大正/近代/现代/uncertain",
   "apply_interval":[0.0,1.0],"evidence_interval":[0.0,1.0],"evidence_source":"frame","confidence":0.0}},
 {{"dimension":"interaction","a_entity_id":"实体A","b_entity_id":"实体B",
   "relation":"保护|对抗|师徒|亲情|同伴|救助|陌生",
   "apply_interval":[0.0,1.0],"evidence_interval":[0.0,1.0],"evidence_source":"frame","confidence":0.0}}
]}}
规则：
- 时间用片段内秒数（0 起算）。apply_interval=该 facet 在片段内成立的区间；
  evidence_interval=支撑它最直接的可见证据区间（如挡刀动作那两秒）。
- 每条 facet 必须绑定到具体实体（interaction 绑实体对）；location/era 可无实体。
- 只报告可见内容，不得根据 IP 常识补写；不确定的维度整条不输出。
- 实体必须优先来自【实体注册表】；画面中出现注册表外人物才新建 ID（visible_ 前缀）。
【实体注册表】{registry}
"""

FACET_PROMPT_VERSION = "facet_v1"


def parse_type_facets(raw: str, *, window_start: float, window_end: float,
                      known_entities: set[str]) -> dict:
    """解析 facet 答案：切片坐标归一回电影轴并夹到窗口内，非法条目丢弃。

    facet 的确认状态由证据推导：有 evidence_source 且 confidence>=0.5 为
    supported，否则 uncertain——窗口级印象不允许直接冒充已确认事实。
    """
    from src.agentic_video.type_dimensions import (INTERACTION_RELATIONS,
                                                   LIBRARY_FACET_DIMENSIONS)
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else {}
    except ValueError:
        payload = {}
    items = payload.get("facets") if isinstance(payload, dict) else None
    facets = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "")
        if dimension not in LIBRARY_FACET_DIMENSIONS:
            continue
        def _interval(key: str) -> list[float] | None:
            value = item.get(key)
            if not isinstance(value, list) or len(value) != 2:
                return None
            try:
                left = float(value[0]) + window_start
                right = float(value[1]) + window_start
            except (TypeError, ValueError):
                return None
            left = max(window_start, min(window_end, left))
            right = max(window_start, min(window_end, right))
            if right <= left:
                return None
            return [round(left, 3), round(right, 3)]
        apply_interval = _interval("apply_interval")
        evidence_interval = _interval("evidence_interval") or apply_interval
        if apply_interval is None:
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        entry = {
            "dimension": dimension,
            "apply_interval": apply_interval,
            "evidence_interval": evidence_interval,
            "evidence_source": str(item.get("evidence_source") or "") or None,
            "confidence": confidence,
            "status": "supported" if confidence >= 0.5 and item.get("evidence_source")
                      else "uncertain",
        }
        if dimension == "interaction":
            a_id, b_id = str(item.get("a_entity_id") or ""), str(item.get("b_entity_id") or "")
            relation = str(item.get("relation") or "")
            if not a_id or not b_id or a_id == b_id or relation not in INTERACTION_RELATIONS:
                continue
            entry.update({"a_entity_id": a_id, "b_entity_id": b_id, "relation": relation})
        else:
            value = str(item.get("value") or "").strip()
            if not value or value == "uncertain":
                continue
            entry["value"] = value[:24]
            if item.get("entity_id"):
                entry["entity_id"] = str(item["entity_id"])
        for key in ("entity_id", "a_entity_id", "b_entity_id"):
            if key in entry:
                entry["registry_known"] = entry[key] in known_entities
                break
        facets.append(entry)
    return {"facets": facets, "parse_status": "json" if block else "failed"}


def run_type_facets(cfg, source: str, *, windows: list[int] | None = None,
                    force: bool = False, runner=None) -> Path:
    """库侧类型维度补充标注（加性：不触碰既有叙事标注/索引）。

    断点续跑 + 依赖缓存键（源哈希+窗口+维度+提示词版本+模型指纹）：维度集或
    提示词变了，对应窗口自动重提；windows 参数用于试点（如 3-5 窗先验质量）。
    """
    from src.agentic_video.narrative_agent import _omni_signature, _stable_hash
    from src.agentic_video.recipe_v2 import sha256_file
    from src.agentic_video.type_dimensions import LIBRARY_FACET_DIMENSIONS

    result_path = cfg.paths.library_dir / "shots" / f"{source}__narrative" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"缺叙事索引镜头产物：{result_path}（先跑 index --profile narrative）")
    output_path = result_path.parent / "type_facets.json"
    saved = {"windows": {}, "completed": {}}
    if output_path.exists() and not force:
        try:
            saved = json.loads(output_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"facet 状态文件损坏：{output_path}") from exc
    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    shots = envelope["output"]["shots"]
    video = Path(envelope["output"]["video"])
    video_sha = sha256_file(video)
    omni_sig = _omni_signature(cfg)

    annotations_path = result_path.parent / "narrative_annotations.json"
    known_entities: set[str] = set()
    if annotations_path.exists():
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        for annotation in (annotations.get("shots") or {}).values():
            known_entities.update(str(value) for value in annotation.get("entity_ids") or [])
    registry = [{"entity_id": entity_id} for entity_id in sorted(known_entities)][:60]

    by_window: dict[int, list[dict]] = {}
    for shot in shots:
        by_window.setdefault(int(shot["window_idx"]), []).append(shot)
    targets = sorted(by_window) if windows is None else \
        sorted(idx for idx in windows if idx in by_window)
    for window_idx in targets:
        window_shots = by_window[window_idx]
        start = min(float(row["start_s"]) for row in window_shots)
        end = max(float(row["end_s"]) for row in window_shots)
        key = _stable_hash({
            "schema": "type_facets_v1", "video_sha": video_sha,
            "window": [start, end], "dimensions": LIBRARY_FACET_DIMENSIONS,
            "prompt_version": FACET_PROMPT_VERSION,
            "prompt_sha": _stable_hash(FACET_PROMPT), **omni_sig})
        cached = saved["completed"].get(str(window_idx))
        if cached == key and not force:
            continue
        if runner is None:
            from src.perception.omni_runner import OmniRunner
            runner = OmniRunner(cfg.perception.get("omni") or {})
        prompt = FACET_PROMPT.format(
            start=start, end=end,
            registry=json.dumps(registry, ensure_ascii=False)[:4000])
        clip_dir = result_path.parent / "facet_clips" / f"{window_idx:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        answer = runner.watch(video, prompt, start_s=start, end_s=end, clip_dir=clip_dir,
                              max_new_tokens=2048, duration_s=end - start)
        parsed = parse_type_facets(answer.text, window_start=start, window_end=end,
                                   known_entities=known_entities)
        parsed["raw_head"] = str(getattr(answer, "text", ""))[:400]   # 审计/排障
        parsed["cache_key"] = key
        saved["windows"][str(window_idx)] = parsed
        saved["completed"][str(window_idx)] = key
        _atomic_write_json(output_path, saved)
    return output_path


def run_narrative_index(cfg, source: str, *, video: Path | None = None,                        force: bool = False, transcriber=None) -> Path:
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
        zones = zone_config(cfg)
        is_movie = duration >= float(zones["min_movie_s"])
        selected = select_narrative_windows(
            candidates, duration, quotas=quotas,
            max_overlap=float(index_cfg.get("max_overlap", 0.35)),
            partitions=int(index_cfg.get("partitions", 6)),
            exclude_head_s=float(zones["head_s"]) if is_movie else 0.0,
            exclude_tail_s=float(zones["tail_s"]) if is_movie else 0.0)
        scan_path.write_text(json.dumps({
            "source": source, "video": str(video), "duration_s": duration,
            "configuration": index_cfg, "quotas": quotas,
            "source_zones": {"head_s": float(zones["head_s"]) if is_movie else 0.0,
                             "tail_s": float(zones["tail_s"]) if is_movie else 0.0},
            "candidate_count": len(candidates), "sample_count": len(samples),
            "selected_windows": selected,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    result = index_selected_windows(cfg, source, video, selected, force=force,
                                    profile="narrative")
    return attach_transcript_to_shots(result, transcript)
