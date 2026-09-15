"""V8 occurrence-first shared film understanding and character batch editing.

The module deliberately separates six layers:
external prior -> local occurrence -> identity binding -> event fact ->
character evidence view -> creative task.  A correction to identity therefore
never rewrites the source-grounded observation that one local subject acted on
another local subject.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.target_v7 import (
    accept_v7_output,
    apply_native_selection,
    extract_native_frames,
    export_frame,
    export_roi,
    finalize_target_microcut,
)
from src.config import AppConfig, repo_root
from src.perception import common
from src.perception.omni_runner import cut_clip


SPEC_VERSION = "character_batch_v8"
CHARACTER_IDS = ("char:xiaohei", "char:wuxian", "char:fengxi")
MENTION_TYPES = {
    "direct_address", "third_person_reference", "self_reference", "narration",
    "uncertain",
}
PRESENCE_STATES = {"candidate", "supported", "not_supported", "unknown"}
IDENTITY_STATES = {"verified", "uncertain", "conflict", "revoked"}
IDENTITY_COMPARE_RESULTS = {"same", "different", "uncertain"}
PREFERRED_TASK_FAMILIES = (
    "adversity_agency_outcome",
    "intervention_and_result",
    "capability_montage",
    "threat_escalation_and_impact",
)

CACHE_INVALIDATION = {
    "source": (
        "mention_index", "coverage", "occurrence_bank", "identity_bindings",
        "event_facts", "character_evidence_views", "creation_queue", "plans", "videos"),
    "asr": ("mention_index", "asr_leads", "dialogue_evidence"),
    "external_prior": ("candidate_roster", "aliases", "search_leads",
                       "unverified_hypotheses"),
    "character_profile": ("identity_bindings", "character_evidence_views",
                          "creation_queue", "plans", "videos"),
    "event_fact": ("character_evidence_views", "creation_queue", "plans", "videos"),
    "reference_task": ("creation_queue", "plans", "videos"),
}


class V8Blocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = ""):
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail
        message = f"{stage}/{reason_code}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def cache_invalidation_targets(change_kind: str) -> tuple[str, ...]:
    """Return dependency targets without over-invalidating independent fact layers."""
    if change_kind not in CACHE_INVALIDATION:
        raise ValueError(f"unsupported V8 change kind: {change_kind}")
    return CACHE_INVALIDATION[change_kind]


def _stable_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "item"


def _write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise V8Blocked("model", "invalid_json_response")
        try:
            value = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError as exc:
            raise V8Blocked("model", "invalid_json_response", str(exc)) from exc
    if not isinstance(value, dict):
        raise V8Blocked("model", "json_response_not_object")
    return value


def read_v8_spec(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    spec = _read_json(path)
    if spec.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"V8 spec_version must be {SPEC_VERSION}")
    characters = spec.get("characters") or []
    ids = tuple(str(row.get("character_id") or "") for row in characters)
    if ids != CHARACTER_IDS:
        raise ValueError(f"V8 first-round characters must be {CHARACTER_IDS}")
    coverage = spec.get("coverage") or {}
    if float(coverage.get("block_s", 0)) != 45.0:
        raise ValueError("V8 coverage block_s must be 45.0")
    if float(coverage.get("fps", 0)) != 2.0:
        raise ValueError("V8 coverage fps must be 2.0")
    if float(coverage.get("retention_ratio", 0)) != 0.10:
        raise ValueError("V8 coverage retention_ratio must be 0.10")
    return spec, hashlib.sha256(path.read_bytes()).hexdigest()


def build_external_character_prior(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Archive research leads without promoting them to source-local truth."""
    candidates = []
    claims = []
    for character in spec.get("characters") or []:
        character_id = str(character["character_id"])
        candidate = {
            "character_id": character_id,
            "display_name": str(character.get("display_name") or character_id),
            "aliases": [str(value) for value in character.get("aliases") or []],
            "selection_basis": "human_fixed_experiment_sample",
            "identity_status": "prior_only",
        }
        candidates.append(candidate)
        for index, raw in enumerate(character.get("external_claims") or []):
            source_tier = str(raw.get("source_tier") or "secondary")
            if source_tier not in {"official", "secondary", "human_task_input"}:
                raise ValueError(f"unsupported external source tier: {source_tier}")
            claims.append({
                "claim_id": f"{character_id}/claim_{index:03d}",
                "character_id": character_id,
                "field": str(raw.get("field") or "investigation_hint"),
                "value": raw.get("value", raw.get("claim")),
                "source_tier": source_tier,
                "source_url": str(raw.get("source_url") or ""),
                "work_scope": str(raw.get("work_scope") or raw.get("work_version")
                                  or spec.get("source") or ""),
                "status": "prior_only",
            })
    result = {
        "schema_version": "external_character_prior_v1",
        "source": spec.get("source"),
        "candidate_selection_claim": (
            "three human-fixed experiment samples; automatic character selection not tested"
        ),
        "candidates": candidates,
        "claims": claims,
        "source_evidence_written": False,
    }
    output_dir = Path(output_dir)
    _write_json(output_dir / "candidate_roster.json", result)
    _write_jsonl(output_dir / "external_claims.jsonl", claims)
    return result


def _mention_type(text: str, matched_name: str) -> str:
    compact = re.sub(r"\s+", "", text)
    name = re.escape(matched_name)
    if re.search(rf"我(?:叫|是){name}", compact):
        return "self_reference"
    if re.search(rf"^(?:喂|哎|啊)?{name}[，,！!？?：:]", compact):
        return "direct_address"
    third_person_markers = (
        "怎么样", "丢了", "找到", "寻找", "放弃", "带", "为了", "身上", "踪迹",
    )
    if any(marker in compact for marker in third_person_markers):
        return "third_person_reference"
    return "uncertain"


def build_mention_index(transcript: Mapping[str, Any] | Iterable[Mapping[str, Any]],
                        characters: Iterable[dict[str, Any]],
                        output_path: Path) -> list[dict[str, Any]]:
    """Create name leads; never bind a visible or speaking occurrence here."""
    aliases: list[tuple[str, str, str]] = []
    for character in characters:
        character_id = str(character["character_id"])
        names = dict.fromkeys(
            [character.get("display_name"), *(character.get("aliases") or [])])
        aliases.extend((str(name), character_id, "canonical_or_alias")
                       for name in names if str(name or "").strip())
        # ASR aliases are noisy transcript spellings, not character aliases and
        # never identity evidence.  They exist only to avoid losing investigation
        # leads such as 风息 -> 风隙/凤曦.
        asr_names = dict.fromkeys(character.get("asr_aliases") or [])
        aliases.extend((str(name), character_id, "asr_alias")
                       for name in asr_names if str(name or "").strip())
    rows = []
    segments = (transcript.get("segments") or transcript.get("utterances") or []) \
        if isinstance(transcript, Mapping) else transcript
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        for name, character_id, match_basis in aliases:
            if name not in text:
                continue
            rows.append({
                "mention_id": f"mention_{index:05d}_{len(rows):05d}",
                "utterance_id": f"utt_{index:05d}",
                "interval": [round(float(
                    segment.get("start_s", segment.get("start", 0))
                    if segment.get("start_ms") is None
                    else float(segment.get("start_ms") or 0) / 1000), 3),
                             round(float(
                    segment.get("end_s", segment.get("end", 0))
                    if segment.get("end_ms") is None
                    else float(segment.get("end_ms") or 0) / 1000), 3)],
                "text": text,
                "matched_name": name,
                "match_basis": match_basis,
                "mention_type": _mention_type(text, name),
                "visual_presence": "unknown",
                "speaker_occurrence_id": None,
                "mentioned_character_ids": [character_id],
                "addressee_occurrence_id": None,
                "responding_occurrence_id": None,
                "status": "lead",
            })
    _write_jsonl(output_path, rows)
    return rows


def coverage_blocks(duration_s: float, *, block_s: float = 45.0,
                    head_s: float = 90.0, tail_s: float = 360.0,
                    min_movie_s: float = 1800.0) -> list[tuple[float, float]]:
    if duration_s <= 0 or block_s <= 0:
        raise ValueError("duration and block size must be positive")
    start = head_s if duration_s >= min_movie_s else 0.0
    end = duration_s - tail_s if duration_s >= min_movie_s else duration_s
    if end <= start:
        return []
    blocks = []
    cursor = float(start)
    while cursor < end - 1e-6:
        boundary = min(end, cursor + block_s)
        blocks.append((round(cursor, 6), round(boundary, 6)))
        cursor = boundary
    return blocks


NEUTRAL_COVERAGE_PROMPT = """Watch this source-movie block as a neutral observer.
Do not identify any character, use names, infer canonical identity, discuss story templates,
or assign editorial functions. Give local subjects temporary labels A/B/C within each region.
Inspect the entire block before selecting regions. Report only concrete visible state changes,
interactions, entrances/exits, or other activity that merits a closer original-video rewatch;
never report routine static presence or generic placeholder descriptions to fill the list.
Return exactly one JSON object with a regions array. Each region must contain interval,
activity, worth_rewatch, temporal_refinement_needed, occurrences, and event_candidates.
Each occurrence must contain local_id, a concrete local_description, a concrete visual_state,
and roi. Each event candidate must contain actor_local_id, a concrete action,
patient_local_id, and the visible_result. Times are seconds relative to this block.
First inspect the full 45-second input, then
evaluate its early [0,15), middle [15,30), and late [30,45] thirds independently.
Select at most one strongest qualifying region per third. Every interval must be 2-6
seconds; do not split the timeline into fixed bins. For a longer action, return the tightest 6-second
excerpt containing its clearest state change and set "temporal_refinement_needed":true.
Every actor_local_id and non-null patient_local_id used by an event must have its own
entry in that same region's occurrences list; never group multiple visible subjects into
one occurrence.
Every returned region must have worth_rewatch=true. If nothing merits rewatch return
{"regions":[]}. Not observed never means absent."""

COVERAGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "v8_neutral_coverage",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array", "maxItems": 3,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "interval": {"type": "array", "minItems": 2,
                                         "maxItems": 2,
                                         "items": {"type": "number"}},
                            "activity": {"type": "string"},
                            "worth_rewatch": {"type": "boolean"},
                            "temporal_refinement_needed": {"type": "boolean"},
                            "occurrences": {"type": "array", "maxItems": 6,
                                "items": {"type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "local_id": {"type": "string"},
                                        "local_description": {"type": "string"},
                                        "visual_state": {"type": "string"},
                                        "roi": {"type": ["array", "null"],
                                                "items": {"type": "number"},
                                                "minItems": 4, "maxItems": 4},
                                    },
                                    "required": ["local_id", "local_description",
                                                 "visual_state", "roi"]}},
                            "event_candidates": {"type": "array", "maxItems": 4,
                                "items": {"type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "actor_local_id": {"type": "string"},
                                        "action": {"type": "string"},
                                        "patient_local_id": {"type": ["string", "null"]},
                                        "visible_result": {"type": "string"},
                                    },
                                    "required": ["actor_local_id", "action",
                                                 "patient_local_id", "visible_result"]}},
                        },
                        "required": ["interval", "activity", "worth_rewatch",
                                     "temporal_refinement_needed", "occurrences",
                                     "event_candidates"],
                    },
                },
            },
            "required": ["regions"], "additionalProperties": False,
        },
    },
}


_FORBIDDEN_NEUTRAL_KEYS = {
    "character_id", "canonical_id", "target_id", "assigned_goal", "editing_role",
    "adversity", "agency", "outcome",
}


def _parse_coverage_payload(text: str) -> tuple[dict[str, Any], str]:
    """Accept the requested object and one harmless model formatting variant."""
    candidate = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise V8Blocked("model", "invalid_json_response", str(exc)) from exc
    if isinstance(value, list):
        return {"regions": value}, "bare_array_wrapped"
    if not isinstance(value, dict):
        raise V8Blocked("model", "json_response_not_object_or_array")
    return value, "object"


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) in _FORBIDDEN_NEUTRAL_KEYS or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _normalize_coverage_response(payload: dict[str, Any], *, block_id: str,
                                 start_s: float, end_s: float) -> tuple[list[dict], list[dict]]:
    occurrences: list[dict] = []
    event_candidates: list[dict] = []
    if _contains_forbidden_key(payload):
        raise V8Blocked("coverage", "neutral_observation_contaminated")
    for region_index, region in enumerate((payload.get("regions") or [])[:3]):
        if not isinstance(region, dict):
            continue
        if region.get("worth_rewatch") is not True:
            continue
        _concrete_text(region.get("activity"), field="coverage_activity")
        interval = region.get("interval") or []
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        rel_start, rel_end = map(float, interval)
        if not 0 <= rel_start < rel_end <= end_s - start_s + 1e-6:
            continue
        if not 2.0 - 1e-6 <= rel_end - rel_start <= 6.0 + 1e-6:
            raise V8Blocked("coverage", "candidate_interval_outside_2_to_6_seconds")
        absolute = [round(start_s + rel_start, 6), round(start_s + rel_end, 6)]
        local_to_global: dict[str, str] = {}
        for occurrence_index, raw in enumerate((region.get("occurrences") or [])[:6]):
            if not isinstance(raw, dict):
                continue
            local_id = str(raw.get("local_id") or chr(65 + occurrence_index))
            occurrence_id = f"occ_{block_id}_{region_index:02d}_{local_id}"
            local_to_global[local_id] = occurrence_id
            row = {
                "occurrence_id": occurrence_id,
                "source_interval": absolute,
                "local_description": _concrete_text(
                    raw.get("local_description"), field="coverage_local_description"),
                "visual_state": _concrete_text(
                    raw.get("visual_state"), field="coverage_visual_state"),
                "roi": raw.get("roi"),
                "coverage_block_id": block_id,
                "observation_status": "coarse_candidate",
            }
            row["observation_sha256"] = _stable_sha(row)
            occurrences.append(row)
        for event_index, raw in enumerate((region.get("event_candidates") or [])[:4]):
            if not isinstance(raw, dict):
                continue
            actor = local_to_global.get(str(raw.get("actor_local_id") or ""))
            patient_local_id = str(raw.get("patient_local_id") or "")
            patient = local_to_global.get(patient_local_id)
            if not actor:
                raise V8Blocked("coverage", "event_actor_missing_occurrence")
            if patient_local_id and not patient:
                raise V8Blocked("coverage", "event_patient_missing_occurrence")
            event_candidates.append({
                "event_candidate_id": f"candidate_{block_id}_{region_index:02d}_{event_index:02d}",
                "source_interval": absolute,
                "actor_occurrence_id": actor,
                "action": _concrete_text(raw.get("action"),
                                         field="coverage_action"),
                "patient_occurrence_id": patient,
                "visible_result": _concrete_text(
                    raw.get("visible_result"), field="coverage_visible_result",
                    allow_empty=True),
                "activity": str(region.get("activity") or "").strip(),
                "worth_rewatch": bool(region.get("worth_rewatch")),
                "status": "candidate",
            })
    return occurrences, event_candidates


def _build_coverage_transport(ffmpeg_bin: str, ffprobe_bin: str,
                              source_video: Path, destination: Path, *,
                              start_s: float, end_s: float,
                              requested_fps: float,
                              source_fps: float | None = None
                              ) -> tuple[Path, dict[str, Any]]:
    """Create a video containing exactly the frames offered to FlashVID.

    Coverage browsing is visual-only.  Sampling before transport makes the
    transmitted frame set auditable even though the OpenAI-compatible response
    does not expose its decoder metadata.
    """
    if requested_fps <= 0 or end_s <= start_s:
        raise ValueError("coverage transport requires a positive interval and FPS")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    clip = destination / "clip.mp4"
    duration_s = float(end_s - start_s)
    requested_frames = max(4, int(duration_s * requested_fps))
    requested_frames -= requested_frames % 2
    transport_fps = requested_frames / duration_s
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-loglevel", "error", "-ss", f"{start_s}", "-to", f"{end_s}",
        "-i", str(source_video), "-map", "0:v:0",
        "-vf", f"fps={transport_fps:.12f},scale='min(1920,iw)':-2,setpts=PTS-STARTPTS",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", str(clip),
    ], timeout_s=300)
    probe = subprocess.run([
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json", str(clip),
    ], capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        raise common.FFmpegError(
            f"coverage transport ffprobe failed: {(probe.stderr or '')[-300:]}")
    payload = json.loads(probe.stdout)
    relative = [
        round(float(row["best_effort_timestamp_time"]), 6)
        for row in payload.get("frames") or []
        if row.get("best_effort_timestamp_time") is not None
    ]
    absolute = [round(float(start_s) + value, 6) for value in relative]
    verified = (
        len(relative) == requested_frames and
        all(0 <= value < duration_s + 1e-6 for value in relative) and
        all(left < right for left, right in zip(relative, relative[1:]))
    )
    audit = {
        "audit_basis": "deterministic_presampled_transport",
        "requested_fps": float(requested_fps),
        "transport_fps": round(transport_fps, 9),
        "effective_fps": round(len(relative) / duration_s, 6),
        "source_fps": source_fps,
        "requested_frame_count": requested_frames,
        "actual_frame_count": len(relative),
        "actual_frame_indices": list(range(len(relative))),
        "actual_frame_timestamps_relative_s": relative,
        "actual_frame_timestamps_absolute_s": absolute,
        "source_time_origin_s": float(start_s),
        "do_sample_frames": False,
        "transport_sha256": sha256_file(clip),
        "sampling_verified": verified,
    }
    _write_json(destination / "sampling_manifest.json", audit)
    if not verified:
        raise V8Blocked("coverage", "sampling_audit_failed", json.dumps({
            "requested": requested_frames, "actual": len(relative),
            "interval": [start_s, end_s],
        }))
    return clip, audit


def build_uniform_coverage_map(cfg: AppConfig, spec: dict[str, Any], output_dir: Path, *,
                               client, source_video: Path, source_sha256: str,
                               reuse_completed: bool = True) -> dict[str, Any]:
    """Run one cheap, character-neutral browse request for every coverage block."""
    output_dir = Path(output_dir)
    coverage = spec["coverage"]
    duration = common.video_duration_s(
        cfg.perception.get("ffprobe_bin", "ffprobe"), Path(source_video))
    blocks = coverage_blocks(
        duration, block_s=float(coverage["block_s"]),
        head_s=float(coverage.get("head_s", 90.0)),
        tail_s=float(coverage.get("tail_s", 360.0)),
        min_movie_s=float(coverage.get("min_movie_s", 1800.0)),
    )
    contract = {
        "source_sha256": source_sha256,
        "prompt_sha256": hashlib.sha256(NEUTRAL_COVERAGE_PROMPT.encode()).hexdigest(),
        "fps": float(coverage["fps"]),
        "retention_ratio": float(coverage["retention_ratio"]),
        "block_s": float(coverage["block_s"]),
    }
    completed: dict[str, dict] = {}
    jobs = []
    ffmpeg = cfg.perception.get("ffmpeg_bin", "ffmpeg")
    ffprobe = cfg.perception.get("ffprobe_bin", "ffprobe")
    source_probe = common.run_ffprobe_json(ffprobe, Path(source_video))
    video_stream = next(
        (row for row in source_probe.get("streams") or []
         if row.get("codec_type") == "video"), {})
    rate_text = str(video_stream.get("avg_frame_rate") or "0/1")
    try:
        numerator, denominator = rate_text.split("/", 1)
        source_fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        source_fps = None

    def stage_for_service(clip: Path, block_id: str) -> tuple[Path, dict[str, Any]]:
        """Copy one transport clip into the server's explicit media allow-list.

        The copy is request-scoped and removed after the synchronous response.  The
        repository copy remains the durable audit artifact.
        """
        configured_root = str((coverage.get("endpoint") or {}).get("media_root") or "").strip()
        clip_digest = sha256_file(clip)
        if not configured_root:
            return clip, {
                "staged": False,
                "source_transport_clip": str(clip),
                "source_transport_clip_sha256": clip_digest,
                "request_transport_clip": str(clip),
                "request_transport_clip_sha256": clip_digest,
            }
        media_root = Path(configured_root).resolve()
        media_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                prefix=f"v8_coverage_{block_id}_", suffix=".mp4",
                dir=media_root, delete=False) as handle:
            staged = Path(handle.name)
        try:
            staged.resolve().relative_to(media_root)
            shutil.copy2(clip, staged)
            staged_digest = sha256_file(staged)
            if staged_digest != clip_digest:
                raise V8Blocked("coverage", "media_staging_hash_mismatch", block_id)
        except Exception:
            staged.unlink(missing_ok=True)
            raise
        return staged, {
            "staged": True,
            "media_root": str(media_root),
            "source_transport_clip": str(clip),
            "source_transport_clip_sha256": clip_digest,
            "request_transport_clip": str(staged),
            "request_transport_clip_sha256": staged_digest,
        }

    def run_one(block_id: str, start: float, end: float) -> dict[str, Any]:
        block_dir = output_dir / "blocks" / block_id
        block_dir.mkdir(parents=True, exist_ok=True)
        result_path = block_dir / "result.json"
        cache_key = _stable_sha({**contract, "source_interval": [start, end]})
        if reuse_completed and result_path.is_file():
            prior = _read_json(result_path)
            if prior.get("cache_key") == cache_key and prior.get("status") == "covered":
                return prior
        clip, transport_sampling = _build_coverage_transport(
            ffmpeg, ffprobe, Path(source_video), block_dir / "transport",
            start_s=start, end_s=end, requested_fps=float(coverage["fps"]),
            source_fps=source_fps)
        forbidden_terms = {
            str(value).strip().lower()
            for character in spec.get("characters") or []
            for value in ([character.get("display_name")]
                          + list(character.get("aliases") or []))
            if str(value or "").strip()
        }
        attempt_audits: list[dict[str, Any]] = []
        for attempt in range(1, 3):
            request_clip, media_transport_audit = stage_for_service(clip, block_id)
            retry_suffix = ("" if attempt == 1 else
                "\nCONTRACT RETRY: inspect the full block again and correct the prior "
                "format failure. Return one object, 2-6 second regions, and define every "
                "event actor/patient as a separate occurrence.")
            try:
                answer = client.watch(
                    request_clip, NEUTRAL_COVERAGE_PROMPT + retry_suffix,
                    duration_s=end - start, image_paths=None, max_tokens=2048,
                    response_format=COVERAGE_RESPONSE_FORMAT)
            finally:
                if request_clip != clip:
                    request_clip.unlink(missing_ok=True)
            raw_path = block_dir / "raw_response.txt"
            raw_path.write_text(str(answer.text), encoding="utf-8")
            attempt_raw_path = block_dir / f"raw_response_attempt_{attempt:02d}.txt"
            attempt_raw_path.write_text(str(answer.text), encoding="utf-8")
            response_envelope_path = block_dir / "response_envelope.json"
            envelope = {"request_audit": answer.request_audit, "response": answer.raw}
            _write_json(response_envelope_path, envelope)
            attempt_envelope_path = block_dir / f"response_envelope_attempt_{attempt:02d}.json"
            _write_json(attempt_envelope_path, envelope)
            attempt_audit = {
                "attempt": attempt, "raw_response_path": str(attempt_raw_path),
                "response_envelope_path": str(attempt_envelope_path),
                "request_audit": answer.request_audit,
                "media_transport_audit": media_transport_audit,
            }
            try:
                raw, response_shape = _parse_coverage_payload(answer.text)
                serialized = json.dumps(raw, ensure_ascii=False).lower()
                if any(term in serialized for term in forbidden_terms):
                    raise V8Blocked("coverage", "neutral_observation_named_character")
                occurrences, events = _normalize_coverage_response(
                    raw, block_id=block_id, start_s=start, end_s=end)
            except Exception as exc:
                attempt_audit["validation_error"] = f"{type(exc).__name__}: {exc}"
                attempt_audits.append(attempt_audit)
                if attempt == 2:
                    raise
                continue
            attempt_audit["validation_error"] = None
            attempt_audits.append(attempt_audit)
            break
        for occurrence in occurrences:
            occurrence["source_video"] = str(source_video)
        for event in events:
            event["source_video"] = str(source_video)
        server_sampling = (answer.raw.get("sampling_audit") or
                           answer.raw.get("video_metadata") or {})
        sampling = {
            **transport_sampling,
            "server_sampling_audit": server_sampling,
            "server_sampling_audit_status": (
                "available" if server_sampling else "unavailable_from_server"),
            "server_requested_frame_count": answer.request_audit.get("requested_frames"),
        }
        if answer.request_audit.get("requested_frames") != transport_sampling[
                "actual_frame_count"]:
            raise V8Blocked("coverage", "sampling_audit_failed",
                            "request count differs from transmitted frame count")
        row = {
            "schema_version": "coverage_block_v1",
            "block_id": block_id,
            "source_interval": [start, end],
            "cache_key": cache_key,
            "status": "covered",
            "not_observed_is_absent": False,
            "occurrences": occurrences,
            "event_candidates": events,
            "request_audit": answer.request_audit,
            "media_transport_audit": media_transport_audit,
            "raw_response_path": str(raw_path),
            "response_envelope_path": str(response_envelope_path),
            "response_shape": response_shape,
            "response_attempts": attempt_audits,
            "sampling_audit": sampling,
            "sampling_audit_status": "verified_transport_frames",
            "raw_response": answer.text,
        }
        _write_json(result_path, row)
        return row

    for index, (start, end) in enumerate(blocks):
        block_id = f"b{index:04d}"
        block_dir = output_dir / "blocks" / block_id
        result_path = block_dir / "result.json"
        cache_key = _stable_sha({**contract, "source_interval": [start, end]})
        if reuse_completed and result_path.is_file():
            prior = _read_json(result_path)
            if prior.get("cache_key") == cache_key and prior.get("status") == "covered":
                completed[block_id] = prior
                continue
        jobs.append((block_id, start, end))
    workers = min(int(coverage.get("workers", 8)), max(1, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, block_id, start, end): (block_id, start, end)
            for block_id, start, end in jobs
        }
        for future in as_completed(futures):
            block_id, start, end = futures[future]
            try:
                completed[block_id] = future.result()
            except Exception as exc:  # a failed block is never marked covered
                failed = {
                    "schema_version": "coverage_block_v1", "block_id": block_id,
                    "source_interval": [start, end], "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                raw_path = output_dir / "blocks" / block_id / "raw_response.txt"
                envelope_path = output_dir / "blocks" / block_id / "response_envelope.json"
                if raw_path.is_file():
                    failed["raw_response_path"] = str(raw_path)
                if envelope_path.is_file():
                    failed["response_envelope_path"] = str(envelope_path)
                completed[block_id] = failed
                _write_json(output_dir / "blocks" / block_id / "result.json", failed)
    ordered = [completed[f"b{index:04d}"] for index in range(len(blocks))]
    occurrences = [row for block in ordered for row in block.get("occurrences") or []]
    events = [row for block in ordered for row in block.get("event_candidates") or []]
    failures = [row for row in ordered if row.get("status") != "covered"]
    result = {
        "schema_version": "uniform_coverage_v1", "source": spec["source"],
        "contract": contract, "duration_s": duration, "block_count": len(ordered),
        "covered_count": len(ordered) - len(failures), "failed_count": len(failures),
        "complete": not failures, "blocks": ordered,
        "occurrence_count": len(occurrences), "event_candidate_count": len(events),
    }
    _write_json(output_dir / "coverage_manifest.json", result)
    _write_jsonl(output_dir / "occurrence_candidates.jsonl", occurrences)
    _write_jsonl(output_dir / "event_candidates.jsonl", events)
    return result


def build_occurrence_bank(coverage_occurrences: Iterable[Mapping[str, Any]],
                          targeted_occurrences: Iterable[Mapping[str, Any]],
                          output_path: Path) -> list[dict[str, Any]]:
    """Combine observation sources without guessing cross-shot character identity."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_kind, source_rows in (("coverage_floor", coverage_occurrences),
                                     ("targeted_lead", targeted_occurrences)):
        for raw in source_rows:
            row = dict(raw)
            if _contains_key_recursive(row, "character_id"):
                raise V8Blocked("occurrence", "occurrence_contains_character_id")
            digest = str(row.get("observation_sha256") or _stable_sha(row))
            if digest in seen:
                continue
            seen.add(digest)
            row.setdefault("observation_source", source_kind)
            row.setdefault("observation_sha256", digest)
            rows.append(row)
    rows.sort(key=lambda row: (
        float((row.get("source_interval") or [float("inf")])[0]),
        str(row.get("occurrence_id") or "")))
    _write_jsonl(output_path, rows)
    return rows


def build_investigation_leads(mention_rows: Iterable[Mapping[str, Any]],
                              coverage_events: Iterable[Mapping[str, Any]],
                              narrative_windows: Iterable[Mapping[str, Any]] = ()) \
        -> list[dict[str, Any]]:
    """Union targeted and uniform leads; no lead is identity or event truth."""
    raw_leads: list[dict[str, Any]] = []
    for mention in mention_rows:
        interval = list(map(float, mention.get("interval") or []))
        if len(interval) == 2 and interval[1] > interval[0]:
            raw_leads.append({
                "source_interval": interval, "lead_kind": "asr_mention",
                "lead_id": mention.get("mention_id"),
                "character_hypotheses": mention.get("mentioned_character_ids") or [],
            })
    for event in coverage_events:
        interval = list(map(float, event.get("source_interval") or []))
        if len(interval) == 2 and interval[1] > interval[0]:
            raw_leads.append({
                "source_interval": interval, "lead_kind": "uniform_coverage_activity",
                "lead_id": event.get("event_candidate_id"), "character_hypotheses": [],
            })
    for index, window in enumerate(narrative_windows):
        interval = list(map(float, window.get("source_interval") or
                            window.get("interval") or []))
        if len(interval) == 2 and interval[1] > interval[0]:
            raw_leads.append({
                "source_interval": interval, "lead_kind": "existing_narrative_window",
                "lead_id": window.get("window_id", f"window_{index:04d}"),
                "character_hypotheses": [],
            })
    unique: dict[tuple[float, float, str], dict[str, Any]] = {}
    for row in raw_leads:
        start, end = row["source_interval"]
        key = (round(start, 3), round(end, 3), row["lead_kind"])
        unique.setdefault(key, row)
    return sorted(unique.values(), key=lambda row: (
        row["source_interval"][0], row["source_interval"][1], row["lead_kind"]))


NEUTRAL_OCCURRENCE_PROMPT = """Observe this source-movie context without identifying
characters or assigning story/editing roles. Work in this order: first list local visual
subjects using temporary labels A/B/C; then describe actions using only labels already
listed. A new shot may create a new local occurrence even when it may depict the same
person. Within one continuous shot, do not fragment one subject without visible reason.

Return exactly one JSON object and nothing else. Required keys are status, occurrences,
event_candidates, left_context_complete, right_context_complete, and boundary_reason.
status must be observed, observed_empty, or unreliable. Use observed_empty only when the
clip is clear and has no concrete subject or event worth recording. Use unreliable for
dissolves, black/corrupted frames, severe motion blur, or insufficient visual evidence.

Each occurrence needs local_id, visible_interval, a concrete visible appearance in
local_description, a concrete current action/state in visual_state, and roi (null or four
normalized coordinates). Each event needs interval, actor_local_id, a concrete visible
action, patient_local_id (or null), and the visible result (empty only when no result is
shown). Times are seconds relative to this input and must stay inside its exact duration.
Every event actor and non-null patient must reference an occurrence in this response.
Do not use placeholder phrases, generic statements, canonical names, or causal claims not
shown by the clip. If the input starts after an action has already begun, set
left_context_complete=false. If it ends before the visible action/result finishes, set
right_context_complete=false and briefly explain boundary_reason."""


NEUTRAL_OBSERVATION_STATES = {"observed", "observed_empty", "unreliable"}
_PLACEHOLDER_TEXT = {
    "visible state", "visible state/activity", "visible verb", "visible action",
    "visible result", "visible result or empty", "visible change or empty",
    "neutral visible appearance", "visible appearance only",
    "visible coarse activity", "人物发生动作", "某个角色在活动", "发生了某种动作",
}


def _single_json_object(text: str) -> dict[str, Any]:
    """Parse one object while rejecting repeated JSON or model commentary."""
    candidate = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    try:
        value, offset = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise V8Blocked("model", "invalid_json_response", str(exc)) from exc
    if candidate[offset:].strip():
        raise V8Blocked("model", "trailing_or_repeated_model_output")
    if not isinstance(value, dict):
        raise V8Blocked("model", "json_response_not_object")
    return value


def _concrete_text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text and allow_empty:
        return ""
    normalized = text.lower().strip(" .,:;!?，。！？：；\"'")
    if (not normalized or normalized in _PLACEHOLDER_TEXT or
            normalized.startswith("visible ") or
            normalized in {"unknown", "n/a", "none", "something happens"}):
        raise V8Blocked("occurrence", f"non_concrete_{field}")
    if not re.search(r"[A-Za-z0-9\u3400-\u9fff]", normalized):
        raise V8Blocked("occurrence", f"non_concrete_{field}")
    return text


def validate_neutral_observation(payload: Mapping[str, Any], *, duration_s: float) \
        -> str:
    """Validate observable content, not merely JSON shape or string length."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if _contains_forbidden_key(payload):
        raise V8Blocked("occurrence", "neutral_observation_contaminated")
    legacy_passed = payload.get("passed")
    status = str(payload.get("status") or (
        "observed" if legacy_passed is True else "observed_empty"
    ))
    if status not in NEUTRAL_OBSERVATION_STATES:
        raise V8Blocked("occurrence", "invalid_observation_status")
    occurrences = payload.get("occurrences") or []
    events = payload.get("event_candidates") or []
    if status != "observed" and (occurrences or events):
        raise V8Blocked("occurrence", "non_observed_status_contains_claims")
    if status == "observed" and not occurrences:
        raise V8Blocked("occurrence", "observed_without_occurrences")
    if status == "unreliable":
        reason = _concrete_text(payload.get("boundary_reason"), field="unreliable_reason")
        if not reason:
            raise V8Blocked("occurrence", "unreliable_without_reason")
        return status

    local_ids: set[str] = set()
    concrete_values: list[str] = []
    for index, occurrence in enumerate(occurrences):
        if not isinstance(occurrence, Mapping):
            raise V8Blocked("occurrence", "invalid_occurrence")
        local_id = str(occurrence.get("local_id") or "").strip()
        if not local_id or local_id in local_ids:
            raise V8Blocked("occurrence", "duplicate_or_missing_local_occurrence_id")
        local_ids.add(local_id)
        _normalize_relative_interval(
            occurrence.get("visible_interval"), start_s=0.0, end_s=duration_s,
            field=f"occurrence_{index}_visible_interval")
        concrete_values.extend([
            _concrete_text(occurrence.get("local_description"),
                           field="local_description"),
            _concrete_text(occurrence.get("visual_state"), field="visual_state"),
        ])
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise V8Blocked("occurrence", "invalid_event_candidate")
        actor = str(event.get("actor_local_id") or "").strip()
        patient = str(event.get("patient_local_id") or "").strip()
        if actor not in local_ids:
            raise V8Blocked("occurrence", "event_actor_missing_occurrence")
        if patient and patient not in local_ids:
            raise V8Blocked("occurrence", "event_patient_missing_occurrence")
        _normalize_relative_interval(
            event.get("interval"), start_s=0.0, end_s=duration_s,
            field=f"event_{index}_interval")
        concrete_values.append(_concrete_text(event.get("action"), field="action"))
        result = _concrete_text(event.get("visible_result"), field="visible_result",
                                allow_empty=True)
        if result:
            concrete_values.append(result)
    duplicates = [value for value in set(concrete_values)
                  if concrete_values.count(value) > 2]
    if duplicates:
        raise V8Blocked("occurrence", "repeated_semantic_fields")
    for key in ("left_context_complete", "right_context_complete"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            raise V8Blocked("occurrence", f"invalid_{key}")
    return status


def investigation_windows(leads: Iterable[Mapping[str, Any]], *, duration_s: float,
                          window_s: float = 6.0) -> list[dict[str, Any]]:
    """Turn heterogeneous leads into bounded, mostly deduplicated rewatch windows."""
    if duration_s <= 0 or window_s <= 0:
        raise ValueError("positive duration and window size required")
    proposed: list[dict[str, Any]] = []
    for lead in leads:
        interval = list(map(float, lead.get("source_interval") or []))
        if len(interval) != 2 or not 0 <= interval[0] < interval[1] <= duration_s + 1e-6:
            continue
        center = (interval[0] + interval[1]) / 2
        start = min(max(0.0, center - window_s / 2), max(0.0, duration_s - window_s))
        end = min(duration_s, start + window_s)
        proposed.append({
            "source_interval": [round(start, 6), round(end, 6)],
            "lead_refs": [{"lead_kind": lead.get("lead_kind"),
                           "lead_id": lead.get("lead_id")}],
        })
    merged: list[dict[str, Any]] = []
    for row in sorted(proposed, key=lambda item: item["source_interval"]):
        if merged:
            previous = merged[-1]
            union_start = min(previous["source_interval"][0], row["source_interval"][0])
            union_end = max(previous["source_interval"][1], row["source_interval"][1])
            if union_end - union_start <= window_s + 0.5:
                center = (union_start + union_end) / 2
                start = min(max(0.0, center - window_s / 2),
                            max(0.0, duration_s - window_s))
                previous["source_interval"] = [round(start, 6),
                                               round(min(duration_s, start + window_s), 6)]
                previous["lead_refs"].extend(row["lead_refs"])
                continue
        merged.append(row)
    for index, row in enumerate(merged):
        row["observation_id"] = f"observation_{index:05d}"
    return merged


def compute_temporal_coverage(intervals: Iterable[Iterable[float]], *,
                              eligible_interval: Iterable[float]) -> dict[str, Any]:
    """Measure time-union coverage instead of treating window count as coverage."""
    eligible = list(map(float, eligible_interval))
    if len(eligible) != 2 or eligible[1] <= eligible[0]:
        raise ValueError("eligible_interval must have increasing endpoints")
    eligible_start, eligible_end = eligible
    clipped: list[list[float]] = []
    for raw in intervals:
        row = list(map(float, raw))
        if len(row) != 2 or row[1] <= row[0]:
            continue
        start = max(eligible_start, row[0])
        end = min(eligible_end, row[1])
        if end > start:
            clipped.append([start, end])
    union: list[list[float]] = []
    for start, end in sorted(clipped):
        if union and start <= union[-1][1] + 1e-6:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    uncovered: list[list[float]] = []
    cursor = eligible_start
    for start, end in union:
        if start > cursor + 1e-6:
            uncovered.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < eligible_end - 1e-6:
        uncovered.append([cursor, eligible_end])
    covered_s = sum(end - start for start, end in union)
    eligible_s = eligible_end - eligible_start
    return {
        "eligible_interval": [eligible_start, eligible_end],
        "eligible_duration_s": round(eligible_s, 6),
        "input_interval_count": len(clipped),
        "union_intervals": [[round(a, 6), round(b, 6)] for a, b in union],
        "covered_duration_s": round(covered_s, 6),
        "coverage_ratio": round(covered_s / eligible_s, 6),
        "uncovered_intervals": [[round(a, 6), round(b, 6)]
                                for a, b in uncovered],
    }


def merge_candidate_regions(candidates: Iterable[Mapping[str, Any]], *,
                            max_gap_s: float = 1.0) -> list[dict[str, Any]]:
    """Merge overlapping leads before Omni context viewing."""
    if max_gap_s < 0:
        raise ValueError("max_gap_s cannot be negative")
    normalized = []
    for index, candidate in enumerate(candidates):
        interval = list(map(float, candidate.get("source_interval") or []))
        if len(interval) != 2 or interval[1] <= interval[0]:
            continue
        refs = list(candidate.get("lead_refs") or [])
        if not refs:
            refs = [{"lead_kind": candidate.get("lead_kind"),
                     "lead_id": candidate.get("lead_id", f"lead_{index:04d}")}]
        normalized.append({"source_interval": interval, "lead_refs": refs})
    merged: list[dict[str, Any]] = []
    for row in sorted(normalized, key=lambda item: item["source_interval"]):
        if merged and row["source_interval"][0] <= (
                merged[-1]["source_interval"][1] + max_gap_s):
            merged[-1]["source_interval"][1] = max(
                merged[-1]["source_interval"][1], row["source_interval"][1])
            merged[-1]["lead_refs"].extend(row["lead_refs"])
        else:
            merged.append({"source_interval": list(row["source_interval"]),
                           "lead_refs": list(row["lead_refs"])})
    for index, row in enumerate(merged):
        row["context_id"] = f"context_{index:05d}"
        row["source_interval"] = [round(value, 6)
                                  for value in row["source_interval"]]
    return merged


def expand_boundary_context(candidate_interval: Iterable[float], *, duration_s: float,
                            current_interval: Iterable[float] | None = None,
                            left_context_complete: bool = True,
                            right_context_complete: bool = True,
                            initial_context_s: float = 10.0,
                            expansion_step_s: float = 4.0,
                            max_context_s: float = 40.0) -> dict[str, Any]:
    """Expand transport context without redefining the underlying event boundary."""
    candidate = list(map(float, candidate_interval))
    if (len(candidate) != 2 or candidate[1] <= candidate[0] or duration_s <= 0 or
            initial_context_s <= 0 or max_context_s < initial_context_s):
        raise ValueError("invalid context expansion arguments")
    if current_interval is None:
        center = sum(candidate) / 2.0
        width = min(max_context_s, max(initial_context_s, candidate[1] - candidate[0]))
        start = max(0.0, center - width / 2.0)
        end = min(duration_s, start + width)
        start = max(0.0, end - width)
    else:
        current = list(map(float, current_interval))
        if len(current) != 2 or current[1] <= current[0]:
            raise ValueError("current_interval must have increasing endpoints")
        start, end = current
        room = max(0.0, max_context_s - (end - start))
        left_add = min(expansion_step_s if not left_context_complete else 0.0, room)
        room -= left_add
        right_add = min(expansion_step_s if not right_context_complete else 0.0, room)
        start = max(0.0, start - left_add)
        end = min(duration_s, end + right_add)
        # Use unused room at a source boundary on the opposite side.
        target_width = min(max_context_s, (current[1] - current[0]) +
                           left_add + right_add)
        if end - start < target_width:
            if start <= 1e-6:
                end = min(duration_s, target_width)
            elif end >= duration_s - 1e-6:
                start = max(0.0, duration_s - target_width)
    interval = [round(start, 6), round(end, 6)]
    return {
        "source_interval": interval,
        "duration_s": round(end - start, 6),
        "at_safety_limit": end - start >= max_context_s - 1e-6,
        "left_source_boundary": start <= 1e-6,
        "right_source_boundary": end >= duration_s - 1e-6,
    }


def _normalize_neutral_observation(payload: Mapping[str, Any], *,
                                   observation_id: str, start_s: float,
                                   end_s: float, source_video: Path,
                                   lead_refs: list[dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _contains_forbidden_key(payload):
        raise V8Blocked("occurrence", "neutral_observation_contaminated")
    duration = end_s - start_s
    occurrences: list[dict[str, Any]] = []
    local_map: dict[str, str] = {}
    for index, raw in enumerate(payload.get("occurrences") or []):
        if not isinstance(raw, Mapping):
            continue
        local_id = str(raw.get("local_id") or chr(65 + index))
        interval = _normalize_relative_interval(
            raw.get("visible_interval") or [0.0, duration], start_s=start_s,
            end_s=end_s, field="occurrence_visible_interval")
        occurrence_id = f"occ_{observation_id}_{_safe_id(local_id)}"
        if occurrence_id in local_map.values():
            raise V8Blocked("occurrence", "duplicate_local_occurrence_id")
        local_map[local_id] = occurrence_id
        row = {
            "schema_version": "occurrence_v1",
            "occurrence_id": occurrence_id,
            "source_interval": interval,
            "local_description": str(raw.get("local_description") or "").strip(),
            "visual_state": str(raw.get("visual_state") or "").strip(),
            "roi": raw.get("roi"),
            "frame_refs": [],
            "observation_id": observation_id,
            "observation_source": "targeted_neutral_omni",
            "lead_refs": lead_refs,
            "source_video": str(source_video),
            "status": "observed",
        }
        row["observation_sha256"] = _stable_sha(row)
        occurrences.append(row)
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("event_candidates") or []):
        if not isinstance(raw, Mapping):
            continue
        actor_local = str(raw.get("actor_local_id") or "")
        patient_local = str(raw.get("patient_local_id") or "")
        actor = local_map.get(actor_local)
        patient = local_map.get(patient_local) if patient_local else None
        if not actor:
            raise V8Blocked("occurrence", "event_actor_missing_occurrence")
        if patient_local and not patient:
            raise V8Blocked("occurrence", "event_patient_missing_occurrence")
        interval = _normalize_relative_interval(
            raw.get("interval") or [0.0, duration], start_s=start_s,
            end_s=end_s, field="event_candidate_interval")
        events.append({
            "event_candidate_id": f"candidate_{observation_id}_{index:02d}",
            "source_interval": interval,
            "actor_occurrence_id": actor,
            "action": str(raw.get("action") or "").strip(),
            "patient_occurrence_id": patient,
            "visible_result": str(raw.get("visible_result") or "").strip(),
            "observation_id": observation_id,
            "lead_refs": lead_refs,
            "source_video": str(source_video),
            "status": "neutral_observation_candidate",
        })
    return occurrences, events


def build_occurrence_bank_from_leads(cfg: AppConfig, spec: Mapping[str, Any],
                                     leads: Iterable[Mapping[str, Any]],
                                     output_dir: Path, *, source_video: Path,
                                     runner: Any,
                                     source_sha256: str | None = None,
                                     reuse_completed: bool = True,
                                     batch_size: int = 32) -> dict[str, Any]:
    """Use resumable, name-free Omni rewatches to create occurrence truth."""
    output_dir = Path(output_dir)
    duration_s = common.video_duration_s(
        cfg.perception.get("ffprobe_bin", "ffprobe"), source_video)
    windows = investigation_windows(leads, duration_s=duration_s, window_s=6.0)
    if batch_size < 1:
        raise ValueError("occurrence observation batch_size must be positive")
    contract = {
        "schema_version": "neutral_occurrence_contract_v2",
        "source_video": str(source_video),
        "source_sha256": source_sha256 or sha256_file(source_video),
        "prompt_sha256": hashlib.sha256(
            NEUTRAL_OCCURRENCE_PROMPT.encode("utf-8")).hexdigest(),
        "fps": 12.0,
        "window_s": 6.0,
        "use_audio_in_video": False,
    }
    contract_sha256 = _stable_sha(contract)
    forbidden_terms = {
        str(value).strip().lower()
        for character in spec.get("characters") or []
        for value in ([character.get("display_name")]
                      + list(character.get("aliases") or []))
        if str(value or "").strip()
    }
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    cached_by_interval: dict[tuple[float, float], dict[str, Any]] = {}
    if reuse_completed and output_dir.is_dir():
        for cached_path in output_dir.glob("*/result.json"):
            try:
                cached = _read_json(cached_path)
                interval = tuple(map(float, cached.get("source_interval") or []))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if (len(interval) == 2 and
                    cached.get("contract_sha256") == contract_sha256 and
                    cached.get("status") in {"observed", "observed_empty"}):
                cached_by_interval[interval] = cached
    for window in windows:
        result_path = output_dir / window["observation_id"] / "result.json"
        cached = None
        if reuse_completed and result_path.is_file():
            direct = _read_json(result_path)
            cached = direct if direct.get("source_interval") == window["source_interval"] \
                else None
        if cached is None:
            cached = cached_by_interval.get(tuple(map(float, window["source_interval"])))
        if cached is not None:
            if (cached.get("contract_sha256") == contract_sha256
                    and cached.get("source_interval") == window["source_interval"]
                    and cached.get("status") in {"observed", "observed_empty"}):
                completed[window["observation_id"]] = cached
                continue
        pending.append(window)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        requests = [{
            "video_path": source_video,
            "prompt": NEUTRAL_OCCURRENCE_PROMPT,
            "kwargs": {
                "start_s": row["source_interval"][0],
                "end_s": row["source_interval"][1],
                "clip_dir": output_dir / row["observation_id"] / "source_clip",
                "fps": 12.0,
                "duration_s": row["source_interval"][1] - row["source_interval"][0],
                "max_new_tokens": 768,
                "use_audio_in_video": False,
            },
        } for row in batch]
        answers = runner.watch_many(requests)
        if len(answers) != len(batch):
            raise V8Blocked("occurrence", "neutral_observation_response_count_mismatch")
        for window, answer in zip(batch, answers):
            observation_id = window["observation_id"]
            observation_dir = output_dir / observation_id
            raw_path = observation_dir / "raw_response.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(str(answer.text), encoding="utf-8")
            _write_json(observation_dir / "response_envelope.json", {
                "sampling": getattr(answer, "sampling", None),
                "gpu_pair": getattr(answer, "gpu_pair", None),
                "raw_response": str(answer.text),
            })
            try:
                payload = _single_json_object(str(answer.text))
                serialized = json.dumps(payload, ensure_ascii=False).lower()
                if any(term in serialized for term in forbidden_terms):
                    raise V8Blocked("occurrence", "neutral_observation_named_character")
                observation_status = validate_neutral_observation(
                    payload, duration_s=(window["source_interval"][1] -
                                         window["source_interval"][0]))
                if observation_status == "observed":
                    new_occurrences, new_events = _normalize_neutral_observation(
                        payload, observation_id=observation_id,
                        start_s=window["source_interval"][0],
                        end_s=window["source_interval"][1], source_video=source_video,
                        lead_refs=window["lead_refs"])
                    status = "observed"
                else:
                    new_occurrences, new_events = [], []
                    status = observation_status
                result = {
                    "schema_version": "neutral_occurrence_result_v2",
                    "contract_sha256": contract_sha256,
                    "observation_id": observation_id,
                    "source_interval": window["source_interval"],
                    "lead_refs": window["lead_refs"],
                    "status": status,
                    "left_context_complete": payload.get("left_context_complete"),
                    "right_context_complete": payload.get("right_context_complete"),
                    "boundary_reason": str(payload.get("boundary_reason") or "").strip(),
                    "occurrences": new_occurrences,
                    "event_candidates": new_events,
                    "raw_response_path": str(raw_path),
                }
            except Exception as exc:
                result = {
                    "schema_version": "neutral_occurrence_result_v2",
                    "contract_sha256": contract_sha256,
                    "observation_id": observation_id,
                    "source_interval": window["source_interval"],
                    "lead_refs": window["lead_refs"],
                    "status": "failed",
                    "failure_class": "verification",
                    "failure_stage": "occurrence",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "raw_response_path": str(raw_path),
                }
            completed[observation_id] = result
            _write_json(observation_dir / "result.json", result)

    ordered = [completed[row["observation_id"]] for row in windows]
    occurrences = [item for result in ordered
                   for item in result.get("occurrences") or []]
    events = [item for result in ordered
              for item in result.get("event_candidates") or []]
    failures = [result for result in ordered if result.get("status") == "failed"]
    unreliable = [result for result in ordered if result.get("status") == "unreliable"]
    build_occurrence_bank([], occurrences, output_dir.parent / "occurrence_bank.jsonl")
    events.sort(key=lambda row: (
        float((row.get("source_interval") or [float("inf")])[0]),
        str(row.get("event_candidate_id") or "")))
    _write_jsonl(output_dir.parent / "event_candidates.jsonl", events)
    result = {
        "schema_version": "occurrence_observation_manifest_v2",
        "contract": contract, "contract_sha256": contract_sha256,
        "lead_count": sum(len(row["lead_refs"]) for row in windows),
        "observation_count": len(windows),
        "passed_observation_count": sum(
            row.get("status") in {"observed", "observed_empty"} for row in ordered),
        "observed_empty_count": sum(
            row.get("status") == "observed_empty" for row in ordered),
        "unreliable_count": len(unreliable),
        "failed_observation_count": len(failures),
        "complete": not failures,
        "reused_observation_count": len(windows) - len(pending),
        "occurrence_count": len(occurrences),
        "event_candidate_count": len(events),
        "coarse_coverage_promoted_directly": False,
        "failures": failures,
    }
    _write_json(output_dir / "occurrence_observation_manifest.json", result)
    return result


def build_context_watch_bank(cfg: AppConfig, spec: Mapping[str, Any],
                             candidates: Iterable[Mapping[str, Any]],
                             output_dir: Path, *, source_video: Path, runner: Any,
                             source_sha256: str | None = None,
                             initial_context_s: float = 10.0,
                             max_context_s: float = 40.0,
                             expansion_step_s: float = 4.0,
                             merge_gap_s: float = 1.0,
                             reuse_completed: bool = True) -> dict[str, Any]:
    """Understand merged candidate regions with iterative native-video context."""
    output_dir = Path(output_dir)
    duration_s = common.video_duration_s(
        cfg.perception.get("ffprobe_bin", "ffprobe"), source_video)
    merged = merge_candidate_regions(candidates, max_gap_s=merge_gap_s)
    contract = {
        "schema_version": "context_watch_contract_v1",
        "source_sha256": source_sha256 or sha256_file(source_video),
        "prompt_sha256": hashlib.sha256(
            NEUTRAL_OCCURRENCE_PROMPT.encode("utf-8")).hexdigest(),
        "fps": 12.0,
        "initial_context_s": float(initial_context_s),
        "max_context_s": float(max_context_s),
        "expansion_step_s": float(expansion_step_s),
        "merge_gap_s": float(merge_gap_s),
        "use_audio_in_video": False,
    }
    contract_sha = _stable_sha(contract)
    forbidden_terms = {
        str(value).strip().lower()
        for character in spec.get("characters") or []
        for value in ([character.get("display_name")]
                      + list(character.get("aliases") or []))
        if str(value or "").strip()
    }
    results: list[dict[str, Any]] = []
    all_occurrences: list[dict[str, Any]] = []
    verified_events: list[dict[str, Any]] = []
    partial_events: list[dict[str, Any]] = []
    for context in merged:
        context_id = str(context["context_id"])
        context_dir = output_dir / context_id
        result_path = context_dir / "result.json"
        if reuse_completed and result_path.is_file():
            cached = _read_json(result_path)
            if (cached.get("contract_sha256") == contract_sha and
                    cached.get("candidate_interval") == context["source_interval"] and
                    cached.get("status") in {"observed", "observed_empty"} and
                    cached.get("context_status") == "complete"):
                results.append(cached)
                all_occurrences.extend(cached.get("occurrences") or [])
                verified_events.extend(cached.get("event_candidates") or [])
                continue
        current = expand_boundary_context(
            context["source_interval"], duration_s=duration_s,
            initial_context_s=initial_context_s, max_context_s=max_context_s)
        attempts: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for attempt_index in range(1, 16):
            start_s, end_s = current["source_interval"]
            prompt = (
                NEUTRAL_OCCURRENCE_PROMPT
                + f"\nThe exact input duration is {end_s - start_s:.6f} seconds."
            )
            try:
                answer = runner.watch(
                    source_video, prompt, start_s=start_s, end_s=end_s,
                    clip_dir=context_dir / f"attempt_{attempt_index:02d}" / "source_clip",
                    fps=12.0, duration_s=end_s - start_s, max_new_tokens=768,
                    use_audio_in_video=False)
                raw_text = str(answer.text)
                attempt_dir = context_dir / f"attempt_{attempt_index:02d}"
                raw_path = attempt_dir / "raw_response.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(raw_text, encoding="utf-8")
                payload = _single_json_object(raw_text)
                serialized = json.dumps(payload, ensure_ascii=False).lower()
                if any(term in serialized for term in forbidden_terms):
                    raise V8Blocked("occurrence", "neutral_observation_named_character")
                observation_status = validate_neutral_observation(
                    payload, duration_s=end_s - start_s)
                left_complete = payload.get("left_context_complete")
                right_complete = payload.get("right_context_complete")
                if not isinstance(left_complete, bool) or not isinstance(
                        right_complete, bool):
                    raise V8Blocked("occurrence", "missing_context_completeness")
                if observation_status == "observed":
                    occurrences, events = _normalize_neutral_observation(
                        payload, observation_id=context_id, start_s=start_s,
                        end_s=end_s, source_video=source_video,
                        lead_refs=context["lead_refs"])
                else:
                    occurrences, events = [], []
                attempt = {
                    "attempt": attempt_index,
                    "source_interval": [start_s, end_s],
                    "duration_s": round(end_s - start_s, 6),
                    "status": observation_status,
                    "left_context_complete": left_complete,
                    "right_context_complete": right_complete,
                    "boundary_reason": str(payload.get("boundary_reason") or "").strip(),
                    "raw_response_path": str(raw_path),
                    "sampling": getattr(answer, "sampling", None),
                    "gpu_pair": getattr(answer, "gpu_pair", None),
                }
                attempts.append(attempt)
                incomplete = observation_status == "observed" and not (
                    left_complete and right_complete)
                if incomplete:
                    expanded = expand_boundary_context(
                        context["source_interval"], duration_s=duration_s,
                        current_interval=current["source_interval"],
                        left_context_complete=left_complete,
                        right_context_complete=right_complete,
                        initial_context_s=initial_context_s,
                        expansion_step_s=expansion_step_s,
                        max_context_s=max_context_s)
                    if expanded["source_interval"] != current["source_interval"]:
                        current = expanded
                        continue
                    context_status = "partial"
                elif observation_status == "unreliable":
                    context_status = "unreliable"
                else:
                    context_status = "complete"
                final = {
                    "schema_version": "context_watch_result_v1",
                    "contract_sha256": contract_sha,
                    "context_id": context_id,
                    "candidate_interval": context["source_interval"],
                    "source_interval": [start_s, end_s],
                    "lead_refs": context["lead_refs"],
                    "status": observation_status,
                    "context_status": context_status,
                    "left_context_complete": left_complete,
                    "right_context_complete": right_complete,
                    "boundary_reason": str(payload.get("boundary_reason") or "").strip(),
                    "occurrences": occurrences,
                    "event_candidates": events,
                    "attempts": attempts,
                }
                break
            except Exception as exc:
                final = {
                    "schema_version": "context_watch_result_v1",
                    "contract_sha256": contract_sha,
                    "context_id": context_id,
                    "candidate_interval": context["source_interval"],
                    "source_interval": current["source_interval"],
                    "lead_refs": context["lead_refs"],
                    "status": "failed",
                    "context_status": "failed",
                    "failure_class": "verification",
                    "failure_stage": getattr(exc, "stage", "context"),
                    "reason_code": getattr(exc, "reason_code",
                                               "context_watch_failed"),
                    "detail": str(exc),
                    "attempts": attempts,
                }
                break
        if final is None:
            raise AssertionError("context watch exhausted without result")
        _write_json(result_path, final)
        results.append(final)
        if final.get("status") == "observed":
            all_occurrences.extend(final.get("occurrences") or [])
            if final.get("context_status") == "complete":
                verified_events.extend(final.get("event_candidates") or [])
            else:
                partial_events.extend(final.get("event_candidates") or [])
    build_occurrence_bank([], all_occurrences, output_dir / "occurrence_bank.jsonl")
    _write_jsonl(output_dir / "event_candidates.jsonl", verified_events)
    _write_jsonl(output_dir / "partial_event_candidates.jsonl", partial_events)
    failures = [row for row in results if row.get("status") == "failed"]
    unreliable = [row for row in results if row.get("status") == "unreliable"]
    partial = [row for row in results if row.get("context_status") == "partial"]
    manifest = {
        "schema_version": "context_watch_manifest_v1",
        "contract": contract,
        "contract_sha256": contract_sha,
        "candidate_count": len(merged),
        "observed_count": sum(row.get("status") == "observed" for row in results),
        "observed_empty_count": sum(
            row.get("status") == "observed_empty" for row in results),
        "unreliable_count": len(unreliable),
        "partial_count": len(partial),
        "failed_count": len(failures),
        "occurrence_count": len(all_occurrences),
        "event_candidate_count": len(verified_events),
        "partial_event_candidate_count": len(partial_events),
        "complete": not failures and not partial,
        "results": results,
    }
    _write_json(output_dir / "context_watch_manifest.json", manifest)
    return manifest


def build_coverage_audit(spec: Mapping[str, Any], *, duration_s: float,
                         global_intervals: Iterable[Iterable[float]],
                         dense_intervals: Iterable[Iterable[float]],
                         legacy_intervals: Iterable[Iterable[float]],
                         output_path: Path) -> dict[str, Any]:
    """Persist comparable global, dense, and legacy time-union coverage."""
    coverage = spec.get("coverage") or {}
    eligible_start = (float(coverage.get("head_s", 90.0))
                      if duration_s >= float(coverage.get("min_movie_s", 1800.0))
                      else 0.0)
    eligible_end = (duration_s - float(coverage.get("tail_s", 360.0))
                    if duration_s >= float(coverage.get("min_movie_s", 1800.0))
                    else duration_s)
    eligible = [eligible_start, max(eligible_start, eligible_end)]
    result = {
        "schema_version": "v81_coverage_audit_v1",
        "movie_duration_s": round(duration_s, 6),
        "eligible_interval": eligible,
        "global_browse": compute_temporal_coverage(
            global_intervals, eligible_interval=eligible),
        "dense_context_watch": compute_temporal_coverage(
            dense_intervals, eligible_interval=eligible),
        "legacy_targeted_windows": compute_temporal_coverage(
            legacy_intervals, eligible_interval=eligible),
        "window_count_is_coverage": False,
    }
    _write_json(Path(output_path), result)
    return result


def select_v81_pilot_windows(spec: Mapping[str, Any], *, duration_s: float,
                             high_value_leads: Iterable[Mapping[str, Any]] = ()) \
        -> list[dict[str, Any]]:
    """Choose 8 known-hard, 6 high-value, and 6 reproducible random windows."""
    pilot = (spec.get("diagnostic") or {}).get("pilot") or {}
    window_s = float(pilot.get("browse_window_s", 45.0))
    known = list(pilot.get("known_hard") or [])
    if len(known) != 8:
        raise ValueError("V8.1 pilot requires exactly eight known-hard windows")

    def window(center: float, *, category: str, reason: str,
               source_ref: Any = None) -> dict[str, Any]:
        start = min(max(0.0, center - window_s / 2.0),
                    max(0.0, duration_s - window_s))
        end = min(duration_s, start + window_s)
        return {
            "pilot_id": "", "category": category,
            "source_interval": [round(start, 6), round(end, 6)],
            "reason": reason, "source_ref": source_ref,
        }

    selected = []
    for row in known:
        interval = list(map(float, row.get("source_interval") or []))
        if len(interval) != 2 or interval[1] <= interval[0]:
            raise ValueError("known-hard pilot interval is invalid")
        selected.append(window(
            sum(interval) / 2.0, category="known_hard",
            reason=str(row.get("reason") or "known hard case"),
            source_ref=row.get("id")))

    def overlaps_existing(candidate: dict[str, Any]) -> bool:
        start, end = candidate["source_interval"]
        return any(min(end, old["source_interval"][1]) -
                   max(start, old["source_interval"][0]) > window_s * 0.5
                   for old in selected)

    valid_high_value: dict[str, list[dict[str, Any]]] = {}
    for raw_lead in high_value_leads:
        lead = dict(raw_lead)
        interval = list(map(float, lead.get("source_interval") or []))
        if len(interval) != 2 or interval[1] <= interval[0]:
            continue
        kind = str(lead.get("lead_kind") or "high-value investigation lead")
        valid_high_value.setdefault(kind, []).append(lead)

    def spread_over_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: (
            sum(map(float, row["source_interval"])) / 2.0,
            str(row.get("lead_id") or "")))
        if len(ordered) <= 6:
            return ordered
        last = len(ordered) - 1
        anchors = []
        for position in (0, last, round(last / 2), round(last / 4),
                         round(3 * last / 4), round(last / 8)):
            if position not in anchors:
                anchors.append(position)
        anchor_set = set(anchors)
        return [ordered[index] for index in anchors] + [
            row for index, row in enumerate(ordered) if index not in anchor_set]

    queues = {
        kind: spread_over_timeline(rows)
        for kind, rows in sorted(valid_high_value.items())
    }
    while (sum(row["category"] == "high_value" for row in selected) < 6
           and any(queues.values())):
        progress = False
        for kind in sorted(queues):
            while queues[kind]:
                lead = queues[kind].pop(0)
                interval = list(map(float, lead["source_interval"]))
                candidate = window(
                    sum(interval) / 2.0, category="high_value",
                    reason=kind, source_ref=lead.get("lead_id"))
                if overlaps_existing(candidate):
                    continue
                selected.append(candidate)
                progress = True
                break
            if sum(row["category"] == "high_value" for row in selected) >= 6:
                break
        if not progress:
            break
    if sum(row["category"] == "high_value" for row in selected) != 6:
        raise ValueError("V8.1 pilot requires at least six distinct high-value leads")

    coverage = spec.get("coverage") or {}
    eligible_start = (float(coverage.get("head_s", 90.0))
                      if duration_s >= float(coverage.get("min_movie_s", 1800.0))
                      else 0.0)
    eligible_end = (duration_s - float(coverage.get("tail_s", 360.0))
                    if duration_s >= float(coverage.get("min_movie_s", 1800.0))
                    else duration_s)
    rng = random.Random(int(pilot.get("random_seed", 20260915)))
    attempts = 0
    while sum(row["category"] == "random" for row in selected) < 6 and attempts < 1000:
        attempts += 1
        center = rng.uniform(eligible_start + window_s / 2.0,
                             eligible_end - window_s / 2.0)
        candidate = window(center, category="random",
                           reason="reproducible_uniform_random_control")
        if not overlaps_existing(candidate):
            selected.append(candidate)
    if sum(row["category"] == "random" for row in selected) != 6:
        raise ValueError("could not select six distinct random pilot windows")
    category_order = {"known_hard": 0, "random": 1, "high_value": 2}
    selected.sort(key=lambda row: (category_order[row["category"]],
                                   row["source_interval"][0]))
    for index, row in enumerate(selected):
        row["pilot_id"] = f"pilot_{index:02d}"
    return selected


def evaluate_browse_arms(arms: Mapping[str, Mapping[str, Any]],
                         ground_truth: Iterable[Mapping[str, Any]] | None = None) \
        -> dict[str, Any]:
    """Compare candidate recall post hoc; ground truth never enters prompts."""
    gt = [dict(row) for row in (ground_truth or [])]
    result = {"schema_version": "v81_browse_arm_comparison_v1", "arms": {}}
    for arm_id in ("A", "B", "C"):
        arm = dict(arms.get(arm_id) or {})
        candidates = [dict(row) for row in arm.get("candidates") or []]
        merged = merge_candidate_regions(candidates, max_gap_s=0.15)
        metrics = {
            "window_count": int(arm.get("window_count") or 0),
            "candidate_count": len(candidates),
            "unique_candidate_count": len(merged),
            "duplicate_rate": round(
                1.0 - len(merged) / len(candidates), 6) if candidates else 0.0,
            "visual_tokens": int(arm.get("visual_tokens") or 0),
            "latency_s": round(float(arm.get("latency_s") or 0), 6),
            "failure_count": int(arm.get("failure_count") or 0),
        }
        if gt:
            hits = 0
            per_goal: dict[str, list[bool]] = {}
            for truth in gt:
                target = list(map(float, truth.get("source_interval") or []))
                hit = len(target) == 2 and any(
                    max(target[0], row["source_interval"][0]) <
                    min(target[1], row["source_interval"][1])
                    for row in candidates)
                hits += int(hit)
                per_goal.setdefault(str(truth.get("goal") or "unspecified"), []).append(hit)
            metrics["candidate_temporal_recall"] = round(hits / len(gt), 6)
            metrics["goal_recall"] = {
                goal: round(sum(values) / len(values), 6)
                for goal, values in per_goal.items()
            }
            metrics["gt_count"] = len(gt)
        result["arms"][arm_id] = metrics
    result["ground_truth_status"] = "available" if gt else "awaiting_human_ground_truth"
    if gt:
        recalls = {arm: row["candidate_temporal_recall"]
                   for arm, row in result["arms"].items()}
        result["diagnosis"] = (
            "temporal_sampling_bottleneck" if recalls["B"] > recalls["A"] else
            "compression_bottleneck" if recalls["C"] > recalls["B"] else
            "lowest_cost_arm_sufficient" if len(set(recalls.values())) == 1 else
            "mixed_or_prompt_bottleneck")
    else:
        result["diagnosis"] = "pending_human_ground_truth"
    return result


def run_v81_diagnostic(cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path, *,
                       clients: Mapping[str, Any], source_video: Path,
                       pilot_windows: Iterable[Mapping[str, Any]],
                       source_sha256: str | None = None,
                       ground_truth: Iterable[Mapping[str, Any]] | None = None) \
        -> dict[str, Any]:
    """Run A/B/C browse recall arms without repeating downstream Omni work."""
    output_dir = Path(output_dir)
    windows = [dict(row) for row in pilot_windows]
    if len(windows) != 20:
        raise ValueError("V8.1 diagnostic requires exactly twenty pilot windows")
    arms_spec = {
        str(row["id"]): dict(row)
        for row in ((spec.get("diagnostic") or {}).get("browse_arms") or [])
    }
    if set(arms_spec) != {"A", "B", "C"} or set(clients) != {"A", "B", "C"}:
        raise ValueError("V8.1 diagnostic requires A/B/C arms and clients")
    _write_json(output_dir / "pilot_windows.json", {
        "schema_version": "v81_pilot_windows_v1", "windows": windows,
    })
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    ffprobe = str(cfg.perception.get("ffprobe_bin", "ffprobe"))
    source_probe = common.run_ffprobe_json(ffprobe, source_video)
    stream = next((row for row in source_probe.get("streams") or []
                   if row.get("codec_type") == "video"), {})
    rate_text = str(stream.get("avg_frame_rate") or "0/1")
    try:
        numerator, denominator = rate_text.split("/", 1)
        source_fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        source_fps = None
    source_digest = source_sha256 or sha256_file(source_video)

    def request_cost(attempts: Iterable[Mapping[str, Any]]) -> tuple[int, float]:
        visual_tokens = 0
        latency_s = 0.0
        for attempt in attempts:
            audit = dict(attempt.get("request_audit") or {})
            usage = dict(audit.get("usage") or {})
            details = dict(usage.get("prompt_tokens_details") or {})
            multimodal = dict(details.get("multimodal_tokens") or {})
            visual_tokens += int(
                usage.get("visual_tokens") or details.get("visual_tokens") or
                multimodal.get("video") or 0)
            latency_s += float(audit.get("latency_s") or 0.0)
        return visual_tokens, latency_s

    def reusable_result(result_path: Path, *, arm: Mapping[str, Any],
                        source_interval: list[float]) -> dict[str, Any] | None:
        if not result_path.is_file():
            return None
        try:
            result = _read_json(result_path)
        except (OSError, json.JSONDecodeError):
            return None
        audit = dict(result.get("request_audit") or {})
        sampling = dict(result.get("sampling_audit") or {})
        if (result.get("status") != "observed" or
                list(map(float, result.get("source_interval") or [])) != source_interval or
                float(audit.get("requested_fps") or -1) != float(arm["fps"]) or
                float(audit.get("retention_ratio") or -1) !=
                float(arm["retention_ratio"]) or
                sampling.get("sampling_verified") is not True or
                not result.get("attempt_audits")):
            return None
        return result

    def stage(clip: Path, arm: Mapping[str, Any], name: str) -> Path:
        media_root_text = str(arm.get("media_root") or "").strip()
        if not media_root_text:
            return clip
        media_root = Path(media_root_text).resolve()
        media_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                prefix=f"v81_{_safe_id(name)}_", suffix=".mp4",
                dir=media_root, delete=False) as handle:
            staged = Path(handle.name)
        shutil.copy2(clip, staged)
        if sha256_file(staged) != sha256_file(clip):
            staged.unlink(missing_ok=True)
            raise V8Blocked("pilot", "media_staging_hash_mismatch")
        return staged

    def run_arm(arm_id: str) -> tuple[str, dict[str, Any]]:
        arm = arms_spec[arm_id]
        client = clients[arm_id]
        arm_dir = output_dir / "arms" / arm_id
        rows, candidates, failures = [], [], []
        for window in windows:
            pilot_id = str(window["pilot_id"])
            start_s, end_s = map(float, window["source_interval"])
            window_dir = arm_dir / pilot_id
            window_dir.mkdir(parents=True, exist_ok=True)
            result_path = window_dir / "result.json"
            cached = reusable_result(
                result_path, arm=arm, source_interval=[start_s, end_s])
            if cached is not None:
                rows.append(cached)
                candidates.extend(cached.get("candidates") or [])
                continue
            attempt_audits: list[dict[str, Any]] = []
            sampling: dict[str, Any] = {}
            try:
                clip, sampling = _build_coverage_transport(
                    ffmpeg, ffprobe, source_video, window_dir / "transport",
                    start_s=start_s, end_s=end_s,
                    requested_fps=float(arm["fps"]), source_fps=source_fps)
                for attempt in range(1, 3):
                    request_clip = stage(
                        clip, arm, f"{arm_id}_{pilot_id}_attempt_{attempt:02d}")
                    retry_suffix = ("" if attempt == 1 else
                        "\nCONTRACT RETRY: Do not return fixed 10- or 15-second bins. "
                        "Return at most three tight 2-6 second excerpts, or an empty "
                        "regions array. ROI values, when present, are normalized image "
                        "coordinates in [0,1], never timestamps.")
                    try:
                        answer = client.watch(
                            request_clip, NEUTRAL_COVERAGE_PROMPT + retry_suffix,
                            duration_s=end_s - start_s, image_paths=None,
                            max_tokens=1024, response_format=COVERAGE_RESPONSE_FORMAT)
                    finally:
                        if request_clip != clip:
                            request_clip.unlink(missing_ok=True)
                    attempt_raw = window_dir / f"raw_response_attempt_{attempt:02d}.txt"
                    attempt_raw.write_text(str(answer.text), encoding="utf-8")
                    attempt_audit = {
                        "attempt": attempt,
                        "raw_response_path": str(attempt_raw),
                        "request_audit": dict(
                            getattr(answer, "request_audit", {}) or {}),
                    }
                    try:
                        payload, response_shape = _parse_coverage_payload(str(answer.text))
                        if response_shape != "object":
                            raise V8Blocked("pilot", "browse_response_not_object")
                        normalized_occurrences, normalized_events = (
                            _normalize_coverage_response(
                                payload, block_id=f"{arm_id}_{pilot_id}",
                                start_s=start_s, end_s=end_s))
                    except Exception as exc:
                        attempt_audit["validation_error"] = (
                            f"{type(exc).__name__}: {exc}")
                        attempt_audits.append(attempt_audit)
                        if attempt == 2:
                            raise
                        continue
                    attempt_audit["validation_error"] = None
                    attempt_audits.append(attempt_audit)
                    break
                raw_path = window_dir / "raw_response.txt"
                raw_path.write_text(str(answer.text), encoding="utf-8")
                window_candidates = []
                for index, region in enumerate(payload.get("regions") or []):
                    rel = list(map(float, region.get("interval") or []))
                    if len(rel) != 2 or not 0 <= rel[0] < rel[1] <= end_s - start_s:
                        continue
                    candidate = {
                        "candidate_id": f"{arm_id}_{pilot_id}_{index:02d}",
                        "arm": arm_id, "pilot_id": pilot_id,
                        "source_interval": [round(start_s + rel[0], 6),
                                            round(start_s + rel[1], 6)],
                        "activity": _concrete_text(
                            region.get("activity"), field="pilot_activity"),
                        "lead_kind": "pilot_browse",
                        "lead_id": f"{arm_id}_{pilot_id}_{index:02d}",
                    }
                    candidates.append(candidate)
                    window_candidates.append(candidate)
                result = {
                    "schema_version": "v81_pilot_browse_result_v1",
                    "pilot_id": pilot_id, "category": window.get("category"),
                    "source_interval": [start_s, end_s], "status": "observed",
                    "response_shape": response_shape, "sampling_audit": sampling,
                    "request_audit": dict(getattr(answer, "request_audit", {}) or {}),
                    "attempt_audits": attempt_audits,
                    "raw_response_path": str(raw_path),
                    "occurrence_candidate_count": len(normalized_occurrences),
                    "event_candidate_count": len(normalized_events),
                    "candidates": window_candidates,
                }
            except Exception as exc:
                result = {
                    "schema_version": "v81_pilot_browse_result_v1",
                    "pilot_id": pilot_id, "category": window.get("category"),
                    "source_interval": [start_s, end_s], "status": "failed",
                    "failure_class": "verification",
                    "failure_stage": getattr(exc, "stage", "pilot"),
                    "reason_code": getattr(exc, "reason_code", "pilot_browse_failed"),
                    "detail": str(exc),
                    "sampling_audit": sampling,
                    "attempt_audits": attempt_audits,
                }
                failures.append(result)
            _write_json(result_path, result)
            rows.append(result)
        total_visual_tokens = 0
        total_latency = 0.0
        for row in rows:
            visual_tokens, latency_s = request_cost(row.get("attempt_audits") or [])
            total_visual_tokens += visual_tokens
            total_latency += latency_s
        failures = [row for row in rows if row.get("status") == "failed"]
        arm_result = {
            "schema_version": "v81_pilot_arm_v1", "arm": arm_id,
            "fps": float(arm["fps"]),
            "retention_ratio": float(arm["retention_ratio"]),
            "backend": str(arm["backend"]), "window_count": len(windows),
            "candidate_count": len(candidates), "candidates": candidates,
            "visual_tokens": total_visual_tokens,
            "latency_s": round(total_latency, 6),
            "failure_count": len(failures), "failures": failures,
            "complete": not failures, "results": rows,
        }
        _write_json(arm_dir / "arm_manifest.json", arm_result)
        return arm_id, arm_result

    arm_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_arm, arm_id) for arm_id in ("A", "B", "C")]
        for future in as_completed(futures):
            arm_id, result = future.result()
            arm_results[arm_id] = result
    comparison = evaluate_browse_arms(arm_results, ground_truth)
    _write_json(output_dir / "arm_comparison.json", comparison)
    combined = []
    for arm_id in ("A", "B", "C"):
        combined.extend(arm_results[arm_id]["candidates"])
    selected = merge_candidate_regions(combined, max_gap_s=1.0)
    for row in selected:
        row["lead_kind"] = "v81_pilot_browse_union"
        row["lead_id"] = row["context_id"]
    _write_jsonl(output_dir / "selected_candidates.jsonl", selected)
    result = {
        "schema_version": "v81_diagnostic_manifest_v1",
        "source_sha256": source_digest,
        "pilot_window_count": len(windows),
        "arms": {arm: {key: value for key, value in arm_results[arm].items()
                         if key not in {"results", "candidates", "failures"}}
                 for arm in ("A", "B", "C")},
        "selected_candidate_count": len(selected),
        "arm_comparison": comparison,
        "complete": all(arm_results[arm]["complete"] for arm in ("A", "B", "C")),
        "identity_or_render_started": False,
    }
    _write_json(output_dir / "experiment_manifest.json", result)
    return result


def event_fact_hash(event_fact: Mapping[str, Any]) -> str:
    if _contains_key_recursive(event_fact, "character_id"):
        raise ValueError("Event Fact must not contain character_id")
    return _stable_sha(event_fact)


def _contains_key_recursive(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key_recursive(item, key)
                                   for item in value.values())
    if isinstance(value, list):
        return any(_contains_key_recursive(item, key) for item in value)
    return False


def build_seed_review_sheet(cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path,
                            *, source_video: Path) -> dict[str, Any]:
    """Export human-reviewable seed proposals without treating them as truth."""
    output_dir = Path(output_dir)
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    proposals: list[dict[str, Any]] = []
    configured = spec.get("seed_proposals") or {}
    for character in spec.get("characters") or []:
        character_id = str(character["character_id"])
        row = configured.get(character_id) or {}
        forms = row.get("forms") or [row]
        for form_index, form in enumerate(forms):
            form_id = form.get("form_id") or row.get("form_id")
            for index, timestamp in enumerate(form.get("source_times_s") or []):
                frame = export_frame(
                    ffmpeg, source_video, float(timestamp),
                    output_dir / character_id.replace(":", "_") /
                    f"form_{form_index:02d}_candidate_{index:02d}.jpg")
                proposals.append({
                    "candidate_id": (
                        f"{character_id}:form_{form_index:02d}:seed_candidate:{index:02d}"),
                    "character_id": character_id,
                    "source_time_s": float(timestamp),
                    "full_frame": str(frame),
                    "sha256": sha256_file(frame),
                    "status": "awaiting_human_review",
                    "suggested_form_id": form_id,
                })
        invalid_times = list(row.get("invalid_source_times_s") or [])
        invalid_times.extend(value for form in forms
                             for value in form.get("invalid_source_times_s") or [])
        for timestamp in dict.fromkeys(invalid_times):
            proposals.append({
                "candidate_id": f"{character_id}:invalid_regression:{float(timestamp):.3f}",
                "character_id": character_id,
                "source_time_s": float(timestamp),
                "status": "forbidden_regression_example",
                "reason": "known_dissolve_or_wrong_identity_seed",
            })
    result = {
        "schema_version": "seed_review_sheet_v1",
        "source_video": str(source_video),
        "source_sha256": sha256_file(source_video),
        "requires_human_confirmation": True,
        "proposals": proposals,
        "characters_missing_proposals": [
            row["character_id"] for row in spec.get("characters") or []
            if not any(item["character_id"] == row["character_id"]
                       and item["status"] == "awaiting_human_review"
                       for item in proposals)
        ],
        "instructions": (
            "For every form, select one clean trusted seed, two cross-shot positives, "
            "and one confusing hard negative. Model proposals are not identity truth."),
    }
    _write_json(output_dir / "seed_review.json", result)
    return result


IDENTITY_COMPARE_PROMPT = """This is an identity-only comparison. The first image is
the subject under test and the second image is a trusted comparison example. Full frames
are identity evidence; any following ROI crops only magnify those same frames. Do not use
actions, story roles, character names, or edit goals. Return strict JSON only:
{"result":"same|different|uncertain","visible_basis":["brief visual facts"]}
Use uncertain whenever the views, occlusion, transformation, or transition do not support
a reliable decision. Do not output a numeric confidence."""


def _identity_result(text: str) -> tuple[str, dict[str, Any]]:
    parsed = _parse_object(text)
    result = str(parsed.get("result") or "").lower().strip()
    if result not in IDENTITY_COMPARE_RESULTS:
        raise V8Blocked("identity", "invalid_identity_response", result)
    if "confidence" in parsed or "margin" in parsed:
        parsed.pop("confidence", None)
        parsed.pop("margin", None)
    return result, parsed


def _inspect_images(runner: Any, image_paths: list[Path], prompt: str) -> Any:
    return runner.inspect_media(image_paths, prompt, max_new_tokens=256)


def _directed_identity_compare(runner: Any, left: Path, right: Path, *,
                               left_roi: Path | None = None,
                               right_roi: Path | None = None) -> dict[str, Any]:
    images = [Path(left), Path(right)]
    # A crop may help, but never replaces either full frame.
    images.extend(path for path in (left_roi, right_roi) if path is not None)
    answer = _inspect_images(runner, images, IDENTITY_COMPARE_PROMPT)
    result, parsed = _identity_result(str(answer.text))
    return {
        "left": str(left), "right": str(right), "input_images": [str(p) for p in images],
        "full_frames_present": str(left) in {str(p) for p in images}
        and str(right) in {str(p) for p in images},
        "result": result, "parsed": parsed, "raw_response": str(answer.text),
    }


def _occurrence_profile_request(candidate: Path, candidate_roi: Path | None,
                                profile: Mapping[str, Any], *, reverse: bool) \
        -> tuple[list[Path], str, dict[str, Any]]:
    examples = profile.get("examples") or {}
    seed = Path(str((examples.get("seed") or {}).get("full_frame") or ""))
    negative_key = next((key for key in sorted(examples)
                         if key.startswith("negative_")), None)
    negative = Path(str((examples.get(negative_key) or {}).get("full_frame") or "")) \
        if negative_key else None
    if not seed.is_file() or negative is None or not negative.is_file():
        raise V8Blocked("identity", "profile_comparison_images_missing",
                        str(profile.get("form_id")))
    if reverse:
        images = [seed, candidate]
        subject = "Image 1 is the trusted positive; Image 2 is the candidate."
    else:
        images = [candidate, seed]
        subject = "Image 1 is the candidate; Image 2 is the trusted positive."
    if candidate_roi is not None:
        images.append(candidate_roi)
    images.append(negative)
    prompt = (
        "Identity-only profile comparison. " + subject +
        " The last image is a visually confusing hard negative. Any candidate ROI is "
        "only a magnified view paired with the candidate full frame. Ignore actions, "
        "character names, story roles, and editing goals. Decide whether the candidate "
        "and trusted positive depict the same subject. Return strict JSON only: "
        '{"result":"same|different|uncertain","visible_basis":["brief facts"]}. '
        "Do not output numeric confidence.")
    metadata = {
        "direction": "profile_to_candidate" if reverse else "candidate_to_profile",
        "input_images": [str(path) for path in images], "candidate_full": str(candidate),
        "trusted_positive": str(seed), "hard_negative": str(negative),
        "candidate_full_present": candidate in images,
        "hard_negative_present": negative in images,
    }
    return images, prompt, metadata


def _occurrence_profile_compare(runner: Any, candidate: Path,
                                candidate_roi: Path | None,
                                profile: Mapping[str, Any], *, reverse: bool) -> dict[str, Any]:
    images, prompt, metadata = _occurrence_profile_request(
        candidate, candidate_roi, profile, reverse=reverse)
    answer = _inspect_images(runner, images, prompt)
    result, parsed = _identity_result(str(answer.text))
    return {**metadata,
        "result": result, "parsed": parsed, "raw_response": str(answer.text),
    }


def _export_identity_example(ffmpeg: str, source_video: Path, example: Mapping[str, Any],
                             destination: Path) -> tuple[Path, Path | None, dict[str, Any]]:
    if example.get("frame_path"):
        full = Path(str(example["frame_path"]))
        if not full.is_absolute():
            full = repo_root() / full
        if not full.is_file():
            raise FileNotFoundError(full)
    else:
        timestamp = float(example["source_time_s"])
        full = export_frame(ffmpeg, source_video, timestamp, destination / "full.jpg")
    roi_path = None
    if example.get("roi") is not None:
        roi_path = export_roi(ffmpeg, full, example["roi"], destination / "roi.jpg")
    audit = {
        "source_time_s": example.get("source_time_s"), "full_frame": str(full),
        "full_frame_sha256": sha256_file(full), "roi": example.get("roi"),
        "roi_crop": str(roi_path) if roi_path else None,
        "roi_crop_sha256": sha256_file(roi_path) if roi_path else None,
    }
    return full, roi_path, audit


def build_character_profiles(cfg: AppConfig, spec: Mapping[str, Any],
                             seed_manifest: Mapping[str, Any], output_dir: Path, *,
                             source_video: Path, runner: Any) -> dict[str, Any]:
    """Validate human-confirmed forms through an all-directed-pairs contract."""
    output_dir = Path(output_dir)
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    invalid: set[tuple[str, float]] = set()
    for character_id, row in (spec.get("seed_proposals") or {}).items():
        values = list(row.get("invalid_source_times_s") or [])
        values.extend(value for form in row.get("forms") or []
                      for value in form.get("invalid_source_times_s") or [])
        invalid.update((str(character_id), round(float(value), 3)) for value in values)
    profiles: list[dict[str, Any]] = []
    manifest_chars = seed_manifest.get("characters") or []
    for character in manifest_chars:
        character_id = str(character.get("character_id") or "")
        for form in character.get("forms") or []:
            form_id = str(form.get("form_id") or "")
            seed = dict(form.get("trusted_seed") or {})
            positives = [dict(row) for row in form.get("positive_examples") or []]
            negatives = [dict(row) for row in form.get("hard_negatives") or []]
            if seed.get("confirmed_by") != "human":
                raise V8Blocked("identity", "trusted_seed_not_human_confirmed", form_id)
            if len(positives) < 2 or not negatives:
                raise V8Blocked("identity", "insufficient_form_examples", form_id)
            seed_time = seed.get("source_time_s")
            if seed_time is not None and (character_id, round(float(seed_time), 3)) in invalid:
                raise V8Blocked("identity", "forbidden_regression_seed", form_id)
            form_dir = output_dir / character_id.replace(":", "_") / _safe_id(form_id)
            exported: dict[str, tuple[Path, Path | None, dict[str, Any]]] = {}
            examples = [("seed", seed)]
            examples += [(f"positive_{index:02d}", row)
                         for index, row in enumerate(positives)]
            examples += [(f"negative_{index:02d}", row)
                         for index, row in enumerate(negatives)]
            for example_id, example in examples:
                exported[example_id] = _export_identity_example(
                    ffmpeg, source_video, example, form_dir / "examples" / example_id)
            positive_ids = ["seed"] + [f"positive_{index:02d}"
                                         for index in range(len(positives))]
            negative_ids = [f"negative_{index:02d}" for index in range(len(negatives))]
            comparisons: list[dict[str, Any]] = []
            for left_id, right_id in itertools.permutations(positive_ids, 2):
                left, left_roi, _ = exported[left_id]
                right, right_roi, _ = exported[right_id]
                comparison = _directed_identity_compare(
                    runner, left, right, left_roi=left_roi, right_roi=right_roi)
                comparison.update({"left_id": left_id, "right_id": right_id,
                                   "expected": "same"})
                comparisons.append(comparison)
            for positive_id in positive_ids:
                for negative_id in negative_ids:
                    for left_id, right_id in ((positive_id, negative_id),
                                              (negative_id, positive_id)):
                        left, left_roi, _ = exported[left_id]
                        right, right_roi, _ = exported[right_id]
                        comparison = _directed_identity_compare(
                            runner, left, right, left_roi=left_roi, right_roi=right_roi)
                        comparison.update({"left_id": left_id, "right_id": right_id,
                                           "expected": "different"})
                        comparisons.append(comparison)
            passed = all(row["result"] == row["expected"] for row in comparisons)
            profile = {
                "schema_version": "character_form_profile_v1",
                "character_id": character_id, "form_id": form_id,
                "status": "usable" if passed else "conflict",
                "profile_state": "human_seed_model_expanded",
                "examples": {key: value[2] for key, value in exported.items()},
                "comparisons": comparisons,
                "confidence_threshold_used": False, "margin_threshold_used": False,
            }
            profile["profile_version"] = _stable_sha(profile)
            profiles.append(profile)
            _write_json(form_dir / "profile.json", profile)
    result = {
        "schema_version": "character_profiles_v1", "profiles": profiles,
        "usable_form_count": sum(row["status"] == "usable" for row in profiles),
        "conflict_form_count": sum(row["status"] != "usable" for row in profiles),
    }
    _write_json(output_dir / "profiles_manifest.json", result)
    return result


def bind_occurrence_identities(cfg: AppConfig, occurrences: Iterable[Mapping[str, Any]],
                               profiles_manifest: Mapping[str, Any], output_dir: Path, *,
                               source_video: Path, runner: Any,
                               occurrence_ids: set[str] | None = None) -> dict[str, Any]:
    """Overlay revocable identity bindings; never edit occurrences or event facts."""
    output_dir = Path(output_dir)
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    usable = [row for row in profiles_manifest.get("profiles") or []
              if row.get("status") == "usable"]
    if not usable:
        raise V8Blocked("identity", "no_usable_character_profile")
    prepared_occurrences: list[tuple[dict[str, Any], Path, Path | None]] = []
    for occurrence in occurrences:
        occurrence = dict(occurrence)
        occurrence_id = str(occurrence.get("occurrence_id") or "")
        if occurrence_ids is not None and occurrence_id not in occurrence_ids:
            continue
        interval = list(map(float, occurrence.get("source_interval") or []))
        if len(interval) != 2 or interval[1] <= interval[0]:
            continue
        frame_dir = output_dir / "occurrence_frames" / _safe_id(occurrence_id)
        frame = export_frame(
            ffmpeg, source_video, sum(interval) / 2.0, frame_dir / "full.jpg")
        roi_frame = None
        if occurrence.get("roi") is not None:
            try:
                roi_frame = export_roi(ffmpeg, frame, occurrence["roi"], frame_dir / "roi.jpg")
            except (TypeError, ValueError, common.FFmpegError):
                roi_frame = None
        prepared_occurrences.append((occurrence, frame, roi_frame))

    requests: list[dict[str, Any]] = []
    request_meta: list[tuple[str, int, bool, dict[str, Any]]] = []
    for occurrence, frame, roi_frame in prepared_occurrences:
        occurrence_id = str(occurrence["occurrence_id"])
        for profile_index, profile in enumerate(usable):
            for reverse in (False, True):
                images, prompt, metadata = _occurrence_profile_request(
                    frame, roi_frame, profile, reverse=reverse)
                requests.append({"image_paths": images, "prompt": prompt,
                                 "kwargs": {"max_new_tokens": 256}})
                request_meta.append((occurrence_id, profile_index, reverse, metadata))
    if hasattr(runner, "inspect_media_many"):
        answers = runner.inspect_media_many(requests)
    else:
        answers = [runner.inspect_media(
            row["image_paths"], row["prompt"], **row["kwargs"]) for row in requests]
    comparisons_by_occurrence: dict[str, dict[int, dict[str, Any]]] = {}
    for meta, answer in zip(request_meta, answers):
        occurrence_id, profile_index, reverse, metadata = meta
        result, parsed = _identity_result(str(answer.text))
        comparison = {**metadata, "result": result, "parsed": parsed,
                      "raw_response": str(answer.text)}
        pair = comparisons_by_occurrence.setdefault(occurrence_id, {}).setdefault(
            profile_index, {
                "character_id": usable[profile_index]["character_id"],
                "form_id": usable[profile_index]["form_id"],
                "profile_version": usable[profile_index]["profile_version"],
            })
        pair["reverse" if reverse else "forward"] = comparison

    bindings: list[dict[str, Any]] = []
    for occurrence, _frame, _roi_frame in prepared_occurrences:
        occurrence_id = str(occurrence["occurrence_id"])
        comparisons = [comparisons_by_occurrence[occurrence_id][index]
                       for index in range(len(usable))]
        matched: list[dict[str, Any]] = []
        had_uncertain = False
        for profile, pair in zip(usable, comparisons):
            forward, reverse = pair["forward"], pair["reverse"]
            pair["symmetric"] = forward["result"] == reverse["result"]
            if forward["result"] == reverse["result"] == "same":
                matched.append(profile)
            if "uncertain" in {forward["result"], reverse["result"]}:
                had_uncertain = True
        if len(matched) == 1:
            selected = matched[0]
            status = "verified"
            character_id = selected["character_id"]
            form_id = selected["form_id"]
            reason = "single_bidirectionally_consistent_profile_match"
        elif len(matched) > 1:
            status, character_id, form_id = "conflict", None, None
            reason = "multiple_character_profiles_match"
        else:
            status, character_id, form_id = "uncertain", None, None
            reason = ("identity_comparison_uncertain" if had_uncertain
                      else "unmatched_occurrence_kept_unlabeled")
        discovery_state = None
        if status == "uncertain" and occurrence.get("continuous_character_context"):
            discovery_state = "unknown_form_candidate"
        row = {
            "schema_version": "identity_binding_v1",
            "occurrence_id": occurrence_id, "character_id": character_id,
            "form_id": form_id, "status": status, "reason": reason,
            "unknown_form_state": discovery_state or "unlabeled_cluster",
            "evidence": comparisons,
            "occurrence_observation_sha256": occurrence.get("observation_sha256"),
            "profile_version": matched[0]["profile_version"] if len(matched) == 1 else None,
            "candidate_pool_may_remain_uncertain": True,
        }
        row["binding_sha256"] = _stable_sha(row)
        bindings.append(row)
    verified = [row for row in bindings if row["status"] == "verified"]
    result = {
        "schema_version": "identity_binding_manifest_v1", "bindings": bindings,
        "metrics": {
            "total": len(bindings), "verified": len(verified),
            "uncertain": sum(row["status"] == "uncertain" for row in bindings),
            "conflict": sum(row["status"] == "conflict" for row in bindings),
            "verified_coverage": round(len(verified) / len(bindings), 6) if bindings else 0.0,
            "false_merge": None, "false_rejection": None,
            "uncertain_threshold_used": False,
        },
    }
    _write_jsonl(output_dir.parent / "identity_bindings.jsonl", bindings)
    _write_json(output_dir / "identity_manifest.json", result)
    return result


def evaluate_identity_bindings(bindings: Iterable[Mapping[str, Any]],
                               heldout_gt: Mapping[str, str]) -> dict[str, Any]:
    """Separate false merge, false rejection, and coverage; uncertainty is allowed."""
    rows = {str(row["occurrence_id"]): row for row in bindings}
    false_merge = 0
    false_rejection = 0
    verified = 0
    audited = 0
    details = []
    for occurrence_id, expected in heldout_gt.items():
        if occurrence_id not in rows:
            continue
        audited += 1
        row = rows[occurrence_id]
        predicted = row.get("character_id") if row.get("status") == "verified" else None
        if predicted is not None:
            verified += 1
        if predicted is not None and predicted != expected:
            false_merge += 1
            outcome = "false_merge"
        elif predicted is None:
            false_rejection += 1
            outcome = "false_rejection_or_uncertain"
        else:
            outcome = "correct"
        details.append({"occurrence_id": occurrence_id, "expected": expected,
                        "predicted": predicted, "outcome": outcome})
    return {
        "schema_version": "identity_heldout_metrics_v1", "audited": audited,
        "false_merge": false_merge, "false_rejection": false_rejection,
        "verified_coverage": round(verified / audited, 6) if audited else 0.0,
        "uncertain_limit_applied": False, "details": details,
    }


def evaluate_event_directions(event_facts: Iterable[Mapping[str, Any]],
                              heldout_gt: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Report global direction accuracy separately from the selected-video hard gate."""
    events = {str(row["event_id"]): row for row in event_facts}
    details = []
    correct = 0
    for event_id, expected in heldout_gt.items():
        event = events.get(event_id)
        if not event or not event.get("facts"):
            details.append({"event_id": event_id, "correct": False,
                            "reason": "event_or_fact_missing"})
            continue
        fact = event["facts"][0]
        passed = (str(fact.get("actor")) == str(expected.get("actor")) and
                  str(fact.get("patient")) == str(expected.get("patient")))
        correct += int(passed)
        details.append({"event_id": event_id, "correct": passed,
                        "predicted_actor": fact.get("actor"),
                        "predicted_patient": fact.get("patient"),
                        "expected_actor": expected.get("actor"),
                        "expected_patient": expected.get("patient")})
    total = len(heldout_gt)
    return {
        "schema_version": "event_direction_metrics_v1", "sample_count": total,
        "minimum_recommended_sample_count": 12,
        "sample_size_sufficient": total >= 12,
        "actor_patient_accuracy": round(correct / total, 6) if total else 0.0,
        "details": details,
    }


NEUTRAL_EVENT_PROMPT = """Observe only the supplied source clip. Local subject labels
and short neutral visual descriptions are provided below solely to keep subjects distinct.
Do not identify a character, guess a canonical name, assign a story role, or discuss how
the clip should be edited. Report visible action direction and state changes only.
Return strict JSON:
{"participants":[{"occurrence_id":"...","role":"actor|patient|observer"}],
 "facts":[{"actor":"occurrence_id","action":"visible verb",
 "patient":"occurrence_id or null","result":"visible state change",
 "relation_requires_separate_check":false}],
 "claim_bounds":{"can_prove":["..."],"cannot_prove":["..."]},
 "core_interval":[relative_start,relative_end],
 "renderable_interval":[relative_start,relative_end],"passed":true}
Set relation_requires_separate_check=true when a cut or off-screen change prevents direct
causal proof. Do not perform that causal check in this response."""


NEUTRAL_RELATION_PROMPT = """Independently inspect whether a visible action before a cut
is related to the reported off-screen or post-cut state change. Keep the supplied local
occurrence IDs; do not identify characters, use names, or assign an editing function.
Return strict JSON:
{"passed":true,"relations":[{"actor":"occurrence_id",
"patient":"occurrence_id or null","related_outcome":true,
"can_prove":["bounded visible relation"],"cannot_prove":["stronger claim"]}]}
Use related_outcome=false whenever continuity does not support the causal link."""


def _normalize_relative_interval(value: Any, *, start_s: float, end_s: float,
                                 field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise V8Blocked("events", f"invalid_{field}")
    left, right = map(float, value)
    duration = end_s - start_s
    if not 0 <= left < right <= duration + 1e-6:
        raise V8Blocked("events", f"out_of_bounds_{field}")
    return [round(start_s + left, 6), round(start_s + right, 6)]


def verify_shared_event_facts(cfg: AppConfig,
                              event_candidates: Iterable[Mapping[str, Any]],
                              occurrences: Iterable[Mapping[str, Any]], output_dir: Path, *,
                              source_video: Path, runner: Any,
                              native_frame_max: int = 60) -> dict[str, Any]:
    """Verify each unique interval once using a name-free action prompt."""
    output_dir = Path(output_dir)
    occurrence_map = {str(row["occurrence_id"]): dict(row) for row in occurrences}
    grouped: dict[tuple[float, float], list[Mapping[str, Any]]] = {}
    for candidate in event_candidates:
        interval = tuple(map(float, candidate.get("source_interval") or []))
        if len(interval) == 2 and interval[1] > interval[0]:
            grouped.setdefault(interval, []).append(candidate)
    facts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    for event_index, ((start_s, end_s), candidates) in enumerate(sorted(grouped.items())):
        occurrence_ids = sorted({
            str(candidate.get(key))
            for candidate in candidates
            for key in ("actor_occurrence_id", "patient_occurrence_id")
            if candidate.get(key)
        })
        local_context = [
            {"occurrence_id": occurrence_id,
             "local_description": occurrence_map.get(occurrence_id, {}).get("local_description"),
             "visual_state": occurrence_map.get(occurrence_id, {}).get("visual_state")}
            for occurrence_id in occurrence_ids
        ]
        # Descriptions may contain only local visual facts; names and task labels never enter.
        prompt = NEUTRAL_EVENT_PROMPT + "\nLocal subjects:\n" + json.dumps(
            local_context, ensure_ascii=False)
        event_id = f"film1_event_{event_index:04d}"
        event_dir = output_dir / event_id
        try:
            clip = cut_clip(
                ffmpeg, source_video, event_dir / "source_clip",
                start_s=start_s, end_s=end_s, include_audio=False)
            answer = runner.watch(
                clip, prompt, fps=12.0, duration_s=end_s - start_s,
                max_new_tokens=768, use_audio_in_video=False)
            parsed = _parse_object(str(answer.text))
            if _contains_key_recursive(parsed, "character_id"):
                raise V8Blocked("events", "event_fact_contains_character_id")
            participants = parsed.get("participants") or []
            normalized_participants = []
            for participant in participants:
                occurrence_id = str(participant.get("occurrence_id") or "")
                role = str(participant.get("role") or "")
                if occurrence_id not in occurrence_ids or role not in {"actor", "patient", "observer"}:
                    raise V8Blocked("events", "invalid_event_participant")
                normalized_participants.append({"occurrence_id": occurrence_id, "role": role})
            normalized_facts = []
            relation_required = False
            for raw in parsed.get("facts") or []:
                actor = str(raw.get("actor") or "")
                patient = raw.get("patient")
                patient = str(patient) if patient else None
                if actor not in occurrence_ids or (patient and patient not in occurrence_ids):
                    raise V8Blocked("events", "fact_references_unknown_occurrence")
                needs_relation = bool(raw.get("relation_requires_separate_check"))
                relation_required = relation_required or needs_relation
                normalized_facts.append({
                    "actor": actor, "action": str(raw.get("action") or "").strip(),
                    "patient": patient, "result": str(raw.get("result") or "").strip(),
                    "relation_verified": (bool(raw.get(
                        "relation_verified", str(raw.get("result") or "").strip()))
                                          if not needs_relation else False),
                    "relation_check_required": needs_relation,
                })
            if parsed.get("passed") is not True or not normalized_facts:
                raise V8Blocked("events", "neutral_event_not_verified")
            relation_audit = None
            if relation_required:
                relation_answer = runner.watch(
                    clip, NEUTRAL_RELATION_PROMPT + "\nLocal occurrence IDs: "
                    + json.dumps(occurrence_ids), fps=12.0,
                    duration_s=end_s - start_s, max_new_tokens=512,
                    use_audio_in_video=False)
                relation_payload = _parse_object(str(relation_answer.text))
                if _contains_key_recursive(relation_payload, "character_id"):
                    raise V8Blocked("relation", "relation_contains_character_id")
                relation_rows = relation_payload.get("relations") or []
                relation_map = {
                    (str(row.get("actor") or ""),
                     str(row.get("patient") or "") if row.get("patient") else None): row
                    for row in relation_rows
                }
                for fact_row in normalized_facts:
                    if not fact_row["relation_check_required"]:
                        continue
                    relation_row = relation_map.get(
                        (fact_row["actor"], fact_row["patient"]))
                    fact_row["relation_verified"] = bool(
                        relation_payload.get("passed") is True and relation_row
                        and relation_row.get("related_outcome") is True)
                relation_audit = {
                    "prompt_sha256": hashlib.sha256(
                        NEUTRAL_RELATION_PROMPT.encode("utf-8")).hexdigest(),
                    "parsed": relation_payload, "raw_response": str(relation_answer.text),
                }
            coarse_core = _normalize_relative_interval(
                parsed.get("core_interval"), start_s=start_s, end_s=end_s,
                field="core_interval")
            renderable = _normalize_relative_interval(
                parsed.get("renderable_interval"), start_s=start_s, end_s=end_s,
                field="renderable_interval")
            if renderable[0] > coarse_core[0] or renderable[1] < coarse_core[1]:
                raise V8Blocked("events", "renderable_does_not_contain_core")
            native_interval = list(coarse_core)
            if native_interval[1] - native_interval[0] > 0.9:
                center = sum(native_interval) / 2.0
                native_interval = [center - 0.45, center + 0.45]
            native = extract_native_frames(
                ffmpeg, source_video, native_interval, event_dir / "boundary" / "native",
                max_frames=int(native_frame_max),
                ffprobe_bin=str(cfg.perception.get("ffprobe_bin", "ffprobe")))
            frame_ids = [row["frame_id"] for row in native["frames"]]
            boundary_answer = runner.inspect_media(
                [Path(row["labeled_path"]) for row in native["frames"]],
                "Select the first frame where the proven action begins, its clearest peak, "
                "and the last frame where the visible result is established. Images are "
                f"ordered as {frame_ids}. Return strict JSON with start_frame, peak_frame, "
                "end_frame using only listed IDs. Do not calculate seconds.",
                max_new_tokens=128)
            native = apply_native_selection(native, _parse_object(str(boundary_answer.text)))
            _write_json(event_dir / "boundary" / "native" /
                        "native_frames_manifest.json", native)
            core = list(map(float, native["selected"]["core_interval"]))
            if renderable[0] > core[0] or renderable[1] < core[1]:
                raise V8Blocked("events", "native_core_outside_renderable")
            fact = {
                "schema_version": "event_fact_v1", "event_id": event_id,
                "interval": [start_s, end_s], "core_interval": core,
                "renderable_interval": renderable, "source_video": str(source_video),
                "participants": normalized_participants, "facts": normalized_facts,
                "claim_bounds": {
                    "can_prove": [str(value) for value in
                                  (parsed.get("claim_bounds") or {}).get("can_prove") or []],
                    "cannot_prove": [str(value) for value in
                                     (parsed.get("claim_bounds") or {}).get("cannot_prove") or []],
                },
                "native_frame_provenance": native,
                "verification": {
                    "status": "verified", "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")).hexdigest(), "clip": str(clip),
                    "clip_sha256": sha256_file(clip), "raw_response": str(answer.text),
                    "shared_candidate_ids": [str(row.get("event_candidate_id"))
                                             for row in candidates],
                    "relation_check": relation_audit,
                },
            }
            fact["event_fact_sha256"] = event_fact_hash(fact)
            facts.append(fact)
            _write_json(event_dir / "event_fact.json", fact)
        except Exception as exc:  # each interval fails independently
            failure = {
                "event_id": event_id, "interval": [start_s, end_s],
                "status": "blocked", "failure_class": "verification",
                "failure_stage": getattr(exc, "stage", "events"),
                "reason_code": getattr(exc, "reason_code", "event_verification_failed"),
                "detail": str(exc),
            }
            failures.append(failure)
            _write_json(event_dir / "failure.json", failure)
    result = {
        "schema_version": "event_fact_manifest_v1", "event_facts": facts,
        "verified_count": len(facts), "blocked_count": len(failures),
        "shared_observation_count": len(grouped), "failures": failures,
    }
    _write_jsonl(output_dir.parent / "event_facts.jsonl", facts)
    _write_json(output_dir / "event_manifest.json", result)
    return result


def derive_character_evidence_views(event_facts: Iterable[Mapping[str, Any]],
                                    identity_bindings: Iterable[Mapping[str, Any]],
                                    output_dir: Path) -> dict[str, Any]:
    """Join immutable event facts with verified, revocable identity bindings."""
    output_dir = Path(output_dir)
    binding_map = {
        str(row["occurrence_id"]): dict(row)
        for row in identity_bindings if row.get("status") == "verified"
    }
    views: list[dict[str, Any]] = []
    for event in event_facts:
        event_fact_hash(event)  # refuse contaminated historical input as well
        for fact_index, fact in enumerate(event.get("facts") or []):
            relation = bool(fact.get("relation_verified"))
            roles = ((str(fact.get("actor") or ""), "actor"),
                     (str(fact.get("patient") or ""), "patient"))
            for occurrence_id, perspective_role in roles:
                if not occurrence_id or occurrence_id not in binding_map:
                    continue
                binding = binding_map[occurrence_id]
                uses: list[str] = []
                if perspective_role == "actor":
                    uses.append("agency")
                    if relation and str(fact.get("result") or "").strip():
                        uses.extend(["caused_visible_change", "outcome"])
                else:
                    uses.extend(["received_impact", "adversity"])
                can_prove = [str(value) for value in
                             (event.get("claim_bounds") or {}).get("can_prove") or []]
                view = {
                    "schema_version": "character_evidence_view_v1",
                    "view_id": f"{event['event_id']}:{fact_index}:{perspective_role}",
                    "character_id": binding["character_id"],
                    "form_id": binding.get("form_id"),
                    "occurrence_id": occurrence_id,
                    "binding_sha256": binding.get("binding_sha256"),
                    "event_id": event["event_id"],
                    "event_fact_sha256": event.get("event_fact_sha256") or event_fact_hash(event),
                    "perspective_role": perspective_role,
                    "evidence_uses": uses,
                    "action": fact.get("action"), "result": fact.get("result"),
                    "relation_verified": relation,
                    "core_interval": event.get("core_interval"),
                    "renderable_interval": event.get("renderable_interval"),
                    "observation_interval": event.get("interval"),
                    "source_video": event.get("source_video"),
                    "claim_bounds": event.get("claim_bounds") or {},
                    "core_expression_fragment": can_prove[0] if can_prove else str(
                        fact.get("result") or fact.get("action") or "").strip(),
                }
                views.append(view)
    by_character = {character_id: [row for row in views
                                   if row["character_id"] == character_id]
                    for character_id in CHARACTER_IDS}
    for character_id, rows in by_character.items():
        destination = output_dir / character_id.replace(":", "_") / "views.jsonl"
        _write_jsonl(destination, rows)
    result = {
        "schema_version": "character_evidence_views_v1", "views": views,
        "counts": {key: len(value) for key, value in by_character.items()},
        "identity_source": "verified_bindings_only",
    }
    _write_json(output_dir / "views_manifest.json", result)
    return result


def _unique_event_views(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (
            float((item.get("renderable_interval") or [float("inf")])[0]),
            str(item.get("event_id")))):
        selected.setdefault(str(row["event_id"]), dict(row))
    return list(selected.values())


def _task(character_id: str, family: str, views: list[dict[str, Any]],
          required_uses: Iterable[str]) -> dict[str, Any]:
    ordered = _unique_event_views(views)
    fragments = [str(row.get("core_expression_fragment") or "").strip()
                 for row in ordered if str(row.get("core_expression_fragment") or "").strip()]
    required = sorted(set(required_uses))
    all_uses = {value for row in ordered for value in row.get("evidence_uses") or []}
    missing = [value for value in required if value not in all_uses]
    task_id = f"{character_id.replace(':', '_')}__{family}"
    return {
        "schema_version": "creative_task_v1", "task_id": task_id,
        "character_id": character_id, "family": family,
        "status": "ready" if not missing else "blocked", "missing_uses": missing,
        "event_ids": [row["event_id"] for row in ordered],
        "view_ids": [row["view_id"] for row in ordered],
        "core_expression": "；".join(dict.fromkeys(fragments)) or "已核验的可见变化",
        "core_expression_source": "event_claim_bounds",
        "model_total_score_used": False,
    }


def _character_task_candidates(character_id: str,
                               rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = [
        ("adversity_agency_outcome", {"adversity", "agency", "outcome"}),
        ("intervention_and_result", {"agency", "caused_visible_change"}),
        ("capability_montage", {"agency"}),
        ("threat_escalation_and_impact", {"agency", "outcome"}),
    ]
    candidates: list[dict[str, Any]] = []
    for family, required in families:
        eligible = [row for row in rows
                    if set(row.get("evidence_uses") or []) & required]
        candidate = _task(character_id, family, eligible, required)
        if family == "capability_montage" and len(set(candidate["event_ids"])) < 2:
            candidate["status"] = "blocked"
            candidate["missing_uses"] = ["second_distinct_agency_event"]
        candidates.append(candidate)
    if not any(row["status"] == "ready" for row in candidates):
        custom_views = _unique_event_views(rows)
        if len(custom_views) >= 2:
            candidates.append(_task(
                character_id, "custom_evidence_arc", custom_views[:4], ()))
    return candidates


def build_character_creation_queue(
        evidence_views: Iterable[Mapping[str, Any]], output_path: Path) -> dict[str, Any]:
    """Choose at most one task per fixed character with deterministic global search."""
    rows = [dict(row) for row in evidence_views]
    candidates_by_character = {
        character_id: _character_task_candidates(
            character_id, [row for row in rows if row.get("character_id") == character_id])
        for character_id in CHARACTER_IDS
    }
    ready_options = [
        [None] + sorted((row for row in candidates_by_character[character_id]
                         if row["status"] == "ready"), key=lambda row: row["task_id"])
        for character_id in CHARACTER_IDS
    ]
    best_combo: tuple[dict[str, Any] | None, ...] = tuple(None for _ in CHARACTER_IDS)
    best_score = (-1, -1, -1)
    for combo in itertools.product(*ready_options):
        chosen = [row for row in combo if row is not None]
        ordered_sequences = [tuple(row["event_ids"]) for row in chosen]
        if len(ordered_sequences) != len(set(ordered_sequences)):
            continue  # changing only a title is not an independent character video
        all_events = [event_id for row in chosen for event_id in row["event_ids"]]
        duplicate_count = len(all_events) - len(set(all_events))
        score = (len(chosen), len(set(all_events)), -duplicate_count)
        if score > best_score:
            best_combo, best_score = combo, score
    selected = [row for row in best_combo if row is not None]
    selected_ids = {row["task_id"] for row in selected}
    all_candidates = [row for character_id in CHARACTER_IDS
                      for row in candidates_by_character[character_id]]
    for row in all_candidates:
        row["selected"] = row["task_id"] in selected_ids
    result = {
        "schema_version": "character_creation_queue_v1",
        "characters": list(CHARACTER_IDS), "candidates": all_candidates,
        "selected_tasks": selected,
        "ready_character_count": len(selected), "maximum_videos": 3,
        "selection_objectives": [
            "deliverable_character_count", "independent_event_coverage",
            "fewer_gaps", "stable_task_id"],
    }
    _write_json(Path(output_path), result)
    return result


def build_reference_driven_character_plan(reference_task: Mapping[str, Any],
                                          task: Mapping[str, Any],
                                          evidence_views: Iterable[Mapping[str, Any]]) \
        -> dict[str, Any]:
    """Map verified event views to a render plan without changing their boundaries."""
    view_map = {str(row["view_id"]): dict(row) for row in evidence_views}
    selected = [view_map[view_id] for view_id in task.get("view_ids") or []
                if view_id in view_map]
    selected = _unique_event_views(selected)
    segments = []
    for index, view in enumerate(selected):
        renderable = list(map(float, view.get("renderable_interval") or []))
        core = list(map(float, view.get("core_interval") or []))
        if len(renderable) != 2 or len(core) != 2 or not (
                renderable[0] <= core[0] < core[1] <= renderable[1]):
            raise V8Blocked("planning", "invalid_verified_event_intervals")
        segments.append({
            "id": f"segment_{index:02d}", "evidence_id": view["view_id"],
            "event_id": view["event_id"], "assigned_goal": task["family"],
            "source_video": view["source_video"],
            "observation_interval": view.get("observation_interval"),
            "core_interval": core, "render_interval": renderable,
            "final_source_interval": renderable,
            "duration_s": round(renderable[1] - renderable[0], 6),
            "identity_binding_sha256": view.get("binding_sha256"),
            "event_fact_sha256": view.get("event_fact_sha256"),
            "selection_reason": "verified_character_view_and_event_claim",
        })
    if not segments:
        raise V8Blocked("planning", "selected_task_has_no_verified_segments")
    measured = reference_task.get("measured_style") or {}
    reference_duration = float(measured.get("reference_duration_s", 21.933))
    p90 = float(measured.get("shot_duration_p90_s") or 0.0)
    duration = round(sum(row["duration_s"] for row in segments), 6)
    reference_units = int(measured.get("meaningful_unit_count") or
                          len(measured.get("shot_intervals") or []))
    soft_min, soft_max = reference_duration * 0.85, reference_duration * 1.15
    plan = {
        "schema_version": "character_edit_plan_v1", "passed": True,
        "task_id": task["task_id"], "target_id": task["character_id"],
        "character_id": task["character_id"], "family": task["family"],
        "core_expression": task["core_expression"], "segments": segments,
        "duration_s": duration, "filler_added": False,
        "minimum_cut_count_required": False,
        "style": {
            "preferred_duration_s": reference_duration,
            "soft_range_s": [round(soft_min, 6), round(soft_max, 6)],
            "segment_preferred_max_s": p90,
            "reference_meaningful_unit_count": reference_units,
            "meaningful_unit_count": len(segments),
            "reference_information_units_per_s": round(
                reference_units / reference_duration, 6) if reference_duration else 0.0,
            "output_information_units_per_s": round(
                len(segments) / duration, 6) if duration else 0.0,
            "section_duration_distribution": list(
                measured.get("section_duration_distribution") or []),
            "shot_duration_distribution": list(
                measured.get("shot_duration_distribution") or []),
        },
        "requires_human_style_review": duration > soft_max,
        "short_complete_content_allowed": duration < soft_min,
    }
    plan["plan_sha256"] = _stable_sha(plan)
    return plan


def run_character_batch(cfg: AppConfig, creation_queue: Mapping[str, Any],
                        reference_task: Mapping[str, Any],
                        evidence_views: Iterable[Mapping[str, Any]], output_dir: Path, *,
                        source_video: Path, bgm_path: Path, runner: Any,
                        force: bool = False) -> dict[str, Any]:
    """Render each ready role independently; one failure never aborts the batch."""
    output_dir = Path(output_dir)
    all_views = [dict(row) for row in evidence_views]
    results: list[dict[str, Any]] = []
    for task in creation_queue.get("selected_tasks") or []:
        character_id, task_id = str(task["character_id"]), str(task["task_id"])
        task_dir = output_dir / "characters" / character_id.replace(":", "_") / task_id
        try:
            plan = build_reference_driven_character_plan(reference_task, task, all_views)
            _write_json(task_dir / "edit_plan.json", plan)
            rendered = finalize_target_microcut(
                cfg, plan, task_dir, source_video=source_video, bgm_path=bgm_path,
                runner=runner, human_acceptance=None, force=force)
            results.append({
                "character_id": character_id, "task_id": task_id,
                "status": "preview_ready" if rendered["acceptance"].get(
                    "automated_passed") else "blocked",
                "acceptance": rendered["acceptance"], "task_dir": str(task_dir),
            })
        except Exception as exc:
            (task_dir / "rendered.mp4").unlink(missing_ok=True)
            failure = {
                "character_id": character_id, "task_id": task_id, "status": "blocked",
                "failure_class": "verification",
                "failure_stage": getattr(exc, "stage", "render"),
                "reason_code": getattr(exc, "reason_code", "character_render_failed"),
                "detail": str(exc), "task_dir": str(task_dir),
            }
            _write_json(task_dir / "failure.json", failure)
            results.append(failure)
    result = {
        "schema_version": "character_batch_acceptance_v1", "passed": False,
        "delivery": "awaiting_human_acceptance",
        "selected_task_count": len(creation_queue.get("selected_tasks") or []),
        "preview_ready_count": sum(row["status"] == "preview_ready" for row in results),
        "final_passed_count": 0, "characters": results,
    }
    _write_json(output_dir / "batch_acceptance.json", result)
    return result


def accept_character_batch(output_dir: Path,
                           human_acceptance: Path | Mapping[str, Any]) -> dict[str, Any]:
    """Promote only independently human-approved, automatically passed character cuts."""
    output_dir = Path(output_dir)
    human = (_read_json(human_acceptance) if isinstance(human_acceptance, Path)
             else dict(human_acceptance))
    prior = _read_json(output_dir / "batch_acceptance.json")
    approvals = human.get("characters") or {}
    results = []
    for row in prior.get("characters") or []:
        task_dir = Path(str(row["task_dir"]))
        approval = approvals.get(row["character_id"]) or approvals.get(row["task_id"]) or {}
        try:
            if approval.get("selected_identity_false_merges") != 0:
                raise V8Blocked("human", "selected_identity_false_merge_not_zero")
            if approval.get("selected_event_direction_errors") != 0:
                raise V8Blocked("human", "selected_event_direction_error_not_zero")
            if approval.get("core_expression_understood") is not True:
                raise V8Blocked("human", "core_expression_not_understood")
            destination = accept_v7_output(task_dir, approval)
            results.append({**row, "status": "passed", "rendered": str(destination),
                            "human_acceptance": approval})
        except Exception as exc:
            (task_dir / "rendered.mp4").unlink(missing_ok=True)
            results.append({**row, "status": "blocked", "human_acceptance": approval,
                            "acceptance_error": str(exc)})
    final_count = sum(row["status"] == "passed" for row in results)
    result = {
        "schema_version": "character_batch_acceptance_v1",
        "passed": final_count > 0, "delivery": "passed" if final_count else "blocked",
        "selected_task_count": prior.get("selected_task_count", len(results)),
        "preview_ready_count": prior.get("preview_ready_count", 0),
        "final_passed_count": final_count, "characters": results,
        "zero_to_three_outputs_allowed": True,
    }
    _write_json(output_dir / "batch_acceptance.json", result)
    return result
