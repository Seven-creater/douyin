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
import re
import shutil
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
    aliases: list[tuple[str, str]] = []
    for character in characters:
        character_id = str(character["character_id"])
        names = dict.fromkeys(
            [character.get("display_name"), *(character.get("aliases") or [])])
        aliases.extend((str(name), character_id) for name in names if str(name or "").strip())
    rows = []
    segments = (transcript.get("segments") or transcript.get("utterances") or []) \
        if isinstance(transcript, Mapping) else transcript
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        for name, character_id in aliases:
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
Inspect the entire block before selecting regions. Report only visible state changes,
interactions, entrances/exits, or other concrete activity that genuinely merits a closer
original-video rewatch; never report routine static presence merely to fill the list.
Return strict JSON:
{"regions":[{"interval":[0.0,2.0],"activity":"visible coarse activity",
"worth_rewatch":true,"occurrences":[{"local_id":"A","local_description":"visible
appearance only","visual_state":"visible state/activity","roi":null}],
"event_candidates":[{"actor_local_id":"A","action":"visible action",
"patient_local_id":null,"visible_result":"visible change or empty"}]}]}
Times are seconds relative to this block. Return at most three regions, each 2-6 seconds.
Every returned region must have worth_rewatch=true. If nothing merits rewatch return
{"regions":[]}. Not observed never means absent."""


_FORBIDDEN_NEUTRAL_KEYS = {
    "character_id", "canonical_id", "target_id", "assigned_goal", "editing_role",
    "adversity", "agency", "outcome",
}


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
        interval = region.get("interval") or []
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        rel_start, rel_end = map(float, interval)
        if not 0 <= rel_start < rel_end <= end_s - start_s + 1e-6:
            continue
        if not 1.5 <= rel_end - rel_start <= 6.5:
            continue
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
                "local_description": str(raw.get("local_description") or "").strip(),
                "visual_state": str(raw.get("visual_state") or "").strip(),
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
            patient = local_to_global.get(str(raw.get("patient_local_id") or ""))
            if not actor:
                continue
            event_candidates.append({
                "event_candidate_id": f"candidate_{block_id}_{region_index:02d}_{event_index:02d}",
                "source_interval": absolute,
                "actor_occurrence_id": actor,
                "action": str(raw.get("action") or "").strip(),
                "patient_occurrence_id": patient,
                "visible_result": str(raw.get("visible_result") or "").strip(),
                "activity": str(region.get("activity") or "").strip(),
                "worth_rewatch": bool(region.get("worth_rewatch")),
                "status": "candidate",
            })
    return occurrences, event_candidates


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
        result_path = block_dir / "result.json"
        cache_key = _stable_sha({**contract, "source_interval": [start, end]})
        if reuse_completed and result_path.is_file():
            prior = _read_json(result_path)
            if prior.get("cache_key") == cache_key and prior.get("status") == "covered":
                return prior
        clip = cut_clip(
            ffmpeg, Path(source_video), block_dir / "transport",
            start_s=start, end_s=end)
        request_clip, media_transport_audit = stage_for_service(clip, block_id)
        try:
            answer = client.watch(
                request_clip, NEUTRAL_COVERAGE_PROMPT, duration_s=end - start,
                image_paths=None, max_tokens=768)
        finally:
            if request_clip != clip:
                request_clip.unlink(missing_ok=True)
        raw = _parse_object(answer.text)
        forbidden_terms = {
            str(value).strip().lower()
            for character in spec.get("characters") or []
            for value in ([character.get("display_name")]
                          + list(character.get("aliases") or []))
            if str(value or "").strip()
        }
        serialized = json.dumps(raw, ensure_ascii=False).lower()
        if any(term in serialized for term in forbidden_terms):
            raise V8Blocked("coverage", "neutral_observation_named_character")
        occurrences, events = _normalize_coverage_response(
            raw, block_id=block_id, start_s=start, end_s=end)
        for occurrence in occurrences:
            occurrence["source_video"] = str(source_video)
        for event in events:
            event["source_video"] = str(source_video)
        sampling = (answer.raw.get("sampling_audit") or
                    answer.raw.get("video_metadata") or {})
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
            "sampling_audit": sampling,
            "sampling_audit_status": "available" if sampling else "unavailable_from_server",
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
    _write_jsonl(output_dir.parent / "occurrence_bank.jsonl", occurrences)
    _write_jsonl(output_dir.parent / "event_candidates.jsonl", events)
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
            clip = cut_clip(ffmpeg, source_video, event_dir / "source_clip",
                            start_s=start_s, end_s=end_s)
            answer = runner.watch(
                clip, prompt, fps=12.0, duration_s=end_s - start_s, max_new_tokens=768)
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
                    duration_s=end_s - start_s, max_new_tokens=512)
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
                max_frames=int(native_frame_max))
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
    plan = {
        "schema_version": "character_edit_plan_v1", "passed": True,
        "task_id": task["task_id"], "target_id": task["character_id"],
        "character_id": task["character_id"], "family": task["family"],
        "core_expression": task["core_expression"], "segments": segments,
        "duration_s": duration, "filler_added": False,
        "minimum_cut_count_required": False,
        "style": {
            "preferred_duration_s": reference_duration,
            "soft_range_s": [round(reference_duration * 0.7, 6),
                             round(reference_duration * 1.2, 6)],
            "segment_preferred_max_s": p90,
        },
        "requires_human_style_review": duration > reference_duration * 1.2,
        "short_complete_content_allowed": duration < reference_duration * 0.7,
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
