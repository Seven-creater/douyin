"""V8.2 target-centred, recall-first film perception.

The reusable architecture is reference-conditioned target-centric perception.
This release implements only its first ``subject`` harvest-policy instance,
whose fixed experimental target is Xiaohei. It intentionally stops at an
auditable target event timeline and
does not import or call the editorial planner, renderer, audio mixer, YOLO,
Ultralytics, a tracker, or a dedicated ReID model.

The durable layers are kept separate:

candidate scene -> observation task -> neutral occurrence/event candidate
                -> revocable character/form binding -> trusted timeline view

Human audit data is consumed only after automatic artifacts have been frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.agentic_video.character_batch import (
    IDENTITY_COMPARE_PROMPT,
    V8Blocked,
    _build_coverage_transport,
    _occurrence_profile_request,
    _parse_object,
    _single_json_object,
    _stable_sha,
    _write_json,
    _write_jsonl,
    build_character_profiles,
    build_seed_review_sheet,
    compute_temporal_coverage,
    coverage_blocks,
    validate_neutral_observation,
)
from src.agentic_video.recipe_v2 import sha256_file
from src.agentic_video.target_v7 import export_frame, export_roi
from src.config import AppConfig
from src.perception import common


SPEC_VERSION = "target_recall_v82"
FIRST_ROUND_TARGET = "char:xiaohei"
FORM_STATUSES = {"identified", "uncertain", "transforming", "unknown_form_candidate"}
CHARACTER_STATUSES = {"supported", "uncertain", "conflict", "rejected"}
QUEUE_STATES = {
    "pending", "running", "completed_observed", "completed_mention_only",
    "completed_uncertain", "needs_context", "failed_infrastructure",
}

V82_NEUTRAL_PROMPT = """Observe only this source-movie clip. Do not identify or name
characters, use an identity album, infer plot, or assign editing/story functions. First
list local visual subjects with temporary labels. A cut may create a new occurrence even
if it might be the same character; identity linking happens later. Then report concrete
visible facts using only those labels.

Return exactly one JSON object:
{"status":"observed|observed_empty|unreliable",
 "occurrences":[{"local_id":"A","visible_interval":[0,1],
 "local_description":"concrete appearance","visual_state":"concrete state",
 "roi":null}],
 "event_candidates":[{"interval":[0,1],"initiator_local_id":"A or null",
 "affected_local_id":"B or null","action_or_state":"concrete visible fact",
 "visible_result":"visible change or empty","relation_status":"supported|not_applicable|uncertain"}],
 "left_context_complete":true,"right_context_complete":true,
 "boundary_reason":"brief visible reason or empty"}

At least one local occurrence must be referenced by every event. Initiator may be null for
a static state or an outcome whose cause is not visible. affected may be null when not
applicable. Do not upgrade temporal order into causation. Use unreliable for dissolves,
severe blur, black/corrupt frames, or insufficient evidence. Use observed_empty only when
the image is clear and nothing concrete is recordable. All times are relative to this
exact input and must remain in bounds. Return no Markdown and no trailing text."""
TERMINAL_QUEUE_STATES = {
    "completed_observed", "completed_mention_only", "completed_uncertain",
    "failed_infrastructure",
}
REQUIRED_FLASHVID_AUDIT_FIELDS = {
    "actual_sampled_frame_count", "actual_frame_indices",
    "actual_frame_timestamps_relative_s", "actual_frame_timestamps_absolute_s",
    "input_image_sizes", "processed_image_grid_thw", "processed_video_grid_thw",
    "video_tokens_before", "video_tokens_after", "actual_retention_ratio",
    "service_startup_args", "service_git_head", "service_git_dirty",
    "service_core_hashes", "video_transport_sha256", "image_sha256",
}


class V82Blocked(V8Blocked):
    """Stable phase/reason failure used by the V8.2 CLI."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "item"


def _interval(value: Any, *, duration_s: float | None = None,
              field: str = "interval") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise V82Blocked("scope", f"invalid_{field}")
    start, end = map(float, value)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise V82Blocked("scope", f"invalid_{field}")
    if duration_s is not None and (start < 0 or end > duration_s + 1e-6):
        raise V82Blocked("scope", f"out_of_bounds_{field}")
    return [round(start, 6), round(end, 6)]


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(float(left[1]), float(right[1])) -
               max(float(left[0]), float(right[0])))


def interval_coverage_ratio(interval: Sequence[float],
                            covering: Iterable[Sequence[float]]) -> float:
    """Return union coverage of one interval, never double-counting overlaps."""
    target = _interval(interval, field="coverage_target")
    audit = compute_temporal_coverage(covering, eligible_interval=target)
    return float(audit["coverage_ratio"])


def _concrete(value: Any, field: str, *, allow_empty: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    normalized = text.lower().strip(" .,:;!?，。！？：；\"'")
    if allow_empty and not normalized:
        return ""
    placeholders = {
        "visible state", "visible verb", "visible result", "visible change or empty",
        "concrete visible fact", "concrete appearance", "concrete state",
        "人物发生动作", "某个角色在活动", "发生了某种动作", "something happens",
    }
    if not normalized or normalized in placeholders or normalized.startswith("visible "):
        raise V82Blocked("observe", f"non_concrete_{field}")
    if not re.search(r"[A-Za-z0-9\u3400-\u9fff]", normalized):
        raise V82Blocked("observe", f"non_concrete_{field}")
    return text


def validate_target_neutral_observation(payload: Mapping[str, Any], *,
                                        duration_s: float) -> str:
    """V8.2 semantic contract with optional initiator/affected roles."""
    base = dict(payload)
    facts = list(base.get("event_candidates") or [])
    base["event_candidates"] = []
    status = validate_neutral_observation(base, duration_s=duration_s)
    if status != "observed" and facts:
        raise V82Blocked("observe", "non_observed_status_contains_facts")
    local_ids = {str(row.get("local_id") or "")
                 for row in base.get("occurrences") or []}
    semantic_values = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, Mapping):
            raise V82Blocked("observe", "invalid_event_candidate")
        _interval(fact.get("interval"), duration_s=duration_s,
                  field=f"event_{index}_interval")
        initiator = fact.get("initiator_local_id")
        affected = fact.get("affected_local_id")
        if initiator in {"", "null", "not_applicable"}:
            initiator = None
        if affected in {"", "null", "not_applicable"}:
            affected = None
        if initiator is not None and str(initiator) not in local_ids:
            raise V82Blocked("observe", "event_initiator_missing_occurrence")
        if affected is not None and str(affected) not in local_ids:
            raise V82Blocked("observe", "event_affected_missing_occurrence")
        if initiator is None and affected is None:
            raise V82Blocked("observe", "event_has_no_local_participant")
        semantic_values.append(_concrete(fact.get("action_or_state"), "action_or_state"))
        result = _concrete(fact.get("visible_result"), "visible_result", allow_empty=True)
        if result:
            semantic_values.append(result)
        relation = str(fact.get("relation_status") or "")
        if relation not in {"supported", "not_applicable", "uncertain"}:
            raise V82Blocked("observe", "invalid_relation_status")
    if any(semantic_values.count(value) > 2 for value in set(semantic_values)):
        raise V82Blocked("observe", "repeated_semantic_fields")
    return status


def read_v82_spec(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    spec = _read_json(path)
    if spec.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"V8.2 spec_version must be {SPEC_VERSION}")
    objective = spec.get("perception_objective") or {}
    if objective.get("center_type") != "subject":
        raise ValueError("V8.2 implements only the subject harvest policy")
    if objective.get("harvest_policy") != "subject":
        raise ValueError("V8.2 harvest_policy must be subject")
    if objective.get("target_id") != FIRST_ROUND_TARGET:
        raise ValueError(f"V8.2 first-round target must be {FIRST_ROUND_TARGET}")
    if objective.get("continuity_key") != "character_identity":
        raise ValueError("V8.2 subject continuity key must be character_identity")
    if not str(spec.get("movie_knowledge_prior") or "").strip():
        raise ValueError("V8.2 requires an external movie knowledge prior")
    forms = spec["target_profile"].get("forms") or []
    expected = {"form_black_cat", "form_black_hair_child", "form_white_hair_child"}
    if {str(row.get("form_id")) for row in forms} != expected:
        raise ValueError(f"V8.2 requires the three Xiaohei forms: {sorted(expected)}")
    browse = spec.get("browse") or {}
    if float(browse.get("block_s", 0)) != 45.0:
        raise ValueError("V8.2 browse block_s must be 45 seconds")
    if float(browse.get("fps", 0)) != 4.0:
        raise ValueError("V8.2 work browse must use 4fps")
    if float(browse.get("retention_ratio", 0)) != .25:
        raise ValueError("V8.2 work browse must use r0.25")
    observe = spec.get("observe") or {}
    if int(observe.get("max_frames", 0)) != 80:
        raise ValueError("V8.2 observation task budget must be 80 frames")
    if float(observe.get("fps", 0)) != 4.0:
        raise ValueError("V8.2 observation tasks must use 4fps")
    serialized = json.dumps(spec, ensure_ascii=False).lower()
    for banned in ("yolo", "ultralytics", "dedicated_reid", "object_tracker"):
        if banned in serialized:
            raise ValueError(f"V8.2 spec contains forbidden detector/tracker token: {banned}")
    return spec, hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_movie_knowledge(
        prior_path: Path, output_dir: Path, *, target_id: str) -> dict[str, Any]:
    """Freeze external movie knowledge as search prior, never source truth."""
    prior = _read_json(Path(prior_path))
    if prior.get("schema_version") != "movie_knowledge_prior_v1":
        raise V82Blocked("knowledge", "unsupported_movie_knowledge_schema")
    sources = list(prior.get("sources") or [])
    source_ids = [str(row.get("source_id") or "") for row in sources]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise V82Blocked("knowledge", "knowledge_sources_missing_or_duplicated")
    if any(not str(row.get("url") or "").startswith("https://") for row in sources):
        raise V82Blocked("knowledge", "knowledge_source_url_invalid")
    source_set = set(source_ids)
    character_priors = list(prior.get("character_priors") or [])
    target = next((dict(row) for row in character_priors
                   if row.get("character_id") == target_id), None)
    if target is None:
        raise V82Blocked("knowledge", "target_character_prior_missing", target_id)
    claims = list(prior.get("claims") or [])
    external_records = [*claims, *character_priors,
                        *(prior.get("relationship_priors") or []),
                        *(prior.get("form_transition_hypotheses") or [])]
    for claim in external_records:
        if claim.get("status") != "prior_to_verify":
            raise V82Blocked("knowledge", "external_claim_promoted_to_source_truth")
        refs = set(map(str, claim.get("source_ids") or []))
        if not refs or not refs.issubset(source_set):
            raise V82Blocked("knowledge", "external_claim_source_missing")
        forbidden = {"source_interval", "occurrence_id", "binding_status"}
        if forbidden.intersection(claim):
            raise V82Blocked("knowledge", "external_claim_contains_source_local_fact")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    movie_identity = dict(prior.get("movie_identity") or {})
    roster = list(prior.get("character_roster") or [])
    relationships = list(prior.get("relationship_priors") or [])
    transitions = list(prior.get("form_transition_hypotheses") or [])
    _write_json(output_dir / "movie_identity.json", movie_identity)
    _write_json(output_dir / "character_roster.json", {"characters": roster})
    for character in character_priors:
        character_id = str(character.get("character_id") or "")
        if not character_id:
            raise V82Blocked("knowledge", "character_prior_id_missing")
        _write_json(output_dir / "character_priors" /
                    f"{_safe_id(character_id)}.json", character)
    _write_json(output_dir / "relationship_priors.json", {"relationships": relationships})
    _write_json(output_dir / "form_transition_hypotheses.json", {
        "hypotheses": transitions})
    _write_jsonl(output_dir / "sources.jsonl", sources)
    _write_jsonl(output_dir / "claims.jsonl", claims)
    manifest = {
        "schema_version": "movie_knowledge_manifest_v82",
        "target_id": target_id,
        "movie_identity": movie_identity,
        "target_character_prior": target,
        "claim_count": len(claims),
        "character_prior_count": len(character_priors),
        "source_count": len(sources),
        "relationship_prior_count": len(relationships),
        "form_transition_hypothesis_count": len(transitions),
        "knowledge_is_prior": True,
        "source_video_is_evidence": True,
        "external_claims_can_bind_occurrences": False,
        "prior_sha256": sha256_file(Path(prior_path)),
    }
    manifest["knowledge_manifest_sha256"] = _stable_sha(manifest)
    _write_json(output_dir / "knowledge_manifest.json", manifest)
    return manifest


def build_target_search_card(
        knowledge_manifest: Mapping[str, Any],
        profiles_manifest: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    """Build the small browse card: search space plus source-confirmed examples."""
    if knowledge_manifest.get("knowledge_is_prior") is not True:
        raise V82Blocked("knowledge", "knowledge_prior_contract_missing")
    if profiles_manifest.get("status") != "ready":
        raise V82Blocked("prepare", "human_confirmed_profile_required_for_search_card")
    target_id = str(knowledge_manifest["target_id"])
    trusted_forms = sorted({
        str(row.get("form_id", "")).split("/")[-1]
        for row in profiles_manifest.get("profiles") or []
        if row.get("status") == "usable"})
    prior_forms = list((knowledge_manifest.get("target_character_prior") or {}).get(
        "known_form_hypotheses") or [])
    card = {
        "schema_version": "target_search_card_v82",
        "target_alias": "S0",
        "target_id_for_audit_only": target_id,
        "center_type": "subject",
        "known_form_hypotheses": prior_forms,
        "trusted_local_form_ids": trusted_forms,
        "identity_album_status": "human_confirmed_model_cross_checked",
        "search_outputs": [
            "known_form_candidate", "uncertain_form_candidate",
            "unknown_form_candidate", "direct_interaction_candidate"],
        "prior_expands_search_space_only": True,
        "external_prior_can_confirm_identity": False,
        "relationship_priors_injected_into_browse_prompt": False,
        "hard_negative_required": True,
    }
    card["search_card_sha256"] = _stable_sha(card)
    _write_json(Path(output_path), card)
    return card


def build_browse_prompt(search_card: Mapping[str, Any]) -> str:
    """Render only target-search prior; relationship and plot claims stay out."""
    if search_card.get("prior_expands_search_space_only") is not True:
        raise V82Blocked("knowledge", "search_card_prior_contract_missing")
    forms = []
    for row in search_card.get("known_form_hypotheses") or []:
        description = str(row.get("visual_description") or "").strip()
        if description:
            forms.append(description)
    if not forms:
        raise V82Blocked("knowledge", "search_card_has_no_form_hypotheses")
    form_lines = "\n".join(f"- {value}" for value in forms)
    prior_section = (
        "External knowledge proposes the following visual search space; it is not "
        f"identity proof:\n{form_lines}\nReport known-form candidates, uncertain or "
        "possible new forms, and direct S0 interactions. Relationship and plot prior "
        "are deliberately withheld.\n")
    return BROWSE_PROMPT.replace(
        "The first video is a 45-second source-movie transport.",
        "The first video is a source-movie transport.").replace(
            "Return one JSON object:", prior_section + "Return one JSON object:")


def prepare_multiform_profile(cfg: AppConfig, spec: Mapping[str, Any],
                              output_dir: Path, *, source_video: Path,
                              seed_manifest: Mapping[str, Any] | None = None,
                              runner: Any | None = None) -> dict[str, Any]:
    """Export proposals, or validate a human-confirmed three-form profile.

    Proposal frames are never promoted to identity truth.  The 5391.5 dissolve
    stays a forbidden regression example even if it resembles a target form.
    """
    output_dir = Path(output_dir)
    target = dict(spec["target_profile"])
    target_id = str(spec["perception_objective"]["target_id"])
    compatibility_spec = {
        "source": spec["source"],
        "characters": [{
            "character_id": target_id,
            "display_name": target.get("display_name", "小黑"),
            "aliases": target.get("aliases") or ["小黑"],
        }],
        "seed_proposals": {target_id: {
            "forms": [
                {"form_id": f"{target_id}/{row['form_id']}",
                 "source_times_s": row.get("proposal_times_s") or []}
                for row in target.get("forms") or []
            ],
            "invalid_source_times_s": target.get("invalid_source_times_s") or [5391.5],
        }},
    }
    review = build_seed_review_sheet(
        cfg, compatibility_spec, output_dir / "review", source_video=source_video)
    if seed_manifest is None:
        result = {
            "schema_version": "multiform_profile_prepare_v82",
            "status": "awaiting_human_confirmation",
            "target_id": target_id,
            "required_forms": [row["form_id"] for row in target.get("forms") or []],
            "seed_review": review,
            "identity_truth_claimed": False,
        }
        _write_json(output_dir / "prepare_manifest.json", result)
        return result
    if runner is None:
        raise V82Blocked("prepare", "omni_runner_required_for_profile_validation")
    profiles = build_character_profiles(
        cfg, compatibility_spec, seed_manifest, output_dir,
        source_video=source_video, runner=runner)
    usable = profiles.get("profiles") or []
    if len(usable) != 3 or any(row.get("status") != "usable" for row in usable):
        raise V82Blocked("prepare", "three_human_confirmed_forms_not_usable")
    result = {
        **profiles,
        "schema_version": "multiform_profile_manifest_v82",
        "status": "ready",
        "target_id": target_id,
        "review_status": "human_confirmed",
        "invalid_regression_times_s": target.get("invalid_source_times_s") or [5391.5],
    }
    mosaics = build_profile_mosaics(
        profiles, output_dir / "mosaics",
        ffmpeg_bin=str(cfg.perception.get("ffmpeg_bin", "ffmpeg")))
    result["browse_album"] = mosaics
    _write_json(output_dir / "profiles_manifest.json", result)
    return result


def _make_mosaic(image_paths: Sequence[Path], destination: Path, *,
                 ffmpeg_bin: str = "ffmpeg") -> Path:
    """Make a deterministic 2x2 labelled album image without a detector."""
    paths = [Path(path) for path in image_paths]
    if not paths or len(paths) > 4 or any(not path.is_file() for path in paths):
        raise V82Blocked("prepare", "mosaic_source_images_missing")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    chains = []
    labels = "ABCD"
    for index, path in enumerate(paths):
        inputs.extend(["-i", str(path)])
        chains.append(
            f"[{index}:v]scale=640:360:force_original_aspect_ratio=decrease,"
            f"pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
            f"drawtext=text='{labels[index]}':x=18:y=18:fontsize=38:"
            f"fontcolor=white:box=1:boxcolor=black@0.7[v{index}]")
    while len(chains) < 4:
        index = len(chains)
        chains.append(f"color=c=black:s=640x360:d=1[v{index}]")
    filter_graph = ";".join(chains) + ";" + \
        "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
    command = [ffmpeg_bin, "-y", *inputs, "-filter_complex", filter_graph,
               "-map", "[out]", "-frames:v", "1", str(destination)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not destination.is_file():
        raise V82Blocked("prepare", "mosaic_generation_failed",
                         (completed.stderr or "")[-500:])
    return destination


def build_profile_mosaics(profiles_manifest: Mapping[str, Any], output_dir: Path, *,
                          ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """Create one positive mosaic per form plus one cross-form negative mosaic."""
    profiles = [row for row in profiles_manifest.get("profiles") or []
                if row.get("status") == "usable"]
    if len(profiles) != 3:
        raise V82Blocked("prepare", "three_usable_forms_required_for_mosaics")
    form_order = {"form_black_cat": 0, "form_black_hair_child": 1,
                  "form_white_hair_child": 2}
    profiles.sort(key=lambda row: form_order.get(
        str(row.get("form_id")).split("/")[-1], 99))
    positives = []
    negatives = []
    for profile in profiles:
        examples = profile.get("examples") or {}
        positive_paths = [Path(str(examples[key]["full_frame"]))
                          for key in sorted(examples)
                          if key == "seed" or key.startswith("positive_")]
        if len(positive_paths) < 3:
            raise V82Blocked("prepare", "three_positive_views_required_for_mosaic")
        form_name = str(profile["form_id"]).split("/")[-1]
        mosaic = _make_mosaic(
            positive_paths[:3], Path(output_dir) / f"positive_{_safe_id(form_name)}.jpg",
            ffmpeg_bin=ffmpeg_bin)
        positives.append({"form_id": form_name, "path": str(mosaic),
                          "sha256": sha256_file(mosaic),
                          "source_images": [str(path) for path in positive_paths[:3]]})
        negative_key = next((key for key in sorted(examples)
                             if key.startswith("negative_")), None)
        if not negative_key:
            raise V82Blocked("prepare", "hard_negative_required_for_mosaic")
        negatives.append(Path(str(examples[negative_key]["full_frame"])))
    negative_mosaic = _make_mosaic(
        negatives, Path(output_dir) / "hard_negatives.jpg", ffmpeg_bin=ffmpeg_bin)
    result = {
        "schema_version": "profile_mosaics_v82",
        "ordered_inputs": [*positives, {
            "form_id": "hard_negatives", "path": str(negative_mosaic),
            "sha256": sha256_file(negative_mosaic),
            "source_images": [str(path) for path in negatives],
        }],
        "input_order_contract": [
            "positive_black_cat", "positive_black_hair_child",
            "positive_white_hair_child", "hard_negatives"],
        "each_subject_visually_discernible": "requires_processor_audit_and_preflight_review",
    }
    _write_json(Path(output_dir) / "mosaics_manifest.json", result)
    return result


def resolve_target_and_form(
        form_comparisons: Iterable[Mapping[str, Any]], *,
        target_id: str = FIRST_ROUND_TARGET,
        hard_negative_matches: bool = False,
        transforming: bool = False,
        continuous_character_context: bool = False) -> dict[str, Any]:
    """Resolve character first, then form; multi-form matches are not conflict."""
    rows = [dict(row) for row in form_comparisons]
    symmetric_same = [row for row in rows
                      if row.get("forward") == row.get("reverse") == "same"]
    had_uncertain = any("uncertain" in {row.get("forward"), row.get("reverse")}
                        for row in rows)
    forms = sorted({str(row["form_id"]) for row in symmetric_same})
    if hard_negative_matches and forms:
        character_status, form_status = "conflict", "uncertain"
        reason = "target_and_different_character_hard_negative_both_match"
    elif forms:
        character_status = "supported"
        form_status = "transforming" if transforming else (
            "identified" if len(forms) == 1 else "uncertain")
        reason = ("one_target_form_supported" if len(forms) == 1
                  else "character_supported_form_ambiguous")
    elif continuous_character_context:
        character_status, form_status = "uncertain", "unknown_form_candidate"
        reason = "known_character_context_but_no_known_form_matches"
    elif had_uncertain:
        character_status, form_status = "uncertain", "uncertain"
        reason = "identity_evidence_uncertain"
    else:
        character_status, form_status = "rejected", "uncertain"
        reason = "all_known_target_forms_rejected"
    return {
        "target_id": target_id if character_status == "supported" else None,
        "character_id": target_id if character_status == "supported" else None,
        "character_status": character_status,
        "form_candidates": forms,
        "form_status": form_status,
        "review_status": "unreviewed",
        "reason": reason,
    }


# Public compatibility name required by the V8.2 implementation plan.
resolve_character_and_form = resolve_target_and_form


def _flashvid_server_audit(answer: Any) -> dict[str, Any]:
    raw = getattr(answer, "raw", {}) or {}
    audit = raw.get("flashvid_audit") or raw.get("token_audit") or {}
    if not isinstance(audit, Mapping):
        return {}
    return dict(audit)


def validate_flashvid_audit(answer: Any, *, expected_fps: float,
                            expected_retention: float,
                            expected_image_count: int,
                            interval: Sequence[float],
                            transport_audit: Mapping[str, Any] | None = None) \
        -> dict[str, Any]:
    """Fail closed unless the server exposes post-processor/token facts."""
    request = dict(getattr(answer, "request_audit", {}) or {})
    server = _flashvid_server_audit(answer)
    missing = sorted(REQUIRED_FLASHVID_AUDIT_FIELDS.difference(server))
    if missing:
        raise V82Blocked(
            "infrastructure", "token_audit_unavailable", ",".join(missing))
    count = int(server["actual_sampled_frame_count"])
    indices = list(server["actual_frame_indices"])
    relative = list(map(float, server["actual_frame_timestamps_relative_s"]))
    absolute = list(map(float, server["actual_frame_timestamps_absolute_s"]))
    if not count or not (count == len(indices) == len(relative) == len(absolute)):
        raise V82Blocked("infrastructure", "sampling_audit_failed", "count mismatch")
    start, end = map(float, interval)
    if any(value < start - 1e-6 or value > end + 1e-6 for value in absolute):
        raise V82Blocked("infrastructure", "sampling_audit_failed", "PTS out of scope")
    actual_ratio = float(server["actual_retention_ratio"])
    before, after = int(server["video_tokens_before"]), int(server["video_tokens_after"])
    if before <= 0 or after <= 0 or after > before:
        raise V82Blocked("infrastructure", "token_audit_invalid")
    video_grids = list(server["processed_video_grid_thw"])
    if len(video_grids) != 1 or len(video_grids[0]) != 3:
        raise V82Blocked("infrastructure", "video_processor_audit_mismatch")
    temporal = int(video_grids[0][0])
    expected_after = (before if expected_retention >= 1.0 else
                      max(math.ceil(before / temporal),
                          int(before * expected_retention)))
    expected_actual_ratio = expected_after / before
    if (after != expected_after or
            abs(actual_ratio - expected_actual_ratio) > max(1 / before, 1e-6)):
        raise V82Blocked("infrastructure", "retention_ratio_mismatch")
    if int(request.get("image_count", -1)) != int(expected_image_count):
        raise V82Blocked("infrastructure", "identity_album_input_count_mismatch")
    if server["video_transport_sha256"] != request.get("transport_clip_sha256"):
        raise V82Blocked("infrastructure", "server_transport_hash_mismatch")
    if list(server["image_sha256"]) != list(request.get("identity_album_sha256") or []):
        raise V82Blocked("infrastructure", "server_identity_album_hash_mismatch")
    effective_fps = count / max(1e-9, end - start)
    if abs(effective_fps - expected_fps) > .5:
        raise V82Blocked("infrastructure", "sampling_fps_mismatch")
    image_sizes = list(server["input_image_sizes"])
    image_grids = list(server["processed_image_grid_thw"])
    if len(image_sizes) != expected_image_count or len(image_grids) != expected_image_count:
        raise V82Blocked("infrastructure", "image_processor_audit_mismatch")
    if transport_audit:
        expected_relative = list(map(
            float, transport_audit.get("actual_frame_timestamps_relative_s") or []))
        if (count != int(transport_audit.get("actual_frame_count", -1)) or
                len(expected_relative) != len(relative) or
                any(abs(left - right) > 1e-4
                    for left, right in zip(expected_relative, relative))):
            raise V82Blocked("infrastructure", "server_transport_pts_mismatch")
    result = {
        "schema_version": "flashvid_input_token_audit_v82",
        "request": request,
        "server": server,
        "effective_fps": round(effective_fps, 6),
        "audit_complete": True,
        "images_compressed_by_flashvid": False,
    }
    return result


BROWSE_PROMPT = """The first video is a 45-second source-movie transport. The still
images are a human-confirmed visual album for one target character: three positive form
mosaics followed by one hard-negative mosaic. Search for possible appearances of that
same character, interactions directly involving it, or a visible result plausibly linked
to it. This is recall-first browsing: broad uncertain ranges are allowed and absence is
never proved here. Do not infer identity from names, dialogue, plot memory, or action type.
Return one JSON object:
{"status":"observed|not_observed|unreliable","overflow":false,
 "regions":[{"interval":[relative_start,relative_end],
 "kind":"target_direct|related_interaction|possible_outcome|uncertain",
 "reason":"concrete visible reason"}]}
Use at most 12 regions. Times belong only to the source video. not_observed means only
that this browse did not observe a candidate; it never means absent."""


def _parse_browse_response(text: str, *, start_s: float, end_s: float,
                           block_id: str, arm: str) -> dict[str, Any]:
    payload = _single_json_object(text)
    status = str(payload.get("status") or "")
    if status not in {"observed", "not_observed", "unreliable"}:
        raise V82Blocked("browsing", "invalid_browse_status")
    raw_regions = payload.get("regions") or []
    if status != "observed" and raw_regions:
        raise V82Blocked("browsing", "non_observed_browse_contains_regions")
    if len(raw_regions) > 12:
        raise V82Blocked("browsing", "browse_region_limit_exceeded")
    duration = end_s - start_s
    regions = []
    for index, raw in enumerate(raw_regions):
        rel = _interval(raw.get("interval"), duration_s=duration,
                        field="browse_relative_interval")
        kind = str(raw.get("kind") or "")
        if kind not in {"target_direct", "related_interaction", "possible_outcome",
                        "uncertain"}:
            raise V82Blocked("browsing", "invalid_browse_region_kind")
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise V82Blocked("browsing", "browse_region_reason_missing")
        absolute = [round(start_s + rel[0], 6), round(start_s + rel[1], 6)]
        regions.append({
            "candidate_id": f"browse_{arm}_{block_id}_{index:02d}",
            "source_interval": absolute,
            "raw_relative_interval": rel,
            "lead_kind": "multiform_visual_browse",
            "lead_id": f"{arm}/{block_id}/{index:02d}",
            "browse_arm": arm,
            "kind": kind,
            "reason": reason,
            "not_observed_is_absent": False,
        })
    return {"status": status, "overflow": bool(payload.get("overflow")),
            "regions": regions, "raw": payload}


def _stage_media(clip: Path, media_root: str | None, prefix: str) -> tuple[Path, dict]:
    if not media_root:
        return clip, {"staged": False, "source": str(clip),
                      "sha256": sha256_file(clip)}
    root = Path(media_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4", dir=root,
                                     delete=False) as handle:
        staged = Path(handle.name)
    shutil.copy2(clip, staged)
    if sha256_file(staged) != sha256_file(clip):
        staged.unlink(missing_ok=True)
        raise V82Blocked("infrastructure", "media_staging_hash_mismatch")
    return staged, {"staged": True, "source": str(clip), "request": str(staged),
                    "sha256": sha256_file(staged)}


def _browse_one_interval(cfg: AppConfig, client: Any, *, source_video: Path,
                         output_dir: Path, interval: Sequence[float], block_id: str,
                         album_images: Sequence[Path], fps: float,
                         retention_ratio: float, arm: str, media_root: str | None,
                         prompt: str = BROWSE_PROMPT) -> dict:
    start_s, end_s = map(float, interval)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    ffprobe = str(cfg.perception.get("ffprobe_bin", "ffprobe"))
    clip, transport = _build_coverage_transport(
        ffmpeg, ffprobe, source_video, output_dir / "transport",
        start_s=start_s, end_s=end_s, requested_fps=fps)
    request_clip, staging = _stage_media(clip, media_root, f"v82_{block_id}_")
    try:
        answer = client.watch(
            request_clip, prompt, duration_s=end_s - start_s,
            image_paths=list(album_images), max_tokens=1536,
            source_time_origin_s=start_s)
    finally:
        if request_clip != clip:
            request_clip.unlink(missing_ok=True)
    raw_path = output_dir / "raw_response.txt"
    raw_path.write_text(str(answer.text), encoding="utf-8")
    parsed = _parse_browse_response(
        str(answer.text), start_s=start_s, end_s=end_s, block_id=block_id, arm=arm)
    audit = validate_flashvid_audit(
        answer, expected_fps=fps, expected_retention=retention_ratio,
        expected_image_count=len(album_images), interval=interval,
        transport_audit=transport)
    result = {
        "schema_version": "character_recall_browse_block_v82",
        "block_id": block_id, "arm": arm,
        "source_interval": [start_s, end_s],
        "status": "covered" if parsed["status"] != "unreliable" else "unreliable",
        "browse_status": parsed["status"], "overflow": parsed["overflow"],
        "regions": parsed["regions"], "transport_audit": transport,
        "media_staging": staging, "flashvid_audit": audit,
        "raw_response_path": str(raw_path),
        "response_sha256": hashlib.sha256(str(answer.text).encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    _write_json(output_dir / "result.json", result)
    return result


def run_target_recall_browse(
        cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path, *,
        source_video: Path, album_images: Sequence[Path], r025_client: Any,
        vanilla_client: Any | None = None, source_sha256: str | None = None,
        reuse_completed: bool = True,
        search_card: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Browse every eligible 45-second block and audit a blind vanilla sample."""
    output_dir = Path(output_dir)
    browse = spec["browse"]
    duration = common.video_duration_s(
        str(cfg.perception.get("ffprobe_bin", "ffprobe")), source_video)
    blocks = coverage_blocks(
        duration, block_s=45.0, head_s=float(browse.get("head_s", 90)),
        tail_s=float(browse.get("tail_s", 360)),
        min_movie_s=float(browse.get("min_movie_s", 1800)))
    source_digest = source_sha256 or sha256_file(source_video)
    browse_prompt = build_browse_prompt(search_card) if search_card else BROWSE_PROMPT
    contract = {
        "schema_version": "character_recall_browse_contract_v82",
        "source_sha256": source_digest,
        "prompt_sha256": hashlib.sha256(browse_prompt.encode()).hexdigest(),
        "search_card_sha256": ((search_card or {}).get("search_card_sha256")
                               if search_card else None),
        "album_sha256": [sha256_file(Path(path)) for path in album_images],
        "fps": 4.0, "retention_ratio": .25, "block_s": 45.0,
    }
    contract_sha = _stable_sha(contract)
    completed: dict[str, dict] = {}

    def browse_with_overflow(interval: Sequence[float], block_id: str,
                             block_dir: Path, depth: int = 0) -> dict:
        result = _browse_one_interval(
            cfg, r025_client, source_video=source_video, output_dir=block_dir,
            interval=interval, block_id=block_id, album_images=album_images,
            fps=4.0, retention_ratio=.25, arm="r025",
            media_root=str(browse.get("media_root") or ""), prompt=browse_prompt)
        if not result.get("overflow"):
            return result
        start, end = map(float, interval)
        if depth >= 3 or end - start <= 6.0:
            raise V82Blocked("browsing", "overflow_unresolved_after_split", block_id)
        midpoint = (start + end) / 2
        children = [browse_with_overflow(
            child_interval, f"{block_id}_s{child_index}",
            block_dir / "overflow" / f"s{child_index}", depth + 1)
            for child_index, child_interval in enumerate(
                ([start, midpoint], [midpoint, end]))]
        result["regions_before_overflow_split"] = result.get("regions") or []
        result["regions"] = [region for child in children
                             for region in child.get("regions") or []]
        result["overflow_children"] = [
            {"block_id": child["block_id"],
             "source_interval": child["source_interval"],
             "result_path": str(block_dir / "overflow" /
                                f"s{index}" / "result.json")}
            for index, child in enumerate(children)]
        result["overflow"] = False
        result["overflow_resolved_by_split"] = True
        _write_json(block_dir / "result.json", result)
        return result

    def work(index: int, interval: Sequence[float]) -> dict:
        block_id = f"b{index:04d}"
        block_dir = output_dir / "r025" / "blocks" / block_id
        result_path = block_dir / "result.json"
        cache_key = _stable_sha({"contract": contract_sha, "interval": interval})
        if reuse_completed and result_path.is_file():
            cached = _read_json(result_path)
            if cached.get("cache_key") == cache_key and cached.get("status") == "covered":
                return cached
        result = browse_with_overflow(interval, block_id, block_dir)
        result["cache_key"] = cache_key
        _write_json(result_path, result)
        return result

    jobs = []
    workers = min(max(1, int(browse.get("workers", 4))), max(1, len(blocks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, interval in enumerate(blocks):
            jobs.append((index, executor.submit(work, index, interval)))
        for index, future in jobs:
            block_id = f"b{index:04d}"
            try:
                completed[block_id] = future.result()
            except Exception as exc:
                failed = {
                    "schema_version": "character_recall_browse_block_v82",
                    "block_id": block_id, "source_interval": list(blocks[index]),
                    "status": "failed", "failure_class": "infrastructure",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                completed[block_id] = failed
                _write_json(output_dir / "r025" / "blocks" / block_id / "result.json", failed)
    ordered = [completed[f"b{index:04d}"] for index in range(len(blocks))]
    not_observed = [row for row in ordered if row.get("browse_status") == "not_observed"]
    audit_sample: list[dict] = []
    if vanilla_client is not None and not_observed:
        by_quarter: dict[int, list[dict]] = defaultdict(list)
        eligible_start, eligible_end = blocks[0][0], blocks[-1][1]
        span = max(1e-9, eligible_end - eligible_start)
        for row in not_observed:
            center = sum(row["source_interval"]) / 2
            quarter = min(3, int(4 * (center - eligible_start) / span))
            by_quarter[quarter].append(row)
        rng = random.Random(int(browse.get("vanilla_audit_seed", 20260915)))
        for quarter in range(4):
            population = by_quarter.get(quarter, [])
            sample_n = max(1, math.ceil(len(population) * .10)) if population else 0
            audit_sample.extend(rng.sample(population, min(sample_n, len(population))))
        for row in audit_sample:
            block_id = str(row["block_id"])
            result = _browse_one_interval(
                cfg, vanilla_client, source_video=source_video,
                output_dir=output_dir / "vanilla_audit" / "blocks" / block_id,
                interval=row["source_interval"], block_id=block_id,
                album_images=album_images, fps=4.0, retention_ratio=1.0,
                arm="vanilla", media_root=str(browse.get("vanilla_media_root") or
                                               browse.get("media_root") or ""),
                prompt=browse_prompt)
            row["vanilla_audit_result"] = result
    vanilla_leaks = [row for row in audit_sample
                     if (row.get("vanilla_audit_result") or {}).get("regions")]
    vanilla_upgrades = []
    if vanilla_leaks and vanilla_client is not None:
        sampled_ids = {str(row["block_id"]) for row in audit_sample}
        for row in not_observed:
            if str(row["block_id"]) in sampled_ids:
                result_row = row.get("vanilla_audit_result") or {}
            else:
                block_id = str(row["block_id"])
                result_row = _browse_one_interval(
                    cfg, vanilla_client, source_video=source_video,
                    output_dir=output_dir / "vanilla_upgrade" / "blocks" / block_id,
                    interval=row["source_interval"], block_id=block_id,
                    album_images=album_images, fps=4.0, retention_ratio=1.0,
                    arm="vanilla", media_root=str(browse.get("vanilla_media_root") or
                                                   browse.get("media_root") or ""),
                    prompt=browse_prompt)
            vanilla_upgrades.append(result_row)
    final_candidates = [
        region for row in [*ordered, *vanilla_upgrades]
        for region in row.get("regions") or []]
    result = {
        "schema_version": "character_recall_browse_manifest_v82",
        "contract": contract, "contract_sha256": contract_sha,
        "duration_s": duration, "block_count": len(ordered),
        "covered_count": sum(row.get("status") == "covered" for row in ordered),
        "failed_count": sum(row.get("status") == "failed" for row in ordered),
        "unreliable_count": sum(row.get("status") == "unreliable" for row in ordered),
        "not_observed_count": len(not_observed),
        "r025_candidate_count": sum(len(row.get("regions") or []) for row in ordered),
        "candidate_count": len(final_candidates),
        "final_candidate_block_ids": sorted({
            str(row["block_id"]) for row in ordered
            if any(_overlap(row["source_interval"], region["source_interval"]) > 0
                   for region in final_candidates)}),
        "blocks": ordered,
        "vanilla_blind_audit": {
            "sample_count": len(audit_sample),
            "leak_count": len(vanilla_leaks),
            "all_remaining_not_observed_require_vanilla": bool(vanilla_leaks),
            "sample_block_ids": [row["block_id"] for row in audit_sample],
            "upgraded_remaining_block_count": len(vanilla_upgrades),
        },
        "complete": all(row.get("status") == "covered" for row in ordered),
    }
    _write_json(output_dir / "browse_manifest.json", result)
    _write_jsonl(output_dir / "visual_candidates.jsonl", final_candidates)
    return result


run_character_recall_browse = run_target_recall_browse


def align_interval_outward(interval: Sequence[float], shot_intervals: Iterable[Sequence[float]],
                           *, duration_s: float) -> list[float]:
    """Snap only outwards; a detector failure leaves the original interval intact."""
    original = _interval(interval, duration_s=duration_s)
    overlaps = [list(map(float, row)) for row in shot_intervals
                if len(row) == 2 and _overlap(original, row) > 0]
    if not overlaps:
        return original
    return [round(max(0.0, min([original[0], *[row[0] for row in overlaps]])), 6),
            round(min(duration_s, max([original[1], *[row[1] for row in overlaps]])), 6)]


def build_candidate_scenes(
        visual_candidates: Iterable[Mapping[str, Any]],
        mention_rows: Iterable[Mapping[str, Any]] = (),
        prior_ranges: Iterable[Mapping[str, Any]] = (),
        neighbor_ranges: Iterable[Mapping[str, Any]] = (), *, duration_s: float,
        shot_intervals: Iterable[Sequence[float]] = (), merge_gap_s: float = 4.0,
        output_path: Path | None = None) -> list[dict[str, Any]]:
    """Build traceable logical scenes from all four lead sources without Top-K."""
    leads: list[dict[str, Any]] = []
    for source_name, rows in (
            ("visual", visual_candidates), ("prior", prior_ranges),
            ("neighbor", neighbor_ranges)):
        for index, raw in enumerate(rows):
            interval = _interval(raw.get("source_interval") or raw.get("interval"),
                                 duration_s=duration_s)
            leads.append({"raw_interval": interval, "aligned_interval": align_interval_outward(
                interval, shot_intervals, duration_s=duration_s),
                "source": source_name, "source_id": raw.get("lead_id") or
                raw.get("candidate_id") or raw.get("id") or f"{source_name}_{index:05d}"})
    for index, raw in enumerate(mention_rows):
        interval = _interval(raw.get("interval"), duration_s=duration_s)
        expanded = [max(0.0, interval[0] - 20.0), min(duration_s, interval[1] + 20.0)]
        leads.append({"raw_interval": interval, "aligned_interval": align_interval_outward(
            expanded, shot_intervals, duration_s=duration_s),
            "source": "asr", "source_id": raw.get("mention_id") or f"mention_{index:05d}"})
    scenes: list[dict[str, Any]] = []
    for lead in sorted(leads, key=lambda row: row["aligned_interval"]):
        interval = lead["aligned_interval"]
        if scenes and interval[0] <= scenes[-1]["source_interval"][1] + merge_gap_s:
            scenes[-1]["source_interval"][1] = max(
                scenes[-1]["source_interval"][1], interval[1])
            scenes[-1]["lead_refs"].append(lead)
        else:
            scenes.append({"source_interval": list(interval), "lead_refs": [lead]})
    for index, scene in enumerate(scenes):
        scene["source_interval"] = [round(value, 6) for value in scene["source_interval"]]
        identity_payload = {
            "source_interval": scene["source_interval"],
            "lead_refs": sorted((str(row.get("source")), str(row.get("source_id")))
                                for row in scene["lead_refs"]),
        }
        scene["candidate_scene_id"] = f"scene_{_stable_sha(identity_payload)[:12]}"
        scene["logical_scene_not_model_batch"] = True
    if output_path is not None:
        _write_jsonl(output_path, scenes)
    return scenes


def build_observation_tasks(
        scenes: Iterable[Mapping[str, Any]], *, fps: float = 4.0,
        max_frames: int = 80, overlap_s: float = 4.0,
        shot_intervals: Iterable[Sequence[float]] = (),
        output_path: Path | None = None) -> list[dict[str, Any]]:
    """Split logical scenes by model capacity without declaring event boundaries."""
    if fps <= 0 or max_frames <= 0:
        raise ValueError("positive observation fps/frame budget required")
    width = max_frames / fps
    if not 0 <= overlap_s < width:
        raise ValueError("observation overlap must be smaller than task width")
    shots = sorted([list(map(float, row)) for row in shot_intervals if len(row) == 2])
    tasks = []
    for scene in scenes:
        scene_id = str(scene["candidate_scene_id"])
        start, end = map(float, scene["source_interval"])
        cursor = start
        part = 0
        while cursor < end - 1e-6:
            target_end = min(end, cursor + width)
            candidates = [row[1] for row in shots
                          if cursor + width * .55 <= row[1] <= target_end]
            task_end = max(candidates) if candidates else target_end
            if task_end <= cursor + 1e-6:
                task_end = target_end
            task_id = f"task_{scene_id}_{part:03d}"
            tasks.append({
                "task_id": task_id, "candidate_scene_id": scene_id,
                "source_interval": [round(cursor, 6), round(task_end, 6)],
                "fps": fps, "max_frames": max_frames,
                "state": "pending", "task_kind": "neutral_observation",
                "lead_refs": list(scene.get("lead_refs") or []),
                "attempt_count": 0, "infrastructure_failures": [],
            })
            if task_end >= end - 1e-6:
                break
            cursor = max(cursor + .001, task_end - overlap_s)
            part += 1
    if output_path is not None:
        _write_jsonl(output_path, tasks)
    return tasks


def make_bridge_task(left: Mapping[str, Any], right: Mapping[str, Any], *,
                     padding_s: float = 2.0) -> dict[str, Any]:
    boundary = (float(left["source_interval"][1]) +
                float(right["source_interval"][0])) / 2
    start = max(float(left["source_interval"][0]), boundary - padding_s)
    end = min(float(right["source_interval"][1]), boundary + padding_s)
    return {
        "task_id": f"bridge_{_safe_id(left['task_id'])}_{_safe_id(right['task_id'])}",
        "candidate_scene_id": left["candidate_scene_id"],
        "source_interval": [round(start, 6), round(end, 6)],
        "fps": float(left.get("fps", 4.0)), "max_frames": 80,
        "state": "pending", "task_kind": "bridge_observation",
        "bridge_sources": [left["task_id"], right["task_id"]],
        "attempt_count": 0, "infrastructure_failures": [],
    }


def make_boundary_context_task(
        task: Mapping[str, Any], *, direction: str, source_duration_s: float,
        max_context_s: float = 40.0, task_width_s: float = 20.0,
        overlap_s: float = 4.0) -> dict[str, Any] | None:
    """Extend an incomplete scene edge without letting context grow forever."""
    if direction not in {"left", "right"}:
        raise ValueError("boundary context direction must be left or right")
    start, end = map(float, task["source_interval"])
    chain = list(map(float, task.get("context_chain_interval") or [start, end]))
    if direction == "left":
        chain_start = max(0.0, max(chain[1] - max_context_s,
                                   chain[0] - (task_width_s - overlap_s)))
        interval = [chain_start, min(source_duration_s, chain[0] + overlap_s)]
        expanded_chain = [chain_start, chain[1]]
    else:
        chain_end = min(source_duration_s, min(chain[0] + max_context_s,
                                               chain[1] + task_width_s - overlap_s))
        interval = [max(0.0, chain[1] - overlap_s), chain_end]
        expanded_chain = [chain[0], chain_end]
    if (interval[1] <= interval[0] + 1e-6 or
            expanded_chain[1] - expanded_chain[0] <=
            chain[1] - chain[0] + 1e-6):
        return None
    identity = _stable_sha({
        "parent": task["task_id"], "direction": direction,
        "interval": interval, "chain": expanded_chain,
    })[:12]
    return {
        "task_id": f"context_{direction}_{identity}",
        "candidate_scene_id": task["candidate_scene_id"],
        "source_interval": [round(value, 6) for value in interval],
        "context_chain_interval": [round(value, 6) for value in expanded_chain],
        "fps": float(task.get("fps", 4.0)), "max_frames": 80,
        "state": "pending", "task_kind": "boundary_context",
        "context_parent_task_id": task["task_id"], "direction": direction,
        "attempt_count": 0, "infrastructure_failures": [],
    }


def build_neighbor_ranges(occurrences: Iterable[Mapping[str, Any]], *,
                          source_duration_s: float, step_s: float = 20.0,
                          clear_intervals_to_stop: int = 2) -> list[dict[str, Any]]:
    """Schedule two outward investigation rings around newly supported targets.

    Two rings operationalise the "two clear neighbouring intervals" stop rule.
    Unreliable or failed results never count as clear; queue evaluation decides
    their status later.
    """
    if step_s <= 0 or clear_intervals_to_stop < 1:
        raise ValueError("invalid neighbour expansion budget")
    rows = []
    for occurrence in occurrences:
        occurrence_id = str(occurrence["occurrence_id"])
        start, end = _interval(occurrence["source_interval"],
                               duration_s=source_duration_s)
        for direction in ("left", "right"):
            for ring in range(clear_intervals_to_stop):
                if direction == "left":
                    left = max(0.0, start - (ring + 1) * step_s)
                    right = max(0.0, start - ring * step_s)
                else:
                    left = min(source_duration_s, end + ring * step_s)
                    right = min(source_duration_s, end + (ring + 1) * step_s)
                if right <= left + 1e-6:
                    continue
                rows.append({
                    "id": f"neighbor_{_safe_id(occurrence_id)}_{direction}_{ring + 1}",
                    "source_interval": [round(left, 6), round(right, 6)],
                    "lead_kind": "verified_target_neighbor",
                    "trigger_occurrence_id": occurrence_id,
                    "direction": direction, "ring": ring + 1,
                    "absence_claimed": False,
                })
    return rows


def split_observation_task_on_oom(task: Mapping[str, Any], *,
                                  overlap_s: float = 4.0) -> list[dict[str, Any]]:
    """Cover the complete failed interval at the next 20 -> 16 -> 12 budget."""
    start, end = map(float, task["source_interval"])
    duration = end - start
    next_width = 16.0 if duration > 16.0 + 1e-6 else (
        12.0 if duration > 12.0 + 1e-6 else None)
    if next_width is None:
        return []
    rows = []
    cursor = start
    index = 0
    while cursor < end - 1e-6:
        child_end = min(end, cursor + next_width)
        rows.append({
            **{key: value for key, value in dict(task).items()
               if key not in {"task_id", "source_interval", "state",
                              "attempt_count", "attempt_failures"}},
            "task_id": f"{task['task_id']}_oom{int(next_width)}_{index:02d}",
            "source_interval": [round(cursor, 6), round(child_end, 6)],
            "state": "pending", "attempt_count": 0,
            "attempt_failures": [], "oom_parent_task_id": task["task_id"],
        })
        if child_end >= end - 1e-6:
            break
        cursor = max(cursor + .001, child_end - overlap_s)
        index += 1
    return rows


def _is_oom_error(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in message or "cuda oom" in message or "cublas" in message


def _normalise_observation(payload: Mapping[str, Any], task: Mapping[str, Any],
                           source_video: Path) -> tuple[list[dict], list[dict]]:
    start, end = map(float, task["source_interval"])
    duration = end - start
    local_map = {}
    occurrences = []
    for index, raw in enumerate(payload.get("occurrences") or []):
        local_id = str(raw["local_id"])
        rel = _interval(raw["visible_interval"], duration_s=duration,
                        field="occurrence_interval")
        occurrence_id = f"occ_{_safe_id(task['task_id'])}_{index:02d}_{_safe_id(local_id)}"
        local_map[local_id] = occurrence_id
        row = {
            "schema_version": "occurrence_v82", "occurrence_id": occurrence_id,
            "source_interval": [round(start + rel[0], 6), round(start + rel[1], 6)],
            "local_description": str(raw["local_description"]),
            "visual_state": str(raw["visual_state"]), "roi": raw.get("roi"),
            "observation_task_id": task["task_id"], "source_video": str(source_video),
            "fact_status": "candidate", "automation_status": "automatic",
            "human_review_status": "unreviewed",
        }
        row["observation_sha256"] = _stable_sha(row)
        occurrences.append(row)
    events = []
    for index, raw in enumerate(payload.get("event_candidates") or []):
        rel = _interval(raw["interval"], duration_s=duration, field="event_interval")
        actor_key = str(raw.get("initiator_local_id") or "")
        patient_key = str(raw.get("affected_local_id") or "")
        actor = local_map.get(actor_key) if actor_key else None
        patient = local_map.get(patient_key) if patient_key else None
        if (actor_key and not actor) or (patient_key and not patient) or not (actor or patient):
            raise V82Blocked("observe", "event_references_unknown_occurrence")
        events.append({
            "schema_version": "neutral_event_candidate_v82",
            "event_candidate_id": f"event_{_safe_id(task['task_id'])}_{index:02d}",
            "source_interval": [round(start + rel[0], 6), round(start + rel[1], 6)],
            "initiator_occurrence_id": actor, "affected_occurrence_id": patient,
            "action": str(raw.get("action_or_state") or ""),
            "visible_result": str(raw.get("visible_result") or raw.get("result") or ""),
            "relation_status": str(raw.get("relation_status") or "uncertain"),
            "fact_status": "candidate", "observation_task_id": task["task_id"],
            "automation_status": "automatic", "human_review_status": "unreviewed",
        })
    return occurrences, events


def run_observe_queue(
        cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path, *,
        source_video: Path, runner: Any, tasks: Iterable[Mapping[str, Any]],
        reuse_completed: bool = True,
        source_duration_s: float | None = None) -> dict[str, Any]:
    """Run/resume neutral observation tasks. Identity is deliberately separate.

    Infrastructure failures are retried twice. OOM retries shrink the execution
    batch interval 20 -> 16 -> 12 seconds without lowering the 4fps sampling.
    Incomplete boundaries become bridge/context tasks rather than inferred facts.
    """
    output_dir = Path(output_dir)
    source_duration = (float(source_duration_s) if source_duration_s is not None else
                       common.video_duration_s(
                           str(cfg.perception.get("ffprobe_bin", "ffprobe")),
                           source_video))
    max_context_s = float(spec.get("observe", {}).get("max_context_s", 40.0))
    queue_path = output_dir / "observe_queue.jsonl"
    prior = {row["task_id"]: row for row in _read_jsonl(queue_path)} \
        if reuse_completed and queue_path.is_file() else {}
    queue_rows = []
    for raw in tasks:
        row = dict(raw)
        cached = prior.get(str(row["task_id"]))
        if cached and cached.get("state") in TERMINAL_QUEUE_STATES:
            row = cached
        if row.get("state") not in QUEUE_STATES:
            raise ValueError(f"invalid queue state: {row.get('state')}")
        queue_rows.append(row)
    _write_jsonl(queue_path, queue_rows)
    contract_sha = hashlib.sha256((V82_NEUTRAL_PROMPT +
        "\nV8.2 roles may be null/not_applicable; no names or editing functions.").encode()).hexdigest()
    occurrences = []
    events = []
    task_results = []
    pending_bridges = []
    pending_oom_splits = []
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in queue_rows:
        by_scene[str(row["candidate_scene_id"])].append(row)
    preloaded_answers: dict[str, Any] = {}
    initial_batch_error = None
    initial_pending = [row for row in queue_rows
                       if row.get("state") not in TERMINAL_QUEUE_STATES]
    if len(initial_pending) > 1 and hasattr(runner, "watch_many"):
        requests = []
        for row in initial_pending:
            start, end = map(float, row["source_interval"])
            requests.append({
                "video_path": source_video, "prompt": V82_NEUTRAL_PROMPT,
                "kwargs": {
                    "start_s": start, "end_s": end,
                    "clip_dir": (output_dir / "tasks" / _safe_id(row["task_id"]) /
                                 "attempt_01" / "source_clip"),
                    "fps": 4.0, "duration_s": end - start,
                    "max_new_tokens": 768, "use_audio_in_video": False,
                    "stop_after_json_object": True,
                },
            })
        try:
            answers = runner.watch_many(requests)
            if len(answers) != len(initial_pending):
                raise V82Blocked("observe", "observation_response_count_mismatch")
            preloaded_answers = {row["task_id"]: answer
                                 for row, answer in zip(
                                     initial_pending, answers, strict=True)}
        except Exception as exc:
            # Fall back to bounded per-task retries so one failed worker does
            # not erase the successful-task accounting contract.
            initial_batch_error = f"{type(exc).__name__}: {exc}"
    for row in queue_rows:
        if row.get("state") in TERMINAL_QUEUE_STATES:
            result_path = output_dir / "tasks" / _safe_id(row["task_id"]) / "result.json"
            if result_path.is_file():
                result = _read_json(result_path)
                task_results.append(result)
                occurrences.extend(result.get("occurrences") or [])
                events.extend(result.get("event_candidates") or [])
            continue
        task_dir = output_dir / "tasks" / _safe_id(row["task_id"])
        row["state"] = "running"
        _write_jsonl(queue_path, queue_rows)
        initial_duration = (float(row["source_interval"][1]) -
                            float(row["source_interval"][0]))
        # Three total attempts at one complete execution interval. OOM does
        # not truncate the tail: it schedules overlapping child tasks that
        # cover the complete parent at the next 20 -> 16 -> 12 budget.
        durations = [initial_duration] * 3
        result = None
        for attempt, width in enumerate(durations, 1):
            original_start, original_end = map(float, row["source_interval"])
            start, end = original_start, min(original_end, original_start + width)
            row["attempt_count"] = attempt
            try:
                if attempt == 1 and row["task_id"] in preloaded_answers:
                    answer = preloaded_answers[row["task_id"]]
                else:
                    answer = runner.watch(
                        source_video, V82_NEUTRAL_PROMPT,
                        start_s=start, end_s=end,
                        clip_dir=task_dir / f"attempt_{attempt:02d}" / "source_clip",
                        fps=4.0, duration_s=end - start, max_new_tokens=768,
                        use_audio_in_video=False, stop_after_json_object=True)
                raw_text = str(answer.text)
                raw_path = task_dir / f"attempt_{attempt:02d}" / "raw_response.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(raw_text, encoding="utf-8")
                payload = _single_json_object(raw_text)
                status = validate_target_neutral_observation(
                    payload, duration_s=end - start)
                if status == "observed":
                    task_used = {**row, "source_interval": [start, end]}
                    new_occ, new_events = _normalise_observation(
                        payload, task_used, source_video)
                    state = "completed_observed"
                elif status == "observed_empty":
                    new_occ, new_events = [], []
                    state = ("completed_mention_only" if any(
                        ref.get("source") == "asr" for ref in row.get("lead_refs") or [])
                             else "completed_uncertain")
                else:
                    new_occ, new_events, state = [], [], "completed_uncertain"
                left_complete = payload.get("left_context_complete") is not False
                right_complete = payload.get("right_context_complete") is not False
                if not left_complete or not right_complete:
                    state = "needs_context"
                result = {
                    "schema_version": "observation_task_result_v82",
                    "contract_sha256": contract_sha, "task_id": row["task_id"],
                    "candidate_scene_id": row["candidate_scene_id"],
                    "source_interval": [start, end], "state": state,
                    "observation_status": status, "occurrences": new_occ,
                    "event_candidates": new_events,
                    "left_context_complete": left_complete,
                    "right_context_complete": right_complete,
                    "boundary_reason": payload.get("boundary_reason"),
                    "sampling": getattr(answer, "sampling", None),
                    "gpu_pair": getattr(answer, "gpu_pair", None),
                    "raw_response_path": str(raw_path),
                }
                if state == "needs_context":
                    scene_tasks = sorted(by_scene[str(row["candidate_scene_id"])],
                                         key=lambda item: item["source_interval"])
                    position = next(i for i, item in enumerate(scene_tasks)
                                    if item["task_id"] == row["task_id"])
                    scheduled = []
                    for direction, complete, neighbor_index in (
                            ("left", left_complete, position - 1),
                            ("right", right_complete, position + 1)):
                        if complete:
                            continue
                        if 0 <= neighbor_index < len(scene_tasks):
                            scheduled.append(make_bridge_task(
                                scene_tasks[min(position, neighbor_index)],
                                scene_tasks[max(position, neighbor_index)]))
                        else:
                            context_task = make_boundary_context_task(
                                row, direction=direction,
                                source_duration_s=source_duration,
                                max_context_s=max_context_s)
                            if context_task is not None:
                                scheduled.append(context_task)
                    pending_bridges.extend(scheduled)
                    state = "completed_uncertain"
                    result["state"] = state
                    result["context_tasks_scheduled"] = [
                        item["task_id"] for item in scheduled]
                    result["partial"] = not bool(scheduled)
                    if not scheduled:
                        result["reason_code"] = "context_safety_limit_reached"
                break
            except Exception as exc:
                verification_failure = isinstance(exc, V8Blocked)
                row.setdefault("attempt_failures", []).append({
                    "attempt": attempt, "duration_s": width,
                    "failure_class": ("verification" if verification_failure
                                      else "infrastructure"),
                    "error": f"{type(exc).__name__}: {exc}"})
                if _is_oom_error(exc):
                    children = split_observation_task_on_oom(row)
                    if children:
                        pending_oom_splits.extend(children)
                        result = {
                            "schema_version": "observation_task_result_v82",
                            "contract_sha256": contract_sha,
                            "task_id": row["task_id"],
                            "candidate_scene_id": row["candidate_scene_id"],
                            "source_interval": row["source_interval"],
                            "state": "completed_uncertain", "occurrences": [],
                            "event_candidates": [], "failure_class": "infrastructure",
                            "reason_code": "oom_split_scheduled",
                            "child_task_ids": [child["task_id"] for child in children],
                            "failures": row["attempt_failures"],
                        }
                        break
                if attempt >= len(durations):
                    result = {
                        "schema_version": "observation_task_result_v82",
                        "contract_sha256": contract_sha, "task_id": row["task_id"],
                        "candidate_scene_id": row["candidate_scene_id"],
                        "source_interval": row["source_interval"],
                        "state": ("completed_uncertain" if verification_failure
                                  else "failed_infrastructure"),
                        "occurrences": [], "event_candidates": [],
                        "failure_class": ("verification" if verification_failure
                                          else "infrastructure"),
                        "reason_code": ("observation_contract_failed_after_retries"
                                        if verification_failure else
                                        "observation_task_failed_after_retries"),
                        "failures": row["attempt_failures"],
                    }
        if result is None:
            raise AssertionError("observe task produced no result")
        row["state"] = result["state"]
        _write_json(task_dir / "result.json", result)
        task_results.append(result)
        _write_jsonl(queue_path, queue_rows)
        occurrences.extend(result.get("occurrences") or [])
        events.extend(result.get("event_candidates") or [])
    known_ids = {row["task_id"] for row in queue_rows}
    for bridge in pending_bridges:
        if bridge["task_id"] not in known_ids:
            queue_rows.append(bridge)
            known_ids.add(bridge["task_id"])
    for child in pending_oom_splits:
        if child["task_id"] not in known_ids:
            queue_rows.append(child)
            known_ids.add(child["task_id"])
    _write_jsonl(queue_path, queue_rows)
    _write_jsonl(output_dir / "occurrence_bank.jsonl", occurrences)
    _write_jsonl(output_dir / "neutral_event_candidates.jsonl", events)
    partial_ranges = [row["source_interval"] for row in task_results
                      if row.get("partial") is True]
    unreliable_ranges = [row["source_interval"] for row in task_results
                         if row.get("observation_status") == "unreliable"]
    incomplete_ranges = [row["source_interval"] for row in queue_rows
                         if row.get("state") in {"pending", "needs_context",
                                                 "failed_infrastructure"}]
    incomplete_ranges.extend(partial_ranges)
    manifest = {
        "schema_version": "observe_queue_manifest_v82",
        "task_count": len(queue_rows),
        "states": {state: sum(row.get("state") == state for row in queue_rows)
                   for state in sorted(QUEUE_STATES)},
        "occurrence_count": len(occurrences), "event_candidate_count": len(events),
        "new_bridge_task_count": len(pending_bridges),
        "new_oom_split_task_count": len(pending_oom_splits),
        "initial_parallel_batch_size": len(preloaded_answers),
        "initial_parallel_batch_error": initial_batch_error,
        "queue_closed": all(row.get("state") in TERMINAL_QUEUE_STATES
                            for row in queue_rows),
        "processing_complete": not incomplete_ranges and all(
            row.get("state") in TERMINAL_QUEUE_STATES for row in queue_rows),
        "incomplete_ranges": incomplete_ranges,
        "partial_ranges": partial_ranges,
        "unreliable_ranges": unreliable_ranges,
    }
    _write_json(output_dir / "observe_queue_manifest.json", manifest)
    return manifest


def bind_target_and_form(
        cfg: AppConfig, occurrences: Iterable[Mapping[str, Any]],
        profiles_manifest: Mapping[str, Any], output_dir: Path, *,
        source_video: Path, runner: Any,
        target_id: str = FIRST_ROUND_TARGET,
        reuse_completed: bool = True) -> dict[str, Any]:
    """Bidirectionally compare all forms, resolving character separately from form."""
    profiles = [row for row in profiles_manifest.get("profiles") or []
                if row.get("status") == "usable" and
                row.get("character_id") == target_id]
    if len(profiles) != 3:
        raise V82Blocked("identity", "three_usable_target_forms_required")
    ffmpeg = str(cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    output_dir = Path(output_dir)
    binding_path = output_dir / "identity_bindings.jsonl"
    profile_contract = _stable_sha([
        {"form_id": row.get("form_id"), "profile_version": row.get("profile_version")}
        for row in profiles])
    cached = {str(row.get("occurrence_id")): row
              for row in (_read_jsonl(binding_path)
                          if reuse_completed and binding_path.is_file() else [])}
    bindings = []
    prepared: dict[str, tuple[dict[str, Any], Path, Path | None]] = {}
    for occurrence in occurrences:
        occurrence = dict(occurrence)
        occurrence_id = str(occurrence["occurrence_id"])
        prior = cached.get(occurrence_id)
        if (prior and prior.get("occurrence_observation_sha256") ==
                occurrence.get("observation_sha256") and
                prior.get("profile_contract_sha256") == profile_contract):
            bindings.append(prior)
            continue
        interval = _interval(occurrence["source_interval"])
        frame_dir = output_dir / "frames" / _safe_id(occurrence_id)
        frame = export_frame(ffmpeg, source_video, sum(interval) / 2,
                             frame_dir / "full.jpg")
        roi = None
        if occurrence.get("roi") is not None:
            try:
                roi = export_roi(ffmpeg, frame, occurrence["roi"], frame_dir / "roi.jpg")
            except Exception:
                roi = None
        prepared[occurrence_id] = (occurrence, frame, roi)

    requests = []
    request_meta = []
    for occurrence_id, (_occurrence, frame, roi) in prepared.items():
        for profile in profiles:
            for reverse in (False, True):
                images, prompt, metadata = _occurrence_profile_request(
                    frame, roi, profile, reverse=reverse)
                requests.append({"image_paths": images, "prompt": prompt,
                                 "kwargs": {"max_new_tokens": 256}})
                request_meta.append({
                    "occurrence_id": occurrence_id, "kind": "positive",
                    "form_id": str(profile["form_id"]).split("/")[-1],
                    "profile_version": profile.get("profile_version"),
                    "direction": "reverse" if reverse else "forward",
                    "metadata": metadata,
                })
            examples = profile.get("examples") or {}
            negative_key = next((key for key in sorted(examples)
                                 if key.startswith("negative_")), None)
            if negative_key:
                negative = Path(str(examples[negative_key]["full_frame"]))
                for reverse in (False, True):
                    images = [negative, frame] if reverse else [frame, negative]
                    if roi is not None:
                        images.append(roi)
                    requests.append({
                        "image_paths": images, "prompt": IDENTITY_COMPARE_PROMPT,
                        "kwargs": {"max_new_tokens": 256},
                    })
                    request_meta.append({
                        "occurrence_id": occurrence_id, "kind": "negative",
                        "form_id": str(profile["form_id"]).split("/")[-1],
                        "direction": "reverse" if reverse else "forward",
                        "metadata": {"input_images": [str(path) for path in images],
                                     "hard_negative": str(negative),
                                     "candidate_full": str(frame)},
                    })
    if requests:
        answers = (runner.inspect_media_many(requests)
                   if hasattr(runner, "inspect_media_many") else [
                       runner.inspect_media(row["image_paths"], row["prompt"],
                                            **row["kwargs"]) for row in requests])
        if len(answers) != len(requests):
            raise V82Blocked("identity", "identity_response_count_mismatch")
    else:
        answers = []
    comparison_data: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: {"positive": {}, "negative": {}})
    for meta, answer in zip(request_meta, answers, strict=True):
        parsed = _parse_object(str(answer.text))
        result_value = str(parsed.get("result") or "").lower()
        if result_value not in {"same", "different", "uncertain"}:
            raise V82Blocked("identity", "invalid_identity_response")
        form_row = comparison_data[meta["occurrence_id"]][meta["kind"]].setdefault(
            meta["form_id"], {"profile_version": meta.get("profile_version"),
                              "evidence": {}})
        form_row[meta["direction"]] = result_value
        form_row["evidence"][meta["direction"]] = {
            **meta["metadata"], "raw_response": str(answer.text), "parsed": parsed}

    for occurrence_id, (occurrence, _frame, _roi) in prepared.items():
        comparisons = []
        negative_matches = []
        for profile in profiles:
            form_id = str(profile["form_id"]).split("/")[-1]
            positive = comparison_data[occurrence_id]["positive"][form_id]
            comparisons.append({"form_id": form_id, **positive})
            negative = comparison_data[occurrence_id]["negative"].get(form_id)
            if negative:
                negative_matches.append({
                    "profile_form_id": form_id, **negative,
                    "symmetric_same": (negative.get("forward") ==
                                       negative.get("reverse") == "same"),
                })
        resolution = resolve_target_and_form(
            comparisons, target_id=target_id,
            hard_negative_matches=any(row["symmetric_same"]
                                      for row in negative_matches),
            continuous_character_context=bool(
                occurrence.get("continuous_character_context")))
        binding = {
            "schema_version": "character_form_binding_v82",
            "occurrence_id": occurrence_id, **resolution,
            "comparisons": comparisons,
            "hard_negative_comparisons": negative_matches,
            "occurrence_observation_sha256": occurrence.get("observation_sha256"),
            "automation_status": "automatic",
            "profile_contract_sha256": profile_contract,
        }
        binding["binding_sha256"] = _stable_sha(binding)
        bindings.append(binding)
    bindings.sort(key=lambda row: str(row["occurrence_id"]))
    _write_jsonl(binding_path, bindings)
    result = {
        "schema_version": "character_form_binding_manifest_v82",
        "bindings": bindings,
        "metrics": {
            "total": len(bindings),
            "supported": sum(row["character_status"] == "supported" for row in bindings),
            "uncertain": sum(row["character_status"] == "uncertain" for row in bindings),
            "conflict": sum(row["character_status"] == "conflict" for row in bindings),
            "model_supported_coverage": round(sum(
                row["character_status"] == "supported" for row in bindings) /
                len(bindings), 6) if bindings else 0.0,
            "false_merge": None, "false_rejection": None,
        },
    }
    _write_json(output_dir / "identity_manifest.json", result)
    return result


bind_character_and_form = bind_target_and_form


V82_FACT_PROMPT = """Independently verify one neutral visual fact from the source clip.
The supplied occurrence IDs are local bookkeeping labels, not character identities. Do
not name a character, use plot knowledge, or assign an editing role. Return one JSON:
{"passed":true,"initiator_occurrence_id":"id or null",
 "affected_occurrence_id":"id or null","action_or_state":"concrete visible fact",
 "visible_result":"concrete visible change or empty",
 "relation_status":"supported|not_applicable|requires_separate_check|uncertain",
 "claim_bounds":{"can_prove":["bounded claims"],"cannot_prove":["stronger claims"]}}
Use null initiator when only a state/result is visible. Use requires_separate_check when a
cut or off-screen change prevents direct causal proof. Use only listed occurrence IDs."""

V82_RELATION_PROMPT = """Independently inspect the continuous source context around a
neutral fact. Decide only whether the visible earlier action supports the later state
change across the cut/off-screen interval. Do not identify characters. Return one JSON:
{"passed":true,"relation_supported":true,
 "can_prove":["bounded relation"],"cannot_prove":["stronger claim"]}
Use relation_supported=false when continuity is insufficient."""


def verify_target_event_facts(
        cfg: AppConfig, event_candidates: Iterable[Mapping[str, Any]],
        occurrences: Iterable[Mapping[str, Any]], *, source_video: Path,
        runner: Any, target_occurrence_ids: set[str], output_dir: Path,
        source_duration_s: float | None = None) -> dict[str, Any]:
    """Recheck facts independently; identity support never promotes a fact itself."""
    output_dir = Path(output_dir)
    occurrence_map = {str(row["occurrence_id"]): dict(row) for row in occurrences}
    duration = source_duration_s or common.video_duration_s(
        str(cfg.perception.get("ffprobe_bin", "ffprobe")), source_video)
    candidates = []
    requests = []
    for raw in event_candidates:
        row = dict(raw)
        participants = {str(value) for value in (
            row.get("initiator_occurrence_id"), row.get("affected_occurrence_id")) if value}
        if not participants.intersection(target_occurrence_ids):
            continue
        source_interval = _interval(row["source_interval"], duration_s=duration)
        start = max(0.0, source_interval[0] - 2.0)
        end = min(duration, source_interval[1] + 2.0)
        context = [{
            "occurrence_id": occurrence_id,
            "local_description": occurrence_map.get(occurrence_id, {}).get(
                "local_description"),
            "visible_state": occurrence_map.get(occurrence_id, {}).get("visual_state"),
        } for occurrence_id in sorted(participants)]
        prompt = V82_FACT_PROMPT + "\nAllowed local occurrences:\n" + json.dumps(
            context, ensure_ascii=False)
        candidates.append({"row": row, "allowed_ids": participants,
                           "source_interval": source_interval,
                           "watch_interval": [start, end], "prompt": prompt})
        requests.append({
            "video_path": source_video, "prompt": prompt,
            "kwargs": {"start_s": start, "end_s": end,
                       "clip_dir": output_dir / _safe_id(
                           row["event_candidate_id"]) / "source_clip",
                       "fps": 12.0, "duration_s": end - start,
                       "max_new_tokens": 512, "use_audio_in_video": False,
                       "stop_after_json_object": True},
        })
    if requests:
        answers = (runner.watch_many(requests) if hasattr(runner, "watch_many") else
                   [runner.watch(item["video_path"], item["prompt"], **item["kwargs"])
                    for item in requests])
    else:
        answers = []
    facts = []
    failures = []
    relation_pending = []
    for candidate, answer in zip(candidates, answers, strict=True):
        raw = candidate["row"]
        event_id = f"verified_{raw['event_candidate_id']}"
        event_dir = output_dir / _safe_id(str(raw["event_candidate_id"]))
        response_path = event_dir / "fact_raw_response.txt"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(str(answer.text), encoding="utf-8")
        try:
            payload = _single_json_object(str(answer.text))
            if payload.get("passed") is not True:
                raise V82Blocked("events", "fact_not_supported")
            initiator = payload.get("initiator_occurrence_id")
            affected = payload.get("affected_occurrence_id")
            initiator = str(initiator) if initiator else None
            affected = str(affected) if affected else None
            if (initiator and initiator not in candidate["allowed_ids"]) or (
                    affected and affected not in candidate["allowed_ids"]):
                raise V82Blocked("events", "fact_references_unknown_occurrence")
            if not (initiator or affected):
                raise V82Blocked("events", "fact_has_no_local_participant")
            relation = str(payload.get("relation_status") or "uncertain")
            if relation not in {"supported", "not_applicable",
                                "requires_separate_check", "uncertain"}:
                raise V82Blocked("events", "invalid_fact_relation_status")
            fact = {
                "schema_version": "target_event_fact_v82", "event_id": event_id,
                "source_interval": candidate["source_interval"],
                "observation_interval": candidate["watch_interval"],
                "initiator_occurrence_id": initiator,
                "affected_occurrence_id": affected,
                "action": _concrete(payload.get("action_or_state"), "verified_action"),
                "visible_result": _concrete(payload.get("visible_result"),
                                             "verified_result", allow_empty=True),
                "fact_status": "supported", "relation_status": (
                    "uncertain" if relation == "requires_separate_check" else relation),
                "claim_bounds": dict(payload.get("claim_bounds") or {}),
                "verification": {"status": "verified",
                                 "prompt_sha256": hashlib.sha256(
                                     candidate["prompt"].encode()).hexdigest(),
                                 "raw_response_path": str(response_path),
                                 "sampling": getattr(answer, "sampling", None)},
                "automation_status": "automatic",
                "human_review_status": "unreviewed",
            }
            facts.append(fact)
            if relation == "requires_separate_check":
                relation_pending.append((fact, candidate, event_dir))
        except Exception as exc:
            failures.append({"event_id": event_id,
                             "failure_class": "verification",
                             "reason": f"{type(exc).__name__}: {exc}",
                             "raw_response_path": str(response_path)})
    for fact, candidate, event_dir in relation_pending:
        request = next(item for item in requests if
                       item["prompt"] == candidate["prompt"])
        answer = runner.watch(
            source_video, V82_RELATION_PROMPT,
            start_s=request["kwargs"]["start_s"], end_s=request["kwargs"]["end_s"],
            clip_dir=event_dir / "relation_clip", fps=12.0,
            duration_s=request["kwargs"]["duration_s"], max_new_tokens=256,
            use_audio_in_video=False, stop_after_json_object=True)
        relation_path = event_dir / "relation_raw_response.txt"
        relation_path.write_text(str(answer.text), encoding="utf-8")
        payload = _single_json_object(str(answer.text))
        fact["relation_status"] = (
            "supported" if payload.get("passed") is True and
            payload.get("relation_supported") is True else "uncertain")
        fact["relation_verification"] = {
            "raw_response_path": str(relation_path), "parsed": payload}
    for fact in facts:
        fact["event_fact_sha256"] = _stable_sha(fact)
    _write_jsonl(output_dir / "event_facts.jsonl", facts)
    result = {"schema_version": "target_event_fact_manifest_v82",
              "event_facts": facts, "verified_count": len(facts),
              "blocked_count": len(failures), "failures": failures}
    _write_json(output_dir / "event_fact_manifest.json", result)
    return result


def build_target_timelines(
        candidate_records: Iterable[Mapping[str, Any]],
        occurrences: Iterable[Mapping[str, Any]],
        bindings: Iterable[Mapping[str, Any]],
        event_facts: Iterable[Mapping[str, Any]], output_dir: Path,
        human_reviews: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write four distinct timeline layers; never relabel human data automatic."""
    output_dir = Path(output_dir)
    bindings_by_occ = {str(row["occurrence_id"]): dict(row) for row in bindings}
    supported_occurrences = []
    for raw in occurrences:
        row = dict(raw)
        binding = bindings_by_occ.get(str(row["occurrence_id"]), {})
        if binding.get("character_status") == "supported":
            supported_occurrences.append({
                **row, "character_binding": binding,
                "automation_status": "model_supported",
                "human_review_status": binding.get("review_status", "unreviewed"),
            })
    automatic_events = []
    supported_ids = {str(row["occurrence_id"]) for row in supported_occurrences}
    for raw in event_facts:
        row = dict(raw)
        participant_ids = {str(item.get("occurrence_id"))
                           for item in row.get("participants") or []
                           if item.get("occurrence_id")}
        if not participant_ids:
            participant_ids = {
                str(value) for value in (
                    row.get("initiator_occurrence_id"), row.get("affected_occurrence_id"))
                if value
            }
        fact_supported = (row.get("fact_status") == "supported" or
                          (row.get("verification") or {}).get("status") == "verified")
        nested_facts = row.get("facts") or []
        relation_supported = (
            row.get("relation_status") in {"supported", "not_applicable"} or
            bool(nested_facts) and all(
                fact.get("relation_verified") is True or
                not fact.get("relation_check_required") for fact in nested_facts))
        if participant_ids.intersection(supported_ids) and fact_supported and relation_supported:
            automatic_events.append({
                **row, "automation_status": "automatic_supported",
                "human_review_status": "unreviewed",
            })
    human_rows = []
    if human_reviews:
        decisions = human_reviews.get("events") or {}
        for row in automatic_events:
            decision = decisions.get(str(row.get("event_id") or
                                         row.get("event_candidate_id")))
            if isinstance(decision, Mapping) and decision.get("confirmed") is True:
                human_rows.append({**row, "automation_status": "automatic_then_reviewed",
                                   "human_review_status": "human_confirmed",
                                   "human_review": dict(decision)})
    candidate_rows = [dict(row) for row in candidate_records]
    _write_jsonl(output_dir / "candidate_records.jsonl", candidate_rows)
    _write_jsonl(output_dir / "automatic_character_occurrences.jsonl",
                 supported_occurrences)
    _write_jsonl(output_dir / "automatic_event_timeline.jsonl", automatic_events)
    _write_jsonl(output_dir / "human_confirmed_timeline.jsonl", human_rows)
    result = {
        "schema_version": "character_timelines_manifest_v82",
        "candidate_record_count": len(candidate_rows),
        "automatic_character_occurrence_count": len(supported_occurrences),
        "automatic_event_count": len(automatic_events),
        "human_confirmed_event_count": len(human_rows),
        "human_confirmed_is_pure_automatic": False,
    }
    _write_json(output_dir / "timeline_manifest.json", result)
    return result


build_character_timelines = build_target_timelines


def _quarter(interval: Sequence[float], eligible: Sequence[float]) -> int:
    center = sum(map(float, interval)) / 2
    span = max(1e-9, float(eligible[1]) - float(eligible[0]))
    return min(3, max(0, int(4 * (center - float(eligible[0])) / span)))


def build_stratified_audit_pack(
        blocks: Iterable[Mapping[str, Any]], *, eligible_interval: Sequence[float],
        output_dir: Path, per_stratum: int = 5, random_seed: int = 20260915,
        source_video: Path | None = None, ffmpeg_bin: str = "ffmpeg",
        append_per_stratum: int = 0, max_total: int = 120) -> dict[str, Any]:
    """Freeze and blind-sample hit/miss x movie-quarter 45-second blocks.

    Repeated calls can append a predeclared number of whole blocks per stratum.
    The frozen population must remain equivalent and existing anonymous IDs are
    retained.
    """
    output_dir = Path(output_dir)
    populations: dict[str, list[dict]] = defaultdict(list)
    frozen = []
    for raw in blocks:
        row = dict(raw)
        status = "hit" if row.get("candidate_hit") else "miss"
        stratum = f"{status}_q{_quarter(row['source_interval'], eligible_interval) + 1}"
        item = {**row, "stratum": stratum}
        frozen.append(item)
        populations[stratum].append(item)
    freeze_sha = _stable_sha(frozen)
    frozen_path = output_dir / "frozen_population.json"
    contacts_path = output_dir / "blind_contact_sheet.json"
    key_path = output_dir / "private_sample_key.json"
    previous_contacts: list[dict[str, Any]] = []
    previous_key: dict[str, Any] = {}
    if append_per_stratum and frozen_path.is_file():
        previous_frozen = _read_json(frozen_path)
        if previous_frozen.get("frozen_sha256") != freeze_sha:
            raise V82Blocked("audit", "frozen_population_changed")
        previous_contacts = list(_read_json(contacts_path).get("items") or [])
        previous_key = _read_json(key_path)
    selected_ids_by_stratum: dict[str, set[str]] = defaultdict(set)
    for value in previous_key.values():
        selected_ids_by_stratum[str(value["stratum"])].add(str(value["block_id"]))
    selected: list[tuple[str, dict[str, Any]]] = []
    design = []
    for stratum in sorted(populations):
        population = populations[stratum]
        already = selected_ids_by_stratum[stratum]
        target_n = min(len(population),
                       (len(already) + append_per_stratum)
                       if append_per_stratum else per_stratum)
        stratum_seed = int(hashlib.sha256(
            f"{random_seed}:{stratum}".encode()).hexdigest()[:16], 16)
        ordered = list(population)
        random.Random(stratum_seed).shuffle(ordered)
        sample = [row for row in ordered
                  if str(row.get("block_id")) not in already][:
                      max(0, target_n - len(already))]
        nh = len(already) + len(sample)
        if nh == 0:
            continue
        design.append({"stratum": stratum, "Nh": len(population), "nh": nh,
                       "sampling_probability": nh / len(population),
                       "design_weight": len(population) / nh})
        selected.extend((stratum, row) for row in sample)
    available_slots = max(0, max_total - len(previous_contacts))
    selected = selected[:available_slots]
    contacts = list(previous_contacts)
    private_key = dict(previous_key)
    for index, (stratum, row) in enumerate(selected, start=len(contacts)):
        anonymous_id = f"audit_{index:03d}"
        media_path = None
        if source_video is not None:
            media_path = output_dir / "media" / f"{anonymous_id}.mp4"
            media_path.parent.mkdir(parents=True, exist_ok=True)
            start, end = map(float, row["source_interval"])
            common.run_ffmpeg(ffmpeg_bin, [
                "-y", "-ss", str(start), "-to", str(end), "-i", str(source_video),
                "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "25",
                "-preset", "veryfast", "-c:a", "aac", str(media_path)], timeout_s=300)
        contacts.append({
            "anonymous_id": anonymous_id, "source_interval": row["source_interval"],
            "media_path": str(media_path) if media_path else None,
            "annotation_fields": {
                "target_appearances": [], "important_visible_events": [],
                "identity_outcomes": [], "actor_affected_checks": [],
                "supported_binding_false_merge": False,
            },
        })
        private_key[anonymous_id] = {
            "block_id": row.get("block_id"), "stratum": stratum,
            "source_interval": row.get("source_interval"),
        }
    selected_count_by_stratum: dict[str, int] = defaultdict(int)
    for value in private_key.values():
        selected_count_by_stratum[str(value["stratum"])] += 1
    design = [{"stratum": name, "Nh": len(population),
               "nh": selected_count_by_stratum[name],
               "sampling_probability": selected_count_by_stratum[name] / len(population),
               "design_weight": len(population) / selected_count_by_stratum[name]}
              for name, population in sorted(populations.items())
              if selected_count_by_stratum[name]]
    _write_json(frozen_path, {
        "frozen_sha256": freeze_sha, "blocks": frozen})
    _write_json(output_dir / "sampling_design.json", {
        "schema_version": "stratified_audit_design_v82", "strata": design,
        "random_seed": random_seed, "per_stratum": per_stratum})
    _write_json(contacts_path, {
        "schema_version": "blind_audit_contact_sheet_v82", "items": contacts,
        "model_outputs_visible": False, "strata_visible": False})
    _write_json(key_path, private_key)
    return {"sample_count": len(contacts), "strata": design,
            "frozen_sha256": freeze_sha, "contacts": contacts,
            "appended_count": len(selected),
            "private_key_path": str(key_path)}


def _appearance_hit(interval: Sequence[float], investigated: Iterable[Sequence[float]]) -> bool:
    duration = float(interval[1]) - float(interval[0])
    if duration < 1.0:
        center = (float(interval[0]) + float(interval[1])) / 2
        return any(float(row[0]) <= center <= float(row[1]) for row in investigated)
    return interval_coverage_ratio(interval, investigated) >= .5


def attach_audit_design(
        annotation_rows: Iterable[Mapping[str, Any]], *,
        private_key: Mapping[str, Any], sampling_design: Mapping[str, Any]) \
        -> list[dict[str, Any]]:
    """Join blind annotations to private strata and compile count contracts."""
    weights = {str(row["stratum"]): float(row["design_weight"])
               for row in sampling_design.get("strata") or []}
    compiled = []
    for raw in annotation_rows:
        row = dict(raw)
        anonymous_id = str(row.get("anonymous_id") or "")
        key = private_key.get(anonymous_id)
        if not isinstance(key, Mapping):
            raise V82Blocked("audit", "unknown_anonymous_audit_id", anonymous_id)
        stratum = str(key["stratum"])
        if stratum not in weights:
            raise V82Blocked("audit", "audit_stratum_missing_design", stratum)
        appearances = list(row.get("target_appearances") or [])
        events = list(row.get("important_visible_events") or [])
        directions = list(row.get("actor_affected_checks") or [])
        identities = list(row.get("identity_outcomes") or [])
        row.update({
            "stratum": stratum,
            "weight": weights[stratum],
            "true_appearances": row.get("true_appearances", len(appearances)),
            "candidate_hits": row.get("candidate_hits", sum(
                int(bool(item.get("candidate_investigated"))) for item in appearances)),
            "investigated_true_appearances": row.get(
                "investigated_true_appearances", sum(
                    int(bool(item.get("candidate_investigated"))) for item in appearances)),
            "identity_hits": row.get("identity_hits", sum(
                int(bool(item.get("candidate_investigated")) and
                    bool(item.get("identity_retained"))) for item in appearances)),
            "important_events": row.get("important_events", len(events)),
            "event_hits": row.get("event_hits", sum(
                int(bool(item.get("entered_event_timeline"))) for item in events)),
            "direction_checks": row.get("direction_checks", len(directions)),
            "direction_hits": row.get("direction_hits", sum(
                int(bool(item.get("correct"))) for item in directions)),
            "identity_supported_predictions": row.get(
                "identity_supported_predictions", sum(
                    int(bool(item.get("model_supported"))) for item in identities)),
            "identity_correct": row.get("identity_correct", sum(
                int(bool(item.get("model_supported")) and bool(item.get("correct")))
                for item in identities)),
            "identity_false_rejections": row.get(
                "identity_false_rejections", sum(
                    int(bool(item.get("is_target")) and
                        not bool(item.get("model_supported"))) for item in identities)),
            "supported_binding_false_merge": int(bool(
                row.get("supported_binding_false_merge", False))),
        })
        compiled.append(row)
    return compiled


def _wilson_interval(successes: float, trials: float, z: float = 1.959964) \
        -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, successes / trials))
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    spread = z * math.sqrt((p * (1 - p) / trials) + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _weighted_ratio(rows: Sequence[Mapping[str, Any]], numerator: str,
                    denominator: str) -> tuple[float | None, float, float]:
    den = sum(float(row["weight"]) * float(row.get(denominator, 0)) for row in rows)
    num = sum(float(row["weight"]) * float(row.get(numerator, 0)) for row in rows)
    if den <= 0:
        return None, 0.0, 0.0
    weights = [float(row["weight"]) * float(row.get(denominator, 0)) for row in rows
               if float(row.get(denominator, 0)) > 0]
    effective_n = (sum(weights) ** 2 / sum(value * value for value in weights)) \
        if weights and sum(value * value for value in weights) else 0.0
    return num / den, num, max(effective_n, 1.0)


def _stratified_cluster_bootstrap(
        rows: Sequence[Mapping[str, Any]], numerator: str, denominator: str, *,
        replicates: int, seed: int) -> list[float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["stratum"])].append(row)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = []
        for stratum_rows in groups.values():
            sample.extend(rng.choice(stratum_rows) for _ in range(len(stratum_rows)))
        ratio, _, _ = _weighted_ratio(sample, numerator, denominator)
        if ratio is not None:
            values.append(ratio)
    return sorted(values)


def _metric_interval(rows: Sequence[Mapping[str, Any]], numerator: str,
                     denominator: str, *, replicates: int, seed: int) -> dict[str, Any]:
    estimate, weighted_success, effective_n = _weighted_ratio(rows, numerator, denominator)
    if estimate is None:
        return {"estimate": None, "ci95": [0.0, 1.0], "information": "missing"}
    boot = _stratified_cluster_bootstrap(
        rows, numerator, denominator, replicates=replicates, seed=seed)
    lo_index = max(0, int(.025 * len(boot)) - 1) if boot else 0
    hi_index = min(len(boot) - 1, int(.975 * len(boot))) if boot else 0
    boot_lo, boot_hi = ((boot[lo_index], boot[hi_index]) if boot else (0.0, 1.0))
    wilson_lo, wilson_hi = _wilson_interval(estimate * effective_n, effective_n)
    # Wilson prevents an all-success/all-failure cluster sample from claiming zero uncertainty.
    ci = [min(boot_lo, wilson_lo), max(boot_hi, wilson_hi)]
    return {"estimate": round(estimate, 6),
            "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "effective_sample_size": round(effective_n, 3),
            "weighted_success_total": round(weighted_success, 6),
            "bootstrap_unit": "whole_45s_block_within_stratum",
            "zero_error_interval_collapsed": False}


def evaluate_v82_recall(
        audit_rows: Iterable[Mapping[str, Any]], *,
        thresholds: Mapping[str, float] | None = None,
        regression_hard_gate_passed: bool = True,
        max_blocks_reached: bool = False, bootstrap_replicates: int = 2000,
        random_seed: int = 20260915, output_path: Path | None = None,
        auxiliary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Design-weighted metrics and PASS/FAIL/INSUFFICIENT_EVIDENCE gate."""
    thresholds = dict(thresholds or {
        "candidate_interval_recall": .90, "identity_timeline_recall": .90,
        "event_fact_coverage": .85, "actor_affected_accuracy": .95,
    })
    rows = [dict(row) for row in audit_rows]
    metrics = {}
    contracts = {
        "candidate_interval_recall": ("candidate_hits", "true_appearances"),
        "identity_timeline_recall": ("identity_hits", "investigated_true_appearances"),
        "event_fact_coverage": ("event_hits", "important_events"),
        "actor_affected_accuracy": ("direction_hits", "direction_checks"),
    }
    decisions = []
    for index, (name, (num, den)) in enumerate(contracts.items()):
        metric = _metric_interval(rows, num, den, replicates=bootstrap_replicates,
                                  seed=random_seed + index)
        threshold = float(thresholds[name])
        lo, hi = metric["ci95"]
        if metric["estimate"] is None:
            decision = "INSUFFICIENT_EVIDENCE"
        elif lo >= threshold:
            decision = "PASS"
        elif hi < threshold:
            decision = "FAIL"
        else:
            decision = "INSUFFICIENT_EVIDENCE"
        metric.update({"threshold": threshold, "decision": decision})
        metrics[name] = metric
        decisions.append(decision)
    false_merge_count = sum(int(row.get("supported_binding_false_merge", 0)) for row in rows)
    identity_precision = _metric_interval(
        rows, "identity_correct", "identity_supported_predictions",
        replicates=bootstrap_replicates, seed=random_seed + 10)
    false_rejections = sum(
        float(row["weight"]) * float(row.get("identity_false_rejections", 0))
        for row in rows)
    if not regression_hard_gate_passed or false_merge_count:
        overall = "FAIL"
    elif "FAIL" in decisions:
        overall = "FAIL"
    elif all(value == "PASS" for value in decisions):
        overall = "PASS"
    else:
        overall = "INSUFFICIENT_EVIDENCE"
    result = {
        "schema_version": "v82_recall_evaluation_v1",
        "decision": overall, "metrics": metrics,
        "supported_binding_false_merge": false_merge_count,
        "diagnostic_metrics": {
            "identity_precision": identity_precision,
            "weighted_false_rejection_count": round(false_rejections, 6),
        },
        "operational_metrics": dict(auxiliary or {}),
        "regression_hard_gate_passed": regression_hard_gate_passed,
        "max_blocks_reached": max_blocks_reached,
        "next_action": (
            "append_two_blinded_blocks_per_nonempty_stratum" if
            overall == "INSUFFICIENT_EVIDENCE" and not max_blocks_reached else
            "keep_insufficient_evidence" if overall == "INSUFFICIENT_EVIDENCE" else
            "none"),
    }
    if output_path is not None:
        _write_json(output_path, result)
    return result


def run_preflight_flashvid_comparison(
        cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path, *,
        source_video: Path, album_images: Sequence[Path],
        r025_client: Any, vanilla_client: Any,
        search_card: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit compressed/vanilla inputs on difficult cases before Omni expansion."""
    output_dir = Path(output_dir)
    cases = [row for row in (spec.get("preflight") or {}).get("cases") or []
             if row.get("flashvid_compare")]
    rows = []
    browse_prompt = build_browse_prompt(search_card) if search_card else BROWSE_PROMPT
    for case in cases:
        case_id = str(case["id"])
        interval = _interval(case["source_interval"])
        arm_rows = {}
        for arm, client, ratio in (
                ("r025", r025_client, .25), ("vanilla", vanilla_client, 1.0)):
            arm_rows[arm] = _browse_one_interval(
                cfg, client, source_video=source_video,
                output_dir=output_dir / case_id / arm, interval=interval,
                block_id=f"preflight_{_safe_id(case_id)}", album_images=album_images,
                fps=4.0, retention_ratio=ratio, arm=arm,
                media_root=str((spec.get("browse") or {}).get("media_root") or ""),
                prompt=browse_prompt)
        rows.append({
            "case_id": case_id, "source_interval": interval, "arms": arm_rows,
            "both_audits_complete": all(
                arm_rows[arm]["flashvid_audit"]["audit_complete"]
                for arm in ("r025", "vanilla")),
            "candidate_status_agrees": (
                arm_rows["r025"]["browse_status"] ==
                arm_rows["vanilla"]["browse_status"]),
        })
    result = {
        "schema_version": "preflight_flashvid_comparison_v82",
        "case_count": len(rows), "cases": rows,
        "passed": bool(rows) and all(row["both_audits_complete"] for row in rows),
        "semantic_agreement_is_not_accuracy_proof": True,
    }
    _write_json(output_dir / "comparison_manifest.json", result)
    return result


def run_v82_preflight(
        cfg: AppConfig, spec: Mapping[str, Any], output_dir: Path, *,
        source_video: Path, runner: Any, profiles_manifest: Mapping[str, Any],
        ground_truth: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run fixed capability cases without leaking GT into prompts.

    Automatic outputs are frozen first. GT may then score them. A missing GT or
    human review keeps the gate at INSUFFICIENT_EVIDENCE rather than passing.
    """
    output_dir = Path(output_dir)
    cases = list((spec.get("preflight") or {}).get("cases") or [])
    if len(cases) != 12:
        raise ValueError("V8.2 preflight requires exactly 12 fixed cases")
    tasks = [{
        "task_id": f"preflight_{_safe_id(case['id'])}",
        "candidate_scene_id": f"preflight_scene_{index:02d}",
        "source_interval": case["source_interval"], "fps": 4.0, "max_frames": 80,
        "state": "pending", "task_kind": "preflight_neutral_observation",
        "lead_refs": [{"source": "preflight", "source_id": case["id"]}],
        "attempt_count": 0, "infrastructure_failures": [],
    } for index, case in enumerate(cases)]
    all_tasks = list(tasks)
    previous_signature = None
    observe = {}
    for _ in range(8):
        observe = run_observe_queue(
            cfg, spec, output_dir / "observe", source_video=source_video,
            runner=runner, tasks=all_tasks, reuse_completed=True)
        persisted = _read_jsonl(output_dir / "observe" / "observe_queue.jsonl")
        known = {str(row["task_id"]) for row in all_tasks}
        all_tasks.extend(row for row in persisted if str(row["task_id"]) not in known)
        signature = tuple((str(row["task_id"]), str(row["state"]))
                          for row in persisted)
        if observe.get("queue_closed") or signature == previous_signature:
            break
        previous_signature = signature
    occurrences = _read_jsonl(output_dir / "observe" / "occurrence_bank.jsonl")
    bindings = bind_target_and_form(
        cfg, occurrences, profiles_manifest, output_dir / "identity",
        source_video=source_video, runner=runner,
        target_id=str(spec["perception_objective"]["target_id"]))
    frozen = {
        "observe_manifest_sha256": sha256_file(output_dir / "observe" /
                                                "observe_queue_manifest.json"),
        "identity_manifest_sha256": sha256_file(output_dir / "identity" /
                                                 "identity_manifest.json"),
        "gt_used_in_prompt": False,
    }
    _write_json(output_dir / "automatic_outputs_frozen.json", frozen)
    scored = []
    gt_cases = {str(row["id"]): row for row in (ground_truth or {}).get("cases") or []}
    task_results = {_read_json(path)["task_id"]: _read_json(path)
                    for path in (output_dir / "observe" / "tasks").glob("*/result.json")}
    binding_by_occurrence = {str(row["occurrence_id"]): row
                             for row in bindings["bindings"]}
    false_merge_count = 0
    for case in cases:
        case_id = str(case["id"])
        gt = gt_cases.get(case_id)
        automatic = task_results.get(f"preflight_{_safe_id(case_id)}")
        score = {"id": case_id, "scored": gt is not None, "passed": None}
        if gt is not None and automatic is not None:
            # The GT file stores explicit posthoc checks; no semantic string matching.
            expected_status = gt.get("expected_status")
            checks = []
            if expected_status:
                checks.append(automatic.get("observation_status") == expected_status)
            minimum = int(gt.get("minimum_occurrences", 0))
            checks.append(len(automatic.get("occurrences") or []) >= minimum)
            if gt.get("requires_event"):
                checks.append(bool(automatic.get("event_candidates")))
            searchable = json.dumps({
                "occurrences": automatic.get("occurrences") or [],
                "events": automatic.get("event_candidates") or [],
            }, ensure_ascii=False).lower()
            checks.extend(str(term).lower() in searchable
                          for term in gt.get("required_terms") or [])
            checks.extend(str(term).lower() not in searchable
                          for term in gt.get("forbidden_terms") or [])
            occurrence_map = {str(row["occurrence_id"]): row
                              for row in automatic.get("occurrences") or []}
            first_event = next(iter(automatic.get("event_candidates") or []), {})
            initiator = occurrence_map.get(str(
                first_event.get("initiator_occurrence_id") or ""), {})
            affected = occurrence_map.get(str(
                first_event.get("affected_occurrence_id") or ""), {})
            initiator_text = json.dumps(initiator, ensure_ascii=False).lower()
            affected_text = json.dumps(affected, ensure_ascii=False).lower()
            required_initiator = [str(term).lower() for term in
                                  gt.get("initiator_description_any") or []]
            required_affected = [str(term).lower() for term in
                                 gt.get("affected_description_any") or []]
            if required_initiator:
                checks.append(any(term in initiator_text for term in required_initiator))
            if required_affected:
                checks.append(any(term in affected_text for term in required_affected))
            case_bindings = [binding_by_occurrence.get(str(row["occurrence_id"]), {})
                             for row in automatic.get("occurrences") or []]
            supported_count = sum(row.get("character_status") == "supported"
                                  for row in case_bindings)
            if gt.get("target_binding") == "none":
                binding_check = supported_count == 0
                checks.append(binding_check)
                if not binding_check:
                    false_merge_count += supported_count
            elif gt.get("target_binding") == "at_least_one":
                checks.append(supported_count >= 1)
            score["passed"] = all(checks)
            score["checks"] = checks
        scored.append(score)
    scored_rows = [row for row in scored if row["scored"]]
    pass_count = sum(row.get("passed") is True for row in scored_rows)
    identity_conflicts = bindings["metrics"]["conflict"]
    hard_ids = {"direction_5412", "dissolve_5391", "mention_only"}
    hard_pass = all(row.get("passed") is True for row in scored
                    if row["id"] in hard_ids and row["scored"])
    infrastructure_complete = bool(observe.get("processing_complete"))
    if not infrastructure_complete:
        decision = "FAIL"
    elif len(scored_rows) < 12:
        decision = "INSUFFICIENT_EVIDENCE"
    elif (pass_count < 11 or not hard_pass or identity_conflicts or false_merge_count):
        decision = "FAIL"
    elif not (ground_truth or {}).get("human_review_confirmed"):
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        decision = "PASS"
    result = {
        "schema_version": "v82_preflight_manifest_v1", "decision": decision,
        "automatic_outputs_frozen": frozen, "case_count": len(cases),
        "scored_case_count": len(scored_rows), "passed_case_count": pass_count,
        "minimum_passed_required": 11, "hard_regressions_passed": hard_pass,
        "identity_false_merge_proxy_conflicts": identity_conflicts,
        "posthoc_identity_false_merge_count": false_merge_count,
        "observe_queue_closed": observe["queue_closed"], "case_results": scored,
        "infrastructure_complete": infrastructure_complete,
        "incomplete_ranges": observe.get("incomplete_ranges") or [],
        "full_movie_observe_allowed": decision == "PASS",
    }
    _write_json(output_dir / "preflight_manifest.json", result)
    return result


def run_v82_diagnostic(*args, **kwargs):
    """Compatibility alias for orchestration code; phases stay explicit in CLI."""
    return run_v82_preflight(*args, **kwargs)
