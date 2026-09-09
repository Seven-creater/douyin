"""Recipe v2 contract, validation, hashing, and one-way v1 migration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RECIPE_VERSION = "2.0"

TRACK_TYPES = ("video", "overlay", "text", "mask", "audio")
STATUSES = ("supported", "uncertain", "unsupported")
OP_TYPES = (
    "hard_cut", "match_cut", "crossfade", "mask_wipe",
    "foreground_occlusion_transition", "luma_flash", "speed_ramp",
    "tracked_mask_fill", "subject_cutout_composite", "text_layer_animation",
    "montage_burst", "zoom_punch", "whip_pan", "beat_freeze", "trim",
    "time_reorder", "crop_reframe", "color_adjust", "opacity_blend",
    "text_overlay", "unknown_edit",
)
LAYER_OP_TYPES = {
    "mask_wipe", "foreground_occlusion_transition", "tracked_mask_fill",
    "subject_cutout_composite", "text_layer_animation", "text_overlay",
    "opacity_blend",
}
POINT_OP_TYPES = {"hard_cut", "match_cut", "luma_flash", "montage_burst"}
SUPPORTED_RENDER_OPS = {
    "hard_cut", "trim", "speed_ramp", "crop_reframe", "zoom_punch",
    "luma_flash", "color_adjust", "opacity_blend", "text_overlay",
    "text_layer_animation", "crossfade", "beat_freeze",
    "tracked_mask_fill", "subject_cutout_composite", "mask_wipe",
    "foreground_occlusion_transition",
}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_recipe(*, reference_id: str, reference_uri: str, sha256: str,
               duration_s: float, fps: float, model: str = "deterministic",
               prompt_version: str = "recipe_v2", seed: int = 0) -> dict:
    """Return the smallest valid multi-track Recipe v2 envelope."""
    return {
        "recipe_version": RECIPE_VERSION,
        "reference": {
            "id": reference_id,
            "uri": reference_uri,
            "sha256": sha256,
            "duration_s": round(float(duration_s), 6),
            "fps": round(float(fps), 6),
        },
        "assets": [],
        "tracks": [
            {"id": "video_main", "type": "video", "z": 0},
            {"id": "audio_main", "type": "audio", "z": 0},
        ],
        "operations": [],
        "provenance": {
            "model": model,
            "prompt_version": prompt_version,
            "tool_calls": [],
            "seed": int(seed),
        },
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_recipe_v2(recipe: Any) -> list[str]:
    """Validate structure, timeline, references, dependencies, and evidence."""
    if not isinstance(recipe, dict):
        return ["recipe must be an object"]
    errors: list[str] = []
    if recipe.get("recipe_version") != RECIPE_VERSION:
        errors.append(f"recipe_version must be {RECIPE_VERSION}")

    ref = recipe.get("reference")
    if not isinstance(ref, dict):
        errors.append("reference must be an object")
        duration = 0.0
    else:
        for key in ("id", "uri", "sha256", "duration_s", "fps"):
            if key not in ref:
                errors.append(f"reference missing {key}")
        duration = float(ref.get("duration_s") or 0)
        if duration <= 0:
            errors.append("reference.duration_s must be > 0")
        if not _is_number(ref.get("fps")) or ref.get("fps", 0) <= 0:
            errors.append("reference.fps must be > 0")
        digest = ref.get("sha256")
        if not isinstance(digest, str) or (digest and len(digest) != 64):
            errors.append("reference.sha256 must be empty or a 64-char digest")

    assets = recipe.get("assets")
    if not isinstance(assets, list):
        errors.append("assets must be a list")
        assets = []
    asset_ids: set[str] = set()
    for idx, asset in enumerate(assets):
        if not isinstance(asset, dict):
            errors.append(f"assets[{idx}] must be an object")
            continue
        aid = asset.get("id")
        if not isinstance(aid, str) or not aid:
            errors.append(f"assets[{idx}].id missing")
        elif aid in asset_ids:
            errors.append(f"duplicate asset id {aid}")
        else:
            asset_ids.add(aid)
        if not isinstance(asset.get("uri"), str) or not asset.get("uri"):
            errors.append(f"assets[{idx}].uri missing")
        source_range = asset.get("source_range")
        if source_range is not None and (
                not isinstance(source_range, list) or len(source_range) != 2
                or not all(_is_number(v) for v in source_range)
                or source_range[1] <= source_range[0]):
            errors.append(f"assets[{idx}].source_range invalid")

    tracks = recipe.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        errors.append("tracks must be a non-empty list")
        tracks = []
    track_ids: set[str] = set()
    for idx, track in enumerate(tracks):
        if not isinstance(track, dict):
            errors.append(f"tracks[{idx}] must be an object")
            continue
        tid = track.get("id")
        if not isinstance(tid, str) or not tid:
            errors.append(f"tracks[{idx}].id missing")
        elif tid in track_ids:
            errors.append(f"duplicate track id {tid}")
        else:
            track_ids.add(tid)
        if track.get("type") not in TRACK_TYPES:
            errors.append(f"tracks[{idx}].type invalid")
        if not isinstance(track.get("z"), int):
            errors.append(f"tracks[{idx}].z must be int")

    operations = recipe.get("operations")
    if not isinstance(operations, list):
        errors.append("operations must be a list")
        operations = []
    op_ids: set[str] = set()
    dependencies: dict[str, list[str]] = {}
    for idx, op in enumerate(operations):
        if not isinstance(op, dict):
            errors.append(f"operations[{idx}] must be an object")
            continue
        prefix = f"operations[{idx}]"
        oid = op.get("id")
        if not isinstance(oid, str) or not oid:
            errors.append(f"{prefix}.id missing")
            oid = f"__invalid_{idx}"
        elif oid in op_ids:
            errors.append(f"duplicate operation id {oid}")
        op_ids.add(oid)
        if op.get("type") not in OP_TYPES:
            errors.append(f"{prefix}.type invalid")
        if op.get("status") not in STATUSES:
            errors.append(f"{prefix}.status invalid")
        interval = op.get("interval")
        if (not isinstance(interval, list) or len(interval) != 2
                or not all(_is_number(v) for v in interval)):
            errors.append(f"{prefix}.interval invalid")
        else:
            start, end = map(float, interval)
            if start < 0 or end < start or end > duration + 1e-6:
                errors.append(f"{prefix}.interval outside reference")
            if end == start and op.get("type") not in POINT_OP_TYPES:
                errors.append(f"{prefix}.interval must have duration")
        if op.get("track_id") not in track_ids:
            errors.append(f"{prefix}.track_id unknown")
        if not isinstance(op.get("params", {}), dict):
            errors.append(f"{prefix}.params must be an object")
        confidence = op.get("confidence")
        if not _is_number(confidence) or not 0 <= confidence <= 1:
            errors.append(f"{prefix}.confidence outside [0,1]")
        evidence = op.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{prefix}.evidence must be a list")
        elif op.get("status") == "supported" and not evidence:
            errors.append(f"{prefix} supported operation requires evidence")
        else:
            for eidx, item in enumerate(evidence):
                if not isinstance(item, dict) or not item.get("source"):
                    errors.append(f"{prefix}.evidence[{eidx}] missing source")
        inputs = op.get("inputs", [])
        if not isinstance(inputs, list):
            errors.append(f"{prefix}.inputs must be a list")
        depends = op.get("depends_on", [])
        if not isinstance(depends, list) or not all(isinstance(v, str) for v in depends):
            errors.append(f"{prefix}.depends_on must be string list")
            depends = []
        dependencies[oid] = list(depends)

    known_inputs = asset_ids | op_ids
    for idx, op in enumerate(operations):
        if not isinstance(op, dict):
            continue
        for item in op.get("inputs", []):
            if item not in known_inputs:
                errors.append(f"operations[{idx}].inputs references unknown {item}")
        for dep in op.get("depends_on", []):
            if dep not in op_ids:
                errors.append(f"operations[{idx}].depends_on references unknown {dep}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            errors.append(f"operation dependency cycle at {node}")
            return
        if node in visited:
            return
        visiting.add(node)
        for dep in dependencies.get(node, []):
            if dep in dependencies:
                visit(dep)
        visiting.remove(node)
        visited.add(node)

    for op_id in dependencies:
        visit(op_id)

    provenance = recipe.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for key in ("model", "prompt_version", "tool_calls", "seed"):
            if key not in provenance:
                errors.append(f"provenance missing {key}")
        if not isinstance(provenance.get("tool_calls"), list):
            errors.append("provenance.tool_calls must be a list")
    return errors


def recipe_hash(recipe: dict) -> str:
    payload = json.dumps(recipe, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_recipe(recipe: dict, path: Path) -> Path:
    errors = validate_recipe_v2(recipe)
    if errors:
        raise ValueError("invalid Recipe v2: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _track_for_op(op_type: str) -> tuple[str, str, int]:
    if op_type in {"text_layer_animation", "text_overlay"}:
        return "text_main", "text", 20
    if op_type in {"tracked_mask_fill", "mask_wipe"}:
        return "mask_main", "mask", 10
    if op_type in {"subject_cutout_composite", "foreground_occlusion_transition",
                   "opacity_blend"}:
        return "overlay_main", "overlay", 10
    return "video_main", "video", 0


def migrate_v1_to_v2(v1: dict, *, reference_uri: str = "",
                     reference_sha256: str = "", fps: float = 24.0) -> dict:
    """One-way migration. Unknown v1 details are retained under params.legacy."""
    video = v1.get("video") or {}
    duration = float(video.get("duration_s") or 0)
    recipe = new_recipe(
        reference_id=str(video.get("aweme_id") or video.get("id") or "unknown"),
        reference_uri=reference_uri or str(video.get("uri") or "unknown"),
        sha256=reference_sha256 or str(video.get("sha256") or ""),
        duration_s=duration,
        fps=fps,
        model="v1_migration",
        prompt_version="recipe_v1_to_v2",
    )
    tracks = {t["id"]: t for t in recipe["tracks"]}
    for idx, old in enumerate(v1.get("operations") or []):
        if not isinstance(old, dict):
            continue
        op_type = old.get("type") if old.get("type") in OP_TYPES else "hard_cut"
        track_id, track_type, z = _track_for_op(op_type)
        tracks.setdefault(track_id, {"id": track_id, "type": track_type, "z": z})
        start = max(0.0, min(duration, float(old.get("t_s") or 0)))
        op_duration = float(old.get("duration") or old.get("duration_s") or 0)
        end = min(duration, start + max(op_duration, 0.0))
        if end == start and op_type not in POINT_OP_TYPES:
            end = min(duration, start + min(0.25, max(0.001, duration - start)))
        evidence = old.get("evidence")
        if isinstance(evidence, dict):
            evidence = [evidence]
        if not isinstance(evidence, list) or not evidence:
            evidence = [{"source": "v1_migration", "timestamp_s": start}]
        known = {"t_s", "type", "duration", "duration_s", "confidence", "evidence"}
        recipe["operations"].append({
            "id": f"op_{idx:04d}",
            "type": op_type,
            "interval": [round(start, 6), round(end, 6)],
            "track_id": track_id,
            "inputs": [],
            "depends_on": [],
            "params": {"legacy": {k: v for k, v in old.items() if k not in known}},
            "evidence": evidence,
            "confidence": float(old.get("confidence", 0.5)),
            "status": "supported" if old.get("type") in OP_TYPES else "uncertain",
        })
    recipe["tracks"] = sorted(tracks.values(), key=lambda t: (t["z"], t["id"]))
    recipe["provenance"]["legacy_segments"] = v1.get("segments") or []
    return recipe
