"""CPU-only condition catalog and deterministic H3 context preview.

P2 previews are deliberately non-executable until real P0 acceptance and
capability/license gates. The compiler copies only reviewed draft fields and
explicitly permitted asset descriptors; it never asks a model to add facts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .generation_p1 import P1_DRAFT_VERSION
from .generation_v9g import route_h3_mode


CATALOG_VERSION = "v9g_condition_assets_p2"
CONTEXT_VERSION = "v9g_context_preview_p2"
NAMESPACES = {"reference_evidence", "target_assets", "committed_memory"}
TRANSFER_KINDS = {
    "identity", "appearance", "scene", "motion", "rhythm", "camera",
    "audio", "continuity",
}
REFERENCE_ALLOWED = {"motion", "rhythm", "camera", "audio"}
MEDIA_TYPES = {"image", "video", "audio", "video_audio"}


class P2Blocked(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)
        self.reason_code = reason_code


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_condition_asset_catalog(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Register local conditions with source hashes and explicit transfer scope."""
    if not isinstance(entries, list):
        raise P2Blocked("asset_entries_invalid")
    assets = []
    seen = set()
    for entry in entries:
        asset_id = entry.get("asset_id")
        namespace = entry.get("namespace")
        media_type = entry.get("media_type")
        if not isinstance(asset_id, str) or not asset_id or asset_id in seen:
            raise P2Blocked("asset_id_invalid_or_duplicate", str(asset_id))
        if namespace not in NAMESPACES or media_type not in MEDIA_TYPES:
            raise P2Blocked("asset_namespace_or_media_invalid", asset_id)
        seen.add(asset_id)
        path = Path(entry.get("path") or "").resolve()
        if not path.is_file():
            raise P2Blocked("asset_file_missing", asset_id)
        source = entry.get("source")
        if not isinstance(source, str) or not source.strip():
            raise P2Blocked("asset_source_missing", asset_id)
        interval = entry.get("source_interval")
        if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or
                                     not all(isinstance(t, (int, float)) for t in interval) or
                                     interval[0] < 0 or interval[0] >= interval[1]):
            raise P2Blocked("asset_source_interval_invalid", asset_id)
        allowed = entry.get("allowed_transfer")
        forbidden = entry.get("forbidden_transfer")
        descriptors = entry.get("transfer_descriptors")
        if (not isinstance(allowed, list) or not isinstance(forbidden, list) or
                not isinstance(descriptors, dict) or not allowed or
                any(kind not in TRANSFER_KINDS for kind in allowed + forbidden) or
                set(allowed) & set(forbidden) or set(descriptors) != set(allowed) or
                any(not isinstance(value, str) or not value.strip()
                    for value in descriptors.values())):
            raise P2Blocked("asset_transfer_policy_invalid", asset_id)
        if namespace == "reference_evidence" and not set(allowed) <= REFERENCE_ALLOWED:
            raise P2Blocked("reference_identity_transfer_forbidden", asset_id)
        if namespace == "committed_memory" and (
                entry.get("accepted") is not True or not entry.get("state_commit_id")):
            raise P2Blocked("uncommitted_memory_forbidden", asset_id)
        assets.append({
            "asset_id": asset_id, "namespace": namespace, "media_type": media_type,
            "path": str(path), "sha256": _file_hash(path), "source": source,
            "source_interval": interval, "allowed_transfer": sorted(set(allowed)),
            "forbidden_transfer": sorted(set(forbidden)),
            "transfer_descriptors": {kind: descriptors[kind] for kind in sorted(descriptors)},
            "state_commit_id": entry.get("state_commit_id") if namespace == "committed_memory"
            else None,
        })
    catalog = {"catalog_version": CATALOG_VERSION,
               "assets": sorted(assets, key=lambda asset: asset["asset_id"])}
    catalog["catalog_sha256"] = _hash(catalog)
    return catalog


def _validated_asset_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if catalog.get("catalog_version") != CATALOG_VERSION:
        raise P2Blocked("catalog_version_invalid")
    payload = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
    if _hash(payload) != catalog.get("catalog_sha256"):
        raise P2Blocked("catalog_hash_mismatch")
    assets = catalog.get("assets") or []
    if len({asset["asset_id"] for asset in assets}) != len(assets):
        raise P2Blocked("asset_id_invalid_or_duplicate")
    for asset in assets:
        if (asset.get("namespace") not in NAMESPACES or
                asset.get("media_type") not in MEDIA_TYPES or
                set(asset.get("allowed_transfer") or []) &
                set(asset.get("forbidden_transfer") or []) or
                set(asset.get("transfer_descriptors") or {}) !=
                set(asset.get("allowed_transfer") or [])):
            raise P2Blocked("asset_transfer_policy_invalid", asset.get("asset_id", ""))
        if (asset["namespace"] == "reference_evidence" and
                not set(asset["allowed_transfer"]) <= REFERENCE_ALLOWED):
            raise P2Blocked("reference_identity_transfer_forbidden", asset["asset_id"])
        path = Path(asset["path"])
        if not path.is_file() or _file_hash(path) != asset["sha256"]:
            raise P2Blocked("asset_hash_mismatch", asset["asset_id"])
    return {asset["asset_id"]: asset for asset in assets}


def compile_h3_context_preview(draft: dict[str, Any], catalog: dict[str, Any], *,
                               unit_id: str) -> dict[str, Any]:
    """Compile six official-style sections without inventing content or granting execution."""
    if (draft.get("contract_version") != P1_DRAFT_VERSION or
            draft.get("execution_authorized") is not False):
        raise P2Blocked("p1_draft_required")
    payload = {key: value for key, value in draft.items() if key != "draft_sha256"}
    if _hash(payload) != draft.get("draft_sha256"):
        raise P2Blocked("p1_draft_hash_mismatch")
    selected = [(section, unit) for section in draft.get("sections") or []
                for unit in section.get("generation_units") or []
                if unit.get("unit_id") == unit_id]
    if len(selected) != 1:
        raise P2Blocked("unit_not_unique", unit_id)
    section, unit = selected[0]
    asset_map = _validated_asset_map(catalog)
    ids = list(dict.fromkeys(unit.get("condition_keys") or []))
    if any(asset_id not in asset_map for asset_id in ids):
        raise P2Blocked("condition_asset_missing", unit_id)
    assets = [asset_map[asset_id] for asset_id in ids]
    definitions = []
    retention = []
    references = []
    label_counts = {"Picture": 0, "Video": 0, "Audio": 0}
    for asset in assets:
        label_kind = ("Picture" if asset["media_type"] == "image" else
                      "Video" if asset["media_type"] in {"video", "video_audio"} else
                      "Audio")
        label_counts[label_kind] += 1
        label = f"<{label_kind} {label_counts[label_kind]}>"
        descriptors = asset["transfer_descriptors"]
        transfer = "; ".join(f"{kind}: {descriptors[kind]}" for kind in sorted(descriptors))
        definitions.append(f"{label} [{asset['asset_id']}]: {transfer}")
        retention.append(f"{label}: permitted transfer only: {', '.join(asset['allowed_transfer'])}")
        references.append({"asset_id": asset["asset_id"], "request_label": label,
                           "type": asset["media_type"], "namespace": asset["namespace"],
                           "allowed_transfer": list(asset["allowed_transfer"]),
                           "forbidden_transfer": list(asset["forbidden_transfer"]),
                           "uri": asset["path"], "sha256": asset["sha256"]})
    continuity = "; ".join(f"{key}={value}" for key, value in sorted(
        unit.get("continuity_ids", {}).items())
        if section["continuity"].get(key) == "required")
    fields = {
        "subject_definitions": "\n".join(definitions) or "none specified",
        "summary": f"Section goal: {section['semantic_goal']}. Unit goal: {unit['primary_causal_goal']}.",
        "retention_analysis": "\n".join(retention) or "no references",
        "detailed_description": (
            f"Required phases in order: {', '.join(unit['semantic_phases'])}. "
            f"Required continuity identifiers: {continuity or 'none specified'}."
        ),
        "overall_soundscape": "not specified by contract",
        "non_diegetic_music": "not specified by contract",
    }
    prompt = "\n".join(f"{key}: {value}" for key, value in fields.items())
    result = {
        "context_version": CONTEXT_VERSION, "execution_authorized": False,
        "draft_sha256": draft["draft_sha256"],
        "catalog_sha256": catalog["catalog_sha256"],
        "unit_id": unit_id, "prompt_fields": fields, "prompt": prompt,
        "references": references,
        "generated_duration_s": unit["generated_duration_s"],
    }
    result["context_sha256"] = _hash(result)
    return result


def preview_h3_route(context: dict[str, Any], capabilities: dict[str, Any], *,
                     first_frame: str | None = None,
                     last_frame: str | None = None) -> dict[str, Any]:
    """Check a supplied capability fixture; never turn a preview into a request."""
    if context.get("context_version") != CONTEXT_VERSION or \
            context.get("execution_authorized") is not False:
        raise P2Blocked("context_preview_required")
    payload = {key: value for key, value in context.items() if key != "context_sha256"}
    if _hash(payload) != context.get("context_sha256"):
        raise P2Blocked("context_hash_mismatch")
    if not capabilities:
        raise P2Blocked("capability_manifest_required")
    route = route_h3_mode(references=context["references"], first_frame=first_frame,
                          last_frame=last_frame, capabilities=capabilities,
                          allow_explicit_split=True)
    return {"execution_authorized": False, "context_sha256": context["context_sha256"],
            "route": route, "first_frame": first_frame, "last_frame": last_frame}
