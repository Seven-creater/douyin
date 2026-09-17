"""H3 condition integrity and CPU-only capability planning.

A media response proves transport, not that a condition affected generation.
All new registries start unverified; fake observations never promote them.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from .generation_v9g import H3_CAPABILITIES, build_h3_request


P4_VERSION = "v9g_h3_condition_integrity_p4"
CAPABILITY_NAMES = (
    "t2va", "first_frame", "last_frame", "first_last_frame",
    "reference_image", "reference_video", "reference_audio", "ref2va_hybrid",
    "ref2va_hybrid_first", "ref2va_hybrid_last", "ref2va_hybrid_first_last",
)
CAPABILITY_STATES = {
    "unverified", "transport_only", "effective", "unsupported", "failed",
}
ASSET_ROLES = {
    "identity_reference", "scene_reference", "motion_reference",
    "interaction_reference", "timing_reference", "audio_reference",
    "first_frame", "last_frame", "continuity_reference",
}
MEDIA_KINDS = {"image", "video", "audio"}
LABEL_RE = re.compile(r"<(Picture|Video|Audio) ([1-9][0-9]*)>")


class P4Blocked(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)
        self.reason_code = reason_code


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_capability_registry(backend_id: str) -> dict[str, Any]:
    if not backend_id:
        raise P4Blocked("backend_id_missing")
    return {"schema_version": P4_VERSION, "backend_id": backend_id,
            "status": "unverified",
            "capabilities": {name: "unverified" for name in CAPABILITY_NAMES},
            "evidence": {}}


def validate_capability_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != P4_VERSION or not registry.get("backend_id") or \
            registry.get("status") not in CAPABILITY_STATES or \
            set(registry.get("capabilities") or {}) != set(CAPABILITY_NAMES) or \
            any(value not in CAPABILITY_STATES
                for value in registry["capabilities"].values()):
        raise P4Blocked("capability_registry_invalid")


def register_condition_asset(*, asset_id: str, kind: str, path: Path,
                             source: dict[str, Any], allowed_roles: list[str],
                             forbidden_roles: list[str] | None = None,
                             request_uri: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    forbidden = forbidden_roles or []
    if not asset_id or kind not in MEDIA_KINDS or not path.is_file() or \
            path.stat().st_size == 0:
        raise P4Blocked("condition_asset_invalid", asset_id)
    if (not isinstance(source, dict) or not source.get("type") or
            not allowed_roles or not set(allowed_roles + forbidden) <= ASSET_ROLES or
            set(allowed_roles) & set(forbidden)):
        raise P4Blocked("condition_asset_policy_invalid", asset_id)
    if source["type"] == "reference_evidence" and \
            "identity_reference" in allowed_roles:
        raise P4Blocked("reference_identity_forbidden", asset_id)
    if source["type"] == "reference_evidence":
        interval = source.get("source_interval")
        if (not source.get("reference_video_sha256") or
                not isinstance(interval, list) or len(interval) != 2 or
                not all(isinstance(value, (int, float)) for value in interval) or
                interval[0] < 0 or interval[0] >= interval[1]):
            raise P4Blocked("reference_source_provenance_missing", asset_id)
    return {"asset_id": asset_id, "kind": kind, "path": str(path),
            "request_uri": request_uri or path.as_uri(), "sha256": _file_hash(path),
            "source": deepcopy(source), "allowed_roles": sorted(set(allowed_roles)),
            "forbidden_roles": sorted(set(forbidden))}


def _condition_capabilities(bindings: list[dict[str, Any]]) -> tuple[str, list[str]]:
    references = [row for row in bindings if row["semantic_role"] not in
                  {"first_frame", "last_frame"}]
    first = any(row["semantic_role"] == "first_frame" for row in bindings)
    last = any(row["semantic_role"] == "last_frame" for row in bindings)
    if references:
        needed = {"reference_image" if row["kind"] == "image" else
                  "reference_video" if row["kind"] == "video" else
                  "reference_audio" for row in references}
        if first or last:
            needed.add("ref2va_hybrid")
            needed.add("ref2va_hybrid_first_last" if first and last else
                       "ref2va_hybrid_first" if first else "ref2va_hybrid_last")
        return "ref2va", sorted(needed)
    if first or last:
        return "fl2va", ["first_last_frame" if first and last else
                          "first_frame" if first else "last_frame"]
    return "t2va", ["t2va"]


def build_h3_condition_request(*, request_id: str, unit_id: str, prompt: str,
                               duration_s: float, assets: list[dict[str, Any]],
                               bindings: list[dict[str, Any]],
                               backend: dict[str, Any],
                               contract_hash: str | None = None,
                               context_sha256: str | None = None) -> dict[str, Any]:
    """Build a non-authorized H3 request and exact condition-to-asset manifest."""
    asset_map = {row["asset_id"]: row for row in assets}
    if len(asset_map) != len(assets) or not request_id or not unit_id or not prompt:
        raise P4Blocked("request_plan_invalid")
    references = []
    keyframes: dict[str, str] = {}
    rows = []
    keyframe_seen = False
    for position, binding in enumerate(bindings):
        asset_id = binding.get("asset_id")
        if asset_id not in asset_map:
            raise P4Blocked("condition_asset_missing", str(asset_id))
        asset = asset_map[asset_id]
        role = binding.get("semantic_role")
        if role not in asset["allowed_roles"] or role in asset["forbidden_roles"]:
            raise P4Blocked("condition_role_forbidden", str(asset_id))
        label = binding.get("request_label")
        if role in {"first_frame", "last_frame"}:
            keyframe_seen = True
            if asset["kind"] != "image" or role in keyframes or label != (
                    "First Frame" if role == "first_frame" else "Last Frame"):
                raise P4Blocked("keyframe_binding_invalid", str(label))
            keyframes[role] = asset["request_uri"]
        else:
            if keyframe_seen:
                raise P4Blocked("condition_order_invalid")
            if asset["kind"] not in MEDIA_KINDS or not isinstance(label, str):
                raise P4Blocked("reference_binding_invalid", str(asset_id))
            references.append({"type": asset["kind"], "uri": asset["request_uri"]})
        rows.append({"condition_id": binding.get("condition_id") or f"cond_{position+1:02d}",
                     "position": position, "asset_id": asset_id,
                     "asset_sha256": asset["sha256"],
                     "asset_record_sha256": _hash(asset), "request_label": label,
                     "semantic_role": role, "required": binding.get("required") is True,
                     "kind": asset["kind"], "request_uri": asset["request_uri"]})
    mode, needed = _condition_capabilities(rows)
    route_caps = {name: True for name in H3_CAPABILITIES}
    request = build_h3_request(
        prompt=prompt, duration_s=duration_s, references=references,
        first_frame=keyframes.get("first_frame"),
        last_frame=keyframes.get("last_frame"), capabilities=route_caps)
    if request["task"] != mode or backend.get("model_variant") != request["_v9g_route"]["variant"]:
        raise P4Blocked("backend_variant_mismatch")
    manifest = {"schema_version": P4_VERSION, "request_id": request_id,
                "unit_id": unit_id, "mode": mode,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "request_sha256": _hash(request), "conditions": rows,
                "expected_capabilities": needed, "backend": deepcopy(backend),
                "contract_hash": contract_hash, "context_sha256": context_sha256,
                "execution_authorized": False}
    manifest["manifest_sha256"] = _hash(manifest)
    return {"request": request, "manifest": manifest}


def plan_h3_request_from_context(context: dict[str, Any], *,
                                 assets: list[dict[str, Any]],
                                 role_assignments: dict[str, str],
                                 backend: dict[str, Any],
                                 first_asset_id: str | None = None,
                                 last_asset_id: str | None = None) -> dict[str, Any]:
    """Bridge P2 preview to P4 request without inferring missing roles or assets."""
    if (context.get("context_version") != "v9g_context_preview_p2" or
            context.get("execution_authorized") is not False or
            _hash({key: value for key, value in context.items()
                   if key != "context_sha256"}) != context.get("context_sha256")):
        raise P4Blocked("p2_context_invalid")
    asset_map = {row["asset_id"]: row for row in assets}
    bindings = []
    namespace_source = {"reference_evidence": "reference_evidence",
                        "target_assets": "target_asset",
                        "committed_memory": "committed_memory"}
    role_transfer = {"identity_reference": "identity",
                     "scene_reference": "scene", "motion_reference": "motion",
                     "interaction_reference": "motion", "timing_reference": "rhythm",
                     "audio_reference": "audio", "continuity_reference": "continuity"}
    for row in context.get("references") or []:
        asset_id = row["asset_id"]
        asset = asset_map.get(asset_id)
        if not asset or asset["sha256"] != row["sha256"] or \
                asset["kind"] != row["type"] or asset_id not in role_assignments:
            raise P4Blocked("p2_condition_binding_missing", asset_id)
        role = role_assignments[asset_id]
        transfer = role_transfer.get(role)
        if (asset.get("source", {}).get("type") !=
                namespace_source.get(row.get("namespace")) or
                transfer not in row.get("allowed_transfer", []) or
                transfer in row.get("forbidden_transfer", [])):
            raise P4Blocked("p2_transfer_policy_conflict", asset_id)
        bindings.append({"asset_id": asset_id, "request_label": row["request_label"].strip("<>"),
                         "semantic_role": role, "required": True})
    for asset_id, label, role in ((first_asset_id, "First Frame", "first_frame"),
                                  (last_asset_id, "Last Frame", "last_frame")):
        if asset_id is not None:
            if asset_id not in asset_map or role_assignments.get(asset_id) != role:
                raise P4Blocked("p2_keyframe_binding_missing", asset_id)
            bindings.append({"asset_id": asset_id, "request_label": label,
                             "semantic_role": role, "required": True})
    selected_ids = {row["asset_id"] for row in bindings}
    if set(role_assignments) != selected_ids:
        raise P4Blocked("unbound_condition_role")
    return build_h3_condition_request(
        request_id=f"context_{context['context_sha256'][:12]}",
        unit_id=context["unit_id"], prompt=context["prompt"],
        duration_s=float(context["generated_duration_s"]),
        assets=[asset_map[asset_id] for asset_id in selected_ids],
        bindings=bindings, backend=backend,
        contract_hash=context["draft_sha256"],
        context_sha256=context["context_sha256"])


def build_h3_capability_smoke_plan(*, fixture_dir: Path | None = None,
                                   backend_id: str = "local_h3_sglang") -> dict[str, Any]:
    """Plan S0–S7 without starting a server or inferring any capability PASS."""
    registry = new_capability_registry(backend_id)
    cases = [
        ("S0", "fl2va", [], ["transport_only"]),
        ("S1", "fl2va", [("first", "First Frame", "first_frame")],
         ["first_frame_similarity", "opening_composition"]),
        ("S2", "fl2va", [("last", "Last Frame", "last_frame")],
         ["last_frame_similarity", "closing_composition"]),
        ("S3", "fl2va", [("first", "First Frame", "first_frame"),
                         ("last", "Last Frame", "last_frame")],
         ["first_frame_similarity", "last_frame_similarity"]),
        ("S4", "ref2va", [("identity", "Picture 1", "identity_reference")],
         ["identity_attributes_visible"]),
        ("S5", "ref2va", [("motion", "Video 1", "motion_reference")],
         ["motion_direction_matches"]),
        ("S6", "ref2va", [("audio", "Audio 1", "audio_reference")],
         ["reference_audio_effect"]),
        ("S7", "ref2va", [("identity", "Picture 1", "identity_reference"),
                          ("first", "First Frame", "first_frame")],
         ["identity_attributes_visible", "first_frame_similarity"]),
    ]
    requirements = {
        "first": "first.png", "last": "last.png", "identity": "reference.png",
        "motion": "reference.mp4", "audio": "reference.wav",
    }
    assets: dict[str, dict[str, Any]] = {}
    if fixture_dir is not None:
        directory = Path(fixture_dir).resolve()
        missing = [name for name in requirements.values()
                   if not (directory / name).is_file()]
        if missing:
            raise P4Blocked("smoke_fixtures_missing", ",".join(missing))
        roles = {
            "first": ["first_frame"], "last": ["last_frame"],
            "identity": ["identity_reference"],
            "motion": ["motion_reference"], "audio": ["audio_reference"],
        }
        for key, filename in requirements.items():
            assets[key] = register_condition_asset(
                asset_id=f"smoke_{key}",
                kind="image" if key in {"first", "last", "identity"} else key.replace(
                    "motion", "video"),
                path=directory / filename, source={"type": "smoke_fixture"},
                allowed_roles=roles[key])
    planned = []
    for case_id, variant, conditions, checks in cases:
        row: dict[str, Any] = {
            "case_id": case_id, "model_variant": variant,
            "condition_specs": [{"fixture": key, "request_label": label,
                                 "semantic_role": role}
                                for key, label, role in conditions],
            "effect_checks": checks,
            "execution_status": "planned_only",
        }
        if assets:
            bindings = [{"condition_id": f"{case_id}_{index}",
                         "asset_id": assets[key]["asset_id"],
                         "request_label": label, "semantic_role": role, "required": True}
                        for index, (key, label, role) in enumerate(conditions, 1)]
            definitions = "\n".join(
                f"<{label}> [{assets[key]['asset_id']}] {role}"
                for key, label, role in conditions if role not in
                {"first_frame", "last_frame"})
            prompt = (f"subject_definitions: {definitions or 'none'}\n"
                      f"summary: capability smoke {case_id}; use only supplied conditions.\n"
                      "detailed_description: Show a clear, observable result.")
            plan = build_h3_condition_request(
                request_id=f"smoke_{case_id}", unit_id=f"SMOKE.{case_id}",
                prompt=prompt, duration_s=4.0,
                assets=[assets[key] for key, _, _ in conditions], bindings=bindings,
                backend={"backend_id": backend_id, "type": "sglang_h3",
                         "model_variant": variant, "endpoint": None})
            row.update(plan)
            row["assets"] = [assets[key] for key, _, _ in conditions]
        planned.append(row)
    return {"schema_version": P4_VERSION, "plan_only": True,
            "backend_registry": registry, "fixture_requirements": requirements,
            "fixtures_ready": bool(assets), "cases": planned}


def _probe_streams(path: Path, ffprobe_bin: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run([
            ffprobe_bin, "-v", "error", "-show_streams", "-of", "json", str(path),
        ], capture_output=True, text=True, check=False)
    except OSError as exc:
        raise P4Blocked("media_probe_unavailable", str(path)) from exc
    if result.returncode:
        raise P4Blocked("media_probe_failed", str(path))
    try:
        return json.loads(result.stdout).get("streams") or []
    except ValueError as exc:
        raise P4Blocked("media_probe_invalid_json", str(path)) from exc


def _check_asset_media(asset: dict[str, Any], ffprobe_bin: str) -> None:
    path = Path(asset["path"])
    if not path.is_file() or path.stat().st_size == 0 or \
            _file_hash(path) != asset["sha256"]:
        raise P4Blocked("asset_sha_mismatch", asset["asset_id"])
    streams = _probe_streams(path, ffprobe_bin)
    kinds = {row.get("codec_type") for row in streams}
    kind = asset["kind"]
    if (kind == "image" and ("video" not in kinds or
                             path.suffix.lower() not in
                             {".png", ".jpg", ".jpeg", ".webp"} or
                             not any(row.get("codec_name") in
                                     {"png", "mjpeg", "webp"} for row in streams)) or
            kind == "video" and ("video" not in kinds or path.suffix.lower() in
                                 {".png", ".jpg", ".jpeg", ".webp"}) or
            kind == "audio" and ("audio" not in kinds or "video" in kinds)):
        raise P4Blocked("asset_media_type_mismatch", asset["asset_id"])


def validate_h3_condition_integrity(request: dict[str, Any],
                                    manifest: dict[str, Any],
                                    assets: list[dict[str, Any]], *,
                                    accepted_commits: dict[str, Any] | None = None,
                                    ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Fail closed before sending any request; validate labels, order, bytes, roles."""
    if manifest.get("schema_version") != P4_VERSION or \
            _hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != \
            manifest.get("manifest_sha256") or _hash(request) != manifest.get("request_sha256"):
        raise P4Blocked("request_manifest_hash_mismatch")
    prompt = str(request.get("prompt") or "")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != manifest["prompt_sha256"]:
        raise P4Blocked("prompt_hash_mismatch")
    asset_map = {row["asset_id"]: row for row in assets}
    rows = manifest.get("conditions") or []
    conditions = request.get("conditions") or []
    if len(asset_map) != len(assets) or len(rows) != len(conditions) or \
            request.get("task") != manifest.get("mode"):
        raise P4Blocked("request_condition_count_or_mode_mismatch")
    if (request.get("model") != "MiniMaxAI/MiniMax-H3" or
            request.get("_v9g_route", {}).get("variant") !=
            manifest.get("backend", {}).get("model_variant")):
        raise P4Blocked("backend_variant_mismatch")
    expected_labels = set()
    modality_counts = {"Picture": 0, "Video": 0, "Audio": 0}
    for index, (binding, condition) in enumerate(zip(rows, conditions)):
        asset = asset_map.get(binding.get("asset_id"))
        if not asset or binding.get("position") != index:
            raise P4Blocked("condition_order_or_asset_mismatch")
        _check_asset_media(asset, ffprobe_bin)
        if (_hash(asset) != binding.get("asset_record_sha256") or
                asset["sha256"] != binding.get("asset_sha256") or
                asset["request_uri"] != binding.get("request_uri") or
                condition.get("uri") != asset["request_uri"] or
                condition.get("type") != asset["kind"]):
            raise P4Blocked("condition_order_or_asset_mismatch", asset["asset_id"])
        role = binding.get("semantic_role")
        if role not in asset.get("allowed_roles", []) or \
                role in asset.get("forbidden_roles", []):
            raise P4Blocked("condition_role_forbidden", asset["asset_id"])
        if asset.get("source", {}).get("type") == "reference_evidence" and \
                role == "identity_reference":
            raise P4Blocked("reference_identity_forbidden", asset["asset_id"])
        if asset.get("source", {}).get("type") == "committed_memory":
            source = asset["source"]
            commit = (accepted_commits or {}).get(source.get("state_commit_id"))
            if (not commit or commit.get("status") != "accepted" or
                    commit.get("unit_id") != source.get("unit_id") or
                    commit.get("snippet_id") != source.get("snippet_id") or
                    commit.get("endpoint_asset_sha256") != asset["sha256"] or
                    source.get("endpoint_kind") != "accepted_snippet_endpoint" or
                    source.get("accepted_snippet_endpoint_s") !=
                    commit.get("accepted_snippet_endpoint_s")):
                raise P4Blocked("committed_memory_not_accepted", asset["asset_id"])
        if role in {"first_frame", "last_frame"}:
            frame_index = 0 if role == "first_frame" else -1
            if condition.get("role") != "keyframe" or \
                    condition.get("frame_index") != frame_index or \
                    binding.get("request_label") != (
                        "First Frame" if frame_index == 0 else "Last Frame"):
                raise P4Blocked("keyframe_condition_mismatch")
        else:
            if condition.get("role") != "reference":
                raise P4Blocked("reference_condition_missing")
            label_type = {"image": "Picture", "video": "Video",
                          "audio": "Audio"}[asset["kind"]]
            modality_counts[label_type] += 1
            expected = f"{label_type} {modality_counts[label_type]}"
            if binding.get("request_label") != expected or \
                    f"<{expected}> [{asset['asset_id']}]" not in prompt:
                raise P4Blocked("prompt_request_label_mismatch", expected)
            expected_labels.add((label_type, modality_counts[label_type]))
    actual_labels = {(kind, int(number)) for kind, number in LABEL_RE.findall(prompt)}
    if actual_labels != expected_labels:
        raise P4Blocked("prompt_request_label_mismatch")
    mode, needed = _condition_capabilities(rows)
    if mode != manifest["mode"] or needed != manifest.get("expected_capabilities"):
        raise P4Blocked("capability_manifest_mismatch")
    return {"passed": True, "request_id": manifest["request_id"],
            "condition_count": len(rows), "expected_capabilities": needed}


def require_effective_capabilities(registry: dict[str, Any],
                                   manifest: dict[str, Any], *,
                                   p0_frozen: bool) -> None:
    if not p0_frozen:
        raise P4Blocked("p0_programs_not_frozen")
    validate_capability_registry(registry)
    if registry.get("backend_id") != manifest.get("backend", {}).get("backend_id"):
        raise P4Blocked("backend_registry_mismatch")
    if registry["status"] != "effective":
        raise P4Blocked("backend_capability_not_verified")
    for name in manifest.get("expected_capabilities") or []:
        if registry.get("capabilities", {}).get(name) != "effective":
            raise P4Blocked("unsupported_or_unverified_capability", name)
        evidence = registry.get("evidence", {}).get(name) or {}
        if (evidence.get("source") != "real" or
                evidence.get("backend_transport_ok") is not True or
                evidence.get("conditioning_effective") is not True or
                not evidence.get("request_sha256")):
            raise P4Blocked("real_capability_evidence_missing", name)


def validate_formal_h3_job(request: dict[str, Any], manifest: dict[str, Any],
                           assets: list[dict[str, Any]],
                           registry: dict[str, Any], *, p0_frozen: bool,
                           accepted_commits: dict[str, Any] | None = None,
                           expected_endpoint: str | None = None,
                           expected_contract_hash: str | None = None,
                           expected_unit_id: str | None = None) \
        -> dict[str, Any]:
    if not p0_frozen:
        raise P4Blocked("p0_programs_not_frozen")
    if not manifest.get("backend", {}).get("endpoint") or (
            expected_endpoint is not None and
            manifest["backend"]["endpoint"].rstrip("/") !=
            expected_endpoint.rstrip("/")):
        raise P4Blocked("backend_endpoint_unverified")
    if (not manifest.get("contract_hash") or
            manifest.get("contract_hash") != expected_contract_hash or
            not manifest.get("context_sha256") or
            manifest.get("unit_id") != expected_unit_id):
        raise P4Blocked("formal_contract_or_unit_binding_missing")
    integrity = validate_h3_condition_integrity(
        request, manifest, assets, accepted_commits=accepted_commits)
    require_effective_capabilities(registry, manifest, p0_frozen=p0_frozen)
    return {"passed": True, "condition_integrity": integrity,
            "capabilities": list(manifest["expected_capabilities"])}


def verify_frozen_program_snapshot(v9_dir: Path, contract: dict[str, Any]) -> bool:
    """Recheck the signed three-Program snapshot immediately before generation."""
    root = Path(v9_dir)
    try:
        acceptance = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
        frozen = json.loads((root / "frozen_program_hashes.json").read_text(
            encoding="utf-8"))
        expected = frozen["program_sha256"]
        if acceptance.get("passed") is not True or \
                acceptance.get("human_passed") is not True or \
                expected != contract.get("v9_program_hashes"):
            return False
        return all(_file_hash(root / name) == digest
                   for name, digest in expected.items())
    except (OSError, ValueError, KeyError):
        return False


def verify_preflight_generation_authorization(
        output_dir: Path, contract: dict[str, Any],
        selected_jobs: list[dict[str, Any]], *,
        expected_endpoints: dict[str, str] | None = None) -> dict[str, Any]:
    """Recheck on-disk authorization immediately before any real H3 request."""
    root = Path(output_dir)
    try:
        authorization = json.loads((root / "formal_generation_authorization.json")
                                   .read_text(encoding="utf-8"))
        saved_contract = json.loads((root / "generation_contract.json").read_text(
            encoding="utf-8"))
        registry_path = root / "capability" / "backend_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        saved_jobs = json.loads((root / "generation_jobs.json").read_text(
            encoding="utf-8"))["jobs"]
    except (OSError, ValueError, KeyError) as exc:
        raise P4Blocked("formal_authorization_artifact_missing") from exc
    if (authorization.get("passed") is not True or
            authorization.get("simulation_only") is not False or
            authorization.get("contract_hash") != contract.get("contract_hash") or
            saved_contract != contract or
            hashlib.sha256(registry_path.read_bytes()).hexdigest() !=
            authorization.get("registry_sha256") or
            not verify_frozen_program_snapshot(
                Path(authorization.get("v9_output") or ""), contract)):
        raise P4Blocked("formal_authorization_stale")
    expected = authorization.get("job_request_sha256") or {}
    saved_job_map = {job["unit_id"]: job for job in saved_jobs}
    selected_job_map = {job["unit_id"]: job for job in selected_jobs}
    if (not expected or len(selected_job_map) != len(selected_jobs) or
            set(expected) != set(selected_job_map)):
        raise P4Blocked("formal_request_set_changed")
    for unit_id, request_hash in expected.items():
        if (unit_id not in saved_job_map or
                _hash(saved_job_map[unit_id]["request"]) != request_hash or
                _hash(selected_job_map[unit_id]["request"]) != request_hash):
            raise P4Blocked("formal_request_changed", unit_id)
        plan_path = root / "generation_conditions" / f"{unit_id}.json"
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise P4Blocked("formal_condition_manifest_missing", unit_id) from exc
        validate_formal_h3_job(selected_job_map[unit_id]["request"], plan["manifest"],
                               plan["assets"], registry, p0_frozen=True,
                               expected_endpoint=(expected_endpoints or {}).get(
                                   selected_job_map[unit_id]["request"].get("task")),
                               expected_contract_hash=contract["contract_hash"],
                               expected_unit_id=unit_id)
    return {"passed": True, "unit_count": len(selected_jobs),
            "contract_hash": contract["contract_hash"]}


def verify_formal_generation_authorization(output_dir: Path) -> dict[str, Any]:
    """Reject simulated or stale results before formal acceptance."""
    root = Path(output_dir)
    try:
        contract = json.loads((root / "generation_contract.json").read_text(
            encoding="utf-8"))
        jobs = json.loads((root / "generation_jobs.json").read_text(
            encoding="utf-8"))["jobs"]
        results = json.loads((root / "generation_results.json").read_text(
            encoding="utf-8"))["results"]
        authorization = json.loads((root / "formal_generation_authorization.json")
                                   .read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        raise P4Blocked("formal_authorization_artifact_missing") from exc
    result_map = {row["unit_id"]: row for row in results}
    expected = authorization.get("job_request_sha256") or {}
    if (not expected or len(result_map) != len(results) or
            set(expected) != set(result_map) or
            any(row.get("simulation_only") is not False for row in results)):
        raise P4Blocked("formal_generation_results_missing_or_simulated")
    selected_jobs = [job for job in jobs if job["unit_id"] in expected]
    return verify_preflight_generation_authorization(root, contract, selected_jobs)


def validate_generated_media_transport(path: Path, *, duration_s: float,
                                       short_edge: int, fps: float = 24.0,
                                       audio_required: bool = True,
                                       ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return {"backend_transport_ok": False, "reason_code": "media_missing_or_empty"}
    try:
        result = subprocess.run([
            ffprobe_bin, "-v", "error", "-show_streams", "-show_format", "-of", "json",
            str(path),
        ], capture_output=True, text=True, check=False)
    except OSError:
        return {"backend_transport_ok": False, "reason_code": "ffprobe_unavailable"}
    if result.returncode:
        return {"backend_transport_ok": False, "reason_code": "ffprobe_failed"}
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return {"backend_transport_ok": False, "reason_code": "ffprobe_invalid_json"}
    video = next((s for s in data.get("streams") or [] if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams") or [] if s.get("codec_type") == "audio"), None)
    if not video:
        return {"backend_transport_ok": False, "reason_code": "video_stream_missing"}
    try:
        actual_duration = float(data["format"]["duration"])
        numerator, denominator = str(video["avg_frame_rate"]).split("/", 1)
        actual_fps = float(numerator) / float(denominator)
        actual_edge = min(int(video["width"]), int(video["height"]))
        audio_hz = int(audio.get("sample_rate") or 0) if audio else None
        audio_channels = int(audio.get("channels") or 0) if audio else None
    except (KeyError, ValueError, ZeroDivisionError):
        return {"backend_transport_ok": False, "reason_code": "media_metadata_invalid"}
    actual = {"duration_s": actual_duration, "fps": actual_fps,
              "width": int(video["width"]), "height": int(video["height"]),
              "video_codec": video.get("codec_name"),
              "audio_present": bool(audio),
              "audio_codec": audio.get("codec_name") if audio else None,
              "audio_sample_rate": audio_hz,
              "audio_channels": audio_channels,
              "sha256": _file_hash(path)}
    passed = (abs(actual_duration - duration_s) <= max(0.5, duration_s * 0.1) and
              abs(actual_fps - fps) <= 0.25 and actual_edge == short_edge and
              video.get("codec_name") == "h264" and
              (not audio_required or (bool(audio) and
                                      actual["audio_codec"] == "aac" and
                                      actual["audio_sample_rate"] == 32000 and
                                      actual["audio_channels"] == 2)))
    return {"backend_transport_ok": passed,
            "reason_code": None if passed else "media_spec_mismatch", "actual": actual}


def evaluate_condition_effect(manifest: dict[str, Any], *,
                              received_conditions: list[dict[str, Any]] | None,
                              checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep transport receipt and observable effect separate, including fake evidence."""
    intended = [(row["request_uri"], row["kind"]) for row in manifest["conditions"]]
    if received_conditions is not None:
        received = [(row.get("uri"), row.get("type")) for row in received_conditions]
        if received != intended:
            return {"status": "ineffective", "conditioning_effective": False,
                    "reason_code": "backend_condition_receipt_mismatch", "checks": checks}
    if not checks:
        return {"status": "unknown", "conditioning_effective": False,
                "reason_code": "effect_checks_missing", "checks": []}
    if any(row.get("passed") is False for row in checks):
        status = "ineffective"
    elif all(row.get("passed") is True and row.get("evidence") for row in checks):
        status = "effective"
    else:
        status = "unknown"
    return {"status": status, "conditioning_effective": status == "effective",
            "checks": deepcopy(checks), "evidence_scope": (
                "fake_only" if all(row.get("source") == "fake" for row in checks) else
                "real_or_mixed")}


class FakeH3ConditionBackend:
    """A transport stub that copies a fixture video and exposes received conditions."""

    def __init__(self, fixture_video: Path, fault: str | None = None) -> None:
        self.fixture_video = Path(fixture_video)
        self.fault = fault

    def generate(self, request: dict[str, Any], output_path: Path) -> dict[str, Any]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.fixture_video, output_path)
        received = deepcopy(request.get("conditions") or [])
        if self.fault == "drop_video":
            received = [row for row in received if row.get("type") != "video"]
        elif self.fault == "reorder_image":
            indexes = [index for index, row in enumerate(received)
                       if row.get("type") == "image" and row.get("role") == "reference"]
            if len(indexes) >= 2:
                received[indexes[0]], received[indexes[1]] = (
                    received[indexes[1]], received[indexes[0]])
        return {"output_path": str(output_path), "received_conditions": received,
                "fake_backend": True}


def run_fake_h3_smoke(plan: dict[str, Any], backend: FakeH3ConditionBackend,
                      output_path: Path, *, short_edge: int,
                      ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    request, manifest = plan["request"], plan["manifest"]
    integrity = validate_h3_condition_integrity(
        request, manifest, plan["assets"], ffprobe_bin=ffprobe_bin)
    response = backend.generate(request, output_path)
    transport = validate_generated_media_transport(
        Path(response["output_path"]), duration_s=float(request["seconds"]),
        short_edge=short_edge, ffprobe_bin=ffprobe_bin)
    effect = evaluate_condition_effect(
        manifest, received_conditions=response["received_conditions"],
        checks=[{"name": "fixture_echo", "passed": True,
                 "evidence": "fake receipt only", "source": "fake"}])
    return {"integrity": integrity, "transport": transport, "effect": effect,
            "registry_updated": False, "real_capability_verified": False}
