"""V9-G reference-conditioned generation contracts and execution loop.

The canonical contract and all research decisions in this module are
first-party.  The implementation deliberately wraps pinned CineCrew/NEWTON
fragments at JSON boundaries; no upstream internal class is a public API.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from third_party.cinecrew import ArtifactTrace
from third_party.newton import NewtonTraceAdapter


CONTRACT_VERSION = "v9g_1"
FILMDSL_ADAPTER_VERSION = "v9g_filmdsl_adapter_v1"
CAPABILITY_VERSION = "v9g_h3_capabilities_v1"
STATE_VERSION = "v9g_state_v1"
H3_CAPABILITIES = (
    "t2va", "fl2va_first", "fl2va_last", "fl2va_first_last",
    "ref2va_image", "ref2va_video", "ref2va_audio",
    "ref2va_hybrid_first", "ref2va_hybrid_last",
    "ref2va_hybrid_first_last",
)
COMPOSITION_MODES = {
    "continuous_clip", "micro_montage", "event_compression_montage",
    "evidence_montage", "dialogue_compression", "reaction_result_pair",
}
NEUTRAL_OBSERVATION_PROMPT = """Observe only what is visibly or audibly present.
Use temporary local subject IDs. Report subjects, initiator, affected subject,
before/after state, scene, clothing/body structure, and anything unreliable.
Do not infer the requested story, desired behavior, or acceptance result.
Return one JSON object and no trailing prose."""


class V9GBlocked(RuntimeError):
    """Stable blocked outcome rather than a silent policy downgrade."""

    def __init__(self, stage: str, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail or reason_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def _parse_one_object(raw: Any, *, stage: str) -> dict[str, Any]:
    text = str(getattr(raw, "text", raw) or "").strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise V9GBlocked(stage, "model_response_invalid_json", str(exc)) from exc
    if not isinstance(value, dict):
        raise V9GBlocked(stage, "model_response_not_one_object")
    return value


def validate_license_gates(code_gate: Path, model_gate: Path) -> dict[str, Any]:
    """Require separate human-reviewed code and model gates."""
    required_common = {"gate_version", "checked_upstream_sha", "reviewed_by",
                       "checked_at", "passed"}
    results: dict[str, Any] = {}
    for kind, path in (("code", Path(code_gate)), ("model", Path(model_gate))):
        if not path.is_file():
            raise V9GBlocked("license", f"{kind}_license_gate_missing", str(path))
        gate = _read_json(path)
        missing = sorted(required_common - set(gate))
        if kind == "model":
            missing += sorted({
                "model_card_revision", "license_revision", "deployment_context",
                "intended_output_use", "distribution_plan",
            } - set(gate))
        if missing:
            raise V9GBlocked("license", f"{kind}_license_gate_incomplete",
                             ",".join(missing))
        if gate.get("passed") is not True or not str(gate.get("reviewed_by") or "").strip() \
                or not gate.get("checked_at"):
            raise V9GBlocked("license", f"{kind}_license_gate_not_passed")
        results[kind] = {
            "path": str(path.resolve()), "sha256": _file_hash(path),
            "checked_upstream_sha": gate["checked_upstream_sha"],
            "reviewed_by": gate["reviewed_by"], "checked_at": gate["checked_at"],
        }
    return {"passed": True, "gates": results}


def _verified_v9_inputs(v9_dir: Path) -> tuple[dict[str, Any], ...]:
    v9_dir = Path(v9_dir)
    acceptance_path = v9_dir / "acceptance.json"
    frozen_path = v9_dir / "frozen_program_hashes.json"
    if not acceptance_path.is_file() or not frozen_path.is_file():
        raise V9GBlocked("contract", "v9_human_acceptance_or_freeze_missing")
    acceptance = _read_json(acceptance_path)
    if acceptance.get("passed") is not True or acceptance.get("human_passed") is not True:
        raise V9GBlocked("contract", "v9_human_acceptance_not_passed")
    frozen = _read_json(frozen_path)
    names = (
        "reference_content_program.json", "reference_edit_program.json",
        "material_requirements.json",
    )
    expected = frozen.get("program_sha256") or {}
    values = []
    for name in names:
        path = v9_dir / name
        if not path.is_file() or expected.get(name) != _file_hash(path):
            raise V9GBlocked("contract", "frozen_v9_program_hash_mismatch", name)
        values.append(_read_json(path))
    return (*values, frozen)


def _partition_phases(phases: list[str], count: int) -> list[list[str]]:
    count = max(1, min(count, len(phases)))
    return [
        phases[math.floor(index * len(phases) / count):
               math.floor((index + 1) * len(phases) / count)]
        for index in range(count)
    ]


def _compile_units(section: dict[str, Any]) -> list[dict[str, Any]]:
    phases = list(section["required_phases"])
    low, high = section["unit_strategy"]["generation_unit_count_range"]
    desired = max(int(low), min(int(high), len(phases) or 1))
    groups = _partition_phases(phases or ["complete_section_meaning"], desired)
    section_budget = float(section["duration_budget_s"])
    duration = min(15.0, max(4.0, section_budget / len(groups)))
    return [{
        "unit_id": f"{section['section_id']}.G{index + 1}",
        "section_id": section["section_id"],
        "semantic_phases": group,
        "target_duration_s": round(duration, 3),
        "ordering_index": index,
    } for index, group in enumerate(groups)]


def compile_generation_contract(v9_dir: Path, target_setting_path: Path,
                                output_path: Path | None = None) -> dict[str, Any]:
    """Compile the only canonical V9-G business object from frozen V9 outputs."""
    content, edit, requirements, frozen = _verified_v9_inputs(v9_dir)
    target_setting_path = Path(target_setting_path)
    if not target_setting_path.is_file():
        raise V9GBlocked("contract", "target_setting_missing", str(target_setting_path))
    target_setting = _read_json(target_setting_path)
    if not isinstance(target_setting, dict) or not target_setting.get("assets"):
        raise V9GBlocked("contract", "target_setting_assets_missing")
    sections = []
    for row in requirements.get("requirements") or []:
        semantic = row.get("semantic_requirement") or {}
        presentation = row.get("presentation_requirement") or {}
        continuity = row.get("continuity_requirement") or {}
        evidence = row.get("evidence_requirement") or {}
        section_id = str(row.get("section_id") or "").strip()
        phases = [str(value) for value in (
            semantic.get("required_semantic_phases") or
            evidence.get("minimum_sufficient_evidence_set") or []) if str(value)]
        section = {
            "section_id": section_id,
            "semantic_goal": str(semantic.get("meaning_to_prove") or "").strip(),
            "composition_mode": presentation.get("composition_mode"),
            "duration_budget_s": float(presentation.get("target_duration_s") or 0),
            "required_phases": phases,
            "continuity": {
                "same_subject": bool(continuity.get("same_subject_required")),
                "same_opponent": bool(continuity.get("same_opponent_required", False)),
                "same_scene": bool(continuity.get("same_scene_required", False)),
                "causal_progression": bool(continuity.get("causal_relation_required")),
            },
            "hard_requirements": [f"phase:{phase}" for phase in phases] + [
                f"continuity:{key}" for key, enabled in {
                    "same_subject": continuity.get("same_subject_required"),
                    "same_opponent": continuity.get("same_opponent_required", False),
                    "same_scene": continuity.get("same_scene_required", False),
                    "causal_progression": continuity.get("causal_relation_required"),
                }.items() if enabled
            ],
            "disallowed_substitutes": list(evidence.get("disallowed_substitutes") or []),
            "unit_strategy": {
                "generation_unit_count_range": list(
                    presentation.get("snippet_count_range") or [1, 1]),
                "source_continuity": presentation.get("source_continuity"),
                "ordering": presentation.get("ordering_constraint"),
            },
            "traceability": deepcopy(row.get("traceability") or {}),
        }
        section["generation_units"] = _compile_units(section)
        sections.append(section)
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "created_at": _now(),
        "v9_program_hashes": deepcopy(frozen["program_sha256"]),
        "target_setting_hash": _file_hash(target_setting_path),
        "target_setting": target_setting,
        "sections": sections,
        "canonical": True,
    }
    validation = validate_generation_contract(contract)
    if not validation["passed"]:
        raise V9GBlocked("contract", "generation_contract_invalid",
                         ";".join(validation["errors"]))
    contract["contract_hash"] = _json_hash(contract)
    if output_path is not None:
        _write_json(output_path, contract)
    return contract


def validate_generation_contract(contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if contract.get("contract_version") != CONTRACT_VERSION:
        errors.append("contract_version_invalid")
    if contract.get("canonical") is not True:
        errors.append("contract_not_canonical")
    if not contract.get("v9_program_hashes") or not contract.get("target_setting_hash"):
        errors.append("contract_provenance_missing")
    if contract.get("contract_hash"):
        unhashed = deepcopy(contract)
        recorded_hash = unhashed.pop("contract_hash")
        if recorded_hash != _json_hash(unhashed):
            errors.append("contract_hash_mismatch")
    if not contract.get("sections"):
        errors.append("contract_sections_empty")
    seen: set[str] = set()
    for section in contract.get("sections") or []:
        section_id = str(section.get("section_id") or "")
        if not section_id or section_id in seen:
            errors.append(f"section_id_invalid:{section_id}")
        seen.add(section_id)
        if not section.get("semantic_goal") or not section.get("required_phases"):
            errors.append(f"section_semantics_missing:{section_id}")
        if section.get("composition_mode") not in COMPOSITION_MODES:
            errors.append(f"composition_mode_invalid:{section_id}")
        if float(section.get("duration_budget_s") or 0) <= 0:
            errors.append(f"duration_budget_invalid:{section_id}")
        low, high = (section.get("unit_strategy") or {}).get(
            "generation_unit_count_range", [0, 0])
        if not (1 <= int(low) <= int(high) <= 12):
            errors.append(f"unit_count_range_invalid:{section_id}")
        for unit in section.get("generation_units") or []:
            if not 4 <= float(unit.get("target_duration_s") or 0) <= 15:
                errors.append(f"h3_duration_unsupported:{unit.get('unit_id')}")
    return {"passed": not errors, "errors": errors}


def adapt_contract_to_filmdsl(contract: dict[str, Any], state_pack: dict[str, Any]) \
        -> dict[str, Any]:
    """Create a derived FilmDSL-shaped dictionary; never mutate the contract."""
    before = _json_hash(contract)
    clips = []
    for section in contract.get("sections") or []:
        for unit in section.get("generation_units") or []:
            clips.append({
                "clip_id": unit["unit_id"],
                "section_id": section["section_id"],
                "narrative_action": {
                    "semantic_goal": section["semantic_goal"],
                    "phases": deepcopy(unit["semantic_phases"]),
                },
                "cinematic_staging": {
                    "continuity": deepcopy(section["continuity"]),
                    "section_invariants": deepcopy(
                        state_pack.get("section_invariants", {}).get(
                            section["section_id"], {})),
                },
                "render_spec": {
                    "duration_s": unit["target_duration_s"],
                    "short_edge": 768, "fps": 24, "audio_hz": 32000,
                },
                "contract_hash": contract["contract_hash"],
            })
    value = {
        "schema_version": FILMDSL_ADAPTER_VERSION,
        "derived_from_contract_hash": contract["contract_hash"],
        "meta": deepcopy(contract.get("target_setting", {}).get("meta") or {}),
        "assets": deepcopy(contract.get("target_setting", {}).get("assets") or {}),
        "memory": deepcopy(state_pack.get("commits") or []),
        "clips": clips,
    }
    if _json_hash(contract) != before:
        raise AssertionError("FilmDSL adapter mutated the canonical contract")
    return value


def build_asset_and_state_pack(contract: dict[str, Any], *,
                               canonical_invariants: dict[str, Any] | None = None,
                               section_invariants: dict[str, Any] | None = None,
                               authorized_mutable_state: dict[str, Any] | None = None) \
        -> dict[str, Any]:
    return {
        "schema_version": STATE_VERSION,
        "contract_hash": contract["contract_hash"],
        "assets": deepcopy(contract.get("target_setting", {}).get("assets") or {}),
        "canonical_invariants": deepcopy(canonical_invariants or {}),
        "section_invariants": deepcopy(section_invariants or {}),
        "authorized_mutable_state": deepcopy(authorized_mutable_state or {}),
        "observed_candidates": [], "commits": [],
    }


def commit_authorized_state(state_pack: dict[str, Any], observed: dict[str, Any], *,
                            raw_unit_passed: bool, snippets_passed: bool,
                            contract_hash: str) -> dict[str, Any]:
    """Append only legal transitions after raw and cropped reviews both pass."""
    state_pack.setdefault("observed_candidates", []).append(deepcopy(observed))
    if state_pack.get("contract_hash") != contract_hash:
        raise V9GBlocked("state", "state_contract_hash_mismatch")
    if not raw_unit_passed or not snippets_passed:
        return {"committed": False, "reason": "review_not_passed"}
    changes = observed.get("state_changes") or {}
    immutable = set(state_pack.get("canonical_invariants") or {})
    section_immutable = set((state_pack.get("section_invariants") or {}).get(
        str(observed.get("section_id") or ""), {}))
    illegal_immutable = sorted(set(changes) & (immutable | section_immutable))
    if illegal_immutable:
        return {"committed": False, "reason": "invariant_changed",
                "fields": illegal_immutable}
    allowed = state_pack.get("authorized_mutable_state") or {}
    for field, transition in changes.items():
        if field not in allowed:
            return {"committed": False, "reason": "unauthorized_state_field",
                    "field": field}
        pair = [transition.get("from"), transition.get("to")]
        if pair not in (allowed[field].get("transitions") or []):
            return {"committed": False, "reason": "unauthorized_transition",
                    "field": field, "transition": pair}
    commit = {
        "commit_id": f"state_commit_{len(state_pack.setdefault('commits', [])) + 1:03d}",
        "created_at": _now(), "contract_hash": contract_hash,
        "unit_id": observed.get("unit_id"), "changes": deepcopy(changes),
    }
    state_pack["commits"].append(commit)
    return {"committed": True, "commit": commit}


def route_h3_mode(*, references: list[dict[str, Any]] | None = None,
                  first_frame: str | None = None, last_frame: str | None = None,
                  capabilities: dict[str, Any] | None = None,
                  allow_explicit_split: bool = False) -> dict[str, Any]:
    references = list(references or [])
    caps = capabilities or {}
    has_first, has_last = bool(first_frame), bool(last_frame)
    suffix = "first_last" if has_first and has_last else "first" if has_first else \
        "last" if has_last else ""
    if references:
        mode = "ref2va"
        reference_capabilities = {
            "image": "ref2va_image", "video": "ref2va_video",
            "video_audio": "ref2va_video", "audio": "ref2va_audio",
        }
        missing_types = [str(row.get("type")) for row in references
                         if row.get("type") not in reference_capabilities]
        if missing_types:
            raise V9GBlocked("capability", "ref2va_reference_type_unsupported",
                             ",".join(missing_types))
        required_reference_caps = {
            reference_capabilities[str(row.get("type"))] for row in references}
        capability = f"ref2va_hybrid_{suffix}" if suffix else \
            sorted(required_reference_caps)[0]
        if suffix and caps.get(capability) is not True:
            if allow_explicit_split:
                return {"mode": None, "variant": None, "split_required": True,
                        "reason_code": "ref2va_hybrid_capability_unavailable",
                        "preserve": ["references", "keyframes"]}
            raise V9GBlocked("capability", "ref2va_hybrid_capability_unavailable",
                             capability)
        unavailable_references = sorted(
            name for name in required_reference_caps if caps and caps.get(name) is not True)
        if unavailable_references:
            raise V9GBlocked("capability", "h3_reference_capability_not_proven",
                             ",".join(unavailable_references))
    elif suffix:
        mode, capability = "fl2va", f"fl2va_{suffix}"
    else:
        mode, capability = "t2va", "t2va"
    if caps and caps.get(capability) is not True:
        raise V9GBlocked("capability", "h3_capability_not_proven", capability)
    return {"mode": mode, "variant": "ref2va" if mode == "ref2va" else "fl2va",
            "capability": capability, "split_required": False}


def build_h3_request(*, prompt: str, duration_s: float, references: list[dict[str, Any]] | None,
                     first_frame: str | None, last_frame: str | None,
                     capabilities: dict[str, Any], seed: int = 1,
                     aspect_ratio: str = "16:9", inference_steps: int = 50) \
        -> dict[str, Any]:
    if not 4 <= float(duration_s) <= 15:
        raise V9GBlocked("generation", "h3_duration_out_of_range", str(duration_s))
    route = route_h3_mode(references=references, first_frame=first_frame,
                          last_frame=last_frame, capabilities=capabilities)
    conditions = []
    for reference in references or []:
        if reference.get("type") not in {"image", "video", "video_audio", "audio"}:
            raise V9GBlocked("generation", "h3_reference_type_invalid")
        row = deepcopy(reference)
        row["role"] = "reference"
        conditions.append(row)
    if first_frame:
        conditions.append({"type": "image", "uri": first_frame,
                           "role": "keyframe", "frame_index": 0})
    if last_frame:
        conditions.append({"type": "image", "uri": last_frame,
                           "role": "keyframe", "frame_index": -1})
    if route["mode"] == "ref2va" and not any(
            row.get("role") == "reference" for row in conditions):
        raise V9GBlocked("generation", "ref2va_reference_required")
    return {
        "model": "MiniMaxAI/MiniMax-H3", "prompt": prompt,
        "seconds": float(duration_s), "task": route["mode"],
        "conditions": conditions,
        "target": {"short_edge": 768, "aspect_ratio": aspect_ratio,
                   "duration_seconds": float(duration_s)},
        "num_outputs_per_prompt": 1, "num_inference_steps": int(inference_steps),
        "flow_shift": 12.0, "audio_flow_shift": 3.0, "seed": int(seed),
        "_v9g_route": route,
    }


class SGLangH3Client:
    """Thin implementation of the official asynchronous SGLang /v1/videos API."""

    def __init__(self, endpoint: str, *, timeout_s: float = 7200.0,
                 poll_s: float = 2.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)

    def generate(self, request: dict[str, Any], output_path: Path) -> dict[str, Any]:
        allowed = {
            "model", "prompt", "seconds", "task", "conditions", "target",
            "num_outputs_per_prompt", "num_inference_steps", "flow_shift",
            "audio_flow_shift", "seed", "quality", "lora_name", "lora_scale",
        }
        payload = {key: value for key, value in request.items() if key in allowed}
        response = requests.post(f"{self.endpoint}/v1/videos", json=payload, timeout=60)
        response.raise_for_status()
        created = response.json()
        job_id = created.get("id")
        if not job_id:
            raise V9GBlocked("generation", "h3_job_id_missing")
        deadline = time.monotonic() + self.timeout_s
        status_payload: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status_response = requests.get(
                f"{self.endpoint}/v1/videos/{job_id}", timeout=30)
            status_response.raise_for_status()
            status_payload = status_response.json()
            status = status_payload.get("status")
            if status == "completed":
                break
            if status == "failed":
                raise V9GBlocked("generation", "h3_job_failed",
                                 json.dumps(status_payload, ensure_ascii=False)[:500])
            time.sleep(self.poll_s)
        else:
            raise V9GBlocked("generation", "h3_job_timeout", str(job_id))
        media = requests.get(f"{self.endpoint}/v1/videos/{job_id}/content",
                             timeout=300)
        media.raise_for_status()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(media.content)
        return {"id": job_id, "status": status_payload,
                "output_path": str(output_path), "sha256": _file_hash(output_path)}


def build_generation_jobs(contract: dict[str, Any], capabilities: dict[str, Any],
                          output_path: Path | None = None) -> list[dict[str, Any]]:
    """Compile Contract units into official H3 requests without changing semantics."""
    target = contract.get("target_setting") or {}
    condition_policy = target.get("generation_conditions") or {}
    default = condition_policy.get("default") or {}
    by_section = condition_policy.get("sections") or {}
    by_unit = condition_policy.get("units") or {}
    jobs = []
    for section in contract.get("sections") or []:
        for unit in section.get("generation_units") or []:
            conditions = deepcopy(default)
            conditions.update(deepcopy(by_section.get(section["section_id"]) or {}))
            conditions.update(deepcopy(by_unit.get(unit["unit_id"]) or {}))
            assets = target.get("assets") or {}
            prompt = (
                f"Semantic goal: {section['semantic_goal']}. "
                f"This unit must visibly cover these phases: "
                f"{', '.join(unit['semantic_phases'])}. "
                f"Maintain the declared subject, scene, opponent and body structure. "
                f"Canonical assets: {json.dumps(assets, ensure_ascii=False, sort_keys=True)}"
            )
            request = build_h3_request(
                prompt=prompt, duration_s=unit["target_duration_s"],
                references=conditions.get("references") or [],
                first_frame=conditions.get("first_frame"),
                last_frame=conditions.get("last_frame"), capabilities=capabilities,
                seed=int(conditions.get("seed", 1000 + len(jobs))),
                aspect_ratio=str(conditions.get("aspect_ratio") or "16:9"),
                inference_steps=int(conditions.get("inference_steps") or 50))
            request.update({"contract_hash": contract["contract_hash"],
                            "section_id": section["section_id"],
                            "unit_id": unit["unit_id"]})
            jobs.append({"unit_id": unit["unit_id"],
                         "section_id": section["section_id"], "request": request})
    if output_path is not None:
        _write_json(output_path, {"contract_hash": contract["contract_hash"],
                                  "jobs": jobs})
    return jobs


def _capability_cases(fixtures: dict[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        ("t2va", "fl2va", {}),
        ("fl2va_first", "fl2va", {"first_frame": fixtures["first"]}),
        ("fl2va_last", "fl2va", {"last_frame": fixtures["last"]}),
        ("fl2va_first_last", "fl2va",
         {"first_frame": fixtures["first"], "last_frame": fixtures["last"]}),
        ("ref2va_image", "ref2va", {"references": [
            {"type": "image", "uri": fixtures["image"]}]}),
        ("ref2va_video", "ref2va", {"references": [
            {"type": "video", "uri": fixtures["video"]}]}),
        ("ref2va_audio", "ref2va", {"references": [
            {"type": "audio", "uri": fixtures["audio"]}]}),
        ("ref2va_hybrid_first", "ref2va", {
            "references": [{"type": "image", "uri": fixtures["image"]}],
            "first_frame": fixtures["first"]}),
        ("ref2va_hybrid_last", "ref2va", {
            "references": [{"type": "image", "uri": fixtures["image"]}],
            "last_frame": fixtures["last"]}),
        ("ref2va_hybrid_first_last", "ref2va", {
            "references": [{"type": "image", "uri": fixtures["image"]}],
            "first_frame": fixtures["first"], "last_frame": fixtures["last"]}),
    ]


def probe_h3_capabilities(clients: dict[str, Any], fixtures: dict[str, str],
                          output_dir: Path) -> dict[str, Any]:
    """Probe every capability with real requests; documentation is not evidence."""
    output_dir = Path(output_dir)
    results: dict[str, Any] = {}
    for name, variant, kwargs in _capability_cases(fixtures):
        permissive = {key: True for key in H3_CAPABILITIES}
        request = build_h3_request(
            prompt=f"V9-G capability smoke {name}", duration_s=4,
            references=kwargs.get("references"), first_frame=kwargs.get("first_frame"),
            last_frame=kwargs.get("last_frame"), capabilities=permissive,
            seed=1000 + len(results))
        request_path = _write_json(output_dir / name / "request.json", request)
        try:
            response = clients[variant].generate(
                request, output_dir / name / "output.mp4")
            passed = Path(response["output_path"]).is_file() and \
                Path(response["output_path"]).stat().st_size > 0
            results[name] = {
                "passed": passed, "variant": variant,
                "request_sha256": _file_hash(request_path),
                "response": response,
            }
        except Exception as exc:  # each capability remains independently auditable
            results[name] = {"passed": False, "variant": variant,
                             "error": f"{type(exc).__name__}: {exc}"}
    manifest = {
        "schema_version": CAPABILITY_VERSION, "created_at": _now(),
        **{name: bool(results.get(name, {}).get("passed")) for name in H3_CAPABILITIES},
        "evidence": results,
    }
    _write_json(output_dir / "capability_manifest.json", manifest)
    return manifest


def _unit_contract(contract: dict[str, Any], unit_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for section in contract.get("sections") or []:
        for unit in section.get("generation_units") or []:
            if unit.get("unit_id") == unit_id:
                return section, unit
    raise V9GBlocked("generation", "generation_unit_not_in_contract", unit_id)


def run_generate_observe_repair(
        contract: dict[str, Any], unit_id: str, output_dir: Path, *,
        generate: Callable[[dict[str, Any], Path], dict[str, Any]],
        neutral_observe: Callable[[Path, str], dict[str, Any]],
        verify_contract: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        plan_repair: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        initial_request: dict[str, Any], max_repairs: int = 2) -> dict[str, Any]:
    """NEWTON-style bounded Planner–Executor–Verifier loop with neutral observation."""
    if max_repairs != 2:
        raise V9GBlocked("policy", "repair_budget_must_equal_two")
    section, unit = _unit_contract(contract, unit_id)
    output_dir = Path(output_dir)
    trace = ArtifactTrace(output_dir, unit_id)
    memory = NewtonTraceAdapter(objective=section["semantic_goal"])
    request = deepcopy(initial_request)
    contract_hash = contract["contract_hash"]
    attempts = []
    for attempt_index in range(max_repairs + 1):
        attempt_dir = output_dir / "units" / unit_id / f"attempt_{attempt_index + 1:03d}"
        request["contract_hash"] = contract_hash
        request["unit_id"] = unit_id
        _write_json(attempt_dir / "request.json", request)
        try:
            generated = generate(request, attempt_dir / "raw_unit.mp4")
            video_path = Path(generated["output_path"])
            observation = neutral_observe(video_path, NEUTRAL_OBSERVATION_PROMPT)
            verification = verify_contract(observation, {
                "contract_hash": contract_hash, "section": section, "unit": unit,
            })
            passed = verification.get("passed") is True
            record = {
                "attempt": attempt_index + 1, "request": request,
                "generated": generated, "neutral_observation": observation,
                "contract_verification": verification, "raw_unit_passed": passed,
                "verifier_independence": "same_model_prompt_isolation",
            }
            attempts.append(record)
            trace.write_artifact(
                f"attempt_{attempt_index + 1:03d}.json", record)
            trace.append("contract_verification", {
                "attempt": attempt_index + 1, "passed": passed,
                "issues": verification.get("issues") or [],
            })
            memory.add_generation(
                turn=attempt_index + 1, prompt=str(request.get("prompt") or ""),
                ref_images=[str(row.get("uri")) for row in request.get("conditions") or []
                            if row.get("type") == "image"],
                video_path=str(video_path), issues=verification.get("issues") or [],
                summary=verification.get("summary"), passed=passed)
            if passed:
                result = {
                    "passed": True, "unit_id": unit_id,
                    "accepted_attempt": attempt_index + 1,
                    "raw_unit_path": str(video_path), "attempts": attempts,
                    "contract_hash": contract_hash,
                    "raw_unit_review": verification,
                    "snippet_review": None, "section_review": None,
                }
                trace.write_artifact("trace.json", memory.to_dict())
                _write_json(output_dir / "units" / unit_id / "unit_result.json", result)
                return result
            if attempt_index < max_repairs:
                repair = plan_repair(record, memory.to_dict())
                if repair.get("delete_hard_requirements"):
                    raise V9GBlocked("repair", "repair_attempted_requirement_deletion")
                request = deepcopy(repair.get("request") or request)
        except V9GBlocked:
            raise
        except Exception as exc:
            attempts.append({"attempt": attempt_index + 1,
                             "error": f"{type(exc).__name__}: {exc}"})
            memory.add_generation(
                turn=attempt_index + 1, prompt=str(request.get("prompt") or ""),
                error=f"{type(exc).__name__}: {exc}")
            if attempt_index >= max_repairs:
                break
    trace.write_artifact("trace.json", memory.to_dict())
    result = {"passed": False, "unit_id": unit_id, "attempts": attempts,
              "contract_hash": contract_hash, "failure_stage": "generation",
              "reason_code": "repair_budget_exhausted"}
    _write_json(output_dir / "units" / unit_id / "unit_result.json", result)
    return result


def run_v9g_generation_jobs(contract: dict[str, Any], jobs: list[dict[str, Any]],
                             output_dir: Path, *, clients: dict[str, SGLangH3Client],
                             runner: Any, capabilities: dict[str, Any],
                             max_units: int | None = None) -> list[dict[str, Any]]:
    """Execute compiled jobs with prompt-isolated Omni observation and verification."""
    output_dir = Path(output_dir)
    selected_jobs = jobs[:max_units] if max_units is not None else jobs
    results = []
    for job in selected_jobs:
        initial = deepcopy(job["request"])

        def generate(request: dict[str, Any], destination: Path) -> dict[str, Any]:
            conditions = request.get("conditions") or []
            references = [row for row in conditions if row.get("role") == "reference"]
            keyframes = {row.get("frame_index"): row.get("uri") for row in conditions
                         if row.get("role") == "keyframe"}
            route = route_h3_mode(
                references=references, first_frame=keyframes.get(0),
                last_frame=keyframes.get(-1), capabilities=capabilities)
            return clients[route["variant"]].generate(request, destination)

        def observe(video: Path, prompt: str) -> dict[str, Any]:
            answer = runner.watch(video, prompt, fps=4.0, max_new_tokens=1024)
            return _parse_one_object(answer, stage="neutral_observation")

        def verify(observation: dict[str, Any], expected: dict[str, Any]) \
                -> dict[str, Any]:
            prompt = (
                "Compare a neutral observation with a canonical generation contract. "
                "Do not invent unobserved facts. Return one JSON object with passed "
                "(boolean), issues (array), hard_requirement_results (object), "
                "invariant_results (object), transition_results (object), and summary.\n"
                f"OBSERVATION={json.dumps(observation, ensure_ascii=False)}\n"
                f"CONTRACT={json.dumps(expected, ensure_ascii=False)}")
            return _parse_one_object(
                runner.ask(prompt, max_new_tokens=1024), stage="contract_verification")

        def repair(record: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
            old = record["request"]
            prompt = (
                "Plan one bounded repair for a failed H3 generation. Preserve every hard "
                "requirement and every existing condition modality. You may refine the "
                "prompt, change seed, or replace a reference/keyframe with an explicitly "
                "supplied alternative. Return one JSON object containing prompt and seed; "
                "optionally references, first_frame, last_frame.\n"
                f"FAILED={json.dumps(record, ensure_ascii=False)}\n"
                f"HISTORY={json.dumps(history, ensure_ascii=False)}")
            proposal = _parse_one_object(
                runner.ask(prompt, max_new_tokens=1024), stage="repair_planning")
            old_conditions = old.get("conditions") or []
            old_refs = [row for row in old_conditions if row.get("role") == "reference"]
            old_keys = {row.get("frame_index"): row.get("uri") for row in old_conditions
                        if row.get("role") == "keyframe"}
            references = proposal.get("references") or old_refs
            first_frame = proposal.get("first_frame") or old_keys.get(0)
            last_frame = proposal.get("last_frame") or old_keys.get(-1)
            repaired = build_h3_request(
                prompt=str(proposal.get("prompt") or old.get("prompt") or ""),
                duration_s=float(old["target"]["duration_seconds"]),
                references=references, first_frame=first_frame, last_frame=last_frame,
                capabilities=capabilities, seed=int(proposal.get("seed", old.get("seed", 1))),
                aspect_ratio=str(old["target"].get("aspect_ratio") or "16:9"),
                inference_steps=int(old.get("num_inference_steps") or 50))
            repaired.update({key: old[key] for key in (
                "contract_hash", "section_id", "unit_id") if key in old})
            return {"request": repaired}

        results.append(run_generate_observe_repair(
            contract, job["unit_id"], output_dir, generate=generate,
            neutral_observe=observe, verify_contract=verify, plan_repair=repair,
            initial_request=initial))
    _write_json(output_dir / "generation_results.json", {
        "contract_hash": contract["contract_hash"], "results": results})
    return results


def select_evidence_moments(event_timeline: list[dict[str, Any]], *,
                            required_phases: list[str], duration_budget_s: float,
                            composition_mode: str) -> dict[str, Any]:
    """Choose the minimum supported evidence set, preserving event order when required."""
    selected = []
    for phase in required_phases:
        candidates = [row for row in event_timeline
                      if row.get("phase") == phase and row.get("supported") is True]
        if not candidates:
            return {"passed": False, "reason_code": "required_phase_missing",
                    "missing_phase": phase, "snippets": []}
        candidates.sort(key=lambda row: (
            float(row["interval"][1]) - float(row["interval"][0]),
            float(row["interval"][0])))
        selected.append(deepcopy(candidates[0]))
    if composition_mode in {"event_compression_montage", "reaction_result_pair"}:
        starts = [float(row["interval"][0]) for row in selected]
        if starts != sorted(starts):
            return {"passed": False, "reason_code": "event_progression_reordered",
                    "snippets": selected}
    total = sum(float(row["interval"][1]) - float(row["interval"][0])
                for row in selected)
    if total > float(duration_budget_s) + 1e-6:
        return {"passed": False, "reason_code": "minimum_evidence_exceeds_budget",
                "duration_s": total, "snippets": selected}
    return {"passed": True, "snippets": selected,
            "duration_s": round(total, 6),
            "selection_policy": "minimum_sufficient_evidence_set"}


def finalize_unit_snippets(cfg: Any, contract: dict[str, Any],
                           unit_results: list[dict[str, Any]], state_pack: dict[str, Any],
                           output_dir: Path, *, runner: Any,
                           force: bool = False) -> list[dict[str, Any]]:
    """Select, render and independently review snippets from accepted raw Units."""
    from src.agentic_video.renderer import render_micro_montage

    output_dir = Path(output_dir)
    for result in unit_results:
        if result.get("passed") is not True:
            continue
        section, unit = _unit_contract(contract, str(result["unit_id"]))
        attempt_index = int(result["accepted_attempt"]) - 1
        attempt = result["attempts"][attempt_index]
        observation = attempt.get("neutral_observation") or {}
        selection = select_evidence_moments(
            observation.get("event_timeline") or [],
            required_phases=list(unit["semantic_phases"]),
            duration_budget_s=float(unit["target_duration_s"]),
            composition_mode=section["composition_mode"])
        result["moment_selection"] = selection
        if not selection["passed"]:
            result["snippet_review"] = {
                "passed": False, "reason_code": selection["reason_code"]}
            continue
        source = Path(result["raw_unit_path"])
        segments = []
        for snippet in selection["snippets"]:
            start, end = map(float, snippet["interval"])
            segments.append({
                "evidence_id": snippet.get("evidence_id") or snippet.get("phase"),
                "render_mode": "micro_clip", "source_video": str(source),
                "render_interval": [start, end], "duration_s": end - start,
                "phase": snippet.get("phase"),
            })
        rendered = render_micro_montage(
            cfg, {"passed": True, "duration_s": selection["duration_s"],
                  "segments": segments, "editorial_policy_version": "v9g_moment_v1"},
            output_dir / "units" / result["unit_id"] / "snippets",
            force=force)
        snippet_path = Path(rendered["content_master"])
        review_prompt = (
            "Review this cropped micro-montage against the stated phases. The raw Unit "
            "being acceptable does not imply these cuts are acceptable. Return one JSON "
            "object with passed (boolean), phase_results (object), continuity_preserved "
            "(boolean), meaning_preserved (boolean), issues (array), and summary.\n"
            f"PHASES={json.dumps(unit['semantic_phases'], ensure_ascii=False)}\n"
            f"SEMANTIC_GOAL={json.dumps(section['semantic_goal'], ensure_ascii=False)}")
        review = _parse_one_object(
            runner.watch(snippet_path, review_prompt, fps=4.0, max_new_tokens=768),
            stage="snippet_review")
        result["snippet_path"] = str(snippet_path)
        result["snippet_review"] = review
        observed_state = {
            "unit_id": result["unit_id"], "section_id": section["section_id"],
            "state_changes": observation.get("state_changes") or {},
        }
        result["state_commit"] = commit_authorized_state(
            state_pack, observed_state, raw_unit_passed=True,
            snippets_passed=review.get("passed") is True,
            contract_hash=contract["contract_hash"])
    _write_json(output_dir / "asset_state_pack.json", state_pack)
    _write_json(output_dir / "generation_results.json", {
        "contract_hash": contract["contract_hash"], "results": unit_results})
    return unit_results


def render_and_review_sections(cfg: Any, contract: dict[str, Any],
                               unit_results: list[dict[str, Any]], output_dir: Path, *,
                               runner: Any, force: bool = False) -> dict[str, Any]:
    """Assemble reviewed Unit snippets, then perform Section and final blind reviews."""
    from src.agentic_video.renderer import render_micro_montage
    from src.perception import common

    output_dir = Path(output_dir)
    by_unit = {str(row.get("unit_id")): row for row in unit_results}
    section_rows = []
    for section in contract.get("sections") or []:
        source_rows = [by_unit.get(unit["unit_id"])
                       for unit in section["generation_units"]]
        if not all(row and row.get("snippet_review", {}).get("passed") is True
                   and row.get("snippet_path") for row in source_rows):
            section_rows.append({"section_id": section["section_id"], "passed": False,
                                 "reason_code": "unit_snippet_not_passed"})
            continue
        segments = []
        for row in source_rows:
            source = Path(row["snippet_path"])
            duration = common.video_duration_s(
                cfg.perception.get("ffprobe_bin", "ffprobe"), source)
            segments.append({"render_mode": "micro_clip", "source_video": str(source),
                             "render_interval": [0.0, duration], "duration_s": duration})
        duration = sum(row["duration_s"] for row in segments)
        duration_over_budget = duration > float(section["duration_budget_s"]) + 1e-6
        completeness_reasons = [str(row.get("semantic_completeness_reason") or "").strip()
                                for row in source_rows]
        if duration_over_budget and not all(completeness_reasons):
            section_rows.append({
                "section_id": section["section_id"], "passed": False,
                "reason_code": "section_duration_exceeds_budget_without_reason",
                "duration_s": duration,
                "duration_budget_s": section["duration_budget_s"],
            })
            continue
        rendered = render_micro_montage(
            cfg, {"passed": True, "duration_s": duration, "segments": segments,
                  "editorial_policy_version": "v9g_section_v1"},
            output_dir / "sections" / section["section_id"], force=force)
        section_path = Path(rendered["content_master"])
        prompt = (
            "Blindly review this Section. Return one JSON object with passed (boolean), "
            "audience_takeaway, event_progression_clear (boolean), identity_stable "
            "(boolean), issues (array), and summary. Do not use outside story knowledge.\n"
            f"REQUIRED_MEANING={json.dumps(section['semantic_goal'], ensure_ascii=False)}")
        review = _parse_one_object(
            runner.watch(section_path, prompt, fps=4.0, max_new_tokens=768),
            stage="section_review")
        section_rows.append({
            "section_id": section["section_id"], "passed": review.get("passed") is True,
            "path": str(section_path), "duration_s": duration,
            "duration_budget_s": section["duration_budget_s"],
            "duration_over_budget": duration_over_budget,
            "semantic_completeness_reasons": completeness_reasons,
            "review": review,
        })
        for row in source_rows:
            row["section_review"] = review
    precheck = assemble_v9g_sections(contract, unit_results)
    sections_passed = precheck["passed"] and all(
        row.get("passed") is True for row in section_rows)
    blind = {"passed": False, "reason_code": "section_review_failed"}
    debug_preview = None
    if sections_passed:
        segments = [{
            "render_mode": "micro_clip", "source_video": row["path"],
            "render_interval": [0.0, row["duration_s"]], "duration_s": row["duration_s"],
        } for row in section_rows]
        total = sum(row["duration_s"] for row in segments)
        rendered = render_micro_montage(
            cfg, {"passed": True, "duration_s": total, "segments": segments,
                  "editorial_policy_version": "v9g_final_preview_v1"},
            output_dir / "final_preview", force=force)
        generated_master = Path(rendered["content_master"])
        debug_preview = output_dir / "debug_preview.mp4"
        shutil.copy2(generated_master, debug_preview)
        blind = _parse_one_object(
            runner.watch(
                debug_preview,
                "Blindly summarize the subject, event progression and visible result. "
                "Return one JSON object with passed (boolean), summary, subject_consistent "
                "(boolean), progression_comprehensible (boolean), and issues (array).",
                fps=4.0, max_new_tokens=768), stage="blind_review")
    assembly = {
        "schema_version": "v9g_section_assembly_v1",
        "contract_hash": contract["contract_hash"], "sections": section_rows,
        "passed": sections_passed and blind.get("passed") is True,
        "debug_preview": str(debug_preview) if debug_preview else None,
    }
    _write_json(output_dir / "section_assembly.json", assembly)
    _write_json(output_dir / "blind_review.json", blind)
    _write_json(output_dir / "generation_results.json", {
        "contract_hash": contract["contract_hash"], "results": unit_results})
    return assembly


def assemble_v9g_sections(contract: dict[str, Any], unit_results: list[dict[str, Any]],
                          output_path: Path | None = None) -> dict[str, Any]:
    by_unit = {str(row.get("unit_id")): row for row in unit_results}
    sections = []
    for section in contract.get("sections") or []:
        rows = [by_unit.get(unit["unit_id"]) for unit in section["generation_units"]]
        passed = all(row and row.get("passed") is True and
                     row.get("snippet_review", {}).get("passed") is True
                     for row in rows)
        sections.append({"section_id": section["section_id"], "passed": passed,
                         "unit_ids": [unit["unit_id"] for unit in section["generation_units"]]})
    result = {"schema_version": "v9g_section_assembly_v1",
              "contract_hash": contract["contract_hash"], "sections": sections,
              "passed": all(row["passed"] for row in sections)}
    if output_path is not None:
        _write_json(output_path, result)
    return result


def evaluate_v9g_experiment(rows: Iterable[dict[str, Any]], *,
                            output_path: Path | None = None) -> dict[str, Any]:
    rows = list(rows)
    required_arms = {"A", "B", "C1", "C2", "D", "E"}
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("arm") in required_arms and row.get("reference_id") and row.get("unit_id"):
            key = (str(row["reference_id"]), str(row["unit_id"]))
            paired.setdefault(key, {})[str(row["arm"])] = row
    complete_pairs = {key: arms for key, arms in paired.items()
                      if set(arms) == required_arms}
    pairs_by_reference: dict[str, int] = {}
    for reference_id, _unit_id in complete_pairs:
        pairs_by_reference[reference_id] = pairs_by_reference.get(reference_id, 0) + 1
    reference_design_passed = (
        len(pairs_by_reference) >= 4 and
        all(count >= 6 for count in pairs_by_reference.values()))
    budget_violations = []
    for key, arms in complete_pairs.items():
        adaptive, random = arms["D"], arms["E"]
        d_seconds = float(adaptive.get("generated_seconds") or 0)
        e_seconds = float(random.get("generated_seconds") or 0)
        seconds_error = abs(d_seconds - e_seconds) / max(d_seconds, e_seconds, 1e-9)
        d_gpu = float(adaptive.get("gpu_minutes") or 0)
        e_gpu = float(random.get("gpu_minutes") or 0)
        gpu_error = abs(d_gpu - e_gpu) / max(d_gpu, e_gpu, 1e-9)
        if (int(adaptive.get("request_count") or 0) !=
                int(random.get("request_count") or 0) or seconds_error > 0.05 or
                gpu_error > 0.05 or
                adaptive.get("resolution") != random.get("resolution") or
                adaptive.get("fps") != random.get("fps") or
                adaptive.get("inference_steps") != random.get("inference_steps") or
                adaptive.get("quality_preset") != random.get("quality_preset") or
                adaptive.get("reference_preprocessing_sha256") !=
                random.get("reference_preprocessing_sha256")):
            budget_violations.append({"reference_id": key[0], "unit_id": key[1],
                                      "generated_seconds_relative_error": seconds_error,
                                      "gpu_minutes_relative_error": gpu_error})
    total_generated = sum(float(row.get("generated_seconds") or 0) for row in rows)
    total_adopted = sum(float(row.get("adopted_seconds") or 0) for row in rows)
    adoption = total_adopted / total_generated if total_generated else 0.0
    result = {
        "schema_version": "v9g_experiment_metrics_v1",
        "unit_count": len(rows),
        "success_count": sum(row.get("passed") is True for row in rows),
        "success_at_budget": (sum(row.get("passed") is True for row in rows) / len(rows)
                              if rows else 0.0),
        "generated_seconds": round(total_generated, 6),
        "adopted_seconds": round(total_adopted, 6),
        "adoption_ratio": round(adoption, 6),
        "waste_ratio": round(1.0 - adoption, 6),
        "gpu_minutes": round(sum(float(row.get("gpu_minutes") or 0) for row in rows), 6),
        "review_cost": round(sum(float(row.get("review_cost") or 0) for row in rows), 6),
        "paired_unit_count": len(complete_pairs),
        "paired_units_by_reference": pairs_by_reference,
        "reference_design_passed": reference_design_passed,
        "required_paired_unit_count": 24,
        "d_e_budget_violations": budget_violations,
        "formal_mechanism_claim_authorized": (
            len(complete_pairs) >= 24 and reference_design_passed and
            not budget_violations),
    }
    result["arms"] = {}
    for arm in sorted(required_arms):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if arm_rows:
            result["arms"][arm] = {
                "unit_count": len(arm_rows),
                "success_at_budget": sum(row.get("passed") is True
                                         for row in arm_rows) / len(arm_rows),
                "generated_seconds": sum(float(row.get("generated_seconds") or 0)
                                         for row in arm_rows),
                "gpu_minutes": sum(float(row.get("gpu_minutes") or 0)
                                   for row in arm_rows),
            }
    if output_path is not None:
        _write_json(output_path, result)
    return result


def accept_v9g(output_dir: Path, human_review_path: Path) -> dict[str, Any]:
    """Final human gate; it never manufactures a delivery artifact."""
    output_dir = Path(output_dir)
    assembly_path = output_dir / "section_assembly.json"
    blind_path = output_dir / "blind_review.json"
    if not assembly_path.is_file() or not blind_path.is_file():
        raise V9GBlocked("acceptance", "assembly_or_blind_review_missing")
    assembly = _read_json(assembly_path)
    blind = _read_json(blind_path)
    human = _read_json(human_review_path)
    required_sections = {row["section_id"] for row in assembly.get("sections") or []}
    human_rows = {str(row.get("section_id")): row
                  for row in human.get("sections") or []}
    human_passed = bool(required_sections) and all(
        human_rows.get(section_id, {}).get("passed") is True
        for section_id in required_sections)
    passed = (assembly.get("passed") is True and blind.get("passed") is True and
              human_passed and human.get("passed") is True)
    result = {
        "schema_version": "v9g_acceptance_v1", "passed": passed,
        "decision": "PASS" if passed else "BLOCKED",
        "all_required_sections_passed": assembly.get("passed") is True,
        "blind_review_passed": blind.get("passed") is True,
        "human_acceptance_passed": human_passed and human.get("passed") is True,
        "formal_artifact_authorized": passed,
    }
    if passed:
        preview = output_dir / "debug_preview.mp4"
        if not preview.is_file():
            raise V9GBlocked("acceptance", "debug_preview_missing")
        formal = output_dir / "rendered.mp4"
        shutil.copy2(preview, formal)
        result["rendered_mp4"] = str(formal)
        result["rendered_sha256"] = _file_hash(formal)
    _write_json(output_dir / "human_review.json", human)
    _write_json(output_dir / "acceptance.json", result)
    return result


def validate_upstream_lock(repo_root: Path, lock_path: Path | None = None) -> dict[str, Any]:
    """Fail clearly if a pinned vendored serialization boundary drifts."""
    repo_root = Path(repo_root)
    lock_path = lock_path or repo_root / "third_party" / "UPSTREAM_LOCK.json"
    lock = _read_json(lock_path)
    errors = []
    entries = []
    for row in lock.get("entries") or []:
        path = repo_root / row["vendored_path"]
        current = _file_hash(path) if path.is_file() else None
        expected = row.get("vendored_sha256")
        if not expected or current != expected:
            errors.append(f"vendored_boundary_drift:{row.get('name')}")
        entries.append({"name": row.get("name"), "path": str(path),
                        "expected": expected, "current": current})
    return {"passed": not errors, "errors": errors, "entries": entries}
