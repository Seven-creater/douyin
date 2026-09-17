"""P5.1: audited SGLang H3 capability transport, not a capability claim.

The official endpoint accepts JSON with server-local ``file://`` condition URIs.
No fake or mock HTTP observation can promote a real capability to effective.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .generation_p4 import (
    P4Blocked, _file_hash, _hash, evaluate_condition_effect,
    validate_capability_registry, validate_generated_media_transport,
    validate_h3_condition_integrity, verify_frozen_program_snapshot,
)
from .generation_v9g import SGLangH3Client, V9GBlocked, _write_json


class P51Blocked(RuntimeError):
    def __init__(self, reason_code: str, failure_class: str = "request_contract") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.failure_class = failure_class


def audit_real_h3_payload(plan: dict[str, Any], payload: dict[str, Any], *,
                          production: bool = False,
                          registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare P4's frozen request/asset bytes with the actual JSON body."""
    request, manifest, assets = plan["request"], plan["manifest"], plan["assets"]
    backend = manifest.get("backend") or {}
    if backend.get("type") != "sglang_h3" or \
            backend.get("protocol") != "sglang_v1_videos_json":
        raise P51Blocked("H3_BACKEND_UNAVAILABLE", "backend_capability")
    try:
        validate_h3_condition_integrity(request, manifest, assets)
    except P4Blocked as exc:
        if exc.reason_code == "asset_sha_mismatch":
            raise P51Blocked("BLOCKED_ASSET_DRIFT") from exc
        raise P51Blocked(exc.reason_code) from exc
    expected = SGLangH3Client.serialize_payload(request)
    if payload != expected or payload.get("conditions") != request.get("conditions") or \
            payload.get("task") != manifest.get("mode"):
        raise P51Blocked("request_payload_mismatch")
    if len(payload.get("conditions") or []) != len(manifest["conditions"]):
        raise P51Blocked("request_condition_count_mismatch")
    if production:
        if registry is None:
            raise P51Blocked("capability_registry_missing")
        validate_capability_registry(registry)
        for name in manifest["expected_capabilities"]:
            if registry["capabilities"].get(name) != "effective":
                raise P51Blocked("production_capability_not_effective")
    elif registry is not None and any(
            registry.get("capabilities", {}).get(name) == "unsupported"
            for name in manifest["expected_capabilities"]):
        raise P51Blocked("backend_capability_unsupported", "backend_capability")
    sent = []
    asset_map = {asset["asset_id"]: asset for asset in assets}
    for index, (binding, condition) in enumerate(zip(
            manifest["conditions"], payload["conditions"])):
        asset = asset_map[binding["asset_id"]]
        if (condition["uri"] != asset["request_uri"] or
                _file_hash(Path(asset["path"])) != asset["sha256"]):
            raise P51Blocked("BLOCKED_ASSET_DRIFT")
        uri = urlparse(condition["uri"])
        if uri.scheme != "file":
            raise P51Blocked("h3_condition_uri_not_server_local")
        sent.append({
            "position": index, "label": binding["request_label"],
            "asset_id": asset["asset_id"], "sha256": asset["sha256"],
            "bytes": Path(asset["path"]).stat().st_size,
            "field_name": f"conditions[{index}].uri",
            "uri": condition["uri"], "type": condition["type"],
            "role": condition["role"],
            "frame_index": condition.get("frame_index"),
        })
    body = SGLangH3Client.wire_body(payload)
    return {"passed": True, "protocol": "sglang_v1_videos_json",
            "request_id": manifest["request_id"],
            "request_manifest_sha256": manifest["manifest_sha256"],
            "payload_sha256": _hash(payload),
            "http_body_sha256": hashlib.sha256(body).hexdigest(),
            "http_body_bytes": len(body),
            "sent_conditions": sent}


def audit_backend_receipt(plan: dict[str, Any], response: dict[str, Any]) \
        -> dict[str, Any]:
    """A server receipt is optional, but if present it must match exactly."""
    accepted = response.get("accepted_conditions")
    if accepted is None:
        return {"available": False, "status": "client_receipt_only"}
    expected = plan["request"].get("conditions") or []
    if accepted != expected:
        raise P51Blocked("request_delivery_mismatch", "request_contract")
    return {"available": True, "status": "matched", "condition_count": len(accepted)}


def _record_state(output_dir: Path, history: list[str], state: str) -> None:
    history.append(state)
    _write_json(output_dir / "job_state.json", {"status": state, "history": history})


def run_h3_capability_case(plan: dict[str, Any], client: SGLangH3Client,
                           output_dir: Path, *, mock_http: bool = False,
                           effect_checks: list[dict[str, Any]] | None = None,
                           short_edge: int = 768) -> dict[str, Any]:
    """Submit/poll/download one audited case; transport never implies effect."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[str] = []
    _record_state(output_dir, history, "planned")
    manifest = plan["manifest"]
    if (manifest["backend"].get("endpoint") or "").rstrip("/") != client.endpoint:
        raise P51Blocked("backend_endpoint_mismatch")
    payload = SGLangH3Client.serialize_payload(plan["request"])
    receipt = audit_real_h3_payload(plan, payload)
    _write_json(output_dir / "request.json", payload)
    _write_json(output_dir / "request_manifest.json", manifest)
    _write_json(output_dir / "condition_assets.json", plan["assets"])
    _write_json(output_dir / "client_payload_receipt.json", receipt)
    (output_dir / "prompt.txt").write_text(payload["prompt"], encoding="utf-8")
    _record_state(output_dir, history, "payload_validated")
    case_id = plan.get("case_id") or manifest["request_id"]
    try:
        created = client.submit(payload)
        _write_json(output_dir / "backend_response.json", created)
        _record_state(output_dir, history, "submitted")
        backend_receipt = audit_backend_receipt(plan, created)
        _write_json(output_dir / "backend_receipt_audit.json", backend_receipt)
        job_id = str(created["id"])
        _record_state(output_dir, history, "running")
        status = client.poll(job_id)
        _write_json(output_dir / "backend_status.json", status)
        _record_state(output_dir, history, "completed")
        media = client.download(job_id, output_dir / "generated.mp4")
        _record_state(output_dir, history, "downloaded")
    except P51Blocked as exc:
        state = "request_delivery_mismatch" if exc.reason_code == \
            "request_delivery_mismatch" else "submit_failed"
        _record_state(output_dir, history, state)
        return {"case_id": case_id, "status": state,
                "reason_code": exc.reason_code, "failure_class": exc.failure_class,
                "history": history}
    except V9GBlocked as exc:
        state = ("backend_failed" if exc.reason_code == "h3_job_failed" else
                 "poll_failed" if exc.reason_code == "h3_job_timeout" else
                 "submit_failed")
        _record_state(output_dir, history, state)
        return {"case_id": case_id, "status": state,
                "reason_code": exc.reason_code, "failure_class": "infrastructure",
                "history": history}
    except (requests.RequestException, OSError, ValueError) as exc:
        state = "submit_failed" if "submitted" not in history else \
            "download_failed" if "completed" in history else "poll_failed"
        _record_state(output_dir, history, state)
        return {"case_id": case_id, "status": state,
                "reason_code": type(exc).__name__,
                "failure_class": "infrastructure", "history": history}
    transport = validate_generated_media_transport(
        Path(media["output_path"]), duration_s=float(payload["seconds"]),
        short_edge=short_edge)
    _write_json(output_dir / "ffprobe.json", transport)
    result_manifest = {
        "request_id": manifest["request_id"], "capability_case_id": case_id,
        "request_manifest_sha256": manifest["manifest_sha256"],
        "backend_job_id": job_id, "output_sha256": _file_hash(
            Path(media["output_path"])),
        "output_bytes": Path(media["output_path"]).stat().st_size,
        "transport": transport,
    }
    _write_json(output_dir / "backend_result_manifest.json", result_manifest)
    if not transport["backend_transport_ok"]:
        _record_state(output_dir, history, "transport_failed")
        return {"case_id": case_id, "status": "transport_failed",
                "reason_code": transport["reason_code"],
                "failure_class": "infrastructure", "history": history}
    _record_state(output_dir, history, "transport_validated")
    _record_state(output_dir, history, "effect_pending")
    try:
        if effect_checks:
            effect = evaluate_condition_effect(
                manifest, received_conditions=created.get("accepted_conditions"),
                checks=effect_checks)
        else:
            effect = {"status": "unverified", "conditioning_effective": False,
                      "reason_code": "effect_checks_not_run", "checks": []}
    except (P4Blocked, ValueError, TypeError, KeyError) as exc:
        _record_state(output_dir, history, "evaluation_failed")
        return {"case_id": case_id, "status": "evaluation_failed",
                "reason_code": type(exc).__name__,
                "failure_class": "infrastructure", "history": history}
    if mock_http or not effect_checks or any(
            row.get("source") not in {"deterministic", "model_judge",
                                      "human", "human_required"}
            for row in effect_checks):
        effect = {**effect, "status": "unverified",
                  "conditioning_effective": False,
                  "evidence_scope": "mock_only" if mock_http else "unverified"}
    _write_json(output_dir / "effect_evaluation.json", effect)
    if effect["status"] in {"effective", "ineffective", "unknown"}:
        _record_state(output_dir, history, effect["status"])
    else:
        _record_state(output_dir, history, "unverified")
    return {"case_id": case_id, "status": effect["status"],
            "transport": transport, "effect": effect,
            "backend_receipt": backend_receipt, "history": history,
            "mock_http": mock_http,
            "request_sha256": manifest["request_sha256"]}


def propose_capability_registry(before: dict[str, Any],
                                results: list[dict[str, Any]], *,
                                real_backend: bool) -> dict[str, Any]:
    """Produce a proposed registry; mock results never change canonical truth."""
    validate_capability_registry(before)
    proposed = deepcopy(before)
    if not real_backend:
        return proposed
    names = {
        "S0": ["t2va"], "S1": ["first_frame"],
        "S2": ["last_frame"], "S3": ["first_last_frame"],
        "S4": ["reference_image"], "S5": ["reference_video"],
        "S6": ["reference_audio"],
        "S7": ["ref2va_hybrid", "ref2va_hybrid_first"],
    }
    for result in results:
        case_id = result["case_id"]
        if case_id not in names or result.get("mock_http") is True:
            continue
        status = result["status"]
        if status == "effective" and (
                result.get("mock_http") is True or
                not result.get("request_sha256") or
                not result.get("effect", {}).get("checks") or
                any(row.get("passed") is not True or not row.get("evidence") or
                    row.get("source") not in {"deterministic", "model_judge",
                                              "human", "human_required"}
                    for row in result["effect"]["checks"])):
            status = "unknown"
        if status in {"transport_failed", "submit_failed", "poll_failed",
                      "download_failed", "backend_failed", "evaluation_failed",
                      "request_delivery_mismatch"}:
            status = "failed"
        elif status in {"unverified", "effect_pending"} and result.get(
                "transport", {}).get("backend_transport_ok"):
            status = "transport_only"
        if case_id == "S0" and status == "effective":
            status = "transport_only"
        if status not in {"unverified", "transport_only", "effective",
                          "ineffective", "unknown", "unsupported", "failed"}:
            status = "unknown"
        for name in names[case_id]:
            proposed["capabilities"][name] = status
            proposed["evidence"][name] = {
                "source": "real", "case_id": case_id,
                "backend_transport_ok": result.get("transport", {}).get(
                    "backend_transport_ok") is True,
                "conditioning_effective": status == "effective",
                "request_sha256": result.get("request_sha256"),
            }
    proposed["status"] = "effective" if any(
        value == "effective" for value in proposed["capabilities"].values()) else \
        "transport_only"
    return proposed


def finalize_capability_registry(canonical_path: Path,
                                 before: dict[str, Any],
                                 results: list[dict[str, Any]], *,
                                 real_backend: bool) -> dict[str, Any]:
    """Commit only a complete real run; all other runs leave canonical bytes intact."""
    path = Path(canonical_path)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != before:
        raise P51Blocked("capability_registry_changed_during_run")
    proposed = propose_capability_registry(before, results,
                                           real_backend=real_backend)
    _write_json(path.parent / "proposed_registry_after.json", proposed)
    if not real_backend:
        return {"committed": False, "reason_code": "mock_backend_only",
                "proposed": proposed}
    case_ids = {row.get("case_id") for row in results}
    failures = {"submit_failed", "poll_failed", "download_failed",
                "backend_failed", "evaluation_failed", "transport_failed",
                "request_delivery_mismatch"}
    if len(results) != 8 or case_ids != {f"S{index}" for index in range(8)} or any(
            row.get("status") in failures or row.get("mock_http") is True or
            not row.get("request_sha256") or
            row.get("transport", {}).get("backend_transport_ok") is not True
            for row in results):
        return {"committed": False, "reason_code": "capability_run_incomplete",
                "proposed": proposed}
    _write_json(path, proposed)
    return {"committed": True, "proposed": proposed}


def verify_capability_run_identity(path: Path, *, expected_code_sha: str,
                                   plan: dict[str, Any]) -> dict[str, Any]:
    """Bind staged S0–S7 results to one checkout, fixture set and endpoint set."""
    identity = {
        "schema_version": "p51_capability_run_identity_v1",
        "expected_code_sha": expected_code_sha,
        "cases": [{"case_id": row["case_id"],
                   "request_sha256": row["manifest"]["request_sha256"],
                   "backend": row["manifest"]["backend"],
                   "asset_sha256": [asset["sha256"] for asset in row["assets"]]}
                  for row in plan["cases"]],
    }
    path = Path(path)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise P51Blocked("capability_run_identity_invalid") from exc
        if previous != identity:
            raise P51Blocked("capability_run_identity_changed")
    else:
        _write_json(path, identity)
    return identity


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, check=False)


def _status_paths(porcelain_z: bytes) -> tuple[list[str], list[str]]:
    tracked, untracked = [], []
    entries = iter(porcelain_z.split(b"\0"))
    for entry in entries:
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise P51Blocked("git_status_porcelain_invalid", "infrastructure")
        path = entry[3:].decode("utf-8", errors="surrogateescape")
        (untracked if entry[:2] == b"??" else tracked).append(path)
        if b"R" in entry[:2] or b"C" in entry[:2]:
            original = next(entries, b"")
            if not original:
                raise P51Blocked("git_status_porcelain_invalid", "infrastructure")
            tracked.append(original.decode("utf-8", errors="surrogateescape"))
    return tracked, untracked


def check_server_sync_safety(local_repo: Path, *, expected_sha: str,
                             remote_head: str, remote_status_z: bytes) \
        -> dict[str, Any]:
    """Read-only comparison; preserve non-conflicting untracked files."""
    repo = Path(local_repo)
    tracked, untracked = _status_paths(remote_status_z)
    base = {"expected_sha": expected_sha, "remote_head": remote_head,
            "protected_untracked_files": untracked,
            "tracked_changes": tracked, "pull_executed": False}
    if tracked:
        return {**base, "status": "BLOCKED_DIRTY_TRACKED"}
    for revision in (remote_head, expected_sha):
        if _git(repo, "cat-file", "-e", f"{revision}^{{commit}}").returncode:
            return {**base, "status": "DIVERGED",
                    "reason_code": "commit_not_available_locally"}
    if remote_head == expected_sha:
        return {**base, "status": "SAFE_TO_FF_PULL", "incoming_paths": []}
    if _git(repo, "merge-base", "--is-ancestor", remote_head,
            expected_sha).returncode:
        return {**base, "status": "DIVERGED"}
    changed = _git(repo, "diff", "--name-only", "-z", "--diff-filter=ACMR",
                   remote_head, expected_sha)
    if changed.returncode:
        raise P51Blocked("incoming_file_diff_failed", "infrastructure")
    incoming = [value.decode("utf-8", errors="surrogateescape")
                for value in changed.stdout.split(b"\0") if value]
    conflicts = sorted(set(incoming) & set(untracked))
    return {**base, "status": ("BLOCKED_UNTRACKED_CONFLICT" if conflicts else
                               "BEHIND_SAFE"),
            "incoming_paths": incoming, "conflicting_untracked_files": conflicts}


def query_server_sync_plan(local_repo: Path, *, expected_sha: str,
                           ssh_target: str, server_root: str) -> dict[str, Any]:
    """Read only remote HEAD/status; never fetch, pull, clean, or reset."""
    remote_root = shlex.quote(server_root)
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            ssh_target]
    head = subprocess.run(base + [f"git -C {remote_root} rev-parse HEAD"],
                          capture_output=True, check=False)
    status = subprocess.run(base + [
        f"git -C {remote_root} status --porcelain=v1 -z --untracked-files=all"],
        capture_output=True, check=False)
    if head.returncode or status.returncode:
        raise P51Blocked("server_git_read_failed", "infrastructure")
    plan = check_server_sync_safety(
        local_repo, expected_sha=expected_sha,
        remote_head=head.stdout.decode("ascii").strip(),
        remote_status_z=status.stdout)
    return {**plan, "ssh_target": ssh_target, "server_root": server_root}


def check_runtime_checkout(repo: Path, expected_sha: str) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain=v1", "-z",
                  "--untracked-files=all")
    if head.returncode or status.returncode:
        raise P51Blocked("runtime_git_read_failed", "infrastructure")
    plan = check_server_sync_safety(
        repo, expected_sha=expected_sha,
        remote_head=head.stdout.decode("ascii").strip(),
        remote_status_z=status.stdout)
    if plan["status"] != "SAFE_TO_FF_PULL":
        raise P51Blocked("server_sync_not_safe")
    return plan


def evaluate_gpu_preflight(gpu_csv: str, apps_csv: str,
                           required_indices: list[int], *,
                           owned_service_pids: set[int] | None = None) \
        -> dict[str, Any]:
    """A free card needs low utilization, low memory and no foreign app."""
    allowed = owned_service_pids or set()
    gpus = {}
    for row in csv.reader(gpu_csv.splitlines()):
        if not row:
            continue
        index, uuid, total, used, utilization = [value.strip() for value in row[:5]]
        gpus[int(index)] = {"index": int(index), "uuid": uuid,
                            "memory_total_mib": int(total),
                            "memory_used_mib": int(used),
                            "utilization_percent": int(utilization), "processes": []}
    for row in csv.reader(apps_csv.splitlines()):
        if not row:
            continue
        uuid, pid, name, used = [value.strip() for value in row[:4]]
        for gpu in gpus.values():
            if gpu["uuid"] == uuid:
                gpu["processes"].append({"pid": int(pid), "name": name,
                                         "used_memory_mib": int(used)})
    if not required_indices or any(index not in gpus for index in required_indices):
        raise P51Blocked("required_gpu_set_invalid", "infrastructure")
    eligible = []
    for index in required_indices:
        gpu = gpus[index]
        pids = {row["pid"] for row in gpu["processes"]}
        own_service_only = bool(pids) and pids <= allowed
        free = (not pids and gpu["memory_used_mib"] <= 1000 and
                gpu["utilization_percent"] <= 5)
        if free or own_service_only:
            eligible.append(index)
    return {"gpu_count": len(gpus), "gpus": list(gpus.values()),
            "required_indices": required_indices,
            "eligible_gpu_sets": [required_indices] if len(eligible) == len(
                required_indices) else [],
            "status": "READY" if len(eligible) == len(required_indices) else
            "BLOCKED_GPU_BUSY"}


def _verify_owned_service_pids(pids: set[int]) -> None:
    if not pids:
        return
    if not hasattr(os, "getuid"):
        raise P51Blocked("owned_service_identity_unavailable", "infrastructure")
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            uid = int(next(line.split()[1] for line in status.splitlines()
                           if line.startswith("Uid:")))
        except (OSError, StopIteration, ValueError) as exc:
            raise P51Blocked("owned_service_identity_unavailable",
                             "infrastructure") from exc
        if uid != os.getuid() or b"sglang" not in command.lower():
            raise P51Blocked("owned_service_identity_mismatch", "infrastructure")


def probe_gpu_preflight(required_indices: list[int], *,
                        owned_service_pids: set[int] | None = None) -> dict[str, Any]:
    _verify_owned_service_pids(owned_service_pids or set())
    gpu = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    apps = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=False)
    if gpu.returncode or apps.returncode:
        raise P51Blocked("nvidia_smi_unavailable", "infrastructure")
    return evaluate_gpu_preflight(
        gpu.stdout, apps.stdout, required_indices,
        owned_service_pids=owned_service_pids)


def require_execute_gates(v9_output: Path, *, expected_code_sha: str,
                          repo: Path, required_gpus: list[int],
                          owned_service_pids: set[int] | None = None) \
        -> dict[str, Any]:
    """P0 freeze, exact checkout and live GPU check all precede HTTP submit."""
    root = Path(v9_output)
    try:
        frozen = json.loads((root / "frozen_program_hashes.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        raise P51Blocked("p0_programs_not_frozen") from exc
    if not verify_frozen_program_snapshot(
            root, {"v9_program_hashes": frozen.get("program_sha256")}):
        raise P51Blocked("p0_programs_not_frozen")
    checkout = check_runtime_checkout(repo, expected_code_sha)
    gpu = probe_gpu_preflight(required_gpus,
                              owned_service_pids=owned_service_pids)
    if gpu["status"] != "READY":
        raise P51Blocked("BLOCKED_GPU_BUSY", "infrastructure")
    return {"passed": True, "p0_frozen": True, "sync": checkout, "gpu": gpu}
