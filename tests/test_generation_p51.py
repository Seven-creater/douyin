"""P5.1 verifies the official JSON transport without H3, Omni or GPU."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from src.agentic_video.generation_p4 import (
    build_h3_capability_smoke_plan, new_capability_registry,
    register_condition_asset,
)
from src.agentic_video.generation_p51 import (
    P51Blocked, audit_real_h3_payload, check_server_sync_safety,
    evaluate_gpu_preflight, finalize_capability_registry,
    propose_capability_registry, require_execute_gates,
    run_h3_capability_case, verify_capability_run_identity,
)
from src.agentic_video.generation_v9g import SGLangH3Client


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    root = tmp_path_factory.mktemp("p51_media")
    for name, color in (("first.png", "red"), ("last.png", "blue"),
                        ("reference.png", "green"), ("second.png", "yellow")):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s=64x64:r=1:d=1", "-frames:v", "1",
            str(root / name)], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=white:s=64x64:r=24:d=4", "-f", "lavfi", "-i",
        "anullsrc=r=32000:cl=stereo", "-t", "4", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(root / "reference.mp4"),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=32000:duration=4",
        str(root / "reference.wav"),
    ], check=True)
    return root


class _MockHTTP:
    def __init__(self, mode: str, video: Path) -> None:
        self.mode = mode
        self.video = video
        self.received: list[dict] = []
        self.body_sha256: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def _json(self, value: dict) -> None:
                body = json.dumps(value).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                assert self.path == "/v1/videos"
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.body_sha256.append(hashlib.sha256(body).hexdigest())
                payload = json.loads(body)
                owner.received.append(payload)
                conditions = deepcopy(payload.get("conditions") or [])
                if owner.mode == "drop_video":
                    conditions = [row for row in conditions if
                                  row.get("type") != "video"]
                if owner.mode == "reorder_image":
                    conditions[:2] = conditions[1::-1]
                self._json({"id": "mock_job_1", "accepted_conditions": conditions})

            def do_GET(self) -> None:
                if self.path == "/v1/videos/mock_job_1":
                    status = ("running" if owner.mode == "timeout" else
                              "failed" if owner.mode == "backend_failed" else
                              "completed")
                    self._json({"status": status})
                    return
                assert self.path == "/v1/videos/mock_job_1/content"
                if owner.mode == "download_failed":
                    self.send_error(503)
                    return
                body = b"not-an-mp4" if owner.mode == "broken" else \
                    owner.video.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _MockHTTP:
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"


def _case(media: Path, endpoint: str, case_id: str) -> dict:
    plan = build_h3_capability_smoke_plan(
        fixture_dir=media,
        backend_endpoints={"fl2va": endpoint, "ref2va": endpoint})
    return next(row for row in plan["cases"] if row["case_id"] == case_id)


def test_official_json_payload_preserves_order_sha_and_hybrid(
        media: Path) -> None:
    case = _case(media, "http://127.0.0.1:30011", "S7")
    payload = SGLangH3Client.serialize_payload(case["request"])
    audit = audit_real_h3_payload(case, payload)
    assert payload["task"] == "ref2va"
    assert [(row["role"], row.get("frame_index")) for row in
            payload["conditions"]] == [("reference", None), ("keyframe", 0)]
    assert [row["label"] for row in audit["sent_conditions"]] == [
        "Picture 1", "First Frame"]
    assert all(row["sha256"] == hashlib.sha256(
        Path(case["assets"][index]["path"]).read_bytes()).hexdigest()
        for index, row in enumerate(audit["sent_conditions"]))
    assert audit["protocol"] == "sglang_v1_videos_json"


def test_payload_drift_variant_and_legacy_block(media: Path) -> None:
    case = _case(media, "http://127.0.0.1:30011", "S5")
    payload = SGLangH3Client.serialize_payload(case["request"])
    changed = deepcopy(payload)
    changed["conditions"] = []
    with pytest.raises(P51Blocked, match="request_payload_mismatch"):
        audit_real_h3_payload(case, changed)
    changed = deepcopy(payload)
    changed["task"] = "t2va"
    with pytest.raises(P51Blocked, match="request_payload_mismatch"):
        audit_real_h3_payload(case, changed)
    legacy = deepcopy(case)
    legacy["manifest"]["backend"]["type"] = "legacy_minimax"
    with pytest.raises(P51Blocked, match="H3_BACKEND_UNAVAILABLE"):
        audit_real_h3_payload(legacy, payload)
    registry = new_capability_registry("local_h3_sglang")
    assert propose_capability_registry(registry, [], real_backend=False) == registry


def test_mock_http_normal_transport_not_effective(media: Path,
                                                  tmp_path: Path) -> None:
    with _MockHTTP("normal", media / "reference.mp4") as mock:
        case = _case(media, mock.endpoint, "S7")
        client = SGLangH3Client(mock.endpoint, timeout_s=1, poll_s=0.001)
        result = run_h3_capability_case(
            case, client, tmp_path / "S7", mock_http=True, short_edge=64,
            effect_checks=[{"source": "mock", "name": "echo", "passed": True,
                            "evidence": "receipt"}])
    assert len(mock.received) == 1
    assert mock.received[0]["conditions"] == case["request"]["conditions"]
    receipt = json.loads((tmp_path / "S7" / "client_payload_receipt.json")
                         .read_text(encoding="utf-8"))
    assert mock.body_sha256 == [receipt["http_body_sha256"]]
    assert result["transport"]["backend_transport_ok"] is True
    assert result["status"] == "unverified"
    assert result["backend_receipt"]["status"] == "matched"
    assert (tmp_path / "S7" / "backend_result_manifest.json").is_file()
    assert (tmp_path / "S7" / "job_state.json").is_file()


@pytest.mark.parametrize("mode,case_id,expected", [
    ("drop_video", "S5", "request_delivery_mismatch"),
    ("reorder_image", "S8", "request_delivery_mismatch"),
    ("broken", "S0", "transport_failed"),
    ("timeout", "S0", "poll_failed"),
    ("backend_failed", "S0", "backend_failed"),
    ("download_failed", "S0", "download_failed"),
])
def test_mock_http_failure_modes(media: Path, tmp_path: Path,
                                 mode: str, case_id: str,
                                 expected: str) -> None:
    with _MockHTTP(mode, media / "reference.mp4") as mock:
        if case_id == "S8":
            case = _case(media, mock.endpoint, "S4")
            case = deepcopy(case)
            second = register_condition_asset(
                asset_id="second", kind="image", path=media / "second.png",
                source={"type": "smoke_fixture"},
                allowed_roles=["identity_reference"])
            case["assets"].append(second)
            # Build a proper two-image P4 request for the reordering adversary.
            from src.agentic_video.generation_p4 import build_h3_condition_request

            case.update(build_h3_condition_request(
                request_id="S8", unit_id="SMOKE.S8",
                prompt="<Picture 1> [smoke_identity] <Picture 2> [second]",
                duration_s=4, assets=case["assets"], bindings=[
                    {"asset_id": "smoke_identity", "request_label": "Picture 1",
                     "semantic_role": "identity_reference", "required": True},
                    {"asset_id": "second", "request_label": "Picture 2",
                     "semantic_role": "identity_reference", "required": True},
                ], backend={"backend_id": "local_h3_sglang", "type": "sglang_h3",
                            "protocol": "sglang_v1_videos_json",
                            "model_variant": "ref2va", "endpoint": mock.endpoint}))
            case["case_id"] = "S8"
        else:
            case = _case(media, mock.endpoint, case_id)
        client = SGLangH3Client(mock.endpoint, timeout_s=0.03, poll_s=0.001)
        result = run_h3_capability_case(
            case, client, tmp_path / mode, mock_http=True, short_edge=64)
    assert result["status"] == expected
    assert result.get("failure_class") != "conditioning_ineffective"
    assert len(mock.received) == 1


def test_atomic_registry_requires_complete_real_run(tmp_path: Path) -> None:
    before = new_capability_registry("local_h3_sglang")
    registry_path = tmp_path / "backend_registry.json"
    registry_path.write_text(json.dumps(before), encoding="utf-8")
    original = registry_path.read_bytes()
    incomplete = finalize_capability_registry(
        registry_path, before, [{"case_id": "S0", "status": "poll_failed"}],
        real_backend=True)
    assert incomplete["committed"] is False
    assert registry_path.read_bytes() == original
    mock = finalize_capability_registry(
        registry_path, before, [{"case_id": f"S{i}", "status": "unverified",
                                 "transport": {"backend_transport_ok": True}}
                                for i in range(8)], real_backend=False)
    assert mock["committed"] is False
    assert registry_path.read_bytes() == original


def test_mock_results_cannot_promote_registry_even_when_marked_real(
        tmp_path: Path) -> None:
    before = new_capability_registry("local_h3_sglang")
    path = tmp_path / "backend_registry.json"
    path.write_text(json.dumps(before), encoding="utf-8")
    results = [{"case_id": f"S{i}", "status": "effective",
                "mock_http": True, "request_sha256": "fake",
                "effect": {"checks": [{"source": "human", "passed": True,
                                       "evidence": "fake"}]}}
               for i in range(8)]
    finalized = finalize_capability_registry(
        path, before, results, real_backend=True)
    assert finalized["committed"] is False
    assert json.loads(path.read_text(encoding="utf-8")) == before


def test_duplicate_case_cannot_commit_registry(tmp_path: Path) -> None:
    before = new_capability_registry("local_h3_sglang")
    path = tmp_path / "backend_registry.json"
    results = [{"case_id": f"S{i}", "status": "unverified"}
               for i in range(8)]
    results.append(deepcopy(results[0]))
    assert finalize_capability_registry(
        path, before, results, real_backend=True)["committed"] is False
    assert not path.exists()


def test_complete_registry_requires_transport_evidence(tmp_path: Path) -> None:
    before = new_capability_registry("local_h3_sglang")
    path = tmp_path / "backend_registry.json"
    results = [{"case_id": f"S{i}", "status": "unverified",
                "request_sha256": f"sha-{i}"} for i in range(8)]
    assert finalize_capability_registry(
        path, before, results, real_backend=True)["committed"] is False
    assert not path.exists()


def test_staged_results_reject_changed_fixtures_or_checkout(
        media: Path, tmp_path: Path) -> None:
    plan = build_h3_capability_smoke_plan(
        fixture_dir=media,
        backend_endpoints={"fl2va": "http://127.0.0.1:30010",
                           "ref2va": "http://127.0.0.1:30011"})
    path = tmp_path / "capability_run_identity.json"
    first = verify_capability_run_identity(
        path, expected_code_sha="commit-a", plan=plan)
    assert verify_capability_run_identity(
        path, expected_code_sha="commit-a", plan=plan) == first
    with pytest.raises(P51Blocked, match="capability_run_identity_changed"):
        verify_capability_run_identity(
            path, expected_code_sha="commit-b", plan=plan)
    changed = deepcopy(plan)
    changed["cases"][4]["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(P51Blocked, match="capability_run_identity_changed"):
        verify_capability_run_identity(
            path, expected_code_sha="commit-a", plan=changed)


def test_asset_drift_blocks_before_http(media: Path) -> None:
    case = _case(media, "http://127.0.0.1:30011", "S4")
    changed = deepcopy(case)
    changed["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(P51Blocked, match="BLOCKED_ASSET_DRIFT"):
        audit_real_h3_payload(
            changed, SGLangH3Client.serialize_payload(changed["request"]))


def test_production_unverified_capability_blocks(media: Path) -> None:
    case = _case(media, "http://127.0.0.1:30011", "S7")
    with pytest.raises(P51Blocked, match="production_capability_not_effective"):
        audit_real_h3_payload(
            case, SGLangH3Client.serialize_payload(case["request"]),
            production=True, registry=new_capability_registry("local_h3_sglang"))


def test_dry_run_serializes_all_cases_without_network(
        media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agentic_video.cli import _reference_generate_v9g, build_parser

    def no_network(*_args, **_kwargs):
        raise AssertionError("dry-run sent a network request")

    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.setattr(requests, "get", no_network)
    args = build_parser().parse_args([
        "reference-generate-v9g", "--phase", "capability",
        "--dry-run-real-backend", "--fixture-dir", str(media),
        "--fl2va-endpoint", "http://127.0.0.1:30010",
        "--ref2va-endpoint", "http://127.0.0.1:30011",
        "--output", str(tmp_path),
    ])
    result = _reference_generate_v9g(args, None)
    assert result["http_requests_sent"] == 0
    assert result["case_count"] == 8
    assert (tmp_path / "capability" / "requests" / "S7" / "request.json").is_file()
    registry = json.loads((tmp_path / "capability" / "backend_registry.json")
                          .read_text(encoding="utf-8"))
    assert set(registry["capabilities"].values()) == {"unverified"}


def test_p0_unfrozen_blocks_before_git_or_gpu(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agentic_video import generation_p51

    monkeypatch.setattr(generation_p51, "check_runtime_checkout",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("git gate reached")))
    with pytest.raises(P51Blocked, match="p0_programs_not_frozen"):
        require_execute_gates(
            tmp_path, expected_code_sha="abc", repo=tmp_path,
            required_gpus=[0])


def test_gpu_busy_blocks_execute_gate_before_backend(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agentic_video import generation_p51

    (tmp_path / "frozen_program_hashes.json").write_text(
        json.dumps({"program_sha256": {}}), encoding="utf-8")
    monkeypatch.setattr(generation_p51, "verify_frozen_program_snapshot",
                        lambda *_args, **_kwargs: True)
    monkeypatch.setattr(generation_p51, "check_runtime_checkout",
                        lambda *_args, **_kwargs: {"status": "SAFE_TO_FF_PULL"})
    monkeypatch.setattr(generation_p51, "probe_gpu_preflight",
                        lambda *_args, **_kwargs: {"status": "BLOCKED_GPU_BUSY"})
    with pytest.raises(P51Blocked, match="BLOCKED_GPU_BUSY"):
        require_execute_gates(
            tmp_path, expected_code_sha="abc", repo=tmp_path,
            required_gpus=[0])


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def test_untracked_conflict_and_preservation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "first.txt").write_text("first", encoding="utf-8")
    _git(repo, "add", "first.txt")
    _git(repo, "commit", "-m", "base")
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "incoming.txt").write_text("new", encoding="utf-8")
    _git(repo, "add", "incoming.txt")
    _git(repo, "commit", "-m", "incoming")
    new = _git(repo, "rev-parse", "HEAD")
    conflict = check_server_sync_safety(
        repo, expected_sha=new, remote_head=old,
        remote_status_z=b"?? incoming.txt\0?? protected.txt\0")
    assert conflict["status"] == "BLOCKED_UNTRACKED_CONFLICT"
    assert conflict["conflicting_untracked_files"] == ["incoming.txt"]
    safe = check_server_sync_safety(
        repo, expected_sha=new, remote_head=old,
        remote_status_z=b"?? protected.txt\0")
    assert safe["status"] == "BEHIND_SAFE"
    assert safe["protected_untracked_files"] == ["protected.txt"]
    assert safe["pull_executed"] is False


def test_tracked_changes_and_divergence_block_sync(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "file.txt").write_text("base", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "base")
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "file.txt").write_text("new", encoding="utf-8")
    _git(repo, "commit", "-am", "new")
    new = _git(repo, "rev-parse", "HEAD")
    dirty = check_server_sync_safety(
        repo, expected_sha=new, remote_head=old,
        remote_status_z=b" M file.txt\0")
    assert dirty["status"] == "BLOCKED_DIRTY_TRACKED"
    diverged = check_server_sync_safety(
        repo, expected_sha=old, remote_head=new, remote_status_z=b"")
    assert diverged["status"] == "DIVERGED"


def test_gpu_busy_requires_no_process_and_low_utilization() -> None:
    gpu = ("0, GPU-a, 49140, 21208, 100\n"
           "1, GPU-b, 49140, 100, 0\n")
    apps = "GPU-a, 3842135, python, 21196\n"
    blocked = evaluate_gpu_preflight(gpu, apps, [0, 1])
    assert blocked["status"] == "BLOCKED_GPU_BUSY"
    assert blocked["eligible_gpu_sets"] == []
    ready = evaluate_gpu_preflight(gpu, apps, [1])
    assert ready["status"] == "READY"
