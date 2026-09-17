"""P4 condition integrity is separate from media transport and model effect."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from src.agentic_video.generation_p4 import (
    FakeH3ConditionBackend, P4Blocked, build_h3_capability_smoke_plan,
    build_h3_condition_request, new_capability_registry, register_condition_asset,
    require_effective_capabilities, run_fake_h3_smoke, plan_h3_request_from_context,
    validate_formal_h3_job, validate_h3_condition_integrity,
    verify_frozen_program_snapshot,
)


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    root = tmp_path_factory.mktemp("p4_media")
    for name, color in (("first.png", "red"), ("last.png", "blue"),
                        ("reference.png", "green"), ("second.png", "yellow")):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s=64x64:r=1:d=1", "-frames:v", "1", str(root / name),
        ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=white:s=64x64:r=24:d=4", "-f", "lavfi", "-i",
        "anullsrc=r=32000:cl=stereo", "-t", "4", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(root / "reference.mp4"),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=32000:duration=4", str(root / "reference.wav"),
    ], check=True)
    return root


def _asset(media: Path, asset_id: str, filename: str, role: str, *,
           source: dict | None = None) -> dict:
    kind = "video" if filename.endswith(".mp4") else \
        "audio" if filename.endswith(".wav") else "image"
    return register_condition_asset(
        asset_id=asset_id, kind=kind, path=media / filename,
        source=source or {"type": "smoke_fixture"}, allowed_roles=[role])


def _plan(media: Path, assets: list[dict], labels: list[str], roles: list[str]) -> dict:
    prompt = "subject_definitions: " + " ".join(
        f"<{label}> [{asset['asset_id']}]" for label, asset, role in
        zip(labels, assets, roles) if role not in {"first_frame", "last_frame"})
    variant = "ref2va" if any(role not in {"first_frame", "last_frame"}
                              for role in roles) else "fl2va"
    return build_h3_condition_request(
        request_id="case", unit_id="U1", prompt=prompt, duration_s=4.0,
        assets=assets, bindings=[{
            "condition_id": f"cond_{index}", "asset_id": asset["asset_id"],
            "request_label": label, "semantic_role": role, "required": True,
        } for index, (asset, label, role) in enumerate(zip(assets, labels, roles), 1)],
        backend={"backend_id": "fake_h3", "model_variant": variant})


def _rehash(plan: dict) -> None:
    plan["manifest"]["request_sha256"] = _hash(plan["request"])
    plan["manifest"]["manifest_sha256"] = _hash({
        key: value for key, value in plan["manifest"].items()
        if key != "manifest_sha256"})


def test_plan_only_keeps_all_capabilities_unverified() -> None:
    plan = build_h3_capability_smoke_plan()
    assert len(plan["cases"]) == 8
    assert plan["fixtures_ready"] is False
    assert {row["case_id"] for row in plan["cases"]} == {
        f"S{index}" for index in range(8)}
    assert set(plan["backend_registry"]["capabilities"].values()) == {"unverified"}
    assert all(row["execution_status"] == "planned_only" for row in plan["cases"])


def test_real_fixture_plan_has_eight_integral_requests(media: Path) -> None:
    plan = build_h3_capability_smoke_plan(fixture_dir=media)
    assert plan["fixtures_ready"] is True
    for case in plan["cases"]:
        result = validate_h3_condition_integrity(
            case["request"], case["manifest"], case["assets"])
        assert result["passed"]
        assert case["manifest"]["execution_authorized"] is False


def test_prompt_video_without_request_video_blocks(media: Path) -> None:
    plan = _plan(media, [], [], [])
    plan["request"]["prompt"] = "subject_definitions: <Video 1> [motion]"
    plan["manifest"]["prompt_sha256"] = hashlib.sha256(
        plan["request"]["prompt"].encode()).hexdigest()
    _rehash(plan)
    with pytest.raises(P4Blocked, match="prompt_request_label_mismatch"):
        validate_h3_condition_integrity(plan["request"], plan["manifest"], [])


def test_unmentioned_picture_and_swapped_pictures_block(media: Path) -> None:
    a = _asset(media, "a", "reference.png", "identity_reference")
    b = _asset(media, "b", "second.png", "identity_reference")
    plan = _plan(media, [a, b], ["Picture 1", "Picture 2"],
                 ["identity_reference", "identity_reference"])
    plan["request"]["prompt"] = "subject_definitions: <Picture 1> [a]"
    plan["manifest"]["prompt_sha256"] = hashlib.sha256(
        plan["request"]["prompt"].encode()).hexdigest()
    _rehash(plan)
    with pytest.raises(P4Blocked, match="prompt_request_label_mismatch"):
        validate_h3_condition_integrity(plan["request"], plan["manifest"], [a, b])
    correct = _plan(media, [a, b], ["Picture 1", "Picture 2"],
                    ["identity_reference", "identity_reference"])
    correct["request"]["conditions"].reverse()
    _rehash(correct)
    with pytest.raises(P4Blocked, match="condition_order_or_asset_mismatch"):
        validate_h3_condition_integrity(correct["request"], correct["manifest"], [a, b])


def test_asset_sha_role_and_memory_provenance_block(media: Path) -> None:
    ref = _asset(media, "ref", "reference.png", "motion_reference",
                 source={"type": "reference_evidence",
                         "reference_video_sha256": "a" * 64,
                         "source_interval": [5.0, 6.0]})
    with pytest.raises(P4Blocked, match="condition_role_forbidden"):
        _plan(media, [ref], ["Picture 1"], ["identity_reference"])
    image = _asset(media, "image", "reference.png", "identity_reference")
    plan = _plan(media, [image], ["Picture 1"], ["identity_reference"])
    changed = deepcopy(image)
    changed["sha256"] = "0" * 64
    with pytest.raises(P4Blocked, match="asset_sha_mismatch"):
        validate_h3_condition_integrity(plan["request"], plan["manifest"], [changed])
    memory = _asset(media, "memory", "first.png", "first_frame", source={
        "type": "committed_memory", "unit_id": "U0", "snippet_id": "SNP0",
        "state_commit_id": "SC0", "endpoint_kind": "raw_unit_last_frame",
        "accepted_snippet_endpoint_s": 2.7,
    })
    memory_plan = _plan(media, [memory], ["First Frame"], ["first_frame"])
    with pytest.raises(P4Blocked, match="committed_memory_not_accepted"):
        validate_h3_condition_integrity(
            memory_plan["request"], memory_plan["manifest"], [memory],
            accepted_commits={"SC0": {
                "status": "accepted", "unit_id": "U0", "snippet_id": "SNP0",
                "accepted_snippet_endpoint_s": 2.7,
                "endpoint_asset_sha256": memory["sha256"]}})
    accepted_memory = deepcopy(memory)
    accepted_memory["source"]["endpoint_kind"] = "accepted_snippet_endpoint"
    accepted_plan = _plan(media, [accepted_memory], ["First Frame"], ["first_frame"])
    with pytest.raises(P4Blocked, match="committed_memory_not_accepted"):
        validate_h3_condition_integrity(
            accepted_plan["request"], accepted_plan["manifest"], [accepted_memory],
            accepted_commits={"SC0": {
                "status": "rejected", "unit_id": "U0", "snippet_id": "SNP0",
                "accepted_snippet_endpoint_s": 2.7,
                "endpoint_asset_sha256": accepted_memory["sha256"]}})
    assert validate_h3_condition_integrity(
        accepted_plan["request"], accepted_plan["manifest"], [accepted_memory],
        accepted_commits={"SC0": {
            "status": "accepted", "unit_id": "U0", "snippet_id": "SNP0",
            "accepted_snippet_endpoint_s": 2.7,
            "endpoint_asset_sha256": accepted_memory["sha256"]}})["passed"]


def test_fake_drop_video_keeps_transport_but_fails_condition(media: Path,
                                                               tmp_path: Path) -> None:
    motion = _asset(media, "motion", "reference.mp4", "motion_reference")
    plan = _plan(media, [motion], ["Video 1"], ["motion_reference"])
    plan["assets"] = [motion]
    honest = run_fake_h3_smoke(
        plan, FakeH3ConditionBackend(media / "reference.mp4"),
        tmp_path / "honest.mp4", short_edge=64)
    assert honest["transport"]["backend_transport_ok"] is True
    assert honest["effect"]["status"] == "effective"
    assert honest["effect"]["evidence_scope"] == "fake_only"
    assert honest["real_capability_verified"] is False
    dropped = run_fake_h3_smoke(
        plan, FakeH3ConditionBackend(media / "reference.mp4", "drop_video"),
        tmp_path / "dropped.mp4", short_edge=64)
    assert dropped["transport"]["backend_transport_ok"] is True
    assert dropped["effect"]["status"] == "ineffective"
    assert dropped["effect"]["conditioning_effective"] is False


def test_fake_image_reordering_is_detected_after_transport(media: Path,
                                                           tmp_path: Path) -> None:
    a = _asset(media, "a", "reference.png", "identity_reference")
    b = _asset(media, "b", "second.png", "identity_reference")
    plan = _plan(media, [a, b], ["Picture 1", "Picture 2"],
                 ["identity_reference", "identity_reference"])
    plan["assets"] = [a, b]
    result = run_fake_h3_smoke(
        plan, FakeH3ConditionBackend(media / "reference.mp4", "reorder_image"),
        tmp_path / "reordered.mp4", short_edge=64)
    assert result["transport"]["backend_transport_ok"] is True
    assert result["effect"]["status"] == "ineffective"


def test_hybrid_and_unverified_capabilities_never_enable_formal_generation(
        media: Path) -> None:
    plan = build_h3_capability_smoke_plan(fixture_dir=media)
    hybrid = next(row for row in plan["cases"] if row["case_id"] == "S7")
    registry = new_capability_registry("local_h3_sglang")
    registry["status"] = "effective"
    registry["capabilities"]["reference_image"] = "effective"
    with pytest.raises(P4Blocked, match="unsupported_or_unverified_capability"):
        require_effective_capabilities(registry, hybrid["manifest"], p0_frozen=True)
    with pytest.raises(P4Blocked, match="p0_programs_not_frozen"):
        validate_formal_h3_job(
            hybrid["request"], hybrid["manifest"], hybrid["assets"],
            registry, p0_frozen=False)


def test_frozen_program_snapshot_rechecks_bytes(tmp_path: Path) -> None:
    names = ("reference_content_program.json", "reference_edit_program.json",
             "material_requirements.json")
    hashes = {}
    for name in names:
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "acceptance.json").write_text(
        '{"passed":true,"human_passed":true}', encoding="utf-8")
    (tmp_path / "frozen_program_hashes.json").write_text(
        json.dumps({"program_sha256": hashes}), encoding="utf-8")
    assert verify_frozen_program_snapshot(tmp_path, {"v9_program_hashes": hashes})
    (tmp_path / names[0]).write_text('{"changed":true}', encoding="utf-8")
    assert not verify_frozen_program_snapshot(tmp_path, {"v9_program_hashes": hashes})


def test_cli_plan_only_needs_no_gpu_or_license_gate(tmp_path: Path) -> None:
    from src.agentic_video.cli import _reference_generate_v9g, build_parser

    args = build_parser().parse_args([
        "reference-generate-v9g", "--phase", "capability", "--plan-only",
        "--output", str(tmp_path),
    ])
    result = _reference_generate_v9g(args, None)
    assert result["plan_only"] is True
    assert result["real_capability_verified"] is False
    capability = tmp_path / "capability"
    for filename in ("capability_plan.json", "backend_registry.json", "manifest.json"):
        assert (capability / filename).is_file()
    registry = json.loads((capability / "backend_registry.json").read_text(
        encoding="utf-8"))
    assert set(registry["capabilities"].values()) == {"unverified"}


def test_cli_fixture_plan_writes_request_asset_and_manifest(
        media: Path, tmp_path: Path) -> None:
    from src.agentic_video.cli import _reference_generate_v9g, build_parser

    args = build_parser().parse_args([
        "reference-generate-v9g", "--phase", "capability", "--plan-only",
        "--fixture-dir", str(media), "--output", str(tmp_path),
    ])
    result = _reference_generate_v9g(args, None)
    assert result["fixtures_ready"] is True
    requests = tmp_path / "capability" / "requests"
    for case_id in (f"S{index}" for index in range(8)):
        case_dir = requests / case_id
        assert all((case_dir / name).is_file() for name in (
            "request.json", "condition_manifest.json", "assets.json"))
    manifest = json.loads((requests / "S7" / "condition_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["execution_authorized"] is False
    assert manifest["expected_capabilities"] == [
        "ref2va_hybrid", "ref2va_hybrid_first", "reference_image"]


def test_p2_context_to_p4_request_keeps_asset_labels_and_hashes(media: Path) -> None:
    from src.agentic_video.generation_p1 import compile_p1_contract_draft
    from src.agentic_video.generation_p2 import (
        build_condition_asset_catalog, compile_h3_context_preview,
    )

    fixture_path = Path(__file__).parent / "fixtures" / "v9g_p1_fixed.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["action_plans"]["S2"][0]["condition_keys"] = [
        "target_identity", "ref_motion"]
    draft = compile_p1_contract_draft(**fixture)
    p2_entries = []
    for asset_id, filename, namespace, kind, descriptor in (
            ("target_identity", "reference.png", "target_assets", "image", "identity"),
            ("ref_motion", "reference.mp4", "reference_evidence", "video", "motion")):
        p2_entries.append({
            "asset_id": asset_id, "namespace": namespace, "media_type": kind,
            "path": str(media / filename), "source": "fixed CPU fixture",
            "source_interval": [0.0, 1.0], "allowed_transfer": [descriptor],
            "forbidden_transfer": [], "transfer_descriptors": {descriptor: asset_id},
        })
    context = compile_h3_context_preview(
        draft, build_condition_asset_catalog(p2_entries), unit_id="S2.G1")
    assets = [
        _asset(media, "target_identity", "reference.png", "identity_reference",
               source={"type": "target_asset"}),
        _asset(media, "ref_motion", "reference.mp4", "motion_reference", source={
            "type": "reference_evidence", "reference_video_sha256": "a" * 64,
            "source_interval": [0.0, 1.0],
        }),
    ]
    plan = plan_h3_request_from_context(
        context, assets=assets, role_assignments={
            "target_identity": "identity_reference", "ref_motion": "motion_reference"},
        backend={"backend_id": "fake_h3", "model_variant": "ref2va"})
    assert [row["request_label"] for row in plan["manifest"]["conditions"]] == [
        "Picture 1", "Video 1"]
    assert validate_h3_condition_integrity(
        plan["request"], plan["manifest"], assets)["passed"]
    mismatched = deepcopy(assets)
    mismatched[1]["source"]["type"] = "target_asset"
    with pytest.raises(P4Blocked, match="p2_transfer_policy_conflict"):
        plan_h3_request_from_context(
            context, assets=mismatched, role_assignments={
                "target_identity": "identity_reference",
                "ref_motion": "motion_reference"},
            backend={"backend_id": "fake_h3", "model_variant": "ref2va"})
