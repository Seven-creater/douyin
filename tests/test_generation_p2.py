"""P2 CPU-only contract-to-condition compilation tests."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.agentic_video.generation_p1 import compile_p1_contract_draft
from src.agentic_video.generation_p2 import (
    P2Blocked,
    build_condition_asset_catalog,
    compile_h3_context_preview,
    preview_h3_route,
)


FIXTURE = Path(__file__).parent / "fixtures" / "v9g_p1_fixed.json"


def _draft() -> dict:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return compile_p1_contract_draft(data["requirements"], data["target_setting"],
                                     data["action_plans"])


def _entry(tmp_path: Path, asset_id: str, namespace: str, media_type: str,
           allowed: dict[str, str], forbidden: list[str] | None = None) -> dict:
    path = tmp_path / f"{asset_id}.bin"
    path.write_bytes(asset_id.encode("ascii"))
    return {
        "asset_id": asset_id, "namespace": namespace, "media_type": media_type,
        "path": str(path), "source": "fixed CPU fixture", "source_interval": [0.0, 1.0],
        "allowed_transfer": list(allowed), "forbidden_transfer": forbidden or [],
        "transfer_descriptors": allowed,
    }


def _catalog(tmp_path: Path) -> dict:
    entries = [
        _entry(tmp_path, "C0_front", "target_assets", "image",
               {"identity": "C0", "appearance": "blue clothing"}),
        _entry(tmp_path, "C1_front", "target_assets", "image",
               {"identity": "C1"}),
        _entry(tmp_path, "arena", "target_assets", "image",
               {"scene": "arena"}),
    ]
    return build_condition_asset_catalog(entries)


def test_catalog_and_context_are_deterministic_and_non_executable(tmp_path: Path) -> None:
    draft, catalog = _draft(), _catalog(tmp_path)
    first = compile_h3_context_preview(draft, catalog, unit_id="S2.G1")
    second = compile_h3_context_preview(draft, catalog, unit_id="S2.G1")
    assert first == second
    assert first["execution_authorized"] is False
    assert list(first["prompt_fields"]) == [
        "subject_definitions", "summary", "retention_analysis",
        "detailed_description", "overall_soundscape", "non_diegetic_music",
    ]
    assert "C0" in first["prompt"]
    assert "arena" in first["prompt"]
    assert "champion" not in first["prompt"].lower()
    assert "not specified by contract" in first["prompt"]


def test_reference_identity_transfer_is_forbidden(tmp_path: Path) -> None:
    entry = _entry(tmp_path, "ref_action", "reference_evidence", "video",
                   {"identity": "reference person"})
    with pytest.raises(P2Blocked, match="reference_identity_transfer_forbidden"):
        build_condition_asset_catalog([entry])


def test_reference_motion_uses_only_permitted_descriptor(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    ref = _entry(tmp_path, "ref_action", "reference_evidence", "video",
                 {"motion": "forward kick"}, ["identity", "appearance"])
    entries = []
    for asset in catalog["assets"]:
        entries.append({**asset})
    entries.append(ref)
    new_catalog = build_condition_asset_catalog(entries)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["action_plans"]["S2"][0]["condition_keys"].append("ref_action")
    draft = compile_p1_contract_draft(**fixture)
    context = compile_h3_context_preview(draft, new_catalog, unit_id="S2.G1")
    assert "forward kick" in context["prompt"]
    assert "reference person" not in context["prompt"]
    assert any(row["asset_id"] == "ref_action" for row in context["references"])


def test_catalog_fails_closed_when_asset_changes(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    Path(catalog["assets"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(P2Blocked, match="asset_hash_mismatch"):
        compile_h3_context_preview(_draft(), catalog, unit_id="S2.G1")


def test_uncommitted_memory_cannot_be_selected(tmp_path: Path) -> None:
    entry = _entry(tmp_path, "memory", "committed_memory", "image",
                   {"continuity": "last accepted position"})
    with pytest.raises(P2Blocked, match="uncommitted_memory_forbidden"):
        build_condition_asset_catalog([entry])


def test_route_preview_preserves_reference_and_keyframe_or_requires_split(
        tmp_path: Path) -> None:
    context = compile_h3_context_preview(_draft(), _catalog(tmp_path), unit_id="S2.G1")
    caps = {"ref2va_image": True, "ref2va_hybrid_first": False}
    route = preview_h3_route(context, caps, first_frame="first.jpg")
    assert route["execution_authorized"] is False
    assert route["route"]["split_required"] is True
    assert route["route"]["preserve"] == ["references", "keyframes"]
    caps["ref2va_hybrid_first"] = True
    assert preview_h3_route(context, caps, first_frame="first.jpg")["route"]["mode"] == "ref2va"
    with pytest.raises(P2Blocked, match="capability_manifest_required"):
        preview_h3_route(context, {})


def test_modified_draft_and_context_cannot_be_compiled_or_routed(tmp_path: Path) -> None:
    draft, catalog = _draft(), _catalog(tmp_path)
    edited = deepcopy(draft)
    edited["sections"][0]["semantic_goal"] = "invented win"
    with pytest.raises(P2Blocked, match="p1_draft_hash_mismatch"):
        compile_h3_context_preview(edited, catalog, unit_id="S2.G1")
    context = compile_h3_context_preview(draft, catalog, unit_id="S2.G1")
    context["prompt"] += " new character"
    with pytest.raises(P2Blocked, match="context_hash_mismatch"):
        preview_h3_route(context, {"ref2va_image": True})
