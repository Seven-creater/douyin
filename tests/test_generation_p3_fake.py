"""P3 fake chain: commit adopted snippet state, not raw tail state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agentic_video.generation_p1 import compile_p1_contract_draft
from src.agentic_video.generation_p2 import build_condition_asset_catalog
from src.agentic_video.generation_p3_fake import (
    FakeRuntimeBlocked, run_fake_progressive_runtime,
)


FIXTURE = Path(__file__).parent / "fixtures" / "v9g_p1_fixed.json"


def _inputs(tmp_path: Path) -> tuple[dict, dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    draft = compile_p1_contract_draft(**data)
    entries = []
    for asset_id, kind in [("C0_front", "identity"), ("C1_front", "identity"),
                           ("arena", "scene")]:
        path = tmp_path / f"{asset_id}.bin"
        path.write_bytes(asset_id.encode("ascii"))
        entries.append({
            "asset_id": asset_id, "namespace": "target_assets", "media_type": "image",
            "path": str(path), "source": "fixed fixture", "source_interval": None,
            "allowed_transfer": [kind], "forbidden_transfer": [],
            "transfer_descriptors": {kind: asset_id},
        })
    return draft, build_condition_asset_catalog(entries)


class FakeChain:
    is_fake_backend = True

    def __init__(self, *, raw_status: str = "PASS", crop_status: str = "PASS") -> None:
        self.raw_status = raw_status
        self.crop_status = crop_status
        self.calls = []

    def generate(self, context: dict, state: dict) -> dict:
        self.calls.append((context["unit_id"], dict(state)))
        return {"unit_id": context["unit_id"], "duration_s": 6.0,
                "raw_tail_state": {"position": "right"}}

    def observe_raw(self, raw: dict) -> dict:
        return {"visible_actions": ["approach", "impact"], "raw_unit_id": raw["unit_id"]}

    def align_phases(self, observed: dict, unit: dict) -> dict:
        return {phase: observed["visible_actions"][0] for phase in unit["semantic_phases"]}

    def verify_contract(self, observed: dict, alignment: dict, unit: dict) -> dict:
        return {"status": self.raw_status}

    def select_moments(self, raw: dict, observed: dict, unit: dict) -> list[dict]:
        return [{"start_s": 0.8, "end_s": 2.7}]

    def review_snippets(self, raw: dict, snippets: list[dict]) -> dict:
        return {"status": self.crop_status}

    def state_at(self, raw: dict, endpoint: float) -> dict:
        return {"pts_s": endpoint,
                "state": {"subject": "C0", "opponent": "C1", "scene": "arena",
                          "event": "E1", "actor_role": "C0_actor", "position": "left"}}


def test_next_unit_uses_accepted_snippet_endpoint_not_raw_tail(tmp_path: Path) -> None:
    draft, catalog = _inputs(tmp_path)
    fake = FakeChain()
    result = run_fake_progressive_runtime(draft, catalog, fake)
    assert result["status"] == "FAKE_CHAIN_PASS"
    assert result["execution_authorized"] is False
    assert len(result["commits"]) == 2
    assert result["commits"][0]["accepted_snippet_endpoint_s"] == 2.7
    assert fake.calls[1][1]["position"] == "left"
    assert fake.calls[1][1]["position"] != "right"


def test_raw_failure_prevents_crop_commit_and_next_unit(tmp_path: Path) -> None:
    draft, catalog = _inputs(tmp_path)
    fake = FakeChain(raw_status="UNKNOWN")
    result = run_fake_progressive_runtime(draft, catalog, fake)
    assert result["status"] == "BLOCKED"
    assert result["trace"][0]["status"] == "UNKNOWN"
    assert result["commits"] == []
    assert len(fake.calls) == 1


def test_crop_failure_prevents_commit_and_next_unit(tmp_path: Path) -> None:
    draft, catalog = _inputs(tmp_path)
    fake = FakeChain(crop_status="SEMANTIC_FAIL")
    result = run_fake_progressive_runtime(draft, catalog, fake)
    assert result["status"] == "BLOCKED"
    assert result["trace"][0]["status"] == "SEMANTIC_FAIL"
    assert result["commits"] == []
    assert len(fake.calls) == 1


def test_wrong_identity_at_snippet_endpoint_is_blocked(tmp_path: Path) -> None:
    draft, catalog = _inputs(tmp_path)
    fake = FakeChain()
    original = fake.state_at

    def wrong(raw: dict, endpoint: float) -> dict:
        value = original(raw, endpoint)
        value["state"]["subject"] = "C1"
        return value

    fake.state_at = wrong
    result = run_fake_progressive_runtime(draft, catalog, fake)
    assert result["status"] == "BLOCKED"
    assert result["trace"][0]["reason_code"] == "required_continuity_broken"
    assert result["commits"] == []


def test_only_explicit_fake_backend_can_run(tmp_path: Path) -> None:
    draft, catalog = _inputs(tmp_path)
    with pytest.raises(FakeRuntimeBlocked, match="fake_backend_required"):
        run_fake_progressive_runtime(draft, catalog, object())
