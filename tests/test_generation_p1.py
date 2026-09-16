from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.agentic_video.generation_p1 import (
    P1Blocked, compile_p1_contract_draft, plan_generation_units,
)
from src.agentic_video.generation_v9g import validate_generation_contract


def _fixture() -> dict:
    path = Path(__file__).parent / "fixtures" / "v9g_p1_fixed.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_generation_units_are_not_four_rendered_snippets() -> None:
    fixture = _fixture()
    draft = compile_p1_contract_draft(**fixture)
    section = draft["sections"][0]
    assert draft["execution_authorized"] is False
    assert len(section["generation_units"]) == 2
    assert section["snippet_count_range"] == [4, 4]
    assert section["rendered_snippets"] == []
    assert [unit["primary_causal_goal"] for unit in section["generation_units"]] == [
        "Establish the confrontation and begin an attack",
        "Complete the attack and show the opponent's reaction",
    ]
    assert not validate_generation_contract(draft)["passed"]


def test_unknown_continuity_cannot_enter_generation_plan() -> None:
    fixture = _fixture()
    fixture["requirements"]["requirements"][0]["continuity_requirement"]["levels"][
        "subject"] = "unknown"
    with pytest.raises(P1Blocked, match="continuity_unknown"):
        compile_p1_contract_draft(**fixture)


def test_unit_plan_cannot_infer_count_from_snippet_count() -> None:
    fixture = _fixture()
    row = fixture["requirements"]["requirements"][0]
    row["presentation_requirement"]["snippet_count_range"] = [9, 9]
    units = plan_generation_units(row, fixture["action_plans"]["S2"])
    assert len(units) == 2


def test_required_phases_and_identity_must_remain_coherent() -> None:
    fixture = _fixture()
    row = fixture["requirements"]["requirements"][0]
    units = deepcopy(fixture["action_plans"]["S2"])
    units[1]["semantic_phases"] = ["visible_result", "decisive_action"]
    with pytest.raises(P1Blocked, match="required_phase_partition_invalid"):
        plan_generation_units(row, units)
    units = deepcopy(fixture["action_plans"]["S2"])
    units[1]["continuity_ids"]["subject"] = "C2"
    with pytest.raises(P1Blocked, match="required_continuity_broken"):
        plan_generation_units(row, units)


def test_p1_draft_is_deterministic_and_never_accepts_legacy_boolean_schema() -> None:
    fixture = _fixture()
    first = compile_p1_contract_draft(**fixture)
    second = compile_p1_contract_draft(**fixture)
    assert first == second
    fixture["requirements"]["requirements"][0]["continuity_requirement"] = {
        "same_subject_required": True,
    }
    with pytest.raises(P1Blocked, match="continuity_schema_invalid"):
        compile_p1_contract_draft(**fixture)
