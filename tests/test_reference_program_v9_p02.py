# -*- coding: utf-8 -*-
"""P0.2 语义对账与编辑分解修复：四个新模块 + 新校验规则的回归锚定。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agentic_video.manifest import json_hash
from src.agentic_video.reference_program_v9 import (
    EDIT_VERSION, V9Blocked,
    apply_montage_reclassification, compile_material_requirements,
    normalize_section_shots, reconcile_section_boundaries,
    semantic_conflict_gate, validate_reference_programs,
    EDIT_PROGRAM_PROMPT, TRANSITION_TYPES,
)
from tests.test_reference_program_v9 import (
    FakeRunner, _conflicts_clean, _content, _edit, _ledger,
    _reconciliation, _section_bank, _section_watch,
)


def _cut_ledger() -> dict:
    ledger = _ledger()
    ledger["boundaries"] = [
        {"boundary_id": "video_start", "frame_id": "f000000",
         "pts_s": 0.0, "kind": "video_start"},
        {"boundary_id": "cut_a", "frame_id": "f000120", "pts_s": 4.0,
         "kind": "cut_candidate"},
        {"boundary_id": "cut_b", "frame_id": "f000123", "pts_s": 4.1,
         "kind": "cut_candidate"},
        {"boundary_id": "cut_c", "frame_id": "f000180", "pts_s": 6.0,
         "kind": "cut_candidate"},
        {"boundary_id": "video_end", "frame_id": "f000900", "pts_s": 30.0,
         "kind": "video_end"},
    ]
    ledger["scene_detection"]["cut_candidates"] = [
        {"boundary_id": "cut_a", "pts_s": 4.0},
        {"boundary_id": "cut_b", "pts_s": 4.1},
        {"boundary_id": "cut_c", "pts_s": 6.0},
    ]
    return ledger


def _cut_section_bank() -> dict:
    shot = {"information_added": "visible action", "edit_function": "kept",
            "event_relation": "same_event", "supporting_deterministic_ids": []}
    return {"reference_sha256": "a" * 64, "sections": [{
        "section_id": "main",
        "shots": [
            {"shot_id": "main.shot_001", "start_boundary_id": "video_start",
             "end_boundary_id": "cut_a", "interval": [0.0, 4.0], **shot},
            {"shot_id": "main.shot_002", "start_boundary_id": "cut_a",
             "end_boundary_id": "cut_b", "interval": [4.0, 4.1], **shot},
            {"shot_id": "main.shot_003", "start_boundary_id": "cut_b",
             "end_boundary_id": "video_end", "interval": [4.1, 30.0], **shot},
        ],
        "cut_assessments": [
            {"boundary_id": "cut_a", "status": "real_cut", "reason": "image change"},
            {"boundary_id": "cut_b", "status": "real_cut", "reason": "image change"},
            {"boundary_id": "cut_c", "status": "not_cut", "reason": "continuous"},
        ],
        "source_interval": [0.0, 30.0], "raw_response_sha256": "d" * 64,
    }]}


def test_shot_normalizer_splits_real_cuts_and_resolves_short_transitions() -> None:
    normalization = normalize_section_shots(_cut_section_bank(), _cut_ledger())
    section = normalization["sections"][0]
    assert [shot["interval"] for shot in section["content_shots"]] == [
        [0.0, 4.0], [4.1, 30.0]]
    assert section["content_shots"][0]["information_added"] == "visible action"
    assert len(section["transition_segments"]) == 1
    transition = section["transition_segments"][0]
    assert transition["interval"] == [4.0, 4.1]
    assert transition["duration_s"] == 0.1
    assert transition["transition_type"] == "unclassified"
    # cut_c 被判 not_cut，不得产生任何镜头边界
    assert 6.0 not in [edge for shot in section["content_shots"]
                       for edge in shot["interval"]]


def test_montage_reclassification_promotes_text_transitions() -> None:
    normalization = normalize_section_shots(_cut_section_bank(), _cut_ledger())
    transition_id = \
        normalization["sections"][0]["transition_segments"][0]["transition_id"]
    montage = {"sections": [{"section_id": "main", "segments": [
        {"segment_id": transition_id, "segment_kind": "transition",
         "interval": [4.0, 4.1], "visible_content": "冠军奖杯照片",
         "on_screen_text": "跆拳道全国冠军", "transition_type": None},
    ]}]}
    result = apply_montage_reclassification(normalization, montage)
    section = result["sections"][0]
    promoted = [shot for shot in section["content_shots"]
                if shot.get("reclassified_from") == transition_id]
    assert len(promoted) == 1
    assert promoted[0]["information_added"] == "冠军奖杯照片"
    assert not section["transition_segments"]

    normalization = normalize_section_shots(_cut_section_bank(), _cut_ledger())
    montage = {"sections": [{"section_id": "main", "segments": [
        {"segment_id": transition_id, "segment_kind": "transition",
         "interval": [4.0, 4.1], "visible_content": "模糊动态画面",
         "on_screen_text": None, "transition_type": "zoom_blur"},
    ]}]}
    result = apply_montage_reclassification(normalization, montage)
    assert result["sections"][0]["transition_segments"][0][
        "transition_type"] == "zoom_blur"
    assert len(result["sections"][0]["content_shots"]) == 2


def test_boundary_reconciliation_moves_unjustified_boundary_and_rewatches(
        tmp_path: Path) -> None:
    bank = _section_bank()
    runner = FakeRunner(
        watch=[_section_watch("competition", "video_start", "cut_001"),
               _section_watch("proofs", "cut_001", "video_end", "cut_002")],
        ask=[{"boundaries": [
            {"boundary_id": "cut_002", "semantic_change": False,
             "change_types": ["none"], "reason": "同一事件仍在延续",
             "recommended_boundary_id": "cut_001"}]}])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    result, reconciled = reconcile_section_boundaries(
        video, _ledger(), bank, tmp_path, runner=runner)
    assert result["boundaries"][0]["action"] == "moved"
    assert result["boundaries"][0]["moved_to"] == "cut_001"
    assert result["rewatched"] == ["competition", "proofs"]
    intervals = {row["section_id"]: row["source_interval"]
                 for row in reconciled["sections"]}
    assert intervals == {"competition": [0.0, 10.0], "proofs": [10.0, 30.0]}
    assert reconciled["reconciled"] is True


def test_unjustified_boundary_without_valid_recommendation_blocks(
        tmp_path: Path) -> None:
    runner = FakeRunner(ask=[{"boundaries": [
        {"boundary_id": "cut_002", "semantic_change": False,
         "change_types": ["none"], "reason": "同一事件仍在延续",
         "recommended_boundary_id": None}]}])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(V9Blocked) as excinfo:
        reconcile_section_boundaries(video, _ledger(), _section_bank(),
                                     tmp_path, runner=runner)
    assert excinfo.value.reason_code == "unjustified_section_boundary"


def test_continuous_clip_contradicting_real_cuts_fails_validation(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    edit["editorial_patterns"][0]["composition_mode"] = "continuous_clip"
    req = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, req, _ledger(),
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("continuous_clip contradicts" in error
               for error in result["errors"])


def test_contested_term_from_conflict_gate_is_banned(tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    content["sections"][0]["reference_specific_fact"] = "这是品势表演"
    req = compile_material_requirements(content, edit, tmp_path)
    conflicts = dict(_conflicts_clean(), must_not_assert=["品势"],
                     unresolved_topics=["事件性质"])
    result = validate_reference_programs(content, edit, req, _ledger(),
                                         _section_bank(),
                                         reconciliation=_reconciliation(),
                                         conflicts=conflicts)
    assert any("asserts contested term" in error for error in result["errors"])


def test_final_section_on_screen_text_must_be_quoted_verbatim(
        tmp_path: Path) -> None:
    ledger = _ledger()
    ledger["ocr"]["normalized_claim_events"] = [
        {"claim_id": "ocr_001", "interval": [25.0, 27.0],
         "text": "但不会剪脚指甲", "evidence_type": "observed_textual_claim"},
    ]
    content = _content()
    edit = _edit()
    req = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, req, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("final_text_quote missing" in error for error in result["errors"])
    content["sections"][-1]["final_text_quote"] = "自由与未来"
    result = validate_reference_programs(content, edit, req, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("final_text_quote not verbatim" in error
               for error in result["errors"])
    content["sections"][-1]["final_text_quote"] = "不会剪脚指甲"
    result = validate_reference_programs(content, edit, req, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert not any("final_text_quote" in error for error in result["errors"])


def test_semantic_conflict_gate_triggers_neutral_recheck_and_bans_terms(
        tmp_path: Path) -> None:
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    runner = FakeRunner(
        watch=[{"observable_actions": ["两人互相攻击", "一人倒地"],
                "subject_count": 2, "mutual_attacks_visible": True,
                "falls_visible": True, "result_visible": True,
                "event_nature_observable": "两人攻防并以一方倒地结束"}],
        ask=[
            {"contradictions": [
                {"topic": "事件性质", "source_a": "section:competition",
                 "quote_a": "两名选手进行对打", "source_b": "probe:q_002",
                 "quote_b": "这是品势表演而非对抗", "material": True}],
             "consistent": False},
            {"resolution": "unresolved", "resolved_description": "",
             "must_not_assert": ["品势"]},
        ])
    draft = json.loads(json.dumps(
        {"core_expression_draft": {"reference_specific": "对打证明能力"},
         "observations": []}))
    resolved = {"questions": [], "probe_history": [], "required_unresolved_ids": []}
    conflicts = semantic_conflict_gate(video, draft, _ledger(), _section_bank(),
                                       resolved, tmp_path, runner=runner)
    assert conflicts["unresolved_topics"] == ["事件性质"]
    assert conflicts["must_not_assert"] == ["品势"]
    assert conflicts["neutral_rechecks"][0]["neutral_recheck"][
        "mutual_attacks_visible"] is True


def test_edit_prompt_uses_finite_ontology_and_transition_refs() -> None:
    assert "multi_angle_action" in EDIT_PROGRAM_PROMPT
    assert "text_led_montage" in EDIT_PROGRAM_PROMPT
    assert "contrast_montage" in EDIT_PROGRAM_PROMPT
    assert "transition_refs" in EDIT_PROGRAM_PROMPT
    assert "shot_C01" in EDIT_PROGRAM_PROMPT
    assert "zoom_blur" in TRANSITION_TYPES
    assert EDIT_VERSION.endswith("_p02")


def test_montage_watch_stops_after_json_and_parses_per_shot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agentic_video.reference_program_v9 as module

    content_shots = [
        {"shot_id": f"main.shot_C{index:02d}",
         "interval": [index * 0.5, index * 0.5 + 0.5],
         "duration_s": 0.5, "information_added": "x", "edit_function": "y",
         "event_relation": "same_event", "interior_real_cuts": []}
        for index in range(1, 5)]
    normalization = {"sections": [{
        "section_id": "main", "interval": [0.0, 2.0], "real_cut_pts": [0.5],
        "content_shots": content_shots, "transition_segments": []}]}
    monkeypatch.setattr(
        module, "_extract_segment_frames",
        lambda ffmpeg_bin, reference, interval, out_dir, count=3: [out_dir / "f0.jpg"])
    answers = [
        {"segment_id": shot["shot_id"], "visible_content": f"内容{index}",
         "on_screen_text": "跆拳道全国冠军" if index == 2 else None,
         "transition_type": None}
        for index, shot in enumerate(content_shots, 1)]
    runner = FakeRunner(inspect=answers)
    ledger = _ledger()
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    result = module.watch_fast_montage_shots(
        video, ledger, normalization, tmp_path, runner=runner,
        ffmpeg_bin="ffmpeg", force=True)
    assert len(result["sections"][0]["segments"]) == 4
    assert all(kwargs.get("stop_after_json_object") is True
               for _images, _prompt, kwargs in runner.inspect_calls)
    assert result["sections"][0]["segments"][1]["on_screen_text"] == "跆拳道全国冠军"
