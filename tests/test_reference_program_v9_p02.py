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
    FakeRunner, _conflicts_clean, _content, _draft, _edit, _ledger,
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


def test_montage_transitions_keep_identity_with_overlay_text() -> None:
    """P0.3：带文字的转场保持 transition 身份，文字记 overlay 属性。"""
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
    transitions = section["transition_segments"]
    assert len(transitions) == 1
    assert transitions[0]["overlay_text"] == "跆拳道全国冠军"
    assert len(section["content_shots"]) == 2  # 不因文字升级

    normalization = normalize_section_shots(_cut_section_bank(), _cut_ledger())
    montage = {"sections": [{"section_id": "main", "segments": [
        {"segment_id": transition_id, "segment_kind": "transition",
         "interval": [4.0, 4.1], "visible_content": "模糊动态画面",
         "on_screen_text": None, "transition_type": "zoom_blur"},
    ]}]}
    result = apply_montage_reclassification(normalization, montage)
    assert result["sections"][0]["transition_segments"][0][
        "transition_type"] == "zoom_blur"
    assert "overlay_text" not in result["sections"][0]["transition_segments"][0]


def _full_watch(section_id: str, boundary_names: list[str],
                *, head_relation: str = "different_event") -> dict:
    pts = {"video_start": 0.0, "cut_001": 10.0, "cut_002": 20.0,
           "cut_003": 25.0, "video_end": 30.0}
    shots = []
    for index, (left, right) in enumerate(
            zip(boundary_names, boundary_names[1:]), 1):
        shots.append({
            "shot_id": f"{section_id}.shot_{index:03d}",
            "start_boundary_id": left, "end_boundary_id": right,
            "interval": [pts[left], pts[right]],
            "information_added": "a visible action changes state",
            "edit_function": "establishes a new stage",
            "event_relation": head_relation if index == 1 else "same_event",
            "supporting_deterministic_ids": []})
    assessments = [{"boundary_id": name, "status": "real_cut",
                    "reason": "visible image change"}
                   for name in boundary_names[1:-1]]
    return {"section_id": section_id, "shots": shots,
            "cut_assessments": assessments,
            "rhythm_claimed": False, "rhythm_evidence_ids": [],
            "unresolved_questions": []}


def test_frame_check_keeps_justified_boundary_without_rewatch(
        tmp_path: Path) -> None:
    """首 shot 为 different_event 时走帧级三维度判定；语义变化成立则保留。"""
    runner = FakeRunner(inspect=[
        {"event_continuity": False, "rhetorical_function_continuity": False,
         "audience_cognition_continuity": False, "semantic_change": True,
         "reason": "提出偏见 vs 开始反驳"}])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    import src.agentic_video.reference_program_v9 as module
    original = module._boundary_frames
    module._boundary_frames = (
        lambda ffmpeg_bin, reference, pts, out_dir, span:
        [out_dir / f"f{index}.jpg" for index in range(6)])
    try:
        result, reconciled = reconcile_section_boundaries(
            video, _ledger(), _section_bank(), tmp_path, runner=runner)
    finally:
        module._boundary_frames = original
    record = result["boundaries"][0]
    assert record["action"] == "kept"
    assert record["attempts"][0]["method"] == "frame_check"
    assert result["rewatched"] == []
    assert reconciled["sections"][0]["source_interval"] == [0.0, 20.0]


def test_iterative_frame_check_moves_until_semantic_change(
        tmp_path: Path) -> None:
    """首次帧判定不成立→移动→复查新边界→第二次成立（reconcile-until-stable）。"""
    ledger = _ledger()
    ledger["boundaries"].insert(3, {
        "boundary_id": "cut_003", "frame_id": "f000750", "pts_s": 25.0,
        "kind": "cut_candidate"})
    bank = _section_bank()
    # proofs 首 shot 为 different_event，避开 same_event 直移路径
    bank["sections"][1]["shots"][0]["event_relation"] = "different_event"
    runner = FakeRunner(
        watch=[_full_watch("competition", ["video_start", "cut_001", "cut_002",
                                           "cut_003"]),
               _full_watch("proofs", ["cut_003", "video_end"])],
        inspect=[
            {"event_continuity": True, "rhetorical_function_continuity": True,
             "audience_cognition_continuity": True, "semantic_change": False,
             "reason": "仍是比赛反应近景"},
            {"event_continuity": False, "rhetorical_function_continuity": False,
             "audience_cognition_continuity": False, "semantic_change": True,
             "reason": "比赛结束，照片蒙太奇开始"}])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    import src.agentic_video.reference_program_v9 as module
    original = module._boundary_frames
    module._boundary_frames = (
        lambda ffmpeg_bin, reference, pts, out_dir, span:
        [out_dir / f"f{index}.jpg" for index in range(6)])
    try:
        result, _reconciled = reconcile_section_boundaries(
            video, ledger, bank, tmp_path, runner=runner)
    finally:
        module._boundary_frames = original
    record = result["boundaries"][0]
    assert record["action"] == "moved"
    assert record["moved_to"] == "cut_003"
    assert len(record["attempts"]) == 2
    assert record["attempts"][1]["semantic_change"] is True


def test_unjustified_boundary_without_valid_recommendation_blocks(
        tmp_path: Path) -> None:
    runner = FakeRunner(inspect=[
        {"event_continuity": True, "rhetorical_function_continuity": True,
         "audience_cognition_continuity": True, "semantic_change": False,
         "reason": "同主题被误判为同事件"}])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    import src.agentic_video.reference_program_v9 as module
    original = module._boundary_frames
    module._boundary_frames = (
        lambda ffmpeg_bin, reference, pts, out_dir, span:
        [out_dir / f"f{index}.jpg" for index in range(6)])
    try:
        with pytest.raises(V9Blocked) as excinfo:
            reconcile_section_boundaries(video, _ledger(), _section_bank(),
                                          tmp_path, runner=runner)
    finally:
        module._boundary_frames = original
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
    assert EDIT_VERSION.endswith("_p03")


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


def test_contract_repair_round_reasks_programs_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agentic_video.reference_program_v9 as module

    def _broken_content() -> dict:
        value = _content()
        value["sections"][0]["continuity_basis"]["spatial_orientation"] = {
            "reason": "orientation visible", "evidence_ids": []}
        return value

    video = tmp_path / "reference.mp4"
    video.write_bytes(b"reference")
    runner = FakeRunner(
        watch=[
            _draft(),
            _section_watch("competition", "video_start", "cut_002", "cut_001"),
            _section_watch("proofs", "cut_002", "video_end"),
        ],
        ask=[
            _conflicts_clean(),
            _broken_content(), _edit(),
            _content(), _edit(),
        ], inspect=[{"event_continuity": False,
                     "rhetorical_function_continuity": False,
                     "audience_cognition_continuity": False,
                     "semantic_change": True, "reason": "new event"}])
    def fake_ledger(_reference, output_dir, **_kwargs):
        ledger = _ledger()
        (Path(output_dir) / "reference_evidence.json").write_text(
            json.dumps(ledger), encoding="utf-8")
        return ledger

    monkeypatch.setattr(module, "build_reference_evidence_ledger", fake_ledger)
    monkeypatch.setattr(module, "_build_review_assets",
                        lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_boundary_frames",
                        lambda ffmpeg_bin, reference, pts, out_dir, span:
                        [out_dir / f"f{index}.jpg" for index in range(6)])
    from types import SimpleNamespace
    cfg = SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg",
                                      "ffprobe_bin": "ffprobe"})
    result = module.run_reference_program_v9(cfg, video, tmp_path / "run",
                                             runner=runner, force=True)
    assert result["decision"] == "PENDING_HUMAN", result
    assert (tmp_path / "run" / "raw_responses" /
            "content_program_repair.txt").is_file()
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["stages"]["program_repair"]["passed_after_repair"] is True
    repaired = json.loads((tmp_path / "run" / "reference_content_program.json")
                          .read_text(encoding="utf-8"))
    assert repaired["contract_repair"] is True


def test_moved_boundary_satisfies_justification_via_moved_to(tmp_path: Path) -> None:
    """边界被对账移动后，校验器必须按 moved_to 的 PTS 认领依据记录。"""
    content = _content()
    # 模拟对账后：competition 结束边界被移动，proofs 从移动点开始
    content["sections"][0]["interval"] = [0.0, 10.0]
    content["sections"][1]["interval"] = [10.0, 30.0]
    edit = _edit()
    req = compile_material_requirements(content, edit, tmp_path)
    moved = {"schema_version": "boundary_reconciliation_v9_p02",
             "boundaries": [
                 {"boundary_id": "cut_002", "between": ["competition", "proofs"],
                  "semantic_change": False, "change_types": ["none"],
                  "reason": "同一事件仍在延续", "action": "moved",
                  "moved_to": "cut_001"}],
             "sections_after": [], "rewatched": []}
    result = validate_reference_programs(content, edit, req, _ledger(),
                                         _section_bank(), reconciliation=moved)
    assert not any("lacks reconciliation justification" in error
                   for error in result["errors"])
    assert not any("unjustified" in error for error in result["errors"])


def test_same_event_head_shot_forces_boundary_move_without_ask(
        tmp_path: Path) -> None:
    """下一 Section 首 shot 报 same_event 时确定性前移，不再依赖模型判定。"""
    ledger = _ledger()
    ledger["boundaries"].insert(3, {
        "boundary_id": "cut_003", "frame_id": "f000750", "pts_s": 25.0,
        "kind": "cut_candidate"})
    bank = _section_bank()
    tail = {"information_added": "对手倒地，主角收势站立", "edit_function": "结果呈现",
            "event_relation": "same_event", "supporting_deterministic_ids": []}
    fresh = {"information_added": "家庭合影照片出现", "edit_function": "新证据",
             "event_relation": "different_event", "supporting_deterministic_ids": []}
    bank["sections"][1]["shots"] = [
        {"shot_id": "proofs.shot_001", "start_boundary_id": "cut_002",
         "end_boundary_id": "cut_003", "interval": [20.0, 25.0], **tail},
        {"shot_id": "proofs.shot_002", "start_boundary_id": "cut_003",
         "end_boundary_id": "video_end", "interval": [25.0, 30.0], **fresh},
    ]
    runner = FakeRunner(
        watch=[_full_watch("competition", ["video_start", "cut_001", "cut_002",
                                           "cut_003"]),
               _full_watch("proofs", ["cut_003", "video_end"])],
        inspect=[{"event_continuity": False,
                  "rhetorical_function_continuity": False,
                  "audience_cognition_continuity": False,
                  "semantic_change": True,
                  "reason": "照片蒙太奇开始"}])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    import src.agentic_video.reference_program_v9 as module
    original = module._boundary_frames
    module._boundary_frames = (
        lambda ffmpeg_bin, reference, pts, out_dir, span:
        [out_dir / f"f{index}.jpg" for index in range(6)])
    try:
        result, _reconciled = reconcile_section_boundaries(
            video, ledger, bank, tmp_path, runner=runner)
    finally:
        module._boundary_frames = original
    record = result["boundaries"][0]
    assert record["action"] == "moved"
    assert record["moved_to"] == "cut_003"
    assert record["attempts"][0]["method"] == "same_event_rule"
    assert record["attempts"][1]["method"] == "frame_check"
    assert not runner.ask_calls


def test_content_sections_must_match_reconciled_observed_sections(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    content["sections"][0]["interval"] = [0.0, 5.0]
    content["sections"][0]["end_boundary_id"] = "cut_001"
    req = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, req, _ledger(),
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("diverge from reconciled" in error for error in result["errors"])


def test_edit_builder_aligns_mode_constants_deterministically(
        tmp_path: Path) -> None:
    from src.agentic_video.reference_program_v9 import (
        build_reference_edit_program, normalize_section_shots)

    content = _content()
    from src.agentic_video.reference_program_v9 import _attach_intervals
    _attach_intervals(content, _ledger(),
                      keys=("meaningful_units", "sections"), stage="edit_program")
    raw_edit = _edit()
    # 模型错误一：多镜头段标 continuous_clip；错误二：event_compression 缺模式常量
    raw_edit["editorial_patterns"][0]["composition_mode"] = "continuous_clip"
    raw_edit["editorial_patterns"][1]["composition_mode"] = "event_compression_montage"
    raw_edit["editorial_patterns"][1]["source_continuity"] = "continuous_required"
    raw_edit["editorial_patterns"][1]["ordering_constraint"] = "source_order"
    raw_edit["editorial_patterns"][1]["source_semantics"] = "one_long_event"
    bank = _section_bank()
    for shot in bank["sections"][0]["shots"]:
        shot["event_relation"] = "same_event"
    normalization = normalize_section_shots(bank, _ledger())
    runner = FakeRunner(ask=[raw_edit])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    edit = build_reference_edit_program(
        content, _ledger(), tmp_path, runner=runner,
        section_observations=bank, normalization=normalization, force=True)
    first = edit["editorial_patterns"][0]
    assert first["composition_mode"] == "multi_angle_action"
    second = edit["editorial_patterns"][1]
    assert second["composition_mode"] == "event_compression_montage"
    assert second["source_continuity"] == "non_contiguous_allowed"
    assert second["ordering_constraint"] == "preserve_event_progression"
    aligns = {row["field"]: row for row in edit["programmatic_mode_alignment"]}
    assert aligns["composition_mode"]["from"] == "continuous_clip"
    assert aligns["source_continuity"]["basis"] == "mode_mandated_constant"


def test_first_section_on_screen_assertion_must_be_hook_quoted(
        tmp_path: Path) -> None:
    """P0.3：第一个含屏幕文字的 Section 必须以 hook_text_quote 保留待反驳命题。"""
    ledger = _ledger()
    ledger["ocr"]["normalized_claim_events"] = [
        {"claim_id": "ocr_001", "interval": [1.0, 3.0],
         "text": "人们常常觉得失去了双手就会变成一个废人",
         "evidence_type": "observed_textual_claim"},
    ]
    content = _content()
    edit = _edit()
    req = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, req, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("hook_text_quote missing" in error for error in result["errors"])
    content["sections"][0]["hook_text_quote"] = "专业跆拳道运动员"
    result = validate_reference_programs(content, edit, req, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("hook_text_quote not verbatim" in error
               for error in result["errors"])
    content["sections"][0]["hook_text_quote"] = "失去了双手就会变成一个废人"
    result = validate_reference_programs(content, edit, req, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert not any("hook_text_quote" in error for error in result["errors"])


def test_montage_modes_require_snippet_floor(tmp_path: Path) -> None:
    """P0.3：蒙太奇类模式 snippet_count_range 下限必须 >=2。"""
    content = _content()
    edit = _edit()
    edit["editorial_patterns"][1]["composition_mode"] = "evidence_montage"
    edit["editorial_patterns"][1]["snippet_count_range"] = [1, 1]
    req = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, req, _ledger(),
                                         _section_bank(),
                                         reconciliation=_reconciliation())
    assert any("snippet_count_range min must be >= 2" in error
               for error in result["errors"])


def test_snippet_range_aligns_to_observed_shot_count(tmp_path: Path) -> None:
    from src.agentic_video.reference_program_v9 import (
        build_reference_edit_program, normalize_section_shots)

    content = _content()
    from src.agentic_video.reference_program_v9 import _attach_intervals
    _attach_intervals(content, _ledger(),
                      keys=("meaningful_units", "sections"), stage="edit_program")
    raw_edit = _edit()
    raw_edit["editorial_patterns"][1]["composition_mode"] = "evidence_montage"
    raw_edit["editorial_patterns"][1]["snippet_count_range"] = [1, 1]
    bank = _section_bank()
    for shot in bank["sections"][0]["shots"]:
        shot["event_relation"] = "same_event"
    normalization = normalize_section_shots(bank, _ledger())
    runner = FakeRunner(ask=[raw_edit])
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    edit = build_reference_edit_program(
        content, _ledger(), tmp_path, runner=runner,
        section_observations=bank, normalization=normalization, force=True)
    # proofs 段 1 个内容镜头：evidence_montage → 下限 max(2, ceil(0.6)) = 2
    assert edit["editorial_patterns"][1]["snippet_count_range"] == [2, 2]
    fields = {row["field"] for row in edit["programmatic_mode_alignment"]}
    assert "snippet_count_range" in fields


def test_dense_cut_completion_fills_missing_assessments(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """整段 watch 漏判的密集切点由逐对帧补判；补判不完整则 BLOCKED。"""
    import src.agentic_video.reference_program_v9 as module

    bank = _cut_section_bank()
    # 模拟整段 watch 只交了 cut_a 的作业，cut_b/cut_c 缺席
    bank["sections"][0]["cut_assessments"] = [
        {"boundary_id": "cut_a", "status": "real_cut", "reason": "image change"}]
    bank["sections"][0]["pending_cut_assessments"] = ["cut_b", "cut_c"]
    monkeypatch.setattr(
        module, "_boundary_frames",
        lambda ffmpeg_bin, reference, pts, out_dir, span:
        [out_dir / f"f{index}.jpg" for index in range(6)])
    monkeypatch.setattr(module.common, "run_ffmpeg", lambda *args, **kwargs: None)
    runner = FakeRunner(inspect=[
        {"assessments": [
            {"boundary_id": "cut_b", "status": "real_cut",
             "reason": "内容切换"},
            {"boundary_id": "cut_c", "status": "not_cut",
             "reason": "同一画面微小变化"}]}])
    row = module._complete_dense_cut_assessments(
        tmp_path / "reference.mp4", _cut_ledger(), bank["sections"][0],
        tmp_path, runner=runner, ffmpeg_bin="ffmpeg")
    assessed = {item["boundary_id"]: item for item in row["cut_assessments"]}
    assert assessed["cut_b"]["status"] == "real_cut"
    assert assessed["cut_c"]["status"] == "not_cut"
    assert "[dense-frame-check]" in assessed["cut_b"]["reason"]

    # 补判不完整 → BLOCKED
    runner2 = FakeRunner(inspect=[
        {"assessments": [
            {"boundary_id": "cut_b", "status": "real_cut", "reason": "ok"}]}])
    bank["sections"][0]["cut_assessments"] = [
        {"boundary_id": "cut_a", "status": "real_cut", "reason": "image change"}]
    bank["sections"][0]["pending_cut_assessments"] = ["cut_b", "cut_c"]
    with pytest.raises(V9Blocked) as excinfo:
        module._complete_dense_cut_assessments(
            tmp_path / "reference.mp4", _cut_ledger(), bank["sections"][0],
            tmp_path, runner=runner2, ffmpeg_bin="ffmpeg")
    assert excinfo.value.reason_code == "cut_assessment_missing"
