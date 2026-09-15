from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.cli import build_parser
from src.agentic_video.reference_program_v9 import (
    CONTENT_VERSION, EDIT_VERSION, V9Blocked,
    accept_reference_programs, build_reference_content_program,
    build_reference_edit_program, build_reference_understanding_draft,
    compile_material_requirements, merge_cut_candidates,
    resolve_reference_questions, run_reference_program_v9,
    validate_reference_programs, _parse_one_object,
)


def _ledger(duration: float = 30.0, frame_count: int = 901) -> dict:
    frames = [{"frame_id": f"f{index:06d}", "pts_s": round(index / 30, 6)}
              for index in range(frame_count)]
    return {
        "schema_version": "reference_evidence_v9",
        "reference": {"duration_s": duration, "sha256": "a" * 64},
        "native_frame_count": frame_count,
        "native_frames": frames,
        "boundaries": [
            {"boundary_id": "video_start", "frame_id": "f000000",
             "pts_s": 0.0, "kind": "video_start"},
            {"boundary_id": "cut_001", "frame_id": "f000300",
             "pts_s": 10.0, "kind": "cut_candidate"},
            {"boundary_id": "cut_002", "frame_id": "f000600",
             "pts_s": 20.0, "kind": "cut_candidate"},
            {"boundary_id": "video_end", "frame_id": f"f{frame_count - 1:06d}",
             "pts_s": duration, "kind": "video_end"},
        ],
        "scene_detection": {"cut_candidates": [
            {"boundary_id": "cut_001", "frame_id": "f000300", "pts_s": 10.0},
            {"boundary_id": "cut_002", "frame_id": "f000600", "pts_s": 20.0},
        ]},
        "motion": {"motion_peaks": []},
        "ocr": {"normalized_claim_events": [], "full_text": ""},
        "transcript": {"segments": [], "full_text": ""},
        "audio": {"beat_analysis": {"beat_points_s": []},
                  "silence_intervals": [], "intensity": {}},
        "vlm_frame_policy": {
            "native_frames_used_for_deterministic_analysis": frame_count,
            "native_frames_sent_in_one_global_vlm_call": 0,
            "global_semantic_watch_fps": 4.0, "flashvid_loaded": False,
        },
    }


def _draft(*, questions=None) -> dict:
    return {
        "core_expression_draft": {
            "reference_specific": "specific",
            "transferable_structure": "establish condition then prove capability",
        },
        "reference_specific_terms": ["specific person"],
        "observations": [
            {"evidence_id": "ev_visual", "evidence_type": "observed_visual",
             "description": "a visible action changes another subject's state",
             "start_boundary_id": "video_start", "end_boundary_id": "cut_001",
             "supporting_deterministic_ids": []},
            {"evidence_id": "ev_text", "evidence_type": "observed_textual_claim",
             "description": "the frame contains a capability claim",
             "start_boundary_id": "cut_002", "end_boundary_id": "video_end",
             "supporting_deterministic_ids": []},
        ],
        "meaningful_units": [
            {"unit_id": "unit_01", "purpose": "establish condition",
             "start_boundary_id": "video_start", "end_boundary_id": "cut_001",
             "evidence_ids": ["ev_visual"]},
            {"unit_id": "unit_02", "purpose": "show action",
             "start_boundary_id": "cut_001", "end_boundary_id": "cut_002",
             "evidence_ids": ["ev_visual"]},
            {"unit_id": "unit_03", "purpose": "show independent proof",
             "start_boundary_id": "cut_002", "end_boundary_id": "video_end",
             "evidence_ids": ["ev_text"]},
        ],
        "section_drafts": [],
        "unresolved_questions": questions or [],
    }


def _content() -> dict:
    return {
        "schema_version": CONTENT_VERSION,
        "core_expression": {
            "reference_specific": "specific",
            "transferable_structure": "establish a visible constraint then prove capability",
        },
        "reference_specific_terms": ["specific person"],
        "perception_focus": {
            "primary_focus": {"type": "subject"},
            "continuity_requirements": ["same_subject_across_sections"],
            "important_evidence_types": ["visually_obvious_condition", "capability_action"],
            "important_relations": ["subject_to_action", "action_to_visible_result"],
        },
        "reference_observations": _draft()["observations"],
        "meaningful_units": _draft()["meaningful_units"],
        "sections": [
            {"section_id": "competition", "start_boundary_id": "video_start",
             "end_boundary_id": "cut_002", "interval": [0.0, 20.0],
             "reference_specific_fact": "specific event",
             "transferable_structure": "show a long capability event through decisive phases",
             "audience_takeaway": "the subject can act effectively",
             "cognition_change": {"before": "uncertain", "after": "capability is visible"},
             "content_function": "proof", "relation_to_previous": "contrast",
             "evidence_ids": ["ev_visual"], "unit_ids": ["unit_01", "unit_02"]},
            {"section_id": "proofs", "start_boundary_id": "cut_002",
             "end_boundary_id": "video_end", "interval": [20.0, 30.0],
             "reference_specific_fact": "specific claims",
             "transferable_structure": "accumulate independent evidence for one proposition",
             "audience_takeaway": "the capability generalizes",
             "cognition_change": {"before": "one example", "after": "multiple proofs"},
             "content_function": "expansion", "relation_to_previous": "broadens proof",
             "evidence_ids": ["ev_text"], "unit_ids": ["unit_03"]},
        ],
        "unresolved_questions": [], "probe_history": [],
    }


def _edit() -> dict:
    return {
        "schema_version": EDIT_VERSION,
        "operations": [
            {"operation_id": "op_001", "section_id": "competition",
             "operation_type": "hard_cut", "start_boundary_id": "video_start",
             "end_boundary_id": "cut_001", "interval": [0.0, 10.0],
             "time_source": "deterministic_boundary_registry", "evidence_ids": ["ev_visual"],
             "support_status": "supported", "purpose": "compression"},
        ],
        "editorial_patterns": [
            {"section_id": "competition", "duration_budget_s": 5.0,
             "composition_mode": "event_compression_montage",
             "source_semantics": "one_long_event",
             "semantic_phases": ["setup", "decisive_action", "visible_result"],
             "snippet_count_range": [2, 4],
             "source_continuity": "non_contiguous_allowed",
             "semantic_continuity": "required",
             "ordering_constraint": "preserve_event_progression",
             "individual_duration_policy": "minimum_sufficient_duration",
             "audience_requirement": "blind viewer understands progression",
             "evidence_ids": ["ev_visual"]},
            {"section_id": "proofs", "duration_budget_s": 4.0,
             "composition_mode": "evidence_montage",
             "source_semantics": "multiple_events",
             "semantic_phases": ["independent_proof_a", "independent_proof_b"],
             "snippet_count_range": [2, 4],
             "source_continuity": "non_contiguous_allowed",
             "semantic_continuity": "required",
             "ordering_constraint": "claim_consistency",
             "individual_duration_policy": "minimum_sufficient_duration",
             "audience_requirement": "blind viewer understands common proposition",
             "evidence_ids": ["ev_text"]},
        ],
        "measured_style": {
            "reference_duration_s": 30.0, "meaningful_unit_count": 3,
            "section_duration_distribution": [20.0, 10.0],
            "shot_duration_distribution": [10.0, 10.0, 10.0],
            "information_interval_distribution": [10.0, 10.0],
            "time_source": "deterministic_native_pts",
        },
    }


class FakeRunner:
    def __init__(self, *, watch=None, ask=None, inspect=None):
        self.watch_values = list(watch or [])
        self.ask_values = list(ask or [])
        self.inspect_values = list(inspect or [])
        self.watch_calls = []
        self.ask_calls = []

    @staticmethod
    def _answer(value):
        text = value if isinstance(value, str) else json.dumps(value)
        return SimpleNamespace(text=text, input_tokens=10,
                               output_tokens=20, elapsed_s=.1,
                               input_build_s=.1, frames_estimate=88,
                               sampling={"sampling_verified": True}, gpu_pair="0,1")

    def watch(self, video, prompt, **kwargs):
        self.watch_calls.append((video, prompt, kwargs))
        return self._answer(self.watch_values.pop(0))

    def ask(self, prompt, **kwargs):
        self.ask_calls.append((prompt, kwargs))
        return self._answer(self.ask_values.pop(0))

    def inspect_media(self, images, prompt, **kwargs):
        return self._answer(self.inspect_values.pop(0))


def test_cli_exposes_v9_and_keeps_v82() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "reference-program-v9", "--output", "run", "--gpu-pairs", "0,1",
    ])
    assert args.reference.endswith("7682719919410072847/video.mp4")
    accepted = parser.parse_args([
        "reference-program-v9-accept", "--output", "run",
        "--human-review", "human.json",
    ])
    assert accepted.command == "reference-program-v9-accept"
    assert parser.parse_args([
        "character-recall-v82", "--phase", "prepare", "--output", "v82",
    ]).command == "character-recall-v82"


def test_cut_candidates_merge_within_two_native_frames() -> None:
    frames = _ledger(duration=1.0, frame_count=31)["native_frames"]
    merged = merge_cut_candidates([
        {"time_s": .3, "signal": "scene_change", "threshold": .1},
        {"time_s": .333, "signal": "histogram", "threshold": .2},
        {"time_s": .8, "signal": "scene_change", "threshold": .3},
    ], frames)
    assert len(merged) == 2
    assert merged[0]["signals"] == ["histogram", "scene_change"]


def test_model_response_must_be_exactly_one_json_object() -> None:
    assert _parse_one_object('```json\n{"ok": true}\n```', stage="test") == {"ok": True}
    for raw in ('{"a": 1} trailing', '{"a": 1}{"b": 2}', '[1, 2]'):
        with pytest.raises(V9Blocked):
            _parse_one_object(raw, stage="test")


def test_global_watch_uses_four_fps_without_flashvid_or_native_frame_batch(tmp_path: Path) -> None:
    draft = _draft()
    runner = FakeRunner(watch=[draft])
    result = build_reference_understanding_draft(
        tmp_path / "reference.mp4", _ledger(frame_count=658), tmp_path, runner=runner)
    assert result["global_watch"] == {
        "fps": 4.0, "flashvid_loaded": False, "whole_reference": True}
    assert runner.watch_calls[0][2]["fps"] == 4.0
    assert '"native_frames":' not in runner.watch_calls[0][1]
    assert '"native_frame_count":658' not in runner.watch_calls[0][1]
    assert result["observations"][1]["evidence_type"] == "observed_textual_claim"


def test_required_question_probe_budget_remains_blocked(tmp_path: Path) -> None:
    question = {"id": "q1", "question": "what changes", "importance": "required",
                "gap_type": "event_structure", "selected_probe": "dense_video",
                "start_boundary_id": "video_start", "end_boundary_id": "cut_001",
                "interval": [0.0, 10.0], "status": "open"}
    runner = FakeRunner(watch=[
        {"question_id": "q1", "status": "open", "answer": "still unclear",
         "evidence": [{"evidence_type": "unsupported", "description": "unclear"}],
         "new_questions": []},
    ])
    result = resolve_reference_questions(
        tmp_path / "reference.mp4", _draft(questions=[question]), _ledger(), tmp_path,
        runner=runner, max_rounds=1, max_probes_per_round=1)
    assert result["required_unresolved_ids"] == ["q1"]
    with pytest.raises(V9Blocked, match="q1"):
        build_reference_content_program(_draft(), result, _ledger(), tmp_path,
                                        runner=runner)


def test_probe_selection_must_match_the_unresolved_gap(tmp_path: Path) -> None:
    question = {"id": "q1", "question": "where are the beats",
                "importance": "required", "gap_type": "rhythm",
                "selected_probe": "dense_video", "interval": [0.0, 10.0],
                "status": "open"}
    with pytest.raises(V9Blocked, match="gap='rhythm'"):
        resolve_reference_questions(
            tmp_path / "reference.mp4", _draft(questions=[question]), _ledger(),
            tmp_path, runner=FakeRunner(), max_rounds=1)


def test_program_validation_accepts_event_compression_and_evidence_montage(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    requirements = compile_material_requirements(content, edit, tmp_path)
    validation = validate_reference_programs(content, edit, requirements, _ledger())
    assert validation["passed"], validation["errors"]
    event = requirements["requirements"][0]
    assert event["presentation_requirement"] == {
        "target_duration_s": 5.0,
        "composition_mode": "event_compression_montage",
        "snippet_count_range": [2, 4],
        "individual_duration_policy": "minimum_sufficient_duration",
        "source_continuity": "non_contiguous_allowed",
        "semantic_continuity": "required",
        "ordering_constraint": "preserve_event_progression",
        "audience_requirement": "blind viewer understands progression",
    }
    assert event["semantic_requirement"]["required_event_understanding"].startswith(
        "understand_full_source_event")
    assert requirements["requirements"][1]["continuity_requirement"][
        "same_event_required"] is False


def test_two_coarse_windows_cannot_swallow_many_meaningful_units(tmp_path: Path) -> None:
    content = _content()
    content["meaningful_units"].append({
        "unit_id": "unit_04", "purpose": "new detail", "start_boundary_id": "cut_002",
        "end_boundary_id": "video_end", "evidence_ids": ["ev_text"]})
    content["sections"][1]["unit_ids"].append("unit_04")
    requirements = compile_material_requirements(content, _edit(), tmp_path)
    result = validate_reference_programs(content, _edit(), requirements, _ledger())
    assert any("two-window collapse" in error for error in result["errors"])


def test_claim_cannot_be_upgraded_to_visual_fact(tmp_path: Path) -> None:
    content = _content()
    content["reference_observations"][1]["verified_as_visual_fact"] = True
    requirements = compile_material_requirements(content, _edit(), tmp_path)
    result = validate_reference_programs(content, _edit(), requirements, _ledger())
    assert any("claim upgraded" in error for error in result["errors"])


def test_requirement_transfer_gate_rejects_reference_and_target_leakage(tmp_path: Path) -> None:
    content = _content()
    content["sections"][0]["transferable_structure"] = "找一个没有双手的跆拳道女性"
    requirements = compile_material_requirements(content, _edit(), tmp_path)
    result = validate_reference_programs(content, _edit(), requirements, _ledger())
    assert any("leaked" in error for error in result["errors"])
    requirements["requirements"][0]["semantic_requirement"]["meaning_to_prove"] = "找小黑"
    result = validate_reference_programs(content, _edit(), requirements, _ledger())
    assert any("小黑" in error for error in result["errors"])


def test_event_compression_order_evidence_montage_causality_and_dialogue_integrity(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    edit["editorial_patterns"][0]["ordering_constraint"] = "source_order"
    edit["editorial_patterns"][1]["source_semantics"] = "one_long_event"
    requirements = compile_material_requirements(content, edit, tmp_path)
    requirements["requirements"][1]["continuity_requirement"]["same_event_required"] = True
    result = validate_reference_programs(content, edit, requirements, _ledger())
    assert any("progression" in error for error in result["errors"])
    assert any("fabricates one event" in error for error in result["errors"])
    assert any("same-event causality" in error for error in result["errors"])

    edit = _edit()
    edit["editorial_patterns"][1]["composition_mode"] = "dialogue_compression"
    edit["editorial_patterns"][1]["source_semantics"] = "dialogue"
    requirements = compile_material_requirements(content, edit, tmp_path)
    dialogue = requirements["requirements"][1]["evidence_requirement"][
        "dialogue_integrity"]
    assert dialogue["complete_utterances_required"] is True
    dialogue["meaning_and_causality_must_not_change"] = False
    result = validate_reference_programs(content, edit, requirements, _ledger())
    assert any("dialogue integrity" in error for error in result["errors"])


def test_fake_runner_evidence_gap_probe_to_three_programs(tmp_path: Path) -> None:
    question = {"id": "q1", "question": "which phases", "importance": "required",
                "gap_type": "event_structure", "selected_probe": "dense_video",
                "start_boundary_id": "video_start", "end_boundary_id": "cut_001",
                "status": "open"}
    raw_draft = _draft(questions=[question])
    runner = FakeRunner(
        watch=[raw_draft, {
            "question_id": "q1", "status": "resolved", "answer": "three phases",
            "evidence": [{"evidence_type": "observed_visual",
                          "description": "action advances through visible phases"}],
            "new_questions": [],
        }],
        ask=[_content(), _edit()],
    )
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    ledger = _ledger()
    draft = build_reference_understanding_draft(video, ledger, tmp_path, runner=runner)
    resolved = resolve_reference_questions(video, draft, ledger, tmp_path,
                                           runner=runner)
    content = build_reference_content_program(draft, resolved, ledger, tmp_path,
                                              runner=runner)
    edit = build_reference_edit_program(content, ledger, tmp_path, runner=runner)
    requirements = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, requirements, ledger)
    assert result["passed"], result["errors"]
    assert resolved["required_unresolved_ids"] == []
    assert {path.name for path in tmp_path.iterdir()} >= {
        "reference_content_program.json", "reference_edit_program.json",
        "material_requirements.json", "probe_log.jsonl",
    }


def test_fake_runner_full_v9_orchestration_stops_pending_human(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agentic_video.reference_program_v9 as module

    video = tmp_path / "reference.mp4"
    video.write_bytes(b"reference")
    runner = FakeRunner(watch=[_draft()], ask=[_content(), _edit()])
    monkeypatch.setattr(module, "build_reference_evidence_ledger",
                        lambda *args, **kwargs: _ledger())
    monkeypatch.setattr(module, "_build_review_assets",
                        lambda *args, **kwargs: [])
    cfg = SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"})
    result = run_reference_program_v9(cfg, video, tmp_path / "run",
                                      runner=runner, force=True)
    assert result["decision"] == "PENDING_HUMAN"
    assert result["passed"] is False
    assert result["long_video_perception_authorized"] is False
    assert (tmp_path / "run" / "human_review.template.json").is_file()
    assert not (tmp_path / "run" / "frozen_program_hashes.json").exists()


def test_human_acceptance_requires_every_section_and_transfer_test(tmp_path: Path) -> None:
    (tmp_path / "reference_content_program.json").write_text(
        json.dumps(_content()), encoding="utf-8")
    (tmp_path / "reference_edit_program.json").write_text(
        json.dumps(_edit()), encoding="utf-8")
    (tmp_path / "material_requirements.json").write_text(
        json.dumps(compile_material_requirements(_content(), _edit(), tmp_path)),
        encoding="utf-8")
    (tmp_path / "acceptance.json").write_text(json.dumps({
        "automated_passed": True, "validation": {"passed": True}}), encoding="utf-8")
    checks = {
        "visible_content_correct": True, "event_phases_correct": True,
        "composition_mode_correct": True, "text_audio_rhythm_role_correct": True,
        "material_requirement_searchable": True,
    }
    review = {
        "sections": [{"section_id": section_id, "passed": True, "checks": dict(checks)}
                     for section_id in ("competition", "proofs")],
        "requirement_transfer_test": {"passed": False},
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    assert accept_reference_programs(tmp_path, path)["decision"] == "BLOCKED"
    review["requirement_transfer_test"]["passed"] = True
    path.write_text(json.dumps(review), encoding="utf-8")
    result = accept_reference_programs(tmp_path, path)
    assert result["decision"] == "PASS"
    assert (tmp_path / "frozen_program_hashes.json").is_file()
