from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.cli import build_parser
from src.agentic_video.manifest import json_hash
from src.agentic_video.reference_program_v9 import (
    CONTENT_VERSION, EDIT_VERSION, V9Blocked,
    accept_reference_programs, build_reference_content_program,
    build_reference_edit_program, build_reference_understanding_draft,
    build_section_observations, collect_reference_gaps,
    compile_material_requirements, merge_cut_candidates,
    normalize_section_shots, apply_montage_reclassification,
    reconcile_section_boundaries, semantic_conflict_gate,
    resolve_reference_questions, run_reference_program_v9,
    validate_reference_programs, _parse_one_object, _read_cached_output,
    _review_binding, HUMAN_REVIEW_VERSION,
    GLOBAL_WATCH_PROMPT, SECTION_WATCH_PROMPT,
    CONTENT_PROGRAM_PROMPT, EDIT_PROGRAM_PROMPT, CONTINUITY_DIMENSIONS,
    TRANSITION_TYPES,
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
        "section_drafts": [
            {"section_id": "competition", "start_boundary_id": "video_start",
             "end_boundary_id": "cut_002", "evidence_ids": ["ev_visual"],
             "unit_ids": ["unit_01", "unit_02"]},
            {"section_id": "proofs", "start_boundary_id": "cut_002",
             "end_boundary_id": "video_end", "evidence_ids": ["ev_text"],
             "unit_ids": ["unit_03"]},
        ],
        "unresolved_questions": questions or [],
    }


def _content() -> dict:
    continuity = {
        "subject": "required", "opponent": "not_required", "scene": "preferred",
        "event": "required", "actor_role": "required",
        "spatial_orientation": "preferred",
    }
    basis = {key: {"reason": "visible section evidence",
                   "evidence_ids": ["ev_visual"] if level in {"required", "preferred"} else []}
             for key, level in continuity.items()}
    proof_continuity = dict(continuity, event="not_required", opponent="not_required")
    proof_basis = {key: {"reason": "independent visual evidence",
                         "evidence_ids": ["ev_text"] if level in {"required", "preferred"} else []}
                   for key, level in proof_continuity.items()}
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
             "evidence_ids": ["ev_visual"], "unit_ids": ["unit_01", "unit_02"],
             "continuity": continuity, "continuity_basis": basis},
            {"section_id": "proofs", "start_boundary_id": "cut_002",
             "end_boundary_id": "video_end", "interval": [20.0, 30.0],
             "reference_specific_fact": "specific claims",
             "transferable_structure": "accumulate independent evidence for one proposition",
             "audience_takeaway": "the capability generalizes",
             "cognition_change": {"before": "one example", "after": "multiple proofs"},
             "content_function": "expansion", "relation_to_previous": "broadens proof",
             "evidence_ids": ["ev_text"], "unit_ids": ["unit_03"],
             "continuity": proof_continuity, "continuity_basis": proof_basis},
        ],
        "unresolved_questions": [], "probe_history": [],
    }


def _edit() -> dict:
    bank = _section_bank()
    return {
        "schema_version": EDIT_VERSION,
        "direct_section_watch_sha256": json_hash(bank),
        "section_rewatch_refs": [
            {"section_id": row["section_id"],
             "raw_response_sha256": row["raw_response_sha256"],
             "shot_ids": [shot["shot_id"] for shot in row["shots"]]}
            for row in bank["sections"]],
        "operations": [
            {"operation_id": "op_001", "section_id": "competition",
             "operation_type": "hard_cut", "start_boundary_id": "video_start",
             "end_boundary_id": "cut_001", "interval": [0.0, 10.0],
             "time_source": "deterministic_boundary_registry", "evidence_ids": ["ev_visual"],
             "support_status": "supported", "purpose": "compression"},
        ],
        "editorial_patterns": [
            {"section_id": "competition", "duration_budget_s": 5.0,
             "shot_refs": ["competition.shot_C01", "competition.shot_C02"],
             "transition_refs": [],
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
             "shot_refs": ["proofs.shot_C01"],
             "transition_refs": [],
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


def _section_watch(section_id: str, start: str, end: str, cut: str | None = None) -> dict:
    boundary_pts = {"video_start": 0.0, "cut_001": 10.0, "cut_002": 20.0,
                    "cut_003": 25.0, "video_end": 30.0}
    pairs = [(start, cut), (cut, end)] if cut else [(start, end)]
    return {
        "section_id": section_id,
        "shots": [{"shot_id": f"{section_id}.shot_{index:03d}",
                   "start_boundary_id": left, "end_boundary_id": right,
                   "interval": [boundary_pts[left], boundary_pts[right]],
                   "information_added": "a visible action changes another subject's state",
                   "edit_function": "establishes a new stage",
                   "event_relation": ("different_event" if index == 1
                                      else "same_event"),
                   "supporting_deterministic_ids": []}
                  for index, (left, right) in enumerate(pairs, 1)],
        "cut_assessments": ([{"boundary_id": cut, "status": "real_cut",
                              "reason": "visible image change"}] if cut else []),
        "rhythm_claimed": False, "rhythm_evidence_ids": [],
        "unresolved_questions": [],
    }


def _section_bank() -> dict:
    return {"reference_sha256": "a" * 64, "sections": [
        dict(_section_watch("competition", "video_start", "cut_002", "cut_001"),
             source_interval=[0.0, 20.0], raw_response_sha256="b" * 64),
        dict(_section_watch("proofs", "cut_002", "video_end"),
             source_interval=[20.0, 30.0], raw_response_sha256="c" * 64),
    ]}


def _reconciliation() -> dict:
    return {"schema_version": "boundary_reconciliation_v9_p02",
            "boundaries": [
                {"boundary_id": "cut_002", "between": ["competition", "proofs"],
                 "semantic_change": True, "change_types": ["event"],
                 "reason": "a different event begins here",
                 "action": "kept", "moved_to": None}],
            "sections_after": [], "rewatched": []}


def _conflicts_clean() -> dict:
    return {"schema_version": "semantic_conflicts_v9_p02",
            "contradictions": [], "neutral_rechecks": [],
            "unresolved_topics": [], "must_not_assert": []}


class FakeRunner:
    def __init__(self, *, watch=None, ask=None, inspect=None):
        self.watch_values = list(watch or [])
        self.ask_values = list(ask or [])
        self.inspect_values = list(inspect or [])
        self.watch_calls = []
        self.ask_calls = []
        self.inspect_calls = []

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
        self.inspect_calls.append((images, prompt, kwargs))
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
    assert _parse_one_object('```json\n{"ok": true}', stage="test") == {"ok": True}
    for raw in ('{"a": 1} trailing', '{"a": 1}{"b": 2}', '[1, 2]',
                '```json\n{"ok": true} trailing'):
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


def test_production_watch_prompts_do_not_spoil_reference_answer() -> None:
    for prompt in (GLOBAL_WATCH_PROMPT, SECTION_WATCH_PROMPT):
        for spoiler in ("没有双手", "跆拳道", "比赛", "能力展示", "全国冠军"):
            assert spoiler not in prompt


def test_edit_model_does_not_generate_deterministic_style_statistics() -> None:
    assert '"measured_style"' not in EDIT_PROGRAM_PROMPT
    assert "真实 PTS" in EDIT_PROGRAM_PROMPT


def test_content_prompt_spells_out_full_continuity_basis_contract() -> None:
    basis_block = CONTENT_PROGRAM_PROMPT.split('"continuity_basis"', 1)[1]
    for dimension in CONTINUITY_DIMENSIONS:
        assert f'"{dimension}"' in basis_block
    assert "一一对应" in CONTENT_PROGRAM_PROMPT
    assert '"primary_focus": {' in CONTENT_PROGRAM_PROMPT


def test_legacy_cache_without_source_and_tool_hash_is_unavailable(tmp_path: Path) -> None:
    cache = tmp_path / "ocr.json"
    cache.write_text(json.dumps({"output": {"text_events": [{"text": "stale"}]}}),
                     encoding="utf-8")
    output, audit = _read_cached_output(
        cache, reference_sha256="a" * 64, tool_version="tool-v2")
    assert output is None
    assert audit["status"] == "unverified_source"


def test_section_edit_rewatches_source_video_and_keeps_cut_evidence(tmp_path: Path) -> None:
    ledger = _ledger()
    draft = _draft()
    draft["section_drafts"][0]["interval"] = [0.0, 20.0]
    draft["section_drafts"][1]["interval"] = [20.0, 30.0]
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake")
    runner = FakeRunner(watch=[
        _section_watch("competition", "video_start", "cut_002", "cut_001"),
        _section_watch("proofs", "cut_002", "video_end"),
    ])
    bank = build_section_observations(video, draft, ledger, tmp_path,
                                      runner=runner)
    assert len(runner.watch_calls) == 2
    assert runner.watch_calls[0][2]["start_s"] == 0.0
    assert runner.watch_calls[0][2]["end_s"] == 20.0
    assert runner.watch_calls[0][2]["max_new_tokens"] == 4096
    assert '"cut_001"' in runner.watch_calls[0][1]
    assert '"cut_candidates_to_assess":["cut_001"]' in runner.watch_calls[0][1]
    assert bank["sections"][0]["shots"][0]["interval"] == [0.0, 10.0]
    assert (tmp_path / "raw_responses" / "section_competition.txt").is_file()


def test_section_end_boundary_is_not_an_assessable_internal_cut() -> None:
    import src.agentic_video.reference_program_v9 as module

    evidence = module._section_evidence(_ledger(), [0.0, 10.0])
    assert any(row["boundary_id"] == "cut_001"
               for row in evidence["boundaries"])
    assert all(row["boundary_id"] != "cut_001"
               for row in evidence["cut_candidates"])


def test_signal_gap_only_when_current_explanation_is_incomplete() -> None:
    ledger = _ledger()
    complete = {"sections": [{
        "section_id": "whole", "source_interval": [0.0, 30.0],
        "shots": [{"start_boundary_id": "video_start", "end_boundary_id": "cut_001",
                   "information_added": "specific change"},
                  {"start_boundary_id": "cut_001", "end_boundary_id": "cut_002",
                   "information_added": "another specific change"},
                  {"start_boundary_id": "cut_002", "end_boundary_id": "video_end",
                   "information_added": "final specific change"}],
        "cut_assessments": [
            {"boundary_id": "cut_001", "status": "real_cut"},
            {"boundary_id": "cut_002", "status": "not_cut"}],
        "rhythm_claimed": False,
    }]}
    assert collect_reference_gaps(_draft(), complete, ledger) == []
    complete["sections"][0]["cut_assessments"].pop()
    gaps = collect_reference_gaps(_draft(), complete, ledger)
    assert len(gaps) == 1
    assert gaps[0]["gap_source"] == "signal_audit"
    assert gaps[0]["trigger_ids"] == ["cut_002"]


def test_required_signal_gap_upgrades_overlapping_optional_question() -> None:
    optional = {"id": "soft", "question": "粗看是否有阶段变化？",
                "gap_type": "event_structure", "importance": "optional",
                "selected_probe": "dense_video", "interval": [0.0, 10.0],
                "status": "open", "trigger_ids": ["cut_001"]}
    required = {"id": "hard", "question": "每一刀新增了什么信息？",
                "gap_type": "event_structure", "importance": "required",
                "selected_probe": "dense_video", "status": "open",
                "trigger_ids": ["cut_002"]}
    bank = {"sections": [{"section_id": "s", "source_interval": [0.0, 10.0],
                          "shots": [{"supporting_deterministic_ids": []}],
                          "cut_assessments": [],
                          "unresolved_questions": [required]}]}
    gaps = collect_reference_gaps(_draft(questions=[optional]), bank, _ledger())
    assert len(gaps) == 1
    assert gaps[0]["importance"] == "required"
    assert "每一刀新增了什么信息" in gaps[0]["question"]
    assert gaps[0]["trigger_ids"] == ["cut_001", "cut_002"]


def test_optional_native_frame_gap_over_limit_is_kept_open_without_blocking(
        tmp_path: Path) -> None:
    question = {
        "id": "optional_biography", "question": "What age did this happen?",
        "importance": "optional", "gap_type": "visual_detail",
        "selected_probe": "native_frames", "status": "open",
        "interval": [0.0, 30.0],
    }
    runner = FakeRunner()
    result = resolve_reference_questions(
        tmp_path / "reference.mp4", _draft(questions=[question]), _ledger(),
        tmp_path, runner=runner)
    resolved_question = result["questions"][0]
    assert resolved_question["status"] == "open"
    assert resolved_question["probe_disposition"] == "skipped_scope_too_wide"
    assert result["required_unresolved_ids"] == []
    assert result["probe_history"][0]["execution_status"] == "skipped"
    assert result["probe_history"][0]["candidate_frame_count"] == 901
    assert runner.inspect_calls == []


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


@pytest.mark.parametrize("evidence_type", ["unsupported", "inferred"])
def test_required_probe_cannot_resolve_from_unsupported_evidence(
        tmp_path: Path, evidence_type: str) -> None:
    question = {"id": "q1", "question": "画面是否证明了结果？",
                "importance": "required", "gap_type": "event_structure",
                "selected_probe": "dense_video", "interval": [0.0, 10.0],
                "status": "open"}
    runner = FakeRunner(watch=[{
        "question_id": "q1", "status": "resolved", "answer": "当前无法证明",
        "evidence": [{"evidence_type": evidence_type,
                      "description": "画面不支持这一结论"}],
    }])
    result = resolve_reference_questions(
        tmp_path / "reference.mp4", _draft(questions=[question]), _ledger(),
        tmp_path, runner=runner, max_rounds=1)
    assert result["required_unresolved_ids"] == ["q1"]
    assert result["questions"][0]["status"] == "open"


def test_probe_selection_must_match_the_unresolved_gap(tmp_path: Path) -> None:
    question = {"id": "q1", "question": "where are the beats",
                "importance": "required", "gap_type": "rhythm",
                "selected_probe": "dense_video", "interval": [0.0, 10.0],
                "status": "open"}
    with pytest.raises(V9Blocked, match="gap='rhythm'"):
        resolve_reference_questions(
            tmp_path / "reference.mp4", _draft(questions=[question]), _ledger(),
            tmp_path, runner=FakeRunner(), max_rounds=1)


def test_ocr_probe_carries_verified_text_events_and_source_interval(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agentic_video.reference_program_v9 as module

    ledger = _ledger()
    ledger["ocr"]["normalized_claim_events"] = [{
        "claim_id": "ocr_001", "text": "a displayed claim",
        "interval": [1.0, 2.0],
    }]
    ledger["ocr_provenance"] = {"status": "reused_verified_artifact",
                                 "source_sha256": ledger["reference"]["sha256"]}
    question = {"id": "q_ocr", "question": "what does text convey?",
                "importance": "required", "gap_type": "text_claim",
                "selected_probe": "ocr_context", "interval": [0.0, 10.0],
                "status": "open"}
    frame = tmp_path / "ocr_frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(module, "_ocr_probe_frames", lambda *args, **kwargs: (
        [frame], [{"claim_id": "ocr_001", "frame_id": "f000030",
                   "pts_s": 1.0, "path": str(frame), "sha256": "frame_hash"}]))
    monkeypatch.setattr(module, "cut_clip", lambda *args, **kwargs: tmp_path / "clip.mp4")
    runner = FakeRunner(inspect=[{
        "question_id": "q_ocr", "status": "resolved", "answer": "text makes a claim",
        "evidence": [{"evidence_type": "observed_textual_claim",
                      "description": "words appear on screen"}],
    }])
    result = resolve_reference_questions(
        tmp_path / "reference.mp4", _draft(questions=[question]), ledger,
        tmp_path, runner=runner)
    assert not result["required_unresolved_ids"]
    assert "ocr_001" in runner.inspect_calls[0][1]
    assert runner.inspect_calls[0][0] == [frame]
    assert runner.inspect_calls[0][2]["video_path"] == tmp_path / "clip.mp4"
    audit = result["probe_history"][0]
    assert "ocr_text_events" in audit["evidence_fields"]
    assert audit["evidence_provenance"]["source_sha256"] == "a" * 64
    assert Path(audit["input_path"]).is_file()


def test_ocr_probe_refuses_unverified_cache(tmp_path: Path) -> None:
    question = {"id": "q_ocr", "question": "what does text convey?",
                "importance": "required", "gap_type": "text_claim",
                "selected_probe": "ocr_context", "interval": [0.0, 10.0],
                "status": "open"}
    with pytest.raises(V9Blocked, match="ocr_context"):
        resolve_reference_questions(
            tmp_path / "reference.mp4", _draft(questions=[question]),
            _ledger(), tmp_path, runner=FakeRunner())


def test_program_validation_accepts_event_compression_and_evidence_montage(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    requirements = compile_material_requirements(
        content, edit, tmp_path,
        normalization=normalize_section_shots(_section_bank(), _ledger()))
    validation = validate_reference_programs(content, edit, requirements, _ledger(),
                                             _section_bank(),
                                             reconciliation=_reconciliation())
    assert validation["passed"], validation["errors"]
    event = requirements["requirements"][0]
    presentation = event["presentation_requirement"]
    assert {key: presentation[key] for key in (
        "target_duration_s", "composition_mode", "snippet_count_range",
        "individual_duration_policy", "source_continuity",
        "semantic_continuity", "ordering_constraint",
        "audience_requirement")} == {
        "target_duration_s": 5.0,
        "composition_mode": "event_compression_montage",
        "snippet_count_range": [2, 4],
        "individual_duration_policy": "minimum_sufficient_duration",
        "source_continuity": "non_contiguous_allowed",
        "semantic_continuity": "required",
        "ordering_constraint": "preserve_event_progression",
        "audience_requirement": "blind viewer understands progression",
    }
    assert presentation["reference_content_shot_count"] == 2
    assert presentation["reference_transition_count"] == 0
    assert event["semantic_requirement"]["required_event_understanding"].startswith(
        "understand_full_source_event")
    assert requirements["requirements"][1]["continuity_requirement"][
        "levels"]["event"] == "not_required"


def test_edit_validation_rejects_missing_section_rewatch_or_shot_refs(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    req = compile_material_requirements(content, edit, tmp_path)
    assert validate_reference_programs(content, edit, req, _ledger(),
                                       _section_bank(),
                                       reconciliation=_reconciliation())["passed"]
    assert any("direct Section watch" in message for message in
               validate_reference_programs(content, edit, req, _ledger())["errors"])
    edit["editorial_patterns"][0]["shot_refs"] = []
    assert any("shot refs missing" in message for message in
               validate_reference_programs(content, edit, req, _ledger(),
                                           _section_bank(),
                                           reconciliation=_reconciliation())["errors"])


def test_two_coarse_windows_cannot_swallow_many_meaningful_units(tmp_path: Path) -> None:
    content = _content()
    content["meaningful_units"].append({
        "unit_id": "unit_04", "purpose": "new detail", "start_boundary_id": "cut_002",
        "end_boundary_id": "video_end", "evidence_ids": ["ev_text"]})
    content["sections"][1]["unit_ids"].append("unit_04")
    requirements = compile_material_requirements(content, _edit(), tmp_path)
    result = validate_reference_programs(content, _edit(), requirements, _ledger(),
                                         _section_bank())
    assert any("two-window collapse" in error for error in result["errors"])


def test_claim_cannot_be_upgraded_to_visual_fact(tmp_path: Path) -> None:
    content = _content()
    content["reference_observations"][1]["verified_as_visual_fact"] = True
    requirements = compile_material_requirements(content, _edit(), tmp_path)
    result = validate_reference_programs(content, _edit(), requirements, _ledger(),
                                         _section_bank())
    assert any("claim upgraded" in error for error in result["errors"])


def test_requirement_transfer_gate_rejects_reference_and_target_leakage(tmp_path: Path) -> None:
    content = _content()
    content["sections"][0]["transferable_structure"] = "找一个没有双手的跆拳道女性"
    requirements = compile_material_requirements(content, _edit(), tmp_path)
    result = validate_reference_programs(content, _edit(), requirements, _ledger(),
                                         _section_bank())
    assert any("leaked" in error for error in result["errors"])
    requirements["requirements"][0]["semantic_requirement"]["meaning_to_prove"] = "找小黑"
    result = validate_reference_programs(content, _edit(), requirements, _ledger(),
                                         _section_bank())
    assert any("小黑" in error for error in result["errors"])


def test_event_compression_order_evidence_montage_causality_and_dialogue_integrity(
        tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    edit["editorial_patterns"][0]["ordering_constraint"] = "source_order"
    edit["editorial_patterns"][1]["source_semantics"] = "one_long_event"
    requirements = compile_material_requirements(content, edit, tmp_path)
    requirements["requirements"][1]["continuity_requirement"]["levels"]["event"] = "required"
    result = validate_reference_programs(content, edit, requirements, _ledger(),
                                         _section_bank())
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
    result = validate_reference_programs(content, edit, requirements, _ledger(),
                                         _section_bank())
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
    edit = build_reference_edit_program(
        content, ledger, tmp_path, runner=runner,
        section_observations=_section_bank())
    assert len(runner.ask_calls) == 2
    assert all(kwargs["stop_after_json_object"] is True
               for _prompt, kwargs in runner.ask_calls)
    requirements = compile_material_requirements(content, edit, tmp_path)
    result = validate_reference_programs(content, edit, requirements, ledger,
                                         _section_bank(),
                                         reconciliation=_reconciliation())
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
    runner = FakeRunner(watch=[
        _draft(),
        _section_watch("competition", "video_start", "cut_002", "cut_001"),
        _section_watch("proofs", "cut_002", "video_end"),
    ], ask=[
        _conflicts_clean(),
        _content(), _edit(),
    ], inspect=[{"event_continuity": False,
                 "rhetorical_function_continuity": False,
                 "audience_cognition_continuity": False,
                 "semantic_change": True, "reason": "new event begins"}])
    def fake_ledger(_reference, output_dir, **_kwargs):
        ledger = _ledger()
        (Path(output_dir) / "reference_evidence.json").write_text(
            json.dumps(ledger), encoding="utf-8")
        return ledger
    monkeypatch.setattr(module, "build_reference_evidence_ledger",
                        fake_ledger)
    monkeypatch.setattr(module, "_build_review_assets",
                        lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_boundary_frames",
                        lambda ffmpeg_bin, reference, pts, out_dir, span:
                        [out_dir / f"f{index}.jpg" for index in range(6)])
    cfg = SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"})
    result = run_reference_program_v9(cfg, video, tmp_path / "run",
                                      runner=runner, force=True)
    assert result["decision"] == "PENDING_HUMAN"
    assert result["passed"] is False
    assert result["long_video_perception_authorized"] is False
    assert (tmp_path / "run" / "human_review.template.json").is_file()
    assert not (tmp_path / "run" / "frozen_program_hashes.json").exists()
    for stage in ("p0_a", "p0_b", "p0_c", "p0_d"):
        assert list((tmp_path / "run" / "stages" / stage).glob("*/manifest.json"))


def test_human_acceptance_requires_every_section_and_transfer_test(tmp_path: Path) -> None:
    (tmp_path / "reference_evidence.json").write_text(
        json.dumps(_ledger()), encoding="utf-8")
    (tmp_path / "section_observations.json").write_text(
        json.dumps(_section_bank()), encoding="utf-8")
    (tmp_path / "boundary_reconciliation.json").write_text(
        json.dumps(_reconciliation()), encoding="utf-8")
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
        "schema_version": HUMAN_REVIEW_VERSION,
        "reviewer": "human reviewer",
        "sections": [{"section_id": section_id, "passed": True, "checks": dict(checks)}
                     for section_id in ("competition", "proofs")],
        "requirement_transfer_test": {"passed": False},
        "review_binding": _review_binding(tmp_path),
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    assert accept_reference_programs(tmp_path, path)["decision"] == "BLOCKED"
    review["requirement_transfer_test"]["passed"] = True
    path.write_text(json.dumps(review), encoding="utf-8")
    result = accept_reference_programs(tmp_path, path)
    assert result["decision"] == "PASS"
    assert (tmp_path / "frozen_program_hashes.json").is_file()


def test_human_review_rejects_changed_program_after_signoff(tmp_path: Path) -> None:
    content = _content()
    edit = _edit()
    (tmp_path / "reference_evidence.json").write_text(
        json.dumps(_ledger()), encoding="utf-8")
    (tmp_path / "section_observations.json").write_text(
        json.dumps(_section_bank()), encoding="utf-8")
    (tmp_path / "boundary_reconciliation.json").write_text(
        json.dumps(_reconciliation()), encoding="utf-8")
    (tmp_path / "reference_content_program.json").write_text(
        json.dumps(content), encoding="utf-8")
    (tmp_path / "reference_edit_program.json").write_text(
        json.dumps(edit), encoding="utf-8")
    compile_material_requirements(content, edit, tmp_path)
    (tmp_path / "acceptance.json").write_text(json.dumps({
        "automated_passed": True, "validation": {"passed": True}}), encoding="utf-8")
    checks = dict.fromkeys((
        "visible_content_correct", "event_phases_correct",
        "composition_mode_correct", "text_audio_rhythm_role_correct",
        "material_requirement_searchable"), True)
    review = {"schema_version": HUMAN_REVIEW_VERSION,
              "review_binding": _review_binding(tmp_path), "reviewer": "reviewer",
              "sections": [{"section_id": name, "passed": True, "checks": checks}
                           for name in ("competition", "proofs")],
              "requirement_transfer_test": {"passed": True}}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    content["sections"][0]["audience_takeaway"] = "changed after review"
    (tmp_path / "reference_content_program.json").write_text(
        json.dumps(content), encoding="utf-8")
    result = accept_reference_programs(tmp_path, path)
    assert result["decision"] == "BLOCKED"
    assert result["reason_code"] == "stale_human_review"
    assert not (tmp_path / "frozen_program_hashes.json").exists()


def test_unknown_continuity_requires_traced_amendment_and_new_transfer_review(
        tmp_path: Path) -> None:
    content = _content()
    content["sections"][0]["continuity"]["opponent"] = "unknown"
    (tmp_path / "reference_evidence.json").write_text(
        json.dumps(_ledger()), encoding="utf-8")
    (tmp_path / "section_observations.json").write_text(
        json.dumps(_section_bank()), encoding="utf-8")
    (tmp_path / "boundary_reconciliation.json").write_text(
        json.dumps(_reconciliation()), encoding="utf-8")
    (tmp_path / "reference_content_program.json").write_text(
        json.dumps(content), encoding="utf-8")
    (tmp_path / "reference_edit_program.json").write_text(
        json.dumps(_edit()), encoding="utf-8")
    compile_material_requirements(content, _edit(), tmp_path)
    (tmp_path / "acceptance.json").write_text(json.dumps({
        "automated_passed": True, "validation": {"passed": True}}), encoding="utf-8")
    checks = dict.fromkeys((
        "visible_content_correct", "event_phases_correct",
        "composition_mode_correct", "text_audio_rhythm_role_correct",
        "material_requirement_searchable"), True)
    review = {"schema_version": HUMAN_REVIEW_VERSION,
              "reviewer": "human reviewer",
              "sections": [{"section_id": name, "passed": True, "checks": checks}
                           for name in ("competition", "proofs")],
              "requirement_transfer_test": {"passed": True},
              "continuity_resolutions": [],
              "review_binding": _review_binding(tmp_path)}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    assert accept_reference_programs(tmp_path, path)["decision"] == "BLOCKED"
    assert not (tmp_path / "frozen_program_hashes.json").exists()
    review["continuity_resolutions"] = [{
        "section_id": "competition", "dimension": "opponent",
        "level": "not_required", "reason": "source frames show no recurring opponent",
        "evidence_ids": ["ev_visual"],
    }]
    path.write_text(json.dumps(review), encoding="utf-8")
    amended = accept_reference_programs(tmp_path, path)
    assert amended["decision"] == "BLOCKED"
    assert "continuity_amendment_requires_rereview" in amended["detail"]
    assert not (tmp_path / "frozen_program_hashes.json").exists()
    review["continuity_resolutions"] = []
    review["review_binding"] = _review_binding(tmp_path)
    path.write_text(json.dumps(review), encoding="utf-8")
    assert accept_reference_programs(tmp_path, path)["decision"] == "PASS"
