from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agentic_video.evidence_matcher import (
    build_v61_edit_plan, oracle_pattern_matches, validate_pattern_contract,
)
from src.agentic_video.evidence_units import EvidenceUnitV2, evidence_v2_from_dict
from src.perception.dual_resolution_miner import (
    _analysis_intervals, merge_coarse_regions, merge_observed_evidence,
    parse_coarse_response, parse_dense_response,
    validate_sampling,
)
from src.perception.omni_runner import OmniRunner


def _unit(unit_id, interval, *, form="dynamic_action", subject="subject_A",
          action="action", attributes=("motion",)):
    return EvidenceUnitV2(
        id=unit_id, core_interval=interval,
        container_interval=(interval[0] - .3, interval[1] + .3),
        observation={"subject_call_id": "person_A", "action": action},
        source_form=form, attributes=attributes, visual_strength=.9,
        confidence=.9, subject_track_id=subject, source_video="source.mp4")


def _slot(slot_id, *, forms=None, group=None, fusion=True):
    row = {
        "id": slot_id, "required": True,
        "allowed_source_forms": forms or ["dynamic_action"],
        "render_modes": ["micro_clip", "keyframe_hold"],
        "slot_fit_threshold": .7, "allow_slot_fusion": fusion,
    }
    if group:
        row["same_subject_group"] = group
    return row


def test_evidence_v2_is_fact_only_and_reads_legacy_v6():
    unit = _unit("v2", (10, 10.8))
    payload = unit.to_dict()
    assert "semantic_role" not in payload
    assert "render_mode" not in payload
    legacy = evidence_v2_from_dict({
        "id": "old", "interval": [20, 21], "unit_type": "dialogue_excerpt",
        "editing_role": "hook", "description": "提出问题", "confidence": .8,
    })
    assert legacy.source_form == "dialogue_span"
    assert legacy.metadata["legacy_v6"]["editing_role"] == "hook"


def test_sampling_audit_uses_reader_indices_and_absolute_origin():
    class Video:
        shape = (12, 3, 224, 224)

    audit = OmniRunner._sampling_audit(
        Video(), {"fps": 60, "total_num_frames": 360,
                  "frames_indices": list(range(0, 360, 30)),
                  "video_backend": "fake"},
        requested_fps=2, source_origin_s=100)
    assert audit["sampling_verified"] is True
    assert audit["actual_frame_count"] == 12
    assert audit["effective_fps"] == 2
    assert audit["actual_frame_timestamps_absolute_s"][1] == 100.5


def test_sampling_gate_distinguishes_coarse_dense_and_missing_metadata():
    valid = {"sampling_verified": True, "actual_frame_count": 12,
             "effective_fps": 2.0,
             "actual_frame_timestamps_absolute_s": list(range(12))}
    assert validate_sampling(valid, fps_min=1.5, fps_max=2.5) == []
    assert validate_sampling(valid, fps_min=10, fps_max=13) == [
        "effective_fps_outside_range:2"]
    assert validate_sampling({}, fps_min=1.5, fps_max=2.5) == [
        "sampling_metadata_unverified"]


def test_transport_windows_are_six_seconds_and_cover_scope():
    rows = _analysis_intervals(
        {"shots": [{"start_s": 0, "end_s": .2}]}, (100, 115),
        max_watch_s=6, overlap_s=.5)
    assert rows[0][1:] == (100, 106)
    assert rows[-1][1:] == (109, 115)
    assert all(right - left == pytest.approx(6)
               for _, left, right in rows)


def test_coarse_regions_are_transport_hints_not_editorial_roles():
    raw = json.dumps({"timebase": "relative", "candidate_regions": [{
        "interval": [1, 1.5], "source_form": "dynamic_action",
        "observation": {"action": "falls"}, "attributes": ["knockdown"],
        "salience": .9, "confidence": .8,
    }]})
    rows, rejected = parse_coarse_response(
        raw, watch_id="c0", container=(100, 106), scope=(100, 110),
        source_video="movie.mp4")
    assert not rejected
    assert rows[0]["interval"] == [100.25, 102.25]
    assert "semantic_role" not in rows[0]


def test_nested_omni_fact_fields_are_normalized_without_changing_observation():
    raw = json.dumps({"timebase": "relative", "candidate_regions": [{
        "interval": [1, 3], "observation": {
            "subject_call_id": "person_A", "action": "speaks",
            "source_form": "dialogue_span", "attributes": ["speech"],
            "salience": .8, "confidence": .9,
        }}]})
    rows, rejected = parse_coarse_response(
        raw, watch_id="c0", container=(100, 106), scope=(100, 110),
        source_video="movie.mp4")
    assert not rejected
    assert rows[0]["source_form"] == "dialogue_span"
    assert rows[0]["attributes"] == ["speech"]
    assert rows[0]["salience"] == .8
    assert "source_form" not in rows[0]["observation"]


def test_dense_core_longer_than_two_seconds_is_not_rejected():
    raw = json.dumps({"timebase": "relative", "evidence": [{
        "core_interval": [1, 3.6], "source_form": "dynamic_action",
        "observation": {"action": "falls_and_rolls"}, "confidence": .9,
    }]})
    units, rejected = parse_dense_response(
        raw, watch_id="d0", coarse_watch_id="c0", container=(10, 16),
        scope=(10, 20), source_video="movie.mp4")
    assert not rejected
    assert units[0].duration_s == pytest.approx(2.6)


def test_overlap_chunks_merge_same_fact_and_keep_provenance():
    left = EvidenceUnitV2(
        id="a", core_interval=(5.7, 6.0), container_interval=(0, 6),
        observation={"subject_description": "黑发少年", "action": "falls"},
        source_form="dynamic_action", attributes=("knockdown",),
        visual_strength=.8, confidence=.8, source_video="movie.mp4",
        sampling_provenance={"dense_rewatch_id": "d0"})
    right = EvidenceUnitV2(
        id="b", core_interval=(5.7, 6.4), container_interval=(5.5, 11.5),
        observation={"subject_description": "黑发少年", "action": "falls"},
        source_form="dynamic_action", attributes=("knockdown",),
        visual_strength=.9, confidence=.9, source_video="movie.mp4",
        sampling_provenance={"dense_rewatch_id": "d1"})
    merged = merge_observed_evidence([left, right])
    assert len(merged) == 1
    assert merged[0].core_interval == (5.7, 6.4)
    assert merged[0].metadata["merged_from_watch_ids"] == ["d0", "d1"]


def test_overlap_coarse_regions_are_deduplicated_before_dense_budget():
    rows = [{
        "id": "a", "interval": [5.0, 7.0], "source_form": "dynamic_action",
        "observation": {"action": "falls"}, "attributes": ["knockdown"],
        "salience": .8, "confidence": .8, "source_video": "movie.mp4",
        "coarse_watch_id": "c0",
    }, {
        "id": "b", "interval": [5.5, 7.5], "source_form": "dynamic_action",
        "observation": {"action": "falls"}, "attributes": ["knockdown"],
        "salience": .9, "confidence": .9, "source_video": "movie.mp4",
        "coarse_watch_id": "c1",
    }]
    merged = merge_coarse_regions(rows)
    assert len(merged) == 1
    assert merged[0]["interval"] == [5.0, 7.5]
    assert merged[0]["merged_from_watch_ids"] == ["c0", "c1"]


def test_pattern_contract_separates_source_forms_and_render_modes():
    assert not validate_pattern_contract({"slots": [_slot("hook")]})
    reasons = validate_pattern_contract({"slots": [{
        "id": "hook", "allowed_source_forms": ["micro_clip"],
        "render_modes": ["dynamic_action"],
    }]})
    assert "pattern_source_form_unsupported:hook" in reasons
    assert "pattern_render_mode_unsupported:hook" in reasons


def test_dialogue_hook_is_preserved_but_not_eligible_as_visual_hook():
    unit = _unit("dialogue", (1, 2), form="dialogue_span")
    bank = {"units": [unit.to_dict()]}
    pattern = {"slots": [_slot("hook", forms=["visual_instant", "dynamic_action"])]}
    matches = {"matches": [{"evidence_id": "dialogue", "slot_fit": {"hook": .95}}]}
    plan = build_v61_edit_plan(bank, pattern, matches, min_duration_s=.1)
    assert plan["semantic_hook_available"] is True
    assert plan["eligible_visual_hook"] is False
    assert plan["slot_diagnostics"][0]["excluded"][0]["source_form"] == "dialogue_span"


def test_main_arc_rejects_subject_switch():
    units = [
        _unit("hook", (1, 2), subject="subject_H"),
        _unit("struggle", (3, 4), subject="subject_A"),
        _unit("reversal", (5, 6), subject="subject_B"),
        _unit("payoff", (7, 8), subject="subject_A"),
    ]
    pattern = {"slots": [
        _slot("hook"), _slot("struggle", group="arc"),
        _slot("reversal", group="arc"), _slot("payoff", group="arc"),
    ]}
    matches = {"matches": [{"evidence_id": unit.id,
                             "slot_fit": {unit.id: 1.0}} for unit in units]}
    plan = build_v61_edit_plan(
        {"units": [unit.to_dict() for unit in units]}, pattern, matches,
        min_duration_s=.1)
    assert plan["passed"] is False
    assert "reversal" in plan["missing_slots"]


def test_pattern_can_explicitly_allow_subject_switch():
    units = [_unit("a", (1, 2), subject="subject_A"),
             _unit("b", (3, 4), subject="subject_B")]
    pattern = {"subject_switch_allowed": True, "slots": [
        _slot("struggle", group="arc"), _slot("reversal", group="arc")]}
    matches = {"matches": [
        {"evidence_id": "a", "slot_fit": {"struggle": 1.0}},
        {"evidence_id": "b", "slot_fit": {"reversal": 1.0}}]}
    plan = build_v61_edit_plan(
        {"units": [unit.to_dict() for unit in units]}, pattern, matches,
        min_duration_s=.1)
    assert plan["passed"] is True
    assert [row["subject_track_id"] for row in plan["segments"]] == [
        "subject_A", "subject_B"]


def test_adjacent_slots_can_fuse_and_render_once():
    unit = _unit("hit", (5, 6))
    bank = {"units": [unit.to_dict()]}
    pattern = {"slots": [_slot("reversal"), _slot("payoff")]}
    matches = {"matches": [{"evidence_id": "hit",
                             "slot_fit": {"reversal": .9, "payoff": .9}}]}
    plan = build_v61_edit_plan(bank, pattern, matches, min_duration_s=.1)
    assert plan["passed"] is True
    assert len(plan["segments"]) == 1
    assert plan["segments"][0]["assigned_slots"] == ["reversal", "payoff"]
    assert plan["segments"][0]["render_once"] is True


def test_oracle_matches_bypass_automatic_matcher():
    unit = _unit("oracle_fact", (5, 6))
    oracle = {"facts": [{"id": "oracle_fact", "expected_slots": ["payoff"]}]}
    matches = oracle_pattern_matches({"units": [unit.to_dict()]}, oracle)
    assert matches["oracle"] is True
    assert matches["matches"][0]["slot_fit"] == {"payoff": 1.0}


def test_v61_diagnostic_fake_runner_synthetic_media(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from src.agentic_video.evidence_diagnostic import run_evidence_diagnostic

    source = tmp_path / "source.mp4"
    reference = tmp_path / "reference.mp4"
    bgm = tmp_path / "bgm.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=320x180:rate=24:duration=20", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=20", "-shortest", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)], check=True)
    shutil.copy2(source, reference)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=220:duration=20", "-c:a", "aac", str(bgm)], check=True)

    class FakeRunner:
        gpu_pairs = ["0,1", "2,3", "4,5", "6,7"]

        @staticmethod
        def _sampling(request):
            kwargs = request.get("kwargs") or {}
            fps = float(kwargs.get("fps", 2))
            duration = float(kwargs["end_s"] - kwargs["start_s"])
            count = max(1, round(fps * duration))
            return {"sampling_verified": True, "actual_frame_count": count,
                    "effective_fps": fps,
                    "actual_frame_timestamps_absolute_s": [
                        kwargs["start_s"] + index / fps for index in range(count)]}

        def watch_many(self, requests):
            answers = []
            for index, request in enumerate(requests):
                prompt = request["prompt"]
                if "candidate_regions" in prompt:
                    text = json.dumps({"timebase": "relative", "candidate_regions": [{
                        "interval": [1, 3], "source_form": "dynamic_action",
                        "observation": {"subject_call_id": "person_A",
                                        "subject_description": "同一人物",
                                        "action": "visible change"},
                        "attributes": ["motion"], "salience": .9, "confidence": .9,
                    }]})
                elif '"evidence"' in prompt:
                    text = json.dumps({"timebase": "relative", "evidence": [{
                        "core_interval": [1.2, 2.0], "source_form": "dynamic_action",
                        "observation": {"subject_call_id": "person_A",
                                        "subject_description": "同一人物",
                                        "action": "visible change"},
                        "attributes": ["motion"], "visual_strength": .9,
                        "confidence": .9,
                    }]})
                else:
                    text = json.dumps({
                        "core_message": "同一人物经历困境后反击并得到结果",
                        "hook_clear": True, "montage_coherent": True,
                        "functionless_span_present": False,
                        "visible_evidence_roles": ["困境", "反击", "结果"],
                        "audible_dialogue_present": True, "speech_clear": True,
                        "music_present": index == 1,
                    })
                answers.append(SimpleNamespace(
                    text=text,
                    sampling=(self._sampling(request)
                              if "start_s" in (request.get("kwargs") or {}) else None),
                    gpu_pair=self.gpu_pairs[index % 4]))
            return answers

        def ask(self, prompt, **_kwargs):
            if "局部人物轨迹" in prompt:
                rows = json.loads(prompt.split("输入：", 1)[1])
                return SimpleNamespace(text=json.dumps({"links": [
                    {"evidence_id": row["evidence_id"],
                     "subject_track_id": "local_subject_01", "confidence": .9}
                    for row in rows], "uncertain_evidence_ids": []}))
            evidence = json.loads(prompt.split("Evidence：", 1)[1])
            return SimpleNamespace(text=json.dumps({"matches": [
                {"evidence_id": row["evidence_id"],
                 "slot_fit": {"hook": .9, "struggle": .9,
                              "reversal": .9, "payoff": .9}}
                for row in evidence]}))

    pattern = {"slots": [
        _slot("hook"), _slot("struggle", group="arc"),
        _slot("reversal", group="arc"), _slot("payoff", group="arc"),
    ]}
    spec = {
        "spec_version": "evidence_diagnostic_v1", "source": "test",
        "reference": str(reference), "bgm_path": str(bgm),
        "controls": {"w15": {"start_s": 0, "end_s": 5},
                     "action": {"start_s": 5, "end_s": 10}},
        "pattern": pattern,
        "perception": {"coarse_fps": 2, "dense_fps": 12,
                       "max_watch_s": 6, "watch_overlap_s": .5,
                       "dense_padding_s": .5, "max_dense_regions": 8,
                       "source_scene_threshold": .3,
                       "min_shot_duration_s": .15},
        "duration": {"min_s": 4, "max_s": 20, "max_segment_s": 4,
                     "pre_roll_s": .3, "post_roll_s": .2},
    }
    facts = []
    for index, (slot, left) in enumerate(zip(
            ["hook", "struggle", "reversal", "payoff"], [5.2, 6.4, 7.6, 8.8])):
        facts.append({
            "id": f"oracle_{index}", "core_interval": [left, left + .8],
            "container_interval": [left - .3, left + 1.1],
            "observation": {"subject_call_id": "person_A", "action": slot},
            "subject_track_id": "local_subject_01",
            "source_form": "dynamic_action", "attributes": [slot],
            "expected_slots": [slot],
        })
    oracle = {"schema_version": "evidence_oracle_v1",
              "control_scope_valid": True, "facts": facts}
    cfg = SimpleNamespace(
        perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"},
        generation={"assemble": {"width": 320, "height": 180}},
        library={"narrative_render": {"bgm_path": str(bgm),
                                      "bgm_mix_volume": .25}},
        paths=SimpleNamespace(library_dir=tmp_path / "library"))
    result = run_evidence_diagnostic(
        cfg, spec, tmp_path / "run", runner=FakeRunner(),
        oracle_path=oracle, source_video=source)
    assert result["acceptance"]["diagnostic_completed"] is True
    assert result["diagnosis"]["oracle_plan_passed"] is True
    assert (tmp_path / "run" / "arms" / "oracle_planner" / "preview.mp4").is_file()
    assert not (tmp_path / "run" / "rendered.mp4").exists()
