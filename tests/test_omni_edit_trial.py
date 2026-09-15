from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.cli import build_parser
from src.agentic_video.omni_edit_trial import (
    OmniEditTrialBlocked,
    normalize_edit_plan,
    run_omni_edit_trial,
    validate_perception,
)
from src.config import load_config


def _perception() -> dict:
    return {
        "status": "observed",
        "subjects": [
            {"subject_id": "S0", "appearance": "small white child",
             "anchor_match": "S0"},
            {"subject_id": "N0", "appearance": "large dark beast",
             "anchor_match": "N0"},
        ],
        "events": [
            {"event_id": "e1", "interval": [4.2, 5.2], "initiator": "S0",
             "affected": "N0", "visible_action": "S0 raises an arm toward N0",
             "state_before": "N0 approaches", "state_after": "N0 moves away",
             "relation_supported": True, "target_involvement": "direct"},
            {"event_id": "e2", "interval": [5.2, 8.0], "initiator": None,
             "affected": "N0", "visible_action": "N0 travels backward and lands",
             "state_before": "airborne", "state_after": "on ground",
             "relation_supported": True, "target_involvement": "related"},
        ],
        "claim_bounds": {"can_prove": ["visible displacement"],
                         "cannot_prove": ["the whole fight ended"]},
        "scene_summary": "A small white child acts before a large dark subject lands.",
    }


def test_cli_exposes_controlled_omni_edit_trial() -> None:
    args = build_parser().parse_args(["omni-edit-trial", "--output", "run"])
    assert args.command == "omni-edit-trial"
    assert args.spec.endswith("lxh1_omni_edit_trial.json")


def test_perception_keeps_human_target_and_hard_negative_separate() -> None:
    result = validate_perception(_perception(), duration_s=12.0)
    assert result["identity_source"] == "human_confirmed_anchors"
    assert result["target_event_ids"] == ["e1", "e2"]
    bad = _perception()
    bad["subjects"][1]["anchor_match"] = "S0"
    with pytest.raises(OmniEditTrialBlocked, match="hard_negative_merged_into_target"):
        validate_perception(bad, duration_s=12.0)


def test_plan_is_evidence_bound_source_ordered_and_scoped(tmp_path: Path) -> None:
    evidence = validate_perception(_perception(), duration_s=12.0)
    source = tmp_path / "source.mp4"
    plan = normalize_edit_plan({
        "passed": True,
        "summary": "target action and visible result",
        "segments": [
            {"id": "a", "relative_interval": [4.0, 5.4],
             "purpose": "action", "evidence_event_ids": ["e1"]},
            {"id": "b", "relative_interval": [5.4, 8.2],
             "purpose": "result", "evidence_event_ids": ["e2"]},
        ],
    }, perception=evidence, context_start_s=5408.0,
        editing_scope=[5410.0, 5419.0], source_video=source,
        max_segments=3, max_total_s=9.0)
    assert plan["duration_s"] == pytest.approx(4.2)
    assert plan["segments"][0]["render_interval"] == [5412.0, 5413.4]
    assert plan["identity_automation_claimed"] is False

    with pytest.raises(OmniEditTrialBlocked, match="outside_editing_scope"):
        normalize_edit_plan({
            "passed": True,
            "segments": [{"relative_interval": [0.0, 3.0],
                          "evidence_event_ids": ["e1"]}],
        }, perception=evidence, context_start_s=5408.0,
            editing_scope=[5410.0, 5419.0], source_video=source,
            max_segments=3, max_total_s=9.0)


def test_trial_always_delivers_debug_not_formal(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agentic_video import omni_edit_trial as module

    source = tmp_path / "source.mp4"
    bgm = tmp_path / "bgm.m4a"
    source.write_bytes(b"source")
    bgm.write_bytes(b"bgm")
    context = tmp_path / "context.mp4"
    context.write_bytes(b"context")

    def fake_export(_ffmpeg, _source, _time, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"image")
        return destination

    render_dir = tmp_path / "out" / "render"
    variants = {
        "source_only": render_dir / "variants" / "source_only" / "rendered.mp4",
        "bgm_mix": render_dir / "variants" / "bgm_mix" / "rendered.mp4",
    }
    for path in variants.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    monkeypatch.setattr(module, "cut_clip", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(module, "export_frame", fake_export)
    monkeypatch.setattr(module, "render_micro_montage", lambda *_args, **_kwargs: {
        "content_master": render_dir / "content_master.mp4",
        "variants": variants, "duration_s": 4.2})
    monkeypatch.setattr(module, "review_planned_segments",
                        lambda *_args, **_kwargs: {"passed": True})
    blind_row = {
        "parsed": True, "hook_clear": True, "montage_coherent": True,
        "functionless_span_present": False, "too_short_intervals": [],
        "redundant_intervals": [], "subject_relation_clear": True,
        "supported_result": "N0 lands", "core_message": "S0 acts and N0 lands",
        "audible_dialogue_present": False, "music_present": True,
    }
    monkeypatch.setattr(module, "blind_review_variants",
                        lambda *_args, **_kwargs: {
                            "source_only": dict(blind_row, music_present=False),
                            "bgm_mix": dict(blind_row)})

    class FakeRunner:
        def inspect_media(self, *_args, **_kwargs):
            return SimpleNamespace(text=json.dumps(_perception()), input_tokens=10,
                                   output_tokens=5, elapsed_s=.1)

        def ask(self, *_args, **_kwargs):
            return SimpleNamespace(text=json.dumps({
                "passed": True, "summary": "target action and result",
                "segments": [
                    {"id": "a", "relative_interval": [4.0, 5.4],
                     "purpose": "action", "evidence_event_ids": ["e1"]},
                    {"id": "b", "relative_interval": [5.4, 8.2],
                     "purpose": "result", "evidence_event_ids": ["e2"]},
                ]}), input_tokens=8, output_tokens=4, elapsed_s=.1)

    spec = {
        "spec_version": "omni_editorial_trial_v1",
        "context_interval": [5408.0, 5420.0],
        "editing_scope": [5410.0, 5419.0],
        "identity_anchors": {"positive_source_times_s": [5412.75, 5417.0],
                             "hard_negative_source_times_s": [5418.0]},
        "perception_fps": 12.0, "max_segments": 3, "max_total_s": 9.0,
    }
    output = tmp_path / "out"
    result = run_omni_edit_trial(
        load_config(), spec, output, source_video=source, bgm_path=bgm,
        runner=FakeRunner())
    acceptance = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert result["delivery"] == "debug_only"
    assert acceptance["automated_editorial_passed"] is True
    assert (output / "debug_preview.mp4").is_file()
    assert not (output / "rendered.mp4").exists()
