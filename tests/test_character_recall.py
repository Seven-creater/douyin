from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video import character_recall as v82
from src.agentic_video.cli import build_parser
from src.config import load_config


def test_v82_spec_is_target_centric_subject_policy() -> None:
    spec, digest = v82.read_v82_spec(Path(
        "config/experiments/lxh1_v82_target_recall.json"))
    assert len(digest) == 64
    assert spec["perception_objective"] == {
        "center_type": "subject",
        "continuity_key": "character_identity",
        "target_id": "char:xiaohei",
        "harvest_goal": "target_related_scenes",
        "harvest_policy": "subject",
    }
    source = Path(v82.__file__).read_text(encoding="utf-8").lower()
    assert "import yolo" not in source
    assert "import ultralytics" not in source


def test_movie_knowledge_is_prior_and_browse_card_contains_no_relationships(
        tmp_path: Path) -> None:
    manifest = v82.bootstrap_movie_knowledge(
        Path("config/knowledge/luoxiaohei1_movie_prior.json"),
        tmp_path / "knowledge", target_id="char:xiaohei")
    assert manifest["knowledge_is_prior"] is True
    assert manifest["external_claims_can_bind_occurrences"] is False
    claims = (tmp_path / "knowledge" / "claims.jsonl").read_text(encoding="utf-8")
    assert '"status": "prior_to_verify"' in claims
    profile = {
        "status": "ready",
        "profiles": [
            {"status": "usable", "form_id": f"char:xiaohei/{form}"}
            for form in ("form_black_cat", "form_black_hair_child",
                         "form_white_hair_child")],
    }
    card = v82.build_target_search_card(
        manifest, profile, tmp_path / "search_card.json")
    prompt = v82.build_browse_prompt(card)
    assert "黑色小猫形态" in prompt
    assert "无限" not in prompt
    assert "风息" not in prompt
    assert card["external_prior_can_confirm_identity"] is False


def test_multi_form_match_supports_character_without_forcing_form() -> None:
    result = v82.resolve_target_and_form([
        {"form_id": "black", "forward": "same", "reverse": "same"},
        {"form_id": "white", "forward": "same", "reverse": "same"},
        {"form_id": "cat", "forward": "different", "reverse": "different"},
    ])
    assert result["character_status"] == "supported"
    assert result["form_status"] == "uncertain"
    assert result["form_candidates"] == ["black", "white"]


def test_target_and_hard_negative_match_is_character_conflict() -> None:
    result = v82.resolve_target_and_form([
        {"form_id": "white", "forward": "same", "reverse": "same"},
    ], hard_negative_matches=True)
    assert result["character_status"] == "conflict"
    assert result["character_id"] is None


def _complete_server_audit() -> dict:
    return {
        "actual_sampled_frame_count": 8,
        "actual_frame_indices": list(range(8)),
        "actual_frame_timestamps_relative_s": [i / 4 for i in range(8)],
        "actual_frame_timestamps_absolute_s": [10 + i / 4 for i in range(8)],
        "input_image_sizes": [[1280, 720]] * 4,
        "processed_image_grid_thw": [[1, 45, 80]] * 4,
        "processed_video_grid_thw": [[4, 20, 50]],
        "video_tokens_before": 1000,
        "video_tokens_after": 250,
        "actual_retention_ratio": .25,
        "service_startup_args": ["--retention-ratio", "0.25"],
        "service_git_head": "abc123",
        "service_git_dirty": False,
        "service_core_hashes": {"src/flashvid_vllm/model.py": "deadbeef"},
        "video_transport_sha256": "video-sha",
        "image_sha256": ["image-sha"] * 4,
    }


def test_flashvid_audit_is_fail_closed() -> None:
    incomplete = SimpleNamespace(raw={}, request_audit={"image_count": 4})
    with pytest.raises(v82.V82Blocked, match="token_audit_unavailable"):
        v82.validate_flashvid_audit(
            incomplete, expected_fps=4, expected_retention=.25,
            expected_image_count=4, interval=[10, 12])

    complete = SimpleNamespace(
        raw={"flashvid_audit": _complete_server_audit()},
        request_audit={"image_count": 4, "transport_clip_sha256": "video-sha",
                       "identity_album_sha256": ["image-sha"] * 4})
    audit = v82.validate_flashvid_audit(
        complete, expected_fps=4, expected_retention=.25,
        expected_image_count=4, interval=[10, 12])
    assert audit["audit_complete"] is True
    assert audit["server"]["video_tokens_before"] == 1000
    assert audit["server"]["video_tokens_after"] == 250


def test_short_flashvid_clip_reports_effective_floor_not_requested_ratio() -> None:
    server = _complete_server_audit()
    server.update({
        "actual_sampled_frame_count": 4,
        "actual_frame_indices": list(range(4)),
        "actual_frame_timestamps_relative_s": [i / 4 for i in range(4)],
        "actual_frame_timestamps_absolute_s": [10 + i / 4 for i in range(4)],
        "processed_video_grid_thw": [[2, 20, 50]],
        "video_tokens_before": 500,
        "video_tokens_after": 250,
        "actual_retention_ratio": .5,
    })
    answer = SimpleNamespace(
        raw={"flashvid_audit": server},
        request_audit={"image_count": 4, "transport_clip_sha256": "video-sha",
                       "identity_album_sha256": ["image-sha"] * 4})
    result = v82.validate_flashvid_audit(
        answer, expected_fps=4, expected_retention=.25,
        expected_image_count=4, interval=[10, 11])
    assert result["server"]["actual_retention_ratio"] == .5


def test_shot_alignment_only_expands_outward() -> None:
    assert v82.align_interval_outward(
        [10, 12], [[9, 11], [11, 13]], duration_s=100) == [9.0, 13.0]
    assert v82.align_interval_outward(
        [10, 12], [], duration_s=100) == [10.0, 12.0]


def test_candidate_scene_keeps_all_sources_and_raw_intervals(tmp_path: Path) -> None:
    rows = v82.build_candidate_scenes(
        [{"candidate_id": "v", "source_interval": [10, 20]}],
        [{"mention_id": "m", "interval": [30, 31]}],
        [{"id": "p", "source_interval": [50, 55]}],
        [{"id": "n", "source_interval": [70, 75]}],
        duration_s=100, merge_gap_s=1,
        output_path=tmp_path / "scenes.jsonl")
    assert sum(len(row["lead_refs"]) for row in rows) == 4
    assert all(row["logical_scene_not_model_batch"] for row in rows)
    assert all("raw_interval" in ref for row in rows for ref in row["lead_refs"])
    assert (tmp_path / "scenes.jsonl").is_file()


def test_long_scene_is_split_by_capacity_with_overlap() -> None:
    tasks = v82.build_observation_tasks([{
        "candidate_scene_id": "s", "source_interval": [0, 52], "lead_refs": [],
    }], fps=4, max_frames=80, overlap_s=4)
    assert len(tasks) == 3
    assert all(row["source_interval"][1] - row["source_interval"][0] <= 20
               for row in tasks)
    assert tasks[1]["source_interval"][0] == tasks[0]["source_interval"][1] - 4
    assert tasks[0]["candidate_scene_id"] == tasks[-1]["candidate_scene_id"]


def test_incomplete_scene_edge_expands_outward_with_safety_budget() -> None:
    task = {
        "task_id": "t0", "candidate_scene_id": "s0",
        "source_interval": [100, 120], "fps": 4,
    }
    left = v82.make_boundary_context_task(
        task, direction="left", source_duration_s=500, max_context_s=40)
    assert left is not None
    assert left["source_interval"] == [84.0, 104.0]
    assert left["context_chain_interval"] == [84.0, 120.0]
    capped = v82.make_boundary_context_task(
        {**left, "source_interval": [84, 104]}, direction="left",
        source_duration_s=500, max_context_s=40)
    assert capped is not None
    assert capped["context_chain_interval"] == [80.0, 120.0]
    assert v82.make_boundary_context_task(
        {**capped, "source_interval": [80, 88]}, direction="left",
        source_duration_s=500, max_context_s=40) is None


def test_v82_neutral_contract_allows_state_without_initiator() -> None:
    payload = {
        "status": "observed",
        "occurrences": [{
            "local_id": "B", "visible_interval": [0, 1],
            "local_description": "dark figure on broken ground",
            "visual_state": "slides backward and stops", "roi": None,
        }],
        "event_candidates": [{
            "interval": [0, 1], "initiator_local_id": None,
            "affected_local_id": "B", "action_or_state": "slides backward",
            "visible_result": "stops on broken ground", "relation_status": "uncertain",
        }],
        "left_context_complete": False, "right_context_complete": True,
        "boundary_reason": "input begins after the cause",
    }
    assert v82.validate_target_neutral_observation(payload, duration_s=2) == "observed"


def test_observe_queue_keeps_fact_and_identity_separate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    payload = {
        "status": "observed",
        "occurrences": [{
            "local_id": "A", "visible_interval": [0, 1],
            "local_description": "small pale figure", "visual_state": "raises one arm",
            "roi": None,
        }, {
            "local_id": "B", "visible_interval": [1, 2],
            "local_description": "large dark figure", "visual_state": "moves backward",
            "roi": None,
        }],
        "event_candidates": [{
            "interval": [.5, 2], "initiator_local_id": "A",
            "affected_local_id": "B", "action_or_state": "extends one arm",
            "visible_result": "the dark figure moves backward",
            "relation_status": "supported",
        }],
        "left_context_complete": True, "right_context_complete": True,
        "boundary_reason": "",
    }

    class Runner:
        def watch(self, *_args, **kwargs):
            assert kwargs["fps"] == 4
            return SimpleNamespace(text=json.dumps(payload), sampling={"ok": True},
                                   gpu_pair="0,1")

    task = {
        "task_id": "t", "candidate_scene_id": "s", "source_interval": [10, 16],
        "fps": 4, "max_frames": 80, "state": "pending",
        "task_kind": "neutral_observation", "lead_refs": [],
        "attempt_count": 0, "infrastructure_failures": [],
    }
    result = v82.run_observe_queue(
        cfg, {}, tmp_path / "observe", source_video=source,
        runner=Runner(), tasks=[task], source_duration_s=100)
    assert result["queue_closed"] is True
    assert result["processing_complete"] is True
    occurrence_text = (tmp_path / "observe" / "occurrence_bank.jsonl").read_text()
    event_text = (tmp_path / "observe" / "neutral_event_candidates.jsonl").read_text()
    assert "character_id" not in occurrence_text
    assert "character_id" not in event_text
    assert "initiator_occurrence_id" in event_text


def test_observe_queue_batches_independent_tasks(tmp_path: Path) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    payload = {
        "status": "observed_empty", "occurrences": [], "event_candidates": [],
        "left_context_complete": True, "right_context_complete": True,
        "boundary_reason": "",
    }

    class Runner:
        batch_sizes = []

        def watch_many(self, requests):
            self.batch_sizes.append(len(requests))
            return [SimpleNamespace(text=json.dumps(payload), sampling={"ok": True})
                    for _ in requests]

        def watch(self, *_args, **_kwargs):
            raise AssertionError("successful initial batch must not run serially")

    tasks = [{
        "task_id": f"t{i}", "candidate_scene_id": f"s{i}",
        "source_interval": [i * 10, i * 10 + 6], "fps": 4, "max_frames": 80,
        "state": "pending", "task_kind": "neutral_observation", "lead_refs": [],
        "attempt_count": 0, "infrastructure_failures": [],
    } for i in range(2)]
    runner = Runner()
    result = v82.run_observe_queue(
        cfg, {}, tmp_path / "observe", source_video=source,
        runner=runner, tasks=tasks, source_duration_s=100)
    assert runner.batch_sizes == [2]
    assert result["initial_parallel_batch_size"] == 2
    assert result["queue_closed"] is True


def test_oom_split_preserves_complete_parent_interval(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    class Runner:
        def watch(self, *_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

    task = {
        "task_id": "long", "candidate_scene_id": "scene", "source_interval": [0, 20],
        "fps": 4, "max_frames": 80, "state": "pending",
        "task_kind": "neutral_observation", "lead_refs": [],
        "attempt_count": 0, "infrastructure_failures": [],
    }
    result = v82.run_observe_queue(
        load_config(), {}, tmp_path / "observe", source_video=source,
        runner=Runner(), tasks=[task], source_duration_s=100)
    queue = [json.loads(line) for line in (tmp_path / "observe" /
                                            "observe_queue.jsonl").read_text().splitlines()]
    children = [row for row in queue if row.get("oom_parent_task_id") == "long"]
    assert result["queue_closed"] is False
    assert result["processing_complete"] is False
    assert result["new_oom_split_task_count"] == 2
    assert children[0]["source_interval"][0] == 0
    assert children[-1]["source_interval"][1] == 20
    assert all(row["source_interval"][1] - row["source_interval"][0] <= 16
               for row in children)


def test_timeline_requires_identity_fact_and_relation_support(tmp_path: Path) -> None:
    occurrence = {"occurrence_id": "occ_a", "source_interval": [1, 2]}
    binding = {"occurrence_id": "occ_a", "character_status": "supported",
               "review_status": "unreviewed"}
    good = {
        "event_id": "good", "participants": [{"occurrence_id": "occ_a"}],
        "facts": [{"relation_verified": True, "relation_check_required": True}],
        "verification": {"status": "verified"},
    }
    bad = {
        "event_id": "bad", "participants": [{"occurrence_id": "occ_a"}],
        "facts": [{"relation_verified": False, "relation_check_required": True}],
        "verification": {"status": "verified"},
    }
    result = v82.build_target_timelines(
        [], [occurrence], [binding], [good, bad], tmp_path)
    assert result["automatic_character_occurrence_count"] == 1
    assert result["automatic_event_count"] == 1
    automatic = (tmp_path / "automatic_event_timeline.jsonl").read_text()
    assert '"good"' in automatic
    assert '"bad"' not in automatic


def test_binding_batches_workers_and_keeps_multiform_as_form_uncertainty(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "candidate.jpg"
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(v82, "export_frame", lambda *_args, **_kwargs: candidate)
    profiles = []
    for form in ("form_black_cat", "form_black_hair_child", "form_white_hair_child"):
        seed = tmp_path / f"{form}_seed.jpg"
        negative = tmp_path / f"{form}_negative.jpg"
        seed.write_bytes(b"seed")
        negative.write_bytes(b"negative")
        profiles.append({
            "status": "usable", "character_id": "char:xiaohei",
            "form_id": f"char:xiaohei/{form}", "profile_version": form,
            "examples": {
                "seed": {"full_frame": str(seed)},
                "negative_00": {"full_frame": str(negative)},
            },
        })

    class Runner:
        batch_size = 0

        def inspect_media_many(self, requests):
            self.batch_size = len(requests)
            # Per form: positive fwd/rev, negative fwd/rev.
            values = ["same", "same", "different", "different",
                      "same", "same", "different", "different",
                      "different", "different", "different", "different"]
            return [SimpleNamespace(text=json.dumps({"result": value}))
                    for value in values]

    runner = Runner()
    result = v82.bind_target_and_form(
        load_config(), [{"occurrence_id": "occ", "source_interval": [1, 2],
                         "observation_sha256": "obs"}],
        {"profiles": profiles}, tmp_path / "identity", source_video=tmp_path / "x.mp4",
        runner=runner)
    assert runner.batch_size == 12
    binding = result["bindings"][0]
    assert binding["character_status"] == "supported"
    assert binding["form_status"] == "uncertain"
    assert binding["form_candidates"] == ["form_black_cat", "form_black_hair_child"]


def test_fact_verification_allows_unknown_initiator(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    occurrence = {
        "occurrence_id": "occ_b", "local_description": "dark figure",
        "visual_state": "slides backward", "source_interval": [10, 11],
    }
    candidate = {
        "event_candidate_id": "e", "source_interval": [10, 11],
        "initiator_occurrence_id": None, "affected_occurrence_id": "occ_b",
    }
    payload = {
        "passed": True, "initiator_occurrence_id": None,
        "affected_occurrence_id": "occ_b", "action_or_state": "slides backward",
        "visible_result": "stops beside broken stone", "relation_status": "uncertain",
        "claim_bounds": {"can_prove": ["the figure moves"],
                         "cannot_prove": ["who caused it"]},
    }

    class Runner:
        def watch_many(self, requests):
            assert len(requests) == 1
            return [SimpleNamespace(text=json.dumps(payload), sampling={"ok": True})]

    result = v82.verify_target_event_facts(
        load_config(), [candidate], [occurrence], source_video=source,
        runner=Runner(), target_occurrence_ids={"occ_b"},
        output_dir=tmp_path / "events", source_duration_s=100)
    assert result["verified_count"] == 1
    assert result["event_facts"][0]["initiator_occurrence_id"] is None
    assert result["event_facts"][0]["relation_status"] == "uncertain"


def test_stratified_pack_hides_stratum_and_model_status(tmp_path: Path) -> None:
    blocks = [{"block_id": f"b{i}", "source_interval": [i * 45, (i + 1) * 45],
               "candidate_hit": i % 2 == 0} for i in range(16)]
    result = v82.build_stratified_audit_pack(
        blocks, eligible_interval=[0, 720], output_dir=tmp_path,
        per_stratum=1, random_seed=1)
    public = json.loads((tmp_path / "blind_contact_sheet.json").read_text())
    assert result["sample_count"] == 8
    assert all("stratum" not in row for row in public["items"])
    assert all("candidate_hit" not in row for row in public["items"])
    design = json.loads((tmp_path / "sampling_design.json").read_text())
    assert all(row["design_weight"] == row["Nh"] / row["nh"]
               for row in design["strata"])


def test_stratified_pack_appends_without_resampling_existing(tmp_path: Path) -> None:
    blocks = [{"block_id": f"b{i}", "source_interval": [i * 45, (i + 1) * 45],
               "candidate_hit": i % 2 == 0} for i in range(32)]
    first = v82.build_stratified_audit_pack(
        blocks, eligible_interval=[0, 1440], output_dir=tmp_path,
        per_stratum=1, random_seed=1)
    first_key = json.loads((tmp_path / "private_sample_key.json").read_text())
    second = v82.build_stratified_audit_pack(
        blocks, eligible_interval=[0, 1440], output_dir=tmp_path,
        per_stratum=1, random_seed=1, append_per_stratum=1)
    second_key = json.loads((tmp_path / "private_sample_key.json").read_text())
    assert second["sample_count"] > first["sample_count"]
    assert all(second_key[key] == value for key, value in first_key.items())


def test_audit_design_is_joined_privately_and_counts_are_compiled() -> None:
    rows = v82.attach_audit_design(
        [{
            "anonymous_id": "audit_000",
            "target_appearances": [{
                "candidate_investigated": True, "identity_retained": True}],
            "important_visible_events": [{"entered_event_timeline": False}],
            "actor_affected_checks": [{"correct": True}],
            "identity_outcomes": [
                {"model_supported": True, "correct": True, "is_target": True}],
        }],
        private_key={"audit_000": {"stratum": "hit_q1", "block_id": "b0"}},
        sampling_design={"strata": [{
            "stratum": "hit_q1", "design_weight": 4.0}]})
    assert rows[0]["stratum"] == "hit_q1"
    assert rows[0]["weight"] == 4.0
    assert rows[0]["candidate_hits"] == 1
    assert rows[0]["event_hits"] == 0
    assert rows[0]["identity_correct"] == 1


def test_interval_coverage_does_not_double_count_overlaps() -> None:
    assert v82.interval_coverage_ratio([0, 10], [[0, 8], [2, 9]]) == .9


def _audit_rows(count: int, success: int) -> list[dict]:
    rows = []
    for index in range(count):
        hit = int(index < success)
        rows.append({
            "stratum": f"{'hit' if index % 2 else 'miss'}_q{index % 4 + 1}",
            "weight": 1,
            "true_appearances": 1, "candidate_hits": hit,
            "investigated_true_appearances": 1, "identity_hits": hit,
            "important_events": 1, "event_hits": hit,
            "direction_checks": 1, "direction_hits": hit,
            "supported_binding_false_merge": 0,
        })
    return rows


def test_three_state_statistical_gate_and_zero_error_uncertainty() -> None:
    insufficient = v82.evaluate_v82_recall(
        _audit_rows(8, 8), bootstrap_replicates=100, random_seed=1)
    assert insufficient["decision"] == "INSUFFICIENT_EVIDENCE"
    assert insufficient["metrics"]["candidate_interval_recall"]["ci95"][0] < 1
    passed = v82.evaluate_v82_recall(
        _audit_rows(120, 120), bootstrap_replicates=100, random_seed=1)
    assert passed["decision"] == "PASS"
    failed = v82.evaluate_v82_recall(
        _audit_rows(120, 20), bootstrap_replicates=100, random_seed=1)
    assert failed["decision"] == "FAIL"


def test_false_merge_is_an_absolute_failure() -> None:
    rows = _audit_rows(120, 120)
    rows[0]["supported_binding_false_merge"] = 1
    result = v82.evaluate_v82_recall(rows, bootstrap_replicates=100)
    assert result["decision"] == "FAIL"


def test_cli_exposes_only_non_editorial_v82_phases() -> None:
    args = build_parser().parse_args([
        "character-recall-v82", "--phase", "timeline", "--output", "out"])
    assert args.phase == "timeline"
    input_args = build_parser().parse_args([
        "character-recall-v82", "--phase", "preflight-input", "--output", "out"])
    assert input_args.phase == "preflight-input"
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "character-recall-v82", "--phase", "render", "--output", "out"])
