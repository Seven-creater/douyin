from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.agentic_video.cli import build_parser
from src.agentic_video.evidence_units import EvidenceUnitV3
from src.agentic_video.target_v7 import (
    REQUIRED_GOALS, V7Blocked, _normalize_candidates, apply_native_selection,
    build_reference_driven_edit_plan, build_target_album, compare_browse_arms,
    diagnose_browse, evaluate_target_album, extract_native_frames, finalize_target_microcut,
    oracle_evidence_bank,
    prepare_reference_task, run_browse_escalation, run_browse_matrix,
    transport_windows, validate_album_consistency, verify_target_evidence,
)
from src.config import load_config
from src.perception.flashvid_client import (
    FlashVIDClient,
    FlashVIDEndpoint,
    OpenAICompatibleClient,
)


def _candidate(candidate_id: str, arm: str, goal: str, start: float) -> dict:
    return {
        "id": candidate_id, "arm": arm, "goal_hypotheses": [goal],
        "observation_interval": [start, start + 0.5],
    }


def test_v7_modules_do_not_import_detector_tracker_or_reid() -> None:
    roots = [
        Path("src/agentic_video/target_v7.py"),
        Path("src/perception/flashvid_client.py"),
    ]
    banned = ("ultralytics", "yolo", "tracker", "reid")
    for path in roots:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(f"import {name}" in source or f"from {name}" in source
                       for name in banned)


def test_cli_exposes_v7_phase_and_accept_commands() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "evidence-v7-target", "--phase", "browse", "--output", "run",
    ])
    assert args.command == "evidence-v7-target"
    assert args.phase == "browse"
    accepted = parser.parse_args([
        "evidence-v7-accept", "--output", "run", "--human-acceptance", "human.json",
    ])
    assert accepted.command == "evidence-v7-accept"


def test_v7_cli_writes_stable_blocked_acceptance(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agentic_video import cli

    def blocked(_args, _cfg):
        raise V7Blocked("browsing", "required_goal_not_observed_after_vanilla")

    monkeypatch.setattr(cli, "_evidence_v7_target_impl", blocked)
    args = SimpleNamespace(output=str(tmp_path), phase="verify")
    result = cli._evidence_v7_target(args, None)
    acceptance = json.loads((tmp_path / "acceptance.json").read_text(encoding="utf-8"))
    assert result["delivery"] == "blocked"
    assert acceptance["failure_class"] == "content"
    assert acceptance["failure_stage"] == "browsing"


def test_flashvid_arm_contract_and_explicit_even_frames() -> None:
    for arm, fps, ratio, backend, expected in (
        ("A", 2, .1, "flashvid", 12), ("B", 4, .1, "flashvid", 24),
        ("C", 4, .25, "flashvid", 24), ("D", 4, 1, "native_bypass", 24),
    ):
        endpoint = FlashVIDEndpoint(arm, "http://localhost/v1", "model",
                                    fps, ratio, backend)
        assert FlashVIDClient.requested_frames(6, endpoint.fps) == expected
    with pytest.raises(ValueError, match="native_bypass"):
        FlashVIDEndpoint("D", "x", "m", 4, .25, "flashvid")


def test_flashvid_request_contains_images_video_and_no_second_sampling(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"i{index}.jpg"
        path.write_bytes(b"image")
        paths.append(path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    captured = {}

    class Transport:
        def chat(self, payload):
            captured.update(payload)
            return ({"choices": [{"message": {"content": "{}"}}], "usage": {}},
                    .1, "request-hash")

    client = FlashVIDClient(FlashVIDEndpoint(
        "B", "http://localhost/v1", "model", 4, .1, "flashvid"))
    client.transport = Transport()
    answer = client.watch(video, "prompt", duration_s=6, image_paths=paths)
    content = captured["messages"][0]["content"]
    assert [row["type"] for row in content] == [
        "image_url", "image_url", "image_url", "image_url", "video_url", "text"]
    assert captured["mm_processor_kwargs"] == {"do_sample_frames": False}
    assert captured["media_io_kwargs"]["video"] == {"num_frames": 24, "fps": -1}
    assert answer.request_audit["image_count"] == 4


def test_openai_transport_preserves_file_uri_for_vllm() -> None:
    transport = OpenAICompatibleClient("http://localhost/v1")
    assert transport.local_file_urls_as_paths is False


def test_transport_windows_cover_scope_with_only_transport_overlap() -> None:
    windows = transport_windows(5385, 5460, duration_s=6, overlap_s=.5)
    assert windows[0] == (5385.0, 5391.0)
    assert windows[-1][1] == 5460.0
    assert all(right[0] == pytest.approx(left[1] - .5)
               for left, right in zip(windows, windows[1:]))


def test_not_observed_is_not_rewritten_as_absent() -> None:
    payload = {"candidates": [
        {"relative_interval": [0, 1], "relation": "not_observed"},
        {"relative_interval": [1, 2], "relation": "target_direct",
         "goal_hypotheses": ["agency"], "observation": {"action": "moves"}},
    ]}
    rows = _normalize_candidates(
        payload, arm="A", window_id="w000", start_s=10, end_s=16)
    assert len(rows) == 1
    assert rows[0]["relation"] == "target_direct"
    assert "absent" not in json.dumps(rows)


def test_album_contract_checks_all_directed_edges_without_confidence() -> None:
    positives = [{"id": value} for value in ("seed", "p1", "p2")]
    negatives = [{"id": "n1"}]
    calls = []

    def compare(left, right, direction):
        calls.append((left["id"], right["id"], direction))
        return "different" if "n1" in {left["id"], right["id"]} else "same"

    checks = validate_album_consistency(positives, negatives, compare=compare)
    assert len(checks) == 12  # 3*2 positive directions + 3*1*2 negative directions
    assert all("confidence" not in row and "margin" not in row for row in checks)


def test_album_asymmetry_or_uncertainty_blocks_freeze() -> None:
    positives = [{"id": value} for value in ("seed", "p1", "p2")]
    negatives = [{"id": "n1"}]

    def compare(left, right, _direction):
        if left["id"] == "p1" and right["id"] == "seed":
            return "uncertain"
        return "different" if "n1" in {left["id"], right["id"]} else "same"

    with pytest.raises(V7Blocked, match="target_album_consistency_failed"):
        validate_album_consistency(positives, negatives, compare=compare)


def test_heldout_identity_audit_is_posthoc_and_counts_false_merges() -> None:
    album = {
        "positive": [{"id": "seed", "source_time_s": 1},
                     {"id": "wrong", "source_time_s": 5}],
        "hard_negative": [{"id": "right_negative", "source_time_s": 6}],
    }
    oracle = {"identity_gt": [{"interval": [4.5, 6.5], "label": "different"}]}
    audit = evaluate_target_album(album, oracle)
    assert audit["gt_entered_album_prompts"] is False
    assert audit["hard_negative_false_merges"] == 1
    assert audit["unlabeled_count"] == 0


def test_target_album_expansion_is_identity_only_and_always_uses_full_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed = tmp_path / "seed.jpg"
    seed.write_bytes(b"seed")
    prompts, image_sets = [], []

    def fake_export(_ffmpeg, _source, _time, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(destination.name.encode())
        return destination

    def fake_crop(_ffmpeg, _full, _roi, destination):
        destination.write_bytes(destination.name.encode())
        return destination

    monkeypatch.setattr("src.agentic_video.target_v7.export_frame", fake_export)
    monkeypatch.setattr("src.agentic_video.target_v7.export_roi", fake_crop)

    class Runner:
        def inspect_media(self, images, prompt, **_kwargs):
            prompts.append(prompt)
            image_sets.append([Path(value).name for value in images])
            different = any("negative" in Path(value).name for value in images)
            return SimpleNamespace(
                text=json.dumps({"result": "different" if different else "same"}),
                input_tokens=1, output_tokens=1)

    proposals = [
        {"id": "positive_1", "source_time_s": 2, "candidate_class": "possible_same",
         "roi": [0, 0, .5, .5]},
        {"id": "positive_2", "source_time_s": 3, "candidate_class": "possible_same",
         "roi": [0, 0, .5, .5]},
        {"id": "negative_1", "source_time_s": 4, "candidate_class": "hard_negative",
         "roi": [.5, 0, 1, .5]},
    ]
    album = build_target_album(
        SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg"}),
        {"source_scope": {"start_s": 0, "end_s": 10}}, tmp_path / "album",
        trusted_seed=seed, source_video=tmp_path / "source.mp4",
        vanilla_client=None, runner=Runner(), candidate_proposals=proposals)
    assert album["status"] == "trusted_seed_model_expanded"
    assert album["uses_model_confidence"] is False
    assert all("adversity" not in prompt and "agency" not in prompt and
               "outcome" not in prompt for prompt in prompts)
    assert all(any("full" in name or name == "seed.jpg" for name in names)
               for names in image_sets)


def test_reference_task_uses_native_semantics_and_deterministic_times(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"reference")
    monkeypatch.setattr("src.agentic_video.target_v7.common.video_duration_s",
                        lambda *_args: 21.933)
    monkeypatch.setattr("src.agentic_video.target_v7.detect_shots", lambda *_args, **_kwargs: {
        "shots": [
            {"start_s": 0.0, "end_s": 1.0, "duration_s": 1.0},
            {"start_s": 1.0, "end_s": 3.0, "duration_s": 2.0},
        ], "shot_count": 2})

    class Vanilla:
        def watch(self, _video, prompt, **kwargs):
            assert kwargs["duration_s"] == 21.933
            assert "deterministic" in prompt.lower()
            return SimpleNamespace(text=json.dumps({
                "edit_sections": [{"id": "actual", "shot_indices": [0, 1],
                                   "purpose": "measured", "goal_ids": ["agency"]}]}),
                                   request_audit={"backend": "native_bypass"})

    task = prepare_reference_task(
        SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"}),
        {"reference_video": str(reference), "reference_analysis": {}},
        tmp_path / "out", vanilla_client=Vanilla())
    assert task["provenance"]["flashvid_loaded"] is False
    assert task["edit_sections"][0]["source_interval"] == [0.0, 3.0]
    assert task["measured_style"]["shot_duration_p90_s"] == pytest.approx(1.9)


def test_browse_matrix_keeps_arm_requests_isolated(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.agentic_video.target_v7.cut_clip", lambda *_args, **kwargs: (
        (Path(_args[2]) / "clip.mp4") if False else tmp_path / "clip.mp4"))
    (tmp_path / "clip.mp4").write_bytes(b"clip")
    images = []
    for index in range(4):
        path = tmp_path / f"album_{index}.jpg"
        path.write_bytes(b"image")
        images.append(path)

    class Client:
        def __init__(self, arm, fps, ratio, backend):
            self.endpoint = SimpleNamespace(
                arm=arm, fps=fps, retention_ratio=ratio, backend=backend)
            self.calls = []

        def watch(self, _clip, _prompt, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                text=json.dumps({"candidates": [{"relative_interval": [0, .5],
                    "relation": "target_direct", "goal_hypotheses": ["agency"]}]}),
                request_audit={"usage": {}, "latency_s": .1})

    clients = {
        arm: Client(arm, fps, ratio, backend)
        for arm, fps, ratio, backend in (
            ("A", 2, .1, "flashvid"), ("B", 4, .1, "flashvid"),
            ("C", 4, .25, "flashvid"), ("D", 4, 1, "native_bypass"))
    }
    spec = {
        "source_scope": {"start_s": 0, "end_s": 6},
        "transport": {"window_s": 6, "overlap_s": .5},
        "browse_arms": {arm: {"fps": client.endpoint.fps,
                              "retention_ratio": client.endpoint.retention_ratio,
                              "backend": client.endpoint.backend}
                        for arm, client in clients.items()},
    }
    album = {"positive": [{"full_frame": str(path)} for path in images[:3]],
             "hard_negative": [{"full_frame": str(images[3])}]}
    result = run_browse_matrix(
        SimpleNamespace(perception={"ffmpeg_bin": "ffmpeg"}), spec,
        tmp_path / "browse", clients=clients, source_video=tmp_path / "source.mp4",
        reference_task={"evidence_goals": [{"id": value} for value in REQUIRED_GOALS]},
        target_album=album)
    assert set(result["arms"]) == set("ABCD")
    assert all(client.calls[0]["image_paths"] == images for client in clients.values())


def test_escalation_leaves_exhausted_arm_and_fills_all_goals() -> None:
    arms = {
        "A": {"candidates": [_candidate("a", "A", "adversity", 1)]},
        "B": {"candidates": [_candidate("b", "B", "agency", 2)]},
        "C": {"candidates": [_candidate("c", "C", "outcome", 3)]},
        "D": {"candidates": []},
    }

    def verify(row):
        return {"id": row["id"], "eligible_goals": row["goal_hypotheses"]}

    evidence, logs = run_browse_escalation(arms, verify=verify)
    assert [row["id"] for row in evidence] == ["a", "b", "c"]
    assert [(row["previous_arm"], row["next_arm"]) for row in logs] == [
        ("A", "B"), ("B", "C")]


def test_escalation_only_blocks_after_vanilla_exhausted() -> None:
    arms = {arm: {"candidates": []} for arm in "ABCD"}
    with pytest.raises(V7Blocked) as error:
        run_browse_escalation(arms, verify=lambda _row: None)
    assert error.value.failure_stage == "browsing"
    assert error.value.reason_code == "required_goal_not_observed_after_vanilla"


def test_browse_diagnosis_distinguishes_sampling_and_compression() -> None:
    metrics = {
        "A": {"candidate_temporal_recall": .2},
        "B": {"candidate_temporal_recall": .4},
        "C": {"candidate_temporal_recall": .8},
        "D": {"candidate_temporal_recall": 1.0},
    }
    assert diagnose_browse(metrics) == [
        "temporal_sampling_insufficient_at_2fps",
        "r010_compression_too_aggressive",
        "custom_port_or_compression_semantic_loss",
    ]


def test_gt_metrics_are_posthoc_and_use_center_or_iou() -> None:
    oracle = {"facts": [{"core_interval": [10, 11], "expected_goals": ["agency"]}]}
    arms = {arm: {"window_count": 1, "failures": [], "request_audits": [],
                  "candidates": ([{"observation_interval": [9.8, 10.2],
                                   "goal_hypotheses": ["agency"]}]
                                 if arm == "B" else [])}
            for arm in "ABCD"}
    result = compare_browse_arms(arms, oracle=oracle)
    assert result["gt_used_in_prompts"] is False
    assert result["metrics"]["B"]["candidate_temporal_hits"] == 1


def _verified(goal: str, start: float) -> dict:
    return {
        "id": f"ev_{goal}", "eligible_goals": [goal],
        "renderable_interval": [start, start + .7],
        "core_interval": [start + .1, start + .5],
        "source_video": "source.mp4",
    }


def test_planner_derives_duration_from_reference_and_adds_no_filler() -> None:
    reference = {
        "edit_sections": [
            {"id": "s1", "goal_ids": ["adversity"]},
            {"id": "s2", "goal_ids": ["agency"]},
            {"id": "s3", "goal_ids": ["outcome"]},
        ],
        "measured_style": {"reference_duration_s": 21.933,
                           "shot_duration_p90_s": .6},
    }
    bank = {"evidence": [_verified("adversity", 1), _verified("agency", 2),
                         _verified("outcome", 3)]}
    plan = build_reference_driven_edit_plan(reference, bank)
    assert plan["passed"] is True
    assert plan["style"]["soft_range_s"] == pytest.approx([15.3531, 26.3196])
    assert plan["short_complete_content_allowed"] is True
    assert plan["filler_added"] is False
    assert all("semantic_completeness_reason" in row for row in plan["segments"])


def test_related_outcome_requires_relation_verification() -> None:
    base = dict(
        id="ev", observation_interval=(1, 2), core_interval=(1.2, 1.5),
        renderable_interval=(1.1, 1.7), observation={},
        source_form="dynamic_action", target_relation="related_outcome",
        identity_verification={"result": "same"},
        action_verification={"passed": True},
    )
    with pytest.raises(ValueError, match="relation check"):
        EvidenceUnitV3(**base)
    assert EvidenceUnitV3(
        **base, relation_verification={"passed": True}).target_relation == "related_outcome"


def test_oracle_bypasses_browse_but_reuses_planner_contract(tmp_path: Path) -> None:
    oracle = {"facts": [
        {"id": "a", "core_interval": [1, 1.3], "container_interval": [.8, 1.5],
         "expected_goals": ["adversity"], "observation": {},
         "source_form": "dynamic_action"},
        {"id": "b", "core_interval": [2, 2.3], "container_interval": [1.8, 2.5],
         "expected_goals": ["agency"], "observation": {},
         "source_form": "dynamic_action"},
        {"id": "c", "core_interval": [3, 3.3], "container_interval": [2.8, 3.5],
         "expected_goals": ["outcome"], "observation": {},
         "source_form": "dynamic_action"},
    ]}
    bank = oracle_evidence_bank(oracle, tmp_path / "source.mp4")
    assert bank["bypassed"] == ["browsing", "automatic_verification"]
    assert bank["evidence"][2]["relation_verification"]["passed"] is True


def test_native_selection_maps_ids_to_pts_without_model_arithmetic() -> None:
    manifest = {"frames": [
        {"frame_id": "f000", "pts_s": 10.0},
        {"frame_id": "f001", "pts_s": 10.016667},
        {"frame_id": "f002", "pts_s": 10.033333},
    ]}
    selected = apply_native_selection(
        manifest, {"start_frame": "f000", "peak_frame": "f001", "end_frame": "f002"})
    assert selected["selected"]["core_interval"] == [10.0, 10.033333]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_native_frame_export_uses_all_source_frames_and_stable_ids(tmp_path: Path) -> None:
    video = tmp_path / "sixty.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=160x90:rate=60:duration=1", "-c:v", "libx264", str(video),
    ], check=True)
    manifest = extract_native_frames("ffmpeg", video, [0.0, .8], tmp_path / "frames")
    assert 47 <= len(manifest["frames"]) <= 49
    assert manifest["frames"][0]["frame_id"] == "f000"
    assert Path(manifest["frames"][0]["labeled_path"]).is_file()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_fake_flashvid_fake_omni_synthetic_video_full_v7_chain(tmp_path: Path) -> None:
    source, bgm = tmp_path / "source.mp4", tmp_path / "bgm.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=320x180:rate=60:duration=3", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264",
        "-c:a", "aac", str(source),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=220:duration=3", "-c:a", "aac", str(bgm),
    ], check=True)
    album_images = []
    for index, timestamp in enumerate((.1, .2, .3, .4)):
        path = tmp_path / f"album_{index}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", str(timestamp),
            "-i", str(source), "-frames:v", "1", str(path),
        ], check=True)
        album_images.append(path)
    album = {"positive": [{"full_frame": str(path)} for path in album_images[:3]],
             "hard_negative": [{"full_frame": str(album_images[3])}]}

    class BrowseClient:
        def __init__(self, arm, fps, ratio, backend):
            self.endpoint = SimpleNamespace(
                arm=arm, fps=fps, retention_ratio=ratio, backend=backend)

        def watch(self, _clip, _prompt, **_kwargs):
            rows = [
                {"relative_interval": [.1, .7], "relation": "target_direct",
                 "goal_hypotheses": ["adversity"]},
                {"relative_interval": [1.0, 1.6], "relation": "target_direct",
                 "goal_hypotheses": ["agency"]},
                {"relative_interval": [2.0, 2.7], "relation": "possible_outcome",
                 "goal_hypotheses": ["outcome"]},
            ]
            return SimpleNamespace(text=json.dumps({"candidates": rows}),
                                   request_audit={"usage": {}, "latency_s": .01})

    clients = {arm: BrowseClient(arm, fps, ratio, backend)
               for arm, fps, ratio, backend in (
                   ("A", 2, .1, "flashvid"), ("B", 4, .1, "flashvid"),
                   ("C", 4, .25, "flashvid"), ("D", 4, 1, "native_bypass"))}
    spec = {
        "source_scope": {"start_s": 0, "end_s": 3},
        "transport": {"window_s": 3, "overlap_s": .5},
        "browse_arms": {arm: {"fps": client.endpoint.fps,
                              "retention_ratio": client.endpoint.retention_ratio,
                              "backend": client.endpoint.backend}
                        for arm, client in clients.items()},
    }
    reference = {
        "evidence_goals": [{"id": goal} for goal in REQUIRED_GOALS],
        "edit_sections": [{"id": goal, "goal_ids": [goal]} for goal in REQUIRED_GOALS],
        "measured_style": {"reference_duration_s": 3.0,
                           "shot_duration_p90_s": .8},
    }
    cfg = load_config()
    browse = run_browse_matrix(
        cfg, spec, tmp_path / "browse", clients=clients, source_video=source,
        reference_task=reference, target_album=album)

    class Omni:
        def inspect_media(self, images, prompt, **_kwargs):
            if "Select exact native" in prompt:
                last = len(images) - 2
                return SimpleNamespace(text=json.dumps({
                    "start_frame": "f001", "peak_frame": f"f{last // 2:03d}",
                    "end_frame": f"f{last:03d}"}), input_tokens=1, output_tokens=1)
            return SimpleNamespace(text='{"result":"same"}',
                                   input_tokens=1, output_tokens=1)

        def watch(self, _video, prompt, **_kwargs):
            if "Coarsely localize" in prompt:
                text = '{"relative_interval":[0.05,0.55]}'
            elif "Independently inspect" in prompt:
                text = ('{"passed":true,"relation":"related_outcome",'
                        '"can_prove":["change follows action"],"cannot_prove":[]}')
            elif "第一次看到" in prompt:
                text = json.dumps({
                    "core_message": "同一主体先受压随后行动并出现结果",
                    "hook_clear": True, "montage_coherent": True,
                    "functionless_span_present": False,
                    "visible_evidence_roles": ["困境", "反击", "结果"],
                    "audible_dialogue_present": False, "speech_clear": True,
                    "music_present": True,
                }, ensure_ascii=False)
            else:
                text = json.dumps({
                    "passed": True, "actor": "S0", "patient": "other",
                    "visible_action": "moves", "state_before": "before",
                    "state_after": "after", "can_prove": ["visible move"],
                    "cannot_prove": [], "source_form": "dynamic_action"})
            return SimpleNamespace(text=text, sampling={"sampling_verified": True})

        def inspect_media_many(self, requests):
            return [self.inspect_media(row["image_paths"], row["prompt"],
                                       **row.get("kwargs", {})) for row in requests]

        def watch_many(self, requests):
            return [self.watch(row["video_path"], row["prompt"],
                               **row.get("kwargs", {})) for row in requests]

    runner = Omni()
    bank = verify_target_evidence(
        cfg, spec, tmp_path / "agent", runner=runner, source_video=source,
        browse_arms=browse["arms"], target_album=album)
    assert bank["omni_execution"] == "stage_batched"
    plan = build_reference_driven_edit_plan(reference, bank)
    assert plan["passed"] is True
    final = finalize_target_microcut(
        cfg, plan, tmp_path / "final", source_video=source, bgm_path=bgm,
        runner=runner, force=True)
    assert final["acceptance"]["automated_passed"] is True
    assert final["acceptance"]["delivery"] == "blocked"
    assert not (tmp_path / "final" / "rendered.mp4").exists()
