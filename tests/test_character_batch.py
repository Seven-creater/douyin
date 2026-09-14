from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video import character_batch as v8
from src.agentic_video import cli
from src.agentic_video.cli import build_parser
from src.config import load_config


def _spec() -> dict:
    return {
        "spec_version": "character_batch_v8",
        "source": "luoxiaohei1",
        "characters": [
            {"character_id": "char:xiaohei", "display_name": "小黑",
             "aliases": ["小黑"]},
            {"character_id": "char:wuxian", "display_name": "无限",
             "aliases": ["无限"]},
            {"character_id": "char:fengxi", "display_name": "风息",
             "aliases": ["风息"]},
        ],
        "coverage": {"block_s": 45.0, "fps": 2.0, "retention_ratio": .1},
        "seed_proposals": {"char:xiaohei": {"invalid_source_times_s": [5391.5]}},
    }


def _event(event_id: str = "event_1") -> dict:
    row = {
        "schema_version": "event_fact_v1", "event_id": event_id,
        "interval": [1.0, 2.0], "core_interval": [1.2, 1.6],
        "renderable_interval": [1.0, 1.9], "source_video": "source.mp4",
        "participants": [
            {"occurrence_id": f"{event_id}_A", "role": "actor"},
            {"occurrence_id": f"{event_id}_B", "role": "patient"},
        ],
        "facts": [{"actor": f"{event_id}_A", "action": "strikes",
                   "patient": f"{event_id}_B", "result": "B moves back",
                   "relation_verified": True}],
        "claim_bounds": {"can_prove": ["B moves back"],
                         "cannot_prove": ["battle ends"]},
    }
    row["event_fact_sha256"] = v8.event_fact_hash(row)
    return row


def test_v8_cli_and_fixed_spec_contract(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "character-batch", "--phase", "coverage", "--output", "run",
    ])
    assert args.command == "character-batch"
    accepted = parser.parse_args([
        "character-batch-accept", "--output", "run",
        "--human-acceptance", "human.json",
    ])
    assert accepted.command == "character-batch-accept"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec()), encoding="utf-8")
    spec, digest = v8.read_v8_spec(path)
    assert spec["source"] == "luoxiaohei1"
    assert len(digest) == 64


def test_checked_in_v8_endpoints_use_actual_served_model_names() -> None:
    spec = json.loads(Path(
        "config/experiments/lxh1_v8_character_batch.json").read_text(encoding="utf-8"))
    assert spec["coverage"]["endpoint"]["model"].endswith("FlashVID-r010")
    assert spec["reference_backend"]["model"].endswith("FlashVID-r100")
    for arm in spec["targeted_escalation"]["arms"]:
        assert arm["model"].endswith(f"FlashVID-r{int(arm['retention_ratio'] * 100):03d}")


def test_v8_cli_classifies_unhandled_bootstrap_failure(tmp_path: Path,
                                                       monkeypatch) -> None:
    monkeypatch.setattr(cli, "_character_batch_impl",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("x")))
    args = SimpleNamespace(output=str(tmp_path), phase="bootstrap")
    with pytest.raises(KeyError):
        cli._character_batch(args, None)
    failure = json.loads((tmp_path / "bootstrap_failure.json").read_text(encoding="utf-8"))
    assert failure["failure_class"] == "infrastructure"
    assert failure["failure_stage"] == "bootstrap"


def test_v8_module_has_no_detector_tracker_or_reid_import() -> None:
    source = Path("src/agentic_video/character_batch.py").read_text(
        encoding="utf-8").lower()
    for banned in ("import yolo", "from yolo", "import ultralytics",
                   "from ultralytics", "import tracker", "from tracker",
                   "import reid", "from reid"):
        assert banned not in source


def test_external_prior_does_not_write_source_truth(tmp_path: Path) -> None:
    spec = _spec()
    spec["characters"][0]["external_claims"] = [{
        "claim": "candidate appearance lead", "source_tier": "secondary",
        "source_url": "https://example.test/prior",
    }]
    result = v8.build_external_character_prior(spec, tmp_path)
    assert result["source_evidence_written"] is False
    assert result["claims"][0]["status"] == "prior_only"
    assert result["claims"][0]["value"] == "candidate appearance lead"


def test_event_fact_rejects_character_id_and_binding_does_not_change_hash() -> None:
    event = _event()
    before = event["event_fact_sha256"]
    binding = {"occurrence_id": "event_1_A", "character_id": "char:xiaohei",
               "status": "verified"}
    binding["character_id"] = "char:fengxi"
    assert event["event_fact_sha256"] == before
    contaminated = {**event, "participants": [{
        "occurrence_id": "event_1_A", "character_id": "char:xiaohei"}]}
    with pytest.raises(ValueError, match="character_id"):
        v8.event_fact_hash(contaminated)


def test_cache_invalidation_preserves_independent_visual_facts() -> None:
    assert "event_facts" not in v8.cache_invalidation_targets("asr")
    assert "occurrence_bank" not in v8.cache_invalidation_targets("character_profile")
    assert "event_facts" not in v8.cache_invalidation_targets("external_prior")
    assert "videos" in v8.cache_invalidation_targets("event_fact")


def test_neutral_prompts_do_not_receive_names_or_editorial_functions() -> None:
    text = (v8.NEUTRAL_COVERAGE_PROMPT + v8.NEUTRAL_OCCURRENCE_PROMPT
            + v8.NEUTRAL_EVENT_PROMPT).lower()
    for forbidden in ("小黑", "无限", "风息", "char:xiaohei", "adversity",
                      "agency", "outcome"):
        assert forbidden not in text


def test_mention_is_lead_not_visual_binding(tmp_path: Path) -> None:
    rows = v8.build_mention_index({"segments": [{
        "start_ms": 1000, "end_ms": 2000, "text": "我们还没找到小黑",
    }]}, _spec()["characters"], tmp_path / "mentions.jsonl")
    assert rows[0]["mention_type"] == "third_person_reference"
    assert rows[0]["visual_presence"] == "unknown"
    assert rows[0]["speaker_occurrence_id"] is None
    assert "character_id" not in rows[0]


def test_asr_alias_is_only_an_investigation_lead(tmp_path: Path) -> None:
    spec = _spec()
    spec["characters"][2]["asr_aliases"] = ["凤曦"]
    rows = v8.build_mention_index({"segments": [{
        "start_ms": 1000, "end_ms": 2000, "text": "发现凤曦的踪迹",
    }]}, spec["characters"], tmp_path / "mentions.jsonl")
    assert len(rows) == 1
    assert rows[0]["matched_name"] == "凤曦"
    assert rows[0]["match_basis"] == "asr_alias"
    assert rows[0]["mentioned_character_ids"] == ["char:fengxi"]
    assert rows[0]["visual_presence"] == "unknown"
    prior = v8.build_external_character_prior(spec, tmp_path / "prior")
    fengxi = next(row for row in prior["candidates"]
                  if row["character_id"] == "char:fengxi")
    assert "凤曦" not in fengxi["aliases"]


def test_uniform_coverage_blocks_exclude_head_tail_without_gaps() -> None:
    blocks = v8.coverage_blocks(5400, block_s=45, head_s=90, tail_s=360)
    assert blocks[0] == (90.0, 135.0)
    assert blocks[-1] == (4995.0, 5040.0)
    assert all(left[1] == right[0] for left, right in zip(blocks, blocks[1:]))


def test_uniform_coverage_drops_regions_not_worth_rewatch() -> None:
    payload = {"regions": [
        {"interval": [0.0, 2.0], "activity": "standing",
         "worth_rewatch": False,
         "occurrences": [{"local_id": "A", "visual_state": "standing"}],
         "event_candidates": [{"actor_local_id": "A", "action": "standing"}]},
        {"interval": [5.0, 8.0], "activity": "one subject strikes another",
         "worth_rewatch": True,
         "occurrences": [{"local_id": "A", "visual_state": "arm moves"}],
         "event_candidates": [{"actor_local_id": "A", "action": "strikes"}]},
    ]}
    occurrences, events = v8._normalize_coverage_response(
        payload, block_id="b0000", start_s=90.0, end_s=135.0)
    assert len(occurrences) == 1
    assert len(events) == 1
    assert occurrences[0]["source_interval"] == [95.0, 98.0]


def test_coverage_parser_wraps_bare_array_but_keeps_interval_contract() -> None:
    payload, shape = v8._parse_coverage_payload('[{"interval":[1,4]}]')
    assert shape == "bare_array_wrapped"
    assert payload == {"regions": [{"interval": [1, 4]}]}
    with pytest.raises(v8.V8Blocked, match="candidate_interval_outside"):
        v8._normalize_coverage_response({"regions": [{
            "interval": [0, 10], "worth_rewatch": True,
            "occurrences": [], "event_candidates": [],
        }]}, block_id="b", start_s=0, end_s=45)


def test_short_final_coverage_block_uses_flashvid_even_frame_contract(
        tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_ffmpeg(_binary, args, **_kwargs):
        captured["filter"] = args[args.index("-vf") + 1]
        Path(args[-1]).write_bytes(b"video")

    class Probe:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"frames": [
            {"best_effort_timestamp_time": str(index * 24.52 / 48)}
            for index in range(48)
        ]})

    monkeypatch.setattr(v8.common, "run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(v8.subprocess, "run", lambda *_args, **_kwargs: Probe())
    _clip, audit = v8._build_coverage_transport(
        "ffmpeg", "ffprobe", tmp_path / "source.mp4", tmp_path / "transport",
        start_s=5715.0, end_s=5739.52, requested_fps=2.0, source_fps=60.0)
    assert audit["requested_frame_count"] == 48
    assert audit["actual_frame_count"] == 48
    assert audit["sampling_verified"] is True
    assert audit["effective_fps"] == pytest.approx(48 / 24.52, abs=1e-6)
    assert f"fps={48 / 24.52:.12f}" in captured["filter"]


def test_coverage_event_cannot_reference_missing_occurrence() -> None:
    with pytest.raises(v8.V8Blocked, match="event_patient_missing_occurrence"):
        v8._normalize_coverage_response({"regions": [{
            "interval": [0, 3], "worth_rewatch": True,
            "occurrences": [{"local_id": "A", "visual_state": "raises arm"}],
            "event_candidates": [{"actor_local_id": "A", "action": "strikes",
                                  "patient_local_id": "B"}],
        }]}, block_id="b", start_s=0, end_s=45)


def test_occurrence_bank_deduplicates_without_identity_and_leads_are_not_truth(
        tmp_path: Path) -> None:
    occurrence = {"occurrence_id": "occ_A", "source_interval": [1, 2],
                  "local_description": "small light subject",
                  "observation_sha256": "same"}
    rows = v8.build_occurrence_bank(
        [occurrence], [occurrence], tmp_path / "occurrences.jsonl")
    assert len(rows) == 1
    assert "character_id" not in rows[0]
    leads = v8.build_investigation_leads(
        [{"mention_id": "m", "interval": [3, 4],
          "mentioned_character_ids": ["char:xiaohei"]}],
        [{"event_candidate_id": "e", "source_interval": [5, 6]}])
    assert {row["lead_kind"] for row in leads} == {
        "asr_mention", "uniform_coverage_activity"}
    assert all("status" not in row for row in leads)


def test_coverage_rejects_character_name_in_neutral_response(tmp_path: Path,
                                                             monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    monkeypatch.setattr(v8.common, "video_duration_s", lambda *_: 10.0)
    monkeypatch.setattr(v8.common, "run_ffprobe_json", lambda *_: {"streams": []})
    monkeypatch.setattr(v8, "_build_coverage_transport", lambda *_args, **_kwargs: (
        clip, {"actual_frame_count": 20, "sampling_verified": True}))

    class Client:
        def watch(self, *_args, **_kwargs):
            return SimpleNamespace(
                text=json.dumps({"regions": [{"interval": [1, 3],
                    "activity": "小黑 moves", "occurrences": [],
                    "event_candidates": []}]}), raw={},
                request_audit={"requested_frames": 20})

    spec = _spec()
    spec["coverage"].update({"head_s": 0, "tail_s": 0, "min_movie_s": 100})
    result = v8.build_uniform_coverage_map(
        cfg, spec, tmp_path / "coverage", client=Client(), source_video=source,
        source_sha256="x")
    assert result["failed_count"] == 1
    assert result["blocks"][0]["status"] == "failed"


def test_coverage_stages_transport_inside_service_allowlist(tmp_path: Path,
                                                             monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"transport-media")
    media_root = tmp_path / "service-media"
    monkeypatch.setattr(v8.common, "video_duration_s", lambda *_: 10.0)
    monkeypatch.setattr(v8.common, "run_ffprobe_json", lambda *_: {"streams": []})
    monkeypatch.setattr(v8, "_build_coverage_transport", lambda *_args, **_kwargs: (
        clip, {"actual_frame_count": 20, "sampling_verified": True}))

    class Client:
        request_path = None

        def watch(self, video, *_args, **_kwargs):
            self.request_path = Path(video)
            assert self.request_path.parent == media_root.resolve()
            assert self.request_path.read_bytes() == clip.read_bytes()
            return SimpleNamespace(
                text=json.dumps({"regions": []}), raw={},
                request_audit={"transport_clip": str(video), "requested_frames": 20})

    client = Client()
    spec = _spec()
    spec["coverage"].update({"head_s": 0, "tail_s": 0, "min_movie_s": 100})
    spec["coverage"]["endpoint"] = {"media_root": str(media_root)}
    result = v8.build_uniform_coverage_map(
        cfg, spec, tmp_path / "coverage", client=client, source_video=source,
        source_sha256="x")
    block = result["blocks"][0]
    assert block["status"] == "covered"
    audit = block["media_transport_audit"]
    assert audit["staged"] is True
    assert audit["source_transport_clip_sha256"] == audit["request_transport_clip_sha256"]
    assert client.request_path is not None
    assert not client.request_path.exists()
    assert clip.is_file()
    assert block["sampling_audit_status"] == "verified_transport_frames"
    assert (tmp_path / "coverage" / "occurrence_candidates.jsonl").is_file()
    assert not (tmp_path / "occurrence_bank.jsonl").exists()


def test_coverage_retries_one_model_contract_failure(tmp_path: Path,
                                                      monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"transport")
    monkeypatch.setattr(v8.common, "video_duration_s", lambda *_: 10.0)
    monkeypatch.setattr(v8.common, "run_ffprobe_json", lambda *_: {"streams": []})
    monkeypatch.setattr(v8, "_build_coverage_transport", lambda *_args, **_kwargs: (
        clip, {"actual_frame_count": 20, "sampling_verified": True}))

    class Client:
        calls = 0

        def watch(self, _video, prompt, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                text = ""
            else:
                assert "CONTRACT RETRY" in prompt
                text = json.dumps({"regions": []})
            return SimpleNamespace(text=text, raw={},
                request_audit={"requested_frames": 20})

    client = Client()
    spec = _spec()
    spec["coverage"].update({"head_s": 0, "tail_s": 0, "min_movie_s": 100})
    result = v8.build_uniform_coverage_map(
        cfg, spec, tmp_path / "coverage", client=client, source_video=source,
        source_sha256="x")
    block = result["blocks"][0]
    assert block["status"] == "covered"
    assert client.calls == 2
    assert len(block["response_attempts"]) == 2
    assert block["response_attempts"][0]["validation_error"]
    assert block["response_attempts"][1]["validation_error"] is None


def test_occurrence_rewatch_replaces_coarse_group_with_local_subjects(
        tmp_path: Path, monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(v8.common, "video_duration_s", lambda *_: 20.0)

    class Runner:
        def watch_many(self, requests):
            assert len(requests) == 1
            assert "小黑" not in requests[0]["prompt"]
            assert requests[0]["kwargs"]["use_audio_in_video"] is False
            return [SimpleNamespace(text=json.dumps({
                "occurrences": [
                    {"local_id": "A", "visible_interval": [0, 3],
                     "local_description": "small light subject",
                     "visual_state": "raises an arm"},
                    {"local_id": "B", "visible_interval": [0, 3],
                     "local_description": "tall dark subject",
                     "visual_state": "moves backward"},
                ],
                "event_candidates": [{"interval": [1, 2.5],
                    "actor_local_id": "A", "action": "strikes",
                    "patient_local_id": "B", "visible_result": "B moves back"}],
                "passed": True,
            }), sampling={"sampling_verified": True}, gpu_pair="0,1")]

    leads = [{"source_interval": [5, 7], "lead_kind": "coverage",
              "lead_id": "coarse_group"}]
    result = v8.build_occurrence_bank_from_leads(
        cfg, _spec(), leads, tmp_path / "observations", source_video=source,
        runner=Runner())
    occurrence_rows = [json.loads(line) for line in (
        tmp_path / "occurrence_bank.jsonl").read_text().splitlines()]
    event_rows = [json.loads(line) for line in (
        tmp_path / "event_candidates.jsonl").read_text().splitlines()]
    assert result["occurrence_count"] == 2
    assert {row["local_description"] for row in occurrence_rows} == {
        "small light subject", "tall dark subject"}
    assert event_rows[0]["actor_occurrence_id"] != event_rows[0][
        "patient_occurrence_id"]
    assert all("character_id" not in json.dumps(row) for row in occurrence_rows)


def test_occurrence_rewatch_treats_empty_as_observed_and_reuses_it(
        tmp_path: Path, monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(v8.common, "video_duration_s", lambda *_: 20.0)

    class FirstRunner:
        calls = 0

        def watch_many(self, requests):
            self.calls += 1
            return [SimpleNamespace(text=json.dumps({
                "occurrences": [], "event_candidates": [], "passed": False,
            }), sampling={"sampling_verified": True}, gpu_pair="0,1")
                    for _ in requests]

    first_runner = FirstRunner()
    leads = [{"source_interval": [5, 7], "lead_kind": "coverage",
              "lead_id": "quiet_region"}]
    first = v8.build_occurrence_bank_from_leads(
        cfg, _spec(), leads, tmp_path / "observations", source_video=source,
        runner=first_runner, source_sha256="source-hash")
    assert first["complete"] is True
    assert first["observed_empty_count"] == 1
    assert first["failed_observation_count"] == 0

    class NoCallRunner:
        def watch_many(self, _requests):
            raise AssertionError("a completed empty observation must be reused")

    second = v8.build_occurrence_bank_from_leads(
        cfg, _spec(), leads, tmp_path / "observations", source_video=source,
        runner=NoCallRunner(), source_sha256="source-hash")
    assert second["complete"] is True
    assert second["reused_observation_count"] == 1


def test_occurrence_cache_survives_observation_id_shift(tmp_path: Path,
                                                        monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(v8.common, "video_duration_s", lambda *_: 30.0)

    class Runner:
        call_sizes = []

        def watch_many(self, requests):
            self.call_sizes.append(len(requests))
            return [SimpleNamespace(text=json.dumps({
                "occurrences": [], "event_candidates": [], "passed": False,
            }), sampling={}, gpu_pair="0,1") for _ in requests]

    runner = Runner()
    later = {"source_interval": [20, 22], "lead_kind": "coverage", "lead_id": "b"}
    v8.build_occurrence_bank_from_leads(
        cfg, _spec(), [later], tmp_path / "observations", source_video=source,
        runner=runner, source_sha256="source-hash")
    earlier = {"source_interval": [2, 4], "lead_kind": "asr", "lead_id": "a"}
    second = v8.build_occurrence_bank_from_leads(
        cfg, _spec(), [earlier, later], tmp_path / "observations",
        source_video=source, runner=runner, source_sha256="source-hash")
    assert runner.call_sizes == [1, 1]
    assert second["reused_observation_count"] == 1


def test_investigation_windows_merge_near_duplicate_leads() -> None:
    rows = v8.investigation_windows([
        {"source_interval": [10, 12], "lead_kind": "asr", "lead_id": "a"},
        {"source_interval": [10.2, 12.2], "lead_kind": "coverage", "lead_id": "b"},
    ], duration_s=100, window_s=6)
    assert len(rows) == 1
    assert len(rows[0]["lead_refs"]) == 2
    assert rows[0]["source_interval"][1] - rows[0]["source_interval"][0] == 6


def test_profile_uses_all_directed_pairs_and_no_confidence(tmp_path: Path,
                                                           monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    def fake_export(_ffmpeg, _source, _example, destination):
        full = destination / "full.jpg"
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(str(destination).encode())
        return full, None, {"full_frame": str(full), "full_frame_sha256": "x",
                            "roi": None, "roi_crop": None, "roi_crop_sha256": None}

    monkeypatch.setattr(v8, "_export_identity_example", fake_export)

    class Runner:
        def inspect_media(self, paths, _prompt, **_kwargs):
            is_negative = any("negative" in str(path) for path in paths[:2])
            return SimpleNamespace(text=json.dumps({
                "result": "different" if is_negative else "same",
                "confidence": .99,
            }))

    manifest = {"characters": [{"character_id": "char:xiaohei", "forms": [{
        "form_id": "char:xiaohei/form_white_small",
        "trusted_seed": {"source_time_s": 5412.7, "confirmed_by": "human"},
        "positive_examples": [{"source_time_s": 1}, {"source_time_s": 2}],
        "hard_negatives": [{"source_time_s": 3}],
    }]}]}
    result = v8.build_character_profiles(
        cfg, _spec(), manifest, tmp_path / "profiles",
        source_video=source, runner=Runner())
    profile = result["profiles"][0]
    assert profile["status"] == "usable"
    assert len(profile["comparisons"]) == 12
    assert all("confidence" not in row["parsed"] for row in profile["comparisons"])


def test_old_dissolve_seed_is_blocked(tmp_path: Path) -> None:
    manifest = {"characters": [{"character_id": "char:xiaohei", "forms": [{
        "form_id": "char:xiaohei/form_wrong",
        "trusted_seed": {"source_time_s": 5391.5, "confirmed_by": "human"},
        "positive_examples": [{}, {}], "hard_negatives": [{}],
    }]}]}
    with pytest.raises(v8.V8Blocked, match="forbidden_regression_seed"):
        v8.build_character_profiles(
            load_config(), _spec(), manifest, tmp_path / "profiles",
            source_video=tmp_path / "source.mp4", runner=object())


def test_occurrence_binding_uses_full_frame_positive_and_hard_negative(
        tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.jpg"
    candidate.write_bytes(b"candidate")
    seed = tmp_path / "seed.jpg"
    seed.write_bytes(b"seed")
    negative = tmp_path / "negative.jpg"
    negative.write_bytes(b"negative")
    monkeypatch.setattr(v8, "export_frame", lambda *_args, **_kwargs: candidate)

    class Runner:
        def __init__(self):
            self.calls = []

        def inspect_media(self, paths, _prompt, **_kwargs):
            self.calls.append(paths)
            return SimpleNamespace(text='{"result":"same"}')

    runner = Runner()
    profiles = {"profiles": [{
        "character_id": "char:xiaohei", "form_id": "white", "status": "usable",
        "profile_version": "p", "examples": {
            "seed": {"full_frame": str(seed)},
            "negative_00": {"full_frame": str(negative)},
        }}]}
    result = v8.bind_occurrence_identities(
        load_config(), [{"occurrence_id": "occ", "source_interval": [1, 2],
                         "observation_sha256": "o"}], profiles,
        tmp_path / "identity", source_video=source, runner=runner)
    assert result["bindings"][0]["status"] == "verified"
    assert len(runner.calls) == 2
    assert all(candidate in paths and seed in paths and negative in paths
               for paths in runner.calls)


def test_character_views_keep_event_direction_for_both_sides(tmp_path: Path) -> None:
    event = _event()
    bindings = [
        {"occurrence_id": "event_1_A", "character_id": "char:xiaohei",
         "form_id": "white", "status": "verified", "binding_sha256": "a"},
        {"occurrence_id": "event_1_B", "character_id": "char:fengxi",
         "form_id": "dark", "status": "verified", "binding_sha256": "b"},
    ]
    result = v8.derive_character_evidence_views([event], bindings, tmp_path)
    xiaohei = next(row for row in result["views"] if row["character_id"] == "char:xiaohei")
    fengxi = next(row for row in result["views"] if row["character_id"] == "char:fengxi")
    assert xiaohei["perspective_role"] == "actor"
    assert "agency" in xiaohei["evidence_uses"]
    assert fengxi["perspective_role"] == "patient"
    assert "received_impact" in fengxi["evidence_uses"]
    assert event["facts"][0]["actor"] == "event_1_A"


def test_shared_event_uses_separate_relation_and_native_frame_boundary(
        tmp_path: Path, monkeypatch) -> None:
    cfg = load_config()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    monkeypatch.setattr(v8, "cut_clip", lambda *_args, **_kwargs: clip)

    def native(_ffmpeg, _source, interval, destination, **_kwargs):
        frames = []
        for index, pts in enumerate((1.3, 1.4, 1.5)):
            path = destination / f"f{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frame")
            frames.append({"frame_id": f"f{index:03d}", "pts_s": pts,
                           "labeled_path": str(path), "path": str(path), "sha256": "x"})
        return {"source_interval": interval, "source_fps": 60, "frames": frames,
                "selected": None}

    monkeypatch.setattr(v8, "extract_native_frames", native)

    class Runner:
        def __init__(self):
            self.prompts = []

        def watch(self, _video, prompt, **_kwargs):
            self.prompts.append(prompt)
            if "Independently inspect" in prompt:
                return SimpleNamespace(text=json.dumps({
                    "passed": True, "relations": [{"actor": "occ_A",
                    "patient": "occ_B", "related_outcome": True,
                    "can_prove": ["B changes after A"], "cannot_prove": []}]}))
            return SimpleNamespace(text=json.dumps({
                "participants": [{"occurrence_id": "occ_A", "role": "actor"},
                                 {"occurrence_id": "occ_B", "role": "patient"}],
                "facts": [{"actor": "occ_A", "action": "strikes",
                           "patient": "occ_B", "result": "B falls",
                           "relation_requires_separate_check": True}],
                "claim_bounds": {"can_prove": ["B falls"],
                                 "cannot_prove": ["battle ends"]},
                "core_interval": [.2, .6], "renderable_interval": [0, .9],
                "passed": True}))

        def inspect_media(self, _images, _prompt, **_kwargs):
            return SimpleNamespace(text=json.dumps({
                "start_frame": "f000", "peak_frame": "f001", "end_frame": "f002"}))

    runner = Runner()
    occurrences = [
        {"occurrence_id": "occ_A", "local_description": "light small subject",
         "visual_state": "raises an arm", "source_interval": [1, 2]},
        {"occurrence_id": "occ_B", "local_description": "dark tall subject",
         "visual_state": "moves back", "source_interval": [1, 2]},
    ]
    candidates = [{"event_candidate_id": "c1", "source_interval": [1, 2],
                   "actor_occurrence_id": "occ_A", "patient_occurrence_id": "occ_B"}]
    result = v8.verify_shared_event_facts(
        cfg, candidates, occurrences, tmp_path / "events", source_video=source,
        runner=runner)
    fact = result["event_facts"][0]
    assert fact["facts"][0]["relation_verified"] is True
    assert fact["core_interval"] == [1.3, 1.5]
    assert fact["native_frame_provenance"]["selected"]["peak_frame"] == "f001"
    assert any("Independently inspect" in prompt for prompt in runner.prompts)
    assert "character_id" not in json.dumps(fact)


def test_custom_arc_survives_preferred_family_mismatch(tmp_path: Path) -> None:
    views = []
    for index in range(2):
        views.append({
            "view_id": f"v{index}", "character_id": "char:wuxian",
            "event_id": f"quiet_{index}", "evidence_uses": ["companionship"],
            "renderable_interval": [index * 2.0, index * 2.0 + 1],
            "core_expression_fragment": f"quiet fact {index}",
        })
    queue = v8.build_character_creation_queue(views, tmp_path / "queue.json")
    selected = queue["selected_tasks"]
    assert len(selected) == 1
    assert selected[0]["family"] == "custom_evidence_arc"


def test_one_character_failure_does_not_abort_other_preview(tmp_path: Path,
                                                            monkeypatch) -> None:
    queue = {"selected_tasks": [
        {"task_id": "x", "character_id": "char:xiaohei", "family": "custom",
         "core_expression": "x", "view_ids": ["vx"]},
        {"task_id": "w", "character_id": "char:wuxian", "family": "custom",
         "core_expression": "w", "view_ids": ["vw"]},
    ]}
    views = [
        {"view_id": "vx", "event_id": "ex", "source_video": "source.mp4",
         "renderable_interval": [0, 1], "core_interval": [.2, .8],
         "observation_interval": [0, 1], "binding_sha256": "bx",
         "event_fact_sha256": "ex"},
        {"view_id": "vw", "event_id": "ew", "source_video": "source.mp4",
         "renderable_interval": [1, 2], "core_interval": [1.2, 1.8],
         "observation_interval": [1, 2], "binding_sha256": "bw",
         "event_fact_sha256": "ew"},
    ]
    reference = {"measured_style": {"reference_duration_s": 2,
                                     "shot_duration_p90_s": 1}}

    def fake_finalize(_cfg, plan, output, **_kwargs):
        if plan["character_id"] == "char:xiaohei":
            raise v8.V8Blocked("render", "synthetic_failure")
        Path(output).mkdir(parents=True, exist_ok=True)
        return {"acceptance": {"automated_passed": True}}

    monkeypatch.setattr(v8, "finalize_target_microcut", fake_finalize)
    result = v8.run_character_batch(
        load_config(), queue, reference, views, tmp_path / "run",
        source_video=tmp_path / "source.mp4", bgm_path=tmp_path / "bgm.m4a",
        runner=object())
    assert result["preview_ready_count"] == 1
    assert {row["status"] for row in result["characters"]} == {"blocked", "preview_ready"}


def test_plan_uses_verified_boundaries_without_filler() -> None:
    task = {"task_id": "t", "character_id": "char:xiaohei",
            "family": "intervention_and_result", "core_expression": "B moves back",
            "view_ids": ["v"]}
    view = {"view_id": "v", "event_id": "e", "source_video": "source.mp4",
            "renderable_interval": [1.0, 2.5], "core_interval": [1.4, 2.0],
            "observation_interval": [.5, 3], "binding_sha256": "b",
            "event_fact_sha256": "e"}
    plan = v8.build_reference_driven_character_plan(
        {"measured_style": {"reference_duration_s": 21.933,
                            "shot_duration_p90_s": 1.2}}, task, [view])
    assert plan["segments"][0]["final_source_interval"] == [1.0, 2.5]
    assert plan["filler_added"] is False
    assert plan["short_complete_content_allowed"] is True


def test_batch_acceptance_requires_zero_selected_identity_and_direction_errors(
        tmp_path: Path, monkeypatch) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (tmp_path / "batch_acceptance.json").write_text(json.dumps({
        "characters": [{"character_id": "char:xiaohei", "task_id": "t",
                        "task_dir": str(task_dir), "status": "preview_ready"}],
        "selected_task_count": 1, "preview_ready_count": 1,
    }), encoding="utf-8")
    called = []
    monkeypatch.setattr(v8, "accept_v7_output",
                        lambda *_args, **_kwargs: called.append(True))
    result = v8.accept_character_batch(tmp_path, {"characters": {
        "char:xiaohei": {"passed": True, "selected_identity_false_merges": 1,
                          "selected_event_direction_errors": 0,
                          "core_expression_understood": True}}})
    assert result["final_passed_count"] == 0
    assert not called
