from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video import modality_isolation as isolation
from src.agentic_video.creative_dna_v2 import (
    DNAV2Error, build_writer_payload, publish_dna)
from src.agentic_video.migration_eval import judge_case
from src.agentic_video.p0_r1 import P0R1Blocked, _absolute_response_times
from src.agentic_video.provenance import (
    APPROVAL_POLICY_VERSION, propagate_revocation,
    validate_approval_binding)
from src.agentic_video.reference_claims import (
    CLAIM_LEDGER_VERSION, ClaimLedger, build_event_draft,
    build_validated_reference, validate_claim)
from src.agentic_video.workspace import Workspace


def _claim(claim_id: str = "C1", **updates):
    value = {
        "claim_id": claim_id, "subject": "E1", "predicate": "appears_in",
        "object": "scene_A", "interval": [0.0, 1.0], "modality": "V",
        "polarity": "POSITIVE", "visibility": "VISIBLE",
        "epistemic_status": "SUPPORTED",
        "support_refs": [{"ref_id": "F1", "kind": "claim"}],
        "producer": "isolated_visual", "source_sha": "a" * 64,
        "schema_version": CLAIM_LEDGER_VERSION, "record_status": "ACTIVE",
        "supersedes": None,
        "semantic_review": {"decision": "supports", "reviewer": "reviewer"},
    }
    value.update(updates)
    return value


def test_visual_request_contract_checks_actual_streams(tmp_path: Path,
                                                       monkeypatch) -> None:
    media = tmp_path / "visual.mp4"
    media.write_bytes(b"not-a-real-video")
    monkeypatch.setattr(isolation, "_stream_types", lambda *a, **k: ["video"])
    payload = {
        "request_id": "v1", "channel": "V", "interval": [0, 1],
        "anonymous_subject_ids": ["E1"],
        "observation_dimensions": ["actions"], "sampling": {"fps": 4},
        "mask_artifact_sha": "x"}
    audit = isolation.audit_request(
        channel="V", prompt="observe generic actions", payload=payload,
        media_paths=[media], forbidden_markers=["source secret"])
    assert audit["contract_passed"]
    monkeypatch.setattr(isolation, "_stream_types",
                        lambda *a, **k: ["video", "audio"])
    with pytest.raises(isolation.IsolationError) as excinfo:
        isolation.audit_request(channel="V", prompt="generic", payload=payload,
                                media_paths=[media])
    assert excinfo.value.reason_code == "visual_request_contains_audio"


def test_request_gate_does_not_echo_forbidden_phrase(tmp_path: Path,
                                                     monkeypatch) -> None:
    media = tmp_path / "v.mp4"
    media.write_bytes(b"x")
    monkeypatch.setattr(isolation, "_stream_types", lambda *a, **k: ["video"])
    payload = {"request_id": "v", "channel": "V", "interval": [0, 1],
               "anonymous_subject_ids": ["E1"],
               "observation_dimensions": [], "sampling": {},
               "mask_artifact_sha": "s"}
    secret = "never echo this reference phrase"
    with pytest.raises(isolation.IsolationError) as excinfo:
        isolation.audit_request(channel="V", prompt=secret, payload=payload,
                                media_paths=[media], forbidden_markers=[secret])
    assert secret not in str(excinfo.value)


def test_legacy_dense_probe_is_silent_and_question_blind(
        tmp_path: Path, monkeypatch) -> None:
    import src.agentic_video.reference_program_v9 as module

    evidence = {"boundaries": [], "cut_candidates": [],
                "ocr_text_events": [], "asr_segments": [],
                "beat_points_s": [], "audio_intensity_change_peaks": []}
    monkeypatch.setattr(module, "_section_evidence",
                        lambda ledger, interval: evidence)

    class Runner:
        calls = []

        def watch(self, video, prompt, **kwargs):
            self.calls.append((video, prompt, kwargs))
            return SimpleNamespace(text=json.dumps({
                "question_id": "Q1", "status": "resolved", "answer": "observed",
                "evidence": [{"evidence_type": "observed_visual",
                              "description": "anonymous action"}],
                "new_questions": []}))

    runner = Runner()
    secret_question = "expected answer hidden from visual executor"
    module._run_reference_probe(
        tmp_path / "source.mp4",
        {"id": "Q1", "question": secret_question,
         "gap_type": "event_structure", "selected_probe": "dense_video",
         "interval": [0.0, 1.0]},
        {"reference": {"sha256": "a" * 64}}, tmp_path / "probe",
        runner=runner, ffmpeg_bin="ffmpeg")
    assert runner.calls[0][2]["use_audio_in_video"] is False
    assert secret_question not in runner.calls[0][1]


def test_negative_claim_requires_continuous_visible_coverage() -> None:
    claim = _claim(polarity="NEGATIVE")
    assert "negative_claim_lacks_continuous_visible_coverage" in validate_claim(claim)


def test_clip_local_times_are_validated_then_shifted() -> None:
    value = {"claims": [{"claim_id": "C1", "interval": [0.1, 0.5]}],
             "events": [{"event_id": "E1", "interval": [0.0, 0.8]}],
             "coverage": {"observed_intervals": [[0.0, 1.0]],
                          "masked_or_unjudgeable": []}}
    shifted = _absolute_response_times(
        value, start_s=16.7, duration_s=1.0, stage="test")
    assert shifted["claims"][0]["interval"] == [16.8, 17.2]
    with pytest.raises(P0R1Blocked):
        _absolute_response_times(
            {"claims": [{"claim_id": "C1", "interval": [0, 2]}],
             "events": [], "coverage": {}},
            start_s=0, duration_s=1, stage="test")
    claim["support_refs"][0]["coverage"] = {
        "kind": "continuous", "interval": [0.0, 1.0]}
    assert "negative_claim_lacks_continuous_visible_coverage" not in validate_claim(
        claim)
    claim["visibility"] = "OCCLUDED"
    assert "negative_claim_lacks_continuous_visible_coverage" in validate_claim(claim)


def test_claim_conflict_requires_exact_alignment_and_comparable_modality() -> None:
    ledger = ClaimLedger(source_sha="a" * 64)
    ledger.add_claim(_claim("C1"), verify_files=False)
    ledger.add_claim(_claim("C2", polarity="NEGATIVE", support_refs=[{
        "ref_id": "F2", "kind": "claim", "coverage": {
            "kind": "continuous", "interval": [0, 1]}}]), verify_files=False)
    assert len(ledger.find_conflicts()) == 1
    ledger = ClaimLedger(source_sha="a" * 64)
    ledger.add_claim(_claim("C1"), verify_files=False)
    ledger.add_claim(_claim("C2", polarity="NEGATIVE", modality="T",
                            support_refs=[{"ref_id": "T1", "kind": "claim",
                                           "coverage": {"kind": "continuous",
                                                        "interval": [0, 1]}}]),
                     verify_files=False)
    assert ledger.find_conflicts() == []


def test_validated_reference_contains_evidence_not_interpretation() -> None:
    ledger = ClaimLedger(source_sha="a" * 64)
    ledger.add_claim(_claim(), verify_files=False)
    events = build_event_draft([{
        "event_id": "EV1", "participants": ["E1"],
        "action_claim_ids": ["C1"], "object_ids": [], "ordering": "observed",
        "outcome_claim_ids": [], "interval": [0.0, 1.0]}], ledger)
    value = build_validated_reference(
        ledger, events,
        deterministic_timeline={"segments": [{"segment_id": "S1",
                                                "interval": [0, 1]}]},
        coverage={"full": {"observed_intervals": [[0, 1]]}},
        critical_claim_ids=["C1"])
    assert value["accepted_claims"][0]["claim_id"] == "C1"
    assert "interpretation" not in value and "creative_dna" not in value


def _dna_audit(description: str = "abstract observer") -> dict:
    return {
        "variables": [
            {"variable_id": "V1", "kind": "observer", "description": description},
            {"variable_id": "V2", "kind": "proposition",
             "description": "bounded proposition"}],
        "relations": [{"relation_id": "R1", "type": "logical",
                       "source": "V1", "target": "V2", "condition": "evidence"}],
        "constraints": [{"constraint_id": "C1", "rule": "scope stays bounded",
                         "required": True}],
        "free_slots": [{"slot_id": "S1", "kind": "domain",
                        "constraint_ids": ["C1"]}],
        "editing_relations": [{"relation_id": "E1",
                               "editing_operation": "ordered reveal",
                               "information_effect": "revision",
                               "conditions": ["evidence visible"]}],
        "supported_by": {"R1": ["C1"]},
        "concrete_bindings": [], "anti_invariants": [], "confidence": 0.9,
    }


def test_dna_publish_is_whitelisted_and_rejects_reference_binding() -> None:
    reference = {"accepted_claims": [{"claim_id": "C1",
                                       "object": "very specific source action"}]}
    published = publish_dna(_dna_audit(), validated_reference=reference)
    assert "supported_by" not in published
    assert "concrete_bindings" not in published
    assert set(build_writer_payload(published)) == {
        "schema_version", "variables", "relations", "constraints",
        "free_slots", "editing_relations"}
    with pytest.raises(DNAV2Error) as excinfo:
        publish_dna(_dna_audit("very specific source action"),
                    validated_reference=reference)
    assert excinfo.value.reason_code == "dna_publish_reference_binding_leak"


def test_migration_judge_never_receives_label_or_reference() -> None:
    published = publish_dna(_dna_audit(), validated_reference={})

    class Runner:
        prompt = ""

        def ask(self, prompt, **kwargs):
            self.prompt = prompt
            return SimpleNamespace(text=json.dumps({
                "variable_mapping": [{"variable_id": "V1",
                                      "candidate_binding": "observer"}],
                "preserved_relation_ids": ["R1"],
                "violated_constraint_ids": [], "unsupported_relation_ids": [],
                "accept": True, "reason_codes": []}))

    runner = Runner()
    judge_case(published, {"case_id": "x", "expected_accept": False,
                           "candidate": {"events": ["new domain"]}},
               runner=runner)
    assert "expected_accept" not in runner.prompt
    assert "accepted_claims" not in runner.prompt


def test_workspace_invalidation_crosses_graph_only_intermediate(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "run")
    ws.set_dependency("middle", "source")
    ws.set_dependency("consumer", "middle")
    ws.write_draft("source", {"v": 1})
    ws.commit("source")
    ws.write_draft("consumer", {"v": 1})
    ws.commit("consumer")
    ws.write_draft("source", {"v": 2})
    ws.commit("source")
    assert ws.get_status("consumer") == "stale"


def test_cross_run_copy_invalidated_by_source_sha(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "consumer")
    ws.write_draft("copy", {"v": 1}, metadata={
        "derived_from": [{"artifact_id": "source:v1", "version": "v1",
                          "sha": "deadbeef"}]})
    ws.commit("copy")
    changed = propagate_revocation([ws.root], ["deadbeef"])
    assert changed and Workspace(ws.root).get_status("copy") == "stale"


def test_approval_binds_candidate_parents_and_policy() -> None:
    parents = [{"artifact_id": "candidate", "sha": "abc"},
               {"artifact_id": "view:front", "sha": "def"}]
    approval = {"candidate_id": "candidate", "candidate_sha": "abc",
                "parent_shas": parents,
                "approval_policy_version": APPROVAL_POLICY_VERSION,
                "decision": "approve", "reviewer": "human"}
    assert validate_approval_binding(
        approval, candidate_id="candidate", candidate_sha="abc",
        parent_shas=parents) == []
    approval["parent_shas"] = parents[:1]
    assert "parent_shas_mismatch" in validate_approval_binding(
        approval, candidate_id="candidate", candidate_sha="abc",
        parent_shas=parents)
