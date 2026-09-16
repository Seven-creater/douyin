from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from src.agentic_video.generation_v9g import (
    H3_CAPABILITIES,
    V9GBlocked,
    adapt_contract_to_filmdsl,
    accept_v9g,
    assemble_v9g_sections,
    build_asset_and_state_pack,
    build_generation_jobs,
    build_h3_request,
    commit_authorized_state,
    compile_generation_contract,
    evaluate_v9g_experiment,
    finalize_unit_snippets,
    probe_h3_capabilities,
    render_and_review_sections,
    route_h3_mode,
    run_generate_observe_repair,
    run_v9g_generation_jobs,
    select_evidence_moments,
    validate_generation_contract,
    validate_license_gates,
    validate_upstream_lock,
)


def _dump(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v9_dir(tmp_path: Path) -> Path:
    root = tmp_path / "v9"
    content = {"schema_version": "reference_content_program_v9", "sections": [
        {"section_id": "S1", "transferable_structure": "prove a subject can act",
         "interval": [0, 5], "evidence_ids": ["ev1"]},
    ]}
    edit = {"schema_version": "reference_edit_program_v9",
            "editorial_patterns": [{"section_id": "S1"}]}
    requirements = {"schema_version": "material_requirements_v9", "requirements": [{
        "requirement_id": "req_01", "section_id": "S1",
        "semantic_requirement": {
            "meaning_to_prove": "a stable subject performs an effective action",
            "required_event_understanding": "understand_full_source_event",
            "required_semantic_phases": ["setup", "action", "visible_result"],
        },
        "presentation_requirement": {
            "target_duration_s": 6.0,
            "composition_mode": "event_compression_montage",
            "snippet_count_range": [2, 3],
            "source_continuity": "non_contiguous_allowed",
            "ordering_constraint": "preserve_event_progression",
        },
        "continuity_requirement": {
            "same_subject_required": True, "same_opponent_required": True,
            "same_scene_required": True, "causal_relation_required": True,
        },
        "evidence_requirement": {
            "minimum_sufficient_evidence_set": ["setup", "action", "visible_result"],
            "disallowed_substitutes": ["result_without_action"],
        },
        "traceability": {"content_section_id": "S1"},
    }]}
    values = {
        "reference_content_program.json": content,
        "reference_edit_program.json": edit,
        "material_requirements.json": requirements,
    }
    hashes = {}
    for name, value in values.items():
        path = _dump(root / name, value)
        hashes[name] = _sha(path)
    _dump(root / "acceptance.json", {"passed": True, "human_passed": True})
    _dump(root / "frozen_program_hashes.json", {"program_sha256": hashes})
    return root


def _contract(tmp_path: Path) -> dict:
    target = _dump(tmp_path / "target_setting.json", {
        "meta": {"setting": "training hall"},
        "assets": {"subject": {"id": "C0"}, "opponent": {"id": "C1"}},
    })
    return compile_generation_contract(_v9_dir(tmp_path), target)


def _caps() -> dict[str, bool]:
    return {name: True for name in H3_CAPABILITIES}


def test_contract_requires_human_pass_and_frozen_hashes(tmp_path: Path) -> None:
    v9 = _v9_dir(tmp_path)
    _dump(v9 / "acceptance.json", {"passed": False, "human_passed": False})
    with pytest.raises(V9GBlocked, match="v9_human_acceptance_not_passed"):
        compile_generation_contract(v9, _dump(tmp_path / "target.json", {
            "assets": {"subject": {"id": "C0"}}}))


def test_frozen_program_tampering_blocks_contract(tmp_path: Path) -> None:
    v9 = _v9_dir(tmp_path)
    _dump(v9 / "material_requirements.json", {"tampered": True})
    with pytest.raises(V9GBlocked, match="frozen_v9_program_hash_mismatch"):
        compile_generation_contract(v9, _dump(tmp_path / "target.json", {
            "assets": {"subject": {"id": "C0"}}}))


def test_contract_compiler_keeps_long_event_as_multiple_units(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    section = contract["sections"][0]
    assert section["composition_mode"] == "event_compression_montage"
    assert [unit["semantic_phases"] for unit in section["generation_units"]] == [
        ["setup"], ["action"], ["visible_result"]]
    assert all(4 <= unit["target_duration_s"] <= 15
               for unit in section["generation_units"])
    assert validate_generation_contract(contract)["passed"]


def test_filmdsl_is_derived_and_cannot_mutate_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    original = deepcopy(contract)
    state = build_asset_and_state_pack(contract)
    filmdsl = adapt_contract_to_filmdsl(contract, state)
    filmdsl["clips"][0]["narrative_action"]["semantic_goal"] = "changed"
    assert contract == original
    assert filmdsl["derived_from_contract_hash"] == contract["contract_hash"]


def test_code_and_model_license_gates_block_independently(tmp_path: Path) -> None:
    code = _dump(tmp_path / "code.json", {
        "gate_version": "v1", "checked_upstream_sha": "a", "reviewed_by": "r",
        "checked_at": "2026-09-16", "passed": True})
    model = _dump(tmp_path / "model.json", {
        "gate_version": "v1", "checked_upstream_sha": "b", "reviewed_by": "",
        "checked_at": None, "passed": False, "model_card_revision": "b",
        "license_revision": "c", "deployment_context": "research",
        "intended_output_use": "research", "distribution_plan": "none"})
    with pytest.raises(V9GBlocked, match="model_license_gate_not_passed"):
        validate_license_gates(code, model)
    value = json.loads(model.read_text(encoding="utf-8"))
    value.update({"reviewed_by": "r", "checked_at": "2026-09-16", "passed": True})
    _dump(model, value)
    assert validate_license_gates(code, model)["passed"]


def test_h3_routes_t2va_fl2va_ref2va_and_hybrid() -> None:
    caps = _caps()
    assert route_h3_mode(capabilities=caps)["mode"] == "t2va"
    assert route_h3_mode(first_frame="file:///f.png", capabilities=caps)["mode"] == "fl2va"
    assert route_h3_mode(references=[{"type": "image", "uri": "file:///r.png"}],
                         capabilities=caps)["mode"] == "ref2va"
    route = route_h3_mode(
        references=[{"type": "image", "uri": "file:///r.png"}],
        first_frame="file:///f.png", last_frame="file:///l.png", capabilities=caps)
    assert route["capability"] == "ref2va_hybrid_first_last"


def test_hybrid_failure_requires_explicit_split_without_dropping_conditions() -> None:
    caps = _caps()
    caps["ref2va_hybrid_first"] = False
    split = route_h3_mode(
        references=[{"type": "image", "uri": "file:///r.png"}],
        first_frame="file:///f.png", capabilities=caps, allow_explicit_split=True)
    assert split == {
        "mode": None, "variant": None, "split_required": True,
        "reason_code": "ref2va_hybrid_capability_unavailable",
        "preserve": ["references", "keyframes"],
    }
    with pytest.raises(V9GBlocked):
        route_h3_mode(references=[{"type": "image", "uri": "file:///r.png"}],
                      first_frame="file:///f.png", capabilities=caps)


def test_h3_request_uses_official_keyframe_indices_and_keeps_reference() -> None:
    request = build_h3_request(
        prompt="act", duration_s=5,
        references=[{"type": "image", "uri": "file:///ref.png"}],
        first_frame="file:///first.png", last_frame="file:///last.png",
        capabilities=_caps())
    assert request["task"] == "ref2va"
    assert [row["frame_index"] for row in request["conditions"]
            if row["role"] == "keyframe"] == [0, -1]
    assert any(row["role"] == "reference" for row in request["conditions"])


def test_three_layer_state_only_commits_authorized_transition(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    state = build_asset_and_state_pack(
        contract, canonical_invariants={"face": "A"},
        section_invariants={"S1": {"clothes": "blue"}},
        authorized_mutable_state={"pose": {"transitions": [["standing", "kicking"]]}})
    observed = {"unit_id": "S1.G1", "section_id": "S1",
                "state_changes": {"pose": {"from": "standing", "to": "kicking"}}}
    result = commit_authorized_state(
        state, observed, raw_unit_passed=True, snippets_passed=True,
        contract_hash=contract["contract_hash"])
    assert result["committed"] and len(state["commits"]) == 1


def test_rejected_or_invariant_changing_attempt_never_commits(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    state = build_asset_and_state_pack(
        contract, canonical_invariants={"face": "A"},
        section_invariants={"S1": {"clothes": "blue"}},
        authorized_mutable_state={})
    rejected = commit_authorized_state(
        state, {"unit_id": "S1.G1", "section_id": "S1", "state_changes": {}},
        raw_unit_passed=False, snippets_passed=True,
        contract_hash=contract["contract_hash"])
    illegal = commit_authorized_state(
        state, {"unit_id": "S1.G1", "section_id": "S1",
                "state_changes": {"face": {"from": "A", "to": "B"}}},
        raw_unit_passed=True, snippets_passed=True,
        contract_hash=contract["contract_hash"])
    assert rejected["committed"] is False
    assert illegal["reason"] == "invariant_changed"
    assert state["commits"] == []


class _FakeH3:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request: dict, output_path: Path) -> dict:
        self.requests.append(deepcopy(request))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        return {"output_path": str(output_path), "sha256": _sha(output_path)}


def test_capability_manifest_is_based_on_real_call_results(tmp_path: Path) -> None:
    fixtures = {key: f"file:///{key}" for key in (
        "first", "last", "image", "video", "audio")}
    fl2va, ref2va = _FakeH3(), _FakeH3()
    manifest = probe_h3_capabilities(
        {"fl2va": fl2va, "ref2va": ref2va}, fixtures, tmp_path)
    assert all(manifest[name] for name in H3_CAPABILITIES)
    hybrid = json.loads((tmp_path / "ref2va_hybrid_first_last" / "request.json")
                        .read_text(encoding="utf-8"))
    assert any(row["role"] == "reference" for row in hybrid["conditions"])
    assert {row.get("frame_index") for row in hybrid["conditions"]} >= {0, -1}


def test_generate_observe_repair_is_bounded_and_prompt_is_neutral(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    unit_id = contract["sections"][0]["generation_units"][0]["unit_id"]
    prompts = []
    calls = {"verify": 0}

    def generate(request: dict, output: Path) -> dict:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"video")
        return {"output_path": str(output)}

    def observe(video: Path, prompt: str) -> dict:
        prompts.append(prompt)
        return {"subjects": ["occ_A", "occ_B"], "initiator": "occ_A"}

    def verify(observation: dict, expected: dict) -> dict:
        calls["verify"] += 1
        return {"passed": calls["verify"] == 3,
                "issues": [] if calls["verify"] == 3 else ["action unclear"]}

    def repair(record: dict, trace: dict) -> dict:
        request = deepcopy(record["request"])
        request["prompt"] += " clarify visible action"
        return {"request": request}

    result = run_generate_observe_repair(
        contract, unit_id, tmp_path, generate=generate, neutral_observe=observe,
        verify_contract=verify, plan_repair=repair,
        initial_request={"prompt": "initial", "conditions": []})
    assert result["passed"] and result["accepted_attempt"] == 3
    assert len(result["attempts"]) == 3
    assert all("hard requirement" not in prompt.lower() for prompt in prompts)
    assert all("expected action" not in prompt.lower() for prompt in prompts)


def test_repair_cannot_delete_hard_requirements(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    unit_id = contract["sections"][0]["generation_units"][0]["unit_id"]

    def generate(request: dict, output: Path) -> dict:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"video")
        return {"output_path": str(output)}

    with pytest.raises(V9GBlocked, match="repair_attempted_requirement_deletion"):
        run_generate_observe_repair(
            contract, unit_id, tmp_path, generate=generate,
            neutral_observe=lambda *_: {"observed": True},
            verify_contract=lambda *_: {"passed": False, "issues": ["bad"]},
            plan_repair=lambda *_: {"delete_hard_requirements": True},
            initial_request={"prompt": "initial", "conditions": []})


def test_event_moment_selection_preserves_order_and_budget() -> None:
    timeline = [
        {"phase": "setup", "interval": [0.0, 1.0], "supported": True},
        {"phase": "action", "interval": [4.0, 5.2], "supported": True},
        {"phase": "result", "interval": [8.0, 9.5], "supported": True},
    ]
    result = select_evidence_moments(
        timeline, required_phases=["setup", "action", "result"],
        duration_budget_s=4.0, composition_mode="event_compression_montage")
    assert result["passed"] and result["duration_s"] == 3.7
    reordered = select_evidence_moments(
        timeline, required_phases=["result", "action"], duration_budget_s=4.0,
        composition_mode="event_compression_montage")
    assert reordered["reason_code"] == "event_progression_reordered"


def test_raw_unit_pass_does_not_imply_snippet_or_section_pass(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    results = []
    for section in contract["sections"]:
        for unit in section["generation_units"]:
            results.append({"unit_id": unit["unit_id"], "passed": True,
                            "snippet_review": {"passed": False}})
    assembled = assemble_v9g_sections(contract, results)
    assert assembled["passed"] is False


def test_adoption_and_waste_ratios_include_failed_material() -> None:
    result = evaluate_v9g_experiment([
        {"passed": True, "generated_seconds": 10, "adopted_seconds": 4,
         "gpu_minutes": 3, "review_cost": 1},
        {"passed": False, "generated_seconds": 10, "adopted_seconds": 0,
         "gpu_minutes": 3, "review_cost": 1},
    ])
    assert result["success_at_budget"] == 0.5
    assert result["adoption_ratio"] == 0.2
    assert result["waste_ratio"] == 0.8
    assert result["formal_mechanism_claim_authorized"] is False


def test_d_and_e_require_real_equal_budget_and_24_complete_pairs() -> None:
    rows = []
    for unit_index in range(24):
        for arm in ("A", "B", "C1", "C2", "D", "E"):
            rows.append({
                "reference_id": f"R{unit_index // 6}", "unit_id": f"U{unit_index}",
                "arm": arm, "passed": True, "request_count": 3,
                "generated_seconds": 12, "adopted_seconds": 4,
                "resolution": "768p", "fps": 24, "inference_steps": 50,
                "quality_preset": "lossless", "gpu_minutes": 10,
                "reference_preprocessing_sha256": "same",
            })
    result = evaluate_v9g_experiment(rows)
    assert result["paired_unit_count"] == 24
    assert result["formal_mechanism_claim_authorized"] is True
    rows[-1]["generated_seconds"] = 20
    result = evaluate_v9g_experiment(rows)
    assert result["formal_mechanism_claim_authorized"] is False
    assert len(result["d_e_budget_violations"]) == 1


def test_upstream_lock_matches_vendored_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_upstream_lock(root)
    assert result["passed"], result["errors"]


def test_fake_h3_fake_omni_contract_to_raw_unit_chain(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    jobs = build_generation_jobs(contract, _caps())

    class Answer:
        def __init__(self, text: str) -> None:
            self.text = text

    class Runner:
        def watch(self, video: Path, prompt: str, **kwargs) -> Answer:
            assert "contract" not in prompt.lower()
            return Answer(json.dumps({
                "subjects": ["occ_A", "occ_B"], "initiator": "occ_A",
                "affected": "occ_B", "state_changes": {},
                "event_timeline": [{"phase": "setup", "interval": [0, 1],
                                    "supported": True}],
            }))

        def ask(self, prompt: str, **kwargs) -> Answer:
            return Answer(json.dumps({"passed": True, "issues": [],
                                      "hard_requirement_results": {},
                                      "invariant_results": {},
                                      "transition_results": {}, "summary": "ok"}))

    results = run_v9g_generation_jobs(
        contract, jobs, tmp_path / "run",
        clients={"fl2va": _FakeH3(), "ref2va": _FakeH3()}, runner=Runner(),
        capabilities=_caps())
    assert len(results) == 3
    assert all(row["passed"] for row in results)
    assert all(row["contract_hash"] == contract["contract_hash"] for row in results)


def test_cli_exposes_all_v9g_phases() -> None:
    from src.agentic_video.cli import build_parser

    parser = build_parser()
    for phase in ("license", "capability", "contract", "pilot", "generate",
                  "select", "assemble", "evaluate"):
        args = parser.parse_args([
            "reference-generate-v9g", "--phase", phase, "--output", "out"])
        assert args.phase == phase


def test_fake_backends_complete_raw_snippet_section_and_human_gates(
        tmp_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    source = tmp_path / "synthetic.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=160x90:r=24:d=1.5", "-f", "lavfi", "-i",
        "anullsrc=r=32000:cl=stereo", "-shortest", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    ], check=True)
    contract = _contract(tmp_path)
    jobs = build_generation_jobs(contract, _caps())

    class VideoH3:
        def generate(self, request: dict, output_path: Path) -> dict:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_path)
            return {"output_path": str(output_path), "sha256": _sha(output_path)}

    phase_by_unit = {"S1.G1": "setup", "S1.G2": "action",
                     "S1.G3": "visible_result"}

    class Answer:
        def __init__(self, value: dict) -> None:
            self.text = json.dumps(value)

    class Runner:
        def watch(self, video: Path, prompt: str, **kwargs) -> Answer:
            if prompt.startswith("Observe only"):
                unit_id = next(key for key in phase_by_unit if key in str(video))
                return Answer({
                    "subjects": ["occ_A", "occ_B"], "initiator": "occ_A",
                    "affected": "occ_B", "state_changes": {},
                    "event_timeline": [{
                        "phase": phase_by_unit[unit_id], "interval": [0.1, 0.6],
                        "supported": True,
                    }],
                })
            if prompt.startswith("Review this cropped"):
                return Answer({"passed": True, "phase_results": {},
                               "continuity_preserved": True,
                               "meaning_preserved": True, "issues": [],
                               "summary": "ok"})
            if prompt.startswith("Blindly review this Section"):
                return Answer({"passed": True, "audience_takeaway": "action succeeds",
                               "event_progression_clear": True,
                               "identity_stable": True, "issues": [], "summary": "ok"})
            return Answer({"passed": True, "summary": "one subject acts and gets a result",
                           "subject_consistent": True,
                           "progression_comprehensible": True, "issues": []})

        def ask(self, prompt: str, **kwargs) -> Answer:
            return Answer({"passed": True, "issues": [],
                           "hard_requirement_results": {}, "invariant_results": {},
                           "transition_results": {}, "summary": "ok"})

    run_dir = tmp_path / "run"
    runner = Runner()
    results = run_v9g_generation_jobs(
        contract, jobs, run_dir,
        clients={"fl2va": VideoH3(), "ref2va": VideoH3()}, runner=runner,
        capabilities=_caps())
    from src.config import load_config
    cfg = load_config()
    cfg.generation.setdefault("assemble", {}).update({"width": 160, "height": 90})
    state = build_asset_and_state_pack(contract)
    finalized = finalize_unit_snippets(
        cfg, contract, results, state, run_dir, runner=runner)
    assert all(row["snippet_review"]["passed"] for row in finalized)
    assembly = render_and_review_sections(
        cfg, contract, finalized, run_dir, runner=runner)
    assert assembly["passed"]
    human = _dump(tmp_path / "human.json", {
        "passed": True, "sections": [{"section_id": "S1", "passed": True}]})
    accepted = accept_v9g(run_dir, human)
    assert accepted["passed"]
    assert (run_dir / "rendered.mp4").is_file()
