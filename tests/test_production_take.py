# -*- coding: utf-8 -*-
"""生产化单条素材工作流（p0521 五项修正）的回归锚定。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agentic_video import production_take as pt
from src.agentic_video.generation_long_take import (
    LT0Blocked, _write_json, build_long_take_request,
    compile_long_take_brief, default_subject_constraints,
    validate_evidence_quality, validate_subject_scoped_actions)
from src.agentic_video.production_take import (
    ProductionBlocked, apply_human_override, evaluate_take_against_card,
    validate_attribute_transmission, validate_repair_plan)
from tests.test_generation_long_take import _p04e_fixtures


def _pack_with_approval(approved: bool = True) -> dict:
    quality = ({"visibility": "clear", "subject_attribution": "unambiguous",
                "critical_region_complete": True,
                "human_approved": True} if approved else
               {"visibility": "blurred", "subject_attribution": "unambiguous",
                "critical_region_complete": True, "human_approved": False})
    return {"schema_version": "x", "entries": [
        {"role": "c0_identity", "path": "/a.jpg", "sha256": "a",
         "uri": "file:///a.jpg", "evidence_quality": quality},
        {"role": "c0_body", "path": "/b.jpg", "sha256": "b",
         "uri": "file:///b.jpg", "evidence_quality": quality},
        {"role": "c1_opponent", "path": "/c.jpg", "sha256": "c",
         "uri": "file:///c.jpg"},
    ]}


def _brief_with_constraints() -> dict:
    fixtures = _p04e_fixtures()
    return compile_long_take_brief(
        fixtures["content_program"], fixtures["edit_program"],
        fixtures["requirement"], fixtures["section_observations"],
        constraints=default_subject_constraints())


def _request(brief: dict) -> dict:
    return build_long_take_request(
        brief, pack=_pack_with_approval(),
        visual_reference={"path": str(Path("/ref.mp4").resolve())},
        recipe="canonical_plus_video", seed=1001, seconds=12,
        aspect_ratio="3:4")


def test_evidence_quality_gate_blocks_unclear_reference() -> None:
    """修正 1：四环齐全但证据本身证明不了属性 → BLOCK BEFORE H3。"""
    constraints = default_subject_constraints()
    with pytest.raises(LT0Blocked) as excinfo:
        validate_evidence_quality(constraints, _pack_with_approval(False))
    assert excinfo.value.reason_code == "evidence_quality_gate"
    validate_evidence_quality(constraints, _pack_with_approval(True))


def test_subject_scoped_action_check_not_string_scan() -> None:
    """修正 2：禁止句/C1 手部动作不误触发；主角手部动作才拦。"""
    brief = _brief_with_constraints()
    constraints = default_subject_constraints()
    # 模板动作全是踢击/放松——不拦；wire 里"Do not add hands"也不会
    # 被扫到（检查只看结构化 claims）
    validate_subject_scoped_actions(brief, constraints)
    conflicting = dict(brief)
    conflicting["subject1_action_claims"] = [
        {"action": "grabs the opponent's collar", "phase": "action",
         "source": "test"}]
    with pytest.raises(LT0Blocked) as excinfo:
        validate_subject_scoped_actions(conflicting, constraints)
    assert excinfo.value.reason_code == "hand_action_contradicts_no_hands"


def test_wire_prompt_uses_three_layer_morphology() -> None:
    """修正 3：正面形态（指向参考）+语义+负面 guard；不发明关节位置。"""
    request = _request(_brief_with_constraints())
    prompt = request["prompt"]
    assert ("Preserve the exact upper-limb morphology visible in "
            "<Picture 2> throughout the target video." in prompt)
    assert "This defining morphology includes the absence of hands." in prompt
    assert ("Do not synthesize hands, fingers, or hand-shaped gloves for "
            "<Subject 1>." in prompt)
    assert ("Do not transfer <Subject 1>'s body attributes to "
            "<Subject 2>." in prompt)
    # 不发明形态：不出现具体关节位置断言
    assert "wrist" not in prompt.lower() or "wrists" not in prompt
    assert "elbow ends" not in prompt
    # retention 增行：critical attribute 逐条 fully_preserved
    assert "C0.hand_morphology (throughout): fully_preserved" in prompt


def test_attribute_transmission_four_links() -> None:
    brief = _brief_with_constraints()
    constraints = default_subject_constraints()
    pack = _pack_with_approval(True)
    request = _request(brief)
    mapping = validate_attribute_transmission(constraints, brief, request,
                                              pack)
    assert {row["attribute_id"] for row in mapping} == {
        "C0.target_character", "C0.hand_morphology"}
    assert all(row["in_prompt"] and row["evidence_quality_ok"]
               for row in mapping)
    # 断链：换掉 prompt 里的形态句 → 拦
    broken_request = json.loads(json.dumps(request))
    broken_request["prompt"] = broken_request["prompt"].replace(
        "Preserve the exact upper-limb morphology visible in "
        "<Picture 2> throughout the target video.", "She is an athlete.")
    with pytest.raises(ProductionBlocked) as excinfo:
        validate_attribute_transmission(constraints, brief, broken_request,
                                        pack)
    assert excinfo.value.reason_code == "attribute_missing_in_prompt"


def _attribute_result(verdicts: dict[str, str]) -> dict:
    return {"take_id": "t", "duration_s": 12.0,
            "attribute_verdicts": {
                key: {"verdict": value, "ownership_confirmed": True,
                      "evidence_interval": [0, 1], "notes": "x"}
                for key, value in verdicts.items()},
            "samples": [], "scope_note": "checked spans only"}


def test_machine_verdict_semantics_and_human_override() -> None:
    """修正 4：unknown 不折算 pass；人工 override 单独落档不覆盖机器判定。"""
    observation = {"observation": {}, "identity": {"frames": []}}
    evaluation = evaluate_take_against_card(
        observation, _attribute_result({
            "C0.target_character": "pass",
            "C0.hand_morphology": "unknown"}))
    assert evaluation["machine_verdict"] == "unknown"
    assert evaluation["final_acceptance"] is False
    # 人工 override：machine_verdict 保持 unknown，审计链不失真
    apply_human_override(evaluation, "pass",
                         "full-resolution manual inspection confirms "
                         "morphology")
    assert evaluation["machine_verdict"] == "unknown"
    assert evaluation["human_override"]["decision"] == "pass"
    assert evaluation["final_acceptance"] is True
    # fail 优先级最高
    evaluation = evaluate_take_against_card(
        observation, _attribute_result({
            "C0.target_character": "pass",
            "C0.hand_morphology": "fail"}))
    assert evaluation["machine_verdict"] == "fail"
    # 全 pass 才机器采纳
    evaluation = evaluate_take_against_card(
        observation, _attribute_result({
            "C0.target_character": "pass",
            "C0.hand_morphology": "pass"}))
    assert evaluation["machine_verdict"] == "pass"
    assert evaluation["final_acceptance"] is True


def test_repair_requires_real_condition_delta() -> None:
    """修正 5：空 delta / 仅换 seed / 条件不变 → STOP。"""
    request = _request(_brief_with_constraints())
    old_conditions = request["conditions"]
    good = {"condition_delta": {"old": old_conditions,
                                "new": old_conditions + [
                                    {"type": "image",
                                     "uri": "file:///body2.jpg",
                                     "role": "reference"}]},
            "why_this_change_addresses_failure":
                "second complementary body view strengthens morphology "
                "evidence"}
    validate_repair_plan(good, request)
    with pytest.raises(ProductionBlocked) as excinfo:
        validate_repair_plan({"condition_delta": {}}, request)
    assert excinfo.value.reason_code == "repair_without_condition_delta"
    with pytest.raises(ProductionBlocked):
        validate_repair_plan({"seed_only": True,
                              "condition_delta": {"old": [], "new": ["x"]}},
                             request)
    identical = {"condition_delta": {"old": old_conditions,
                                     "new": json.loads(
                                         json.dumps(old_conditions))},
                 "why_this_change_addresses_failure": "same"}
    with pytest.raises(ProductionBlocked):
        validate_repair_plan(identical, request)


def test_production_plan_only_runs_all_gates(tmp_path: Path,
                                              monkeypatch: pytest.
                                              MonkeyPatch) -> None:
    from src.agentic_video import generation_long_take as lt0
    p04e = _p04e_fixtures()
    p04e_dir = tmp_path / "p04e"
    p04e_dir.mkdir()
    mapping = {"content_program": "reference_content_program.json",
               "edit_program": "reference_edit_program.json",
               "requirement": "material_requirements.json",
               "section_observations": "section_observations.json"}
    for key, value in p04e.items():
        (p04e_dir / mapping[key]).write_text(json.dumps(value),
                                             encoding="utf-8")
    (p04e_dir / "review_assets").mkdir()
    (p04e_dir / "review_assets" / "section_02.mp4").write_bytes(b"clip")
    monkeypatch.setattr(
        lt0, "probe_media_geometry",
        lambda ffprobe, video: {"width": 720, "height": 1018,
                                "duration_s": 11.6})

    def fake_extract(ffmpeg_bin, reference, timestamp, jpg,
                     crop_filter=None):
        jpg.write_bytes(b"f")

    monkeypatch.setattr(lt0, "_extract_frame", fake_extract)
    monkeypatch.setattr(lt0, "_file_hash", lambda path: "hash")
    monkeypatch.setattr(lt0.common, "run_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(lt0, "detect_active_crop", lambda *a, **k: None)
    # production_take 直接导入的符号要打在其自身命名空间
    monkeypatch.setattr(
        pt, "probe_media_geometry",
        lambda ffprobe, video: {"width": 720, "height": 1018,
                                "duration_s": 11.6})

    picks = {"picks": {"c0_identity": "candidate_03",
                       "c0_body": "candidate_03",
                       "c1_opponent": "candidate_05"},
             "evidence_quality": {
                 "c0_identity": {"visibility": "clear",
                                 "subject_attribution": "unambiguous",
                                 "critical_region_complete": True,
                                 "human_approved": True},
                 "c0_body": {"visibility": "clear",
                             "subject_attribution": "unambiguous",
                             "critical_region_complete": True,
                             "human_approved": True}}}
    summary = pt.run_production_take(
        _Cfg(), p04e_dir, tmp_path / "out", reference=tmp_path / "v.mp4",
        pack_picks=picks, plan_only=True)
    assert summary["phase"] == "planned"
    assert summary["gates_passed"] == ["wire_audit", "evidence_quality",
                                       "subject_scoped_actions",
                                       "attribute_transmission"]
    plan = json.loads((tmp_path / "out" / "production_plan.json")
                      .read_text(encoding="utf-8"))
    assert plan["budget"]["max_h3_requests"] == 2
    assert plan["budget"]["allow_seed_sweep"] is False
    assert {row["attribute_id"] for row in plan["attribute_transmission"]} == {
        "C0.target_character", "C0.hand_morphology"}
    # 证据未批准 → 计划阶段即拦
    with pytest.raises(LT0Blocked) as excinfo:
        pt.run_production_take(
            _Cfg(), p04e_dir, tmp_path / "out2", reference=tmp_path / "v.mp4",
            pack_picks={"picks": picks["picks"]}, plan_only=True)
    assert excinfo.value.reason_code == "evidence_quality_gate"


class _Cfg:
    perception = {"omni": {}}
