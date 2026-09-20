# -*- coding: utf-8 -*-
"""M3-A Storyboard 回归锚定：schema 校验 / 依赖门 / Reference Router /
帧对 L1 / blocked 不崩溃。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.agentic_video.skills.registry import SkillBlocked
from src.agentic_video.storyboard.shot_schema import (
    SHOT_PLAN_SCHEMA_VERSION, validate_shot_plan)
from src.agentic_video.storyboard.skills import (
    build_m3_registry, pick_pilot_shot, select_references)
from src.agentic_video.validators import run_test
from src.agentic_video.workspace import Workspace

# ---- fixtures ----

SCREENPLAY = {
    "title": "T", "target_duration": 22,
    "scenes": [{"scene_id": "SC01", "role": "setup", "beats": [
        {"beat_id": "B1", "purpose": "situation_setup",
         "visual_action": "a woman checks her phone"},
        {"beat_id": "B2", "purpose": "counter_evidence",
         "visual_action": "her face freezes as she reads"},
        {"beat_id": "B3", "purpose": "visible_result",
         "visual_action": "she puts the phone face down"}]}]}

ASSET_GRAPH = {
    "schema_version": "asset_graph_v1",
    "assets": [
        {"asset_id": "C0", "type": "character", "tier": "A",
         "immutable": {"face": "oval"}, "status": "active"},
        {"asset_id": "L01", "type": "location", "tier": "A",
         "status": "active"},
        {"asset_id": "P01", "type": "prop", "tier": "B",
         "status": "active"}]}


def _shot(shot_id="SH01", beat="B1", **over):
    shot = {"shot_id": shot_id, "beat_id": beat,
            "narrative_role": "situation_setup",
            "duration_budget_s": [3.5, 4.5],
            "characters": ["C0"], "location": "L01",
            "props": [], "wardrobe": [], "motion_refs": [],
            "camera": {"shot_size": "medium", "movement": "static"},
            "start_state": "phone idle in her hand",
            "end_state": "she looks at the screen",
            "visual_events": ["lift phone", "look"]}
    shot.update(over)
    return shot


def _plan(shots):
    return {"schema_version": SHOT_PLAN_SCHEMA_VERSION, "shots": shots}


class _Runner:
    """文本 ask + inspect_media 假 Omni。"""

    def __init__(self, ask_texts=(), media_texts=()) -> None:
        self.ask_texts = list(ask_texts)
        self.media_texts = list(media_texts)
        self.prompts: list[str] = []

    def ask(self, prompt, **kw):
        self.prompts.append(prompt)
        text = self.ask_texts.pop(0) if self.ask_texts else "{}"
        return SimpleNamespace(text=text)

    def inspect_media(self, image_paths, prompt, **kw):
        text = self.media_texts.pop(0) if self.media_texts \
            else '{"passed": true}'
        return SimpleNamespace(text=text)


def _ws(tmp_path: Path, *, with_c0: str = "committed") -> Workspace:
    """committed 剧本链 + 可选 C0 final 资产（with_c0: committed/draft/None）。"""
    ws = Workspace(tmp_path / "run")
    ws.state["goal"] = {"target_artifact": "storyboard_frames",
                        "required_status": "committed"}
    for name, data in (("creative_dna", {"narrative_invariants": {"x": 1}}),
                       ("screenplay", SCREENPLAY),
                       ("asset_graph", ASSET_GRAPH)):
        ws.write_draft(name, data)
        ws.record_dependency_snapshot(name)
        ws.commit(name)
    ws.set_dependency("shot_plan", "screenplay", "committed")
    ws.set_dependency("storyboard_frames", "shot_plan", "committed")
    if with_c0:
        files = ws.root / "03_asset_studio" / "files"
        files.mkdir(parents=True, exist_ok=True)
        views_4k = {}
        for view in ("front", "face"):
            p = files / f"C0_{view}_v1.png"
            Image.new("RGB", (108, 192), (120, 120, 120)).save(p)
            views_4k[view] = {"file": f"03_asset_studio/files/{p.name}",
                              "sha": f"{view}-sha"}
        ws.write_draft("asset:C0", {"asset_id": "C0",
                                    "views_4k": views_4k})
        ws.record_dependency_snapshot("asset:C0")
        if with_c0 == "committed":
            ws.commit("asset:C0")
    return ws


# ---- schema 校验 ----

def test_valid_plan_passes() -> None:
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 duration_budget_s=[5.0, 6.5],
                                 start_state="she scans the message",
                                 end_state="her smile drops"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="phone in her hand",
                       end_state="phone face down on table")])
    assert validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH) == []


def test_missing_beat_and_bad_refs_fail() -> None:
    plan = _plan([_shot(), _shot("SH02", "B2", characters=["C9"],
                                 start_state="a", end_state="b")])
    failures = validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "beat_coverage" and "B3" in f["detail"]
               for f in failures)
    assert any(f["check"] == "asset_ref" and "C9" in f["detail"]
               for f in failures)


def test_causality_and_duration_and_camera_fail() -> None:
    plan = _plan([
        _shot(),  # OK
        _shot("SH02", "B2", start_state="same", end_state="same",
              duration_budget_s=[3.0, 2.0],
              camera={"shot_size": "hologram", "movement": "teleport"}),
        _shot("SH03", "B3", start_state="a", end_state="b")])
    failures = validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "causality" for f in failures)
    assert any(f["check"] == "duration" for f in failures)
    assert any(f["check"] == "shot_size" for f in failures)
    assert any(f["check"] == "movement" for f in failures)


def test_event_density_fail() -> None:
    plan = _plan([_shot(visual_events=["a", "b", "c", "d"]),
                  _shot("SH02", "B2", start_state="a", end_state="b"),
                  _shot("SH03", "B3", start_state="a", end_state="b")])
    failures = validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "event_density" for f in failures)


def test_total_duration_window() -> None:
    plan = _plan([_shot(duration_budget_s=[12.0, 13.0]),
                  _shot("SH02", "B2", duration_budget_s=[12.0, 13.0],
                       start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[12.0, 13.0],
                       start_state="a", end_state="b")])
    failures = validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "total_duration" for f in failures)


# ---- Skill + 依赖门（核心） ----

def test_plan_skill_schema_in_prompt_and_commit(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 duration_budget_s=[5.0, 6.5],
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b")])
    runner = _Runner(ask_texts=[json.dumps(plan)])
    registry = build_m3_registry(runner=runner)
    registry.execute({"skill": "plan_storyboard"}, ws)
    assert SHOT_PLAN_SCHEMA_VERSION in runner.prompts[0]  # schema 原生内嵌
    report = run_test("test_shot_plan", ws, runner=runner)
    assert report["passed"], report["failures"]
    ws.record_dependency_snapshot("shot_plan")
    ws.commit("shot_plan")
    assert ws.get_status("shot_plan") == "committed"


def test_repair_injects_failures_and_patches_locally(
        tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    bad = _plan([_shot(), _shot("SH02", "B2", characters=["C9"],
                                duration_budget_s=[5.0, 6.5],
                                start_state="a", end_state="b"),
                 _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b")])
    ws.write_draft("shot_plan", bad)
    good = _plan([_shot(), _shot("SH02", "B2", characters=["C0"],
                                 duration_budget_s=[5.0, 6.5],
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b")])
    runner = _Runner(ask_texts=[json.dumps(good)])
    registry = build_m3_registry(runner=runner)
    failures = validate_shot_plan(bad, SCREENPLAY, ASSET_GRAPH)
    registry.execute({"skill": "repair_shot_plan"}, ws,
                     test_failures=failures, target="SH02")
    assert "C9" in runner.prompts[0]  # 失败明细进 prompt
    assert run_test("test_shot_plan", ws, runner=runner)["passed"]


def test_gate_blocks_when_location_not_committed(tmp_path: Path) -> None:
    """C0 已 COMMIT 但 L01 未产 → 带 location 的 shot BLOCKED。"""
    ws = _ws(tmp_path / "loc", with_c0="committed")
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 location="L01",
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", start_state="a", end_state="b")])
    ws.write_draft("shot_plan", plan)
    ws.record_dependency_snapshot("shot_plan")
    ws.commit("shot_plan")
    registry = build_m3_registry(runner=None)
    try:
        registry.execute({"skill": "generate_storyboard_frame",
                          "target": "SH02"}, ws)
        raise AssertionError("must be blocked")
    except SkillBlocked as blocked:
        assert "asset:L01" in str(blocked)


def test_router_picks_face_for_closeup(tmp_path: Path) -> None:
    ws = _ws(tmp_path, with_c0="committed")
    refs, missing = select_references(
        _shot(camera={"shot_size": "closeup", "movement": "static"},
              location=None), ws)
    assert missing == []
    assert refs == ["03_asset_studio/files/C0_face_v1.png"]
    refs2, _ = select_references(
        _shot(location=None), ws)
    assert refs2 == ["03_asset_studio/files/C0_front_v1.png"]


def test_dependency_gate_blocks_without_committed_assets(
        tmp_path: Path) -> None:
    """核心：C0 未 COMMIT（或 location 缺）→ BLOCKED，绝不烧 GPU。"""
    for c0_state in ("draft", None):
        ws = _ws(tmp_path / f"case_{c0_state}", with_c0=c0_state)
        plan = _plan([_shot(), _shot("SH02", "B2",
                                     narrative_role="counter_evidence",
                                     location=None,
                                     start_state="a", end_state="b"),
                      _shot("SH03", "B3", start_state="a",
                            end_state="b")])
        ws.write_draft("shot_plan", plan)
        ws.record_dependency_snapshot("shot_plan")
        ws.commit("shot_plan")
        registry = build_m3_registry(runner=None)
        try:
            registry.execute(
                {"skill": "generate_storyboard_frame",
                 "target": "SH02"}, ws)
            raise AssertionError("must be blocked")
        except SkillBlocked as blocked:
            assert "STORYBOARD:SH02 BLOCKED" in str(blocked)
            assert "asset:C0" in str(blocked) or \
                "backend" in str(blocked)


def test_pilot_shot_prefers_counter_evidence_min_missing(
        tmp_path: Path) -> None:
    ws = _ws(tmp_path, with_c0="committed")
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 location=None,
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", start_state="a", end_state="b")])
    assert pick_pilot_shot(plan, ws)["shot_id"] == "SH02"


def test_agent_loop_survives_blocked(tmp_path: Path) -> None:
    """agent_loop 里 BLOCKED 落 trace 继续跑（不崩溃）。"""
    from src.agentic_video.agent_v4 import agent_loop
    from src.agentic_video.no_progress import NoProgressDetector
    ws = _ws(tmp_path / "loop", with_c0=None)
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 location=None,
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", start_state="a", end_state="b")])
    ws.write_draft("shot_plan", plan)
    ws.record_dependency_snapshot("shot_plan")
    ws.commit("shot_plan")
    registry = build_m3_registry(runner=None)
    runner = _Runner(ask_texts=[
        '{"skill": "generate_storyboard_frame", "target": "SH02"}']
        * 6)
    result = agent_loop(ws, registry, controller_runner=runner,
                        max_steps=6,
                        no_progress_detector=NoProgressDetector(limit=3))
    assert result["stop_reason"] == "no_progress"  # 全被门拦住 → 停滞停机
    trace = ws.recent_trace(6)
    assert any(t.get("tests", {}).get("failures", [{}])[0].get("check")
               == "dependency_gate" for t in trace)


# ---- 帧对 L1 ----

def test_dict_state_causality_and_entity_refs() -> None:
    """per-entity dict state：至少一键变化；entity 前缀必须存在。"""
    plan = _plan([
        _shot(start_state={"C0.pose": "stand", "P01.state": "off"},
              end_state={"C0.pose": "lean in", "P01.state": "on"},
              duration_budget_s=[5.0, 6.5]),
        _shot("SH02", "B2",
              duration_budget_s=[5.0, 6.5],
              start_state={"C0.pose": "a"}, end_state={"C0.pose": "b"}),
        _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
              start_state="a", end_state="b")])
    assert validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH) == []

    # 无变化的 dict state
    bad = _plan([_shot(start_state={"C0.pose": "x"},
                       end_state={"C0.pose": "x"},
                       duration_budget_s=[5.0, 6.5]),
                 _shot("SH02", "B2", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b"),
                 _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b")])
    failures = validate_shot_plan(bad, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "causality" for f in failures)

    # 未知 entity 前缀
    bad2 = _plan([_shot(start_state={"C9.pose": "a"},
                        end_state={"C9.pose": "b"},
                        duration_budget_s=[5.0, 6.5]),
                  _shot("SH02", "B2", duration_budget_s=[5.0, 6.5],
                        start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                        start_state="a", end_state="b")])
    failures = validate_shot_plan(bad2, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "state_entity_ref" for f in failures)


def test_camera_angle_vocab() -> None:
    plan = _plan([_shot(camera={"shot_size": "medium",
                                "angle": "warp_speed",
                                "movement": "static"}),
                  _shot("SH02", "B2", duration_budget_s=[5.0, 6.5],
                        start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                        start_state="a", end_state="b")])
    failures = validate_shot_plan(plan, SCREENPLAY, ASSET_GRAPH)
    assert any(f["check"] == "camera_angle" for f in failures)


def test_storyboard_dependencies_validator(tmp_path: Path) -> None:
    """READY/BLOCKED 清单：C0 committed + location 未产 → 仅无地点镜 READY。"""
    ws = _ws(tmp_path / "deps", with_c0="committed")
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 location=None,
                                 duration_budget_s=[5.0, 6.5],
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                        start_state="a", end_state="b")])
    ws.write_draft("shot_plan", plan)
    report = run_test("test_storyboard_dependencies", ws, runner=None)
    assert not report["passed"]
    assert "1/3 shots READY" in report["detail"]
    blocked_ids = {f["shot_id"] for f in report["failures"]}
    assert blocked_ids == {"SH01", "SH03"}  # L01 未产 → 带 location 的全 BLOCKED


def test_pilot_shot_prefers_simpler(tmp_path: Path) -> None:
    """同缺资产时选人物+道具更少的镜头。"""
    ws = _ws(tmp_path / "simple", with_c0="committed")
    plan = _plan([_shot(), _shot("SH02", "B2",
                                 narrative_role="counter_evidence",
                                 location=None, props=["P01", "P02"],
                                 duration_budget_s=[5.0, 6.5],
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", location=None,
                        narrative_role="counter_evidence",
                        duration_budget_s=[5.0, 6.5],
                        start_state="a", end_state="b")])
    # P01/P02 不在 fixture 资产表里会挂校验——但选镜只看复杂度
    assert pick_pilot_shot(plan, ws)["shot_id"] == "SH03"


def test_acceptance_report(tmp_path: Path) -> None:
    """验收报告：candidate 语义 + 修复史 + 结构化 recommended refs。"""
    from src.agentic_video.asset_studio.acceptance import (
        build_acceptance_report)
    ws = _ws(tmp_path / "acc", with_c0="committed")
    ws.write_draft("asset:C0_master", {
        "identity_description": "a woman",
        "master": {"file": "03_asset_studio/files/C0_master_front_v1.png",
                   "sha": "m1", "generator": "qwen-image"}})
    ws.record_dependency_snapshot("asset:C0_master")
    ws.commit("asset:C0_master")
    views = {"views": {v: {"file": f"03_asset_studio/files/C0_{v}_v1.png",
                           "sha4k": f"{v}-sha", "file4k":
                           f"03_asset_studio/files/C0_{v}_v1.png"}
                       for v in ("front", "profile", "back", "face")}}
    ws.write_draft("asset:C0_views", views)
    ws.record_dependency_snapshot("asset:C0_views")
    ws.commit("asset:C0_views")
    ws.write_draft("asset:C0_candidate", {
        "identity_description": "a woman",
        "views_4k": {v: {"file": views["views"][v]["file4k"],
                         "sha": f"{v}-sha"}
                     for v in ("front", "profile", "back", "face")},
        "sheet": {"file": "03_asset_studio/files/C0_sheet_v1.png",
                  "sha": "s1"}})
    ws.record_dependency_snapshot("asset:C0_candidate")
    ws.commit("asset:C0_candidate")
    ws.record_trace(5, {"skill": "repair_character_view",
                        "target": "profile"}, {"target": "profile"},
                    {"passed": True})

    report = build_acceptance_report(ws, "C0")
    # final（_ws 里的 asset:C0 views_4k fixture）→ committed
    assert report["overall"] == "committed"
    assert report["views"]["profile"]["repair_count"] == 1
    assert report["views"]["front"]["repair_count"] == 0
    rec = report["recommended_reference_roles"]
    # P1-5：结构化——view id 永远合法，修复史用显式字段
    assert rec["face_identity"]["primary"] == ["face"]
    assert rec["face_identity"]["fallback"] == ["front"]
    assert rec["side_body"]["requires_extra_verification"] is True
    assert rec["side_body"]["note"]
    all_ids = [v for entry in rec.values()
               for v in entry["primary"] + entry["fallback"]]
    assert all(v in ("front", "profile", "back", "face") for v in all_ids)


def test_inspect_never_commits(tmp_path: Path) -> None:
    """M3-A dry-run 教训：无 validator 的只读动作不得触发 commit。"""
    from src.agentic_video.agent_v4 import agent_loop
    ws = _ws(tmp_path / "no_commit")
    bad = _plan([_shot(), _shot("SH02", "B2", characters=["C9"],
                                duration_budget_s=[5.0, 6.5],
                                start_state="a", end_state="b"),
                 _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b")])
    ws.write_draft("shot_plan", bad)  # draft，会被校验打挂
    registry = build_m3_registry(runner=None)
    runner = _Runner(ask_texts=['{"skill": "inspect_shot_plan"}'] * 4)
    result = agent_loop(ws, registry, controller_runner=runner,
                        max_steps=4)
    # inspect 循环跑了，但 shot_plan 绝不能变成 committed
    assert ws.get_status("shot_plan") != "committed"
    assert not result["goal_satisfied"]


def test_repair_unwraps_current_shot_plan_wrapper(tmp_path: Path) -> None:
    """Omni 照抄 payload 键 current_shot_plan → 必须剥出内层。"""
    ws = _ws(tmp_path / "unwrap")
    ws.write_draft("shot_plan", _plan([_shot()]))
    good = _plan([_shot(), _shot("SH02", "B2",
                                 duration_budget_s=[5.0, 6.5],
                                 start_state="a", end_state="b"),
                  _shot("SH03", "B3", duration_budget_s=[5.0, 6.5],
                       start_state="a", end_state="b")])
    runner = _Runner(ask_texts=[
        json.dumps({"current_shot_plan": good})])  # 镜像包装
    registry = build_m3_registry(runner=runner)
    registry.execute({"skill": "repair_shot_plan"}, ws,
                     test_failures=[{"shot_id": "_root",
                                     "check": "beat_coverage"}])
    repaired = ws.read_artifact("shot_plan")
    assert len(repaired.get("shots") or []) == 3  # 剥出了内层


def test_frame_pair_l1_and_l2(tmp_path: Path) -> None:
    ws = _ws(tmp_path / "frames", with_c0="committed")
    files = ws.root / "04_storyboard" / "files"
    files.mkdir(parents=True, exist_ok=True)
    for name in ("SH02_start_v1.png", "SH02_end_v1.png"):
        Image.new("RGB", (1280, 720), (110, 110, 110)).save(files / name)
    manifest = {
        "shot_id": "SH02",
        "shot": _shot("SH02", "B2", location=None),
        "identity_refs": ["03_asset_studio/files/C0_face_v1.png"],
        "start_frame": {"file": "04_storyboard/files/SH02_start_v1.png"},
        "end_frame": {"file": "04_storyboard/files/SH02_end_v1.png"}}
    ws.write_draft("storyboard_frames", manifest)
    report = run_test("test_storyboard_frame_pair", ws,
                      runner=_Runner(media_texts=['{"passed": true}']))
    assert report["passed"], report["failures"]

    # 非 16:9 → L1 FAIL（Omni 不烧）
    class Boom:
        def inspect_media(self, *a, **kw):  # pragma: no cover
            raise AssertionError("L1 fail must not call Omni")

    Image.new("RGB", (800, 800), (110, 110, 110)).save(
        files / "SH02_start_v2.png")
    manifest["start_frame"]["file"] = \
        "04_storyboard/files/SH02_start_v2.png"
    ws.write_draft("storyboard_frames", manifest)
    report = run_test("test_storyboard_frame_pair", ws, runner=Boom())
    assert not report["passed"]
    assert any("aspect" in f["check"] for f in report["failures"])
