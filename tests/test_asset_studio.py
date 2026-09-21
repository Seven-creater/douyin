# -*- coding: utf-8 -*-
"""M2-A Character Asset Studio 回归锚定：全链 commit / L1 先行 /
per-view 局部修复 / 预算门 / 冒号前置条件 / 背景启发式。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.agentic_video.asset_studio.image_io import (
    check_background_clean, compose_identity_sheet, upscale_lanczos)
from src.agentic_video.asset_studio.skills import build_m2a_registry
from src.agentic_video.validators import run_test
from src.agentic_video.workspace import Workspace


# ---- 假后端：灰底 + 中心色块（过 L1） ----

def _figure_image(tint: tuple[int, int, int],
                  size: tuple[int, int] = (576, 1024)) -> Image.Image:
    from PIL import ImageDraw
    img = Image.new("RGB", size, (128, 128, 128))
    w, h = size
    draw = ImageDraw.Draw(img)
    # 中央"人物"色块（保持 4% 边缘为纯灰背景，过 L1 启发式）
    x0 = round(w * 0.28)
    x1 = round(w * 0.72)
    y0 = round(h * 0.12)
    y1 = round(h * 0.82)
    draw.rectangle([x0, y0, x1, y1], fill=tint)
    return img


class FakeT2I:
    def generate(self, prompt, out_path, *, width, height, seed, steps=30):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _figure_image((140, 90, 70), (width, height)).save(path)
        return {"path": str(path), "width": width, "height": height,
                "seed": seed}


class FakeEdit:
    def edit(self, reference_path, prompt, out_path, *, width=1152,
             height=2048, seed=0, steps=30):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tint = {"front": (140, 90, 70), "profile": (120, 80, 90),
                "back": (100, 70, 100), "face": (150, 100, 80)}.get(
            path.stem.split("_")[1] if "_" in path.stem else path.stem,
            (120, 120, 120))
        _figure_image(tint, (width, height)).save(path)
        return {"path": str(path), "width": width, "height": height,
                "seed": seed}


class FakeMultiView:
    def __init__(self) -> None:
        self.edit_backend = FakeEdit()
        self.edit_calls: list[str] = []
        self.last_repair_kwargs: dict = {}

    def generate_views(self, master_path, view_ids, out_dir,
                       identity_description, seed_base, name_prefix=""):
        results = {}
        for offset, view in enumerate(view_ids):
            out = out_dir / f"{name_prefix}{view}_v1.png"
            self.edit_calls.append(view)
            results[view] = self.edit_backend.edit(
                master_path, view, out, seed=seed_base + offset)
        return results

    def edit_single(self, reference_path, view, out_path,
                    identity_description, seed, failures=None, attempt=1):
        self.edit_calls.append(f"repair:{view}")
        self.last_repair_kwargs = {"failures": failures,
                                   "attempt": attempt, "seed": seed}
        return self.edit_backend.edit(reference_path, view, out_path,
                                      seed=seed)


class FakeUpscale:
    def upscale(self, src_path, out_path):
        path = upscale_lanczos(src_path, out_path)
        return {"path": str(path)}


class FakeOmniRunner:
    """ask（choose_action）与 inspect_media（validator）分通道脚本。"""

    def __init__(self, ask_texts=(), media_texts=()) -> None:
        self.ask_texts = list(ask_texts)
        self.media_texts = list(media_texts)
        self.media_calls: list[list[str]] = []

    def ask(self, prompt, **kw):
        text = self.ask_texts.pop(0) if self.ask_texts \
            else '{"skill": "inspect_character_asset"}'
        return SimpleNamespace(text=text)

    def inspect_media(self, image_paths, prompt, **kw):
        self.media_calls.append([str(p) for p in image_paths])
        text = self.media_texts.pop(0) if self.media_texts \
            else '{"passed": true}'
        return SimpleNamespace(text=text)


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "run")
    # 夜链目标 = candidate（P0-5：final COMMIT 必须等人审）
    ws.state["goal"] = {"target_artifact": "asset:C0_candidate",
                        "required_status": "committed"}
    graph = {"schema_version": "asset_graph_v1",
             "assets": [
                 {"asset_id": "C0", "type": "character", "tier": "A",
                  "canonical": {"description": "a young woman, tall"},
                  "immutable": {"face": "oval face", "body_build": "slim"},
                  "status": "active"},
                 {"asset_id": "L01", "type": "location", "tier": "A",
                  "status": "active"}]}
    ws.write_draft("asset_graph", graph)
    ws.record_dependency_snapshot("asset_graph")
    ws.commit("asset_graph")
    from src.agentic_video.asset_studio.workspace_setup import (
        prepare_m2a_workspace)
    prepare_m2a_workspace(ws)  # 生产依赖注册（P0-7）
    return ws


def _registry() -> tuple:
    mv = FakeMultiView()
    reg = build_m2a_registry(t2i=FakeT2I(), multiview=mv,
                             upscale=FakeUpscale())
    return reg, mv


# ---- 全链（agent_loop 集成） ----

def test_full_chain_master_to_candidate(tmp_path: Path) -> None:
    """机器链终点 = candidate COMMIT；final 必须等人审（P0-5）。"""
    from src.agentic_video.agent_v4 import agent_loop
    ws = _ws(tmp_path)
    reg, _ = _registry()
    runner = FakeOmniRunner(
        ask_texts=[
            '{"skill": "generate_character_master", "reason": "start"}',
            '{"skill": "generate_character_multiview", "reason": "next"}',
            '{"skill": "upscale_character_view", "reason": "views pass"}',
            '{"skill": "compose_identity_sheet", "reason": "4k done"}'],
        media_texts=[
            '{"passed": true, "target_fidelity": {"passed": true},'
            ' "failures": []}',
            '{"passed": true, "views": ['
            '{"view": "front", "same_person": true, "view_correct": true},'
            '{"view": "profile", "same_person": true, "view_correct": true},'
            '{"view": "back", "same_person": true, "view_correct": true},'
            '{"view": "face", "same_person": true, "view_correct": true}]}',
            '{"passed": true, "failures": []}'])
    result = agent_loop(ws, reg, controller_runner=runner,
                        max_steps=10, budget="gpu")
    assert result["goal_satisfied"], result
    assert result["stop_reason"] == "goal_satisfied"
    assert ws.get_status("asset:C0_candidate") == "committed"
    # P0-5 核心：人审未发生，final 绝不 COMMIT
    assert ws.effective_status("asset:C0") != "committed"
    # 四张单图 + sheet 落盘且 4K
    files = ws.root / "03_asset_studio" / "files"
    for view in ("front", "profile", "back", "face"):
        matches = sorted(files.glob(f"C0_{view}*"))
        assert matches, f"missing {view} files"
    sheet = json.loads(
        (ws.root / "03_asset_studio" / "C0_candidate" / "v1.json"
         ).read_text(encoding="utf-8"))
    for view, entry in sheet["views_4k"].items():
        img = Image.open(ws.root / entry["file"])
        assert max(img.size) >= 3840
    assert (ws.root / sheet["sheet"]["file"]).is_file()


def test_human_gate_final_commit_flow(tmp_path: Path) -> None:
    """approve 流程：SHA 绑定 → final COMMIT；错绑/缺审批全 BLOCKED。"""
    from src.agentic_video.agent_v4 import agent_loop
    from src.agentic_video.skills.registry import SkillBlocked
    ws = _ws(tmp_path)
    reg, _ = _registry()
    runner = FakeOmniRunner(
        ask_texts=[
            '{"skill": "generate_character_master"}',
            '{"skill": "generate_character_multiview"}',
            '{"skill": "upscale_character_view"}',
            '{"skill": "compose_identity_sheet"}'],
        media_texts=[
            '{"passed": true, "target_fidelity": {"passed": true}}',
            '{"passed": true, "views": []}',
            '{"passed": true}'])
    agent_loop(ws, reg, controller_runner=runner, max_steps=6,
               budget="gpu")
    assert ws.get_status("asset:C0_candidate") == "committed"

    # 无人审 → approve 被 precondition 拦
    spec = reg.get("approve_character_asset")
    assert not spec.can_run(ws)[0]

    # 错绑 SHA 的审批 → execute 拦截
    ws.write_draft("human_approval:C0",
                   {"decision": "approve", "candidate_sha": "deadbeef",
                    "views": {}})
    ws.commit("human_approval:C0")
    try:
        reg.execute({"skill": "approve_character_asset"}, ws)
        raise AssertionError("must be blocked")
    except SkillBlocked as blocked:
        assert "candidate_sha_mismatch" in str(blocked)

    # 正确绑定 → final COMMIT
    candidate = ws.read_artifact("asset:C0_candidate")
    from src.agentic_video.provenance import APPROVAL_POLICY_VERSION
    parent_shas = [
        {"artifact_id": "asset:C0_candidate",
         "sha": ws.get_sha("asset:C0_candidate")},
        *[{"artifact_id": f"view:{view}", "sha": entry["sha"]}
          for view, entry in candidate["views_4k"].items()],
        {"artifact_id": "identity_sheet", "sha": candidate["sheet"]["sha"]},
    ]
    ws.write_draft("human_approval:C0",
                   {"decision": "approve", "reviewer": "human",
                    "candidate_id": "asset:C0_candidate",
                    "candidate_sha": ws.get_sha("asset:C0_candidate"),
                    "parent_shas": parent_shas,
                    "approval_policy_version": APPROVAL_POLICY_VERSION,
                    "views": {v: e["sha"] for v, e in
                              candidate["views_4k"].items()}})
    ws.commit("human_approval:C0")
    result = reg.execute({"skill": "approve_character_asset"}, ws)
    assert result["artifact"] == "asset:C0"
    report = run_test("test_asset_approved", ws, runner=None)
    assert report["passed"], report["failures"]
    ws.commit("asset:C0")
    assert ws.effective_status("asset:C0") == "committed"


def test_l1_failure_skips_omni(tmp_path: Path) -> None:
    """比例错（非 9:16）→ L1 FAIL，Omni 一次都不调。"""
    ws = _ws(tmp_path)
    reg, _ = _registry()

    class Boom:
        def ask(self, *a, **kw):  # pragma: no cover
            raise AssertionError("L1 fail must not call Omni")

        def inspect_media(self, *a, **kw):  # pragma: no cover
            raise AssertionError("L1 fail must not call inspect_media")

    # 手工塞一张 1:1 master
    out = ws.root / "03_asset_studio" / "files" / "C0_master_front_v1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _figure_image((140, 90, 70), (800, 800)).save(out)
    ws.write_draft("asset:C0_master", {
        "asset_id": "C0",
        "master": {"file": "03_asset_studio/files/C0_master_front_v1.png",
                   "view": "front"}})
    report = run_test("test_character_master", ws, runner=Boom())
    assert not report["passed"]
    assert any(f["check"] == "aspect" for f in report["failures"])


def test_repair_only_touches_failing_view(tmp_path: Path) -> None:
    """profile FAIL → 只重生成 profile；其余视图文件与 sha 原样。"""
    ws = _ws(tmp_path)
    reg, mv = _registry()
    reg.execute({"skill": "generate_character_master"}, ws)
    reg.execute({"skill": "generate_character_multiview"}, ws)
    views = ws.read_artifact("asset:C0_views")

    failing = FakeOmniRunner(media_texts=[
        '{"passed": false, "failures": ['
        '{"view": "profile", "check": "identity", '
        '"detail": "face shape drift"}]}'])
    report = run_test("test_character_multiview", ws, runner=failing)
    assert not report["passed"]

    reg.execute({"skill": "repair_character_view", "target": "profile"}, ws,
                test_failures=report["failures"], target="profile")
    repaired = ws.read_artifact("asset:C0_views")
    for view in ("front", "back", "face"):
        assert (repaired["views"][view]["file"]
                == views["views"][view]["file"])
        assert (repaired["views"][view]["sha"]
                == views["views"][view]["sha"])
    assert repaired["views"]["profile"]["file"] != \
        views["views"]["profile"]["file"]
    assert mv.edit_calls == ["profile", "back", "face", "repair:profile"]
    # v2 是新版本（不可变语义）
    assert (ws.root / "03_asset_studio" / "C0_views"
            / "v2.json").is_file()


def test_budget_gate_blocks_gpu(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.write_draft("asset:C0_master", {"asset_id": "C0"})  # exists
    reg, _ = _registry()
    cheap = [s["name"] for s in reg.runnable_skills(ws, "cheap_text")]
    assert "generate_character_master" not in cheap
    assert "inspect_character_asset" in cheap
    gpu = [s["name"] for s in reg.runnable_skills(ws, "gpu")]
    assert "generate_character_master" in gpu


def test_colon_precondition_resolution(tmp_path: Path) -> None:
    """"asset:C0_master:committed" → artifact=asset:C0_master。"""
    ws = _ws(tmp_path)
    reg, _ = _registry()
    spec = reg.get("generate_character_multiview")
    assert spec.preconditions == ["asset:C0_master:committed"]
    can, reason = spec.can_run(ws)
    assert not can and "asset:C0_master" in reason
    ws.write_draft("asset:C0_master", {"asset_id": "C0"})
    ws.record_dependency_snapshot("asset:C0_master")
    ws.commit("asset:C0_master")
    assert spec.can_run(ws)[0] is True


def test_background_cleanliness_heuristic() -> None:
    gray = Image.new("RGB", (200, 356), (128, 128, 128))
    assert check_background_clean(gray)
    noisy = Image.new("RGB", (200, 356), (10, 200, 40))
    assert not check_background_clean(noisy)


def test_identity_sheet_is_overview(tmp_path: Path) -> None:
    """sheet 只是 overview：2×2 布局，四格内容来自四张单图。"""
    d = tmp_path / "views"
    d.mkdir()
    paths = {}
    for view, tint in (("front", (140, 90, 70)), ("profile", (120, 80, 90)),
                       ("back", (100, 70, 100)), ("face", (150, 100, 80))):
        p = d / f"{view}.png"
        _figure_image(tint).save(p)
        paths[view] = p
    sheet = compose_identity_sheet(paths, tmp_path / "sheet.png")
    img = Image.open(sheet)
    # 2×2 + gaps：宽 > 2×cell 即结构成立
    assert img.width > 2 * 640 and img.height > 2 * 1138


# ---- 修复轮回归（外部 review P0/P1 锚定） ----

def test_unknown_validator_fails_closed(tmp_path: Path) -> None:
    """P0-3：typo 的 validator 名必须 FAIL（不得静默放行）。"""
    ws = _ws(tmp_path)
    report = run_test("test_character_assset", ws, runner=None)  # typo
    assert report["passed"] is False
    assert report["failures"][0]["check"] == "validator_registry"


def test_omni_json_tolerates_trailing_text() -> None:
    """夜链真机教训：Omni 视觉模式在 JSON 后追加说明文字——
    raw_decode 提取首个完整对象，解析不再挂（Extra data）。"""
    from types import SimpleNamespace
    from src.agentic_video.asset_studio.validators import _parse_omni_json
    trailing = ('{"passed": true, "target_fidelity": {"passed": true},'
                '\n "failures": []}\n'
                '根据以上检查，这张全身照符合所有要求。人物姿态自然，'
                '背景为中性灰无缝背景。')
    value = _parse_omni_json(SimpleNamespace(text=trailing))
    assert value["passed"] is True
    fenced = ('```json\n{"passed": false, "failures": '
              '[{"check": "x"}]}\n```\n补充说明文字')
    value2 = _parse_omni_json(SimpleNamespace(text=fenced))
    assert value2["passed"] is False


def test_front_view_repair_forbidden(tmp_path: Path) -> None:
    """P1-3：front = canonical master，禁止局部修（必须重修 master）。"""
    from src.agentic_video.skills.registry import SkillBlocked
    ws = _ws(tmp_path)
    reg, _ = _registry()
    reg.execute({"skill": "generate_character_master"}, ws)
    reg.execute({"skill": "generate_character_multiview"}, ws)
    try:
        reg.execute({"skill": "repair_character_view", "target": "front"},
                    ws, target="front")
        raise AssertionError("must be blocked")
    except SkillBlocked as blocked:
        assert "master_invalid" in str(blocked)


def test_repair_diagnostic_and_seed_varies(tmp_path: Path) -> None:
    """P1-2：失败原因注入 prompt；seed 随 attempt 递进。"""
    ws = _ws(tmp_path)
    reg, mv = _registry()
    reg.execute({"skill": "generate_character_master"}, ws)
    reg.execute({"skill": "generate_character_multiview"}, ws)
    failures = [{"view": "profile", "check": "view_angle",
                 "detail": "only 60 degrees, not strict 90"}]
    reg.execute({"skill": "repair_character_view", "target": "profile"}, ws,
                test_failures=failures, target="profile")
    assert mv.last_repair_kwargs["failures"] == failures
    assert mv.last_repair_kwargs["attempt"] == 1
    seed1 = mv.last_repair_kwargs["seed"]
    reg.execute({"skill": "repair_character_view", "target": "profile"}, ws,
                test_failures=failures, target="profile")
    seed2 = mv.last_repair_kwargs["seed"]
    assert seed2 != seed1  # 同失败重修不再抽同一 seed
    assert mv.last_repair_kwargs["attempt"] == 2


def test_budget_hard_gate_blocks_hallucinated_gpu(tmp_path: Path) -> None:
    """P0-4：cheap_text 预算下 Omni 幻觉 GPU skill → validate_action 拒。"""
    from src.agentic_video.skills.registry import SkillBlocked
    ws = _ws(tmp_path)
    reg, _ = _registry()
    allowed = {row["name"] for row in reg.runnable_skills(ws, "cheap_text")}
    assert "generate_character_master" not in allowed
    try:
        reg.validate_action(
            {"skill": "generate_character_master"}, ws,
            budget="cheap_text", allowed_skill_names=allowed)
        raise AssertionError("must be blocked")
    except SkillBlocked as blocked:
        assert "action_not_runnable" in str(blocked)


def test_gen_worker_edit_uses_image_list_not_image1(tmp_path: Path) -> None:
    """P0-1 契约回归：Edit 参考图必须是 image=[PIL]，绝无 image1。"""
    from src.agentic_video.asset_studio import gen_worker

    seen_kwargs: dict = {}

    class StrictFakePipe:
        def __call__(self, **kwargs):
            seen_kwargs.update(kwargs)
            assert "image1" not in kwargs, "worker used image1!"
            assert isinstance(kwargs.get("image"), list)
            assert isinstance(kwargs["image"][0], Image.Image)
            assert kwargs.get("negative_prompt"), "CFG needs negative"
            assert kwargs.get("true_cfg_scale", 0) > 1
            return SimpleNamespace(images=[_figure_image(
                (120, 80, 90), (1152, 2048))])

    ref = tmp_path / "master.png"
    _figure_image((140, 90, 70), (1152, 2048)).save(ref)
    out = tmp_path / "profile.png"
    task = {"prompt": "test", "reference_path": str(ref),
            "width": 1152, "height": 2048, "seed": 7,
            "out_path": str(out)}
    result = gen_worker._run(StrictFakePipe(), "edit", task)
    assert out.is_file()
    assert result["seed"] == 7
    assert "image1" not in seen_kwargs


def test_gen_worker_t2i_has_no_image(tmp_path: Path) -> None:
    from src.agentic_video.asset_studio import gen_worker

    class FakeT2IPipe:
        def __call__(self, **kwargs):
            assert "image" not in kwargs and "image1" not in kwargs
            assert kwargs.get("num_inference_steps") == 50
            return SimpleNamespace(images=[_figure_image(
                (140, 90, 70), (1152, 2048))])

    out = tmp_path / "master.png"
    gen_worker._run(FakeT2IPipe(), "t2i",
                    {"prompt": "x", "width": 1152, "height": 2048,
                     "seed": 1, "out_path": str(out)})
    assert out.is_file()


def test_recursive_dependency_invalidation_four_chain(
        tmp_path: Path) -> None:
    """P0-6：A→B→C→D 四层全 commit，A 变 → B/C/D 全 stale，goal 翻假。"""
    ws = Workspace(tmp_path / "dag")
    ws.state["goal"] = {"target_artifact": "D", "required_status": "committed"}
    for name, up in (("B", "A"), ("C", "B"), ("D", "C")):
        ws.set_dependency(name, up, "committed")
    for name in ("A", "B", "C", "D"):
        ws.write_draft(name, {"v": name})
        ws.commit(name)  # commit 自带快照
    assert ws.goal_satisfied()
    # 上游 A 变了
    ws.write_draft("A", {"v": "A2"})
    ws.commit("A")
    assert ws.get_status("B") == "stale"
    assert ws.get_status("C") == "stale"  # 递归（旧实现只标一层）
    assert ws.get_status("D") == "stale"
    assert not ws.goal_satisfied()
    # lazy 双保险：手动把 C 标回 committed（模拟 eager 漏标），
    # effective_status 仍应判 stale
    ws.state["artifacts"]["C"]["versions"][
        ws.state["artifacts"]["C"]["active_version"]]["status"] = \
        "committed"
    ws._save()
    assert ws.effective_status("C") == "stale"
    assert ws.effective_status("D") == "stale"


def test_identity_description_empty_does_not_raise() -> None:
    """P1-6：canonical/immutable 全空不 NameError。"""
    from src.agentic_video.asset_studio.skills import _identity_description
    assert _identity_description({}) == "character"
    assert _identity_description(
        {"asset_id": "C7"}) == "C7"


def test_asset_graph_completeness_contract() -> None:
    """P1-4：剧本 required 资产缺失/type 错 → FAIL（确定性）。"""
    from src.agentic_video.skills.asset_schema import validate_asset_graph
    screenplay = {"characters": [{"id": "C0"}],
                  "scenes": [{"location": "L01", "beats": [
                      {"beat_id": "B1",
                       "production_requirements": {
                           "characters": ["C0"], "location": "L01",
                           "props": ["P01"], "motion_refs": []}}]}]}
    graph = {"schema_version": "asset_graph_v1",
             "assets": [
                 {"asset_id": "C0", "type": "character", "tier": "A",
                  "immutable": {"face": "x"}, "status": "active"},
                 {"asset_id": "L01", "type": "prop", "tier": "A",
                  "status": "active"}]}  # L01 type 错 + P01 缺失
    failures = validate_asset_graph(graph, screenplay)
    assert any(f["check"] == "completeness" and f["asset_id"] == "P01"
               for f in failures)
    assert any(f["check"] == "type_mismatch" and f["asset_id"] == "L01"
               for f in failures)


def test_preflight_smoke_returns_manifest() -> None:
    """preflight 可运行并返回 (failures, manifest)——本机无 diffusers
    时 failures 非空但不崩（服务器上必须空）。"""
    from src.agentic_video.asset_studio.preflight import check_contracts
    failures, manifest = check_contracts()
    assert isinstance(failures, list)
    assert "python" in manifest
