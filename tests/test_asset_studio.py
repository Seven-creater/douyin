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
                    identity_description, seed):
        self.edit_calls.append(f"repair:{view}")
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
    ws.state["goal"] = {"target_artifact": "asset:C0",
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
    ws.set_dependency("asset:C0_master", "asset_graph", "committed")
    ws.set_dependency("asset:C0_views", "asset:C0_master", "committed")
    ws.set_dependency("asset:C0", "asset:C0_views", "committed")
    return ws


def _registry() -> tuple:
    mv = FakeMultiView()
    reg = build_m2a_registry(t2i=FakeT2I(), multiview=mv,
                             upscale=FakeUpscale())
    return reg, mv


# ---- 全链（agent_loop 集成） ----

def test_full_chain_master_to_commit(tmp_path: Path) -> None:
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
            '{"passed": true, "failures": []}',
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
    # 四张单图 + sheet 落盘且 4K
    files = ws.root / "03_asset_studio" / "files"
    for view in ("front", "profile", "back", "face"):
        matches = sorted(files.glob(f"C0_{view}*"))
        assert matches, f"missing {view} files"
    sheet = json.loads(
        (ws.root / "03_asset_studio" / "C0" / "v1.json").read_text(
            encoding="utf-8"))
    for view, entry in sheet["views_4k"].items():
        img = Image.open(ws.root / entry["file"])
        assert max(img.size) >= 3840
    assert (ws.root / sheet["sheet"]["file"]).is_file()


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
