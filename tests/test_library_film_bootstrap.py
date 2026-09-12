"""P1.5 Film Knowledge Bootstrap 单测：Identity Gate / 两层 Identity / 状态机。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.library.film_bootstrap import (annotation_view, merge_pack_registry,
                                        parse_identity_answer,
                                        parse_roster_answer, pick_representative_timestamps,
                                        resolve_tier, sanitize_canonical_id,
                                        update_pack_from_annotations)


def _vote(title="", confidence=0.0, recognized=None):
    return {"recognized": recognized if recognized is not None else bool(title),
            "title": title, "installment": "", "confidence": confidence,
            "basis": "", "franchise_slug": ""}


def test_identity_gate_requires_cross_view_agreement():
    """外审二轮：自报 confidence 不可信——A/B 双投一致才进 prior。"""
    agree_high = resolve_tier(_vote("罗小黑战记2", 0.92), _vote("罗小黑战记2", 0.88))
    assert agree_high["tier"] == "model_prior"
    # 单侧高置信但两投不一致 → 搜证，不得直接 prior
    disagree = resolve_tier(_vote("罗小黑战记2", 0.97), _vote("雾山五行", 0.6))
    assert disagree["tier"] == "search"
    assert disagree["votes_agree"] is False
    # 元数据佐证任一投票 → prior（产品模式：文件名是便宜合法信号）
    meta = resolve_tier(_vote("罗小黑战记2", 0.7), _vote("", 0.1),
                        "luoxiaohei_movie2_1080p.mkv".replace("luoxiaohei_movie2", "罗小黑战记2"))
    assert meta["tier"] == "model_prior" and meta["metadata_match"]


def test_identity_gate_metadata_conflict_blocks_injection():
    """模型认 A + 元数据说 B → identity_conflict：错 title 会诱导幻觉，不注入。"""
    conflict = resolve_tier(_vote("雾山五行", 0.9), _vote("雾山五行", 0.85),
                            "罗小黑战记2.mkv")
    assert conflict["tier"] == "identity_conflict"
    low = resolve_tier(_vote("", 0.2), _vote("", 0.1))
    assert low["tier"] == "unknown"


def test_metadata_prior_tier_when_vision_refuses_but_filename_names_film():
    """四轮冒烟实锤：Qwen3-Omni 对罗小黑视觉拒认（角色帧也诚实说不知道）。
    视觉全盲 + 文件名含明确作品名 → metadata_prior（roster 照生成、全
    proposed；绑定层要求画面吻合才生效）；mask 元数据后（benchmark 模式）
    同样输入回到 unknown——泄漏源可控。"""
    from src.library.film_bootstrap import title_from_metadata
    filename = "罗小黑1 2019.4K.H265.60fps.10bit.Dolby.5.1.chs.mkv"
    assert title_from_metadata(filename) == "罗小黑1"
    blind = resolve_tier(_vote("", 0.0), _vote("", 0.0), filename)
    assert blind["tier"] == "metadata_prior"
    assert blind["metadata_title_extracted"] == "罗小黑1"
    masked = resolve_tier(_vote("", 0.0), _vote("", 0.0), "")
    assert masked["tier"] == "unknown"
    # 元数据只是乱码技术噪声（无 CJK）→ 仍 unknown
    assert resolve_tier(_vote("", 0.0), _vote("", 0.0), "2160p.WEB-DL.HQ.mkv")["tier"] \
        == "unknown"


def test_sanitize_canonical_id_enforces_franchise_namespace():
    """char:<franchise>:<slug> 三段式（防多素材库同名碰撞）；单段自动补 franchise。"""
    assert sanitize_canonical_id("char:luoxiaohei:wuxian", "luoxiaohei") \
        == "char:luoxiaohei:wuxian"
    assert sanitize_canonical_id("char:Wuxian", "luoxiaohei") \
        == "char:luoxiaohei:wuxian"
    assert sanitize_canonical_id("垃圾!!!", "luoxiaohei") is None
    assert sanitize_canonical_id("char:x", "") is None            # 无 franchise 不给


def _pack(tier="model_prior"):
    return {
        "schema": "knowledge_pack_v1",
        "work_identity": {"tier": tier, "title": "罗小黑战记2", "status": "proposed"},
        "franchise_slug": "luoxiaohei",
        "entities": [
            {"canonical_id": "char:luoxiaohei:xiaohei", "name": "罗小黑",
             "aliases": ["小黑"], "type": "creature", "appearance": "黑猫/黑发少年",
             "source": "model_prior", "status": "proposed"},
            {"canonical_id": "char:luoxiaohei:wuxian", "name": "无限",
             "aliases": [], "type": "human", "appearance": "黑发执行人",
             "source": "model_prior", "status": "proposed"},
        ],
        "relations": [{"a": "char:luoxiaohei:xiaohei", "b": "char:luoxiaohei:wuxian",
                       "relation": "同行", "source": "model_prior", "status": "proposed"}],
        "plot_hypotheses": [{"segment": "A", "summary": "小黑流浪",
                             "source": "model_prior", "status": "proposed"}],
    }


def test_annotation_view_is_recognition_prior_only():
    """红线：标注器只拿 roster/aliases/appearance——关系与剧情绝不进。"""
    view = annotation_view(_pack())
    assert [row["canonical_id"] for row in view] == \
        ["char:luoxiaohei:xiaohei", "char:luoxiaohei:wuxian"]
    dumped = json.dumps(view, ensure_ascii=False)
    assert "同行" not in dumped and "流浪" not in dumped
    # tier 未达 model_prior（含 unverified 降级）→ 空：错 title 会诱导幻觉
    assert annotation_view(_pack(tier="model_prior_unverified")) == []
    assert annotation_view(_pack(tier="search")) == []
    assert annotation_view(None) == []


def test_merge_pack_registry_puts_candidates_first():
    merged = json.loads(merge_pack_registry(
        {"0": {"entity_ids": ["vis_1"], "entity_names": ["黑发少年"]}},
        annotation_view(_pack())))
    ids = [row["entity_id"] for row in merged]
    assert ids[:2] == ["char:luoxiaohei:xiaohei", "char:luoxiaohei:wuxian"]
    assert "vis_1" in ids                                  # 窗口累积表排后不丢


def test_parse_bindings_whitelist_and_two_layer_identity():
    from src.agentic_video.narrative_index import parse_window_annotations
    raw = json.dumps({
        "shots": [{"shot_idx": 3, "entity_ids": ["vis_021"], "entity_names": ["长发男性"],
                   "event_id": "e1", "bindings": [
                       {"local_entity_id": "vis_021",
                        "canonical_entity_id": "char:luoxiaohei:wuxian",
                        "binding_status": "supported", "binding_confidence": 0.86,
                        "binding_evidence": ["长发", "黑衣", "持刀"]},
                       {"local_entity_id": "vis_021",
                        "canonical_entity_id": "char:madeup:someone",
                        "binding_status": "supported"}]}],
        "causal_links": []}, ensure_ascii=False)
    parsed = parse_window_annotations(
        raw, valid_shot_ids={3}, valid_canonicals={"char:luoxiaohei:wuxian"})
    row = parsed["shots"]["3"]
    assert row["entity_ids"] == ["vis_021"]                 # 本地视觉身份永远保留
    assert len(row["bindings"]) == 1                        # 白名单外的自造 canonical 被丢
    assert row["bindings"][0]["canonical_entity_id"] == "char:luoxiaohei:wuxian"


def test_row_identity_keys_consume_supported_bindings():
    """两层 Identity 消费：supported 绑定直接给 canonical 键（跨窗同人的正路）；
    conflict 绑定不贡献（先验与视觉矛盾以视觉为准）。"""
    from src.library.entity_registry import row_identity_keys
    row = {"video_stem": "luoxiaohei2__narrative", "window_idx": 7,
           "entity_ids": ["vis_021"], "entity_names": ["长发男性"],
           "bindings": [
               {"local_entity_id": "vis_021",
                "canonical_entity_id": "char:luoxiaohei:wuxian",
                "binding_status": "supported", "binding_confidence": 0.86}]}
    keys = row_identity_keys(row, {})
    assert "char:luoxiaohei:wuxian" in keys
    assert "id:luoxiaohei2/w7/vis_021" in keys              # 窗口作用域本地键仍在
    conflict_row = {**row, "bindings": [
        {"local_entity_id": "vis_021", "canonical_entity_id": "char:luoxiaohei:wuxian",
         "binding_status": "conflict"}]}
    assert "char:luoxiaohei:wuxian" not in row_identity_keys(conflict_row, {})


def test_binding_state_machine_and_auto_registry_backfill(tmp_path):
    """supported→verified 只认独立 GT（防循环论证）；conflict 如实降级；
    source_entities 回填可被 build_alias_maps 反解消费。"""
    from src.library.entity_registry import build_alias_maps

    cfg = SimpleNamespace(library={},
                          paths=SimpleNamespace(library_dir=tmp_path))
    shots_dir = tmp_path / "shots" / "luoxiaohei2__narrative"
    shots_dir.mkdir(parents=True)
    pack = _pack()
    (shots_dir / "knowledge_pack.json").write_text(
        json.dumps(pack, ensure_ascii=False), encoding="utf-8")
    annotations = {"shots": {
        "3": {"window_idx": 3, "bindings": [
            {"local_entity_id": "vis_021", "canonical_entity_id": "char:luoxiaohei:wuxian",
             "binding_status": "supported", "binding_confidence": 0.9}]},
        "9": {"window_idx": 9, "bindings": [
            {"local_entity_id": "vis_007", "canonical_entity_id": "char:luoxiaohei:wuxian",
             "binding_status": "supported", "binding_confidence": 0.8}]},
        "11": {"window_idx": 11, "bindings": [
            {"local_entity_id": "vis_030", "canonical_entity_id": "char:luoxiaohei:xiaohei",
             "binding_status": "conflict"}]},
    }}
    (shots_dir / "narrative_annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False), encoding="utf-8")
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"verified_entities": ["char:luoxiaohei:wuxian"]}),
                  encoding="utf-8")

    report = update_pack_from_annotations(cfg, "luoxiaohei2", gt_path=gt)
    assert report["entities"]["char:luoxiaohei:wuxian"]["status"] == "verified"
    assert report["entities"]["char:luoxiaohei:xiaohei"]["status"] == "conflict"
    updated = json.loads((shots_dir / "knowledge_pack.json").read_text(encoding="utf-8"))
    assert updated["entities"][1]["binding_windows"] == [3, 9]

    auto = json.loads((tmp_path / "entity_registry.auto.json").read_text(encoding="utf-8"))
    refs = auto["char:luoxiaohei:wuxian"]["source_entities"]
    aliases, source_map, windowed = build_alias_maps(auto)
    assert windowed[("luoxiaohei2", "w3", "vis_021")] == "char:luoxiaohei:wuxian"
    assert windowed[("luoxiaohei2", "w9", "vis_007")] == "char:luoxiaohei:wuxian"
    assert set(refs) == {"luoxiaohei2/w3/vis_021", "luoxiaohei2/w9/vis_007"}


def test_auto_registry_merges_without_overwriting_handwritten(tmp_path):
    from src.library.entity_registry import load_entity_registry
    from src.config import AppConfig, PathsCfg

    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, template={}, generation={},
        logging_level="INFO", perception={},
        library={"entities": {"registry_path": str(tmp_path / "hand.json")}},
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "per", generation_dir=tmp_path / "g",
                       library_dir=tmp_path / "lib"))
    (tmp_path / "hand.json").write_text(json.dumps(
        {"char:xiaohei": {"aliases": ["小黑", "黑猫"], "source_entities": []}}),
        encoding="utf-8")
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib" / "entity_registry.auto.json").write_text(json.dumps(
        {"char:xiaohei": {"aliases": ["自动别名"], "source_entities": ["luoxiaohei1/e005"]},
         "char:new": {"aliases": ["新角色"], "source_entities": []}}), encoding="utf-8")
    merged = load_entity_registry(cfg)
    assert merged["char:xiaohei"]["aliases"] == ["小黑", "黑猫"]   # 手写别名优先
    assert merged["char:xiaohei"]["source_entities"] == ["luoxiaohei1/e005"]
    assert merged["char:new"]["aliases"] == ["新角色"]


def test_parse_roster_and_timestamps():
    stamps = pick_representative_timestamps(600.0, n_uniform=6, head_s=90, tail_s=60)
    assert len(stamps) == 6 and all(90 < t < 540 for t in stamps)
    roster = parse_roster_answer(json.dumps({
        "franchise_slug": "luoxiaohei",
        "entities": [
            {"canonical_id": "char:luoxiaohei:xiaohei", "name": "罗小黑",
             "aliases": ["小黑"], "type": "creature", "appearance": "黑猫"},
            {"canonical_id": "char:wuxian", "name": "无限", "appearance": "执行人"}],
        "relations": [{"a": "char:luoxiaohei:xiaohei", "b": "char:luoxiaohei:wuxian",
                       "relation": "同行"}],
        "plot_arcs": [{"segment": "A", "summary": "流浪"}]}, ensure_ascii=False),
        franchise="luoxiaohei", max_entities=12)
    ids = [e["canonical_id"] for e in roster["entities"]]
    assert ids == ["char:luoxiaohei:xiaohei", "char:luoxiaohei:wuxian"]
    assert roster["entities"][0]["status"] == "proposed"     # 全部待验证
    assert roster["relations"] and roster["plot_hypotheses"]  # 归档但不进标注


def test_parse_identity_answer_tolerates_garbage():
    assert parse_identity_answer("not json")["recognized"] is False
    parsed = parse_identity_answer(json.dumps(
        {"recognized": True, "title": "罗小黑战记2", "confidence": "1.7",
         "basis": "画风"}))
    assert parsed["confidence"] == 1.0                        # 夹到 [0,1]
