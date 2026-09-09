from __future__ import annotations

import json
from types import SimpleNamespace

from src.agentic_video.narrative_agent import build_narrative_material
from src.agentic_video.narrative import (new_narrative_program,
                                          validate_narrative_program)


def _evidence(t: float = 0.5) -> list[dict]:
    return [{"source": "frame", "interval": [t, t + 0.1],
             "frame": 12, "confidence": 0.9}]


def valid_program() -> dict:
    program = new_narrative_program(
        reference_id="ref", reference_uri="ref.mp4", sha256="a" * 64,
        duration_s=12.0, fps=24.0)
    program["intent"] = {
        "topic": "救助", "message": "善意带来改变",
        "content_type": "real_story", "evidence": _evidence(),
        "confidence": 0.85, "status": "supported",
    }
    program["entities"] = [
        {"id": "person", "kind": "person", "name_or_role": "救助者",
         "aliases": [], "evidence": _evidence(), "confidence": 0.9,
         "status": "supported"},
        {"id": "cat", "kind": "animal", "name_or_role": "小猫",
         "aliases": [], "evidence": _evidence(1.0), "confidence": 0.9,
         "status": "supported"},
    ]
    program["events"] = [
        {"id": "e1", "interval": [0.0, 4.0], "participants": ["person", "cat"],
         "action": "发现被困小猫", "state_before": "小猫被困",
         "state_after": "开始救助", "evidence": _evidence(),
         "confidence": 0.9, "status": "supported"},
        {"id": "e2", "interval": [4.0, 9.0], "participants": ["person", "cat"],
         "action": "救出小猫", "state_before": "正在救助",
         "state_after": "小猫脱险", "evidence": _evidence(5.0),
         "confidence": 0.9, "status": "supported"},
        {"id": "e3", "interval": [9.0, 12.0], "participants": ["person", "cat"],
         "action": "小猫恢复安全", "state_before": "刚脱险",
         "state_after": "安全", "evidence": _evidence(10.0),
         "confidence": 0.9, "status": "supported"},
    ]
    program["causal_links"] = [
        {"from_event": "e1", "to_event": "e2", "relation": "motivates",
         "evidence": _evidence(3.5), "confidence": 0.8, "status": "supported"},
        {"from_event": "e2", "to_event": "e3", "relation": "causes",
         "evidence": _evidence(8.5), "confidence": 0.8, "status": "supported"},
    ]
    program["arc"] = [
        {"role": "hook", "event_ids": ["e1"]},
        {"role": "conflict", "event_ids": ["e1"]},
        {"role": "choice", "event_ids": ["e2"]},
        {"role": "climax", "event_ids": ["e2"]},
        {"role": "consequence", "event_ids": ["e3"]},
        {"role": "resolution", "event_ids": ["e3"]},
    ]
    program["emotion_curve"] = [
        {"interval": [0.0, 4.0], "emotion": "担忧", "intensity": 0.7,
         "evidence": _evidence(), "confidence": 0.8, "status": "supported"},
        {"interval": [9.0, 12.0], "emotion": "安心", "intensity": 0.8,
         "evidence": _evidence(10.0), "confidence": 0.8, "status": "supported"},
    ]
    return program


def test_valid_narrative_program():
    program = valid_program()
    assert program["status"] == "uncertain"
    assert program["evidence"] == []
    assert validate_narrative_program(program) == []


def test_supported_program_requires_top_level_evidence():
    program = valid_program()
    program["status"] = "supported"
    assert any("narrative program requires evidence" in error
               for error in validate_narrative_program(program))


def test_supported_claims_require_evidence_and_known_entities():
    program = valid_program()
    program["events"][0]["evidence"] = []
    program["events"][1]["participants"] = ["missing"]
    errors = validate_narrative_program(program)
    assert any("requires evidence" in error for error in errors)
    assert any("unknown entity" in error for error in errors)


def test_causal_graph_must_be_acyclic_and_forward():
    program = valid_program()
    program["causal_links"].append({
        "from_event": "e3", "to_event": "e1", "relation": "causes",
        "evidence": _evidence(), "confidence": 0.8, "status": "supported"})
    errors = validate_narrative_program(program)
    assert any("cycle" in error for error in errors)
    assert any("backward" in error for error in errors)


def test_utterance_cannot_cross_reference_duration():
    program = valid_program()
    program["utterances"] = [{
        "id": "u1", "interval": [11.0, 13.0], "speaker_id": "person",
        "original": "助ける", "translation_zh": "我要救它",
        "evidence": _evidence(11.0), "confidence": 0.8, "status": "supported"}]
    assert any("outside reference" in e for e in validate_narrative_program(program))


def test_narrative_material_reads_curated_context_as_unverified_hypothesis(tmp_path):
    videos = tmp_path / "videos"
    perception = tmp_path / "perception"
    reference = videos / "ref"
    reference.mkdir(parents=True)
    (reference / "context.json").write_text(json.dumps({
        "title": "父亲帮女儿练习用脚写字",
        "author": "良田",
        "hashtags": ["父亲", "自信"],
    }, ensure_ascii=False), encoding="utf-8")
    cfg = SimpleNamespace(paths=SimpleNamespace(
        videos_dir=videos, perception_dir=perception))

    material = build_narrative_material(cfg, "ref")

    assert "仅作待验证假设，不是证据" in material
    assert "父亲帮女儿练习用脚写字" in material
    assert "外部作者：良田" in material
    assert "外部话题：父亲,自信" in material
