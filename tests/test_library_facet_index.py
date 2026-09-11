from __future__ import annotations

import json
from types import SimpleNamespace

from src.library.build_index import collect_rows


def _env(tmp_path, shots):
    shots_dir = tmp_path / "library" / "shots" / "src__narrative"
    shots_dir.mkdir(parents=True)
    (shots_dir / "result.json").write_text(
        json.dumps({"output": {"shots": shots}}, ensure_ascii=False), encoding="utf-8")
    return shots_dir


def test_facets_bind_to_shot_intervals_not_whole_window(tmp_path):
    """P1 红线：facet 按 apply_interval 实质重叠落镜头，窗口印象不复制给整窗；
    uncertain facet 不进索引；emotion 不再只标注不消费。"""
    shots_dir = _env(tmp_path, [
        {"shot_idx": 0, "window_idx": 0, "start_s": 10.0, "end_s": 14.0, "video": "m.mp4",
         "video_stem": "src__narrative", "duration_s": 4.0, "source": "src",
         "action_score": 0.5, "dialogue": []},
        {"shot_idx": 1, "window_idx": 0, "start_s": 30.0, "end_s": 34.0, "video": "m.mp4",
         "video_stem": "src__narrative", "duration_s": 4.0, "source": "src",
         "action_score": 0.5, "dialogue": []},
    ])
    (shots_dir / "captions.json").write_text(
        json.dumps({"0": "雪地战斗", "1": "室内对谈"}), encoding="utf-8")
    (shots_dir / "narrative_annotations.json").write_text(json.dumps({"shots": {
        "0": {"entity_ids": ["e0", "e1"], "entity_names": ["少年", "少女"],
              "event_id": "w000_e0", "event_summary": "雪中护送", "story_role": "conflict",
              "emotion": "紧张", "focus_x": 0.5, "dialogue": [], "annotation_confidence": 0.9},
        "1": {"entity_ids": ["e2"], "entity_names": ["长者"], "event_id": "w000_e1",
              "event_summary": "屋内谈话", "story_role": "context", "emotion": "uncertain",
              "focus_x": 0.5, "dialogue": [], "annotation_confidence": 0.8},
    }}), encoding="utf-8")
    # facet 适用区间只覆盖 10-13：室内镜头（30-34）不得继承
    (shots_dir / "type_facets.json").write_text(json.dumps({"windows": {"0": {"facets": [
        {"dimension": "location", "value": "雪野", "apply_interval": [10.0, 13.0],
         "evidence_interval": [10.0, 12.0], "evidence_source": "frame",
         "confidence": 0.9, "status": "supported"},
        {"dimension": "interaction", "a_entity_id": "e0", "b_entity_id": "e1",
         "relation": "保护", "apply_interval": [10.5, 12.0],
         "evidence_interval": [11.0, 12.0], "evidence_source": "frame",
         "confidence": 0.8, "status": "supported"},
        {"dimension": "era", "value": "大正", "apply_interval": [10.0, 34.0],
         "evidence_interval": [10.0, 11.0], "evidence_source": "frame",
         "confidence": 0.9, "status": "supported"},
        {"dimension": "location", "value": "低置信地点", "apply_interval": [10.0, 13.0],
         "evidence_interval": [10.0, 11.0], "evidence_source": "", "confidence": 0.3,
         "status": "uncertain"},
    ]}}, "completed": {"0": "k"}}, ensure_ascii=False), encoding="utf-8")

    rows = collect_rows(SimpleNamespace(
        paths=SimpleNamespace(library_dir=tmp_path / "library")))
    snow, indoor = rows[0], rows[1]
    assert snow["facets"]["locations"] == ["雪野"]
    assert snow["facets"]["interactions"][0]["relation"] == "保护"
    assert snow["facets"]["era"] == "大正"
    assert "地点：雪野" in snow["search_text"]
    assert "关系：保护" in snow["search_text"]
    assert "时代：大正" in snow["search_text"]
    assert "情绪：紧张" in snow["search_text"]          # emotion 进检索文本
    assert "低置信地点" not in snow["search_text"]       # uncertain 不进
    assert indoor["facets"]["locations"] == []          # 窗口印象不复制给整窗
    assert "地点" not in indoor["search_text"]
    assert "情绪" not in indoor["search_text"]          # uncertain emotion 不入


def test_index_unchanged_without_facets_file(tmp_path):
    """加性约束：没有 type_facets.json 时行结构零变化（不新增 facets 键）；
    老格式镜头行（预告片时代，无 window_idx 键）不炸（2026-09-11 夜间实锤）。"""
    shots_dir = _env(tmp_path, [
        {"shot_idx": 0, "window_idx": 0, "start_s": 0.0, "end_s": 4.0, "video": "m.mp4",
         "video_stem": "src__narrative", "duration_s": 4.0, "source": "src",
         "action_score": 0.5, "dialogue": []},
        {"shot_idx": 1, "start_s": 10.0, "end_s": 12.0, "video": "old.mp4",
         "video_stem": "trailer", "duration_s": 2.0, "source": "trailer",
         "action_score": 0.4, "dialogue": []},          # 无 window_idx 的老行
    ])
    (shots_dir / "captions.json").write_text(
        json.dumps({"0": "训练", "1": "旧预告"}), encoding="utf-8")
    rows = collect_rows(SimpleNamespace(
        paths=SimpleNamespace(library_dir=tmp_path / "library")))
    assert len(rows) == 2 and "facets" not in rows[0]
    assert rows[0]["search_text"].startswith("训练")
