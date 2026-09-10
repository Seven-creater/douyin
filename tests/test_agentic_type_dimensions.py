from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agentic_video.narrative_index import (FACET_PROMPT,
                                               parse_type_facets,
                                               run_type_facets)
from src.agentic_video.type_dimensions import (LIBRARY_FACET_DIMENSIONS,
                                               dimension_config,
                                               resolve_dimensions)


def test_resolve_dimensions_unions_type_defaults_and_template_needs():
    """content_type 只给默认维度；模板需求 ∪ 进来且去重、默认在前。"""
    dims = resolve_dimensions("growth_story", ["location", "被低估的能力", "location"])
    assert dims[:3] == ("age_appearance", "time_span", "setback")
    assert "location" in dims and dims.count("location") == 1
    assert "被低估的能力" in dims                       # 模板自定义维度透传
    assert resolve_dimensions("unknown-type") == ("location", "relationship")


def test_dimension_config_overrides_from_config():
    cfg = SimpleNamespace(library={"type_dimensions": {
        "real_story": ["location", "custom_dim"]}})
    sets = dimension_config(cfg)
    assert sets["real_story"] == ("location", "custom_dim")
    assert sets["growth_story"][0] == "age_appearance"   # 未覆盖的保持默认


def test_facet_prompt_binds_entity_and_intervals():
    assert "apply_interval" in FACET_PROMPT and "evidence_interval" in FACET_PROMPT
    assert "每条 facet 必须绑定到具体实体" in FACET_PROMPT
    assert "不确定的维度整条不输出" in FACET_PROMPT
    assert "interaction" in FACET_PROMPT


def _facet_raw(**overrides) -> str:
    facet = {"dimension": "age_appearance", "entity_id": "e0", "value": "少年",
             "apply_interval": [1.0, 8.0], "evidence_interval": [1.0, 3.0],
             "evidence_source": "frame", "confidence": 0.9}
    facet.update(overrides)
    return json.dumps({"facets": [facet]}, ensure_ascii=False)


def test_parse_type_facets_normalizes_clip_time_to_movie_axis():
    parsed = parse_type_facets(_facet_raw(), window_start=5000.0, window_end=5045.0,
                               known_entities={"e0"})
    facet = parsed["facets"][0]
    assert facet["apply_interval"] == [5001.0, 5008.0]    # 切片坐标归一
    assert facet["evidence_interval"] == [5001.0, 5003.0]
    assert facet["status"] == "supported" and facet["registry_known"] is True


def test_parse_type_facets_drops_junk_and_marks_unknown_entities():
    raw = json.dumps({"facets": [
        {"dimension": "rescue_scene", "value": "救助", "apply_interval": [0, 5]},   # 非库维度
        {"dimension": "age_appearance", "entity_id": "eX", "value": "青年",
         "apply_interval": [0, 5], "evidence_source": "frame", "confidence": 0.9},  # 注册表外
        {"dimension": "age_appearance", "value": "uncertain",
         "apply_interval": [0, 5]},                                                # uncertain 值
        {"dimension": "interaction", "a_entity_id": "e0", "b_entity_id": "e0",
         "relation": "保护", "apply_interval": [0, 5]},                             # 自指交互
        {"dimension": "interaction", "a_entity_id": "e0", "b_entity_id": "e1",
         "relation": "保护", "apply_interval": [0, 5], "evidence_source": "frame",
         "confidence": 0.8},
    ]}, ensure_ascii=False)
    parsed = parse_type_facets(raw, window_start=0.0, window_end=45.0,
                               known_entities={"e0", "e1"})
    facets = parsed["facets"]
    assert [f["dimension"] for f in facets] == ["age_appearance", "interaction"]
    assert facets[0]["registry_known"] is False        # 新实体保留但标记
    assert facets[0]["status"] == "supported"
    assert facets[1]["relation"] == "保护" and facets[1]["registry_known"] is True
    # 无 evidence_source 的低置信条目是 uncertain，不冒充已确认
    low = parse_type_facets(
        _facet_raw(evidence_source="", confidence=0.3), window_start=0.0,
        window_end=45.0, known_entities={"e0"})["facets"][0]
    assert low["status"] == "uncertain"


class _FacetRunner:
    def __init__(self):
        self.watches = 0

    def watch(self, video, prompt, **kwargs):
        self.watches += 1
        return SimpleNamespace(text=_facet_raw(
            apply_interval=[0.5, 6.0], evidence_interval=[0.5, 2.0]))


def _facet_cfg(tmp_path: Path, *, with_state: bool = False):
    shots_dir = tmp_path / "library" / "shots" / "src__narrative"
    shots_dir.mkdir(parents=True)
    (shots_dir / "result.json").write_text(json.dumps({
        "output": {"video": str(tmp_path / "movie.mp4"), "shots": [
            {"shot_idx": 0, "window_idx": 0, "start_s": 10.0, "end_s": 14.0},
            {"shot_idx": 1, "window_idx": 0, "start_s": 14.0, "end_s": 18.0},
            {"shot_idx": 2, "window_idx": 1, "start_s": 100.0, "end_s": 106.0},
        ]}}), encoding="utf-8")
    (shots_dir / "narrative_annotations.json").write_text(json.dumps({
        "shots": {"0": {"entity_ids": ["e0"], "entity_names": ["少年"]}}}),
        encoding="utf-8")
    (tmp_path / "movie.mp4").write_bytes(b"fake-movie")
    return SimpleNamespace(paths=SimpleNamespace(library_dir=tmp_path / "library"),
                           perception={})


def test_run_type_facets_pilot_windows_and_resume(tmp_path):
    cfg = _facet_cfg(tmp_path)
    runner = _FacetRunner()
    path = run_type_facets(cfg, "src", windows=[0, 1], runner=runner)
    assert runner.watches == 2                                   # 试点只跑指定窗
    state = json.loads(path.read_text(encoding="utf-8"))
    assert set(state["completed"]) == {"0", "1"}
    assert state["windows"]["0"]["facets"][0]["dimension"] == "age_appearance"

    run_type_facets(cfg, "src", windows=[0, 1], runner=runner)    # 键全同 → 全命中
    assert runner.watches == 2

    import src.agentic_video.narrative_index as ni
    original = ni.FACET_PROMPT
    ni.FACET_PROMPT = original + "\n【v2 试炼】"
    try:
        run_type_facets(cfg, "src", windows=[0], runner=runner)   # 提示词变 → 窗 0 重提
    finally:
        ni.FACET_PROMPT = original
    assert runner.watches == 3                                   # 只有窗 0 重跑


def test_library_facet_dimensions_cover_union_needs():
    assert set(LIBRARY_FACET_DIMENSIONS) == {"age_appearance", "location", "era",
                                             "interaction"}
