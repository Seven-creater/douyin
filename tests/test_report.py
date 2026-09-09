"""report 单测：指标计算（校准目标形状）/ 清单生成 / markdown 渲染 / ingest 元数据。"""
from __future__ import annotations

import json

from src.library.report import build_checklist, recipe_metrics, render_markdown


def _recipe(ops, duration=25.0):
    return {"video": {"duration_s": duration}, "tempo_bpm_est": 112.4,
            "segments": [{"start": 0, "end": duration}],
            "operations": ops}


BEATS = [1.0, 1.5, 2.0, 2.5, 3.0]


def test_recipe_metrics_reproduces_gap_shape():
    """校准形状：参考多事件低对齐，重剪少事件高对齐（分析文档 35/16、85.7%/100%）。"""
    ref = _recipe([{"t_s": 1.01, "type": "hard_cut", "confidence": .8},
                   {"t_s": 2.0, "type": "luma_flash", "confidence": .7},
                   {"t_s": 2.6, "type": "tracked_mask_fill", "confidence": .6},
                   {"t_s": 4.4, "type": "montage_burst", "confidence": .8}])
    glm = _recipe([{"t_s": 1.0, "type": "hard_cut", "confidence": .8},
                   {"t_s": 2.5, "type": "hard_cut", "confidence": .8}])
    ma = recipe_metrics(ref, BEATS)
    mb = recipe_metrics(glm, BEATS)
    assert ma["n_ops"] == 4 and mb["n_ops"] == 2
    assert ma["beat_aligned_pct"] == 75.0            # 3/4 落拍（4.4 不落）
    assert mb["beat_aligned_pct"] == 100.0           # 重剪版全落拍
    assert ma["op_hist"]["hard_cut"] == 1 and ma["n_montage"] == 1
    assert ma["ops_per_min"] == round(4 * 60 / 25, 1)
    assert mb["tempo_bpm_est"] == 112.4 and mb["mean_seg_s"] == 25.0


def test_build_checklist_covers_claims_and_misses():
    ref = _recipe([{"t_s": 2.0, "type": "tracked_mask_fill", "confidence": .6}])
    glm = _recipe([])
    items = build_checklist({"ref": ref, "glm": glm})
    claimed = [i for i in items if i["video"] == "ref"]
    assert len(claimed) == 1 and claimed[0]["phenomenon"] == "tracked_mask_fill"
    absent = [i for i in items if "未在任一片中主张" in i["claim"]]
    assert len(absent) >= 10                          # 漏检对照项（除 hard_cut）
    frees = [i for i in items if "意外发现" in i["phenomenon"]]
    assert len(frees) == 2


def test_render_markdown_has_tables_and_calibration():
    ref = _recipe([{"t_s": 2.0, "type": "tracked_mask_fill", "confidence": .6,
                    "evidence": {"quote": "人像内部纹理变化"}}])
    glm = _recipe([{"t_s": 1.0, "type": "hard_cut", "confidence": .8}])
    ma, mb = recipe_metrics(ref, BEATS), recipe_metrics(glm, BEATS)
    text = render_markdown("ref", "glm", ma, mb, ref, glm,
                           build_checklist({"ref": ref, "glm": glm}))
    assert "85.7%" in text and "~35" in text          # 校准目标常驻
    assert "| n_ops | 1 | 1 |" in text
    assert "tracked_mask_fill" in text and "[ ]" in text


def test_ingest_writes_metadata(tmp_path):
    from src.config import AppConfig, PathsCfg
    from src.library.ingest import run_ingest

    cfg = AppConfig(
        wellbyte={}, ranking={}, download={}, perception={}, template={},
        generation={}, logging_level="INFO",
        paths=PathsCfg(raw_dir=tmp_path / "r", processed_dir=tmp_path / "p",
                       videos_dir=tmp_path / "v", logs_dir=tmp_path / "l",
                       perception_dir=tmp_path / "per",
                       generation_dir=tmp_path / "g", library_dir=tmp_path / "lib"))
    src = tmp_path / "卡点视频.mp4"
    src.write_bytes(b"0" * 1024)
    dst = run_ingest(cfg, src, "reedit_glm_001", kind="reedit", ref="7668161")
    assert dst.exists() and dst.name == "video.mp4"
    meta = json.loads((dst.parent / "metadata.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "reedit" and meta["references"] == ["7668161"]
    assert meta["aweme_id"] == "reedit_glm_001"
    # 幂等：二跑不炸不覆盖
    run_ingest(cfg, src, "reedit_glm_001", kind="reedit", ref="7668161")
    try:
        run_ingest(cfg, src, "reedit_glm_001", kind="bad")
        assert False, "kind 非法应抛错"
    except ValueError:
        pass
