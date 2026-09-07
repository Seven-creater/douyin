"""template schema/prompts 单测（纯函数，~20 用例）。"""
from __future__ import annotations

import json

from src.template.prompts import build_repair_prompt, build_template_prompt, dedupe_json_repetition
from src.template.schema import (
    extract_json_block,
    normalize_template,
    parse_and_validate,
    parse_template_lenient,
    validate_template,
)

DUR = 22.0
BOUNDS = [0.0, 3.2, 7.8, 12.1, 22.0]
BEATS = [0.5, 1.0, 2.1, 3.6, 5.0]


def valid_template() -> dict:
    return {
        "trend_summary": "水上飞人反差",
        "core_meme": "静止突然升空",
        "timeline": [
            {"start": 0.0, "end": 3.2, "role": "setup", "visual": "男子踩板", "speech": None, "text": None},
            {"start": 3.2, "end": 12.1, "role": "climax", "visual": "飞板升空", "speech": None, "text": "水上飞人"},
            {"start": 12.1, "end": 22.0, "role": "ending", "visual": "落水收尾", "speech": None, "text": None},
        ],
        "audio": {"bgm": "电子乐", "beat_points": [0.5, 2.1], "speech": []},
        "fixed_elements": ["静止→升空反转"], "replaceable_elements": ["主角人物"],
        "generation_plan": ["生成踩板起步镜头"],
    }


# ---------- extract_json_block ----------

def test_extract_plain_json():
    assert json.loads(extract_json_block('{"a": 1}')) == {"a": 1}


def test_extract_with_fence_and_think():
    text = '<think>推理</think>```json\n{"a": {"b": "含{花括号}的中文"}}\n```'
    assert json.loads(extract_json_block(text))["a"]["b"] == "含{花括号}的中文"


def test_extract_with_trailing_text():
    out = extract_json_block('好的，结果如下：{"a": 1} 以上就是分析。')
    assert out == '{"a": 1}'


def test_extract_none_when_no_json():
    assert extract_json_block("没有任何 JSON") is None


def test_extract_nested_braces_in_strings():
    assert json.loads(extract_json_block('{"s": "a{b}c}"}'))["s"] == "a{b}c}"


# ---------- validate ----------

def test_validate_ok():
    res = validate_template(valid_template(), duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert res.ok, res.errors
    assert res.template["timeline"][0]["start"] == 0.0


def test_validate_missing_key_rejected():
    obj = valid_template(); del obj["core_meme"]
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert not res.ok and any("core_meme" in e for e in res.errors)


def test_validate_time_out_of_range_rejected():
    obj = valid_template(); obj["timeline"][1]["end"] = 99.0
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert not res.ok and any("end" in e for e in res.errors)


def test_validate_overlap_rejected():
    obj = valid_template(); obj["timeline"][1]["start"] = 2.0  # 与前段重叠
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert not res.ok and any("重叠" in e for e in res.errors)


def test_validate_bad_role_rejected():
    obj = valid_template(); obj["timeline"][0]["role"] = "开场"
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert not res.ok and any("role" in e for e in res.errors)


def test_validate_fabricated_beat_rejected():
    obj = valid_template(); obj["audio"]["beat_points"] = [7.77]
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert not res.ok and any("自造" in e for e in res.errors)


def test_validate_beats_without_known_rejected():
    obj = valid_template()  # beat_points 非空但 known 为空
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=[])
    assert not res.ok and any("未提供节拍" in e for e in res.errors)


def test_validate_snap_warning_is_soft():
    obj = valid_template(); obj["timeline"][1]["start"] = 5.0  # 离边界 7.8 差 2.8s
    res = validate_template(obj, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert res.ok  # 只警告不拒
    assert any("镜头边界" in w for w in res.warnings)


def test_normalize_strips_and_rounds():
    obj = valid_template()
    obj["trend_summary"] = "  padded  "
    obj["timeline"][0]["visual"] = " v "
    n = normalize_template(obj)
    assert n["trend_summary"] == "padded"
    assert n["timeline"][0]["visual"] == "v"
    assert n["audio"]["beat_points"] == [0.5, 2.1]


# ---------- lenient / parse_and_validate ----------

def test_lenient_partial_two_keys():
    text = '前言。 "trend_summary": "好的", 一些噪声 "core_meme": "梗", 尾巴'
    obj, mode = parse_template_lenient(text)
    assert mode == "partial"
    assert obj["trend_summary"] == "好的" and obj["core_meme"] == "梗"


def test_lenient_fail_when_nothing():
    assert parse_template_lenient("完全不是 JSON")[0] is None


def test_parse_and_validate_json_mode():
    text = "说明文字\n" + json.dumps(valid_template(), ensure_ascii=False)
    res, mode = parse_and_validate(text, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert mode == "json" and res.ok


def test_parse_and_validate_fence_mode():
    text = "```json\n" + json.dumps(valid_template(), ensure_ascii=False) + "\n```"
    res, mode = parse_and_validate(text, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert mode == "json" and res.ok


def test_parse_and_validate_invalid_returns_json_invalid():
    obj = valid_template(); obj["timeline"][0]["role"] = "bad"
    res, mode = parse_and_validate(json.dumps(obj, ensure_ascii=False),
                                   duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert mode == "json_invalid" and not res.ok


def test_parse_and_validate_partial_mode():
    text = '"trend_summary": "x", "core_meme": "y",'
    res, mode = parse_and_validate(text, duration_s=DUR, shot_boundaries=BOUNDS, beat_points_known=BEATS)
    assert mode in ("partial", "partial_invalid")


# ---------- prompts ----------

def test_dedupe_json_repetition():
    text = '{"trend_summary": "a", "core_meme": "b"} {"trend_summary": "a", "core_meme": "b"}'
    out = dedupe_json_repetition(text)
    assert out.count('"trend_summary"') == 1
    assert out.count('"core_meme"') == 1


def test_dedupe_no_repetition_passthrough():
    assert dedupe_json_repetition('{"trend_summary": "a"}') == '{"trend_summary": "a"}'


def test_build_prompts():
    p = build_template_prompt("材料X")
    assert "材料X" in p and "硬性规则" in p and "trend_summary" in p
    r = build_repair_prompt("材料X", '{"bad": 1}', ["缺顶层键 timeline"])
    assert "上次输出" in r and "缺顶层键 timeline" in r
