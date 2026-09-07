"""generation S2 单测：prompts / rewrite 解析 / variant 解析与预设。"""
from __future__ import annotations

import json

from src.generation import prompts as gp
from src.generation.models import GenerationPlan, GenerationUnit, VariantSpec, fnv1a
from src.generation.rewrite import fallback_prompts, parse_unit_prompts, rewrite_unit_prompts
from src.generation.variant import parse_variant_specs, preset_variants, snake_case, validate_variants


def make_plan() -> GenerationPlan:
    u0 = GenerationUnit(unit_id=0, roles=["setup", "buildup"], timeline_span=(0.0, 8.4),
                        duration_s=8.4, num_frames=226, seed_base=1,
                        visual_brief="纸箱滚动；女性跑入", speech_lines=["这箱子里到底是什么"],
                        segments=[])
    u1 = GenerationUnit(unit_id=1, roles=["twist", "ending"], timeline_span=(8.4, 16.9),
                        duration_s=8.5, num_frames=226, seed_base=2,
                        visual_brief="纸箱打开惊喜；魔性舞蹈", speech_lines=[],
                        segments=[])
    return GenerationPlan(template_id="t1", source_duration_s=16.9,
                          audio_brief="电子流行", units=[u0, u1])


VARIANT = VariantSpec(variant_id="giftbox", label="礼盒变体",
                      substitutions={"道具（纸箱）": "红丝带礼盒"})


# ---------- prompts ----------

def test_build_rewrite_prompt_contains_all():
    p = gp.build_rewrite_prompt(make_plan(), VARIANT)
    assert "纸箱滚动" in p and "这箱子里到底是什么" in p
    assert "红丝带礼盒" in p and "giftbox" in p or "礼盒变体" in p
    assert '"unit"' in p and "Vertical 9:16" in p


def test_build_variant_proposal_prompt():
    p = gp.build_variant_proposal_prompt("反差梗", ["固定钩子"], ["道具", "场景"], 3)
    assert "固定钩子" in p and "道具" in p and "3" in p


def test_build_t2va_prompt():
    p = gp.build_t2va_prompt("a box rolls", "electronic pop", "台词", 5.2)
    assert p.startswith("Vertical 9:16") and 'The person says: "台词"' in p


# ---------- parse_unit_prompts ----------

def test_parse_unit_prompts_ok():
    text = '```json\n[{"unit": 1, "prompt": "B"}, {"unit": 0, "prompt": "A"}]\n```'
    assert parse_unit_prompts(text, 2) == ["A", "B"]


def test_parse_unit_prompts_missing_unit_rejected():
    assert parse_unit_prompts('[{"unit": 0, "prompt": "A"}]', 2) is None


def test_parse_unit_prompts_garbage():
    assert parse_unit_prompts("不是 JSON", 2) is None
    assert parse_unit_prompts('{"unit": 0}', 1) is None  # 不是数组


def test_parse_unit_prompts_dedupe_repetition():
    from src.generation.prompts import dedupe_array_repetition

    text = '[{"unit": 0, "prompt": "A"}, {"unit": 1, "prompt": "B"}] [{"unit": 0, "prompt": "A"}]'
    assert parse_unit_prompts(dedupe_array_repetition(text, '"unit"'), 2) == ["A", "B"]
    assert dedupe_array_repetition('[{"unit": 0}]', '"unit"') == '[{"unit": 0}]'  # 无重复不动


# ---------- rewrite（mock ask） ----------

class FakeRunner:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def ask(self, prompt, *, max_new_tokens=None):
        self.calls.append(prompt)
        text = self.texts.pop(0)
        class A:  # 简答壳
            input_tokens = 100; output_tokens = 50; elapsed_s = 1.0
        return A.__new__(A) or type("A", (), {"text": text, "input_tokens": 100, "output_tokens": 50, "elapsed_s": 1.0})()


def _runner_with(text):
    return type("R", (), {"ask": staticmethod(lambda prompt, max_new_tokens=None:
        type("A", (), {"text": text, "input_tokens": 100, "output_tokens": 50, "elapsed_s": 1.0})())})()


def test_rewrite_json_mode():
    text = json.dumps([{"unit": 0, "prompt": "Vertical 9:16 ... gift box"},
                       {"unit": 1, "prompt": "Vertical 9:16 ... dance"}], ensure_ascii=False)
    out, meta = rewrite_unit_prompts(_runner_with(text), make_plan(), VARIANT)
    assert meta["mode"] == "json" and len(out) == 2
    assert "gift box" in out[0].prompt
    assert out[0].seed == fnv1a("t1|giftbox|0") % 2**31
    assert out[1].seed != out[0].seed


def test_rewrite_fallback_on_garbage():
    out, meta = rewrite_unit_prompts(_runner_with("完全不是 JSON"), make_plan(), VARIANT, max_retries=0)
    assert meta["mode"] == "fallback"
    assert "纸箱滚动" in out[0].prompt or "礼盒" in out[0].prompt  # fallback 带替换注记


def test_fallback_prompts_contain_substitution():
    out = fallback_prompts(make_plan(), VARIANT)
    assert "红丝带礼盒" in out[0].prompt
    assert 'The person says: "这箱子里到底是什么"' in out[0].prompt


# ---------- variant ----------

def test_parse_variant_specs_ok():
    text = '[{"variant_id": "Gift Box!", "label": "礼盒", "substitutions": {"道具": "礼盒"}}]'
    specs = parse_variant_specs(text, 3)
    assert specs and specs[0].variant_id == "gift_box"
    assert specs[0].substitutions == {"道具": "礼盒"}


def test_parse_variant_specs_truncates_to_n():
    text = json.dumps([{"variant_id": f"v{i}", "label": "x", "substitutions": {"a": "b"}} for i in range(5)])
    assert len(parse_variant_specs(text, 3)) == 3


def test_snake_case():
    assert snake_case("Gift Box!") == "gift_box"
    assert snake_case("") == "variant"


def test_preset_variants_fuzzy_match():
    repl = ["道具（如用购物袋代替纸箱）", "服装与造型", "具体场景"]
    specs = preset_variants(repl, 3)
    assert len(specs) == 3
    assert any("礼盒" in " ".join(s.substitutions.values()) for s in specs)
    # 至少有一个键能对上 replaceable
    assert any(any(k in r or r in k for r in repl) for s in specs for k in s.substitutions)


def test_validate_variants_warns_on_unknown_key():
    repl = ["道具"]
    w = validate_variants([VariantSpec("v", "x", {"完全无关的键": "值"})], repl)
    assert w and "完全无关的键" in w[0]
    assert validate_variants([VariantSpec("v", "x", {"道具": "值"})], repl) == []
