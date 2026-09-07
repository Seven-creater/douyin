"""variant：变体规格（propose=ask() 提议 / preset=预设池 / file=显式文件）。"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from src.generation import prompts as gen_prompts
from src.generation.models import VariantSpec
from src.template.schema import extract_json_block

logger = logging.getLogger(__name__)

_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def snake_case(text: str) -> str:
    s = _SNAKE_RE.sub("_", (text or "").lower()).strip("_")
    return s or "variant"


def parse_variant_specs(text: str, n: int) -> list[VariantSpec] | None:
    block = extract_json_block(text)
    if not block:
        return None
    try:
        arr = json.loads(block)
    except ValueError:
        return None
    specs = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        subs = item.get("substitutions")
        if not isinstance(subs, dict) or not subs:
            continue
        specs.append(VariantSpec(
            variant_id=snake_case(str(item.get("variant_id") or f"variant{len(specs)}")),
            label=str(item.get("label") or item.get("variant_id") or "变体"),
            substitutions={str(k): str(v) for k, v in subs.items()},
        ))
    return specs[:n] if specs else None


def propose_variants(runner, template: dict, n: int, *,
                     max_new_tokens: int = 2048, max_retries: int = 1
                     ) -> tuple[list[VariantSpec], dict]:
    """ask() 一次提议；失败 → 预设池。"""
    prompt = gen_prompts.build_variant_proposal_prompt(
        template.get("core_meme") or "", template.get("fixed_elements") or [],
        template.get("replaceable_elements") or [], n,
    )
    t0 = time.time()
    last = ""
    for attempt in range(1 + max_retries):
        answer = runner.ask(prompt if attempt == 0 else prompt + "\n\n请严格只输出 JSON 数组。"
                            + f"\n上次输出开头：{last[:200]}", max_new_tokens=max_new_tokens)
        last = answer.text
        specs = parse_variant_specs(gen_prompts.dedupe_array_repetition(answer.text, '"variant_id"'), n)
        if specs:
            return specs, {"mode": "propose", "attempts": attempt + 1,
                           "usage": {"input_tokens": answer.input_tokens,
                                     "output_tokens": answer.output_tokens,
                                     "elapsed_s": answer.elapsed_s}}
    logger.warning("[variants] 提议失败，使用预设池")
    return preset_variants(template.get("replaceable_elements") or [], n), \
        {"mode": "preset", "attempts": 1 + max_retries, "usage": {"total_s": round(time.time() - t0, 1)}}


def preset_variants(replaceable: list[str], n: int) -> list[VariantSpec]:
    """预设池：把池内的新值模糊分配到 replaceable 元素上。"""
    specs = []
    for i, (vid, label, subs) in enumerate(gen_prompts.PRESET_SUBSTITUTION_POOL[:n]):
        mapping = {}
        for key in replaceable:
            for pool_key, val in subs.items():
                if pool_key in key or key in pool_key:
                    mapping[key] = val
                    break
        if not mapping:  # 对不上就把整包塞给第一个元素
            mapping = {replaceable[0]: "；".join(subs.values())} if replaceable else {}
        specs.append(VariantSpec(variant_id=vid, label=label, substitutions=mapping))
    return specs


def load_variant_file(path: Path) -> list[VariantSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("variants") or []
    return [VariantSpec(variant_id=snake_case(str(it["variant_id"])), label=str(it.get("label", "")),
                        substitutions={str(k): str(v) for k, v in it["substitutions"].items()})
            for it in items]


def validate_variants(specs: list[VariantSpec], replaceable: list[str]) -> list[str]:
    """substitutions 键应能对上 replaceable 条目（模糊包含）；对不上 → 警告串。"""
    warnings = []
    for s in specs:
        for key in s.substitutions:
            if not any(key in r or r in key for r in replaceable):
                warnings.append(f"{s.variant_id}: 替换键「{key}」未在 replaceable_elements 中找到对应项")
    return warnings
