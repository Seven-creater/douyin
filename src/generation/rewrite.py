"""rewrite：ask() 一次重写全部 unit prompts（失败 → 中文 brief 直填降级）。"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.config import AppConfig
from src.generation import models, prompts as gen_prompts
from src.generation.models import GenerationPlan, VariantSpec
from src.perception import common
from src.template.schema import extract_json_block

logger = logging.getLogger(__name__)


def parse_unit_prompts(text: str, n_units: int) -> list[str] | None:
    """抽取 JSON 数组 [{"unit": i, "prompt": "..."}] → 按 unit 排序的 prompt 列表；失败 None。"""
    block = extract_json_block(text)
    if not block:
        return None
    try:
        arr = json.loads(block)
    except ValueError:
        return None
    if not isinstance(arr, list):
        return None
    by_unit = {}
    for item in arr:
        if isinstance(item, dict) and isinstance(item.get("prompt"), str) and item["prompt"].strip():
            by_unit[int(item.get("unit", -1))] = item["prompt"].strip()
    if sorted(by_unit) != list(range(n_units)):
        return None
    return [by_unit[i] for i in range(n_units)]


def fallback_prompts(plan: GenerationPlan, variant: VariantSpec) -> list[models.UnitPrompt]:
    """降级：中文 brief 直填脚手架（无英文润色，但可用）。"""
    subs_note = "；".join(f"{k}={v}" for k, v in variant.substitutions.items())
    out = []
    for u in plan.units:
        speech = u.speech_lines[0] if u.speech_lines else None
        visual = f"{u.visual_brief}（变体替换：{subs_note}）" if subs_note else u.visual_brief
        out.append(models.UnitPrompt(
            unit_id=u.unit_id,
            prompt=gen_prompts.build_t2va_prompt(visual, plan.audio_brief, speech, u.duration_s),
            seed=(models.fnv1a(f"{plan.template_id}|{variant.variant_id}|{u.unit_id}") % 2**31),
        ))
    return out


def rewrite_unit_prompts(runner, plan: GenerationPlan, variant: VariantSpec, *,
                         max_new_tokens: int = 2048, max_retries: int = 1
                         ) -> tuple[list[models.UnitPrompt], dict]:
    """1 次生成 + 至多 1 次重试；仍失败 → fallback。返回 (prompts, meta{mode,usage})。"""
    prompt = gen_prompts.build_rewrite_prompt(plan, variant)
    last_text = ""
    t0 = time.time()
    for attempt in range(1 + max_retries):
        answer = runner.ask(prompt if attempt == 0 else prompt + "\n\n上次输出无法解析，请严格只输出 JSON 数组。"
                            + f"\n上次输出开头：{last_text[:200]}", max_new_tokens=max_new_tokens)
        last_text = answer.text
        texts = parse_unit_prompts(gen_prompts.dedupe_array_repetition(answer.text, '"unit"'), len(plan.units))
        if texts is not None:
            meta = {"mode": "json", "attempts": attempt + 1,
                    "usage": {"input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
                              "elapsed_s": answer.elapsed_s, "total_s": round(time.time() - t0, 1)}}
            out = [models.UnitPrompt(unit_id=u.unit_id, prompt=t,
                                     seed=models.fnv1a(f"{plan.template_id}|{variant.variant_id}|{u.unit_id}") % 2**31)
                   for u, t in zip(plan.units, texts)]
            return out, meta
        logger.warning("[rewrite %s] 第 %d 次解析失败", variant.variant_id, attempt + 1)
    logger.warning("[rewrite %s] 降级为 fallback 中文 prompt", variant.variant_id)
    return fallback_prompts(plan, variant), {"mode": "fallback", "attempts": 1 + max_retries,
                                             "usage": {"total_s": round(time.time() - t0, 1)}}


def run_for_variant(cfg: AppConfig, plan: GenerationPlan, variant: VariantSpec, *,
                    force: bool = False, runner=None) -> Path | None:
    g_cfg = (cfg.generation or {}).get("rewrite") or {}
    vdir = cfg.paths.generation_dir / plan.template_id / "variants" / variant.variant_id
    pdir = vdir / "prompts"
    pdir.mkdir(parents=True, exist_ok=True)
    params = {"prompt_version": "rewrite_v1", "variant": variant.variant_id}
    if common.done_or_skip(pdir, params, force=force):
        logger.info("[rewrite %s] 已有产物，跳过", variant.variant_id)
        return pdir / "result.json"

    if runner is None:
        from src.perception.omni_runner import OmniRunner

        runner = OmniRunner(cfg.perception.get("omni") or {})

    unit_prompts, meta = rewrite_unit_prompts(
        runner, plan, variant,
        max_new_tokens=int(g_cfg.get("max_new_tokens", 2048)),
        max_retries=int(g_cfg.get("max_retries", 1)),
    )
    output = {"variant_id": variant.variant_id, "mode": meta["mode"], "attempts": meta["attempts"],
              "usage": meta.get("usage"),
              "unit_prompts": [{"unit_id": p.unit_id, "prompt": p.prompt, "seed": p.seed}
                               for p in unit_prompts]}
    path = common.write_result_json(pdir, tool="rewrite_prompts", aweme_id=plan.template_id,
                                    params=params, output=output)
    common.append_metric(cfg.paths.generation_dir / "metrics.jsonl",
                         tool="rewrite_prompts", aweme_id=plan.template_id, status="ok",
                         elapsed_s=meta.get("usage", {}).get("elapsed_s"),
                         input_tokens=meta.get("usage", {}).get("input_tokens"),
                         output_tokens=meta.get("usage", {}).get("output_tokens"),
                         extra={"variant": variant.variant_id, "mode": meta["mode"]})
    logger.info("[rewrite %s] mode=%s %d prompts → %s", variant.variant_id, meta["mode"],
                len(unit_prompts), path)
    return path
