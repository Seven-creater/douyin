"""extract_template：感知产物 → 结构化 Trend Template JSON（Phase 3 核心）。

CLI：python -m src.template.extract_template --aweme-id <id> [--force] [--dry-run]
  --dry-run 只打印装配后的上下文与预算统计，不调模型（S4 验证用）
流程：context → ask(TEMPLATE_PROMPT) → 抽取/校验 → （失败）repair 重试 → （仍失败）lenient 降级；
失败不写 result.json（幂等，重跑即重试），raw 输出存 raw_answer.txt。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common
from src.template import context as tctx
from src.template import schema
from src.template.prompts import build_repair_prompt, build_template_prompt, dedupe_json_repetition

logger = logging.getLogger(__name__)


def _coverage_penalty(timeline: list[dict], duration_s: float) -> float:
    """覆盖罚分 = 尾部缺失 + 段间空隙总和（秒）。用于修复版与保底版择优。"""
    if not timeline:
        return duration_s
    tail = max(0.0, duration_s - float(timeline[-1].get("end") or 0))
    gaps = sum(max(0.0, float(timeline[i].get("start") or 0) - float(timeline[i - 1].get("end") or 0))
               for i in range(1, len(timeline)))
    return round(tail + gaps, 2)


def synthesize(cfg: AppConfig, aweme_id: str, *, force: bool = False, runner=None,
               max_new_tokens: int | None = None) -> Path | None:
    t_cfg = cfg.template or {}
    max_chars = int(t_cfg.get("max_context_chars", 6000))
    max_retries = int(t_cfg.get("max_retries", 1))
    max_new = int(max_new_tokens or t_cfg.get("max_new_tokens", 2048))

    bundle = tctx.load_perception_bundle(cfg.paths.perception_dir, aweme_id, cfg.paths.videos_dir)
    duration_s = tctx.duration_of(bundle)
    boundaries = tctx.shot_boundaries_of(bundle)
    beats_known = tctx.beat_points_known_of(bundle)
    ctx, ctx_stats = tctx.build_context(bundle, max_chars=max_chars)

    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "template")
    params = {"prompt_version": "template_v1", "max_context_chars": max_chars,
              "tools_used": ctx_stats["tools_used"]}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[template %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    if runner is None:
        from src.perception.omni_runner import OmniRunner

        runner = OmniRunner(cfg.perception.get("omni") or {})

    t0 = time.time()
    raw_answers: list[str] = []
    result = None
    mode = "fail"
    attempts_meta = []
    fallback = None  # (template, warnings)：覆盖修复前的可用 json 结果

    for attempt in range(1 + max_retries):
        prompt = build_template_prompt(ctx) if attempt == 0 else None
        if attempt > 0:
            prev = schema.extract_json_block(raw_answers[-1]) or raw_answers[-1]
            prompt = build_repair_prompt(ctx, prev, attempts_meta[-1]["errors"])
        answer = runner.ask(prompt, max_new_tokens=max_new)
        raw = dedupe_json_repetition(answer.text)
        raw_answers.append(answer.text)
        res, m = schema.parse_and_validate(raw, duration_s=duration_s,
                                           shot_boundaries=boundaries, beat_points_known=beats_known)
        attempts_meta.append({
            "attempt": attempt, "mode": m, "errors": res.errors,
            "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
            "elapsed_s": answer.elapsed_s,
        })
        if m == "json":
            cov = [w for w in res.warnings if "未覆盖" in w]
            if cov and attempt < max_retries and fallback is None:
                # 覆盖不完整（尾部缺失/段间空隙）→ 带着问题清单 repair 一次；首版结果保底
                fallback = (res.template, res.warnings)
                attempts_meta[-1]["errors"] = cov
                logger.warning("[template %s] 覆盖不完整，触发修复重试：%s", aweme_id, cov[:2])
                continue
            result, mode = res.template, m
            break
        if m == "partial":
            result, mode = res.template if res.ok else None, m
            break
        logger.warning("[template %s] 第 %d 次尝试 mode=%s errors=%s", aweme_id, attempt + 1, m,
                       res.errors[:3])
        # json_invalid → 走 repair 重试

    if fallback is not None:
        if result is None or mode != "json":
            result, mode = fallback[0], "json"   # 修复重试失败 → 用首版 json 保底
        elif any("未覆盖" in w for w in res.warnings) and \
                _coverage_penalty(result["timeline"], duration_s) >= \
                _coverage_penalty(fallback[0]["timeline"], duration_s):
            result, mode = fallback[0], "json"   # 修复版覆盖仍不达标 → 择优保底

    total_s = time.time() - t0
    if result is None:
        (tdir / "raw_answer.txt").write_text(
            "\n\n===== ATTEMPT =====\n\n".join(raw_answers), encoding="utf-8")
        common.append_metric(
            common.metrics_path_for(cfg.paths.perception_dir),
            tool="extract_template", aweme_id=aweme_id, status="error",
            elapsed_s=round(total_s, 2), extra={"mode": mode, "errors": attempts_meta[-1]["errors"][:5]},
        )
        logger.error("[template %s] 解析失败（mode=%s），raw 已存 raw_answer.txt；重跑即重试", aweme_id, mode)
        return None

    usage = attempts_meta[-1]
    final_warnings = (fallback[1] if (fallback is not None and result is fallback[0])
                      else res.warnings) if result is not None else []
    output = {
        "template": result,
        "parse_meta": {
            "mode": mode, "attempts": len(attempts_meta),
            "errors": [e for a in attempts_meta for e in a["errors"]][:20],
            "warnings": [w for w in final_warnings][:20],
        },
        "usage": {
            "elapsed_s": usage["elapsed_s"], "total_s": round(total_s, 2),
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "context_chars": ctx_stats["chars"], "context_est_tokens": ctx_stats["est_tokens"],
        },
        "video": {"aweme_id": aweme_id, "duration_s": duration_s},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = common.write_result_json(tdir, tool="extract_template", aweme_id=aweme_id,
                                    params=params, output=output)
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="extract_template", aweme_id=aweme_id, status="ok",
        elapsed_s=usage["elapsed_s"], input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        extra={"mode": mode, "attempts": len(attempts_meta)},
    )
    logger.info("[template %s] mode=%s %d 段时间线 %d 固定/%d 可替换 → %s", aweme_id, mode,
                len(result.get("timeline") or []), len(result.get("fixed_elements") or []),
                len(result.get("replaceable_elements") or []), path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="热点模板抽取")
    common.add_common_cli_args(ap)
    ap.add_argument("--dry-run", action="store_true", help="只打印上下文与预算，不调模型")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="超长视频时间线被 2048 截断时调大（如 4096）")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="template_extract")

    if args.dry_run:
        ids = [args.aweme_id] if args.aweme_id else common.list_aweme_ids(cfg.paths.videos_dir)
        for aweme_id in ids:
            try:
                bundle = tctx.load_perception_bundle(cfg.paths.perception_dir, aweme_id, cfg.paths.videos_dir)
            except FileNotFoundError as exc:
                print(f"{aweme_id}: {exc}")
                continue
            ctx, stats = tctx.build_context(bundle, max_chars=int((cfg.template or {}).get("max_context_chars", 6000)))
            print(f"{aweme_id}: {stats['chars']} 字 est≈{stats['est_tokens']} tok tools={len(stats['tools_used'])}")
        common.emit_status_line("ok", dry_run=True, n=len(ids))
        return 0

    try:
        aweme_id, _ = common.resolve_target(cfg, args)
        path = synthesize(cfg, aweme_id, force=args.force, max_new_tokens=args.max_new_tokens)
        if path is None:
            common.emit_status_line("error", aweme_id=aweme_id, error="parse failed (see raw_answer.txt)")
            return 1
        common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract_template 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
