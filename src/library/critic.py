"""critic：Omni 自审看成片（含音频）→ 结构化问题清单 → 优化动作 → 重渲染循环。

用户 2026-09-08 指令："让 omni 自己审核观看，知道自己哪里做得不好，然后优化"。
首版实测反馈：布局/字体/比例奇怪（16:9 源居中裁 9:16 所致），卡点不错。

问题分类（ASPECTS）：构图裁切 crop_weird / 画面比例 aspect / 字幕文字干扰 text_artifact /
节奏 rhythm / 素材相关性 relevance / 转场 transition / 清晰度 quality。
动作映射（确定性，不靠 LLM 改片）：
  aspect/crop_weird 多处或严重 → 全局换裁切策略 center↔blurpad
  relevance（带 t_s）→ 定位切序号换镜头（排除已用，换需求句重检索）
  其余记录进报告（V1 不自动处理）

CLI：python -m src.library.critic --template-id <tid> [--rounds 2]
产物：data/library/stories/<tid>_beatcut_v*/critique.json + critique_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common
from src.template.prompts import dedupe_json_repetition
from src.template.schema import extract_json_block

logger = logging.getLogger(__name__)

ASPECTS = ("crop_weird", "aspect", "text_artifact", "rhythm",
           "relevance", "transition", "quality")

CRITIC_PROMPT = """你是苛刻的短视频剪辑审片人。看这条用素材库镜头自动剪出来的卡点成片
（含音频），对照给它的定位（抖音竖屏卡点混剪），找出所有做得不好的地方。

【只输出一个 JSON 对象，无围栏无解释】
{"score": 0,                              # 1-10 整体质量
 "verdict": "一句话总评",
 "issues": [{"aspect": "crop_weird",      # 只能取：crop_weird/aspect/text_artifact/
                                             rhythm/relevance/transition/quality
              "t_s": 0.0,                 # 问题出现时刻（看不准给区间中点）
              "severity": "high",          # high/medium/low
              "note": "具体现象",           # 如"主体被裁掉一半""背景字幕残缺"
              "suggest": "改进建议"}],
 "good_points": ["做得好的地方"]}

审片要点：主体完整性（人物被裁？）、画面比例观感、画面里的文字/字幕是否残缺怪异、
切换是否踩点、素材与音乐氛围是否搭、白帧转场是否自然。不要客套，问题越多越有价值。"""


def parse_critique(raw: str) -> dict | None:
    block = extract_json_block(dedupe_json_repetition(raw, first_key='"score"'))
    if not block:
        return None
    try:
        obj = json.loads(block)
    except ValueError:
        return None
    if not isinstance(obj, dict) or "issues" not in obj:
        return None
    for i, iss in enumerate(obj.get("issues") or []):
        if isinstance(iss, dict) and iss.get("aspect") not in ASPECTS:
            iss["aspect"] = "quality"        # 归一越界分类，不拒收（审片是软任务）
    return obj


def run_critic(cfg: AppConfig, video: Path, *, runner=None) -> dict:
    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
    ans = runner.watch(video, CRITIC_PROMPT, max_new_tokens=2048)
    crit = parse_critique(ans.text)
    if crit is None:                          # 审片失败不阻塞循环：空清单+raw
        crit = {"score": None, "verdict": "critique 解析失败", "issues": [],
                "good_points": [], "raw_head": ans.text[:300]}
    crit["_elapsed_s"] = ans.elapsed_s
    return crit


def derive_actions(critique: dict, plan: list[tuple[float, float]],
                   current_crop: str) -> dict:
    """critique → 确定性动作。返回 {switch_crop, reswap_cut_indices}。"""
    issues = [i for i in critique.get("issues") or [] if isinstance(i, dict)]
    layout = [i for i in issues if i.get("aspect") in ("aspect", "crop_weird")]
    actions: dict = {"switch_crop": None, "reswap_cut_indices": [], "rationale": []}
    if layout and (len(layout) >= 2 or any(i.get("severity") == "high" for i in layout)):
        actions["switch_crop"] = "blurpad" if current_crop != "blurpad" else "center"
        actions["rationale"].append(f"布局类问题 {len(layout)} 处 → 切换裁切策略")
    swap_idx = []
    for i in issues:
        if i.get("aspect") != "relevance" or i.get("severity") == "low":
            continue
        t = i.get("t_s")
        if isinstance(t, (int, float)):
            for idx, (s, e) in enumerate(plan):
                if s - 0.3 <= t <= e + 0.3:
                    swap_idx.append(idx)
                    break
    actions["reswap_cut_indices"] = sorted(set(swap_idx))[:8]
    if swap_idx:
        actions["rationale"].append(f"素材不相关 {len(swap_idx)} 处 → 换镜头")
    return actions


def apply_swaps(rows, emb, embedder, plan, critique_issues, used: set) -> dict[int, int]:
    """对被点名的切重新检索换镜头（排除已用与原镜头，查询按批评意见偏移）。"""
    import numpy as np

    swaps: dict[int, int] = {}
    for idx, (s, e) in plan_idx_iter(plan, critique_issues):
        t_mid = (s + e) / 2
        # 查询句：取第 idx+1 个池句偏移（避开原句的相似邻域）
        from src.library.beat_cut import QUERY_POOL
        q = QUERY_POOL[(idx + 3) % len(QUERY_POOL)]
        qv = embedder.embed([q])[0]
        cos = (emb @ qv.reshape(-1)).ravel()
        for ri in np.argsort(-cos)[:60]:
            if int(ri) in used or rows[ri]["duration_s"] < 0.45:
                continue
            swaps[idx] = int(ri)
            used.add(int(ri))
            break
    return swaps


def plan_idx_iter(plan, issues):
    for i in issues:
        if not isinstance(i, dict) or i.get("aspect") != "relevance":
            continue
        t = i.get("t_s")
        if not isinstance(t, (int, float)):
            continue
        for idx, (s, e) in enumerate(plan):
            if s - 0.3 <= t <= e + 0.3:
                yield idx, (s, e)
                break


def run_loop(cfg: AppConfig, template_id: str, *, rounds: int = 2) -> Path:
    from src.library.beat_cut import run_beat_cut
    from src.library.build_index import E5Embedder, load_index

    rows, emb = load_index(cfg)
    embedder = E5Embedder(cfg.library.get("embed") or {})
    from src.perception.omni_runner import OmniRunner
    runner = OmniRunner(cfg.perception.get("omni") or {})

    # 读 v1 切点计划（与首渲一致）
    v1_meta = json.loads((cfg.paths.library_dir / "stories" / f"{template_id}_beatcut"
                          / "beatcut.json").read_text(encoding="utf-8"))
    plan = [(c["start"], c["end"]) for c in v1_meta["cuts"]]

    critiques, current_crop = [], v1_meta.get("crop", "center")
    vid = cfg.paths.library_dir / "stories" / f"{template_id}_beatcut" / "final.mp4"
    logger.info("[critic] 第 1 次审片：%s", vid)
    crit = run_critic(cfg, vid, runner=runner)
    (cfg.paths.library_dir / "stories" / f"{template_id}_beatcut" / "critique_v1.json") \
        .write_text(json.dumps(crit, ensure_ascii=False, indent=1), encoding="utf-8")
    critiques.append({"version": "v1", "crop": current_crop, "critique": crit})
    logger.info("[critic] v1 score=%s issues=%d", crit.get("score"),
                len(crit.get("issues") or []))

    swaps: dict[int, int] = {}
    used_rows: set[int] = set()
    for r in range(2, rounds + 1):
        actions = derive_actions(crit, plan, current_crop)
        if not actions["switch_crop"] and not actions["reswap_cut_indices"]:
            logger.info("[critic] 无可执行动作，循环结束")
            break
        if actions["switch_crop"]:
            current_crop = actions["switch_crop"]
        issues = [i for i in crit.get("issues") or [] if isinstance(i, dict)]
        new_swaps = apply_swaps(rows, emb, embedder, plan, issues, used_rows)
        swaps.update(new_swaps)
        logger.info("[critic] v%d 动作：crop=%s 换 %d 镜头",
                    r, current_crop, len(new_swaps))
        vid = run_beat_cut(cfg, template_id, force=True, crop=current_crop,
                           swap_shots=swaps or None, tag=f"_v{r}")
        crit = run_critic(cfg, vid, runner=runner)
        meta_p = cfg.paths.library_dir / "stories" / f"{template_id}_beatcut_v{r}" / "critique.json"
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(crit, ensure_ascii=False, indent=1), encoding="utf-8")
        critiques.append({"version": f"v{r}", "crop": current_crop,
                          "swaps": len(swaps), "critique": crit})
        logger.info("[critic] v%d score=%s issues=%d", r, crit.get("score"),
                    len(crit.get("issues") or []))

    report = cfg.paths.library_dir / "stories" / f"{template_id}_beatcut" / "critique_report.md"
    lines = [f"# critic 自审循环：{template_id}", ""]
    for c in critiques:
        k = c["critique"]
        lines.append(f"## {c['version']}（crop={c.get('crop')}）score={k.get('score')} · "
                     f"{len(k.get('issues') or [])} 个问题")
        lines.append(f"> {k.get('verdict', '')}")
        for i in (k.get("issues") or [])[:8]:
            if isinstance(i, dict):
                lines.append(f"- [{i.get('severity')}] {i.get('aspect')} @ {i.get('t_s')}s："
                             f"{i.get('note', '')} → {i.get('suggest', '')}")
        if k.get("good_points"):
            lines.append(f"**好的一面**：{'；'.join(k['good_points'][:3])}")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[critic] 报告 → %s；最终成片：%s", report, vid)
    common.emit_status_line("ok", output=str(vid), report=str(report))
    return vid


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Omni 自审-优化循环")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_critic")
    try:
        run_loop(cfg, args.template_id, rounds=args.rounds)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("critic 循环失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
