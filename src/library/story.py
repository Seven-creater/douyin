"""story：C2——把热门模板改写成「蜘蛛侠素材库世界观」的故事板。

输入：Template JSON（Phase 3 产物）+ 素材库能力摘要（镜头描述抽样）
输出：storyboard.json——每段含 主体/动作/景别/情绪 + 英文 CLIP 检索句
约束：段数与模板时间线一致（结构节拍必须复刻）；角色只能是库内存在的
（彼得/蜘蛛侠/反派——由能力摘要佐证）；不满足的段落改写故事而非硬凑。

CLI：python -m src.library.story --template-id <id> [--force]
产物：data/library/stories/<template_id>/storyboard.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.library.build_index import load_index
from src.perception import common
from src.template.prompts import dedupe_json_repetition
from src.template.schema import extract_json_block

logger = logging.getLogger(__name__)

STORY_PROMPT_HEADER = """你是短视频改编策划。给你一份抖音热门模板（时间线+固定元素+可替换元素）和一个
「蜘蛛侠电影素材库」的能力摘要。请把模板改写成用这个素材库就能拍出的蜘蛛侠版故事板。

【硬性规则】
1. 只输出一个 JSON 数组，无围栏无解释，段数与模板 timeline 完全一致，第 i 段的
   start/end 必须逐字等于模板第 i 段。
2. 每段字段：{"start", "end", "subject", "action", "shot_scale", "mood",
   "need_zh", "caption_zh"}；need_zh 是给素材检索用的一句中文需求
   （主体+动作+场景，如「主体：彼得帕克摘下蜘蛛面具；动作：震惊看向前方；场景：特写」）。
3. subject 只能是能力摘要里出现过的角色/元素（蜘蛛侠、彼得帕克、面具、战衣、
   反派等）；库里没有的场景（如外星飞船内部）必须改成库里有的等价物。
4. 保留模板的固定元素与节拍结构（悬念/反转/收尾位置不变）；台词字幕原样保留到 caption_zh。
5. 30 字内中文一句话说明该段画面（caption_zh 开头写【字幕】若有）。
"""


def capability_summary(rows: list[dict], n_sample: int = 40) -> str:
    """库能力摘要：规模统计 + 镜头描述抽样（供 LLM 知道库里有什么）。"""
    caps = [r["caption"] for r in rows if r.get("caption")]
    videos = sorted({r["video_stem"] for r in rows})
    lines = [f"[素材库规模] {len(rows)} 个镜头，来自 {len(videos)} 部预告/合集；"
             f"镜头时长分布 P50={_pct([r['duration_s'] for r in rows], 50):.1f}s "
             f"P90={_pct([r['duration_s'] for r in rows], 90):.1f}s"]
    step = max(1, len(caps) // n_sample)
    lines += f"[镜头描述抽样]（{min(len(caps), n_sample)} 条）" + "\n".join(
        f"- {c}" for c in caps[::step][:n_sample])
    return "\n".join(lines)


def _pct(vals: list[float], p: int) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100))] if s else 0.0


def build_story_prompt(template: dict, summary: str) -> str:
    tl = template.get("timeline") or []
    tl_txt = "\n".join(
        f"- {s['start']}~{s['end']}s [{s.get('role')}] 画面:{s.get('visual') or '无'}"
        f" 字幕:{s.get('text') or '无'}" for s in tl)
    fixed = "\n".join(f"- {f}" for f in template.get("fixed_elements") or [])
    return (STORY_PROMPT_HEADER
            + f"\n【模板核心梗】{template.get('core_meme')}\n"
            + f"\n【固定元素（必须保留）】\n{fixed}\n"
            + f"\n【模板时间线】\n{tl_txt}\n"
            + f"\n【素材库能力摘要】\n{summary}\n")


def validate_storyboard(board: list, template: dict) -> list[str]:
    """硬校验：段数一致、时间戳逐字对齐、字段齐全非空。返回错误清单。"""
    errors: list[str] = []
    tl = template.get("timeline") or []
    if not isinstance(board, list) or not board:
        return ["storyboard 不是非空数组"]
    if len(board) != len(tl):
        errors.append(f"段数 {len(board)} != 模板 {len(tl)}")
        return errors
    for i, (seg, t) in enumerate(zip(board, tl)):
        for k in ("subject", "action", "shot_scale", "mood", "need_zh", "caption_zh"):
            if not str(seg.get(k) or "").strip():
                errors.append(f"storyboard[{i}].{k} 缺失或为空")
        if abs(float(seg.get("start", -1)) - float(t["start"])) > 0.05 \
                or abs(float(seg.get("end", -1)) - float(t["end"])) > 0.05:
            errors.append(f"storyboard[{i}] 时间 {seg.get('start')}~{seg.get('end')}"
                          f" 与模板 {t['start']}~{t['end']} 不一致")
    return errors


def parse_storyboard(raw: str) -> list | None:
    block = extract_json_block(raw)
    if not block:
        return None
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, list) else None
    except ValueError:
        return None


def run_story(cfg: AppConfig, template_id: str, *, force: bool = False, runner=None) -> Path:
    s_cfg = cfg.library.get("story") or {}
    out_dir = cfg.paths.library_dir / "stories" / template_id
    out_path = out_dir / "storyboard.json"
    if out_path.exists() and not force:
        logger.info("[story %s] 已有产物，跳过", template_id)
        return out_path

    template = json.loads(
        (cfg.paths.perception_dir / template_id / "template" / "result.json")
        .read_text(encoding="utf-8"))["output"]["template"]
    rows, _emb = load_index(cfg)
    summary = capability_summary(rows)

    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})

    prompt = build_story_prompt(template, summary)
    max_new = int(s_cfg.get("max_new_tokens", 2048))
    max_retries = int(s_cfg.get("max_retries", 1))
    board = None
    attempts: list[dict] = []
    for attempt in range(1 + max_retries):
        p = prompt if attempt == 0 else (
            "你上次的输出未通过校验。错误：\n- " + "\n- ".join(attempts[-1]["errors"])
            + "\n请修正后只返回完整 JSON 数组。\n\n" + prompt)
        ans = runner.ask(p, max_new_tokens=max_new)
        raw = dedupe_json_repetition(ans.text, first_key='"start"')
        cand = parse_storyboard(raw)
        errs = validate_storyboard(cand, template) if cand else ["JSON 解析失败"]
        attempts.append({"attempt": attempt, "errors": errs[:8],
                         "elapsed_s": ans.elapsed_s})
        logger.info("[story %s] 第 %d 次：%d 错误", template_id, attempt + 1, len(errs))
        if not errs:
            board = cand
            break
    if board is None:
        # 保底：模板段直填通用蜘蛛侠检索句（结构不丢，检索质量降级）
        logger.warning("[story %s] LLM 故事板未过校验，降级为模板直填", template_id)
        board = [{"start": s["start"], "end": s["end"],
                  "subject": "蜘蛛侠", "action": s.get("visual") or "动作场面",
                  "shot_scale": "medium", "mood": "uncertain",
                  "need_zh": f"主体：蜘蛛侠；动作：{s.get('visual') or '动作场面'}",
                  "caption_zh": s.get("text") or s.get("visual") or ""}
                 for s in template["timeline"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"template_id": template_id, "attempts": attempts, "segments": board},
        ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[story %s] %d 段 → %s", template_id, len(board), out_path)
    common.emit_status_line("ok", segments=len(board))
    return out_path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="模板→蜘蛛侠故事板")
    ap.add_argument("--template-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_story")
    try:
        run_story(cfg, args.template_id, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("story 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
