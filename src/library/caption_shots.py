"""caption_shots：素材库 B2——轻量 VLM（Qwen2-VL-7B）给每个镜头关键帧写一句话中文描述。

分层索引的第 2 层（详见 README）：CLIP 管全量视觉检索（零 token），本工具只给镜头
落一句可读描述（供故事板 LLM 读库能力摘要 + 关键词辅助检索）；Omni 大模型只在
C3 检索命中后按需看候选片段，绝不直扫全库。

CLI：python -m src.library.caption_shots [--source trailers] [--limit N] [--force]
产物：data/library/shots/<stem>/captions.json（{shot_idx: caption}，断点续跑）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "用中文描述这个电影镜头，按「主体：…；动作：…；场景：…；情绪/氛围：…；"
    "镜头尺度：…；视觉强度：…」格式，"
    "每项 10 字内。主体写画面可确认的具体身份或外观（如 黑发持刀少年/白发成年男性），"
    "只有画面能确认时才写角色名；不得依据 IP 常识或示例猜角色。动作写清在做什么，"
    "不要评价画质，不要猜测画面外内容。"
)

# P1.5：pack 候选表在场时允许"特征一致才写角色名"—— Recognition Prior 帮
# 跨窗主体描述稳定（V3 病灶：同角色 caption 在 黑发少年/长发男性/持刀人 间漂移），
# 但仍禁止凭 IP 常识乱写（候选表≠画面证据）。
CAPTION_PROMPT_V2 = (
    "用中文描述这个电影镜头，按「主体：…；动作：…；场景：…；情绪/氛围：…；"
    "镜头尺度：…；视觉强度：…」格式，每项 10 字内。"
    "候选角色表（待验证假设）：{roster}"
    "主体写画面可确认的具体身份或外观；仅当画面特征与候选角色明显一致"
    "（发色/服装/体型/标志物）时才写该角色名，否则写可确认外观，禁止猜。"
    "动作写清在做什么，不要评价画质，不要猜测画面外内容。"
)


def _roster_line(shots_dir: Path) -> str:
    """pack 存在且 tier=model_prior 时返回候选表一行（无则空串 → 用 v1）。"""
    from src.library.film_bootstrap import annotation_view, load_pack

    pack_path = shots_dir / "knowledge_pack.json"
    if not pack_path.exists():
        return ""
    try:
        view = annotation_view(load_pack_from(pack_path))
    except Exception:                                     # noqa: BLE001 - caption 层不因 pack 崩
        return ""
    if not view:
        return ""
    rows = "；".join(f"{row['name']}({row['canonical_id']}:"
                    f"{row['appearance']})" for row in view[:10])
    return rows[:600] + "。"


def load_pack_from(pack_path: Path):
    import json as _json

    pack = _json.loads(pack_path.read_text(encoding="utf-8"))
    return pack if isinstance(pack, dict) else None


def effective_caption_version(shots_dir: Path, base_version: str) -> str:
    """captions.json 缓存版本：pack 在场时 v3（候选表），否则基线版本。"""
    return base_version + "+pack" if _roster_line(shots_dir) else base_version


class Qwen2VLRunner:
    """Qwen2-VL-7B 单图描述（16G 单卡；transformers 5.8 兼容旧架构）。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        path = self.cfg.get("model_path")
        logger.info("[vl] 加载 Qwen2-VL: %s", path)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda:0")
        self._processor = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28,
                                                        max_pixels=512 * 28 * 28)
        self._model.eval()

    def unload(self):
        """显式卸载（cli._index 在 caption 后接 Omni 标注前调用——同进程
        双模型叠加会 OOM，局部变量释放不归还 allocator 缓存）。"""
        if self._model is None:
            return
        del self._model
        self._model = None
        self._processor = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self, image_path: Path, max_new_tokens: int = 96,
                 prompt: str | None = None) -> str:
        self.load()
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt or CAPTION_PROMPT}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def describe_many(self, image_paths: list[Path], max_new_tokens: int = 120,
                      prompt: str | None = None) -> str:
        """Describe beginning/middle/end frames as one temporal shot."""
        self.load()
        import torch
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in image_paths]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text",
                        "text": (prompt or CAPTION_PROMPT) + "三帧按时间顺序排列。"})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self._processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def caption_video(cfg: AppConfig, shots_result: Path, *, force: bool = False,
                  limit: int | None = None, runner: Qwen2VLRunner | None = None) -> Path:
    cap_cfg = cfg.library.get("captions") or {}
    max_new = int(cap_cfg.get("max_new_tokens", 120))
    tdir = shots_result.parent
    out_path = tdir / "captions.json"
    # P1.5：pack 候选表在场 → v2 prompt（特征一致才写角色名）+ 版本失效重写
    roster = _roster_line(tdir)
    prompt = CAPTION_PROMPT_V2.format(roster=roster) if roster else None
    version = effective_caption_version(tdir, str(cap_cfg.get("prompt_version",
                                                              "shot_caption_v2")))
    captions: dict[str, str] = {}
    if out_path.exists() and not force:
        captions = json.loads(out_path.read_text(encoding="utf-8"))
        stored = captions.get("_prompt_version")
        if stored is None:                               # 旧文件无版本标记 → 视为基线
            stored = str(cap_cfg.get("prompt_version", "shot_caption_v2"))
        if stored != version:
            captions = {}                                 # 版本换代（如 pack 出现）：全量重写
    env = json.loads(shots_result.read_text(encoding="utf-8"))
    shots = env["output"]["shots"]

    own = runner is None
    if runner is None:
        runner = Qwen2VLRunner(cap_cfg)
    n_new = 0
    for s in shots:
        if str(s["shot_idx"]) in captions:
            continue
        if limit is not None and n_new >= limit:
            break
        try:
            kfs = [Path(path) for path in s.get("kfs") or [s["kf"]]]
            if len(kfs) > 1 and hasattr(runner, "describe_many"):
                cap = runner.describe_many(kfs, max_new_tokens=max_new, prompt=prompt)
            else:
                cap = runner.describe(kfs[0], max_new_tokens=max_new, prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[caption %s#%d] 失败：%s", s["video_stem"], s["shot_idx"], exc)
            cap = ""
        captions[str(s["shot_idx"])] = cap
        n_new += 1
        if n_new % 25 == 0:
            out_path.write_text(json.dumps(captions, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            logger.info("[caption %s] %d 条", s["video_stem"], len(captions))
    out_path.write_text(json.dumps({**captions, "_prompt_version": version},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("[caption %s] 共 %d 条 → %s", shots[0]["video_stem"], len(captions), out_path)
    if own:
        common.emit_status_line("ok", video=shots[0]["video_stem"], captions=len(captions))
    return out_path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="素材库镜头描述（Qwen2-VL）")
    ap.add_argument("--source", default="trailers")
    ap.add_argument("--limit", type=int, default=None, help="每视频新增条数上限（试跑用）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_caption")

    shots_dir = cfg.paths.library_dir / "shots"
    results = sorted(shots_dir.glob("*/result.json"))
    if not results:
        logger.error("先跑 index_shots：%s 下无产物", shots_dir)
        return 1
    runner = Qwen2VLRunner(cfg.library.get("captions") or {})
    for r in results:
        try:
            caption_video(cfg, r, force=args.force, limit=args.limit, runner=runner)
        except Exception:  # noqa: BLE001
            logger.exception("[caption] %s 失败，继续下一个", r)
    common.emit_status_line("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
