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
    "用中文描述这个电影画面，按「主体：…；动作：…；场景：…；情绪/氛围：…」格式，"
    "每项 10 字内。主体写具体（如 彼得帕克/蜘蛛侠战衣/章鱼博士），动作写清在做什么，"
    "不要评价画质，不要猜测画面外内容。"
)


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

    def describe(self, image_path: Path, max_new_tokens: int = 96) -> str:
        self.load()
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def caption_video(cfg: AppConfig, shots_result: Path, *, force: bool = False,
                  limit: int | None = None, runner: Qwen2VLRunner | None = None) -> Path:
    cap_cfg = cfg.library.get("captions") or {}
    max_new = int(cap_cfg.get("max_new_tokens", 120))
    tdir = shots_result.parent
    out_path = tdir / "captions.json"
    captions: dict[str, str] = {}
    if out_path.exists() and not force:
        captions = json.loads(out_path.read_text(encoding="utf-8"))
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
            cap = runner.describe(Path(s["kf"]), max_new_tokens=max_new)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[caption %s#%d] 失败：%s", s["video_stem"], s["shot_idx"], exc)
            cap = ""
        captions[str(s["shot_idx"])] = cap
        n_new += 1
        if n_new % 25 == 0:
            out_path.write_text(json.dumps(captions, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            logger.info("[caption %s] %d 条", s["video_stem"], len(captions))
    out_path.write_text(json.dumps(captions, ensure_ascii=False, indent=1), encoding="utf-8")
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
