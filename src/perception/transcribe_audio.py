"""transcribe_audio：FunASR SenseVoiceSmall + fsmn-vad → 全文/毫秒级分段/音频事件标签。

CLI：python -m src.perception.transcribe_audio --aweme-id <id> [--language auto] [--force]
首次运行自动从 ModelScope 下载模型 ~1GB（缓存 ~/.cache/modelscope）。
SenseVoice 单段上限 30s → fsmn-vad 预分段（config vad_max_single_segment_ms）。
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<\|([^|]+)\|>")
_LANG_TAGS = {"ZH", "EN", "YUE", "JA", "KO", "AUTO"}  # 与 extract_tags 的 upper() 对齐
_PROC_TAGS = {"WITHITN"}  # 处理标记（ITN），非音频事件
# 情感标签单独归档（对模板分析也有用）
_EMO_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "EMO_UNKNOWN", "FEARDISGUST", "SURPRISED"}
# rich_transcription_postprocess 会把事件/情感标签转成 emoji 留在文本里 → 清掉
# （🎼=BGM U+1F3BC、😊👏😂😢😡😱🤢😮 等）
_EMOJI_RE = re.compile(
    "[\U0001F3BC\U0001F3B5\U0001F60A\U0001F622\U0001F621\U0001F631\U0001F62E"
    "\U0001F922\U0001F602\U0001F44F\U0001F600]"
)


def _clean_text(text: str) -> str:
    return _EMOJI_RE.sub("", _TAG_RE.sub("", text)).strip()


def extract_tags(text: str) -> tuple[list[str], list[str]]:
    """返回 (audio_events, emotions)，从原始输出里抽 <|BGM|>/<|APPLAUSE|>/... 标签。"""
    tags = [t.upper() for t in _TAG_RE.findall(text)]
    events = sorted({t for t in tags if t not in _LANG_TAGS | _PROC_TAGS and t not in _EMO_TAGS})
    emotions = sorted({t for t in tags if t in _EMO_TAGS})
    return events, emotions


def parse_sensevoice_result(raw: dict, *, postprocess=None) -> dict:
    """raw = funasr generate 返回的 res[0]（dict）。postprocess 为 rich_transcription_postprocess
    （延迟注入便于单测；None 时只做标签剥离）。"""
    raw_text = raw.get("text") or ""
    events, emotions = extract_tags(raw_text)
    clean = _clean_text(postprocess(raw_text) if postprocess else raw_text)

    segments = []
    for si in raw.get("sentence_info") or []:
        if si.get("text"):
            segments.append({
                "start_ms": int(si.get("start") or 0),
                "end_ms": int(si.get("end") or 0),
                "text": _clean_text(str(si.get("text"))),
            })
    return {
        "full_text": clean,
        "segments": segments,
        "segments_available": bool(segments),
        "audio_events": events,
        "emotions": emotions,
    }


def load_transcriber(*, model: str, vad_model: str, vad_max_segment_ms: int, device: str):
    """延迟 import funasr；首次自动下载模型。"""
    from funasr import AutoModel

    return AutoModel(
        model=model,
        trust_remote_code=True,
        vad_model=vad_model,
        vad_kwargs={"max_single_segment_time": vad_max_segment_ms},
        device=device,
    )


def transcribe_with_model(model, video_path: Path, *, language: str = "auto") -> dict:
    # merge_vad/merge_length_s/batch_size_s：官方 SenseVoice 示例参数（让 sentence_info 出毫秒分段）
    res = model.generate(input=str(video_path), language=language, use_itn=True,
                         batch_size_s=60, merge_vad=True, merge_length_s=15)
    post = None
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        post = rich_transcription_postprocess
    except ImportError:
        pass
    return parse_sensevoice_result(res[0], postprocess=post)


def run_for_video(cfg: AppConfig, aweme_id: str, *, force: bool = False,
                  language: str | None = None, model=None):
    p_cfg = common.perception_cfg(cfg)
    t_cfg = p_cfg.get("transcribe") or {}
    language = language or t_cfg.get("language", "auto")

    video = common.resolve_video_path(cfg.paths.videos_dir, aweme_id)
    tdir = common.tool_dir_for(cfg.paths.perception_dir, aweme_id, "transcribe")
    params = {"model": t_cfg.get("model", "iic/SenseVoiceSmall"), "language": language}
    if common.done_or_skip(tdir, params, force=force):
        logger.info("[asr %s] 已有产物，跳过", aweme_id)
        return tdir / "result.json"

    if model is None:
        t0 = time.time()
        model = load_transcriber(
            model=t_cfg.get("model", "iic/SenseVoiceSmall"),
            vad_model=t_cfg.get("vad_model", "fsmn-vad"),
            vad_max_segment_ms=int(t_cfg.get("vad_max_single_segment_ms", 30000)),
            device=t_cfg.get("device", "cuda:0"),
        )
        logger.info("[asr] 模型加载 %.1fs", time.time() - t0)

    duration_s = common.video_duration_s(p_cfg.get("ffprobe_bin", "ffprobe"), video)
    t0 = time.time()
    output = transcribe_with_model(model, video, language=language)
    output.update({"engine": "funasr_sensevoice", "language": language, "duration_s": duration_s})

    path = common.write_result_json(tdir, tool="transcribe_audio", aweme_id=aweme_id,
                                    params=params, output=output)
    (tdir / "transcript.txt").write_text(output["full_text"], encoding="utf-8")
    common.append_metric(
        common.metrics_path_for(cfg.paths.perception_dir),
        tool="transcribe_audio", aweme_id=aweme_id, status="ok",
        elapsed_s=round(time.time() - t0, 2),
        extra={"segments": len(output["segments"]), "audio_events": output["audio_events"]},
    )
    logger.info("[asr %s] %d 段 %d 字 事件=%s → %s", aweme_id, len(output["segments"]),
                len(output["full_text"]), output["audio_events"], path)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="ASR 转写（SenseVoiceSmall）")
    common.add_common_cli_args(ap)
    ap.add_argument("--language", default=None, help="zh/en/yue/ja/ko/auto")
    args = ap.parse_args(argv)
    cfg = common.load_cfg(args)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="perception_asr")
    try:
        aweme_id, _ = common.resolve_target(cfg, args)
        path = run_for_video(cfg, aweme_id, force=args.force, language=args.language)
        common.emit_status_line("ok", output=str(path))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("transcribe_audio 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
