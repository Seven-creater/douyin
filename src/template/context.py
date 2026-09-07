"""上下文装配：感知 bundle → 紧凑中文上下文（纯函数，本地可测）。"""
from __future__ import annotations

import json
from pathlib import Path

from src.perception import common

OMNI_SECTIONS_USED = ("content_summary", "timeline", "bgm_sfx", "visual_shots", "template_elements")
# speech 节仅在 ASR 为空时纳入（避免与转写重复且互相矛盾）
_MAX_OMNI_SECTION_CHARS = 400
_MAX_TEMPLATE_ELEMENTS_CHARS = 600
_MAX_OCR_EVENTS = 15
_MAX_ASR_CHARS = 400
_TITLE_MAX = 60


def estimate_tokens(text: str) -> int:
    """Qwen 分词近似：CJK≈1 token/字，ASCII≈0.3 token/字（仅预算记账用）。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return int(cjk + (len(text) - cjk) * 0.3)


def load_perception_bundle(perception_dir: Path, aweme_id: str, videos_dir: Path) -> dict:
    """读全部上游产物。omni_full 缺失 → FileNotFoundError（硬依赖）；其余容忍缺失（值为 None）。"""
    def _read(tool: str) -> dict | None:
        env = common.read_result_json(Path(perception_dir) / aweme_id / tool)
        return env["output"] if env else None

    omni = _read("omni_full")
    if omni is None:
        raise FileNotFoundError(
            f"缺少 omni_full 产物（模板抽取的硬依赖）：先跑 run_baseline --ids {aweme_id}"
        )
    meta = None
    meta_path = Path(videos_dir) / aweme_id / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {
        "aweme_id": aweme_id,
        "metadata": meta,
        "inspect": _read("inspect"),
        "shots": _read("shots"),
        "ocr": _read("ocr"),
        "transcribe": _read("transcribe"),
        "beats": _read("beats"),
        "omni_full": omni,
    }


def _cut(s: str, limit: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[:limit] + "…"


def build_context(bundle: dict, *, max_chars: int = 6000) -> tuple[str, dict]:
    """返回 (中文紧凑上下文, stats)。超预算按 speech→visual_shots→ocr 顺序裁剪。"""
    lines: list[str] = []
    stats = {"tools_used": [], "sections_truncated": [], "chars": 0, "est_tokens": 0}

    meta = bundle.get("metadata") or {}
    wb = meta.get("wellbyte") or {}
    title = _cut(wb.get("title") or "", _TITLE_MAX)
    stats_like = (wb.get("stats") or {})
    lines.append(f"[标题] {title or '无'}")
    lines.append(f"[热度] 点赞 {stats_like.get('likes')} / 播放 {stats_like.get('views')}")
    stats["tools_used"].append("metadata")

    insp = bundle.get("inspect")
    if insp:
        lines.append(f"[规格] {insp.get('duration_s', 0):.1f} 秒，{insp.get('width')}x{insp.get('height')}，"
                     f"{insp.get('fps')}fps")
        stats["tools_used"].append("inspect")
    else:
        lines.append("[规格] 未知")

    shots = bundle.get("shots")
    boundaries: list[float] = []
    if shots:
        boundaries = list(shots.get("boundaries_s") or [])
        durs = "/".join(f"{s.get('duration_s', 0):.1f}" for s in shots.get("shots") or [])
        lines.append(f"[镜头] 边界(秒)：「{'」/「'.join(f'{b:g}' for b in boundaries)}」，"
                     f"共 {shots.get('shot_count')} 镜，各镜时长 {durs}")
        stats["tools_used"].append("shots")

    ocr = bundle.get("ocr")
    if ocr:
        events = ocr.get("text_events") or []
        shown = events[:_MAX_OCR_EVENTS]
        for e in shown:
            lines.append(f"[字幕] {e.get('t_start_s', 0):g}-{e.get('t_end_s', 0):g}s 「{e.get('text', '')}」")
        if len(events) > len(shown):
            lines.append(f"[字幕] 另有 {len(events) - len(shown)} 条略")
        stats["tools_used"].append("ocr")
    else:
        lines.append("[字幕] 未提供")

    asr = bundle.get("transcribe")
    asr_text = ""
    if asr:
        asr_text = asr.get("full_text") or ""
        events = ",".join(asr.get("audio_events") or [])
        lines.append(f"[语音转写·无时间戳] {_cut(asr_text, _MAX_ASR_CHARS)}")
        lines.append(f"[音频事件] {events or '无'}（ASR 无时间戳，speech 落位请按叙事时间线推断概括）")
        stats["tools_used"].append("transcribe")
    else:
        lines.append("[语音转写] 未提供")

    beats = bundle.get("beats")
    if beats and beats.get("beat_points_s"):
        pts = ", ".join(f"{p:g}" for p in beats["beat_points_s"][:60])
        lines.append(f"[节拍点·仅供 beat_points 选择] {pts}"
                     f"{' …' if len(beats['beat_points_s']) > 60 else ''}")
        stats["tools_used"].append("beats")
    else:
        lines.append("[节拍点] 未检测（beat_points 必须留空数组）")

    omni = bundle.get("omni_full") or {}
    sections = omni.get("sections") or {}
    header = "[视频整体理解]"
    for key in OMNI_SECTIONS_USED:
        if not asr_text and key == "speech":
            pass  # ASR 空时把 speech 节也纳入
        elif key == "speech":
            continue
        val = sections.get(key) or ""
        if not val.strip():
            continue
        limit = _MAX_TEMPLATE_ELEMENTS_CHARS if key == "template_elements" else _MAX_OMNI_SECTION_CHARS
        cut = _cut(val, limit)
        if cut != val.strip():
            stats["sections_truncated"].append(key)
        lines.append(f"{header}·{key}: {cut}" if header else f"·{key}: {cut}")
        header = ""
    stats["tools_used"].append("omni_full")

    context = "\n".join(lines)
    # 超预算二次裁剪（简单可靠：整体截断保头保尾）
    if len(context) > max_chars:
        keep_head = int(max_chars * 0.85)
        keep_tail = int(max_chars * 0.1)
        context = context[:keep_head] + "\n…（材料超长已截断）…\n" + context[-keep_tail:]
        stats["sections_truncated"].append("__context__")
    stats["chars"] = len(context)
    stats["est_tokens"] = estimate_tokens(context)
    return context, stats


def duration_of(bundle: dict) -> float:
    insp = bundle.get("inspect") or {}
    return float(insp.get("duration_s") or (bundle.get("omni_full") or {}).get("video", {}).get("duration_s") or 0.0)


def shot_boundaries_of(bundle: dict) -> list[float]:
    return list((bundle.get("shots") or {}).get("boundaries_s") or [])


def beat_points_known_of(bundle: dict) -> list[float]:
    return list((bundle.get("beats") or {}).get("beat_points_s") or [])
